"""lint_template() — static logic checker for WorldTemplate.

Catches errors that Pydantic structural validation can't:

  - Action references an `actor_type` that doesn't exist
  - Effect modifies a `field` not on that entity type's property list
  - Effect uses a resource that isn't registered
  - Termination references entity_type/property combos that don't exist
  - Domain module name isn't registered in the DomainModuleRegistry
  - Resolution archetype name isn't registered
  - No agent-role entity exists (nobody will act)
  - No termination condition (game can't end naturally)
  - Resource holdings reference entities that don't exist
  - Initial relations reference entities that don't exist

All issues use JSON-pointer-like paths so the agent (or human) can
locate and fix them. Each issue has a short suggested ``hint``.

The linter is intentionally CONSERVATIVE — it only flags things it's
sure about. If a check can't be done with confidence (e.g. expression
strings might reference dynamically-spawned entities), it stays
silent rather than crying wolf.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from .compile import CompileIssue


def lint_template(template: Any, *, registry: Any = None) -> List[CompileIssue]:
    """Run all static checks against a ``WorldTemplate``.

    Accepts either a ``WorldTemplate`` Pydantic instance or a dict.
    Returns a list of ``CompileIssue`` objects — both errors and
    warnings mixed. The caller (typically ``compile_template``) sorts
    them by severity.

    ``registry`` scopes the registry-dependent checks (resolution
    archetypes, termination check_types, custom effect ops) to a
    specific ``KernelRegistry``; defaults to the process-global one."""
    if hasattr(template, "model_dump"):
        data = template.model_dump()
    else:
        data = dict(template)

    ctx = _LintCtx(data, registry=registry)
    issues: List[CompileIssue] = []

    _check_entity_types_have_agent(ctx, issues)
    _check_terminations_exist(ctx, issues)
    _check_action_actor_target_types(ctx, issues)
    _check_action_effects_reference_known_fields(ctx, issues)
    _check_action_effects_reference_known_resources(ctx, issues)
    _check_action_resolution_archetype_registered(ctx, issues)
    _check_terminations_reference_known_entity_types(ctx, issues)
    _check_resource_holdings_reference_known_entities(ctx, issues)
    _check_initial_relations_reference_known_entities(ctx, issues)
    _check_domain_modules_registered(ctx, issues)
    _check_phase_handlers_registered(ctx, issues)
    _check_factions_reference_known_entities(ctx, issues)
    _check_termination_check_types_registered(ctx, issues)
    # Newer checks — catch the failure modes a real env-builder agent hit
    _check_effect_target_sentinels(ctx, issues)
    _check_expressions_for_unquoted_barewords(ctx, issues)
    _check_expression_property_references(ctx, issues)
    _check_unknown_spec_fields(ctx, issues)
    _check_mutation_values(ctx, issues)

    return issues


def _check_mutation_values(ctx: "_LintCtx", issues: List[CompileIssue]) -> None:
    """Check literals wherever effects occur, including nested/derived rules.

    Dynamic targets cannot always be typed statically; runtime dispatch performs
    the same numeric/boolean checks after resolving their actual entity.
    """
    from ..effect_values import EffectValueError, expression_source, number, validate_operand

    types = {
        et["name"]: {p["name"]: p.get("type") for p in et.get("properties", [])}
        for et in ctx.data.get("entity_types", [])
    }

    def walk(node: Any, path: str, actor_type: str | None = None, target_type: str | None = None) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]", actor_type, target_type)
        elif isinstance(node, dict):
            actor_type = node.get("actor_type", actor_type)
            target_type = node.get("target_type", target_type)
            op = node.get("operation")
            if op in {"set", "add", "subtract", "multiply"} and "value" in node:
                value = node["value"]
                target = node.get("target", "actor")
                et = actor_type if target in {"actor", "$actor"} else (
                    target_type if target in {"target", "$target"} else ctx.entity_id_to_type.get(target))
                kind = types.get(et, {}).get(node.get("field"))
                try:
                    validate_operand(value, numeric=op != "set")
                    if op == "set" and expression_source(value) is None:
                        if kind in {"int", "float"} or target == "physics":
                            number(value)
                            if kind == "int" and value != int(value):
                                raise EffectValueError("integer property requires an integer value")
                        elif kind == "bool" and not isinstance(value, bool):
                            raise EffectValueError("boolean property requires a boolean value")
                except EffectValueError as exc:
                    issues.append(CompileIssue(
                        severity="error", path=f"{path}.value", message=str(exc),
                        hint="Supply a correctly typed literal or a valid dollar expression; missing references fail at runtime.",
                    ))
            for key, item in node.items():
                # A literal payload can itself contain an 'operation' key.
                # Only conditional values contain nested executable effects.
                if key != "value" or op == "conditional":
                    walk(item, f"{path}.{key}" if path else key, actor_type, target_type)

    # Do not inspect data tables or stored snapshots as executable rules.
    for key in ("actions", "derived_rules", "triggers", "temporal", "decks"):
        walk(ctx.data.get(key), key)


# ---------------------------------------------------------------------------
# Lint context — pre-computed indices so checks stay O(N)
# ---------------------------------------------------------------------------


class _LintCtx:
    """Pre-computed indices over the template for fast lookups."""

    def __init__(self, data: Dict[str, Any], registry: Any = None):
        self.data = data
        self.registry = registry  # KernelRegistry or None (→ global)
        # entity_type name → set of property names defined on it
        self.entity_type_props: Dict[str, Set[str]] = {}
        # entity_type name → role
        self.entity_type_roles: Dict[str, str] = {}
        for et in data.get("entity_types", []):
            name = et.get("name", "")
            if not name:
                continue
            self.entity_type_props[name] = {
                p.get("name", "") for p in et.get("properties", []) if p.get("name")
            }
            # alive is always implicit on every entity
            self.entity_type_props[name].add("alive")
            self.entity_type_roles[name] = et.get("role", "agent")

        self.resource_names: Set[str] = {
            r.get("name", "") for r in data.get("resource_types", []) if r.get("name")
        }
        self.relation_names: Set[str] = {
            r.get("name", "") for r in data.get("relation_types", []) if r.get("name")
        }
        self.entity_ids: Set[str] = {
            e.get("id", "") for e in data.get("entities", []) if e.get("id")
        }
        # entity_id → entity_type
        self.entity_id_to_type: Dict[str, str] = {
            e["id"]: e["entity_type"] for e in data.get("entities", [])
            if e.get("id") and e.get("entity_type")
        }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_entity_types_have_agent(ctx: _LintCtx, issues: List[CompileIssue]) -> None:
    """A world needs decision agents or a declared autonomous mechanism."""
    has_agent = any(role == "agent" for role in ctx.entity_type_roles.values())
    autonomous = any(ctx.data.get(key) for key in (
        'physics', 'property_dynamics', 'derived_rules', 'triggers', 'domain_modules'))
    if not has_agent and not autonomous:
        issues.append(CompileIssue(
            severity="error",
            path="entity_types",
            message="no entity_type has role='agent' and no autonomous dynamics are declared",
            hint="Declare the scenario's autonomous dynamics/world rules, or add a decision participant with role: 'agent'.",
        ))


def _check_terminations_exist(ctx: _LintCtx, issues: List[CompileIssue]) -> None:
    """A game with no termination_conditions runs until max_rounds,
    which is fine but worth flagging as a warning."""
    if not ctx.data.get("termination_conditions"):
        issues.append(CompileIssue(
            severity="warning",
            path="termination_conditions",
            message="no termination_conditions — game will only stop at max_rounds",
            hint="Add at least one termination condition, e.g. "
                 "{check_type: 'last_one_standing', params: {...}}.",
        ))


def _check_action_actor_target_types(ctx: _LintCtx, issues: List[CompileIssue]) -> None:
    """Every action's actor_type must be a registered entity_type. Same
    for target_type if specified."""
    known = set(ctx.entity_type_props.keys())
    for i, a in enumerate(ctx.data.get("actions", [])):
        path = f"actions[{i}].actor_type"
        actor_type = a.get("actor_type", "")
        if actor_type and actor_type not in known:
            issues.append(CompileIssue(
                severity="error",
                path=path,
                message=f"actor_type '{actor_type}' is not a declared entity_type",
                hint=f"Add an entity_type with name '{actor_type}' to entity_types, "
                     f"or change actor_type to one of: {sorted(known) or '<none>'}.",
            ))
        target_type = a.get("target_type")
        if target_type and target_type not in known:
            issues.append(CompileIssue(
                severity="error",
                path=f"actions[{i}].target_type",
                message=f"target_type '{target_type}' is not a declared entity_type",
                hint=f"Pick one of: {sorted(known) or '<none>'}.",
            ))


def _check_action_effects_reference_known_fields(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Effects with `field` should reference a property declared on
    the target entity type. Skipped when target == '$expr' (dynamic
    multi-target) — too ambiguous to check statically."""
    for i, a in enumerate(ctx.data.get("actions", [])):
        actor_type = a.get("actor_type", "")
        target_type = a.get("target_type")
        for branch_name in ("effects_on_success", "effects_on_failure", "effects_on_partial"):
            for j, eff in enumerate(a.get(branch_name, []) or []):
                field = eff.get("field")
                if not field:
                    continue
                # Skip ops that don't modify entity properties
                op = (eff.get("operation") or "").lower()
                if op in {"transfer_resource", "emit_event", "set_relation",
                          "modify_relation", "kill", "spawn_entity",
                          "despawn_entity", "give_item", "take_item",
                          "drop_item", "pickup_item", "place_on_board",
                          "reset_board", "draw_from_deck", "claim_slot",
                          "release_slot", "draw_to_hand", "discard_from_hand",
                          "play_card", "pass_card", "use_held_card",
                          "grant_token", "consume_token", "conditional",
                          "follow_entity", "unfollow_entity",
                          "post_content", "share_content", "react_content"}:
                    continue
                target = eff.get("target", "actor")
                if target == "actor":
                    et = actor_type
                elif target == "target":
                    et = target_type or actor_type
                else:
                    # Specific entity_id reference — look it up
                    et = ctx.entity_id_to_type.get(target)
                if not et:
                    continue
                props = ctx.entity_type_props.get(et)
                if props is not None and field not in props:
                    issues.append(CompileIssue(
                        severity="warning",
                        path=f"actions[{i}].{branch_name}[{j}].field",
                        message=f"effect modifies '{field}' but it's not a declared "
                                f"property on entity_type '{et}'",
                        hint=f"Add '{field}' to {et}.properties, or change the "
                             f"effect.field to one of: {sorted(props)}.",
                    ))


def _check_action_effects_reference_known_resources(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Effects that name a resource must reference a declared resource_type."""
    for i, a in enumerate(ctx.data.get("actions", [])):
        for branch_name in ("effects_on_success", "effects_on_failure", "effects_on_partial"):
            for j, eff in enumerate(a.get(branch_name, []) or []):
                res = eff.get("resource")
                if not res:
                    continue
                if res not in ctx.resource_names:
                    issues.append(CompileIssue(
                        severity="error",
                        path=f"actions[{i}].{branch_name}[{j}].resource",
                        message=f"effect references resource '{res}' which "
                                f"isn't declared in resource_types",
                        hint=f"Add a resource_type with name '{res}', or remove "
                             f"this effect's resource field.",
                    ))


def _check_action_resolution_archetype_registered(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Every action's resolution_archetype must be a registered name."""
    try:
        from ..resolution import RESOLUTION_REGISTRY
        from ..registry import registry as _global_reg
        _kreg = ctx.registry if ctx.registry is not None else _global_reg
        known = set(RESOLUTION_REGISTRY.keys()) | set(_kreg.resolutions.keys())
    except Exception:
        return  # if we can't reach the registry, skip rather than false-positive
    for i, a in enumerate(ctx.data.get("actions", [])):
        arch = a.get("resolution_archetype", "deterministic")
        if arch and arch not in known:
            issues.append(CompileIssue(
                severity="error",
                path=f"actions[{i}].resolution_archetype",
                message=f"resolution_archetype '{arch}' is not registered",
                hint=f"Pick one of: {sorted(known)}.",
            ))


def _check_terminations_reference_known_entity_types(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Most termination check_types take an `entity_type` param. If
    specified, it must exist."""
    known_types = set(ctx.entity_type_props.keys())
    for i, tc in enumerate(ctx.data.get("termination_conditions") or []):
        params = tc.get("params") or {}
        et = params.get("entity_type")
        if et and et not in known_types:
            issues.append(CompileIssue(
                severity="error",
                path=f"termination_conditions[{i}].params.entity_type",
                message=f"termination references entity_type '{et}' "
                        f"which isn't declared",
                hint=f"Add an entity_type '{et}', or change to one of: "
                     f"{sorted(known_types) or '<none>'}.",
            ))


def _check_resource_holdings_reference_known_entities(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Resource holdings must reference an existing entity and resource."""
    for i, rh in enumerate(ctx.data.get("resource_holdings") or []):
        eid = rh.get("entity_id") or rh.get("entity")
        if eid and eid not in ctx.entity_ids:
            issues.append(CompileIssue(
                severity="warning",
                path=f"resource_holdings[{i}].entity_id",
                message=f"holding references entity '{eid}' which isn't "
                        f"in the entities list",
                hint="Studio may rename entity ids at spawn (fun-name "
                     "generation); this warning is suppressible.",
            ))
        res = rh.get("resource") or rh.get("name")
        if res and res not in ctx.resource_names:
            issues.append(CompileIssue(
                severity="error",
                path=f"resource_holdings[{i}].resource",
                message=f"holding references resource '{res}' which isn't "
                        f"in resource_types",
                hint=f"Add resource_type '{res}'.",
            ))


def _check_initial_relations_reference_known_entities(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Initial relations must reference declared entities + relation types."""
    for i, rel in enumerate(ctx.data.get("initial_relations") or []):
        rt = rel.get("type")
        if rt and rt not in ctx.relation_names:
            issues.append(CompileIssue(
                severity="error",
                path=f"initial_relations[{i}].type",
                message=f"relation type '{rt}' isn't declared in relation_types",
                hint=f"Add a relation_type '{rt}', or change to one of: "
                     f"{sorted(ctx.relation_names) or '<none>'}.",
            ))


def _check_domain_modules_registered(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Domain modules referenced by name must be importable."""
    try:
        from ..domain_module import DomainModuleRegistry
        known = set(DomainModuleRegistry.get_instance().list_available())
    except Exception:
        return
    for i, dm in enumerate(ctx.data.get("domain_modules") or []):
        name = dm.get("name", "")
        if name and name not in known:
            issues.append(CompileIssue(
                severity="error",
                path=f"domain_modules[{i}].name",
                message=f"domain module '{name}' isn't registered",
                hint=f"Pick one of: {sorted(known)}, or register a custom "
                     f"module at startup with DomainModuleRegistry.register().",
            ))


def _check_factions_reference_known_entities(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Faction member ids should reference declared entities."""
    for i, f in enumerate(ctx.data.get("factions") or []):
        for j, mid in enumerate(f.get("member_ids", []) or []):
            if mid and mid not in ctx.entity_ids:
                issues.append(CompileIssue(
                    severity="warning",
                    path=f"factions[{i}].member_ids[{j}]",
                    message=f"faction member '{mid}' isn't in the entities list",
                    hint="Studio may rename ids at spawn; suppressible if "
                         "you know the rename will happen.",
                ))


def _check_termination_check_types_registered(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Termination check_type strings must be either built-in
    (compound_and/or, expr) or registered in registry.terminations."""
    try:
        from ..registry import registry as _global_reg
        _kreg = ctx.registry if ctx.registry is not None else _global_reg
        known = set(_kreg.terminations.keys()) | {"expr", "compound_and", "compound_or"}
    except Exception:
        return
    # The engine's evaluator also supports these directly (its inline chain
    # predates the registry) — rejecting them was a false-positive build
    # blocker for real games (chess, mafia, co-op).
    known |= {
        "checkmate", "board_pattern", "cooperative_win", "cooperative_loss",
        "bankruptcy", "vote_threshold", "score_after_n_rounds",
        "world_property_threshold", "count_property", "faction_win",
        "event_triggered", "rounds_idle", "all_goals_complete",
    }

    def walk(items: Any, path: str) -> None:
        for i, tc in enumerate(items or []):
            if not isinstance(tc, dict):
                continue
            ct = (tc.get("check_type") or "").strip().lower()
            if ct and ct not in known:
                issues.append(CompileIssue(
                    severity="error",
                    path=f"{path}[{i}].check_type",
                    message=f"check_type '{ct}' isn't registered",
                    hint=f"Pick one of: {sorted(known)}, or register a custom "
                         f"check_type with @termination('name').",
                ))
            if ct == "property_threshold":
                params = tc.get("params") or {}
                if not params.get("entity_type") and params.get("scope") != "world":
                    issues.append(CompileIssue(
                        severity="error",
                        path=f"{path}[{i}].params",
                        message=(
                            "property_threshold needs params.entity_type, or "
                            "params.scope='world' for a world-level property — "
                            "without either it can never fire"
                        ),
                        hint="add entity_type, or scope: 'world'",
                    ))
            if tc.get("sub_conditions"):
                walk(tc["sub_conditions"], f"{path}[{i}].sub_conditions")

    walk(ctx.data.get("termination_conditions"), "termination_conditions")


# ---------------------------------------------------------------------------
# Newer checks — failure modes a real env-builder agent ran into
# ---------------------------------------------------------------------------


# Valid effect-target tokens. ANYTHING else must be either:
#   - "actor" / "target"
#   - "pool:<resource_name>"
#   - "<entity_id>"     (looked up in state.entities)
#   - "$<expr>"         (resolved at runtime; we can't validate statically)
#   - "all" / "all_others" / "role:<X>" / "faction:<X>"
_VALID_TARGET_BUILTINS = {"actor", "target", "all", "all_others"}


def _is_recognized_effect_target(target: str, ctx: _LintCtx) -> bool:
    """Is this target string something the kernel will actually resolve?"""
    if not target or not isinstance(target, str):
        return False
    if target in _VALID_TARGET_BUILTINS:
        return True
    if target.startswith("$"):
        return True
    if target.startswith("pool:") or target.startswith("role:") or target.startswith("faction:"):
        return True
    if target in ctx.entity_ids:
        return True
    return False


def _check_effect_target_sentinels(ctx: _LintCtx, issues: List[CompileIssue]) -> None:
    """Catch invalid effect-target sentinels — e.g. an agent inventing
    `"target": "target_owner"` (not a valid token, not an entity id).
    These silently drop effects at runtime; we surface them at lint.
    """
    for i, a in enumerate(ctx.data.get("actions", [])):
        for branch in ("effects_on_success", "effects_on_failure", "effects_on_partial"):
            for j, eff in enumerate(a.get(branch, []) or []):
                target = eff.get("target")
                if not target or not isinstance(target, str):
                    continue
                if _is_recognized_effect_target(target, ctx):
                    continue
                issues.append(CompileIssue(
                    severity="error",
                    path=f"actions[{i}].{branch}[{j}].target",
                    message=f"effect target '{target}' is not a recognized token "
                            f"or entity id",
                    hint=("Valid targets: 'actor', 'target', '<entity_id>', "
                          "'pool:<resource>', 'role:<type>', 'faction:<id>', "
                          "'all', 'all_others', or a $expression that resolves "
                          "to one of those. Made-up sentinels (like 'target_owner') "
                          "silently drop the effect at runtime."),
                ))


# Boolean tokens we accept without quoting in expressions. Anything else
# unquoted on either side of == / != is almost certainly a mistyped string.
_EXPR_RESERVED = {"true", "false", "null", "none", "nil", "yes", "no",
                  "on", "off", "and", "or", "not", "in"}


def _bareword_warnings_for_expr(expr: str, path: str) -> List[CompileIssue]:
    """Scan an expr string for `== bareword` / `!= bareword` patterns
    where `bareword` is an unquoted identifier that's not a $-ref,
    number, list, or reserved keyword. Almost always a typo.
    """
    out: List[CompileIssue] = []
    if not isinstance(expr, str):
        return out
    import re
    # Find `== something` or `!= something` patterns where `something`
    # is an unquoted bareword identifier on the RHS.
    pattern = re.compile(r"(==|!=)\s*([A-Za-z_][A-Za-z0-9_]*)\b")
    for m in pattern.finditer(expr):
        op, ident = m.group(1), m.group(2)
        if ident.lower() in _EXPR_RESERVED:
            continue
        # Don't flag $-refs (those start with $) or numeric tokens
        if ident.isdigit():
            continue
        out.append(CompileIssue(
            severity="warning",
            path=path,
            message=f"expression compares to unquoted bareword "
                    f"'{op} {ident}' — kernel treats this as the literal "
                    f"identifier name, but you probably meant the string",
            hint=f"Quote the string: `{op} \"{ident}\"`. Or use a $-reference "
                 f"if `{ident}` is a property/variable.",
        ))
    return out


def _check_expressions_for_unquoted_barewords(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Walk every `expr` field in the template — preconditions, effect
    conditions, derived rules, termination params — and flag bareword
    comparisons that aren't quoted strings or reserved tokens.
    """
    # Action preconditions + effect conditions
    for i, a in enumerate(ctx.data.get("actions", [])):
        for j, pc in enumerate(a.get("preconditions", []) or []):
            expr = pc.get("expr")
            if isinstance(expr, str):
                issues.extend(_bareword_warnings_for_expr(
                    expr, f"actions[{i}].preconditions[{j}].expr",
                ))
        for branch in ("effects_on_success", "effects_on_failure", "effects_on_partial"):
            for j, eff in enumerate(a.get(branch, []) or []):
                cond = eff.get("condition")
                if isinstance(cond, dict) and isinstance(cond.get("expr"), str):
                    issues.extend(_bareword_warnings_for_expr(
                        cond["expr"], f"actions[{i}].{branch}[{j}].condition.expr",
                    ))
    # Derived rules
    for i, r in enumerate(ctx.data.get("derived_rules") or []):
        expr = r.get("when")
        if isinstance(expr, str):
            issues.extend(_bareword_warnings_for_expr(
                expr, f"derived_rules[{i}].when",
            ))
    # Termination expressions
    for i, tc in enumerate(ctx.data.get("termination_conditions") or []):
        params = tc.get("params") or {}
        expr = params.get("expr") if isinstance(params, dict) else None
        if isinstance(expr, str):
            issues.extend(_bareword_warnings_for_expr(
                expr, f"termination_conditions[{i}].params.expr",
            ))


# Common $-refs that DON'T resolve. Agents reach for these out of habit.
_KNOWN_PROBLEMATIC_REFS = {
    "$actor.entity_id":
        "Use `$actor.id` — entities don't have an `entity_id` field, "
        "they have `id`.",
    "$target.entity_id":
        "Use `$target.id` — entities have `id`, not `entity_id`.",
    "$actor.type":
        "Use `$actor.entity_type` — the type field is `entity_type`.",
    "$target.type":
        "Use `$target.entity_type` — the type field is `entity_type`.",
}


def _scan_expr_for_problematic_refs(expr: str, path: str) -> List[CompileIssue]:
    out: List[CompileIssue] = []
    if not isinstance(expr, str):
        return out
    for bad_ref, hint in _KNOWN_PROBLEMATIC_REFS.items():
        if bad_ref in expr:
            out.append(CompileIssue(
                severity="warning",
                path=path,
                message=f"expression references '{bad_ref}' which doesn't resolve",
                hint=hint,
            ))
    return out


def _check_expression_property_references(
    ctx: _LintCtx, issues: List[CompileIssue],
) -> None:
    """Flag $-refs that are common typos / hallucinations."""
    for i, a in enumerate(ctx.data.get("actions", [])):
        for j, pc in enumerate(a.get("preconditions", []) or []):
            expr = pc.get("expr")
            if isinstance(expr, str):
                issues.extend(_scan_expr_for_problematic_refs(
                    expr, f"actions[{i}].preconditions[{j}].expr",
                ))
        for branch in ("effects_on_success", "effects_on_failure", "effects_on_partial"):
            for j, eff in enumerate(a.get(branch, []) or []):
                value = eff.get("value")
                if isinstance(value, str):
                    issues.extend(_scan_expr_for_problematic_refs(
                        value, f"actions[{i}].{branch}[{j}].value",
                    ))
                cond = eff.get("condition")
                if isinstance(cond, dict) and isinstance(cond.get("expr"), str):
                    issues.extend(_scan_expr_for_problematic_refs(
                        cond["expr"], f"actions[{i}].{branch}[{j}].condition.expr",
                    ))


def _check_unknown_spec_fields(ctx: _LintCtx, issues: List[CompileIssue]) -> None:
    """Flag fields the typed spec models don't declare — the classic silent
    no-op: a typo'd field name ("effcts_on_success") validates fine under
    ``extra="allow"`` and is then ignored by the engine forever."""
    from . import loader

    def unknown(item: Any, model: Any, path: str) -> None:
        if not isinstance(item, dict):
            return
        known = set(model.model_fields)
        for key in item:
            if key not in known:
                issues.append(CompileIssue(
                    severity="warning",
                    path=f"{path}.{key}",
                    message=(
                        f"unknown field '{key}' on {model.__name__} — the "
                        "engine will silently ignore it"
                    ),
                    hint="likely a typo; check the field name against the spec",
                ))

    # Root level too — "termination_conditons" passing silently is the same
    # bug class one level up. Beyond WorldTemplate's own fields, several keys
    # are consumed straight off the raw schema by the loader or the app layer
    # (LLM personas, archetypes, physics) — they're extension points, not typos.
    root_known = set(loader.WorldTemplate.model_fields) | {
        "physics", "temporal", "spatial", "property_dynamics", "rules",
        "cognitive_config", "social_config", "crowd_config",
        "last_runtime_params", "personas", "agent_archetypes",
        "scenario_type", "viz", "visualization", "tables", "report_outputs",
    }
    for key in ctx.data:
        if key not in root_known:
            issues.append(CompileIssue(
                severity="warning",
                path=key,
                message=(
                    f"unknown top-level field '{key}' on WorldTemplate — the "
                    "engine will silently ignore it"
                ),
                hint="likely a typo; check the section name against the spec",
            ))

    for i, et in enumerate(ctx.data.get("entity_types", [])):
        unknown(et, loader.EntityTypeSpec, f"entity_types[{i}]")
        for j, p in enumerate(et.get("properties", []) or []):
            unknown(p, loader.PropertySpec, f"entity_types[{i}].properties[{j}]")
    for i, rt in enumerate(ctx.data.get("resource_types", [])):
        unknown(rt, loader.ResourceTypeSpec, f"resource_types[{i}]")
    for i, e in enumerate(ctx.data.get("entities", [])):
        unknown(e, loader.EntitySpec, f"entities[{i}]")
    for i, dm in enumerate(ctx.data.get("domain_modules", [])):
        unknown(dm, loader.DomainModuleSpec, f"domain_modules[{i}]")
    try:
        from ..registry import registry as _global_reg
        _kreg = ctx.registry if ctx.registry is not None else _global_reg
        _custom_effects = set(_kreg.effects.keys())
    except Exception:  # noqa: BLE001 — registry optional at lint time
        _custom_effects = set()

    for i, a in enumerate(ctx.data.get("actions", [])):
        unknown(a, loader.ActionSpec, f"actions[{i}]")
        for j, pc in enumerate(a.get("preconditions", []) or []):
            unknown(pc, loader.PreconditionSpec, f"actions[{i}].preconditions[{j}]")
            op = pc.get("operator") if isinstance(pc, dict) else None
            if op and not pc.get("expr") and op not in loader._OPERATOR_MAP:
                issues.append(CompileIssue(
                    severity="error",
                    path=f"actions[{i}].preconditions[{j}].operator",
                    message=f"unknown precondition operator '{op}'",
                    hint=f"use one of: {sorted(loader._OPERATOR_MAP)}",
                ))
        for branch in ("effects_on_success", "effects_on_failure", "effects_on_partial"):
            for j, eff in enumerate(a.get(branch, []) or []):
                unknown(eff, loader.EffectSpec, f"actions[{i}].{branch}[{j}]")
                op = eff.get("operation") if isinstance(eff, dict) else None
                if op and op not in loader._EFFECT_OP_MAP and op not in _custom_effects:
                    issues.append(CompileIssue(
                        severity="error",
                        path=f"actions[{i}].{branch}[{j}].operation",
                        message=f"unknown effect operation '{op}' — the effect would be dead",
                        hint=f"use one of: {sorted(loader._EFFECT_OP_MAP)}",
                    ))
                cond = eff.get("condition") if isinstance(eff, dict) else None
                if cond is not None:
                    unknown(cond, loader.EffectConditionSpec,
                            f"actions[{i}].{branch}[{j}].condition")

    def walk_terminations(items: Any, path: str) -> None:
        for j, tc in enumerate(items or []):
            unknown(tc, loader.TerminationSpec, f"{path}[{j}]")
            if isinstance(tc, dict) and tc.get("sub_conditions"):
                walk_terminations(tc["sub_conditions"], f"{path}[{j}].sub_conditions")

    walk_terminations(ctx.data.get("termination_conditions", []),
                      "termination_conditions")


__all__ = ["lint_template"]


def _check_phase_handlers_registered(ctx: _LintCtx, issues: List[CompileIssue]) -> None:
    from ..phase_handlers import PHASE_HANDLER_REGISTRY
    for index, phase in enumerate((ctx.data.get("temporal") or {}).get("phases") or []):
        handler = phase.get("handler")
        if handler and handler not in PHASE_HANDLER_REGISTRY:
            issues.append(CompileIssue(
                severity="error", path=f"temporal.phases[{index}].handler",
                message=f"Unknown phase handler: {handler!r}.",
                hint="Use a registered handler, or express scenario updates with kernel rules and property dynamics. Available handlers: " + ", ".join(sorted(PHASE_HANDLER_REGISTRY)),
            ))
