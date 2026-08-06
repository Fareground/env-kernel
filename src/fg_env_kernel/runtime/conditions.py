"""Condition evaluation — the predicate side of effect and action resolution.

Canonical implementations of the engine's condition helpers: conditional
effect clauses, world-level conditions, multi-target resolution, effect
conditions, target preconditions, and the comparison primitives they share.
Extracted so the engine class stays focused on the tick loop; the engine's
methods are now 1-line delegations into this module.

``composition.py`` duck-types an engine-shaped shim exposing several of these
by the same names, so the ``engine`` first argument is deliberately typed
loosely — anything with the right attributes works.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)


def _evaluate_conditional_clause(engine, spec, *, actor, target, params, result, resolve_val,
) -> bool:
    """Evaluate a CONDITIONAL effect's `if` / `if_expr` clause.

    Supports two forms:
      { "if": { subject, field?, operator, value, check_type? } }
        — uses the existing EffectCondition machinery.
      { "if_expr": "<expression>" }
        — resolves the expression; truthy ⇒ branch fires.
      { "if_compare": [lhs, "op", rhs] }
        — both sides may be expressions; supports == != > >= < <=
    """
    if not isinstance(spec, dict):
        return False

    if "if_expr" in spec:
        return bool(resolve_val(spec["if_expr"]))

    if "if_compare" in spec:
        cmp = spec["if_compare"]
        if not (isinstance(cmp, (list, tuple)) and len(cmp) == 3):
            return False
        lhs = resolve_val(cmp[0])
        op = str(cmp[1])
        rhs = resolve_val(cmp[2])
        try:
            if op == "==": return lhs == rhs
            if op == "!=": return lhs != rhs
            if op == ">":  return lhs > rhs
            if op == ">=": return lhs >= rhs
            if op == "<":  return lhs < rhs
            if op == "<=": return lhs <= rhs
            if op == "in":  return lhs in rhs
            if op == "not_in": return lhs not in rhs
        except TypeError:
            return False
        return False

    if "if" in spec and isinstance(spec["if"], dict):
        from ..action import EffectCondition
        d = spec["if"]
        try:
            cond = EffectCondition(
                subject=str(d.get("subject", "actor")),
                field=d.get("field"),
                operator=str(d.get("operator", "eq")),
                value=resolve_val(d.get("value")),
                check_type=str(d.get("check_type", "property")),
            )
        except Exception:
            return False
        return engine._evaluate_effect_condition(cond, actor, target, params)

    return False

def _evaluate_world_condition(engine, spec: dict) -> bool:
    """Generic world-state condition evaluator shared by
    `cooperative_win` / `cooperative_loss` / `world_property_threshold`.

    Supports the same clause forms as CONDITIONAL effects:
      `if_expr`, `if_compare: [lhs, op, rhs]`, or
      `expr` + `operator` + `value` shorthand.
    """
    from ..effects import resolve_expression
    if not isinstance(spec, dict):
        return False
    if "if_expr" in spec:
        return bool(resolve_expression(spec["if_expr"], state=engine.state, rng=engine._rng))
    if "if_compare" in spec:
        cmp = spec["if_compare"]
        if not (isinstance(cmp, (list, tuple)) and len(cmp) == 3):
            return False
        lhs = resolve_expression(cmp[0], state=engine.state, rng=engine._rng)
        op = str(cmp[1])
        rhs = resolve_expression(cmp[2], state=engine.state, rng=engine._rng)
        try:
            if op == "==": return lhs == rhs
            if op == "!=": return lhs != rhs
            if op == ">":  return lhs > rhs
            if op == ">=": return lhs >= rhs
            if op == "<":  return lhs < rhs
            if op == "<=": return lhs <= rhs
            if op == "in":  return lhs in rhs
        except TypeError:
            return False
        return False
    if "expr" in spec:
        lhs = resolve_expression(spec["expr"], state=engine.state, rng=engine._rng)
        op = spec.get("operator", "gte")
        rhs = spec.get("value", 0)
        try:
            if op == "gte": return lhs >= rhs
            if op == "lte": return lhs <= rhs
            if op == "gt":  return lhs > rhs
            if op == "lt":  return lhs < rhs
            if op == "eq":  return lhs == rhs
            if op == "neq": return lhs != rhs
        except TypeError:
            return False
    return False

def _resolve_multi_target(engine, target_token: str, actor, target):
    """Expand a multi-target token into a list of entities.

    Tokens (Tier-1 EffectDSL):
      "all"          — every alive agent
      "all_others"   — every alive agent except the actor
      "role:X"       — every entity whose entity_type == X (alive)
      "faction:Y"    — every entity in faction Y (alive)

    Returns [] if nothing matches. Single-target tokens
    (`actor`, `target`, or a specific id) should NOT be routed here.
    """
    if target_token == "all":
        return [e for e in engine.state.get_agent_entities() if e.alive]
    if target_token == "all_others":
        actor_id = actor.id if actor else None
        return [
            e for e in engine.state.get_agent_entities()
            if e.alive and e.id != actor_id
        ]
    if target_token.startswith("role:"):
        wanted = target_token.split(":", 1)[1]
        return [
            e for e in engine.state.get_agent_entities()
            if e.alive and getattr(e, "entity_type", None) == wanted
        ]
    if target_token.startswith("faction:"):
        wanted = target_token.split(":", 1)[1]
        mgr = engine.state.factions
        if mgr is None:
            return []
        return [
            e for e in engine.state.get_agent_entities()
            if e.alive and mgr.get_entity_faction(e.id) == wanted
        ]
    return []

def _evaluate_effect_condition(engine, condition, actor, target, params=None) -> bool:
    """Evaluate a runtime condition for a conditional effect.

    ``params`` is threaded so expressions inside the condition can
    reference effect-time variables — critical for ``for_each``
    sub-effects where the iteration variable lives in
    ``$params.<as>``. Without this, conditions referencing
    ``$params.X`` always evaluated False.
    """
    # Modern: full-expression form takes precedence over the legacy
    # subject/operator/field structure when set.
    expr = getattr(condition, "expr", None)
    if expr:
        from ..predicates import evaluate as _predicate_eval
        return _predicate_eval(
            expr,
            actor=actor, target=target,
            params=params or {}, state=engine.state, rng=engine._rng,
        )

    # Resolve the subject entity
    if condition.subject == "actor":
        ent = actor
    elif condition.subject == "target":
        ent = target
    else:
        ent = engine.state.get_entity(condition.subject)

    if ent is None:
        return False

    if condition.check_type == "property":
        val = ent.get(condition.field, 0) if condition.field else 0
        return engine._compare(val, condition.operator, condition.value)

    elif condition.check_type == "alive":
        expected = condition.value if condition.value is not None else True
        return ent.alive == expected

    elif condition.check_type == "has_status":
        status_name = condition.value
        active = engine.state.status_effects.get_active(ent.id)
        has_it = any(ase.definition.name == status_name for ase in active)
        return has_it

    logger.warning(f"Unknown effect condition check_type: '{condition.check_type}'")
    return False  # Unknown check_type: fail safe

def _check_target_preconditions(engine, actor, target, action_def) -> bool:
    """Check preconditions that require both actor and target (IS_ADJACENT, faction checks)."""
    from ..action import Operator
    for pc in action_def.preconditions:
        # Modern expression form supersedes the legacy operator switch.
        # If it referenced $target, the unified evaluator handles it.
        expr = getattr(pc, "expr", None)
        if expr:
            from ..predicates import evaluate as _predicate_eval
            if not _predicate_eval(expr, actor=actor, target=target, state=engine.state, rng=engine._rng):
                return False
            continue
        if pc.operator == Operator.IS_ADJACENT:
            if not target:
                return False
            actor_loc = engine.state.locations.get(actor.id) or actor.location_id
            target_loc = engine.state.locations.get(target.id) or target.location_id
            if not actor_loc or not target_loc:
                return False
            if not engine.state.adjacency:
                # No adjacency defined — consider all locations adjacent
                continue
            from ..pathfinding import Pathfinder
            if not Pathfinder.are_adjacent(engine.state.adjacency, actor_loc, target_loc):
                return False
        elif pc.operator == Operator.SAME_FACTION:
            if not target:
                return False
            if not engine.state.factions.are_allies(actor.id, target.id):
                return False
        elif pc.operator == Operator.DIFFERENT_FACTION:
            if not target:
                return False
            if engine.state.factions.are_allies(actor.id, target.id):
                return False
    return True

def _compare(val, operator: str, target_val) -> bool:
    """Compare a value using a string operator. Type-mismatched operands
    (string property vs numeric target) fail CLOSED, never crash the run."""
    try:
        return _compare_inner(val, operator, target_val)
    except TypeError:
        logger.warning(
            "comparison %r %s %r raised TypeError — condition treated as False",
            val, operator, target_val,
        )
        return False

def _compare_inner(val, operator: str, target_val) -> bool:
    if operator == "gte":
        return val >= target_val
    elif operator == "lte":
        return val <= target_val
    elif operator == "gt":
        return val > target_val
    elif operator == "lt":
        return val < target_val
    elif operator == "eq":
        return val == target_val
    elif operator == "neq" or operator == "ne":
        return val != target_val
    # Unknown operator → fail CLOSED. Returning True made a typo'd
    # condition fire the effect unconditionally — the one fail-open in
    # an otherwise fail-closed layer.
    logger.warning("unknown comparison operator %r — condition treated as False", operator)
    return False


__all__ = [
    "_evaluate_conditional_clause",
    "_evaluate_world_condition",
    "_resolve_multi_target",
    "_evaluate_effect_condition",
    "_check_target_preconditions",
    "_compare",
    "_compare_inner",
]
