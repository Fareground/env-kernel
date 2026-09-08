"""Action composition primitives — declarative ways to compose actions.

These are kernel-registered effect ops that let JSON express complex
mechanics without Python:

  invoke_action — fire another action's effects atomically (castling =
    move_king + move_rook expressed as one action that invokes both)

  for_each — apply a sub-list of effects to each entity in a
    $-resolved collection (fireball = "for each enemy in range, damage")

  sequence — run a list of effects in order, sharing a tiny per-call
    scratch dict so later steps can reference earlier ones via
    $params._seq.X

All three handlers are auto-registered on import via @effect.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .effect_context import EffectContext
from .registry import effect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# invoke_action — fire another action's effect chain
# ---------------------------------------------------------------------------


@effect("invoke_action", replace=True)
def _invoke_action(ctx: EffectContext, spec: Dict[str, Any]) -> Any:
    """Trigger another action's success-effects atomically.

    Schema:
        {
            "operation": "invoke_action",
            "value": {
                "action_name": "castle_kingside",       // required
                "actor": "$actor",                       // optional, defaults to current actor
                "target": "$target",                     // optional
                "params": {"from": "e1", "to": "g1"}     // optional, merged with current params
            }
        }

    The invoked action's preconditions are NOT re-checked — this is
    deliberate atomicity for things like castling (one logical "move"
    triggers both king and rook movement). If you DO want preconditions
    checked, use a trigger instead.
    """
    state = ctx.state
    spec_val = ctx.resolve(spec.get("value")) or {}
    if not isinstance(spec_val, dict):
        return None
    action_name = spec_val.get("action_name")
    if not action_name or action_name not in state.action_definitions:
        logger.warning("invoke_action: unknown action '%s'", action_name)
        return None

    action_def = state.action_definitions[action_name]

    # Resolve actor/target — default to the current ctx values
    actor = ctx.actor
    target = ctx.target
    actor_ref = spec_val.get("actor")
    if isinstance(actor_ref, str) and actor_ref.startswith("$"):
        actor_resolved = ctx.resolve(actor_ref)
        if hasattr(actor_resolved, "id"):
            actor = actor_resolved
        elif isinstance(actor_resolved, str):
            actor = state.entities.get(actor_resolved, actor)
    target_ref = spec_val.get("target")
    if isinstance(target_ref, str) and target_ref.startswith("$"):
        target_resolved = ctx.resolve(target_ref)
        if hasattr(target_resolved, "id"):
            target = target_resolved
        elif isinstance(target_resolved, str):
            target = state.entities.get(target_resolved, target)

    # Merge params (the invoked action sees current params + any overrides)
    merged_params = dict(ctx.params or {})
    overrides = spec_val.get("params") or {}
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            merged_params[k] = ctx.resolve(v)

    # Dispatch the invoked action's effects via the engine's _apply_effects
    # (lazy import to avoid circular)
    from .runtime.effect_dispatch import apply_effects
    sub_changes = apply_effects(
        # We need an engine reference but EffectContext stores state, not engine.
        # Reach for the engine via state.controller if available; else build a
        # shim with the minimum interface apply_effects uses.
        _shim_engine_for_ctx(ctx),
        action_def.effects_on_success,
        actor,
        target,
        merged_params,
        ctx.result,
    )
    return sub_changes if isinstance(sub_changes, list) else None


# ---------------------------------------------------------------------------
# for_each — apply sub-effects per-entity in a collection
# ---------------------------------------------------------------------------


@effect("for_each", replace=True)
def _for_each(ctx: EffectContext, spec: Dict[str, Any]) -> Any:
    """Apply a list of sub-effects to every entity in a collection.

    Schema:
        {
            "operation": "for_each",
            "value": {
                "in": "$within_range($actor.location_id, 2)",  // required: list of entity ids
                "as": "victim",                                  // optional: name in $params
                "effects": [
                    {"operation": "subtract", "target": "$params.victim",
                     "field": "health", "value": 5}
                ]
            }
        }

    For each entity id in the resolved collection:
      - It's exposed as ``$params.<as>`` (default: ``$params.it``)
      - The sub-effects list is applied with that entity as target
    """
    spec_val = spec.get("value") or {}
    if not isinstance(spec_val, dict):
        return None

    collection = ctx.resolve(spec_val.get("in"))
    if not isinstance(collection, list):
        return None

    var_name = spec_val.get("as", "it")
    sub_effects_raw = spec_val.get("effects") or []
    if not sub_effects_raw:
        return None

    # Compile the sub-effects once via the loader's parser
    from .pipeline.loader import _parse_effects
    sub_effects = _parse_effects(sub_effects_raw)
    if not sub_effects:
        return None

    from .runtime.effect_dispatch import apply_effects
    engine = _shim_engine_for_ctx(ctx)
    all_changes: List[dict] = []
    base_params = dict(ctx.params or {})
    for item in collection:
        # Resolve to (entity_id_string, entity_object) so $params.<as>
        # supports both dotted access ($params.t.field) AND target
        # resolution ($params.t as a target string).
        if isinstance(item, str):
            entity_id = item
            entity_obj = ctx.state.entities.get(item) if ctx.state else None
        elif hasattr(item, "id"):
            entity_id = item.id
            entity_obj = item
        else:
            continue
        if not entity_id:
            continue
        scratch_params = dict(base_params)
        # Bind the ENTITY OBJECT (not just the id) so the expression
        # language can do $params.<as>.field. The effect-dispatch
        # target resolver knows to pull `.id` when targeting.
        scratch_params[var_name] = entity_obj if entity_obj is not None else entity_id
        sub_changes = apply_effects(
            engine,
            sub_effects,
            ctx.actor,
            ctx.target,
            scratch_params,
            ctx.result,
        )
        if isinstance(sub_changes, list):
            all_changes.extend(sub_changes)
    return all_changes if all_changes else None


# ---------------------------------------------------------------------------
# sequence — run a list of effects in order with shared scratch params
# ---------------------------------------------------------------------------


@effect("sequence", replace=True)
def _sequence(ctx: EffectContext, spec: Dict[str, Any]) -> Any:
    """Run a list of sub-effects in order. Each step's resolved values
    are stored in ``$params._seq`` for later steps to reference.

    Schema:
        {
            "operation": "sequence",
            "value": {
                "steps": [
                    {"name": "roll", "effect": {"operation": "set", "field": "_roll",
                                                "value": "$dice(2, 6)"}},
                    {"name": "move", "effect": {"operation": "add", "field": "position",
                                                "value": "$actor._roll"}}
                ]
            }
        }

    For simple cases without the scratch dict, just emit a list of
    effects in ``effects_on_success`` — they already run in order. Use
    ``sequence`` only when steps need to communicate.
    """
    spec_val = spec.get("value") or {}
    steps = spec_val.get("steps") or []
    if not steps:
        return None

    from .pipeline.loader import _parse_effects
    from .runtime.effect_dispatch import apply_effects

    engine = _shim_engine_for_ctx(ctx)
    scratch = dict(ctx.params or {})
    scratch.setdefault("_seq", {})
    all_changes: List[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        eff_raw = step.get("effect") or step
        sub_effects = _parse_effects([eff_raw])
        if not sub_effects:
            continue
        sub_changes = apply_effects(
            engine, sub_effects, ctx.actor, ctx.target, scratch, ctx.result,
        )
        if isinstance(sub_changes, list):
            all_changes.extend(sub_changes)
            # Surface step-name → resolved change in scratch._seq
            step_name = step.get("name")
            if step_name and sub_changes:
                scratch["_seq"][step_name] = sub_changes[-1]
    return all_changes if all_changes else None


# ---------------------------------------------------------------------------
# Engine shim — composition primitives need to call into apply_effects,
# which expects an engine-like object. EffectContext doesn't carry one
# directly. We construct a minimal shim from the available context.
# ---------------------------------------------------------------------------


def _shim_engine_for_ctx(ctx: EffectContext) -> Any:
    """Build a minimal engine-like object from an EffectContext for
    recursive calls into apply_effects.

    apply_effects references: state, _rng, _emit_event,
    _evaluate_effect_condition, _evaluate_conditional_clause,
    _resolve_multi_target, _apply_effects (for recursive calls).

    If the state has a SimController with engine handle, prefer that.
    Otherwise build a lightweight shim that forwards what we can.
    """
    state = ctx.state

    # Best path: state.controller may carry an engine ref (set by P5 wiring)
    controller = getattr(state, "controller", None)
    if controller is not None:
        engine_ref = getattr(controller, "engine", None) or getattr(controller, "_engine", None)
        if engine_ref is not None:
            return engine_ref

    # Fallback: construct a minimal shim. Recursive callers only need
    # the attributes apply_effects references. This works for most
    # cases but does not handle the on_event hook.
    class _EngineShim:
        def __init__(self, state, rng, emit):
            self.state = state
            self._rng = rng
            self._emit_event = emit or (lambda *a, **k: None)

        def _evaluate_effect_condition(self, condition, actor, target, params=None) -> bool:
            from .predicates import evaluate as _eval
            expr = getattr(condition, "expr", None)
            if expr:
                return _eval(expr, actor=actor, target=target,
                             params=params or {}, state=self.state, rng=self._rng)
            # Legacy property check
            ent = actor if condition.subject == "actor" else target
            if ent is None:
                return False
            val = ent.get(condition.field, 0) if condition.field else 0
            try:
                v = float(val)
                tv = float(condition.value or 0)
            except (TypeError, ValueError):
                return val == condition.value
            op = condition.operator
            if op == "gte": return v >= tv
            if op == "lte": return v <= tv
            if op == "gt":  return v > tv
            if op == "lt":  return v < tv
            if op == "eq":  return v == tv
            if op == "neq": return v != tv
            return False

        def _evaluate_conditional_clause(self, spec, **kwargs):
            from .runtime.conditions import _evaluate_conditional_clause
            return _evaluate_conditional_clause(self, spec, **kwargs)

        def _resolve_multi_target(self, target_token: str, actor, target):
            # Minimal: support `all` / `all_others` / specific id lookup
            entities = list(self.state.entities.values())
            if target_token == "all":
                return entities
            if target_token == "all_others":
                actor_id = getattr(actor, "id", None)
                return [e for e in entities if e.id != actor_id]
            ent = self.state.entities.get(target_token)
            return [ent] if ent is not None else []

        def _apply_effects(self, effects, actor, target, params, result):
            from .runtime.effect_dispatch import apply_effects
            return apply_effects(self, effects, actor, target, params, result)

    return _EngineShim(state=state, rng=ctx.rng, emit=ctx.emit)


__all__ = ["_invoke_action", "_for_each", "_sequence"]
