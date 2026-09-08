"""Effect dispatch — applies a list of Effects to the world state.

The canonical implementation of ``SimulationEngine._apply_effects``,
extracted here so the engine class stays focused on the tick loop.
The engine's method is now a 1-line delegation to ``apply_effects()``.

Custom effect ops registered via ``@effect("name")`` (see
``fg_env_kernel.registry``) are dispatched from within this module
— that path was already registry-driven (P3) and stays unchanged.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..action import Effect, EffectOperation
from ..resolution import ResolutionResult
from ..effect_values import EffectValueError, number, resolve_value
from ..types import PropertyType

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper imports that the original method body assumes are in scope.
# These match the engine.py module-level imports so the extracted body
# (which references Effect, EffectOperation, ResolutionResult, etc.)
# resolves correctly here.
# ---------------------------------------------------------------------------

# Re-export _coerce_effects so callers that imported it from engine
# can keep doing so via this module too.
from .engine import (  # noqa: E402, F401 — public for back-compat
    _coerce_effects,
    _is_multi_target,
    _resolve_cell_for_board,
)


def apply_effects(
    engine: "SimulationEngine",
    effects: List[Effect],
    actor: Any,
    target: Any,
    params: dict,
    result: ResolutionResult,
) -> List[dict]:
    """Apply a list of effects to the world state. Returns list of changes made."""
    # Observability — zero-cost when KERNEL_METRICS=0 (the default)
    from ..observability import engine_metrics_enabled, timed, metrics
    if engine_metrics_enabled():
        metrics.counter("kernel.effects_dispatched").inc(len(effects))
        with timed("kernel.apply_effects_ms"):
            return _apply_effects_body(engine, effects, actor, target, params, result)
    return _apply_effects_body(engine, effects, actor, target, params, result)


def _apply_effects_body(
    engine: "SimulationEngine",
    effects: List[Effect],
    actor: Any,
    target: Any,
    params: dict,
    result: ResolutionResult,
) -> List[dict]:
    """The actual implementation — separated so observability can wrap it."""
    from ..effects import (
        is_expression as _is_expr,
        resolve_expression as _resolve_expr,
        grant_token as _grant_token,
        consume_token as _consume_token,
    )

    # Lookup of the most recent event in this round — used by
    # $last_event.xxx expressions in declarative effects.
    last_event_payload: Optional[Dict[str, Any]] = None
    try:
        recent = engine.state.event_log.get_round(engine.state.temporal.current_round)
        if recent:
            ev = recent[-1]
            last_event_payload = dict(getattr(ev, "data", None) or {})
            last_event_payload["_event_type"] = getattr(ev, "event_type", None)
    except Exception:
        pass

    def _resolve_val(v: Any) -> Any:
        """Resolve `$…` expressions against the live effect context."""
        if not _is_expr(v):
            return v
        return _resolve_expr(
            v,
            actor=actor, target=target,
            params=params, state=engine.state,
            last_event=last_event_payload,
            result=result,
            rng=engine._rng,
        )

    def _drop(effect_obj: Effect, reason: str, detail: str) -> None:
        """Record a dropped effect as a structured, auditable diagnostic.

        The kernel's design is to skip effects it can't apply rather than
        crash the run. That's the right resilience policy — but a *silent*
        skip means a broken world "looks like it ran fine" (the auditability
        hole). Every skip now emits an ``effect_dropped`` event so it shows
        up in the transcript, traces, and analytics.
        """
        try:
            op = effect_obj.operation
            op_name = op.value if hasattr(op, "value") else str(op)
            engine._emit_event(
                "effect_dropped",
                actor_id=getattr(actor, "id", None),
                data={
                    "reason": reason,
                    "detail": detail,
                    "operation": op_name,
                    "target": effect_obj.target,
                    "field": effect_obj.field,
                },
                narrative=f"Effect '{op_name}' skipped: {detail}",
            )
        except Exception:
            # Diagnostics must never themselves break a run.
            logger.debug("effect_dropped diagnostic failed", exc_info=True)

    changes = []
    for effect in effects:
        # Check conditional effect
        if effect.condition and not engine._evaluate_effect_condition(
            effect.condition, actor, target, params,
        ):
            continue

        # ── Tier 1, 6.C: CONDITIONAL effect ──
        # `value` is { if|if_expr, then: [...], else: [...] }.
        # We pick the branch and recursively apply.
        if effect.operation == EffectOperation.CONDITIONAL:
            spec = effect.value if isinstance(effect.value, dict) else {}
            cond_truthy = engine._evaluate_conditional_clause(
                spec, actor=actor, target=target, params=params,
                result=result, resolve_val=_resolve_val,
            )
            branch_raw = spec.get("then") if cond_truthy else spec.get("else")
            if branch_raw:
                branch_effects = _coerce_effects(branch_raw, engine.registry)
                sub_changes = engine._apply_effects(
                    branch_effects, actor, target, params, result,
                )
                changes.extend(sub_changes)
            continue

        # Resolve $-expressions inside effect.target. Triggers like
        # `target: "$last_event.winner_id"` need this — without it
        # the string was passed through as-is to get_entity() and
        # returned None, silently dropping the effect.
        resolved_target = effect.target
        if isinstance(effect.target, str) and effect.target.startswith("$"):
            rv = _resolve_val(effect.target)
            if isinstance(rv, str) and rv:
                resolved_target = rv
            elif hasattr(rv, "id"):
                # Entity object — pull its id so downstream resolution
                # (which expects a string) keeps working.
                resolved_target = rv.id
            else:
                # Resolution produced nothing usable (e.g. the trigger
                # `$last_event.winner_id` had no winner). The effect would
                # silently no-op; surface it instead.
                _drop(effect, "unresolved_target_expr",
                      f"target expression {effect.target!r} resolved to {rv!r}")
                continue
            # Rewire the local effect view so the rest of the
            # switch sees the resolved target. Use a shallow clone
            # to avoid mutating the schema-side Effect object.
            effect = Effect(
                target=resolved_target,
                operation=effect.operation,
                field=effect.field,
                value=effect.value,
                value_supplied=effect.value_supplied,
                resource=effect.resource,
                relation_type=effect.relation_type,
                description=effect.description,
                condition=effect.condition,
                scale_by_magnitude=effect.scale_by_magnitude,
            )

        # ── Tier 1, 6.B: multi-target expansion ──
        # A target token like `all`, `all_others`, `role:X`,
        # `faction:Y` resolves to a LIST of entities. We expand the
        # effect into one per-entity copy and recursively apply —
        # this keeps the giant switch below single-target.
        if isinstance(effect.target, str) and _is_multi_target(effect.target):
            resolved = engine._resolve_multi_target(effect.target, actor, target)
            if not resolved:
                continue
            expanded = []
            for ent in resolved:
                # Make a shallow copy of the effect with target=this entity's id.
                cloned = Effect(
                    target=ent.id,
                    operation=effect.operation,
                    field=effect.field,
                    value=effect.value,
                    value_supplied=effect.value_supplied,
                    resource=effect.resource,
                    relation_type=effect.relation_type,
                    description=effect.description,
                    condition=effect.condition,
                    scale_by_magnitude=effect.scale_by_magnitude,
                )
                expanded.append(cloned)
            sub_changes = engine._apply_effects(expanded, actor, target, params, result)
            changes.extend(sub_changes)
            continue

        # Scale numeric values by success_degree when scale_by_magnitude is True
        mutation = effect.operation in {
            EffectOperation.SET, EffectOperation.ADD,
            EffectOperation.SUBTRACT, EffectOperation.MULTIPLY,
        }
        supplied = effect.value_supplied if effect.value_supplied is not None else effect.value is not None
        try:
            effective_value = resolve_value(
                effect.value, actor=actor, target=target, params=params,
                state=engine.state, last_event=last_event_payload, result=result, rng=engine._rng,
            ) if mutation else _resolve_val(effect.value)
            if mutation and effect.operation != EffectOperation.SET:
                if not supplied:
                    effective_value = 1 if effect.operation == EffectOperation.MULTIPLY else (result.magnitude if result else 0)
                number(effective_value)
        except EffectValueError as exc:
            _drop(effect, "invalid_effect_value", str(exc))
            continue
        if effect.scale_by_magnitude and result and isinstance(effective_value, (int, float)):
            effective_value = effective_value * result.success_degree

        # ── Physics target: mutate an ODE variable or parameter ──
        # `target: "physics"` with field = variable-or-param name lets
        # declarative effects (shocks, rules) reach the continuous model:
        # "competitor cuts prices in month 3" against an ODE world.
        if effect.target == "physics" and effect.field:
            model = getattr(engine.state, "physics", None)
            if model is None or model.is_empty():
                _drop(effect, "no_physics",
                      "effect targets physics but the world has no model")
                continue
            name = effect.field
            try:
                num = number(effective_value)
            except EffectValueError as exc:
                _drop(effect, "invalid_effect_value", str(exc))
                continue
            var = model.variables.get(name)
            slot = "variable" if var is not None else (
                "param" if name in model.params else None)
            if slot is None:
                _drop(effect, "unknown_physics_name",
                      f"physics has no variable or param {name!r}")
                continue
            if num is None:
                _drop(effect, "non_numeric_value",
                      f"physics effect on {name!r} needs a numeric value")
                continue
            old = var.value if var is not None else model.params[name]
            op_raw = effect.operation
            op_key = op_raw.value if hasattr(op_raw, "value") else str(op_raw)
            new = {"set": num, "add": old + num,
                   "subtract": old - num, "multiply": old * num}.get(op_key)
            if new is None:
                _drop(effect, "unsupported_physics_op",
                      f"physics target supports set/add/subtract/multiply, "
                      f"not {op_key!r}")
                continue
            try:
                number(new)
            except EffectValueError as exc:
                _drop(effect, "invalid_effect_value", str(exc))
                continue
            if var is not None:
                if var.min is not None:
                    new = max(var.min, new)
                if var.max is not None:
                    new = min(var.max, new)
                var.value = float(new)
            else:
                model.params[name] = float(new)
            changes.append({"physics": slot, "field": name,
                            "old": old, "new": float(new)})
            continue

        # Resolve target reference
        if effect.target == "actor":
            ent = actor
        elif effect.target == "target":
            ent = target
        elif isinstance(effect.target, str) and effect.target.startswith("pool:"):
            # 6.R — `pool:NAME` is a sentinel for resource-pool ops.
            # Don't try to look it up as an entity; let the per-op
            # handlers (TRANSFER_RESOURCE) detect it explicitly.
            ent = None
        else:
            ent = engine.state.get_entity(effect.target)

        # ── Plugin registry dispatch ──
        # If the effect.operation is a string (not a built-in
        # EffectOperation enum) AND it's registered in the kernel
        # registry, route through the plugin and skip the legacy
        # switch. Built-in ops keep their original code path so
        # this is a pure additive change.
        if isinstance(effect.operation, str):
            from ..effect_context import EffectContext as _EffCtx
            handler = engine.registry.effects.try_get(effect.operation)
            if handler is not None:
                ctx = _EffCtx(
                    state=engine.state,
                    actor=actor,
                    target=ent if ent is not None else target,
                    params=params,
                    result=result,
                    rng=engine._rng,
                    emit=engine._emit_event,
                )
                try:
                    rv = handler(ctx, {
                        "target": effect.target,
                        "field": effect.field,
                        "value": effective_value,
                        "resource": effect.resource,
                        "relation_type": effect.relation_type,
                        "description": effect.description,
                    })
                except Exception:
                    logger.exception(
                        "Effect handler '%s' raised; skipping",
                        effect.operation,
                    )
                    rv = None
                if ctx.changes:
                    changes.extend(ctx.changes)
                elif isinstance(rv, dict):
                    changes.append(rv)
                elif isinstance(rv, list):
                    changes.extend(c for c in rv if isinstance(c, dict))
                continue

        if ent is None and effect.operation not in (
            EffectOperation.EMIT_EVENT, EffectOperation.TRANSFER_RESOURCE,
        ):
            _drop(effect, "target_not_found",
                  f"no entity for target {effect.target!r}")
            continue

        if mutation and ent is not None and effect.field:
            schema = engine.state.entity_types.get(ent.entity_type)
            prop_schema = schema.get_property_schema(effect.field) if schema else None
            try:
                if effect.operation == EffectOperation.SET:
                    if prop_schema and prop_schema.type in (PropertyType.INT, PropertyType.FLOAT):
                        number(effective_value)
                        if prop_schema.type == PropertyType.INT:
                            if effective_value != int(effective_value):
                                raise EffectValueError(f"{effect.field!r} requires an integer")
                            effective_value = int(effective_value)
                    elif prop_schema and prop_schema.type == PropertyType.BOOL and not isinstance(effective_value, bool):
                        raise EffectValueError(f"{effect.field!r} requires a boolean")
                else:
                    old = number(ent.get(effect.field, 0))
                    operand = number(effective_value)
                    new = old + operand if effect.operation == EffectOperation.ADD else (
                        old - operand if effect.operation == EffectOperation.SUBTRACT else old * operand)
                    number(new)
                    if prop_schema and prop_schema.type not in (PropertyType.INT, PropertyType.FLOAT):
                        raise EffectValueError(f"{effect.field!r} is not a numeric property")
                    if prop_schema and prop_schema.type == PropertyType.INT and new != int(new):
                        raise EffectValueError(f"{effect.field!r} requires an integer result")
            except EffectValueError as exc:
                _drop(effect, "invalid_effect_value", str(exc))
                continue

        if effect.operation == EffectOperation.SET and effect.field:
            old_val = ent.get(effect.field)
            ent.set(effect.field, effective_value)
            changes.append({"entity": ent.id, "field": effect.field, "old": old_val, "new": effective_value})

        elif effect.operation == EffectOperation.ADD and effect.field:
            val = effective_value
            schema = engine.state.entity_types.get(ent.entity_type, None)
            prop_schema = schema.get_property_schema(effect.field) if schema else None
            old_val = ent.get(effect.field, 0)
            ent.modify(effect.field, val, prop_schema)
            changes.append({"entity": ent.id, "field": effect.field, "old": old_val, "new": ent.get(effect.field)})

        elif effect.operation == EffectOperation.MULTIPLY and effect.field:
            multiplier = effective_value
            schema = engine.state.entity_types.get(ent.entity_type, None)
            prop_schema = schema.get_property_schema(effect.field) if schema else None
            old_val = ent.get(effect.field, 0)
            if isinstance(old_val, (int, float)):
                new_val = old_val * multiplier
                # Clamp to property bounds if schema exists
                if prop_schema:
                    if prop_schema.min_value is not None:
                        new_val = max(prop_schema.min_value, new_val)
                    if prop_schema.max_value is not None:
                        new_val = min(prop_schema.max_value, new_val)
                    if prop_schema.type == PropertyType.INT:
                        new_val = int(new_val)
                ent.set(effect.field, new_val)
                changes.append({"entity": ent.id, "field": effect.field, "old": old_val, "new": ent.get(effect.field)})

        elif effect.operation == EffectOperation.SUBTRACT and effect.field:
            val = effective_value
            schema = engine.state.entity_types.get(ent.entity_type, None)
            prop_schema = schema.get_property_schema(effect.field) if schema else None
            old_val = ent.get(effect.field, 0)
            ent.modify(effect.field, -val, prop_schema)
            changes.append({"entity": ent.id, "field": effect.field, "old": old_val, "new": ent.get(effect.field)})

        elif effect.operation == EffectOperation.TRANSFER_RESOURCE and effect.resource:
            # Direction is actor -> recipient. Recipient is normally the
            # action target, but `effect.target` may name a specific
            # entity id (e.g. "pot_main") for games that move resources
            # into a shared sink instead of toward another agent.
            # 6.R: `pool:NAME` target ⇒ transfer to the named resource's
            # unallocated bucket (the "bank" / "treasury"). Same token
            # in reverse (`source: pool:NAME` on a different op) works
            # for paying OUT of the bank.
            amount = params.get("amount", effective_value or 1)
            pool = engine.state.resources.get(effect.resource)
            # Treasury transfer (pool:X)
            if isinstance(effect.target, str) and effect.target.startswith("pool:"):
                if pool and actor:
                    sent = float(amount)
                    if pool.holdings.get(actor.id, 0.0) >= sent:
                        pool.holdings[actor.id] = pool.holdings.get(actor.id, 0.0) - sent
                        pool.unallocated += sent
                        changes.append({
                            "resource": effect.resource,
                            "from": actor.id, "to": "pool",
                            "amount": amount,
                        })
                continue
            recipient = target
            if effect.target not in ("actor", "target", "", None):
                explicit = engine.state.get_entity(effect.target)
                if explicit is not None:
                    recipient = explicit
            if pool and recipient and actor and recipient.id != actor.id:
                success = pool.transfer(actor.id, recipient.id, float(amount))
                if success:
                    changes.append({
                        "resource": effect.resource,
                        "from": actor.id,
                        "to": recipient.id,
                        "amount": amount,
                    })

        elif effect.operation == EffectOperation.SET_RELATION and effect.relation_type:
            if target:
                engine.state.relations.set(actor.id, target.id, effect.relation_type, float(effect.value))
                changes.append({
                    "relation": effect.relation_type,
                    "from": actor.id,
                    "to": target.id,
                    "value": effect.value,
                })

        elif effect.operation == EffectOperation.MODIFY_RELATION and effect.relation_type:
            if target:
                engine.state.relations.modify(actor.id, target.id, effect.relation_type, float(effect.value))
                changes.append({
                    "relation_modify": effect.relation_type,
                    "from": actor.id,
                    "to": target.id,
                    "delta": effect.value,
                })

        elif effect.operation == EffectOperation.MOVE_TO:
            new_loc = str(effect.value)
            old_loc = engine.state.locations.get(ent.id) or ent.location_id
            # Validate adjacency if adjacency graph is defined
            if engine.state.adjacency and old_loc:
                from ..pathfinding import Pathfinder
                if not Pathfinder.are_adjacent(engine.state.adjacency, old_loc, new_loc):
                    logger.warning(
                        f"Entity {ent.id} tried to move from {old_loc} to {new_loc} "
                        f"but they are not adjacent. Move skipped."
                    )
                    continue
            # Check entry requirements
            if not engine.state.location_properties.check_entry(ent, new_loc):
                logger.warning(
                    f"Entity {ent.id} does not meet entry requirements for {new_loc}. Move skipped."
                )
                continue
            # Update entity location_id
            ent.location_id = new_loc
            # Update spatial tracking
            if old_loc and old_loc in engine.state.spatial_index:
                engine.state.spatial_index[old_loc].discard(ent.id)
            if new_loc not in engine.state.spatial_index:
                engine.state.spatial_index[new_loc] = set()
            engine.state.spatial_index[new_loc].add(ent.id)
            engine.state.locations[ent.id] = new_loc
            changes.append({"entity": ent.id, "moved_from": old_loc, "moved_to": new_loc})

        elif effect.operation == EffectOperation.KILL:
            ent.alive = False
            changes.append({"entity": ent.id, "killed": True})

        elif effect.operation == EffectOperation.GRANT_TOKEN:
            # `field` carries the token name; effective_value is the
            # count to grant (default 1). Generic replacement for
            # per-env jail_cards-style counters.
            token_name = effect.field or (effective_value if isinstance(effective_value, str) else None)
            count = int(effective_value) if isinstance(effective_value, (int, float)) else 1
            if token_name:
                new_total = _grant_token(ent, str(token_name), count)
                changes.append({
                    "entity": ent.id, "token_granted": token_name,
                    "count": count, "total": new_total,
                })
                # UI event so visualisations can animate the grant.
                engine._emit_event(
                    "token_granted",
                    actor_id=ent.id,
                    data={"token": token_name, "count": count, "total": new_total},
                )

        elif effect.operation == EffectOperation.CONSUME_TOKEN:
            token_name = effect.field or (effective_value if isinstance(effective_value, str) else None)
            count = int(effective_value) if isinstance(effective_value, (int, float)) else 1
            if token_name:
                spent = _consume_token(ent, str(token_name), count)
                changes.append({
                    "entity": ent.id, "token_consumed": token_name,
                    "count": count, "ok": spent,
                })
                engine._emit_event(
                    "token_consumed",
                    actor_id=ent.id,
                    data={"token": token_name, "count": count, "ok": spent},
                )

        elif effect.operation in (EffectOperation.DRAW_FROM_DECK, EffectOperation.USE_HELD_CARD):
            # Look up the deck by id.
            deck_id = effect.field or "deck"
            deck_mod = None
            if engine.state.domain_modules:
                from ..deck_module import DeckModule
                for _m in engine.state.domain_modules._modules.values():
                    if isinstance(_m, DeckModule) and _m.id == deck_id:
                        deck_mod = _m
                        break
            if deck_mod is None:
                logger.warning(f"DRAW_FROM_DECK: no deck with id={deck_id!r}")
                continue
            # Pull the card
            if effect.operation == EffectOperation.DRAW_FROM_DECK:
                card = deck_mod.draw(actor)
            else:
                card_idx = int(params.get("card_idx", 0))
                card = deck_mod.use_held(actor.id if actor else "", card_idx)
            if card is None:
                changes.append({"deck": deck_id, "drew": None})
                continue
            changes.append({
                "deck": deck_id, "drew": card.get("idx"),
                "text": card.get("text", ""),
            })
            # Apply the card's effects through the same machinery.
            card_effects_raw = card.get("effect") or card.get("effects") or []
            card_effects = _coerce_effects(card_effects_raw, engine.registry)
            if card_effects:
                nested = engine._apply_effects(card_effects, actor, target, params, result)
                changes.extend(nested)
            # Emit a UI-visible event so visualisations can animate.
            engine._emit_event(
                "deck_card_drawn",
                actor_id=actor.id if actor else None,
                data={
                    "deck_id": deck_id,
                    "card_idx": card.get("idx"),
                    "text": card.get("text", ""),
                    "keep_until_used": bool(card.get("keep_until_used")),
                },
                narrative=card.get("text", ""),
            )

        elif effect.operation in (EffectOperation.CLAIM_SLOT, EffectOperation.RELEASE_SLOT):
            # Tier 5b — worker placement.
            slots_id = effect.field or "slots"
            slots_mod = None
            if engine.state.domain_modules:
                from ..slots_module import SlotsModule
                for _m in engine.state.domain_modules._modules.values():
                    if isinstance(_m, SlotsModule) and _m.id == slots_id:
                        slots_mod = _m
                        break
            if slots_mod is None or ent is None:
                continue
            slot_id = str(effective_value) if effective_value is not None else None
            if not slot_id:
                continue
            if effect.operation == EffectOperation.CLAIM_SLOT:
                ok, reward_effects, err = slots_mod.claim(slot_id, ent.id)
                changes.append({
                    "slots": slots_id, "slot_id": slot_id,
                    "entity": ent.id, "claimed": ok, "error": err,
                })
                if ok:
                    engine._emit_event(
                        "slot_claimed",
                        actor_id=ent.id,
                        data={"slots_id": slots_id, "slot_id": slot_id},
                    )
                    # Fire the slot's reward effects, with $claimer
                    # available as the ent.
                    if reward_effects:
                        reward = _coerce_effects(reward_effects, engine.registry)
                        nested = engine._apply_effects(reward, actor, target, params, result)
                        changes.extend(nested)
            else:  # RELEASE_SLOT
                ok = slots_mod.release(slot_id, ent.id)
                changes.append({
                    "slots": slots_id, "slot_id": slot_id,
                    "entity": ent.id, "released": ok,
                })
            continue

        elif effect.operation in (
            EffectOperation.DRAW_TO_HAND, EffectOperation.DISCARD_FROM_HAND,
            EffectOperation.PLAY_CARD, EffectOperation.PASS_CARD,
        ):
            # Tier 5a — per-player hand operations.
            hand_id = effect.field or "hand"
            hand_mod = None
            if engine.state.domain_modules:
                from ..hand_module import HandModule
                for _m in engine.state.domain_modules._modules.values():
                    if isinstance(_m, HandModule) and _m.id == hand_id:
                        hand_mod = _m
                        break
            if hand_mod is None:
                logger.warning(f"hand op: no hand module id={hand_id!r}")
                continue
            if ent is None:
                continue

            if effect.operation == EffectOperation.DRAW_TO_HAND:
                n = int(effective_value) if isinstance(effective_value, (int, float)) else 1
                drawn = hand_mod.draw_to_hand(ent.id, n)
                changes.append({
                    "entity": ent.id, "hand_drew": len(drawn),
                    "card_ids": [c.get("id") for c in drawn],
                })
                engine._emit_event(
                    "hand_drew_cards",
                    actor_id=ent.id,
                    data={"hand_id": hand_id, "drew": [c.get("id") for c in drawn]},
                )
            elif effect.operation == EffectOperation.DISCARD_FROM_HAND:
                card_id = str(effective_value) if effective_value is not None else None
                if card_id is None:
                    continue
                removed = hand_mod.discard(ent.id, card_id)
                if removed:
                    changes.append({
                        "entity": ent.id, "discarded": card_id,
                    })
                    engine._emit_event(
                        "hand_discarded",
                        actor_id=ent.id,
                        data={"hand_id": hand_id, "card_id": card_id},
                    )
            elif effect.operation == EffectOperation.PLAY_CARD:
                card_id = str(effective_value) if effective_value is not None else None
                if card_id is None:
                    continue
                played = hand_mod.play(ent.id, card_id)
                if played:
                    changes.append({"entity": ent.id, "played": card_id})
                    engine._emit_event(
                        "hand_played_card",
                        actor_id=ent.id,
                        data={"hand_id": hand_id, "card_id": card_id,
                              "card": dict(played)},
                        narrative=f"{ent.name} played {played.get('name', card_id)}",
                    )
                    # Recursively apply the card's own effect, if any.
                    card_effects_raw = played.get("effect") or played.get("effects") or []
                    card_effects = _coerce_effects(card_effects_raw, engine.registry)
                    if card_effects:
                        nested = engine._apply_effects(card_effects, actor, target, params, result)
                        changes.extend(nested)
            elif effect.operation == EffectOperation.PASS_CARD:
                # Pass FROM actor TO `target` (action target).
                card_id = str(effective_value) if effective_value is not None else None
                if card_id is None or target is None:
                    continue
                moved = hand_mod.pass_to(actor.id, target.id, card_id)
                if moved:
                    changes.append({
                        "from": actor.id, "to": target.id, "card_id": card_id,
                    })
                    engine._emit_event(
                        "hand_card_passed",
                        actor_id=actor.id, target_id=target.id,
                        data={"hand_id": hand_id, "card_id": card_id},
                    )
            continue

        elif effect.operation == EffectOperation.RESET_BOARD:
            # Wipe all marks on a board (multi-game flow).
            board_id = effect.field or "main"
            if engine.state.domain_modules:
                from ..board_module import BoardModule
                for _m in engine.state.domain_modules._modules.values():
                    if isinstance(_m, BoardModule) and _m._id == board_id:
                        _m.reset_cells()
                        changes.append({"board": board_id, "reset": True})
                        engine._emit_event(
                            "board_reset",
                            data={"board_id": board_id},
                            narrative="Board cleared for next game.",
                        )
                        break
            continue

        elif effect.operation == EffectOperation.PLACE_ON_BOARD:
            # Place a mark on a BoardModule cell.
            # effect.field = board_id (default 'main')
            # effective_value = the mark to place (e.g. actor's 'mark'
            #   property via `$actor.mark`)
            # params['cell'] OR effect.condition (subject=='cell') →
            #   the cell index. For grid boards 1-9 maps to (r,c).
            #
            # Fallback policy: when the LLM omits `cell`, picks an
            # off-board cell, or picks an ALREADY-TAKEN cell, we
            # auto-place on the first empty cell so a turn is never
            # wasted to a silent no-op. The event payload records the
            # original intent + the actual placement.
            board_id = effect.field or "main"
            board_mod = None
            if engine.state.domain_modules:
                from ..board_module import BoardModule
                for _m in engine.state.domain_modules._modules.values():
                    if isinstance(_m, BoardModule) and _m._id == board_id:
                        board_mod = _m
                        break
            if board_mod is None:
                logger.warning(f"PLACE_ON_BOARD: no board with id={board_id!r}")
                continue
            # Resolve cell index — LLM may pass an int, "row,col" string,
            # or a tuple. None / unresolvable values fall through to
            # the empty-cell picker below.
            cell_raw = params.get("cell")
            if cell_raw is None and effect.condition is not None:
                cell_raw = _resolve_val(effect.condition.value)
            cell_pos = _resolve_cell_for_board(board_mod, cell_raw)
            fallback_reason: Optional[str] = None
            # Cell unresolvable OR cell already has a mark → first empty
            if cell_pos is None:
                fallback_reason = "unresolvable_cell"
            elif board_mod.mark_at(cell_pos) not in (None, "", 0):
                fallback_reason = "cell_already_taken"
            if fallback_reason:
                empties = board_mod.empty_cells()
                if not empties:
                    # Board is full — emit a non-placement event and skip.
                    engine._emit_event(
                        "board_mark_placed",
                        actor_id=actor.id if actor else None,
                        data={
                            "board_id": board_id,
                            "cell": None, "mark": effective_value,
                            "placed": False, "reason": "board_full",
                        },
                    )
                    continue
                cell_pos = empties[0]
            # Mark resolution — three sources, in priority order:
            # 1. Board's auto_marks config (`auto_marks: ["X","O"]`).
            #    First unique actor gets slot 0, second gets slot 1.
            #    This is the AUTHORITATIVE path for any game that
            #    declares auto_marks. Eliminates the entire class of
            #    "actor's mark property wasn't set in time" bugs.
            # 2. The effect's `value` ($actor.mark or literal) — used
            #    when auto_marks isn't configured.
            # 3. actor.id fallback if both above resolve null/empty.
            stored_mark: Any = None
            if actor is not None:
                stored_mark = board_mod.mark_for_actor(actor.id)
            if stored_mark is None:
                stored_mark = effective_value
            if stored_mark in (None, "", 0):
                stored_mark = actor.id if actor else "_unknown"
            placed = board_mod.place_mark(cell_pos, stored_mark)
            # Pattern detection — emit a `board_pattern_matched`
            # event when this placement creates a winning run. This
            # is SEPARATE from the `board_pattern` termination
            # predicate: this event lets triggers react (increment
            # game scores, reset the board) WITHOUT ending the sim,
            # enabling best-of-N formats.
            if placed:
                matched = board_mod.check_for_win_pattern(
                    patterns=board_mod._params.get("win_patterns")
                        or ["row_3", "col_3", "diag_3"],
                )
                if matched:
                    engine._emit_event(
                        "board_pattern_matched",
                        actor_id=actor.id if actor else None,
                        data={
                            "board_id": board_id,
                            "mark": matched["mark"],
                            "pattern": matched["pattern"],
                            "cells": matched["cells"],
                            "winner_id": actor.id if actor else None,
                        },
                    )
            changes.append({
                "board": board_id, "cell": cell_pos,
                "mark": stored_mark, "placed": placed,
                "fallback": fallback_reason,
            })
            # Discrete UI event so visualisations can animate the
            # placement instead of having to diff a snapshot.
            engine._emit_event(
                "board_mark_placed",
                actor_id=actor.id if actor else None,
                data={
                    "board_id": board_id,
                    "cell": list(cell_pos) if isinstance(cell_pos, tuple) else cell_pos,
                    "mark": stored_mark,
                    "placed": placed,
                    "fallback": fallback_reason,
                },
            )

        elif effect.operation == EffectOperation.ADVANCE_TRACK and effect.field:
            # Numeric advance on an arbitrary "track" property
            # (board position, score, hand_count). Effective_value
            # supports `$last_event.dice_sum`, `$params.delta`, etc.
            delta = effective_value if isinstance(effective_value, (int, float)) else 0
            if delta:
                old_val = ent.get(effect.field, 0) or 0
                new_val = (old_val + delta)
                # Optional ring length for a closed track
                ring = (effect.condition.value if effect.condition and effect.condition.field == "ring_length" else None)
                if isinstance(ring, int) and ring > 0:
                    new_val = new_val % ring
                ent.set(effect.field, new_val)
                changes.append({
                    "entity": ent.id, "field": effect.field,
                    "old": old_val, "new": new_val, "delta": delta,
                })
                # UI event so visualisations can animate the move.
                engine._emit_event(
                    "track_advanced",
                    actor_id=ent.id,
                    data={
                        "field": effect.field,
                        "old": old_val, "new": new_val, "delta": delta,
                    },
                )

        elif effect.operation == EffectOperation.EMIT_EVENT:
            engine._emit_event(
                "custom_event",
                actor_id=actor.id if actor else None,
                target_id=target.id if target else None,
                data={"event_name": effect.value, "description": effect.description},
                narrative=effect.description,
            )

        elif effect.operation == EffectOperation.APPLY_STATUS:
            # value = status effect name, look up from registered definitions
            status_name = effect.value
            status_defs = engine.state.status_effect_defs
            status_def = status_defs.get(status_name)
            if status_def and ent:
                source = actor.id if actor else None
                round_num = engine.state.temporal.current_round
                engine.state.status_effects.apply(ent.id, status_def, source, round_num)
                changes.append({"entity": ent.id, "status_applied": status_name})

        elif effect.operation == EffectOperation.REMOVE_STATUS:
            status_name = effect.value
            if ent:
                engine.state.status_effects.remove(ent.id, status_name)
                changes.append({"entity": ent.id, "status_removed": status_name})

        elif effect.operation == EffectOperation.DESPAWN_ENTITY:
            # Despawn the entity resolved from effect.target ("actor", "target", or entity_id)
            target_eid = ent.id if ent else (effect.value if isinstance(effect.value, str) else None)
            if target_eid:
                removed_ent = engine.state.despawn_entity(target_eid)
                if removed_ent:
                    changes.append({"entity": target_eid, "despawned": True})
                    engine._emit_event(
                        "entity_despawned",
                        actor_id=actor.id if actor else None,
                        target_id=target_eid,
                        data={"entity_name": removed_ent.name, "entity_type": removed_ent.entity_type},
                        narrative=f"{removed_ent.name} has been removed from the world.",
                    )

        elif effect.operation == EffectOperation.SPAWN_ENTITY:
            spawn_data = effect.value if isinstance(effect.value, dict) else {}
            template = spawn_data.get("template")
            new_id = spawn_data.get("entity_id")
            new_name = spawn_data.get("name", new_id or "unnamed")
            new_loc = spawn_data.get("location")
            prop_overrides = spawn_data.get("property_overrides")
            if template and new_id:
                spawned = engine.state.spawn_entity_from_template(
                    template_name=template,
                    entity_id=new_id,
                    name=new_name,
                    location=new_loc,
                    property_overrides=prop_overrides,
                )
                if spawned:
                    changes.append({"entity": new_id, "spawned": True, "template": template})
                    engine._emit_event(
                        "entity_spawned",
                        actor_id=actor.id if actor else None,
                        data={"entity_id": new_id, "entity_name": new_name, "template": template},
                        narrative=f"{new_name} has appeared in the world.",
                    )

        elif effect.operation == EffectOperation.GIVE_ITEM:
            # Transfer item from actor to target
            item_id = effect.value
            if actor and target and item_id:
                if engine.state.inventory.transfer_item(actor.id, target.id, str(item_id)):
                    changes.append({"item_given": str(item_id), "from": actor.id, "to": target.id})

        elif effect.operation == EffectOperation.TAKE_ITEM:
            # Transfer item from target to actor
            item_id = effect.value
            if actor and target and item_id:
                if engine.state.inventory.transfer_item(target.id, actor.id, str(item_id)):
                    changes.append({"item_taken": str(item_id), "from": target.id, "to": actor.id})

        elif effect.operation == EffectOperation.DROP_ITEM:
            # Drop item from entity's inventory at their location
            item_id = effect.value
            if ent and item_id:
                loc = engine.state.locations.get(ent.id) or (ent.location_id if hasattr(ent, 'location_id') else None)
                if loc and engine.state.inventory.drop_item(ent.id, str(item_id), loc):
                    changes.append({"item_dropped": str(item_id), "entity": ent.id, "location": loc})

        elif effect.operation == EffectOperation.PICKUP_ITEM:
            # Pick up item from ground at entity's location
            item_id = effect.value
            if ent and item_id:
                loc = engine.state.locations.get(ent.id) or (ent.location_id if hasattr(ent, 'location_id') else None)
                if loc and engine.state.inventory.pickup_item(ent.id, str(item_id), loc):
                    changes.append({"item_picked_up": str(item_id), "entity": ent.id, "location": loc})

        elif effect.operation == EffectOperation.AWARD_XP:
            # Award XP to a skill — field = skill name, value = XP amount
            skill_name = effect.field
            xp_amount = effect.value if isinstance(effect.value, (int, float)) else 1.0
            if ent and skill_name:
                level_info = engine.state.skills.award_xp(ent.id, skill_name, xp_amount)
                changes.append({"entity": ent.id, "skill_xp": skill_name, "amount": xp_amount})
                if level_info and level_info.get("leveled_up"):
                    engine._emit_event(
                        "skill_level_up",
                        actor_id=ent.id,
                        data=level_info,
                        narrative=f"{ent.name} leveled up {skill_name} to level {level_info['new_level']}!",
                    )

        elif effect.operation == EffectOperation.CRAFT_ITEM:
            # Execute a recipe — value = recipe name
            recipe_name = effect.value
            if ent and recipe_name:
                craft_result = engine.state.recipes.craft(ent.id, str(recipe_name), engine.state)
                if craft_result:
                    changes.append({"entity": ent.id, "crafted": str(recipe_name), "details": craft_result})
                    engine._emit_event(
                        "item_crafted",
                        actor_id=ent.id,
                        data=craft_result,
                        narrative=f"{ent.name} crafted {recipe_name}.",
                    )
                    # Check for level-up from craft XP
                    if craft_result.get("level_up"):
                        lu = craft_result["level_up"]
                        engine._emit_event(
                            "skill_level_up",
                            actor_id=ent.id,
                            data=lu,
                            narrative=f"{ent.name} leveled up {lu['skill']} to level {lu['new_level']}!",
                        )

        elif effect.operation == EffectOperation.POST_CONTENT:
            # Post social content — value = text, or use reasoning
            if engine.state.social and ent:
                text = str(effect.value) if effect.value else ""
                item = engine.state.social.create_content(
                    author_id=ent.id, text=text,
                    round_number=engine.state.temporal.current_round,
                )
                changes.append({"entity": ent.id, "posted_content": item.id})

        elif effect.operation == EffectOperation.SHARE_CONTENT:
            # Share existing content — value = content_id
            if engine.state.social and ent and effect.value:
                share = engine.state.social.share_content(
                    ent.id, str(effect.value),
                    round_number=engine.state.temporal.current_round,
                )
                if share:
                    changes.append({"entity": ent.id, "shared_content": share.id, "original": str(effect.value)})

        elif effect.operation == EffectOperation.REACT_CONTENT:
            # React to content — value = content_id, field = reaction_type
            if engine.state.social and ent and effect.value:
                reaction_type = effect.field or "like"
                engine.state.social.react_to_content(ent.id, str(effect.value), reaction_type)
                changes.append({"entity": ent.id, "reacted_to": str(effect.value), "reaction": reaction_type})

        elif effect.operation == EffectOperation.FOLLOW_ENTITY:
            # Follow another entity — target = entity to follow
            if engine.state.social and actor and target:
                engine.state.social.social_graph.follow(actor.id, target.id)
                changes.append({"follower": actor.id, "followed": target.id})

        elif effect.operation == EffectOperation.UNFOLLOW_ENTITY:
            # Unfollow another entity
            if engine.state.social and actor and target:
                engine.state.social.social_graph.unfollow(actor.id, target.id)
                changes.append({"unfollower": actor.id, "unfollowed": target.id})

    return changes



__all__ = ["apply_effects"]
