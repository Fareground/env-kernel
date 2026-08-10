"""Action resolution — name to definition, params to types, then apply.

Canonical implementations of ``SimulationEngine._resolve_action_def``,
``_coerce_action_params`` and ``_resolve_and_apply``: the path an
``ActionInstance`` travels from what an agent said it wants to do, through
schema lookup and parameter coercion, to resolved effects on the world.
Extracted so the engine class stays focused on the tick loop; the engine's
methods are now 1-line delegations into this module.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..action import ActionInstance
from ..messaging import Message
from ..resolution import get_resolution
from .engine import _action_suppresses_chat

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _resolve_action_def(engine, entity, action_instance: "ActionInstance"):
    """Resolve an action name to its definition, normalizing it in place.

    Returns the ActionDefinition, or None if the name can't be resolved
    (in which case the appropriate event has already been emitted).

    Used by BOTH the sequential turn path and the parallel/simultaneous
    ``_resolve_and_apply`` path so action-name handling is identical
    everywhere. We auto-apply a correction ONLY for an unambiguous
    case/whitespace match or a high-confidence typo; weaker matches are
    rejected (with suggestions) so we never silently substitute a
    mechanically different action the agent didn't choose.
    """
    action_name = action_instance.action_name
    action_def = engine.state.action_definitions.get(action_name)
    if action_def is not None:
        engine._coerce_action_params(entity, action_def, action_instance)
        return action_def

    from difflib import get_close_matches
    valid = list(engine.state.action_definitions.keys())
    suggestions = get_close_matches(action_name, valid, n=3, cutoff=0.6)
    # Case/whitespace-only mismatches are unambiguous — match them first
    # (difflib is case-sensitive, so "Buy"→"buy" scores only 0.67 and
    # would otherwise be rejected, forfeiting a turn that should plainly
    # execute).
    _norm = action_name.casefold().strip()
    strong = [k for k in valid if k.casefold().strip() == _norm]
    if not strong:
        HIGH_CONFIDENCE = 0.88
        strong = get_close_matches(action_name, valid, n=1, cutoff=HIGH_CONFIDENCE)
    if strong:
        fallback = strong[0]
        engine._emit_event(
            "action_corrected",
            actor_id=entity.id,
            action_name=action_name,
            data={"reason": "typo_autocorrect", "details": {
                "action_name": action_name, "corrected_to": fallback,
                "suggestions": suggestions}},
            narrative=(f"{entity.name} called '{action_name}' — auto-corrected "
                       f"to near-identical '{fallback}'."),
        )
        action_instance.action_name = fallback
        corrected = engine.state.action_definitions[fallback]
        engine._coerce_action_params(entity, corrected, action_instance)
        return corrected

    engine._emit_event(
        "action_failed",
        actor_id=entity.id,
        action_name=action_name,
        data={"reason": "unknown_action", "details": {
            "action_name": action_name, "suggestions": suggestions,
            "valid_actions": valid}},
        narrative=(
            f"{entity.name}'s action '{action_name}' is not defined."
            + (f" Did you mean: {', '.join(suggestions)}?" if suggestions else "")
            + f" Valid actions: {', '.join(valid) or '(none)'}."
        ),
    )
    return None

def _coerce_action_params(engine, entity, action_def, action_instance) -> None:
    """Enforce declared parameter types and bounds on submitted values.

    Declared ``parameters`` on an ActionDefinition were previously
    prompt decoration only — an out-of-range or wrong-typed LLM value
    flowed raw into resolution and effects. Coerce numerics to their
    declared type, clamp to [min, max], and fill a missing value from
    the declaration's default. Every correction is announced with an
    ``action_corrected`` event so the transcript stays honest.
    """
    if not action_def.parameters:
        return
    submitted = action_instance.parameters or {}
    fixes: dict[str, Any] = {}
    for decl in action_def.parameters:
        if not isinstance(decl, dict) or not decl.get("name"):
            continue
        name = str(decl["name"])
        ptype = str(decl.get("type") or "").lower()
        lo, hi = decl.get("min"), decl.get("max")
        if lo is None:
            lo = decl.get("min_value")
        if hi is None:
            hi = decl.get("max_value")
        value = submitted.get(name)
        if value is None:
            if decl.get("default") is not None:
                fixes[name] = decl["default"]
            continue
        if ptype in ("int", "float", "number") or lo is not None or hi is not None:
            try:
                num = float(value)
            except (TypeError, ValueError):
                if decl.get("default") is not None:
                    fixes[name] = decl["default"]
                continue
            clamped = num
            if lo is not None:
                clamped = max(float(lo), clamped)
            if hi is not None:
                clamped = min(float(hi), clamped)
            if ptype == "int":
                clamped = int(round(clamped))
            out = clamped
            if out != value or type(out) is not type(value):
                fixes[name] = out
    if not fixes:
        return
    merged = dict(submitted)
    changed = {
        k: {"submitted": submitted.get(k), "coerced": v}
        for k, v in fixes.items() if submitted.get(k) != v
    }
    merged.update(fixes)
    action_instance.parameters = merged
    if changed:
        engine._emit_event(
            "action_corrected",
            actor_id=entity.id,
            action_name=action_instance.action_name,
            data={"reason": "parameter_coerced", "details": changed},
            narrative=(
                f"{entity.name}'s '{action_instance.action_name}' "
                f"parameters were coerced to their declared bounds: "
                + ", ".join(
                    f"{k}: {v['submitted']!r} → {v['coerced']!r}"
                    for k, v in changed.items())
            ),
        )

def _resolve_and_apply(engine, entity_id: str, action_instance: ActionInstance):
    """Resolve an action and apply its effects to the world state."""
    entity = engine.state.get_entity(entity_id)
    if not entity:
        return

    action_def = engine._resolve_action_def(entity, action_instance)
    if action_def is None:
        return
    action_name = action_instance.action_name

    target = None
    if action_instance.target_id:
        target = engine.state.get_entity(action_instance.target_id)
        if target is None:
            # If the action doesn't declare a target_type, silently drop
            # the stray target instead of failing the whole turn. LLMs
            # sometimes hallucinate target_id for target-less actions
            # (e.g. chess `make_move` with target_id='d7'). Treat that
            # as harmless: clear the target and proceed.
            if not action_def.target_type:
                action_instance.target_id = None
            else:
                engine._emit_event(
                    "action_failed",
                    actor_id=entity_id,
                    action_name=action_name,
                    target_id=action_instance.target_id,
                    data={"reason": "unknown_target", "details": {"target_id": action_instance.target_id}},
                    narrative=f"{entity.name} targeted a nonexistent entity ({action_instance.target_id}).",
                )
                return
    # Targeted action with no target supplied -> reject before resolution.
    if action_def.target_type and target is None:
        engine._emit_event(
            "action_failed",
            actor_id=entity_id,
            action_name=action_name,
            data={"reason": "missing_target"},
            narrative=f"{entity.name} attempted {action_name} without a target.",
        )
        return

    # Some domain modules (e.g. Wordle / Hangman / Sudoku Duel) are
    # sealed-tick deduction races and must suppress public chat /
    # speech / action parameters between competitors. We check that
    # intent ONCE here and apply the flag downstream that would
    # otherwise echo these fields.
    #
    # Critically: `reasoning` is NOT shared between agents — it's
    # a UI-only field shown to the spectator (the user watching the
    # match). Suppressing it removes diagnostic insight without
    # adding any privacy. Only `speech` (the chat-panel content)
    # and `parameters` (the secret guess / letter / cell) need to
    # be wiped from broadcast events.
    suppress_chat = _action_suppresses_chat(engine.state, action_name)
    evt_reasoning = action_instance.reasoning   # always exposed
    evt_speech = "" if suppress_chat else action_instance.speech
    evt_parameters = {} if suppress_chat else action_instance.parameters

    # Emit action_attempted event
    attempted_data = {"parameters": evt_parameters, "reasoning": evt_reasoning, "speech": evt_speech}
    if not action_def.broadcast:
        attempted_data["visible_to"] = [
            p for p in (entity_id, action_instance.target_id) if p
        ]
    engine._emit_event(
        "action_attempted",
        actor_id=entity_id,
        target_id=action_instance.target_id,
        action_name=action_name,
        data=attempted_data,
        narrative=f"{entity.name} attempts to {action_name}" +
                  (f" targeting {target.name}" if target else ""),
    )

    # Handle message actions. A non-broadcast message action is a
    # private channel — deliver to the target only, never the room.
    if action_def.message_action and not suppress_chat:
        content = (action_instance.speech or action_instance.reasoning or action_instance.parameters.get("message", ""))
        if content:
            msg = Message(
                sender_id=entity_id,
                sender_name=entity.name,
                content=content,
                message_type="broadcast" if action_def.broadcast else "direct",
                recipient_id=action_instance.target_id,
            )
            engine.state.messages.post(msg)

    # Resolve action — mirror the sequential path's calling convention
    resolution = get_resolution(action_def.resolution_archetype)

    # Build modified properties with status effect bonuses/penalties
    actor_props = dict(entity.properties)
    actor_props["_entity_id"] = entity_id
    actor_mods = engine.state.status_effects.get_modifiers(entity_id)
    for prop, mod_val in actor_mods.items():
        if prop in actor_props and isinstance(actor_props[prop], (int, float)):
            actor_props[prop] = actor_props[prop] + mod_val

    target_props = None
    if target:
        target_props = dict(target.properties)
        target_mods = engine.state.status_effects.get_modifiers(target.id)
        for prop, mod_val in target_mods.items():
            if prop in target_props and isinstance(target_props[prop], (int, float)):
                target_props[prop] = target_props[prop] + mod_val

    # Domain module hooks: modify properties/params before resolution
    resolution_params = dict(action_def.resolution_params) if action_def.resolution_params else {}
    resolution_params["action_name"] = action_instance.action_name  # So resolution can infer direction
    if engine.state.domain_modules:
        actor_props, target_props = engine.state.domain_modules.modify_resolution(
            actor_props, target_props, action_def, engine.state,
        )
        dm_keys = [k for k in actor_props if k.startswith("_")]
        for k in dm_keys:
            resolution_params[k[1:]] = actor_props.pop(k)

    result = resolution.resolve(
        actor_properties=actor_props,
        target_properties=target_props,
        params=resolution_params,
        action_params=action_instance.parameters,
        rng=engine._rng,
    )

    # Apply effects
    effects = action_def.effects_on_success if result.success else action_def.effects_on_failure
    state_changes = engine._apply_effects(effects, entity, target, action_instance.parameters, result)

    # Notify domain modules
    if engine.state.domain_modules:
        # Make action parameters available to the domain module (e.g. bet amount).
        if result and getattr(result, "details", None) is not None:
            result.details["_action_params"] = dict(action_instance.parameters or {})
            # Forward action_instance's target_id and speech so domain
            # modules see what the LLM chose (not just the resolved
            # target entity, which may be filtered). Speech is wiped
            # for chat-suppressed modules — even the action_resolved
            # event's `details` should carry no in-character text.
            result.details["target_id"] = action_instance.target_id
            result.details["speech"] = "" if suppress_chat else action_instance.speech
        domain_changes = engine.state.domain_modules.post_resolution(
            entity_id, action_instance.action_name, result.success, result, engine.state,
        )
        # Scrub parameter leak from result.details so the downstream
        # action_resolved event broadcast doesn't carry the secret
        # input (e.g. the guess word) for chat-suppressed modules.
        if suppress_chat and getattr(result, "details", None) is not None:
            result.details.pop("_action_params", None)
        # Mirror the sequential path: surface each domain change as an
        # event so the live UI + replay see board updates, captures,
        # check/checkmate, etc. Without this, games like chess emit
        # internal state updates that never reach the viz.
        for change in domain_changes or []:
            # Same shape as the tick_all path: prefer event_type,
            # fall back to type, default to domain_update.
            evt_type = (
                change.get("event_type")
                or change.get("type")
                or "domain_update"
            )
            payload = change.get("data") if isinstance(change.get("data"), dict) else change
            engine._emit_event(
                evt_type,
                actor_id=change.get("actor_id") or entity_id,
                target_id=change.get("target_id"),
                action_name=action_instance.action_name,
                data=payload,
                narrative=change.get("narrative", f"Domain update: {change.get('type', 'unknown')}"),
            )

    # Generate narrative
    narrative = ""
    if engine.narrative_fn:
        try:
            narrative = engine.narrative_fn(entity, target, action_def, action_instance, result, state_changes)
        except Exception:
            narrative = f"{entity.name} {'successfully' if result.success else 'failed to'} {action_name}"
    if not narrative:
        narrative = result.narrative or f"{entity.name} {'successfully' if result.success else 'failed to'} {action_name}"

    # Notify brain
    if engine.outcome_fn:
        engine.outcome_fn(entity_id, action_name, result.success, narrative, result.details)

    # Emit action_resolved event. Non-broadcast actions (hidden votes,
    # night kills) carry visible_to so every downstream consumer —
    # perception, transcripts, analysis — can restrict who sees them.
    resolved_data = {
        "success": result.success,
        "magnitude": result.magnitude,
        "details": result.details,
        "state_changes": state_changes,
    }
    if not action_def.broadcast:
        resolved_data["visible_to"] = [
            p for p in (entity_id, action_instance.target_id) if p
        ]
    engine._emit_event(
        "action_resolved",
        actor_id=entity_id,
        target_id=action_instance.target_id,
        action_name=action_name,
        data=resolved_data,
        narrative=narrative,
    )

    # Surface public speech as a dedicated agent_message event ONLY
    # for actions explicitly declared as `message_action: true`.
    #
    # Previously any action could leak a `speech` field into the
    # chat panel, which meant LLMs spontaneously chatted during
    # non-social games (Battleship, Connect Five, …) by populating
    # the optional speech field on their move actions. The rule now
    # is strict: if the schema doesn't mark the action as a message
    # action, no chat event is emitted — period. Domain modules can
    # additionally opt out via `suppress_chat = True`.
    spoken_pub = ""
    if action_def.message_action and not suppress_chat:
        spoken_pub = (action_instance.speech or "").strip()
        if not spoken_pub:
            spoken_pub = (action_instance.reasoning or "").strip()
    if spoken_pub:
        target_name = target.name if target else None
        engine._emit_event(
            "agent_message",
            actor_id=entity_id,
            target_id=action_instance.target_id,
            action_name=action_name,
            data={
                "sender_name": entity.name,
                "content": spoken_pub,
                "message_type": action_instance.parameters.get("message_type", "broadcast"),
                "recipient_name": target_name,
            },
            narrative=f'{entity.name} says: "{spoken_pub[:200]}"',
        )

    # Track action for crowd agent promotion/demotion
    if engine.state.crowd_agents:
        engine.state.crowd_agents.record_action(entity_id, action_instance.action_name)

    # Record action history
    round_num = engine.state.temporal.current_round
    engine.state.action_history.record(entity_id, action_instance.action_name, result.success, round_num)


__all__ = ["_resolve_action_def", "_coerce_action_params", "_resolve_and_apply"]
