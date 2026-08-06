"""Trigger runtime — emits events + walks the trigger cascade.

Canonical implementations of ``SimulationEngine._emit_event``,
``_fire_trigger`` and the ``_entities_snapshot`` payload they carry.
Extracted so the engine class stays focused on the tick loop; the engine's
methods are now 1-line delegations into this module.

When an event is emitted, schema-declared triggers matching that event type
fire their own effect chains, which can emit more events — capped at depth 4
to prevent runaway. The cascade depth counter uses ``threading.local``
(``engine._cascade_tls``) so parallel-phase threads each get their own counter
and don't race. The ``TriggerEngine`` in ``fg_env_kernel.triggers`` owns the
matching/dispatch logic; only the engine-side glue lives here.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..event import SimEvent
from .engine import _coerce_effects

if TYPE_CHECKING:
    from ..triggers import TriggerSpec
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)


def emit_event(
    engine: "SimulationEngine",
    event_type: str,
    actor_id: Optional[str] = None,
    target_id: Optional[str] = None,
    action_name: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    narrative: str = "",
) -> None:
    """Emit an event to the transcript and walk the trigger cascade.

    Stable public API — prefer this over reaching into the engine method.
    """
    return _emit_event(
        engine,
        event_type=event_type,
        actor_id=actor_id,
        target_id=target_id,
        action_name=action_name,
        data=data,
        narrative=narrative,
    )


def _entities_snapshot(engine,) -> Dict[str, Any]:
    """JSON-safe {entity_id: {name,type,alive,properties[,resources]}} —
    the shape every visualization component reads."""
    import json as _json

    snap: Dict[str, Any] = {}
    for eid, entity in engine.state.entities.items():
        ent: Dict[str, Any] = {
            "name": entity.name,
            "type": entity.entity_type,
            "alive": entity.alive,
            "properties": {},
        }
        for pname, pval in entity.properties.items():
            try:
                _json.dumps(pval)
                ent["properties"][pname] = pval
            except (TypeError, ValueError):
                ent["properties"][pname] = str(pval)
        resources = {}
        for res_name, pool in engine.state.resources.items():
            amt = pool.get(eid)
            if amt is not None and amt != 0:
                resources[res_name] = amt
        if resources:
            ent["resources"] = resources
        snap[eid] = ent
    return snap

def _emit_event(engine,
    event_type: str,
    actor_id: str = None,
    target_id: str = None,
    action_name: str = None,
    data: dict = None,
    narrative: str = "",
):
    """Emit an event to the transcript and optional streaming callback."""
    # Safely get current phase name (index may be past end after advance_phase)
    temporal = engine.state.temporal
    if temporal.phases and 0 <= temporal.current_phase_index < len(temporal.phases):
        phase_name = temporal.phases[temporal.current_phase_index].name
    elif temporal.phases:
        phase_name = temporal.phases[-1].name  # Use last phase as fallback
    else:
        phase_name = ""

    event = SimEvent(
        event_type=event_type,
        round_number=engine.state.temporal.current_round,
        phase=phase_name,
        actor_id=actor_id,
        target_id=target_id,
        action_name=action_name,
        data=data or {},
        narrative=narrative,
    )
    engine.state.event_log.emit(event)

    # Stream to real-time callback if set
    if engine.on_event:
        try:
            engine.on_event(event.to_dict())
        except Exception as e:
            logger.debug(f"on_event callback error (non-fatal): {e}")
        # Live-state contract for visualizations: entity properties +
        # resources after every resolved action / phase step.
        if engine.emit_state_snapshots and event_type in ("action_resolved", "phase_handler"):
            try:
                engine.on_event({
                    "event_type": "state_snapshot",
                    "round_number": event.round_number,
                    "phase": event.phase,
                    "actor_id": None,
                    "target_id": None,
                    "action_name": None,
                    "data": {"entities": engine._entities_snapshot()},
                    "narrative": "",
                    "timestamp": "",
                })
            except Exception as e:  # noqa: BLE001 — viz feed must never break the sim
                logger.debug(f"state_snapshot emission error (non-fatal): {e}")

    # Tier 5a — Triggered effects. Walk schema-declared triggers
    # and fire matching ones. Guarded by `_in_trigger_cascade` to
    # prevent infinite recursion (a trigger that emits the same
    # event it listens to). Cascading triggers are allowed but
    # capped at MAX_DEPTH.
    triggers = getattr(engine, "triggers", None)
    depth = getattr(engine._cascade_tls, "depth", 0)
    if triggers and event_type and depth < 4:
        try:
            matching = triggers.matching(
                event_type, data or {}, actor_id,
                engine.state.temporal.current_round,
            )
            if matching:
                engine._cascade_tls.depth = depth + 1
                try:
                    for spec in matching:
                        engine._fire_trigger(spec, actor_id, target_id, data or {})
                finally:
                    engine._cascade_tls.depth = depth
        except Exception:
            logger.exception("trigger evaluation failed for event %s", event_type)

def _fire_trigger(engine,
    spec: "TriggerSpec",
    actor_id: Optional[str],
    target_id: Optional[str],
    event_data: Dict[str, Any],
) -> None:
    """Apply a triggered effect chain. The triggering event's
    actor/target/data become the new effect context — so effects
    can reference `$event.field` (mapped to params here)."""
    actor = engine.state.get_entity(actor_id) if actor_id else None
    target = engine.state.get_entity(target_id) if target_id else None
    effects = _coerce_effects(spec.effect)
    if not effects:
        return
    # We pass `event_data` through the `params` channel so $params.X
    # in trigger effects can read the triggering event's payload.
    # Also stash actor_id into params['actor'] for convenience.
    params = dict(event_data)
    params.setdefault("_trigger", spec.name or "")
    try:
        engine._apply_effects(effects, actor, target, params, None)
        engine.triggers.record_fired(
            spec, actor_id, engine.state.temporal.current_round,
        )
    except Exception:
        logger.exception("trigger fire failed: %s", spec.name)


__all__ = ["emit_event", "_emit_event", "_fire_trigger", "_entities_snapshot"]
