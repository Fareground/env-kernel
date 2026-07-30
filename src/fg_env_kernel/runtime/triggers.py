"""Trigger runtime — emits events + walks the trigger cascade.

## Current state

``SimulationEngine._emit_event`` and ``SimulationEngine._fire_trigger``
together implement the trigger cascade — when an event is emitted,
schema-declared triggers matching that event type fire their own
effect chains, which can emit more events (capped at depth=4 to
prevent runaway).

The cascade depth counter uses ``threading.local`` (fixed in P1 — see
``self._cascade_tls``) so parallel-phase threads each get their own
counter and don't race.

## Migration path

When extracting:
  1. ``emit_event`` becomes a module-level function taking engine
  2. ``_emit_event`` becomes a 1-line delegation
  3. The TriggerEngine in ``fg_env_kernel.triggers`` already owns
     the matching/dispatch logic — only the engine-side glue moves here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .engine import SimulationEngine


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

    Stable public API. Currently delegates to the engine class method.
    """
    return engine._emit_event(
        event_type=event_type,
        actor_id=actor_id,
        target_id=target_id,
        action_name=action_name,
        data=data,
        narrative=narrative,
    )


__all__ = ["emit_event"]
