"""Simulation event tracking."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SimEvent:
    """A single event in the simulation transcript."""
    event_type: str              # "action_attempted", "action_resolved", "state_change", etc.
    round_number: int
    phase: str
    actor_id: Optional[str] = None
    target_id: Optional[str] = None
    action_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type,
            "round_number": self.round_number,
            "phase": self.phase,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "action_name": self.action_name,
            "data": self.data,
            "narrative": self.narrative,
            "timestamp": self.timestamp,
        }


class EventLog:
    """Append-only event log for a simulation.

    By default the log is unbounded — analytics, snapshots, and transcript
    export read the full history after a run. For long-running or streamed
    simulations (where each event is already pushed out via the engine's
    ``on_event`` callback), pass ``max_events`` to cap in-memory retention to
    the most recent N events and bound memory growth.
    """

    def __init__(self, max_events: Optional[int] = None):
        self._events: List[SimEvent] = []
        self._max_events = max_events

    def emit(self, event: SimEvent):
        """Append an event (trimming oldest if a cap is set)."""
        self._events.append(event)
        if self._max_events is not None and len(self._events) > self._max_events:
            # Drop oldest in a batch to keep this amortized O(1).
            overflow = len(self._events) - self._max_events
            del self._events[:overflow]

    def get_all(self) -> List[SimEvent]:
        """Get all events."""
        return list(self._events)

    def get_round(self, round_number: int) -> List[SimEvent]:
        """Get events for a specific round."""
        return [e for e in self._events if e.round_number == round_number]

    def visible_for(self, observer_id: str) -> List[SimEvent]:
        """Events ``observer_id`` may see.

        An event stamped with ``data["visible_to"]`` (non-broadcast actions —
        night kills, hidden votes) is filtered to the listed participants;
        everything else is public. ANY consumer that feeds events to an
        agent's perception or another player's view must read through this,
        never ``get_all``/``get_round`` directly."""
        out: List[SimEvent] = []
        for e in self._events:
            allowed = e.data.get("visible_to") if isinstance(e.data, dict) else None
            if allowed is None or observer_id in allowed:
                out.append(e)
        return out

    def get_by_actor(self, actor_id: str) -> List[SimEvent]:
        """Get events by a specific actor."""
        return [e for e in self._events if e.actor_id == actor_id]

    def get_by_type(self, event_type: str) -> List[SimEvent]:
        """Get events of a specific type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_recent(self, n: int = 10) -> List[SimEvent]:
        """Get the N most recent events."""
        return self._events[-n:]

    def __len__(self) -> int:
        return len(self._events)

    def to_transcript(self) -> List[dict]:
        """Export full transcript as list of dicts."""
        return [e.to_dict() for e in self._events]
