"""Simulation event tracking."""
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional


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

    Retained events are also indexed by round and by (type, round), so the
    per-round reads and per-type counts that termination checks make every
    round cost the size of the answer rather than the whole history. ``emit``
    and trimming keep the index current; ``_rebuild_index`` recomputes it if
    ``_events`` is ever replaced wholesale.
    """

    def __init__(self, max_events: Optional[int] = None):
        self._events: List[SimEvent] = []
        self._max_events = max_events
        self._rebuild_index()

    # -- index -------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Recompute every index from ``_events``."""
        self._by_round: Dict[Any, Deque[SimEvent]] = {}
        self._type_rounds: Dict[str, Counter] = {}
        self._type_counted: Dict[str, int] = {}
        self._type_max_round: Dict[str, int] = {}
        for event in self._events:
            self._index(event)

    def _index(self, event: SimEvent) -> None:
        self._by_round.setdefault(event.round_number, deque()).append(event)
        rnd = _counted_round(event.round_number)
        if rnd is None:
            return
        etype = event.event_type
        self._type_rounds.setdefault(etype, Counter())[rnd] += 1
        self._type_counted[etype] = self._type_counted.get(etype, 0) + 1
        if rnd > self._type_max_round.get(etype, 0):
            self._type_max_round[etype] = rnd

    def _unindex_oldest(self, event: SimEvent) -> None:
        """Drop ``event`` (the oldest retained event) from the index."""
        bucket = self._by_round[event.round_number]
        bucket.popleft()
        if not bucket:
            del self._by_round[event.round_number]
        rnd = _counted_round(event.round_number)
        if rnd is None:
            return
        etype = event.event_type
        rounds = self._type_rounds[etype]
        rounds[rnd] -= 1
        if not rounds[rnd]:
            del rounds[rnd]
        self._type_counted[etype] -= 1
        # _type_max_round may now overstate the latest round; count_type
        # then takes its exact-sum path, so counts stay correct.

    # -- writes ------------------------------------------------------------

    def emit(self, event: SimEvent):
        """Append an event (trimming oldest if a cap is set)."""
        self._events.append(event)
        self._index(event)
        if self._max_events is not None and len(self._events) > self._max_events:
            # Drop oldest in a batch to keep this amortized O(1).
            overflow = len(self._events) - self._max_events
            for dropped in self._events[:overflow]:
                self._unindex_oldest(dropped)
            del self._events[:overflow]

    # -- reads -------------------------------------------------------------

    def get_all(self) -> List[SimEvent]:
        """Get all events."""
        return list(self._events)

    def get_round(self, round_number: int) -> List[SimEvent]:
        """Get events for a specific round."""
        return list(self._by_round.get(round_number, ()))

    def count_type(self, event_type: str, through_round: int) -> int:
        """Count retained ``event_type`` events stamped in rounds
        ``1..through_round`` inclusive (round-0 setup events excluded).

        Constant time when ``through_round`` is at or past the latest round
        the type was seen in (a check at the current round); otherwise an
        exact sum over that type's distinct rounds."""
        counted = self._type_counted.get(event_type, 0)
        if not counted:
            return 0
        if through_round >= self._type_max_round.get(event_type, 0):
            return counted
        return sum(
            n for rnd, n in self._type_rounds[event_type].items()
            if rnd <= through_round
        )

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


def _counted_round(round_number: Any) -> Optional[int]:
    """The positive integer round an event counts toward, or None.

    Matches what a ``range(1, current + 1)`` scan comparing with ``==`` would
    count: ``3`` and ``3.0`` are round 3; ``0``, ``2.5`` and ``None`` count
    toward no round."""
    if not isinstance(round_number, (int, float)) or round_number < 1:
        return None
    if isinstance(round_number, float) and not round_number.is_integer():
        return None
    return int(round_number)
