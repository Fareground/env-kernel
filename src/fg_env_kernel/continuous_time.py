"""Continuous time / event-driven simulation mode.

In continuous mode, simulation time advances in arbitrary increments rather
than fixed rounds. Agents schedule actions with durations; the engine pops
events from a priority queue and advances time to each event's fire time.

Backward-compatible: DISCRETE mode (the default) is unaffected.
"""
import heapq
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scheduled events
# ---------------------------------------------------------------------------

@dataclass(order=False)
class ScheduledEvent:
    """An event scheduled to fire at a specific simulation time."""
    fire_time: float                # When this event fires
    priority: int = 0               # Lower = higher priority (for tie-breaking)
    event_type: str = "agent_turn"  # "agent_turn", "environment", "custom"
    entity_id: Optional[str] = None # Agent associated with this event
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    cancelled: bool = False

    def __lt__(self, other: "ScheduledEvent") -> bool:
        # Deterministic ordering: time, then priority, then entity_id.
        # Without the entity_id key, two agents scheduled at the same
        # (fire_time, priority) — common once float-drift from staggered
        # turns collides — would pop in heap-internal order, which is not
        # reproducible. entity_id is stable across runs/forks; id (a random
        # uuid) is only the last resort for entity-less events.
        if self.fire_time != other.fire_time:
            return self.fire_time < other.fire_time
        if self.priority != other.priority:
            return self.priority < other.priority
        if (self.entity_id or "") != (other.entity_id or ""):
            return (self.entity_id or "") < (other.entity_id or "")
        return self.id < other.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduledEvent):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict:
        return {
            "fire_time": self.fire_time,
            "priority": self.priority,
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "data": self.data,
            "id": self.id,
            "cancelled": self.cancelled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledEvent":
        return cls(
            fire_time=data["fire_time"],
            priority=data.get("priority", 0),
            event_type=data.get("event_type", "agent_turn"),
            entity_id=data.get("entity_id"),
            data=data.get("data", {}),
            id=data.get("id", uuid.uuid4().hex[:8]),
            cancelled=data.get("cancelled", False),
        )


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------

class EventQueue:
    """Priority queue for scheduled events, ordered by fire_time then priority."""

    def __init__(self):
        self._heap: List[ScheduledEvent] = []
        self._event_map: Dict[str, ScheduledEvent] = {}

    def schedule(self, event: ScheduledEvent):
        """Add an event to the queue."""
        heapq.heappush(self._heap, event)
        self._event_map[event.id] = event

    def pop(self) -> Optional[ScheduledEvent]:
        """Pop the next event. Skips cancelled events. Returns None if empty."""
        while self._heap:
            event = heapq.heappop(self._heap)
            self._event_map.pop(event.id, None)
            if not event.cancelled:
                return event
        return None

    def peek(self) -> Optional[ScheduledEvent]:
        """Look at the next event without removing it. Skips cancelled events."""
        while self._heap and self._heap[0].cancelled:
            event = heapq.heappop(self._heap)
            self._event_map.pop(event.id, None)
        return self._heap[0] if self._heap else None

    def cancel(self, event_id: str) -> bool:
        """Cancel a scheduled event. Returns True if found and cancelled."""
        event = self._event_map.get(event_id)
        if event and not event.cancelled:
            event.cancelled = True
            return True
        return False

    def cancel_for_entity(self, entity_id: str) -> int:
        """Cancel all events for a specific entity. Returns count cancelled."""
        count = 0
        for event in self._event_map.values():
            if event.entity_id == entity_id and not event.cancelled:
                event.cancelled = True
                count += 1
        return count

    def __len__(self) -> int:
        """Count of non-cancelled events (approximate — includes cancelled in heap)."""
        return sum(1 for e in self._event_map.values() if not e.cancelled)

    def is_empty(self) -> bool:
        """Check if queue has any non-cancelled events."""
        while self._heap and self._heap[0].cancelled:
            event = heapq.heappop(self._heap)
            self._event_map.pop(event.id, None)
        return len(self._heap) == 0

    def get_events_for_entity(self, entity_id: str) -> List[ScheduledEvent]:
        """Get all pending (non-cancelled) events for an entity."""
        return [
            e for e in self._event_map.values()
            if e.entity_id == entity_id and not e.cancelled
        ]

    def to_dict(self) -> dict:
        """Serialize the queue."""
        return {
            "events": [e.to_dict() for e in self._event_map.values() if not e.cancelled],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventQueue":
        queue = cls()
        for edata in data.get("events", []):
            event = ScheduledEvent.from_dict(edata)
            if not event.cancelled:
                queue.schedule(event)
        return queue


# ---------------------------------------------------------------------------
# Continuous temporal model
# ---------------------------------------------------------------------------

class ContinuousTemporalModel:
    """
    Continuous-time simulation model with event-driven scheduling.

    Instead of discrete rounds with phases, time is a float that advances
    to the next scheduled event. Agents are scheduled for turns at specific
    times; action durations determine when they can act again.

    The engine's continuous loop:
    1. Pop next event from queue
    2. Advance current_time to event.fire_time
    3. Process event (agent turn, environment event, etc.)
    4. Schedule follow-up events (e.g., next turn after action duration)
    """

    DEFAULT_ACTION_DURATION = 1.0
    DEFAULT_TURN_INTERVAL = 1.0
    # An action MUST advance the clock. A configured (or defaulted) duration of
    # zero or negative would schedule the next turn at the current time, so the
    # event loop pops turns forever without time ever reaching max_time — a hard
    # hang. Every duration is floored to this strictly-positive epsilon so time
    # always moves forward; a misconfigured `action_durations: {move: 0}` costs
    # dense turns, never a wedge.
    MIN_ACTION_DURATION = 1e-9

    # Hard backstop on events processed in one run — the continuous analogue
    # of the discrete loop's max_rounds. The MIN_ACTION_DURATION floor
    # guarantees forward progress; this guarantees *termination* even when a
    # near-zero duration would otherwise need astronomically many tiny steps
    # to reach max_time.
    DEFAULT_MAX_EVENTS = 1_000_000

    def __init__(
        self,
        action_durations: Optional[Dict[str, float]] = None,
        default_turn_interval: float = 1.0,
        max_time: float = 100.0,
        environment_interval: float = 1.0,
        max_events: Optional[int] = None,
    ):
        self.queue = EventQueue()
        self.current_time: float = 0.0
        self.max_time: float = max_time
        self.max_events: int = max_events if max_events is not None else self.DEFAULT_MAX_EVENTS
        self.action_durations: Dict[str, float] = action_durations or {}
        self.default_turn_interval: float = default_turn_interval
        # Cadence (in sim-time units) of recurring "environment" events that
        # drive the world clock forward — physics integration, property drift,
        # world events. 0 or negative disables periodic environment ticks (the
        # world then only changes on agent turns). This is what makes time, not
        # a round counter, advance the continuous world.
        self.environment_interval: float = environment_interval
        self._paused: bool = False
        self._processed_count: int = 0

    def schedule_agent_turn(
        self,
        entity_id: str,
        at_time: Optional[float] = None,
        priority: int = 0,
    ) -> ScheduledEvent:
        """Schedule an agent's next turn."""
        fire_time = at_time if at_time is not None else self.current_time
        event = ScheduledEvent(
            fire_time=fire_time,
            priority=priority,
            event_type="agent_turn",
            entity_id=entity_id,
        )
        self.queue.schedule(event)
        return event

    def schedule_environment_event(
        self,
        at_time: float,
        data: Dict[str, Any],
        entity_id: Optional[str] = None,
        priority: int = 5,
    ) -> ScheduledEvent:
        """Schedule an environment event (weather, spawning, etc.)."""
        event = ScheduledEvent(
            fire_time=at_time,
            priority=priority,
            event_type="environment",
            entity_id=entity_id,
            data=data,
        )
        self.queue.schedule(event)
        return event

    def schedule_custom_event(
        self,
        at_time: float,
        event_type: str,
        data: Dict[str, Any],
        entity_id: Optional[str] = None,
        priority: int = 3,
    ) -> ScheduledEvent:
        """Schedule a custom event."""
        event = ScheduledEvent(
            fire_time=at_time,
            priority=priority,
            event_type=event_type,
            entity_id=entity_id,
            data=data,
        )
        self.queue.schedule(event)
        return event

    def get_action_duration(self, action_name: str) -> float:
        """Get the duration of an action (from configured durations or default),
        floored to a strictly-positive minimum so the clock always advances."""
        configured = self.action_durations.get(action_name, self.DEFAULT_ACTION_DURATION)
        return max(self.MIN_ACTION_DURATION, configured)

    def schedule_next_turn_after_action(
        self,
        entity_id: str,
        action_name: str,
        action_duration: Optional[float] = None,
    ) -> ScheduledEvent:
        """Schedule an agent's next turn after completing an action."""
        duration = action_duration if action_duration is not None else self.get_action_duration(action_name)
        # Floor an explicitly-passed duration too — a caller-supplied 0 is just
        # as capable of wedging the loop as a configured one.
        duration = max(self.MIN_ACTION_DURATION, duration)
        next_time = self.current_time + duration
        return self.schedule_agent_turn(entity_id, at_time=next_time)

    def pop_next_event(self) -> Optional[ScheduledEvent]:
        """Pop and process the next event, advancing time. Returns None if done."""
        if self._paused:
            return None
        if self._processed_count >= self.max_events:
            return None  # termination backstop: too many events this run
        event = self.queue.pop()
        if event is None:
            return None
        if event.fire_time > self.max_time:
            return None
        self.current_time = event.fire_time
        self._processed_count += 1
        return event

    def initialize_agents(self, agent_ids: List[str], stagger: float = 0.0):
        """Schedule initial turns for all agents. Optional stagger offsets each agent."""
        for i, agent_id in enumerate(agent_ids):
            self.schedule_agent_turn(agent_id, at_time=i * stagger)

    def schedule_environment_tick(self, at_time: Optional[float] = None) -> Optional[ScheduledEvent]:
        """Schedule the next recurring environment tick (physics/dynamics clock).

        Fires at ``at_time`` (default: current_time + environment_interval).
        Returns None when periodic ticks are disabled (``environment_interval``
        <= 0) or the tick would land past ``max_time``."""
        if self.environment_interval <= 0:
            return None
        fire = at_time if at_time is not None else self.current_time + self.environment_interval
        if fire > self.max_time:
            return None
        return self.schedule_environment_event(at_time=fire, data={"recurring": True}, priority=5)

    def pause(self):
        """Pause the continuous simulation."""
        self._paused = True

    def resume(self):
        """Resume the continuous simulation."""
        self._paused = False

    def is_paused(self) -> bool:
        return self._paused

    def is_empty(self) -> bool:
        """True when no events are scheduled (a fresh, un-bootstrapped model)."""
        return self.queue.is_empty()

    def is_finished(self) -> bool:
        """Check if simulation should end (no events or past max_time)."""
        next_event = self.queue.peek()
        if next_event is None:
            return True
        return next_event.fire_time > self.max_time

    @property
    def processed_count(self) -> int:
        return self._processed_count

    def to_dict(self) -> dict:
        return {
            "current_time": self.current_time,
            "max_time": self.max_time,
            "max_events": self.max_events,
            "action_durations": dict(self.action_durations),
            "default_turn_interval": self.default_turn_interval,
            "environment_interval": self.environment_interval,
            "queue": self.queue.to_dict(),
            "processed_count": self._processed_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContinuousTemporalModel":
        model = cls(
            action_durations=data.get("action_durations", {}),
            default_turn_interval=data.get("default_turn_interval", 1.0),
            max_time=data.get("max_time", 100.0),
            environment_interval=data.get("environment_interval", 1.0),
            max_events=data.get("max_events"),
        )
        model.current_time = data.get("current_time", 0.0)
        model._processed_count = data.get("processed_count", 0)
        model.queue = EventQueue.from_dict(data.get("queue", {}))
        return model
