"""World events -- environmental forces that act on the simulation.

World events are random, scheduled, or conditional occurrences that affect
the simulation without requiring agent action.  Examples: drought, plague,
festival, earthquake.
"""
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .action import Effect


@dataclass
class WorldEventDefinition:
    """Blueprint for a world event."""
    name: str
    description: str = ""
    trigger_type: str = "random"        # "random", "scheduled", "conditional"

    # Random trigger
    probability: float = 0.0            # Per-round probability (0.1 = 10%)

    # Scheduled trigger
    trigger_rounds: List[int] = field(default_factory=list)

    # Conditional trigger
    trigger_condition: Optional[Dict[str, Any]] = None
    # {"check": "resource_below", "resource": "food", "threshold": 10}
    # {"check": "entity_count_below", "entity_type": "warrior", "threshold": 2}

    # Effects applied when triggered
    effects: List[Effect] = field(default_factory=list)

    # Duration: 0 = instant, >0 = active for N rounds
    duration: int = 0

    # Cooldown: minimum rounds between triggers
    cooldown: int = 0

    # Targeting: None = global (affects all entities), else entity_type name
    target_type: Optional[str] = None

    # Cascading events: trigger other events when this one fires
    cascade_events: List[str] = field(default_factory=list)  # Names of events to trigger
    cascade_delay: int = 0              # Rounds to delay cascade (0 = same round)
    cascade_probability: float = 1.0    # Probability each cascade fires


@dataclass
class DynamicsRule:
    """
    A per-round resource growth/depletion rule.

    Applied automatically each round to model natural processes:
    resource regeneration, population decay, environmental shifts.
    """
    name: str
    resource: str               # Resource name to modify
    rate_per_round: float       # Positive = growth, negative = depletion
    target_type: Optional[str] = None  # Entity type to affect, None = global pool
    condition: Optional[Dict[str, Any]] = None  # Optional condition to check
    min_value: float = 0.0      # Floor (won't go below this)
    max_value: float = float('inf')  # Cap (won't go above this)


@dataclass
class TriggeredEvent:
    """A world event that was triggered this round."""
    definition: WorldEventDefinition
    round_triggered: int
    affected_entities: List[str] = field(default_factory=list)


@dataclass
class ActiveEvent:
    """A world event currently in effect (duration > 0)."""
    definition: WorldEventDefinition
    started_round: int
    remaining_rounds: int


class WorldEventEngine:
    """Evaluates and triggers world events each round."""

    def __init__(
        self,
        event_definitions: List[WorldEventDefinition],
        rng: Optional[random.Random] = None,
    ):
        self.definitions = {e.name: e for e in event_definitions}
        self.rng = rng or random.Random()
        self._last_triggered: Dict[str, int] = {}   # name -> last triggered round
        self._active_events: List[ActiveEvent] = []  # currently active (duration > 0)

    def evaluate(self, state, round_number: int) -> List[TriggeredEvent]:
        """Check all events, return those that triggered this round."""
        triggered = []

        # First, expire active events
        still_active = []
        for ae in self._active_events:
            ae.remaining_rounds -= 1
            if ae.remaining_rounds > 0:
                still_active.append(ae)
        self._active_events = still_active

        # Evaluate each definition
        for defn in self.definitions.values():
            # Check cooldown
            last = self._last_triggered.get(defn.name)
            if last is not None and defn.cooldown > 0:
                if round_number - last < defn.cooldown:
                    continue

            should_trigger = False

            if defn.trigger_type == "random":
                should_trigger = self.rng.random() < defn.probability

            elif defn.trigger_type == "scheduled":
                should_trigger = round_number in defn.trigger_rounds

            elif defn.trigger_type == "conditional":
                should_trigger = self._check_condition(defn.trigger_condition, state)

            if should_trigger:
                # Find affected entities
                affected = self._get_affected_entities(defn, state)

                event = TriggeredEvent(
                    definition=defn,
                    round_triggered=round_number,
                    affected_entities=affected,
                )
                triggered.append(event)
                self._last_triggered[defn.name] = round_number

                # Track duration-based events
                if defn.duration > 0:
                    self._active_events.append(ActiveEvent(
                        definition=defn,
                        started_round=round_number,
                        remaining_rounds=defn.duration,
                    ))

        return triggered

    def get_active_events(self) -> List[ActiveEvent]:
        """Return events currently in effect."""
        return list(self._active_events)

    def _check_condition(self, condition: Optional[Dict], state) -> bool:
        """Evaluate a conditional trigger against world state."""
        if not condition:
            return False

        check = condition.get("check", "")

        if check == "resource_below":
            resource_name = condition.get("resource", "")
            threshold = condition.get("threshold", 0)
            pool = state.resources.get(resource_name)
            if not pool:
                return False
            total = sum(pool.holdings.values()) + pool.unallocated
            return total < threshold

        elif check == "entity_count_below":
            entity_type = condition.get("entity_type", "")
            threshold = condition.get("threshold", 0)
            entities = state.get_entities_by_type(entity_type)
            alive_count = sum(1 for e in entities if e.alive)
            return alive_count < threshold

        return False

    def _get_affected_entities(self, defn: WorldEventDefinition, state) -> List[str]:
        """Get entity IDs affected by this event."""
        if defn.target_type:
            entities = state.get_entities_by_type(defn.target_type)
            return [e.id for e in entities if e.alive]
        else:
            # Global: all alive entities
            return [e.id for e in state.entities.values() if e.alive]


class WorldDynamicsEngine:
    """
    Wraps WorldEventEngine with additional per-round dynamics:
    - Resource growth/depletion rules
    - Cascading event chains
    - Environmental shifts

    Call `tick()` once per round instead of `WorldEventEngine.evaluate()`.
    """

    def __init__(
        self,
        event_engine: WorldEventEngine,
        dynamics_rules: Optional[List[DynamicsRule]] = None,
    ):
        self.event_engine = event_engine
        self.dynamics_rules = dynamics_rules or []
        self._cascade_queue: List[tuple] = []  # (event_name, trigger_round)

    def tick(self, state, round_number: int) -> List[TriggeredEvent]:
        """
        Run one round of world dynamics.

        1. Apply per-round resource growth/depletion
        2. Evaluate events (including scheduled cascades)
        3. Queue new cascades from triggered events
        """
        changes = []

        # 1. Apply dynamics rules
        for rule in self.dynamics_rules:
            if rule.condition and not self.event_engine._check_condition(rule.condition, state):
                continue
            self._apply_dynamics_rule(rule, state)

        # 2. Process cascade queue — fire events whose delay has elapsed
        remaining_cascades = []
        for cascade_name, fire_round in self._cascade_queue:
            if round_number >= fire_round:
                defn = self.event_engine.definitions.get(cascade_name)
                if defn:
                    affected = self.event_engine._get_affected_entities(defn, state)
                    cascade_event = TriggeredEvent(
                        definition=defn,
                        round_triggered=round_number,
                        affected_entities=affected,
                    )
                    changes.append(cascade_event)
                    if defn.duration > 0:
                        self.event_engine._active_events.append(ActiveEvent(
                            definition=defn,
                            started_round=round_number,
                            remaining_rounds=defn.duration,
                        ))
            else:
                remaining_cascades.append((cascade_name, fire_round))
        self._cascade_queue = remaining_cascades

        # 3. Evaluate normal events
        triggered = self.event_engine.evaluate(state, round_number)
        changes.extend(triggered)

        # 4. Queue cascades from newly triggered events
        for event in triggered:
            defn = event.definition
            if defn.cascade_events:
                for cascade_name in defn.cascade_events:
                    if self.event_engine.rng.random() < defn.cascade_probability:
                        fire_round = round_number + defn.cascade_delay
                        self._cascade_queue.append((cascade_name, fire_round))

        return changes

    def _apply_dynamics_rule(self, rule: DynamicsRule, state):
        """Apply a single dynamics rule to the world state."""
        pool = state.resources.get(rule.resource)
        if not pool:
            return

        if rule.target_type:
            # Apply to each entity of the target type
            entities = state.get_entities_by_type(rule.target_type)
            for entity in entities:
                if not entity.alive:
                    continue
                current = pool.get(entity.id)
                new_val = max(rule.min_value, min(rule.max_value, current + rule.rate_per_round))
                delta = new_val - current
                if delta > 0:
                    pool.add(entity.id, delta)
                elif delta < 0:
                    pool.subtract(entity.id, abs(delta))
        else:
            # Apply to global unallocated pool
            new_val = max(rule.min_value, min(rule.max_value, pool.unallocated + rule.rate_per_round))
            pool.unallocated = new_val

    def evaluate(self, state, round_number: int) -> List[TriggeredEvent]:
        """Interface-compatible alias for tick().

        SimulationEngine calls `world_event_engine.evaluate()`, so this
        ensures WorldDynamicsEngine is a drop-in replacement for
        WorldEventEngine.
        """
        return self.tick(state, round_number)

    def get_active_events(self) -> List[ActiveEvent]:
        """Delegate to event engine."""
        return self.event_engine.get_active_events()
