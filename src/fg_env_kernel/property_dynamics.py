"""Autonomous environment dynamics -- property drift, conditional spawning, cascades.

The world evolves on its own each round, independent of agent actions.
Properties drift (weather changes, market prices fluctuate, resources regenerate),
entities spawn when conditions are met, and cascading dynamics chain together
(drought -> crop failure -> famine).
"""
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropertyDriftRule:
    """Per-round property drift rule. Modifies entity properties automatically.

    Drift types:
    - "linear": constant per-round change (rate per round)
    - "sinusoidal": oscillates between -amplitude and +amplitude with given period
    - "random_walk": Gaussian random step each round (mean=0, std=variance)
    - "mean_revert": drifts toward `mean` at `rate` per round
    """
    name: str
    target_type: str                        # Entity type to affect
    property_field: str                     # Property name to modify
    drift_type: str = "linear"              # "linear" | "sinusoidal" | "random_walk" | "mean_revert"
    rate: float = 0.0                       # Per-round change (linear), or reversion speed (mean_revert)
    amplitude: float = 0.0                  # For sinusoidal
    period: int = 10                        # For sinusoidal (rounds per full cycle)
    mean: Optional[float] = None            # For mean_revert (drift target)
    variance: float = 0.0                   # For random_walk (standard deviation)
    condition: Optional[Dict[str, Any]] = None  # Optional trigger condition
    min_value: Optional[float] = None       # Floor clamp
    max_value: Optional[float] = None       # Ceiling clamp
    active: bool = True                     # Can be toggled by cascades

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_type": self.target_type,
            "property_field": self.property_field,
            "drift_type": self.drift_type,
            "rate": self.rate,
            "amplitude": self.amplitude,
            "period": self.period,
            "mean": self.mean,
            "variance": self.variance,
            "condition": self.condition,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PropertyDriftRule":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConditionalSpawnRule:
    """Spawn entities when conditions are met."""
    name: str
    condition_type: str = "entity_count_below"   # "entity_count_below" | "property_threshold" | "round_interval"
    condition_params: Dict[str, Any] = field(default_factory=dict)
    # For entity_count_below: {"entity_type": "rabbit", "threshold": 3}
    # For property_threshold: {"entity_type": "X", "property": "Y", "operator": "lt", "value": Z}
    # For round_interval: {"interval": 5}

    spawn_type: str = ""                    # Entity type name to spawn
    spawn_count: int = 1                    # How many to spawn
    cooldown: int = 5                       # Min rounds between spawns
    spawn_location: Optional[str] = None    # Fixed location (or None for random)
    property_overrides: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    _last_spawn_round: int = field(default=-999, repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "condition_type": self.condition_type,
            "condition_params": self.condition_params,
            "spawn_type": self.spawn_type,
            "spawn_count": self.spawn_count,
            "cooldown": self.cooldown,
            "spawn_location": self.spawn_location,
            "property_overrides": self.property_overrides,
            "last_spawn_round": self._last_spawn_round,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConditionalSpawnRule":
        rule = cls(
            name=data["name"],
            condition_type=data.get("condition_type", "entity_count_below"),
            condition_params=data.get("condition_params", {}),
            spawn_type=data.get("spawn_type", ""),
            spawn_count=data.get("spawn_count", 1),
            cooldown=data.get("cooldown", 5),
            spawn_location=data.get("spawn_location"),
            property_overrides=data.get("property_overrides", {}),
        )
        rule._last_spawn_round = data.get("last_spawn_round", -999)
        return rule


@dataclass
class CascadeRule:
    """Chained dynamics: when condition met, activate/deactivate other drift rules."""
    name: str
    description: str = ""
    condition_type: str = "property_avg_below"   # "property_avg_below" | "property_avg_above" | "entity_count_below" | "resource_below"
    condition_params: Dict[str, Any] = field(default_factory=dict)
    activate_rules: List[str] = field(default_factory=list)    # Drift rule names to activate
    deactivate_rules: List[str] = field(default_factory=list)  # Drift rule names to deactivate
    duration: int = 0                       # How many rounds the activation lasts (0 = permanent)
    cooldown: int = 10                      # Min rounds between triggers

    # Runtime state
    _active_since: Optional[int] = field(default=None, repr=False)
    _last_trigger_round: int = field(default=-999, repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "condition_type": self.condition_type,
            "condition_params": self.condition_params,
            "activate_rules": self.activate_rules,
            "deactivate_rules": self.deactivate_rules,
            "duration": self.duration,
            "cooldown": self.cooldown,
            "active_since": self._active_since,
            "last_trigger_round": self._last_trigger_round,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CascadeRule":
        rule = cls(
            name=data["name"],
            description=data.get("description", ""),
            condition_type=data.get("condition_type", "property_avg_below"),
            condition_params=data.get("condition_params", {}),
            activate_rules=data.get("activate_rules", []),
            deactivate_rules=data.get("deactivate_rules", []),
            duration=data.get("duration", 0),
            cooldown=data.get("cooldown", 10),
        )
        rule._active_since = data.get("active_since")
        rule._last_trigger_round = data.get("last_trigger_round", -999)
        return rule


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PropertyDynamicsEngine:
    """Manages per-round property drift, conditional spawning, and cascade chains.

    Called once per round by SimulationEngine._process_world_events().
    Returns a list of change dicts for event emission.
    """

    def __init__(
        self,
        drift_rules: Optional[List[PropertyDriftRule]] = None,
        spawn_rules: Optional[List[ConditionalSpawnRule]] = None,
        cascade_rules: Optional[List[CascadeRule]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.drift_rules: List[PropertyDriftRule] = drift_rules or []
        self.spawn_rules: List[ConditionalSpawnRule] = spawn_rules or []
        self.cascade_rules: List[CascadeRule] = cascade_rules or []
        self.rng = rng or random.Random()

        # Index drift rules by name for cascade lookup
        self._drift_by_name: Dict[str, PropertyDriftRule] = {
            r.name: r for r in self.drift_rules
        }

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        """Apply all dynamics for one round. Returns list of change events."""
        changes: List[Dict[str, Any]] = []

        # 1. Evaluate cascade rules (may activate/deactivate drift rules)
        changes.extend(self._evaluate_cascades(state, round_number))

        # 2. Apply active drift rules
        changes.extend(self._apply_drift_rules(state, round_number))

        # 3. Check conditional spawns
        changes.extend(self._check_spawn_rules(state, round_number))

        return changes

    # -- Drift application --

    def _apply_drift_rules(self, state: Any, round_number: int) -> List[Dict]:
        changes = []
        for rule in self.drift_rules:
            if not rule.active:
                continue

            # Check optional condition
            if rule.condition and not self._check_condition(rule.condition, state):
                continue

            # Find affected entities
            entities = [
                e for e in state.entities.values()
                if e.entity_type == rule.target_type and e.alive
            ]

            for entity in entities:
                old_value = entity.get(rule.property_field, 0)
                if not isinstance(old_value, (int, float)):
                    continue

                new_value = self._compute_drift(rule, old_value, round_number)

                # Clamp
                if rule.min_value is not None:
                    new_value = max(rule.min_value, new_value)
                if rule.max_value is not None:
                    new_value = min(rule.max_value, new_value)

                if abs(new_value - old_value) > 1e-6:
                    entity.set(rule.property_field, new_value)
                    changes.append({
                        "type": "property_drift",
                        "rule": rule.name,
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "field": rule.property_field,
                        "old_value": round(old_value, 4),
                        "new_value": round(new_value, 4),
                        "drift_type": rule.drift_type,
                        "narrative": (
                            f"{entity.name}'s {rule.property_field} "
                            f"{'increased' if new_value > old_value else 'decreased'} "
                            f"from {old_value:.1f} to {new_value:.1f}"
                        ),
                    })

        return changes

    def _compute_drift(self, rule: PropertyDriftRule, current: float, round_number: int) -> float:
        """Compute new property value based on drift type."""
        if rule.drift_type == "linear":
            return current + rule.rate

        elif rule.drift_type == "sinusoidal":
            # Absolute oscillation around a mean value (not incremental delta).
            # Uses rule.mean as center (defaults to current if not set).
            # Value = mean + amplitude * sin(phase), clamped by min/max.
            period = max(1, rule.period)
            phase = (2 * math.pi * round_number) / period
            center = rule.mean if rule.mean is not None else current
            return center + rule.amplitude * math.sin(phase)

        elif rule.drift_type == "random_walk":
            step = self.rng.gauss(0, rule.variance) if rule.variance > 0 else 0
            return current + step

        elif rule.drift_type == "mean_revert":
            if rule.mean is not None:
                diff = rule.mean - current
                return current + diff * rule.rate
            return current

        return current

    # -- Conditional spawning --

    def _check_spawn_rules(self, state: Any, round_number: int) -> List[Dict]:
        changes = []
        for rule in self.spawn_rules:
            # Check cooldown
            if round_number - rule._last_spawn_round < rule.cooldown:
                continue

            # Check condition
            if not self._check_spawn_condition(rule, state, round_number):
                continue

            # Spawn entities
            for i in range(rule.spawn_count):
                spawn_id = f"spawned_{rule.spawn_type}_{uuid.uuid4().hex[:8]}"
                spawn_name = f"{rule.spawn_type}_{round_number}_{i}"
                entity = state.spawn_entity_from_template(
                    template_name=rule.spawn_type,
                    entity_id=spawn_id,
                    name=spawn_name,
                    location=rule.spawn_location,
                    property_overrides=rule.property_overrides or None,
                )
                if entity:
                    changes.append({
                        "type": "conditional_spawn",
                        "rule": rule.name,
                        "entity_id": spawn_id,
                        "entity_name": spawn_name,
                        "entity_type": rule.spawn_type,
                        "location": rule.spawn_location,
                        "narrative": f"A new {rule.spawn_type} ({spawn_name}) appears in the world.",
                    })

            rule._last_spawn_round = round_number

        return changes

    def _check_spawn_condition(self, rule: ConditionalSpawnRule, state: Any, round_number: int) -> bool:
        """Evaluate a spawn rule's condition against the current state."""
        params = rule.condition_params

        if rule.condition_type == "entity_count_below":
            etype = params.get("entity_type", rule.spawn_type)
            threshold = params.get("threshold", 1)
            alive = [e for e in state.entities.values()
                     if e.entity_type == etype and e.alive]
            return len(alive) < threshold

        elif rule.condition_type == "property_threshold":
            etype = params.get("entity_type", "")
            prop = params.get("property", "")
            op = params.get("operator", "lt")
            value = params.get("value", 0)
            entities = [e for e in state.entities.values()
                        if e.entity_type == etype and e.alive]
            if not entities:
                return False
            # Check if ANY entity meets the threshold
            for entity in entities:
                val = entity.get(prop, 0)
                if isinstance(val, (int, float)):
                    if op == "lt" and val < value:
                        return True
                    if op == "gt" and val > value:
                        return True
                    if op == "lte" and val <= value:
                        return True
                    if op == "gte" and val >= value:
                        return True
            return False

        elif rule.condition_type == "round_interval":
            interval = params.get("interval", 5)
            return round_number > 0 and round_number % interval == 0

        return False

    # -- Cascade evaluation --

    def _evaluate_cascades(self, state: Any, round_number: int) -> List[Dict]:
        changes = []
        for rule in self.cascade_rules:
            # Check if active cascade has expired
            if rule._active_since is not None and rule.duration > 0:
                if round_number - rule._active_since >= rule.duration:
                    # Deactivate
                    for rname in rule.activate_rules:
                        if rname in self._drift_by_name:
                            self._drift_by_name[rname].active = False
                    for rname in rule.deactivate_rules:
                        if rname in self._drift_by_name:
                            self._drift_by_name[rname].active = True
                    changes.append({
                        "type": "cascade_expired",
                        "rule": rule.name,
                        "narrative": f"Cascade effect '{rule.name}' has ended.",
                    })
                    rule._active_since = None
                continue  # Don't re-evaluate while active

            # Check cooldown
            if round_number - rule._last_trigger_round < rule.cooldown:
                continue

            # Check condition
            if self._check_cascade_condition(rule, state):
                # Activate
                for rname in rule.activate_rules:
                    if rname in self._drift_by_name:
                        self._drift_by_name[rname].active = True
                for rname in rule.deactivate_rules:
                    if rname in self._drift_by_name:
                        self._drift_by_name[rname].active = False

                rule._active_since = round_number
                rule._last_trigger_round = round_number
                changes.append({
                    "type": "cascade_triggered",
                    "rule": rule.name,
                    "description": rule.description,
                    "activated": rule.activate_rules,
                    "deactivated": rule.deactivate_rules,
                    "narrative": f"Cascade triggered: {rule.description or rule.name}",
                })

        return changes

    def _check_cascade_condition(self, rule: CascadeRule, state: Any) -> bool:
        """Evaluate a cascade rule's condition."""
        params = rule.condition_params

        if rule.condition_type == "property_avg_below":
            etype = params.get("entity_type", "")
            prop = params.get("property", "")
            threshold = params.get("threshold", 0)
            entities = [e for e in state.entities.values()
                        if e.entity_type == etype and e.alive]
            if not entities:
                return False
            avg = sum(e.get(prop, 0) for e in entities) / len(entities)
            return avg < threshold

        elif rule.condition_type == "property_avg_above":
            etype = params.get("entity_type", "")
            prop = params.get("property", "")
            threshold = params.get("threshold", 0)
            entities = [e for e in state.entities.values()
                        if e.entity_type == etype and e.alive]
            if not entities:
                return False
            avg = sum(e.get(prop, 0) for e in entities) / len(entities)
            return avg > threshold

        elif rule.condition_type == "entity_count_below":
            etype = params.get("entity_type", "")
            threshold = params.get("threshold", 1)
            alive = [e for e in state.entities.values()
                     if e.entity_type == etype and e.alive]
            return len(alive) < threshold

        elif rule.condition_type == "resource_below":
            rname = params.get("resource", "")
            threshold = params.get("threshold", 0)
            pool = state.resources.get(rname)
            if not pool:
                return False
            total = pool.total
            return total < threshold

        return False

    # -- Generic condition check (for drift rule conditions) --

    def _check_condition(self, condition: Dict[str, Any], state: Any) -> bool:
        """Check a simple condition dict against state."""
        ctype = condition.get("type", "")
        if ctype == "round_gte":
            return state.temporal.current_round >= condition.get("value", 0)
        if ctype == "round_lte":
            return state.temporal.current_round <= condition.get("value", 0)
        if ctype == "entity_count_above":
            etype = condition.get("entity_type", "")
            threshold = condition.get("threshold", 0)
            alive = [e for e in state.entities.values()
                     if e.entity_type == etype and e.alive]
            return len(alive) > threshold
        if ctype == "entity_count_below":
            etype = condition.get("entity_type", "")
            threshold = condition.get("threshold", 0)
            alive = [e for e in state.entities.values()
                     if e.entity_type == etype and e.alive]
            return len(alive) < threshold
        return False  # Unknown condition type → reject (fail safe)

    # -- Serialization --

    def to_dict(self) -> dict:
        return {
            "drift_rules": [r.to_dict() for r in self.drift_rules],
            "spawn_rules": [r.to_dict() for r in self.spawn_rules],
            "cascade_rules": [r.to_dict() for r in self.cascade_rules],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PropertyDynamicsEngine":
        engine = cls(
            drift_rules=[PropertyDriftRule.from_dict(d) for d in data.get("drift_rules", [])],
            spawn_rules=[ConditionalSpawnRule.from_dict(d) for d in data.get("spawn_rules", [])],
            cascade_rules=[CascadeRule.from_dict(d) for d in data.get("cascade_rules", [])],
        )
        engine._drift_by_name = {r.name: r for r in engine.drift_rules}
        return engine
