"""World invariants -- runtime assertions checked after every round.

Auto-generated from schema (resource conservation, property bounds) or
user-defined. Violations emit events; severity="error" stops the simulation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .state import WorldState


@dataclass
class WorldInvariant:
    """A runtime assertion about world state."""
    name: str
    check_type: str  # resource_conservation, property_bounds, entity_count_bounds
    params: dict = field(default_factory=dict)
    severity: str = "warning"  # "warning" or "error"


class InvariantChecker:
    """Evaluates world invariants after each round."""

    def __init__(self, invariants: Optional[List[WorldInvariant]] = None):
        self.invariants = invariants or []
        self.violations: List[dict] = []
        self._initial_resource_totals: Dict[str, float] = {}

    def capture_initial_state(self, state: WorldState):
        """Capture initial resource totals for conservation checks."""
        for res_name, pool in state.resources.items():
            total = sum(pool.holdings.values()) + pool.unallocated
            self._initial_resource_totals[res_name] = total

    def check_all(self, state: WorldState, round_number: int) -> List[dict]:
        """Run all invariant checks. Returns list of violations."""
        round_violations = []
        for inv in self.invariants:
            msg = self._evaluate(inv, state)
            if msg:
                violation = {
                    "invariant": inv.name,
                    "check_type": inv.check_type,
                    "severity": inv.severity,
                    "round": round_number,
                    "message": msg,
                }
                round_violations.append(violation)
                self.violations.append(violation)
        return round_violations

    def _evaluate(self, inv: WorldInvariant, state: WorldState) -> Optional[str]:
        """Evaluate a single invariant. Returns error message or None."""
        if inv.check_type == "resource_conservation":
            return self._check_resource_conservation(state, inv.params)
        elif inv.check_type == "property_bounds":
            return self._check_property_bounds(state, inv.params)
        elif inv.check_type == "entity_count_bounds":
            return self._check_entity_count_bounds(state, inv.params)
        return None

    def _check_resource_conservation(self, state: WorldState, params: dict) -> Optional[str]:
        """Verify total of a conserved resource hasn't changed from initial."""
        resource_name = params.get("resource")
        if not resource_name:
            return None
        pool = state.resources.get(resource_name)
        if not pool:
            return None
        current_total = sum(pool.holdings.values()) + pool.unallocated
        expected = self._initial_resource_totals.get(resource_name, current_total)
        if abs(current_total - expected) > 0.001:
            return (
                f"Resource '{resource_name}' total changed: "
                f"expected {expected}, got {current_total} (delta {current_total - expected:+.3f})"
            )
        return None

    def _check_property_bounds(self, state: WorldState, params: dict) -> Optional[str]:
        """Verify entity properties stay within schema bounds."""
        entity_type_name = params.get("entity_type")
        prop_name = params.get("property")
        min_val = params.get("min_value")
        max_val = params.get("max_value")
        if not entity_type_name or not prop_name:
            return None
        for entity in state.entities.values():
            if entity.entity_type != entity_type_name or not entity.alive:
                continue
            val = entity.get(prop_name)
            if val is None:
                continue
            if min_val is not None and val < min_val:
                return (
                    f"Entity '{entity.name}' ({entity.id}): property '{prop_name}' "
                    f"= {val} is below minimum {min_val}"
                )
            if max_val is not None and val > max_val:
                return (
                    f"Entity '{entity.name}' ({entity.id}): property '{prop_name}' "
                    f"= {val} is above maximum {max_val}"
                )
        return None

    def _check_entity_count_bounds(self, state: WorldState, params: dict) -> Optional[str]:
        """Verify alive entity count stays within expected range."""
        entity_type_name = params.get("entity_type")
        min_count = params.get("min_count", 0)
        max_count = params.get("max_count")
        if not entity_type_name:
            return None
        alive = [
            e for e in state.entities.values()
            if e.entity_type == entity_type_name and e.alive
        ]
        count = len(alive)
        if count < min_count:
            return (
                f"Entity type '{entity_type_name}' alive count {count} "
                f"is below minimum {min_count}"
            )
        if max_count is not None and count > max_count:
            return (
                f"Entity type '{entity_type_name}' alive count {count} "
                f"is above maximum {max_count}"
            )
        return None
