"""Location properties -- terrain effects, environmental hazards, and modifiers.

Locations can have modifiers that affect entities occupying them and
tick effects that fire each round on occupants.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LocationModifier:
    """A modifier applied to entities at this location during action resolution."""
    property_name: str              # which entity property to modify
    operation: str                  # "add", "multiply"
    value: float
    entity_type: Optional[str] = None  # None = applies to all types


@dataclass
class LocationTick:
    """An effect that fires each round on entities at this location."""
    property_name: str
    operation: str                  # "add", "subtract"
    value: float
    entity_type: Optional[str] = None  # None = applies to all types


@dataclass
class LocationDefinition:
    """Full definition of a location's properties and effects."""
    id: str
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    modifiers: List[LocationModifier] = field(default_factory=list)
    tick_effects: List[LocationTick] = field(default_factory=list)
    entry_requirements: List[Dict[str, Any]] = field(default_factory=list)


class LocationPropertyManager:
    """Manages location definitions and their effects."""

    def __init__(self):
        self._locations: Dict[str, LocationDefinition] = {}

    def register(self, location: LocationDefinition):
        """Register a location definition."""
        self._locations[location.id] = location

    def get(self, location_id: str) -> Optional[LocationDefinition]:
        """Get a location definition by ID."""
        return self._locations.get(location_id)

    def get_modifiers(self, location_id: str) -> List[LocationModifier]:
        """Get all modifiers for a location."""
        loc = self._locations.get(location_id)
        return list(loc.modifiers) if loc else []

    def get_tick_effects(self, location_id: str) -> List[LocationTick]:
        """Get all tick effects for a location."""
        loc = self._locations.get(location_id)
        return list(loc.tick_effects) if loc else []

    def apply_tick_effects(
        self,
        entity,
        location_id: str,
        entity_type_schemas: dict = None,
    ) -> List[dict]:
        """Apply per-round tick effects to an entity at a location.

        Returns list of change dicts describing what was modified.
        """
        loc = self._locations.get(location_id)
        if not loc:
            return []

        changes = []
        for tick in loc.tick_effects:
            # Filter by entity type if specified
            if tick.entity_type and entity.entity_type != tick.entity_type:
                continue

            old_val = entity.get(tick.property_name, 0)
            if not isinstance(old_val, (int, float)):
                continue

            # Look up schema for bounds
            prop_schema = None
            if entity_type_schemas:
                et_schema = entity_type_schemas.get(entity.entity_type)
                if et_schema:
                    prop_schema = et_schema.get_property_schema(tick.property_name)

            if tick.operation == "add":
                entity.modify(tick.property_name, tick.value, prop_schema)
            elif tick.operation == "subtract":
                entity.modify(tick.property_name, -tick.value, prop_schema)

            new_val = entity.get(tick.property_name)
            if new_val != old_val and new_val is not None:
                changes.append({
                    "entity": entity.id,
                    "field": tick.property_name,
                    "old": old_val if old_val is not None else 0,
                    "new": new_val,
                    "location": location_id,
                    "effect": tick.operation,
                })

        return changes

    def check_entry(self, entity, location_id: str) -> bool:
        """Check if an entity meets the entry requirements for a location.

        Entry requirements are dicts with:
          - "property": property name to check
          - "operator": "gte", "lte", "gt", "lt", "eq"
          - "value": value to compare against
          - "entity_type": optional, only check for this type
        """
        loc = self._locations.get(location_id)
        if not loc:
            return True  # Unknown location, allow entry

        for req in loc.entry_requirements:
            # Filter by entity type if specified
            req_type = req.get("entity_type")
            if req_type and entity.entity_type != req_type:
                continue

            prop = req.get("property")
            if not prop:
                continue

            val = entity.get(prop, 0)
            op = req.get("operator", "gte")
            target = req.get("value", 0)

            if op == "gte" and not (val >= target):
                return False
            elif op == "lte" and not (val <= target):
                return False
            elif op == "gt" and not (val > target):
                return False
            elif op == "lt" and not (val < target):
                return False
            elif op == "eq" and not (val == target):
                return False

        return True

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        return {
            lid: {
                "id": loc.id,
                "name": loc.name,
                "description": loc.description,
                "properties": dict(loc.properties),
                "modifiers": [
                    {"property_name": m.property_name, "operation": m.operation,
                     "value": m.value, "entity_type": m.entity_type}
                    for m in loc.modifiers
                ],
                "tick_effects": [
                    {"property_name": t.property_name, "operation": t.operation,
                     "value": t.value, "entity_type": t.entity_type}
                    for t in loc.tick_effects
                ],
            }
            for lid, loc in self._locations.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocationPropertyManager":
        """Restore from a to_dict snapshot."""
        mgr = cls()
        for lid, loc in (data or {}).items():
            modifiers = [
                LocationModifier(
                    property_name=m["property_name"],
                    operation=m.get("operation", "add"),
                    value=m.get("value", 0),
                    entity_type=m.get("entity_type"),
                )
                for m in loc.get("modifiers", [])
            ]
            ticks = [
                LocationTick(
                    property_name=t["property_name"],
                    operation=t.get("operation", "add"),
                    value=t.get("value", 0),
                    entity_type=t.get("entity_type"),
                )
                for t in loc.get("tick_effects", [])
            ]
            mgr._locations[lid] = LocationDefinition(
                id=loc["id"],
                name=loc.get("name", loc["id"]),
                description=loc.get("description", ""),
                properties=dict(loc.get("properties", {})),
                modifiers=modifiers,
                tick_effects=ticks,
            )
        return mgr
