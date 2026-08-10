"""Entity type definitions and entity instances."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import PropertySchema, PropertyValue, PropertyType


@dataclass
class EntityType:
    """
    Blueprint for a category of entities.

    role determines behavior:
      - "agent":    LLM-driven, takes actions each turn
      - "location": Spatial anchor, can contain other entities
      - "object":   Passive, can be acted upon
      - "abstract": Non-spatial concept (e.g., a faction, a law)
    """
    name: str
    role: str  # "agent", "location", "object", "abstract"
    properties: List[PropertySchema] = field(default_factory=list)
    description: str = ""

    def get_property_schema(self, prop_name: str) -> Optional[PropertySchema]:
        """Get the schema for a named property."""
        for p in self.properties:
            if p.name == prop_name:
                return p
        return None


@dataclass
class Entity:
    """
    A concrete instance of an EntityType within the world state.
    Properties are stored as a mutable dict keyed by property name.
    """
    id: str
    name: str
    entity_type: str  # References EntityType.name
    properties: Dict[str, PropertyValue] = field(default_factory=dict)
    location_id: Optional[str] = None
    alive: bool = True

    def get(self, prop_name: str, default: PropertyValue = None) -> PropertyValue:
        """Get a property value."""
        return self.properties.get(prop_name, default)

    def set(self, prop_name: str, value: PropertyValue):
        """Set a property value."""
        self.properties[prop_name] = value

    def modify(self, prop_name: str, delta: float, schema: Optional[PropertySchema] = None):
        """Add delta to a numeric property, respecting bounds from schema."""
        current = self.properties.get(prop_name, 0)
        if not isinstance(current, (int, float)):
            raise ValueError(f"Cannot modify non-numeric property '{prop_name}'")
        new_val = current + delta
        if schema:
            if schema.min_value is not None:
                new_val = max(schema.min_value, new_val)
            if schema.max_value is not None:
                new_val = min(schema.max_value, new_val)
            if schema.type == PropertyType.INT:
                new_val = int(new_val)
        self.properties[prop_name] = new_val

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": dict(self.properties),
            "location_id": self.location_id,
            "alive": self.alive,
        }
