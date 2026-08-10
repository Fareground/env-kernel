"""Core type system for the simulation kernel."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union


class PropertyType(Enum):
    """Supported property types for entity properties."""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    ENUM = "enum"
    LIST = "list"


# Runtime value for a property
PropertyValue = Union[float, int, str, bool, List[str], None]


@dataclass
class PropertySchema:
    """Schema definition for a single entity property."""
    name: str
    type: PropertyType
    default: PropertyValue = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Optional[List[str]] = None
    description: str = ""
    hidden: bool = False  # If True, not visible to other agents

    def validate(self, value: PropertyValue) -> bool:
        """Check if a value is valid for this schema."""
        if value is None:
            return True
        if self.type == PropertyType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        elif self.type == PropertyType.INT:
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        elif self.type == PropertyType.STRING:
            if not isinstance(value, str):
                return False
        elif self.type == PropertyType.BOOL:
            if not isinstance(value, bool):
                return False
        elif self.type == PropertyType.ENUM:
            if self.enum_values and value not in self.enum_values:
                return False
        elif self.type == PropertyType.LIST:
            if not isinstance(value, list):
                return False
        return True

    def coerce(self, value: Any) -> PropertyValue:
        """Attempt to coerce a value to the correct type."""
        if value is None:
            return self.default
        if self.type == PropertyType.FLOAT:
            return float(value)
        elif self.type == PropertyType.INT:
            return int(value)
        elif self.type == PropertyType.STRING:
            return str(value)
        elif self.type == PropertyType.BOOL:
            return bool(value)
        elif self.type == PropertyType.ENUM:
            s = str(value)
            if self.enum_values and s in self.enum_values:
                return s
            raise ValueError(f"Invalid enum value '{s}'. Valid: {self.enum_values}")
        elif self.type == PropertyType.LIST:
            if isinstance(value, list):
                return value
            return [str(value)]
        return value

    def clamp(self, value: PropertyValue) -> PropertyValue:
        """Clamp a numeric value to schema bounds."""
        if value is None:
            return value
        if self.type in (PropertyType.FLOAT, PropertyType.INT) and isinstance(value, (int, float)):
            if self.min_value is not None:
                value = max(self.min_value, value)
            if self.max_value is not None:
                value = min(self.max_value, value)
            if self.type == PropertyType.INT:
                value = int(value)
        return value
