"""Data connector framework -- pluggable external data sources for simulations.

Connectors bring real-world data (APIs, databases, files) into the simulation
as context for agent decisions and world events. This module defines the
infrastructure; actual connector implementations are registered separately.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type


class ConnectorType(Enum):
    """Supported connector transport types."""
    API = "api"
    DATABASE = "database"
    FILE = "file"
    WEBSOCKET = "websocket"
    CUSTOM = "custom"


class ConnectorStatus(Enum):
    """Lifecycle status of a connector instance."""
    IDLE = "idle"
    CONNECTING = "connecting"
    READY = "ready"
    FETCHING = "fetching"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class MappingRule:
    """Maps connector data fields to simulation state.
    
    source_field: dot-path in the connector payload (e.g. "price.current")
    target_type: where to route — "entity_property", "resource", "event", "context"
    target_entity_type: which entity type to apply to (for entity_property)
    target_field: the property/resource name to set
    transform: optional transformation ("direct", "normalize", "threshold")
    """
    source_field: str
    target_type: str = "context"  # entity_property, resource, event, context
    target_entity_type: Optional[str] = None
    target_field: Optional[str] = None
    transform: str = "direct"
    transform_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorConfig:
    """Configuration for a connector instance."""
    name: str
    connector_type: ConnectorType = ConnectorType.API
    endpoint: str = ""
    auth_config: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 1  # Rounds between refreshes (1 = every round)
    mapping_rules: List[MappingRule] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "connector_type": self.connector_type.value,
            "endpoint": self.endpoint,
            "refresh_interval": self.refresh_interval,
            "mapping_rules": [
                {
                    "source_field": r.source_field,
                    "target_type": r.target_type,
                    "target_entity_type": r.target_entity_type,
                    "target_field": r.target_field,
                    "transform": r.transform,
                    "transform_params": dict(r.transform_params),
                }
                for r in self.mapping_rules
            ],
            "params": dict(self.params),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectorConfig":
        rules = []
        for r in data.get("mapping_rules", []):
            rules.append(MappingRule(
                source_field=r["source_field"],
                target_type=r.get("target_type", "context"),
                target_entity_type=r.get("target_entity_type"),
                target_field=r.get("target_field"),
                transform=r.get("transform", "direct"),
                transform_params=r.get("transform_params", {}),
            ))
        return cls(
            name=data["name"],
            connector_type=ConnectorType(data.get("connector_type", "api")),
            endpoint=data.get("endpoint", ""),
            auth_config=data.get("auth_config", {}),
            refresh_interval=data.get("refresh_interval", 1),
            mapping_rules=rules,
            params=data.get("params", {}),
            description=data.get("description", ""),
        )


@dataclass
class DataPayload:
    """A packet of data fetched from a connector."""
    source: str  # Connector name
    timestamp: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    schema_hint: Optional[Dict[str, str]] = None  # field_name -> type hint

    def get_field(self, dot_path: str, default: Any = None) -> Any:
        """Get a nested field using dot notation: 'price.current' -> data['price']['current']."""
        parts = dot_path.split(".")
        current = self.data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return default
            else:
                return default
        return current

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "timestamp": self.timestamp,
            "data": self.data,
            "schema_hint": self.schema_hint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataPayload":
        return cls(
            source=data["source"],
            timestamp=data.get("timestamp", 0.0),
            data=data.get("data", {}),
            schema_hint=data.get("schema_hint"),
        )


class ConnectorInstance:
    """A live connector instance that fetches data from an external source.
    
    The base implementation stores a fetch_fn callback for flexibility.
    Actual connector types (API, DB, etc.) would subclass or provide fetch_fn.
    """

    def __init__(self, config: ConnectorConfig, fetch_fn: Optional[Callable] = None):
        self.config = config
        self.status = ConnectorStatus.IDLE
        self.last_payload: Optional[DataPayload] = None
        self.last_fetch_round: int = -1
        self.error_message: str = ""
        self._fetch_fn = fetch_fn  # fn(config) -> dict (raw data)

    @property
    def name(self) -> str:
        return self.config.name

    def connect(self):
        """Initialize the connector (validate config, open connections)."""
        self.status = ConnectorStatus.READY
        self.error_message = ""

    def disconnect(self):
        """Clean up the connector."""
        self.status = ConnectorStatus.DISCONNECTED

    def fetch(self, round_number: int) -> Optional[DataPayload]:
        """Fetch data from the external source.
        
        Returns a DataPayload on success, None on failure.
        Updates status and last_payload.
        """
        if self.status == ConnectorStatus.DISCONNECTED:
            return None

        self.status = ConnectorStatus.FETCHING
        try:
            if self._fetch_fn:
                raw_data = self._fetch_fn(self.config)
            else:
                # No fetch function — return empty payload (framework stub)
                raw_data = {}

            payload = DataPayload(
                source=self.config.name,
                timestamp=time.time(),
                data=raw_data if isinstance(raw_data, dict) else {"value": raw_data},
            )
            self.last_payload = payload
            self.last_fetch_round = round_number
            self.status = ConnectorStatus.READY
            self.error_message = ""
            return payload

        except Exception as e:
            self.status = ConnectorStatus.ERROR
            self.error_message = str(e)
            return None

    def needs_refresh(self, current_round: int) -> bool:
        """Check if this connector should fetch new data."""
        if self.status == ConnectorStatus.DISCONNECTED:
            return False
        if self.last_fetch_round < 0:
            return True  # Never fetched
        return (current_round - self.last_fetch_round) >= self.config.refresh_interval

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "last_fetch_round": self.last_fetch_round,
            "last_payload": self.last_payload.to_dict() if self.last_payload else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectorInstance":
        config = ConnectorConfig.from_dict(data["config"])
        instance = cls(config=config)
        instance.status = ConnectorStatus(data.get("status", "idle"))
        instance.last_fetch_round = data.get("last_fetch_round", -1)
        if data.get("last_payload"):
            instance.last_payload = DataPayload.from_dict(data["last_payload"])
        instance.error_message = data.get("error_message", "")
        return instance


class ConnectorRegistry:
    """Global registry of available connector type implementations.
    
    Connector types register a factory function that creates ConnectorInstance
    from a ConnectorConfig. This allows plugging in new connector types at
    runtime without modifying core code.
    """

    _instance: Optional["ConnectorRegistry"] = None

    def __init__(self):
        self._factories: Dict[str, Callable[[ConnectorConfig], ConnectorInstance]] = {}

    @classmethod
    def get_instance(cls) -> "ConnectorRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None

    def register_type(self, type_name: str, factory: Callable[[ConnectorConfig], ConnectorInstance]):
        """Register a connector type factory."""
        self._factories[type_name] = factory

    def create_instance(self, config: ConnectorConfig) -> ConnectorInstance:
        """Create a connector instance using the registered factory, or default."""
        factory = self._factories.get(config.connector_type.value)
        if factory:
            return factory(config)
        # Default: basic ConnectorInstance with no fetch function
        return ConnectorInstance(config=config)

    def list_types(self) -> List[str]:
        """List all registered connector type names."""
        return list(self._factories.keys())


class ConnectorManager:
    """Per-simulation connector manager.
    
    Manages multiple ConnectorInstance objects, ticks them each round,
    applies mapping rules to route data into the simulation state,
    and provides aggregated data for agent perception.
    """

    def __init__(self):
        self._instances: Dict[str, ConnectorInstance] = {}

    def add_instance(self, instance: ConnectorInstance):
        """Add a connector instance."""
        self._instances[instance.name] = instance

    def remove_instance(self, name: str) -> Optional[ConnectorInstance]:
        """Remove and disconnect a connector instance."""
        instance = self._instances.pop(name, None)
        if instance:
            instance.disconnect()
        return instance

    def get_instance(self, name: str) -> Optional[ConnectorInstance]:
        """Get a connector instance by name."""
        return self._instances.get(name)

    def connect_all(self):
        """Connect all instances."""
        for instance in self._instances.values():
            instance.connect()

    def disconnect_all(self):
        """Disconnect all instances."""
        for instance in self._instances.values():
            instance.disconnect()

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        """Tick all connectors: fetch stale data, apply mapping rules.
        
        Returns a list of changes applied to the state.
        """
        changes = []
        for instance in self._instances.values():
            if not instance.needs_refresh(round_number):
                continue

            payload = instance.fetch(round_number)
            if payload is None:
                continue

            # Apply mapping rules
            for rule in instance.config.mapping_rules:
                value = payload.get_field(rule.source_field)
                if value is None:
                    continue

                # Apply transform
                value = self._apply_transform(value, rule)

                change = self._apply_mapping(rule, value, state)
                if change:
                    changes.append(change)

        return changes

    def _apply_transform(self, value: Any, rule: MappingRule) -> Any:
        """Apply a transformation to a fetched value."""
        if rule.transform == "direct":
            return value
        elif rule.transform == "normalize":
            min_val = rule.transform_params.get("min", 0)
            max_val = rule.transform_params.get("max", 100)
            if isinstance(value, (int, float)) and max_val > min_val:
                return (value - min_val) / (max_val - min_val)
            return value
        elif rule.transform == "threshold":
            threshold = rule.transform_params.get("threshold", 0)
            if isinstance(value, (int, float)):
                return value >= threshold
            return value
        return value

    def _apply_mapping(self, rule: MappingRule, value: Any, state: Any) -> Optional[Dict]:
        """Apply a single mapping rule to route data into the simulation state."""
        if rule.target_type == "entity_property" and rule.target_entity_type and rule.target_field:
            # Set property on all entities of the target type
            entities = state.get_entities_by_type(rule.target_entity_type) if hasattr(state, 'get_entities_by_type') else []
            for entity in entities:
                old_val = entity.get(rule.target_field) if hasattr(entity, 'get') else None
                if hasattr(entity, 'set'):
                    entity.set(rule.target_field, value)
            return {
                "type": "entity_property",
                "entity_type": rule.target_entity_type,
                "field": rule.target_field,
                "value": value,
                "count": len(entities),
            }

        elif rule.target_type == "resource" and rule.target_field:
            # Set unallocated resource amount
            if hasattr(state, 'resources'):
                pool = state.resources.get(rule.target_field)
                if pool:
                    pool.unallocated = float(value) if isinstance(value, (int, float)) else pool.unallocated
                    return {
                        "type": "resource",
                        "resource": rule.target_field,
                        "value": value,
                    }

        elif rule.target_type == "event":
            # Data changes are stored as events; engine can emit them
            return {
                "type": "event",
                "source": rule.source_field,
                "value": value,
            }

        elif rule.target_type == "context":
            # Context data — available to agents but doesn't modify state
            return {
                "type": "context",
                "source": rule.source_field,
                "value": value,
            }

        return None

    def inject_context(self, entity_id: str) -> Dict[str, Any]:
        """Get all connector data relevant to an entity for perception injection."""
        context = {}
        for name, instance in self._instances.items():
            if instance.last_payload:
                context[name] = {
                    "data": instance.last_payload.data,
                    "round_fetched": instance.last_fetch_round,
                    "status": instance.status.value,
                }
        return context

    def get_all_data(self) -> Dict[str, Any]:
        """Aggregate all connector payloads for prompt context."""
        data = {}
        for name, instance in self._instances.items():
            if instance.last_payload:
                data[name] = instance.last_payload.data
        return data

    def to_dict(self) -> dict:
        return {
            "instances": {
                name: inst.to_dict()
                for name, inst in self._instances.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectorManager":
        manager = cls()
        for name, inst_data in data.get("instances", {}).items():
            instance = ConnectorInstance.from_dict(inst_data)
            manager._instances[name] = instance
        return manager
