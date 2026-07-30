"""Typed relation graph between entities."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RelationThreshold:
    """A threshold that triggers an event when a relation crosses it."""
    value: float                  # Threshold value
    direction: str = "above"      # "above" or "below"
    event_name: str = ""          # Event name to emit
    description: str = ""
    one_shot: bool = True         # Only triggers once per entity pair


@dataclass
class RelationType:
    """Definition of a relation kind."""
    name: str                    # e.g., "trust", "loyalty", "knows_about"
    symmetric: bool = False      # If True, A->B implies B->A
    default_value: float = 0.0
    min_value: float = -1.0
    max_value: float = 1.0
    description: str = ""
    decay_per_round: float = 0.0          # Rate of drift toward decay_toward
    decay_toward: Optional[float] = None  # Target value (default: default_value)
    thresholds: List[RelationThreshold] = field(default_factory=list)


@dataclass
class RelationEdge:
    """A single directed relation between two entities."""
    from_entity: str
    to_entity: str
    relation_type: str
    value: float = 0.0
    metadata: Dict = field(default_factory=dict)


class RelationGraph:
    """
    The full relation graph. Stores typed directed edges.
    Provides query methods for agent perception and action preconditions.
    """

    def __init__(self):
        self.relation_types: Dict[str, RelationType] = {}
        self._edges: Dict[Tuple[str, str, str], RelationEdge] = {}
        self._triggered_thresholds: set = set()  # (from, to, rtype, threshold_value) tuples

    def register_relation_type(self, rel_type: RelationType):
        """Register a relation type definition."""
        self.relation_types[rel_type.name] = rel_type

    def set(self, from_entity: str, to_entity: str, relation_type: str, value: float, **metadata):
        """Set or update a relation."""
        rt = self.relation_types.get(relation_type)
        if rt:
            value = max(rt.min_value, min(rt.max_value, value))

        edge = RelationEdge(
            from_entity=from_entity,
            to_entity=to_entity,
            relation_type=relation_type,
            value=value,
            metadata=metadata,
        )
        self._edges[(from_entity, to_entity, relation_type)] = edge

        # Handle symmetric relations
        if rt and rt.symmetric:
            reverse = RelationEdge(
                from_entity=to_entity,
                to_entity=from_entity,
                relation_type=relation_type,
                value=value,
                metadata=metadata,
            )
            self._edges[(to_entity, from_entity, relation_type)] = reverse

    def get(self, from_entity: str, to_entity: str, relation_type: str) -> float:
        """Get the value of a specific relation. Returns default if not set."""
        edge = self._edges.get((from_entity, to_entity, relation_type))
        if edge:
            return edge.value
        rt = self.relation_types.get(relation_type)
        return rt.default_value if rt else 0.0

    def modify(self, from_entity: str, to_entity: str, relation_type: str, delta: float):
        """Adjust a relation value by delta."""
        current = self.get(from_entity, to_entity, relation_type)
        self.set(from_entity, to_entity, relation_type, current + delta)

    def get_outgoing(self, entity_id: str, relation_type: Optional[str] = None) -> List[RelationEdge]:
        """Get all outgoing relations from an entity, optionally filtered by type."""
        results = []
        for (from_e, to_e, rtype), edge in self._edges.items():
            if from_e == entity_id:
                if relation_type is None or rtype == relation_type:
                    results.append(edge)
        return results

    def remove_entity(self, entity_id: str):
        """Remove all relation edges involving an entity (both directions)."""
        keys_to_remove = [
            key for key in self._edges
            if key[0] == entity_id or key[1] == entity_id
        ]
        for key in keys_to_remove:
            del self._edges[key]

    def get_incoming(self, entity_id: str, relation_type: Optional[str] = None) -> List[RelationEdge]:
        """Get all incoming relations to an entity."""
        results = []
        for (from_e, to_e, rtype), edge in self._edges.items():
            if to_e == entity_id:
                if relation_type is None or rtype == relation_type:
                    results.append(edge)
        return results

    def query_who(self, entity_id: str, relation_type: str, min_value: float = 0.0) -> List[str]:
        """'Who does entity X trust above threshold?' style queries."""
        return [
            edge.to_entity
            for edge in self.get_outgoing(entity_id, relation_type)
            if edge.value >= min_value
        ]

    def query_by_whom(self, entity_id: str, relation_type: str, min_value: float = 0.0) -> List[str]:
        """'Who trusts entity X above threshold?' style queries."""
        return [
            edge.from_entity
            for edge in self.get_incoming(entity_id, relation_type)
            if edge.value >= min_value
        ]

    def tick(self, round_number: int) -> List[dict]:
        """Apply decay and check thresholds. Returns list of triggered events."""
        triggered_events = []

        for key, edge in list(self._edges.items()):
            rt = self.relation_types.get(edge.relation_type)
            if not rt:
                continue

            # Apply decay
            if rt.decay_per_round > 0:
                target = rt.decay_toward if rt.decay_toward is not None else rt.default_value
                old_value = edge.value
                # Exponential decay toward target
                new_value = old_value + (target - old_value) * rt.decay_per_round
                # Clamp to bounds
                new_value = max(rt.min_value, min(rt.max_value, new_value))
                edge.value = new_value

            # Check thresholds (with hysteresis for non-one-shot thresholds)
            for threshold in rt.thresholds:
                th_key = (edge.from_entity, edge.to_entity, edge.relation_type, threshold.value, threshold.direction)

                if th_key in self._triggered_thresholds:
                    if threshold.one_shot:
                        continue
                    # Hysteresis: non-one-shot must cross BACK before retriggering
                    reset = False
                    if threshold.direction == "above" and edge.value < threshold.value:
                        reset = True
                    elif threshold.direction == "below" and edge.value > threshold.value:
                        reset = True
                    if reset:
                        self._triggered_thresholds.discard(th_key)
                    continue

                crossed = False
                if threshold.direction == "above" and edge.value >= threshold.value:
                    crossed = True
                elif threshold.direction == "below" and edge.value <= threshold.value:
                    crossed = True

                if crossed:
                    self._triggered_thresholds.add(th_key)
                    triggered_events.append({
                        "event_name": threshold.event_name,
                        "description": threshold.description,
                        "from_entity": edge.from_entity,
                        "to_entity": edge.to_entity,
                        "relation_type": edge.relation_type,
                        "value": edge.value,
                        "threshold": threshold.value,
                        "round": round_number,
                    })

        return triggered_events

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "relation_types": [rt.name for rt in self.relation_types.values()],
            "edges": [
                {
                    "from": e.from_entity,
                    "to": e.to_entity,
                    "type": e.relation_type,
                    "value": e.value,
                }
                for e in self._edges.values()
            ],
        }
