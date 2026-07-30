"""Visibility rules, perception rendering, and trend analysis."""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Deque


@dataclass
class VisibilityRule:
    """Defines what an entity type can perceive about the world."""
    observer_type: str  # EntityType.name of the observer (or "*" for all)

    # What entity types are visible (empty = all)
    visible_entity_types: List[str] = field(default_factory=list)

    # What properties of each type are visible
    # {"Merchant": ["name", "gold", "reputation"]}
    # Empty list for a type = all non-hidden properties
    visible_properties: Dict[str, List[str]] = field(default_factory=dict)

    # Spatial visibility
    max_range: Optional[float] = None
    requires_same_location: bool = False

    # Relation visibility
    see_relations_involving_self: bool = True
    see_all_relations: bool = False

    # Resource visibility (empty = own resources only)
    visible_resources: List[str] = field(default_factory=list)


class PerceptionBuilder:
    """
    Given an entity, visibility rules, and world state, produce a
    filtered view of the world for that entity's LLM brain.
    """

    def build_perception(
        self,
        observer_id: str,
        observer_type: str,
        rules: List[VisibilityRule],
        entities: Dict[str, Any],
        entity_types: Dict[str, Any],
        resources: Dict[str, Any],
        relations: Any,
        spatial: Any,
        temporal: Any,
        active_world_events: Optional[List[Dict[str, Any]]] = None,
        faction_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build a filtered state dictionary representing what the observer can see."""
        # Find the most specific rule for this observer type
        rule = None
        for r in rules:
            if r.observer_type == observer_type:
                rule = r
                break
        if rule is None:
            for r in rules:
                if r.observer_type == "*":
                    rule = r
                    break
        if rule is None:
            rule = VisibilityRule(observer_type=observer_type)

        observer = entities.get(observer_id)
        if not observer:
            return {"error": "Observer not found"}

        # Resolve faction info for observer
        observer_faction = None
        observer_faction_name = None
        if faction_manager:
            observer_faction = faction_manager.get_entity_faction(observer_id)
            if observer_faction:
                faction_obj = faction_manager.get_faction(observer_faction)
                observer_faction_name = faction_obj.name if faction_obj else observer_faction

        perception = {
            "self": observer.to_dict() if hasattr(observer, 'to_dict') else {},
            "round": temporal.current_round if temporal else 0,
            "phase": temporal.current_phase.name if temporal and temporal.phases else "action",
            "visible_entities": [],
            "visible_resources": {},
            "visible_relations": [],
            "location": observer.location_id if hasattr(observer, 'location_id') else None,
            "active_world_events": active_world_events or [],
            "faction": observer_faction_name,
        }

        # Filter entities by visibility
        for eid, entity in entities.items():
            if eid == observer_id:
                continue
            if not getattr(entity, 'alive', True):
                continue

            etype = entity.entity_type

            # Check type visibility
            if rule.visible_entity_types and etype not in rule.visible_entity_types:
                continue

            # Check spatial visibility
            if rule.requires_same_location and hasattr(observer, 'location_id'):
                if getattr(entity, 'location_id', None) != observer.location_id:
                    continue
            if rule.max_range is not None and spatial:
                obs_loc = getattr(observer, 'location_id', None)
                ent_loc = getattr(entity, 'location_id', None)
                if obs_loc and ent_loc:
                    dist = spatial.distance(obs_loc, ent_loc)
                    if dist > rule.max_range:
                        continue

            # Filter properties
            visible_props = rule.visible_properties.get(etype, [])
            entity_type_def = entity_types.get(etype)

            filtered_props = {}
            for pname, pval in entity.properties.items():
                # Check if property is hidden in schema
                if entity_type_def:
                    pschema = entity_type_def.get_property_schema(pname)
                    if pschema and pschema.hidden:
                        continue
                # Check if property is in visible list (empty = all non-hidden)
                if visible_props and pname not in visible_props:
                    continue
                filtered_props[pname] = pval

            # Determine faction alliance
            is_ally = False
            if faction_manager and observer_faction:
                entity_faction = faction_manager.get_entity_faction(eid)
                is_ally = entity_faction == observer_faction

            # Confidence level based on spatial distance
            confidence = "certain"
            if spatial and hasattr(observer, 'location_id'):
                obs_loc = getattr(observer, 'location_id', None)
                ent_loc = getattr(entity, 'location_id', None)
                if obs_loc and ent_loc and obs_loc != ent_loc:
                    dist = spatial.distance(obs_loc, ent_loc)
                    if dist <= 1:
                        confidence = "observed"
                    else:
                        confidence = "rumored"

            perception["visible_entities"].append({
                "id": eid,
                "name": entity.name,
                "type": etype,
                "properties": filtered_props,
                "location": getattr(entity, 'location_id', None),
                "is_ally": is_ally,
                "confidence": confidence,
            })

        # Filter resources (always see own)
        for rname, pool in resources.items():
            if rule.visible_resources and rname not in rule.visible_resources:
                # Still show own amount
                own_amount = pool.get(observer_id)
                if own_amount > 0:
                    perception["visible_resources"][rname] = {"own": own_amount}
                continue
            perception["visible_resources"][rname] = {
                "own": pool.get(observer_id),
            }

        # Filter relations
        if relations:
            if rule.see_all_relations:
                for edge in relations.get_outgoing(observer_id) + relations.get_incoming(observer_id):
                    perception["visible_relations"].append({
                        "from": edge.from_entity,
                        "to": edge.to_entity,
                        "type": edge.relation_type,
                        "value": edge.value,
                    })
            elif rule.see_relations_involving_self:
                seen = set()
                for edge in relations.get_outgoing(observer_id):
                    key = (edge.from_entity, edge.to_entity, edge.relation_type)
                    if key not in seen:
                        seen.add(key)
                        perception["visible_relations"].append({
                            "from": edge.from_entity,
                            "to": edge.to_entity,
                            "type": edge.relation_type,
                            "value": edge.value,
                        })
                for edge in relations.get_incoming(observer_id):
                    key = (edge.from_entity, edge.to_entity, edge.relation_type)
                    if key not in seen:
                        seen.add(key)
                        perception["visible_relations"].append({
                            "from": edge.from_entity,
                            "to": edge.to_entity,
                            "type": edge.relation_type,
                            "value": edge.value,
                        })

        return perception


class TrendAnalyzer:
    """
    Tracks perception snapshots over time and detects trends.

    Maintains a circular buffer of recent perceptions per observer.
    Compares current values to historical ones to identify rising/declining
    resources, relationship changes, and entity health trajectories.
    """

    def __init__(self, max_snapshots: int = 5):
        self.max_snapshots = max_snapshots
        self._history: Dict[str, Deque[Dict[str, Any]]] = {}  # observer_id -> deque of snapshots

    def update(self, observer_id: str, perception: Dict[str, Any]):
        """Store a perception snapshot for trend analysis."""
        if observer_id not in self._history:
            self._history[observer_id] = deque(maxlen=self.max_snapshots)

        # Extract key numeric values for comparison
        snapshot = {
            "round": perception.get("round", 0),
            "resources": {},
            "entity_health": {},
            "relations": {},
        }

        # Capture resource values
        for rname, rdata in perception.get("visible_resources", {}).items():
            own = rdata.get("own", 0) if isinstance(rdata, dict) else 0
            snapshot["resources"][rname] = own if isinstance(own, (int, float)) else 0

        # Capture entity properties (health/key numeric values)
        for ve in perception.get("visible_entities", []):
            for prop, val in ve.get("properties", {}).items():
                if isinstance(val, (int, float)):
                    snapshot["entity_health"][f"{ve['id']}.{prop}"] = val

        # Capture relation values
        for rel in perception.get("visible_relations", []):
            key = f"{rel['type']}:{rel['from']}->{rel['to']}"
            rel_val = rel.get("value", 0)
            snapshot["relations"][key] = rel_val if isinstance(rel_val, (int, float)) else 0

        self._history[observer_id].append(snapshot)

    def get_trends(self, observer_id: str) -> Dict[str, Any]:
        """
        Compare current perception to historical snapshots.

        Returns:
        {
            "resource_trends": {"gold": "declining (-15 over 3 rounds)"},
            "entity_trends": {"bob.health": "declining"},
            "relation_trends": {"trust:alice->bob": "improving"},
        }
        """
        history = self._history.get(observer_id, deque())
        if len(history) < 2:
            return {}

        current = history[-1]
        oldest = history[0]
        span = current["round"] - oldest["round"]
        if span <= 0:
            return {}

        trends: Dict[str, Any] = {}

        # Resource trends
        resource_trends = {}
        for rname, current_val in current["resources"].items():
            if rname in oldest["resources"]:
                cv = current_val if isinstance(current_val, (int, float)) else 0
                ov = oldest["resources"][rname]
                ov = ov if isinstance(ov, (int, float)) else 0
                diff = cv - ov
                if abs(diff) >= 1:  # Significant change
                    direction = "rising" if diff > 0 else "declining"
                    resource_trends[rname] = f"{direction} ({diff:+.0f} over {span} rounds)"
        if resource_trends:
            trends["resource_trends"] = resource_trends

        # Entity property trends
        entity_trends = {}
        for key, current_val in current["entity_health"].items():
            if key in oldest["entity_health"]:
                cv = current_val if isinstance(current_val, (int, float)) else 0
                ov = oldest["entity_health"][key]
                ov = ov if isinstance(ov, (int, float)) else 0
                diff = cv - ov
                if abs(diff) >= 1:
                    direction = "rising" if diff > 0 else "declining"
                    entity_trends[key] = direction
        if entity_trends:
            trends["entity_trends"] = entity_trends

        # Relation trends
        relation_trends = {}
        for key, current_val in current["relations"].items():
            if key in oldest["relations"]:
                cv = current_val if isinstance(current_val, (int, float)) else 0
                ov = oldest["relations"][key]
                ov = ov if isinstance(ov, (int, float)) else 0
                diff = cv - ov
                if abs(diff) >= 0.05:  # Relations are typically 0-1
                    direction = "improving" if diff > 0 else "deteriorating"
                    relation_trends[key] = f"{direction} ({diff:+.2f})"
        if relation_trends:
            trends["relation_trends"] = relation_trends

        return trends
