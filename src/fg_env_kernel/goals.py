"""Agent goals -- persistent objectives with measurable completion conditions.

Goals give agents direction, enable progress tracking, and support
win-condition narratives. Each entity can have multiple goals with
priorities, and goals are evaluated each round.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoalCondition:
    """A measurable condition for goal completion."""
    check_type: str               # "property_gte", "property_lte", "has_resource",
                                  # "at_location", "entity_alive", "entity_dead",
                                  # "relation_gte", "has_item"
    subject: str                  # "self" or specific entity_id
    field: Optional[str] = None   # Property/resource/relation name
    value: Any = None             # Value to compare against


@dataclass
class Goal:
    """A goal owned by an entity."""
    id: str
    description: str
    owner_id: str
    conditions: List[GoalCondition] = field(default_factory=list)
    priority: int = 1             # higher = more important
    completed: bool = False
    completed_round: Optional[int] = None
    failed: bool = False


class GoalTracker:
    """Manages entity goals, evaluation, and completion tracking."""

    def __init__(self):
        self._goals: Dict[str, List[Goal]] = {}   # entity_id -> list of goals

    def add_goal(self, entity_id: str, goal: Goal):
        """Add a goal for an entity."""
        if entity_id not in self._goals:
            self._goals[entity_id] = []
        self._goals[entity_id].append(goal)

    def get_goals(self, entity_id: str) -> List[Goal]:
        """Get all goals for an entity."""
        return list(self._goals.get(entity_id, []))

    def get_active_goals(self, entity_id: str) -> List[Goal]:
        """Get goals that are not completed and not failed."""
        return [
            g for g in self._goals.get(entity_id, [])
            if not g.completed and not g.failed
        ]

    def get_completed(self, entity_id: str) -> List[Goal]:
        """Get completed goals for an entity."""
        return [g for g in self._goals.get(entity_id, []) if g.completed]

    def remove_entity(self, entity_id: str):
        """Remove all goals for an entity."""
        self._goals.pop(entity_id, None)

    def mark_failed(self, goal_id: str, entity_id: str):
        """Mark a goal as failed."""
        for goal in self._goals.get(entity_id, []):
            if goal.id == goal_id:
                goal.failed = True
                break

    def evaluate_goals(self, entity_id: str, state, round_number: int) -> List[dict]:
        """Evaluate all active goals for an entity. Returns events for newly completed goals."""
        events = []
        for goal in self.get_active_goals(entity_id):
            if self._check_goal(goal, entity_id, state):
                goal.completed = True
                goal.completed_round = round_number
                events.append({
                    "event": "goal_completed",
                    "entity_id": entity_id,
                    "goal_id": goal.id,
                    "goal_description": goal.description,
                    "round": round_number,
                })
        return events

    def _check_goal(self, goal: Goal, owner_id: str, state) -> bool:
        """Check if all conditions of a goal are met."""
        if not goal.conditions:
            return False  # Goals with no conditions are never auto-complete
        for cond in goal.conditions:
            if not self._check_condition(cond, owner_id, state):
                return False
        return True

    def _check_condition(self, cond: GoalCondition, owner_id: str, state) -> bool:
        """Evaluate a single goal condition against the current state."""
        # Resolve subject
        subject_id = owner_id if cond.subject == "self" else cond.subject

        if cond.check_type == "property_gte":
            entity = state.entities.get(subject_id)
            if not entity:
                return False
            val = entity.get(cond.field, 0) if cond.field else 0
            return val >= cond.value

        elif cond.check_type == "property_lte":
            entity = state.entities.get(subject_id)
            if not entity:
                return False
            val = entity.get(cond.field, 0) if cond.field else 0
            return val <= cond.value

        elif cond.check_type == "has_resource":
            pool = state.resources.get(cond.field)
            if not pool:
                return False
            amount = pool.get(subject_id)
            return amount >= (cond.value or 0)

        elif cond.check_type == "at_location":
            loc = state.locations.get(subject_id)
            if loc is None:
                entity = state.entities.get(subject_id)
                loc = entity.location_id if entity else None
            return loc == cond.value

        elif cond.check_type == "entity_alive":
            entity = state.entities.get(subject_id)
            return entity is not None and entity.alive

        elif cond.check_type == "entity_dead":
            entity = state.entities.get(subject_id)
            if entity is None:
                return True  # Not in world = dead
            return not entity.alive

        elif cond.check_type == "relation_gte":
            # field = relation_type, value = threshold, subject = target entity
            # Check if owner has relation >= value with subject
            edges = state.relations.get_outgoing(owner_id)
            for edge in edges:
                if edge.to_entity == subject_id and edge.relation_type == cond.field:
                    return edge.value >= cond.value
            return False

        elif cond.check_type == "has_item":
            return state.inventory.has_item(subject_id, cond.value)

        elif cond.check_type == "skill_gte":
            # field = skill name, value = required level
            skill_level = state.skills.get_level(subject_id, cond.field or "")
            return skill_level >= (cond.value or 0)

        elif cond.check_type == "has_item_type":
            # value = item_type name
            return state.inventory.has_item_type(subject_id, cond.value)

        return False  # Unknown check_type

    def all_goals_complete(self, entity_id: str) -> bool:
        """Check if all goals for an entity are completed (or failed)."""
        goals = self._goals.get(entity_id, [])
        if not goals:
            return False
        return all(g.completed or g.failed for g in goals)

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        return {
            eid: [
                {
                    "id": g.id,
                    "description": g.description,
                    "priority": g.priority,
                    "completed": g.completed,
                    "completed_round": g.completed_round,
                    "failed": g.failed,
                    "conditions": [
                        {
                            "check_type": c.check_type,
                            "subject": c.subject,
                            "field": c.field,
                            "value": c.value,
                        }
                        for c in g.conditions
                    ],
                }
                for g in goals
            ]
            for eid, goals in self._goals.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalTracker":
        """Restore from a to_dict snapshot."""
        tracker = cls()
        for eid, goal_list in (data or {}).items():
            for gd in goal_list:
                conditions = [
                    GoalCondition(
                        check_type=c["check_type"],
                        subject=c.get("subject", "self"),
                        field=c.get("field"),
                        value=c.get("value"),
                    )
                    for c in gd.get("conditions", [])
                ]
                tracker._goals.setdefault(eid, []).append(Goal(
                    id=gd["id"],
                    description=gd.get("description", ""),
                    owner_id=eid,
                    priority=gd.get("priority", 1),
                    completed=gd.get("completed", False),
                    completed_round=gd.get("completed_round"),
                    failed=gd.get("failed", False),
                    conditions=conditions,
                ))
        return tracker
