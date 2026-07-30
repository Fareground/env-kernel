"""Multi-turn agent planning and theory of mind.

Agents can form multi-step plans toward goals, track progress, and
detect when plans become infeasible (triggering replanning). Theory
of mind lets agents build models of other agents' goals and predict
their next actions based on observed behavior.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Plan structures
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a multi-step plan."""
    action: str                         # Action name to perform
    target_id: Optional[str] = None     # Target entity for the action
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""          # Human-readable expected result
    status: StepStatus = StepStatus.PENDING
    precondition: Optional[str] = None  # Optional: what must be true before this step
    actual_outcome: str = ""            # Filled in after execution

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target_id": self.target_id,
            "parameters": self.parameters,
            "expected_outcome": self.expected_outcome,
            "status": self.status.value,
            "precondition": self.precondition,
            "actual_outcome": self.actual_outcome,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        return cls(
            action=data["action"],
            target_id=data.get("target_id"),
            parameters=data.get("parameters", {}),
            expected_outcome=data.get("expected_outcome", ""),
            status=StepStatus(data.get("status", "pending")),
            precondition=data.get("precondition"),
            actual_outcome=data.get("actual_outcome", ""),
        )


@dataclass
class Plan:
    """A multi-step plan tied to a goal."""
    id: str
    owner_id: str
    goal_description: str               # What this plan aims to achieve
    steps: List[PlanStep] = field(default_factory=list)
    current_step: int = 0               # Index of the next step to execute
    round_created: int = 0
    round_last_validated: int = 0
    active: bool = True
    invalidated: bool = False           # Set when plan becomes infeasible
    invalidation_reason: str = ""

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step, or None if plan is complete/empty."""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance(self, success: bool = True, outcome: str = "") -> bool:
        """Mark current step as complete/failed and advance. Returns True if plan complete."""
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step.status = StepStatus.COMPLETED if success else StepStatus.FAILED
            step.actual_outcome = outcome
            self.current_step += 1

        # Check if plan is complete
        return self.current_step >= len(self.steps)

    def remaining_steps(self) -> List[PlanStep]:
        """Get steps that haven't been executed yet."""
        return [s for s in self.steps[self.current_step:] if s.status == StepStatus.PENDING]

    def progress_fraction(self) -> float:
        """Return completion fraction (0.0 to 1.0)."""
        if not self.steps:
            return 1.0
        completed = sum(1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED))
        return completed / len(self.steps)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "goal_description": self.goal_description,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "round_created": self.round_created,
            "round_last_validated": self.round_last_validated,
            "active": self.active,
            "invalidated": self.invalidated,
            "invalidation_reason": self.invalidation_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        plan = cls(
            id=data["id"],
            owner_id=data["owner_id"],
            goal_description=data.get("goal_description", ""),
            current_step=data.get("current_step", 0),
            round_created=data.get("round_created", 0),
            round_last_validated=data.get("round_last_validated", 0),
            active=data.get("active", True),
            invalidated=data.get("invalidated", False),
            invalidation_reason=data.get("invalidation_reason", ""),
        )
        plan.steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return plan


# ---------------------------------------------------------------------------
# Theory of Mind
# ---------------------------------------------------------------------------

@dataclass
class AgentModel:
    """
    Theory-of-mind model: what one agent believes about another.

    Built from observing the target's actions over time. Tracks inferred
    goals, predicted next action, and behavioral patterns.
    """
    target_id: str
    target_name: str = ""
    inferred_goals: List[str] = field(default_factory=list)
    predicted_next_action: Optional[str] = None
    behavioral_pattern: str = ""        # e.g. "aggressive", "cooperative", "hoarding"
    observed_actions: List[Dict[str, Any]] = field(default_factory=list)  # Recent action log
    confidence: float = 0.5             # How confident observer is in this model
    last_updated_round: int = 0

    # Keep last N observed actions to detect patterns
    MAX_OBSERVATIONS = 10

    def record_action(self, action_name: str, target_id: Optional[str], success: bool, round_num: int):
        """Record an observed action by the modeled agent."""
        self.observed_actions.append({
            "action": action_name,
            "target_id": target_id,
            "success": success,
            "round": round_num,
        })
        # Trim to max
        if len(self.observed_actions) > self.MAX_OBSERVATIONS:
            self.observed_actions = self.observed_actions[-self.MAX_OBSERVATIONS:]

        self.last_updated_round = round_num
        self._update_predictions()

    def _update_predictions(self):
        """Update predicted action and behavioral pattern from observations."""
        if not self.observed_actions:
            return

        # Predict next action: most frequent recent action
        from collections import Counter
        action_counts = Counter(obs["action"] for obs in self.observed_actions[-5:])
        if action_counts:
            self.predicted_next_action = action_counts.most_common(1)[0][0]

        # Detect behavioral pattern from action tendencies
        self.behavioral_pattern = self._detect_pattern()

        # Confidence increases with more observations
        self.confidence = min(1.0, 0.3 + 0.07 * len(self.observed_actions))

    def _detect_pattern(self) -> str:
        """Detect a high-level behavioral pattern from observed actions."""
        if len(self.observed_actions) < 3:
            return "unknown"

        action_names = [obs["action"] for obs in self.observed_actions]
        targets = [obs.get("target_id") for obs in self.observed_actions if obs.get("target_id")]

        # Count categories by action name keywords
        aggressive_keywords = {"attack", "steal", "raid", "sabotage", "fight", "ambush", "threaten"}
        cooperative_keywords = {"trade", "help", "heal", "share", "gift", "cooperate", "ally", "negotiate"}
        gathering_keywords = {"gather", "mine", "harvest", "collect", "forage", "fish", "hunt"}

        aggressive_count = sum(1 for a in action_names if any(kw in a.lower() for kw in aggressive_keywords))
        cooperative_count = sum(1 for a in action_names if any(kw in a.lower() for kw in cooperative_keywords))
        gathering_count = sum(1 for a in action_names if any(kw in a.lower() for kw in gathering_keywords))

        total = len(action_names)
        if aggressive_count / total > 0.4:
            return "aggressive"
        elif cooperative_count / total > 0.4:
            return "cooperative"
        elif gathering_count / total > 0.4:
            return "resource-focused"

        # Check if they target the same entity repeatedly
        if targets:
            from collections import Counter
            target_counts = Counter(targets)
            most_targeted, count = target_counts.most_common(1)[0]
            if count >= 3:
                return "fixated"

        # Check if all actions are the same (monotonous)
        unique_actions = set(action_names[-5:])
        if len(unique_actions) == 1:
            return "repetitive"

        return "versatile"

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "target_name": self.target_name,
            "inferred_goals": list(self.inferred_goals),
            "predicted_next_action": self.predicted_next_action,
            "behavioral_pattern": self.behavioral_pattern,
            "observed_actions": list(self.observed_actions),
            "confidence": self.confidence,
            "last_updated_round": self.last_updated_round,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentModel":
        return cls(
            target_id=data["target_id"],
            target_name=data.get("target_name", ""),
            inferred_goals=data.get("inferred_goals", []),
            predicted_next_action=data.get("predicted_next_action"),
            behavioral_pattern=data.get("behavioral_pattern", ""),
            observed_actions=data.get("observed_actions", []),
            confidence=data.get("confidence", 0.5),
            last_updated_round=data.get("last_updated_round", 0),
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class PlanManager:
    """Manages per-agent plans and theory-of-mind agent models."""

    def __init__(self, max_models_per_agent: int = 5):
        self._plans: Dict[str, Plan] = {}          # plan_id -> Plan
        self._agent_plans: Dict[str, str] = {}     # entity_id -> active plan_id
        self._agent_models: Dict[str, Dict[str, AgentModel]] = {}  # observer_id -> {target_id -> AgentModel}
        self._max_models = max_models_per_agent
        self._plan_counter: int = 0

    # -- Plan management --

    def set_plan(
        self,
        owner_id: str,
        goal_description: str,
        steps: List[PlanStep],
        round_num: int = 0,
    ) -> Plan:
        """Create and activate a new plan for an agent, replacing any existing one."""
        self._plan_counter += 1
        plan_id = f"plan_{owner_id}_{self._plan_counter}"

        # Deactivate old plan
        old_plan_id = self._agent_plans.get(owner_id)
        if old_plan_id and old_plan_id in self._plans:
            self._plans[old_plan_id].active = False

        plan = Plan(
            id=plan_id,
            owner_id=owner_id,
            goal_description=goal_description,
            steps=steps,
            round_created=round_num,
            round_last_validated=round_num,
        )
        self._plans[plan_id] = plan
        self._agent_plans[owner_id] = plan_id
        return plan

    def remove_entity(self, entity_id: str) -> None:
        """Remove all plans and agent-models owned by or referencing this entity."""
        plan_id = self._agent_plans.pop(entity_id, None)
        if plan_id:
            self._plans.pop(plan_id, None)
        self._agent_models.pop(entity_id, None)
        # Also remove this entity as a target inside other observers' models.
        for models in self._agent_models.values():
            models.pop(entity_id, None)

    def get_active_plan(self, entity_id: str) -> Optional[Plan]:
        """Get the current active plan for an agent."""
        plan_id = self._agent_plans.get(entity_id)
        if not plan_id:
            return None
        plan = self._plans.get(plan_id)
        if plan and plan.active and not plan.invalidated:
            return plan
        return None

    def advance_plan(self, entity_id: str, success: bool = True, outcome: str = "") -> bool:
        """Advance the agent's active plan. Returns True if plan completed."""
        plan = self.get_active_plan(entity_id)
        if not plan:
            return False
        completed = plan.advance(success, outcome)
        if completed:
            plan.active = False
        return completed

    def invalidate_plan(self, entity_id: str, reason: str = ""):
        """Mark an agent's active plan as invalidated (needs replanning)."""
        plan = self.get_active_plan(entity_id)
        if plan:
            plan.invalidated = True
            plan.invalidation_reason = reason

    # -- Plan validation --

    def validate_plan(self, entity_id: str, valid_actions: List[str], round_num: int) -> bool:
        """
        Validate remaining steps of the active plan against available actions.

        Returns True if plan is still feasible. Invalidates if not.
        """
        plan = self.get_active_plan(entity_id)
        if not plan:
            return True  # No plan to validate

        plan.round_last_validated = round_num

        for step in plan.remaining_steps():
            if step.action not in valid_actions:
                plan.invalidated = True
                plan.invalidation_reason = f"Action '{step.action}' is no longer available"
                return False

        return True

    def needs_replanning(self, entity_id: str, round_num: int, replan_interval: int = 5) -> bool:
        """Check if the agent needs to create or refresh a plan."""
        plan = self.get_active_plan(entity_id)
        if not plan:
            return True  # No plan at all
        if plan.invalidated:
            return True  # Plan was invalidated
        # Periodic replanning
        if round_num - plan.round_created >= replan_interval:
            return True
        return False

    # -- Theory of Mind --

    def update_agent_model(
        self,
        observer_id: str,
        target_id: str,
        target_name: str,
        action_name: str,
        action_target_id: Optional[str],
        success: bool,
        round_num: int,
    ):
        """Update an observer's theory-of-mind model of a target agent."""
        if observer_id == target_id:
            return  # Don't model self

        if observer_id not in self._agent_models:
            self._agent_models[observer_id] = {}

        models = self._agent_models[observer_id]

        if target_id not in models:
            # Check capacity
            if len(models) >= self._max_models:
                # Evict least recently updated model
                oldest_id = min(models, key=lambda k: models[k].last_updated_round)
                del models[oldest_id]
            models[target_id] = AgentModel(target_id=target_id, target_name=target_name)

        models[target_id].record_action(action_name, action_target_id, success, round_num)

    def get_agent_model(self, observer_id: str, target_id: str) -> Optional[AgentModel]:
        """Get an observer's model of a target agent."""
        models = self._agent_models.get(observer_id, {})
        return models.get(target_id)

    def get_all_models(self, observer_id: str) -> Dict[str, AgentModel]:
        """Get all theory-of-mind models for an observer."""
        return dict(self._agent_models.get(observer_id, {}))

    def summarize_for_prompt(
        self,
        entity_id: str,
        max_plan_steps: int = 3,
        max_models: int = 5,
    ) -> Dict[str, Any]:
        """
        Build a prompt-ready summary of the agent's plan and agent models.

        Returns dict with optional "plan" and "agent_models" keys.
        """
        result = {}

        # Active plan
        plan = self.get_active_plan(entity_id)
        if plan:
            current = plan.get_current_step()
            remaining = plan.remaining_steps()[:max_plan_steps]
            plan_summary = {
                "goal": plan.goal_description,
                "progress": f"{plan.current_step}/{len(plan.steps)}",
                "current_step": current.to_dict() if current else None,
                "upcoming_steps": [s.to_dict() for s in remaining[1:]] if len(remaining) > 1 else [],
            }
            result["plan"] = plan_summary

        # Agent models (top N by confidence)
        models = self.get_all_models(entity_id)
        if models:
            sorted_models = sorted(models.values(), key=lambda m: m.confidence, reverse=True)
            model_summaries = []
            for m in sorted_models[:max_models]:
                summary = {
                    "agent": m.target_name or m.target_id,
                    "pattern": m.behavioral_pattern,
                    "predicted_action": m.predicted_next_action,
                    "confidence": round(m.confidence, 2),
                }
                if m.inferred_goals:
                    summary["inferred_goals"] = m.inferred_goals[:2]
                model_summaries.append(summary)
            result["agent_models"] = model_summaries

        return result

    # -- Serialization --

    def to_dict(self) -> dict:
        return {
            "plans": {pid: p.to_dict() for pid, p in self._plans.items()},
            "agent_plans": dict(self._agent_plans),
            "agent_models": {
                obs_id: {tid: m.to_dict() for tid, m in models.items()}
                for obs_id, models in self._agent_models.items()
            },
            "max_models": self._max_models,
            "plan_counter": self._plan_counter,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanManager":
        mgr = cls(max_models_per_agent=data.get("max_models", 5))
        mgr._plan_counter = data.get("plan_counter", 0)
        mgr._agent_plans = data.get("agent_plans", {})
        for pid, pdata in data.get("plans", {}).items():
            mgr._plans[pid] = Plan.from_dict(pdata)
        for obs_id, models_data in data.get("agent_models", {}).items():
            mgr._agent_models[obs_id] = {
                tid: AgentModel.from_dict(mdata)
                for tid, mdata in models_data.items()
            }
        return mgr
