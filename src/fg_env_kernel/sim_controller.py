"""Mid-simulation controller -- event injection, breakpoints, agent takeover, narrative directives.

Provides runtime control surfaces for a running simulation:
- Event injection: insert events/scenarios mid-run
- Breakpoints: pause when conditions are met
- Agent takeover: human controls an agent for N turns
- Narrative directives: automated story beats triggered by conditions
"""
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class EventInjection:
    """A queued event to inject into the simulation."""
    id: str = ""
    event_type: str = "injected_event"
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    target_entities: List[str] = field(default_factory=list)  # Empty = global
    round_delay: int = 0  # 0 = next round, >0 = delayed
    effects: List[Dict[str, Any]] = field(default_factory=list)  # Effect dicts to apply
    _queued_at_round: int = -1

    def __post_init__(self):
        if not self.id:
            self.id = f"inj_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "description": self.description,
            "data": self.data,
            "target_entities": self.target_entities,
            "round_delay": self.round_delay,
            "effects": self.effects,
            "_queued_at_round": self._queued_at_round,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventInjection":
        inj = cls(
            id=data.get("id", ""),
            event_type=data.get("event_type", "injected_event"),
            description=data.get("description", ""),
            data=data.get("data", {}),
            target_entities=data.get("target_entities", []),
            round_delay=data.get("round_delay", 0),
            effects=data.get("effects", []),
        )
        inj._queued_at_round = data.get("_queued_at_round", -1)
        return inj


@dataclass
class BreakpointCondition:
    """A condition for a breakpoint to trigger."""
    check_type: str = "property_threshold"
    # Same format as TerminationCondition params
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"check_type": self.check_type, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict) -> "BreakpointCondition":
        return cls(
            check_type=data.get("check_type", "property_threshold"),
            params=data.get("params", {}),
        )


@dataclass
class Breakpoint:
    """A conditional breakpoint that pauses or logs when triggered."""
    id: str = ""
    name: str = ""
    condition: BreakpointCondition = field(default_factory=BreakpointCondition)
    action: str = "pause"  # "pause", "log", "callback"
    one_shot: bool = False  # If True, auto-remove after triggering
    enabled: bool = True
    _triggered_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"bp_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "condition": self.condition.to_dict(),
            "action": self.action,
            "one_shot": self.one_shot,
            "enabled": self.enabled,
            "_triggered_count": self._triggered_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Breakpoint":
        bp = cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            condition=BreakpointCondition.from_dict(data.get("condition", {})),
            action=data.get("action", "pause"),
            one_shot=data.get("one_shot", False),
            enabled=data.get("enabled", True),
        )
        bp._triggered_count = data.get("_triggered_count", 0)
        return bp


@dataclass
class AgentTakeover:
    """A human takeover of an agent for N turns."""
    entity_id: str
    remaining_turns: int = 1
    _human_decision_fn: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "remaining_turns": self.remaining_turns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTakeover":
        return cls(
            entity_id=data["entity_id"],
            remaining_turns=data.get("remaining_turns", 1),
        )


@dataclass
class NarrativeDirective:
    """A story beat that triggers when conditions are met."""
    id: str = ""
    name: str = ""
    condition: BreakpointCondition = field(default_factory=BreakpointCondition)
    narrative_event: str = ""  # Event description to inject
    event_data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = evaluated first
    one_shot: bool = True  # Most story beats fire once
    enabled: bool = True
    _fired: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = f"dir_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "condition": self.condition.to_dict(),
            "narrative_event": self.narrative_event,
            "event_data": self.event_data,
            "priority": self.priority,
            "one_shot": self.one_shot,
            "enabled": self.enabled,
            "_fired": self._fired,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NarrativeDirective":
        d = cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            condition=BreakpointCondition.from_dict(data.get("condition", {})),
            narrative_event=data.get("narrative_event", ""),
            event_data=data.get("event_data", {}),
            priority=data.get("priority", 0),
            one_shot=data.get("one_shot", True),
            enabled=data.get("enabled", True),
        )
        d._fired = data.get("_fired", False)
        return d


class SimController:
    """Mid-simulation controller.
    
    Manages event injections, breakpoints, agent takeovers, and
    narrative directives. Checked by the engine at various points
    during the simulation loop.
    """

    def __init__(self):
        self._pending_injections: List[EventInjection] = []
        self._breakpoints: Dict[str, Breakpoint] = {}
        self._takeovers: Dict[str, AgentTakeover] = {}
        self._directives: Dict[str, NarrativeDirective] = {}

    # -------------------------------------------------------------------
    # Event Injection
    # -------------------------------------------------------------------

    def inject_event(self, injection: EventInjection, current_round: int = 0):
        """Queue an event for injection."""
        injection._queued_at_round = current_round
        self._pending_injections.append(injection)

    def process_pending_injections(self, current_round: int) -> List[EventInjection]:
        """Process and return injections whose delay has elapsed.
        
        Returns list of EventInjections that should fire this round.
        Removes them from the pending queue.
        """
        ready = []
        remaining = []
        for inj in self._pending_injections:
            target_round = inj._queued_at_round + inj.round_delay
            if current_round >= target_round:
                ready.append(inj)
            else:
                remaining.append(inj)
        self._pending_injections = remaining
        return ready

    def get_pending_injections(self) -> List[EventInjection]:
        """Get all pending injections (for inspection)."""
        return list(self._pending_injections)

    # -------------------------------------------------------------------
    # Breakpoints
    # -------------------------------------------------------------------

    def add_breakpoint(self, bp: Breakpoint):
        """Add a breakpoint."""
        self._breakpoints[bp.id] = bp

    def remove_breakpoint(self, bp_id: str) -> Optional[Breakpoint]:
        """Remove a breakpoint by ID."""
        return self._breakpoints.pop(bp_id, None)

    def get_breakpoint(self, bp_id: str) -> Optional[Breakpoint]:
        """Get a breakpoint by ID."""
        return self._breakpoints.get(bp_id)

    def list_breakpoints(self) -> List[Breakpoint]:
        """List all breakpoints."""
        return list(self._breakpoints.values())

    def check_breakpoints(self, state: Any, round_number: int) -> List[Breakpoint]:
        """Evaluate all breakpoints. Returns list of triggered breakpoints.
        
        One-shot breakpoints are auto-removed after triggering.
        """
        triggered = []
        to_remove = []

        for bp in self._breakpoints.values():
            if not bp.enabled:
                continue
            if self._evaluate_condition(bp.condition, state, round_number):
                bp._triggered_count += 1
                triggered.append(bp)
                if bp.one_shot:
                    to_remove.append(bp.id)

        for bp_id in to_remove:
            self._breakpoints.pop(bp_id, None)

        return triggered

    # -------------------------------------------------------------------
    # Agent Takeover
    # -------------------------------------------------------------------

    def start_takeover(self, entity_id: str, num_turns: int = 1,
                       decision_fn: Optional[Callable] = None):
        """Start human takeover of an agent."""
        self._takeovers[entity_id] = AgentTakeover(
            entity_id=entity_id,
            remaining_turns=num_turns,
            _human_decision_fn=decision_fn,
        )

    def end_takeover(self, entity_id: str) -> Optional[AgentTakeover]:
        """End takeover of an agent."""
        return self._takeovers.pop(entity_id, None)

    def is_taken_over(self, entity_id: str) -> Optional[AgentTakeover]:
        """Check if an agent is under human control. Returns takeover or None."""
        return self._takeovers.get(entity_id)

    def tick_takeovers(self):
        """Decrement remaining turns on all takeovers. Remove expired ones."""
        expired = []
        for eid, takeover in self._takeovers.items():
            takeover.remaining_turns -= 1
            if takeover.remaining_turns <= 0:
                expired.append(eid)
        for eid in expired:
            self._takeovers.pop(eid, None)

    def get_active_takeovers(self) -> Dict[str, AgentTakeover]:
        """Get all active takeovers."""
        return dict(self._takeovers)

    # -------------------------------------------------------------------
    # Narrative Directives
    # -------------------------------------------------------------------

    def add_directive(self, directive: NarrativeDirective):
        """Add a narrative directive."""
        self._directives[directive.id] = directive

    def remove_directive(self, directive_id: str) -> Optional[NarrativeDirective]:
        """Remove a directive by ID."""
        return self._directives.pop(directive_id, None)

    def check_directives(self, state: Any, round_number: int) -> List[NarrativeDirective]:
        """Evaluate all directives. Returns those that should fire this round.
        
        Directives are evaluated in priority order (highest first).
        One-shot directives are disabled after firing.
        """
        # Sort by priority descending
        sorted_directives = sorted(
            self._directives.values(),
            key=lambda d: d.priority,
            reverse=True,
        )

        fired = []
        for directive in sorted_directives:
            if not directive.enabled:
                continue
            if directive._fired and directive.one_shot:
                continue
            if self._evaluate_condition(directive.condition, state, round_number):
                directive._fired = True
                if directive.one_shot:
                    directive.enabled = False
                fired.append(directive)

        return fired

    def list_directives(self) -> List[NarrativeDirective]:
        """List all directives."""
        return list(self._directives.values())

    # -------------------------------------------------------------------
    # Condition Evaluation
    # -------------------------------------------------------------------

    def _evaluate_condition(self, condition: BreakpointCondition,
                            state: Any, round_number: int) -> bool:
        """Evaluate a breakpoint/directive condition against current state.
        
        Supports the same check_types as TerminationCondition.
        """
        check = condition.check_type
        params = condition.params

        if check == "property_threshold":
            entity_type = params.get("entity_type")
            prop = params.get("property")
            operator = params.get("operator", "gte")
            value = params.get("value", 0)
            if not entity_type or not prop:
                return False
            entities = state.get_entities_by_type(entity_type) if hasattr(state, 'get_entities_by_type') else []
            for e in entities:
                if not e.alive:
                    continue
                v = e.get(prop, 0)
                if self._compare(v, operator, value):
                    return True
            return False

        elif check == "round_reached":
            target_round = params.get("round", 0)
            return round_number >= target_round

        elif check == "entity_at_location":
            entity_id = params.get("entity_id")
            location = params.get("location")
            if not entity_id or not location:
                return False
            if hasattr(state, 'locations'):
                return state.locations.get(entity_id) == location
            return False

        elif check == "resource_threshold":
            resource = params.get("resource")
            operator = params.get("operator", "lte")
            value = params.get("value", 0)
            if not resource:
                return False
            if hasattr(state, 'resources'):
                pool = state.resources.get(resource)
                if pool:
                    total = sum(pool.holdings.values()) + pool.unallocated
                    return self._compare(total, operator, value)
            return False

        elif check == "all_dead":
            entity_type = params.get("entity_type")
            if not entity_type:
                return False
            entities = state.get_entities_by_type(entity_type) if hasattr(state, 'get_entities_by_type') else []
            return len(entities) > 0 and all(not e.alive for e in entities)

        elif check == "event_occurred":
            event_type = params.get("event_type")
            if not event_type or not hasattr(state, 'event_log'):
                return False
            events = state.event_log.get_by_type(event_type)
            min_count = params.get("count", 1)
            return len(events) >= min_count

        elif check == "always":
            return True

        elif check == "never":
            return False

        return False

    @staticmethod
    def _compare(val, operator: str, target_val) -> bool:
        """Compare values using a string operator."""
        try:
            if operator == "gte":
                return val >= target_val
            elif operator == "lte":
                return val <= target_val
            elif operator == "gt":
                return val > target_val
            elif operator == "lt":
                return val < target_val
            elif operator == "eq":
                return val == target_val
            elif operator == "neq":
                return val != target_val
        except (TypeError, ValueError):
            return False
        return False

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "pending_injections": [inj.to_dict() for inj in self._pending_injections],
            "breakpoints": {
                bp_id: bp.to_dict()
                for bp_id, bp in self._breakpoints.items()
            },
            "takeovers": {
                eid: t.to_dict()
                for eid, t in self._takeovers.items()
            },
            "directives": {
                d_id: d.to_dict()
                for d_id, d in self._directives.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimController":
        ctrl = cls()
        for inj_data in data.get("pending_injections", []):
            ctrl._pending_injections.append(EventInjection.from_dict(inj_data))
        for bp_id, bp_data in data.get("breakpoints", {}).items():
            ctrl._breakpoints[bp_id] = Breakpoint.from_dict(bp_data)
        for eid, t_data in data.get("takeovers", {}).items():
            ctrl._takeovers[eid] = AgentTakeover.from_dict(t_data)
        for d_id, d_data in data.get("directives", {}).items():
            ctrl._directives[d_id] = NarrativeDirective.from_dict(d_data)
        return ctrl
