"""Turn-based temporal model."""
import random as random_module
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TimeMode(Enum):
    """Temporal mode for the simulation."""
    DISCRETE = "discrete"      # Turn-based (default, existing behavior)
    CONTINUOUS = "continuous"   # Event-driven with real-time scheduling


@dataclass
class Phase:
    """A named phase within a round."""
    name: str
    description: str = ""
    # Which entity roles act during this phase (empty = all agent roles)
    active_roles: List[str] = field(default_factory=list)
    # Initiative / turn order
    initiative_type: str = "fixed"       # "fixed", "random", "property_based", "round_robin_shuffle"
    initiative_property: Optional[str] = None  # Property name for property_based (e.g., "speed")
    initiative_descending: bool = True    # True = highest property goes first
    # Automated handler that runs before agent turns
    handler: Optional[str] = None                                   # Registry key (e.g., "card_deal")
    handler_params: Dict[str, Any] = field(default_factory=dict)    # Handler-specific config
    # Resolution mode for this phase:
    #   "sequential"   — each agent acts and resolves before the next (default)
    #   "simultaneous" — every eligible agent submits an action against the
    #                    SAME perception, then all actions resolve in a batch.
    #                    This is the "commit-then-reveal" primitive: no agent
    #                    can react to another's move within the phase. Used
    #                    for RPS, sealed bids, blind voting, simultaneous
    #                    moves in social deduction games.
    resolution_mode: str = "sequential"


@dataclass
class TemporalModel:
    """
    Turn-based temporal tracking with optional phases and time mapping.

    A round consists of phases. Each phase, eligible agents choose actions.
    Default: single phase "action" per round.

    Time mapping (optional): ``round_duration_seconds`` maps each round to
    a real-world duration.  When set, agents receive time context in their
    perception and the frontend can render time-based chart labels.
    None means legacy mode — rounds are just counters.
    """
    mode: TimeMode = TimeMode.DISCRETE
    phases: List[Phase] = field(default_factory=lambda: [Phase(name="action")])
    current_round: int = 0
    current_phase_index: int = 0
    turn_order: List[str] = field(default_factory=list)
    current_turn_index: int = 0

    # ── Time mapping (optional) ──
    round_duration_seconds: Optional[int] = None   # None = legacy, rounds have no time
    sim_start_iso: Optional[str] = None            # ISO 8601 anchor for absolute timestamps
    time_unit_label: Optional[str] = None          # "minute", "hour", "day", "week"

    @property
    def current_phase(self) -> Phase:
        """Get the current phase. Returns an inert default if no phases
        are declared yet (used during partial Studio drafts so the engine
        doesn't hard-crash on schemas being built incrementally)."""
        if not self.phases:
            return Phase(name="(no phases)", active_roles=[], description="")
        # Clamp index defensively too — a malformed advance_phase could
        # otherwise leave us out of bounds.
        idx = max(0, min(self.current_phase_index, len(self.phases) - 1))
        return self.phases[idx]

    # ── Time helpers ──

    def _auto_unit_label(self) -> str:
        """Derive a human unit label from round_duration_seconds."""
        if self.time_unit_label:
            return self.time_unit_label
        rds = self.round_duration_seconds or 0
        if rds >= 604800:
            return "week"
        if rds >= 86400:
            return "day"
        if rds >= 3600:
            return "hour"
        if rds >= 60:
            return "minute"
        return "second"

    def current_time_label(self) -> Optional[str]:
        """Human-readable label for the current round, e.g. 'Hour 5' or 'Day 15'."""
        if self.round_duration_seconds is None:
            return None
        unit = self._auto_unit_label()
        return f"{unit.capitalize()} {self.current_round}"

    def round_to_time_label(self, round_num: int) -> str:
        """Convert any round number to a time label.

        Negative rounds (pre-sim history) produce labels like 'Hour -5'.
        """
        if self.round_duration_seconds is None:
            return f"R{round_num}"
        unit = self._auto_unit_label()
        return f"{unit.capitalize()} {round_num}"

    def time_context(self, max_rounds: int) -> Optional[dict]:
        """Full time context dict for agent perception.

        Returns None if time is not configured (legacy mode).
        """
        if self.round_duration_seconds is None:
            return None
        rds = self.round_duration_seconds
        unit = self._auto_unit_label()
        elapsed = self.current_round * rds
        total = max_rounds * rds
        pct = (self.current_round / max_rounds * 100) if max_rounds > 0 else 0

        # Human-readable total duration (use the same unit as round_duration)
        if rds >= 604800:
            total_label = f"{max_rounds}-week"
        elif rds >= 86400:
            total_label = f"{max_rounds}-day"
        elif rds >= 3600:
            total_label = f"{max_rounds}-hour"
        elif rds >= 60:
            total_label = f"{max_rounds}-minute"
        else:
            total_label = f"{max_rounds}-second"

        return {
            "current_label": f"{unit.capitalize()} {self.current_round}",
            "elapsed_seconds": elapsed,
            "total_seconds": total,
            "progress_pct": round(pct, 1),
            "round_duration_seconds": rds,
            "unit": unit,
            "human_summary": (
                f"{unit.capitalize()} {self.current_round} of a {total_label} "
                f"trading session ({pct:.0f}% complete)"
            ),
        }

    def advance_turn(self) -> Optional[str]:
        """Advance to the next turn. Returns next actor entity_id, or None if phase complete.

        Note: The SimulationEngine iterates turn_order directly in _run_phase()
        rather than calling this method, for flexibility. This method is available
        for external code (e.g. step-through debugging, custom engine loops).
        """
        self.current_turn_index += 1
        if self.current_turn_index >= len(self.turn_order):
            return None
        return self.turn_order[self.current_turn_index]

    def advance_phase(self) -> bool:
        """Advance to the next phase. Returns True if moved, False if round complete."""
        self.current_phase_index += 1
        self.current_turn_index = 0
        if self.current_phase_index >= len(self.phases):
            return False
        return True

    def advance_round(self):
        """Move to the next round, resetting phase and turn counters."""
        self.current_round += 1
        self.current_phase_index = 0
        self.current_turn_index = 0

    def set_turn_order(self, entity_ids: List[str]):
        """Set the turn order for the current phase."""
        self.turn_order = entity_ids
        self.current_turn_index = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        # current_phase_index may be past end after advance_phase() exhausts phases
        if self.current_phase_index < len(self.phases):
            phase_name = self.phases[self.current_phase_index].name
        else:
            phase_name = self.phases[-1].name if self.phases else "none"
        d = {
            "mode": self.mode.value,
            "current_round": self.current_round,
            "current_phase": phase_name,
            "current_turn_index": self.current_turn_index,
            "turn_order": self.turn_order,
            "phases": [p.name for p in self.phases],
        }
        if self.round_duration_seconds is not None:
            d["round_duration_seconds"] = self.round_duration_seconds
        if self.sim_start_iso:
            d["sim_start_iso"] = self.sim_start_iso
        if self.time_unit_label:
            d["time_unit_label"] = self.time_unit_label
        return d


class TurnOrderResolver:
    """Determines turn order based on phase initiative settings."""

    @staticmethod
    def resolve(
        agents: List[Any],
        phase: Phase,
        round_number: int,
        rng: Optional[random_module.Random] = None,
    ) -> List[str]:
        """Resolve turn order for a list of agents in a given phase.

        Args:
            agents: List of Entity objects (must have .id and .properties)
            phase: The current Phase (contains initiative settings)
            round_number: Current round number
            rng: Optional seeded Random instance for deterministic results

        Returns:
            List of entity IDs in turn order
        """
        if not agents:
            return []

        ids = [a.id for a in agents]

        if phase.initiative_type == "fixed":
            return ids

        if rng is None:
            rng = random_module.Random()

        if phase.initiative_type == "random":
            shuffled = list(ids)
            rng.shuffle(shuffled)
            return shuffled

        if phase.initiative_type == "property_based":
            prop = phase.initiative_property or "speed"
            # Sort by property value
            sorted_agents = sorted(
                agents,
                key=lambda a: float(a.get(prop, 0) if hasattr(a, 'get') else getattr(a, prop, 0)),
                reverse=phase.initiative_descending,
            )
            return [a.id for a in sorted_agents]

        if phase.initiative_type == "round_robin_shuffle":
            shuffled = list(ids)
            # Use round_number as part of seed for varying but deterministic order
            round_rng = random_module.Random(rng.random() + round_number)
            round_rng.shuffle(shuffled)
            return shuffled

        # Fallback to fixed
        return ids
