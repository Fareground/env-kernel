"""PhaseStateMachineModule — declarative multi-step rituals (Phase 4).

Replaces the bespoke phase-loop code in poker betting rounds, mafia
day/night cycles, parliament debates, dispute_trial procedures,
united_nations deliberation, judged_contest rounds — anywhere a game
has named steps that gate which actions are legal and that advance
based on real conditions (everyone voted, N rounds elapsed, all
folded but one, etc.).

Sits as a DomainModule (not a phase_handler) because it needs the
`post_resolution` + `filter_valid_actions` + `tick` event hooks the
DomainModule base already provides.

═══════════════════════════════════════════════════════════════════════
USAGE — mafia day/night cycle
═══════════════════════════════════════════════════════════════════════

    domain_modules:
      - name: state_machine
        params:
          id: "mafia_cycle"
          initial_state: "night"
          states:
            night:
              actions_allowed: ["kill", "save", "investigate"]
              transitions:
                - when: "all_required_acted"
                  required_actions: ["kill", "save", "investigate"]
                  to: "day_discussion"
            day_discussion:
              actions_allowed: ["speak", "accuse"]
              transitions:
                - when: "rounds_in_state >= 3"
                  to: "voting"
            voting:
              actions_allowed: ["vote"]
              transitions:
                - when: "all_voted"
                  to: "day_resolved"
            day_resolved:
              actions_allowed: []
              transitions:
                - when: "event_emitted"
                  event_type: "town_wins"
                  to: "game_over"
                - when: "always"
                  to: "night"
              on_enter:
                - { operation: "emit_event", value: "mafia_day_resolved" }
            game_over:
              terminal: true

═══════════════════════════════════════════════════════════════════════
USAGE — poker betting round
═══════════════════════════════════════════════════════════════════════

    domain_modules:
      - name: state_machine
        params:
          id: "betting"
          initial_state: "preflop"
          states:
            preflop:
              actions_allowed: ["fold", "check", "call", "raise", "all_in"]
              transitions:
                - when: "all_remaining_called_or_all_in"
                  to: "flop"
                - when: "only_one_remaining"
                  to: "showdown"
            flop: { ... }            # similar

═══════════════════════════════════════════════════════════════════════
PREDICATE LIBRARY (the `when` clause)
═══════════════════════════════════════════════════════════════════════

  - "always"                           — always fires
  - "all_required_acted"               — every alive agent has taken one
                                         of `required_actions` in this state
  - "all_voted"                        — alias for all_required_acted with
                                         required_actions=["vote"]
  - "all_acted"                        — every alive agent has taken any
                                         action this state
  - "rounds_in_state >= N"             — N rounds elapsed since enter
  - "actions_in_state >= N"            — N total actions in this state
  - "only_one_remaining"               — only one agent is not eliminated
  - "all_remaining_called_or_all_in"   — every non-folded agent has
                                         "called" or "all_in" this state
  - "event_emitted"                    — a specific event_type has been
                                         logged since state entry
  - "entity_property_eq"               — entity_id.property == value
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from .domain_module import DomainModule

logger = logging.getLogger(__name__)


class PhaseStateMachineModule(DomainModule):
    """Declarative state machine over named states with transitions
    driven by predicates evaluated against engine state."""

    def __init__(
        self,
        name: str = "state_machine",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, params=params or {})

        self._id: str = self._params.get("id", "state_machine")
        self._states_spec: Dict[str, Dict[str, Any]] = dict(self._params.get("states") or {})
        self._initial_state: str = self._params.get(
            "initial_state",
            next(iter(self._states_spec.keys()), "init") if self._states_spec else "init",
        )

        # Runtime tracking
        self._current_state: str = self._initial_state
        # Rounds since this state was entered
        self._rounds_in_state: int = 0
        # Track which agents have done which actions since state entry
        self._actions_in_state: Dict[str, List[str]] = {}  # actor_id → [action_name, ...]
        # Events emitted since state entry (event_type strings)
        self._events_in_state: Set[str] = set()
        # Round when current state was entered
        self._state_entered_round: int = 0
        # Whether the state machine has ever fired its initial on_enter
        self._initialised: bool = False

    @property
    def description(self) -> str:
        return f"State machine '{self._id}' (current: {self._current_state})"

    # ────────────────────────────────────────────────────────────── #
    # Inspection
    # ────────────────────────────────────────────────────────────── #

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def id(self) -> str:
        return self._id

    @property
    def is_terminal(self) -> bool:
        return bool(self._state_spec().get("terminal"))

    def _state_spec(self) -> Dict[str, Any]:
        return self._states_spec.get(self._current_state, {})

    # ────────────────────────────────────────────────────────────── #
    # tick — initial entry + round counter
    # ────────────────────────────────────────────────────────────── #

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        if not self._initialised:
            self._initialised = True
            self._state_entered_round = round_number
            # Run on_enter for initial state. We can't apply EffectDSL
            # effects here (we don't have an actor), but we can emit a
            # status event so the UI can react.
            events.append({
                "type": "state_machine_state_entered",
                "machine_id": self._id,
                "state": self._current_state,
                "round": round_number,
            })

        # Update rounds_in_state
        self._rounds_in_state = max(0, round_number - self._state_entered_round)

        # Tick-driven transitions (e.g. `rounds_in_state >= N`,
        # `event_emitted`, predicates that don't need a fresh action).
        transitioned = self._maybe_transition(state=state, round_number=round_number)
        events.extend(transitioned)
        return events

    # ────────────────────────────────────────────────────────────── #
    # filter_valid_actions — gate by state's actions_allowed
    # ────────────────────────────────────────────────────────────── #

    def filter_valid_actions(
        self,
        entity_id: str,
        valid_actions: List[str],
        state: Any,
    ) -> List[str]:
        spec = self._state_spec()
        allowed = spec.get("actions_allowed")
        if allowed is None:
            return valid_actions  # state doesn't constrain
        if not allowed:
            return []  # state declares no actions (e.g. "resolved")
        return [a for a in valid_actions if a in set(allowed)]

    # ────────────────────────────────────────────────────────────── #
    # post_resolution — track action + try action-driven transitions
    # ────────────────────────────────────────────────────────────── #

    def post_resolution(
        self,
        actor_id: str,
        action_name: str,
        success: bool,
        result: Any,
        state: Any,
    ) -> List[Dict[str, Any]]:
        if not success:
            return []
        # If the current state declares `reset_others_on`, a matching
        # action wipes every OTHER actor's per-state log. Used by
        # betting rounds: a raise/bet forces previous callers to act
        # again.
        spec = self._state_spec()
        resets = spec.get("reset_others_on") or []
        if action_name in resets:
            mine = self._actions_in_state.get(actor_id, [])
            self._actions_in_state = {actor_id: list(mine)}
        # Track this action under the current state.
        self._actions_in_state.setdefault(actor_id, []).append(action_name)
        # Try transitions that depend on the latest action.
        return self._maybe_transition(state=state, round_number=getattr(state.temporal, "current_round", 0))

    # ────────────────────────────────────────────────────────────── #
    # Transition logic
    # ────────────────────────────────────────────────────────────── #

    def _maybe_transition(
        self, state: Any, round_number: int, _depth: int = 0,
    ) -> List[Dict[str, Any]]:
        """Evaluate the current state's transitions. If one fires,
        switch to its target state, emit on_transit/on_enter effects,
        and recursively re-evaluate (so chained transitions like
        `always` collapse on a single tick).

        `_depth` guards against cyclic `always` transitions; capped at
        len(states) which is the maximum non-repeating chain length."""
        out: List[Dict[str, Any]] = []
        if self.is_terminal:
            return out
        max_chain = max(8, len(self._states_spec) + 1)
        if _depth >= max_chain:
            logger.warning(
                f"state_machine[{self._id}]: aborting transition cascade at depth {_depth} "
                f"(cycle detected near state {self._current_state!r})"
            )
            return out

        spec = self._state_spec()
        transitions = spec.get("transitions", []) or []
        for trans in transitions:
            if not isinstance(trans, dict):
                continue
            if self._predicate(trans, state, round_number):
                target = trans.get("to")
                if not target or target not in self._states_spec:
                    logger.warning(
                        f"state_machine[{self._id}]: transition from "
                        f"{self._current_state} has invalid target {target!r}"
                    )
                    continue
                # Fire transition event
                out.append({
                    "type": "state_machine_transition",
                    "machine_id": self._id,
                    "from": self._current_state,
                    "to": target,
                    "predicate": trans.get("when"),
                    "round": round_number,
                })
                # Optional on_transit effects (JSON-style, applied via
                # the engine's _apply_effects elsewhere — we just emit
                # an event the engine can re-walk if needed).
                on_transit = trans.get("on_transit") or trans.get("on_exit")
                if on_transit:
                    out.append({
                        "type": "state_machine_on_transit",
                        "machine_id": self._id, "from": self._current_state, "to": target,
                        "effects": on_transit,
                    })
                # Switch state
                self._current_state = target
                self._state_entered_round = round_number
                self._rounds_in_state = 0
                self._actions_in_state.clear()
                self._events_in_state.clear()
                out.append({
                    "type": "state_machine_state_entered",
                    "machine_id": self._id, "state": target, "round": round_number,
                })
                # on_enter effects for the new state
                on_enter = (self._state_spec() or {}).get("on_enter")
                if on_enter:
                    out.append({
                        "type": "state_machine_on_enter",
                        "machine_id": self._id, "state": target,
                        "effects": on_enter,
                    })
                # Cascade: re-evaluate immediately (handles `always` chains
                # and terminal-collapse).
                out.extend(self._maybe_transition(state, round_number, _depth + 1))
                break  # only one transition fires per tick
        return out

    # ── Predicate evaluator ──

    def _predicate(self, trans: Dict[str, Any], state: Any, round_number: int) -> bool:
        when = trans.get("when")
        if not when:
            return False

        # Convenience shortcuts that parse a simple expression
        if when.startswith("rounds_in_state"):
            return self._cmp_int(when, "rounds_in_state", self._rounds_in_state)
        if when.startswith("actions_in_state"):
            total = sum(len(v) for v in self._actions_in_state.values())
            return self._cmp_int(when, "actions_in_state", total)

        if when == "always":
            return True

        if when == "all_acted":
            agents = self._alive_agents(state)
            return bool(agents) and all(
                aid in self._actions_in_state for aid in (a.id for a in agents)
            )

        if when == "all_required_acted" or when == "all_voted":
            required = trans.get("required_actions")
            if when == "all_voted" and required is None:
                required = ["vote"]
            required = set(required or [])
            agents = self._alive_agents(state)
            if not agents:
                return False
            for ag in agents:
                actions = self._actions_in_state.get(ag.id, [])
                if not any(a in required for a in actions):
                    return False
            return True

        if when == "only_one_remaining":
            alive = self._alive_agents(state)
            # An agent is "remaining" unless they have property
            # `folded` / `eliminated` / `bankrupt` set truthy.
            exclude = trans.get("exclude_when", ["folded", "eliminated", "bankrupt"])
            remaining = [
                a for a in alive
                if not any(bool(a.get(p)) for p in exclude)
            ]
            return len(remaining) == 1

        if when == "all_remaining_called_or_all_in":
            alive = self._alive_agents(state)
            for a in alive:
                if bool(a.get("folded")) or bool(a.get("eliminated")):
                    continue
                acts = set(self._actions_in_state.get(a.id, []))
                if not (acts & {"call", "check", "all_in", "raise"}):
                    return False
            return True

        if when == "event_emitted":
            wanted = trans.get("event_type")
            if not wanted:
                return False
            return wanted in self._events_in_state

        if when == "entity_property_eq":
            entity_id = trans.get("entity_id")
            prop = trans.get("property")
            value = trans.get("value")
            if not entity_id or not prop:
                return False
            ent = state.get_entity(entity_id) if hasattr(state, "get_entity") else None
            if ent is None:
                return False
            try:
                return ent.get(prop) == value
            except Exception:
                return False

        logger.warning(f"state_machine[{self._id}]: unknown predicate {when!r}")
        return False

    @staticmethod
    def _cmp_int(expr: str, lhs_name: str, lhs_val: int) -> bool:
        """Parse 'rounds_in_state >= 3' style. Supports >= > <= < == !=."""
        tail = expr[len(lhs_name):].strip()
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if tail.startswith(op):
                try:
                    rhs = int(tail[len(op):].strip())
                except ValueError:
                    return False
                if op == ">=": return lhs_val >= rhs
                if op == ">":  return lhs_val > rhs
                if op == "<=": return lhs_val <= rhs
                if op == "<":  return lhs_val < rhs
                if op == "==": return lhs_val == rhs
                if op == "!=": return lhs_val != rhs
        return False

    def _alive_agents(self, state: Any) -> List[Any]:
        agents = list(state.get_agent_entities()) if hasattr(state, "get_agent_entities") else []
        return [a for a in agents if getattr(a, "alive", True)]

    # ────────────────────────────────────────────────────────────── #
    # Event watcher — call this hook from engine when a custom event
    # is emitted so `event_emitted` predicate can wire up.
    # ────────────────────────────────────────────────────────────── #

    def observe_event(self, event_type: str) -> None:
        self._events_in_state.add(event_type)

    # ────────────────────────────────────────────────────────────── #
    # Perception
    # ────────────────────────────────────────────────────────────── #

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        spec = self._state_spec()
        return {
            "machine_id": self._id,
            "current_state": self._current_state,
            "rounds_in_state": self._rounds_in_state,
            "actions_allowed": list(spec.get("actions_allowed") or []),
            "your_actions_this_state": list(self._actions_in_state.get(entity_id, [])),
            "terminal": self.is_terminal,
        }

    # ────────────────────────────────────────────────────────────── #
    # Serialisation
    # ────────────────────────────────────────────────────────────── #

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["state"] = {
            "current_state": self._current_state,
            "state_entered_round": self._state_entered_round,
            "rounds_in_state": self._rounds_in_state,
            "actions_in_state": {k: list(v) for k, v in self._actions_in_state.items()},
            "events_in_state": list(self._events_in_state),
            "initialised": self._initialised,
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseStateMachineModule":
        mod = cls(name=data.get("name", "state_machine"), params=data.get("params", {}))
        s = data.get("state", {})
        mod._current_state = s.get("current_state", mod._initial_state)
        mod._state_entered_round = int(s.get("state_entered_round") or 0)
        mod._rounds_in_state = int(s.get("rounds_in_state") or 0)
        mod._actions_in_state = {k: list(v) for k, v in (s.get("actions_in_state") or {}).items()}
        mod._events_in_state = set(s.get("events_in_state") or [])
        mod._initialised = bool(s.get("initialised", False))
        return mod


__all__ = ["PhaseStateMachineModule"]
