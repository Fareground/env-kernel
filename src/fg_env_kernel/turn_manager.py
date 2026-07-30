"""TurnManagerModule — declarative active-player rotation.

Replaces the ~80 lines of turn-rotation code that every custom
DomainModule used to ship. Configured entirely via schema:

  domain_modules:
    - name: turn_manager
      params:
        active_role: "MonopolyPlayer"        # entity_type that takes turns
        strategy: "round_robin"               # | "random_per_round" | "property_based"
        initiative_property: "speed"          # for property_based
        skip_when: ["bankrupt", "eliminated", "folded"]  # property names that are truthy
        extra_turn_event: "rolled_doubles"    # if this event_type was last, current player goes again
        max_extra_turns: 3                    # ... up to this many consecutive
        advance_on_actions: ["end_turn"]      # which actions move the pointer
        actions_per_turn: 1                   # how many actions before auto-advance

The module is intentionally minimal — it exposes:
  - the **active player id** via perception (`turn_manager.active_player`)
  - `filter_valid_actions`: returns [] for non-active players (so the
    engine's action listing is correct without each game re-implementing
    "it's not your turn" logic)
  - `post_resolution`: advances the pointer when an `advance_on_actions`
    action resolves; resets the doubles counter on others; applies the
    max_extra_turns cap.

Games that need MORE complex turn rules (in_jail jail_turns counter,
poker fold-skip variants) can still subclass; but ~80% of envs are
satisfied by the declarative form.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .domain_module import DomainModule

logger = logging.getLogger(__name__)


class TurnManagerModule(DomainModule):
    """Generic, schema-configured turn manager."""

    def __init__(
        self,
        name: str = "turn_manager",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, params=params or {})

        # Config
        self._active_role: Optional[str] = self._params.get("active_role")
        self._strategy: str = self._params.get("strategy", "round_robin")
        self._initiative_property: str = self._params.get("initiative_property", "speed")
        self._skip_when: List[str] = list(self._params.get("skip_when", []) or [])
        self._extra_turn_event: Optional[str] = self._params.get("extra_turn_event")
        self._max_extra_turns: int = int(self._params.get("max_extra_turns", 0))
        self._advance_on_actions: List[str] = list(
            self._params.get("advance_on_actions", ["end_turn"]) or []
        )
        self._actions_per_turn: int = int(self._params.get("actions_per_turn", 1))

        # RNG for random_per_round ordering — pinned to the sim seed by the
        # engine via reseed(). Defaults to an unseeded RNG only if never
        # injected (e.g. used standalone in a test).
        import random as _random
        self._rng = _random.Random()

        # Runtime state
        self._turn_order: List[str] = []
        self._turn_index: int = 0
        self._actions_taken_this_turn: int = 0
        self._extra_turns_taken: int = 0
        # The last *advance_on_actions* action's full event payload — used by
        # `extra_turn_event` lookahead. We capture this from post_resolution.
        self._last_advance_event: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #

    def reseed(self, rng) -> None:
        """Pin random_per_round turn shuffling to the sim seed (engine calls
        this at startup, same as the deck/hand modules)."""
        self._rng = rng

    @property
    def description(self) -> str:
        return "Declarative turn-order manager (active player, skip predicate, extra-turn rules)."

    @property
    def custom_actions(self) -> List[str]:
        return []

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _seed(self, state: Any) -> None:
        agents = self._eligible_agents(state)
        if not agents:
            return
        existing = set(self._turn_order)
        # Append agents we haven't seen yet (arena matches inject the
        # user's agent AFTER initial setup — without this, the late
        # arrival is permanently shut out of the rotation even though
        # they're free to act).
        new_agents = [a for a in agents if a.id not in existing]
        if not self._turn_order:
            # First seed — apply strategy to the full set.
            ordered = list(agents)
            if self._strategy == "random_per_round":
                self._rng.shuffle(ordered)
            elif self._strategy == "property_based":
                ordered.sort(
                    key=lambda a: float(a.get(self._initiative_property, 0) or 0),
                    reverse=True,
                )
            self._turn_order = [a.id for a in ordered]
            self._turn_index = 0
        elif new_agents:
            # Late arrivals — append to the rotation in order they appeared.
            self._turn_order.extend(a.id for a in new_agents)

    def _eligible_agents(self, state: Any) -> List[Any]:
        if hasattr(state, "get_agent_entities"):
            agents = list(state.get_agent_entities())
        else:
            agents = []
        if self._active_role:
            agents = [a for a in agents if getattr(a, "entity_type", None) == self._active_role]
        return agents

    # ------------------------------------------------------------------ #
    # Active player & skip logic
    # ------------------------------------------------------------------ #

    def _is_skipped(self, entity_id: str, state: Any) -> bool:
        if not self._skip_when:
            return False
        ent = state.get_entity(entity_id) if hasattr(state, "get_entity") else None
        if ent is None:
            return True  # entity gone → skip
        for prop in self._skip_when:
            try:
                if bool(ent.get(prop)):
                    return True
            except Exception:
                continue
        return False

    def active_player_id(self, state: Any) -> Optional[str]:
        """Return the current active player WITHOUT mutating turn state.

        Skipped players are bypassed by a local scan; the persistent
        `_turn_index` only advances when `_advance_pointer` is called
        from the post-action commit path. Mutating from a read path
        previously caused turn-order drift on repeated reads."""
        if not self._turn_order:
            return None
        n = len(self._turn_order)
        cursor = self._turn_index
        for _ in range(n):
            pid = self._turn_order[cursor]
            if not self._is_skipped(pid, state):
                return pid
            cursor = (cursor + 1) % n
        return None

    def commit_skipped(self, state: Any) -> None:
        """Permanently advance past any currently-skipped players. Call
        this from tick/post-action paths when you want the persistent
        index to skip over players who can't act this round."""
        if not self._turn_order:
            return
        n = len(self._turn_order)
        for _ in range(n):
            pid = self._turn_order[self._turn_index]
            if not self._is_skipped(pid, state):
                return
            self._turn_index = (self._turn_index + 1) % n

    # ------------------------------------------------------------------ #
    # Tick — seed + emit active player on round start
    # ------------------------------------------------------------------ #

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        self._seed(state)
        # tick is the canonical advance point — commit skip-cursor here.
        self.commit_skipped(state)
        active = self.active_player_id(state)
        if active is None:
            return []
        return [{
            "type": "turn_manager_active",
            "round": round_number,
            "active_player": active,
            "turn_index": self._turn_index,
        }]

    # ------------------------------------------------------------------ #
    # filter_valid_actions — non-active players have no actions
    # ------------------------------------------------------------------ #

    def filter_valid_actions(
        self,
        entity_id: str,
        valid_actions: List[str],
        state: Any,
    ) -> List[str]:
        active = self.active_player_id(state)
        if active is None:
            return valid_actions
        if entity_id != active:
            # Non-active player. Keep only "non-turn-consuming" actions
            # (speech/communication) — concrete games may relax this via
            # subclassing; the default is conservative.
            return []
        return valid_actions

    # ------------------------------------------------------------------ #
    # post_resolution — advance pointer, handle extra turns
    # ------------------------------------------------------------------ #

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

        events: List[Dict[str, Any]] = []

        # Did this action emit an "extra-turn" event? Check the most
        # recent event log entries from the engine if available.
        gets_extra_turn = False
        if self._extra_turn_event and hasattr(state, "event_log"):
            try:
                recent = state.event_log.get_round(state.temporal.current_round)
                # Walk backwards — most recent event first.
                for ev in reversed(recent[-12:]):
                    if getattr(ev, "event_type", None) == self._extra_turn_event:
                        gets_extra_turn = True
                        self._last_advance_event = (
                            ev.data if hasattr(ev, "data") else {}
                        )
                        break
            except Exception:
                pass

        if action_name in self._advance_on_actions:
            if gets_extra_turn and self._extra_turns_taken < self._max_extra_turns:
                self._extra_turns_taken += 1
                self._actions_taken_this_turn = 0
                events.append({
                    "type": "turn_manager_extra_turn",
                    "actor": actor_id,
                    "extra_turn_count": self._extra_turns_taken,
                    "max": self._max_extra_turns,
                })
            else:
                self._advance_pointer(state)
                events.append({
                    "type": "turn_manager_advanced",
                    "actor": actor_id,
                    "next_active": self.active_player_id(state),
                })
            return events

        # Non-advance action: increment per-turn counter; auto-advance
        # when we've hit `actions_per_turn`.
        self._actions_taken_this_turn += 1
        if (
            self._actions_per_turn > 0
            and self._actions_taken_this_turn >= self._actions_per_turn
            and not self._advance_on_actions
        ):
            self._advance_pointer(state)
            events.append({
                "type": "turn_manager_advanced",
                "actor": actor_id,
                "reason": "actions_per_turn_reached",
                "next_active": self.active_player_id(state),
            })
        return events

    def _advance_pointer(self, state: Any) -> None:
        if not self._turn_order:
            return
        n = len(self._turn_order)
        self._turn_index = (self._turn_index + 1) % n
        self._actions_taken_this_turn = 0
        self._extra_turns_taken = 0

    # ------------------------------------------------------------------ #
    # Perception — surface whose turn it is
    # ------------------------------------------------------------------ #

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        active = self.active_player_id(state)
        is_active = entity_id == active
        return {
            "active_player": active,
            "is_your_turn": is_active,
            "turn_index": self._turn_index,
            "turn_order": list(self._turn_order),
            "extra_turns_taken": self._extra_turns_taken,
        }

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["state"] = {
            "turn_order": list(self._turn_order),
            "turn_index": self._turn_index,
            "actions_taken_this_turn": self._actions_taken_this_turn,
            "extra_turns_taken": self._extra_turns_taken,
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "TurnManagerModule":
        mod = cls(name=data.get("name", "turn_manager"), params=data.get("params", {}))
        s = data.get("state", {})
        mod._turn_order = list(s.get("turn_order") or [])
        mod._turn_index = int(s.get("turn_index") or 0)
        mod._actions_taken_this_turn = int(s.get("actions_taken_this_turn") or 0)
        mod._extra_turns_taken = int(s.get("extra_turns_taken") or 0)
        return mod


__all__ = ["TurnManagerModule"]
