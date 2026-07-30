"""SlotsModule — claimable action slots (worker placement, Tier 5b).

The defining mechanic of euro-style worker-placement games (Agricola,
Lords of Waterdeep, Stone Age, Caverna). Each "slot" is a board-level
action position that an agent CLAIMS by sending a worker there. While
claimed it's unavailable to others. At round end (or phase reset)
workers come home and slots reopen.

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

    domain_modules:
      - name: slots
        params:
          id: "actions"
          reset_on: "round_end"               # or "manual" or "phase:cleanup"
          slots:
            - { id: "gather_wood",  capacity: 1,
                reward: [{ operation: "add", target: "$claimer", field: "wood", value: 2 }] }
            - { id: "gather_food",  capacity: 2,
                reward: [{ operation: "add", target: "$claimer", field: "food", value: 3 }] }
            - { id: "build_house",  capacity: 1, cost: { wood: 4, food: 2 },
                reward: [{ operation: "grant_token", target: "$claimer", field: "house" }] }

  actions (auto-added by the module):
    - claim_slot(slot_id)   — sends a worker to the named slot; fires reward
    - release_slot(slot_id) — recall a worker (rare; usually automatic on reset)

═══════════════════════════════════════════════════════════════════════
PUBLIC API
═══════════════════════════════════════════════════════════════════════

  available_slots()            → list of (slot_id, remaining_capacity)
  is_claimed_by(slot_id, eid)  → bool
  claim(slot_id, entity_id)    → (ok, reward_effects, error?)
  release(slot_id, entity_id)  → ok
  reset()                      → release all
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .domain_module import DomainModule

logger = logging.getLogger(__name__)


class SlotsModule(DomainModule):
    """Claimable action slots (worker placement)."""

    def __init__(self, name: str = "slots", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params or {})
        self._id: str = self._params.get("id", "slots")
        self._reset_on: str = self._params.get("reset_on", "round_end")
        raw_slots = self._params.get("slots") or []
        # slot_id -> {capacity, reward, cost, claimers: set}
        self._slots: Dict[str, Dict[str, Any]] = {}
        for s in raw_slots:
            if not isinstance(s, dict) or "id" not in s:
                continue
            sid = str(s["id"])
            self._slots[sid] = {
                "id": sid,
                "capacity": int(s.get("capacity", 1)),
                "reward": list(s.get("reward") or []),
                "cost": dict(s.get("cost") or {}),
                "claimers": [],   # entity ids currently here
                "description": s.get("description", ""),
            }
        self._last_round_reset: int = -1

    @property
    def description(self) -> str:
        return f"Slots '{self._id}' ({len(self._slots)} slots)"

    @property
    def custom_actions(self) -> List[str]:
        return ["claim_slot", "release_slot"]

    @property
    def id(self) -> str:
        return self._id

    # ── Lifecycle ──

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        # Reset at round start when reset_on == round_end (we reset on the
        # FOLLOWING round's tick so all post_resolution effects from the
        # previous round can still see who claimed what).
        if self._reset_on == "round_end" and round_number > self._last_round_reset:
            self._last_round_reset = round_number
            if round_number > 0:
                return self._reset_all()
        return []

    def _reset_all(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for sid, slot in self._slots.items():
            if slot["claimers"]:
                events.append({
                    "type": "slots_reset",
                    "slot_id": sid,
                    "released": list(slot["claimers"]),
                })
                slot["claimers"] = []
        return events

    # ── Inspection ──

    def available_slots(self) -> List[Tuple[str, int]]:
        return [
            (sid, slot["capacity"] - len(slot["claimers"]))
            for sid, slot in self._slots.items()
            if len(slot["claimers"]) < slot["capacity"]
        ]

    def is_claimed_by(self, slot_id: str, entity_id: str) -> bool:
        slot = self._slots.get(slot_id)
        return bool(slot and entity_id in slot["claimers"])

    def claim(self, slot_id: str, entity_id: str) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        slot = self._slots.get(slot_id)
        if not slot:
            return False, [], "no_such_slot"
        if entity_id in slot["claimers"]:
            return False, [], "already_claimed_by_you"
        if len(slot["claimers"]) >= slot["capacity"]:
            return False, [], "slot_full"
        slot["claimers"].append(entity_id)
        return True, list(slot["reward"]), None

    def release(self, slot_id: str, entity_id: str) -> bool:
        slot = self._slots.get(slot_id)
        if not slot or entity_id not in slot["claimers"]:
            return False
        slot["claimers"].remove(entity_id)
        return True

    # ── Filter actions: only allow claim_slot on AVAILABLE slots ──

    def filter_valid_actions(
        self, entity_id: str, valid_actions: List[str], state: Any,
    ) -> List[str]:
        # Module-level filter doesn't see action parameters; we permit
        # `claim_slot` to always show as an option. The PER-SLOT
        # availability is conveyed via perception (`available_slots`).
        return valid_actions

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        # Surface current slot state so the LLM picks valid slots.
        slot_view = []
        for sid, slot in self._slots.items():
            slot_view.append({
                "id": sid,
                "capacity": slot["capacity"],
                "free": slot["capacity"] - len(slot["claimers"]),
                "claimed_by": list(slot["claimers"]),
                "cost": slot["cost"],
                "description": slot["description"],
                "you_have_claimed": entity_id in slot["claimers"],
            })
        return {
            "slots_id": self._id,
            "slots": slot_view,
            "available_count": sum(1 for _, n in self.available_slots()),
        }

    # ── Serialisation ──

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["state"] = {
            sid: {"claimers": list(slot["claimers"])}
            for sid, slot in self._slots.items()
        }
        base["state"]["_last_round_reset"] = self._last_round_reset
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "SlotsModule":
        mod = cls(name=data.get("name", "slots"), params=data.get("params", {}))
        for sid, snap in (data.get("state") or {}).items():
            if sid == "_last_round_reset":
                mod._last_round_reset = int(snap or -1)
                continue
            if sid in mod._slots and isinstance(snap, dict):
                mod._slots[sid]["claimers"] = list(snap.get("claimers", []))
        return mod


__all__ = ["SlotsModule"]
