"""TradeModule — declarative player-to-player negotiation (Phase 3).

Generalises the propose/counter/accept/reject flow proven in
Monopoly into a kernel module that any env can opt into via schema.
Replaces ~400 lines of bespoke trade code in monopoly (and is ready
for any future game with negotiation — dispute settlements, faction
deals, resource swaps).

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

    domain_modules:
      - name: trade
        params:
          asset_types: ["property", "money", "tokens"]
          counter_depth_cap: 6
          pin_proposer_turn: true       # active player can't end_turn
                                        # while their offer is open
          force_recipient_respond: true # recipient's only legal actions
                                        # are accept/reject/counter

  actions:
    - name: propose_trade
    - name: accept_trade
    - name: reject_trade
    - name: counter_trade

═══════════════════════════════════════════════════════════════════════
ASSET TYPES (built-in)
═══════════════════════════════════════════════════════════════════════

  "money"     — numeric property `money` (override via params.money_property)
  "property"  — generic ownership relation; per-game validator
  "tokens"    — entries from `effects.get_tokens(entity)`

The "property" asset uses a per-env validator named in `property_validator`
(reuses the named-rules registry). Default: "owned_only".

═══════════════════════════════════════════════════════════════════════
PUBLIC API
═══════════════════════════════════════════════════════════════════════

  module.propose(proposer, target, offer)  → events
  module.accept(actor)                     → events (executes the swap)
  module.reject(actor)                     → events
  module.counter(actor, counter_offer)     → events
  module.pending_for(entity_id)            → offer | None
  module.outgoing_from(entity_id)          → (recipient_id, offer) | None
  module.is_pinned(entity_id)              → bool  (proposer waiting)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .domain_module import DomainModule

logger = logging.getLogger(__name__)


class TradeModule(DomainModule):
    """Generic propose/counter/accept/reject — Phase 3."""

    def __init__(self, name: str = "trade", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params or {})

        # Config
        self._asset_types: List[str] = list(self._params.get("asset_types", ["money"]) or [])
        self._counter_cap: int = int(self._params.get("counter_depth_cap", 6))
        self._money_property: str = self._params.get("money_property", "money")
        self._pin_proposer: bool = bool(self._params.get("pin_proposer_turn", True))
        self._force_respond: bool = bool(self._params.get("force_recipient_respond", True))
        # Validator name for the "property" asset type — names match the
        # registry below. Defaults to the most permissive (owner only).
        self._property_validator: str = self._params.get(
            "property_validator", "owned_only",
        )
        # Configurable property/relation keys. The mapping below is
        # per-asset-type and tells the executor how to do the swap.
        self._property_owner_property: str = self._params.get(
            "property_owner_property", "owner",
        )

        # Runtime state
        # _pending[recipient_id] = offer dict
        self._pending: Dict[str, Dict[str, Any]] = {}
        # Full log of every proposal / response.
        self._log: List[Dict[str, Any]] = []

    @property
    def description(self) -> str:
        return f"Generic trade module (assets: {', '.join(self._asset_types)})"

    @property
    def custom_actions(self) -> List[str]:
        return ["propose_trade", "accept_trade", "reject_trade", "counter_trade"]

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        return []

    # ────────────────────────────────────────────────────────────── #
    # Inspection helpers used by filter_valid_actions / perception
    # ────────────────────────────────────────────────────────────── #

    def pending_for(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """The incoming offer addressed to `entity_id`, if any."""
        return self._pending.get(entity_id)

    def outgoing_from(self, entity_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """The offer this entity has open against someone else."""
        for recipient, offer in self._pending.items():
            if offer.get("proposer") == entity_id:
                return recipient, offer
        return None

    def is_pinned(self, entity_id: str) -> bool:
        """True if `entity_id` is a proposer with an unanswered offer
        (and pin_proposer_turn is on)."""
        return self._pin_proposer and self.outgoing_from(entity_id) is not None

    # ────────────────────────────────────────────────────────────── #
    # filter_valid_actions — recipient is force-channeled to respond
    # ────────────────────────────────────────────────────────────── #

    def filter_valid_actions(
        self, entity_id: str, valid_actions: List[str], state: Any,
    ) -> List[str]:
        # If a player has an INCOMING offer, only the response actions
        # should be available (force_recipient_respond=True).
        if self._force_respond and entity_id in self._pending:
            allowed = {"accept_trade", "reject_trade", "counter_trade"}
            return [a for a in valid_actions if a in allowed]
        # If a player has an OUTGOING offer and pin_proposer_turn=True,
        # block `end_turn` so they can't roll past the unresolved deal.
        if self._pin_proposer and self.outgoing_from(entity_id) is not None:
            return [a for a in valid_actions if a != "end_turn"]
        return valid_actions

    # ────────────────────────────────────────────────────────────── #
    # Core handlers
    # ────────────────────────────────────────────────────────────── #

    def propose(
        self,
        proposer: Any,
        target: Any,
        offer: Dict[str, Any],
        state: Any,
    ) -> List[Dict[str, Any]]:
        if proposer is None or target is None:
            return [{"type": "trade_failed", "reason": "missing_entity"}]
        if proposer.id == target.id:
            return [{"type": "trade_failed", "reason": "self_trade"}]
        # Validate now so junk offers don't sit in the queue.
        err = self._validate_offer(proposer, target, offer, state)
        if err:
            return [{"type": "trade_failed", "reason": err}]

        normalised = dict(offer)
        normalised["proposer"] = proposer.id
        normalised["target"] = target.id
        normalised.setdefault("counter_depth", 0)
        self._pending[target.id] = normalised
        self._log.append({"outcome": "proposed", **normalised})
        return [{
            "type": "trade_proposed",
            "proposer": proposer.id, "target": target.id,
            "offer": _redact(normalised),
        }]

    def accept(self, actor: Any, state: Any) -> List[Dict[str, Any]]:
        offer = self._pending.pop(actor.id, None) if actor else None
        if not offer:
            return [{"type": "trade_failed", "reason": "no_open_offer"}]
        proposer = state.get_entity(offer["proposer"]) if hasattr(state, "get_entity") else None
        if proposer is None or not proposer.alive:
            return [{"type": "trade_cancelled", "reason": "proposer_gone"}]
        # Re-validate before executing.
        err = self._validate_offer(proposer, actor, offer, state, recheck=True)
        if err:
            return [{"type": "trade_cancelled", "reason": err}]
        executed = self._execute(proposer, actor, offer, state)
        self._log.append({"outcome": "accepted", **offer})
        return [{
            "type": "trade_accepted",
            "proposer": proposer.id, "target": actor.id,
            "offer": _redact(offer),
            "moves": executed,
        }]

    def reject(self, actor: Any, state: Any) -> List[Dict[str, Any]]:
        offer = self._pending.pop(actor.id, None) if actor else None
        if not offer:
            return [{"type": "trade_failed", "reason": "no_open_offer"}]
        self._log.append({"outcome": "rejected", **offer})
        return [{
            "type": "trade_rejected",
            "proposer": offer.get("proposer"), "target": actor.id,
            "offer": _redact(offer),
        }]

    def counter(
        self, actor: Any, counter_offer: Dict[str, Any], state: Any,
    ) -> List[Dict[str, Any]]:
        original = self._pending.pop(actor.id, None) if actor else None
        if not original:
            return [{"type": "trade_failed", "reason": "no_open_offer"}]
        original_proposer = state.get_entity(original["proposer"]) if hasattr(state, "get_entity") else None
        if original_proposer is None or not original_proposer.alive:
            return [{"type": "trade_cancelled", "reason": "proposer_gone"}]
        depth = int(original.get("counter_depth", 0)) + 1
        if depth > self._counter_cap:
            self._log.append({"outcome": "rejected", "reason": "counter_cap", **original})
            return [{
                "type": "trade_rejected",
                "proposer": original["proposer"], "target": actor.id,
                "reason": "counter_cap",
            }]
        # Flip roles: actor is now the proposer; original proposer is the new target.
        flipped = dict(counter_offer)
        flipped["proposer"] = actor.id
        flipped["target"] = original_proposer.id
        flipped["counter_depth"] = depth
        err = self._validate_offer(actor, original_proposer, flipped, state)
        if err:
            # Restore the original offer so they can try again.
            self._pending[actor.id] = original
            return [{"type": "trade_failed", "reason": err}]
        self._pending[original_proposer.id] = flipped
        self._log.append({"outcome": "countered", **flipped})
        return [{
            "type": "trade_countered",
            "proposer": actor.id, "target": original_proposer.id,
            "offer": _redact(flipped), "depth": depth,
        }]

    # ────────────────────────────────────────────────────────────── #
    # Validation + execution
    # ────────────────────────────────────────────────────────────── #

    def _validate_offer(
        self, proposer: Any, target: Any, offer: Dict[str, Any],
        state: Any, recheck: bool = False,
    ) -> Optional[str]:
        # Empty offer
        if not _has_any_assets(offer):
            return "empty_offer"
        # Money
        give_money = int(offer.get("give_money") or 0)
        want_money = int(offer.get("want_money") or 0)
        if give_money > 0 and float(proposer.get(self._money_property, 0) or 0) < give_money:
            return "proposer_short_on_cash"
        if want_money > 0 and float(target.get(self._money_property, 0) or 0) < want_money:
            return "target_short_on_cash"
        # Properties (using the configured validator)
        validator = _PROPERTY_VALIDATORS.get(self._property_validator, _val_owned_only)
        for prop_id in offer.get("give_properties", []) or []:
            ok, why = validator(prop_id, proposer, state, self)
            if not ok:
                return f"give_property:{why}"
        for prop_id in offer.get("want_properties", []) or []:
            ok, why = validator(prop_id, target, state, self)
            if not ok:
                return f"want_property:{why}"
        return None

    def _execute(
        self, proposer: Any, target: Any, offer: Dict[str, Any], state: Any,
    ) -> List[Dict[str, Any]]:
        moves: List[Dict[str, Any]] = []
        # Properties: change the owner relation/property.
        for prop_id in offer.get("give_properties", []) or []:
            self._set_property_owner(prop_id, target.id, state)
            moves.append({"property": prop_id, "from": proposer.id, "to": target.id})
        for prop_id in offer.get("want_properties", []) or []:
            self._set_property_owner(prop_id, proposer.id, state)
            moves.append({"property": prop_id, "from": target.id, "to": proposer.id})
        # Money
        gm = int(offer.get("give_money") or 0)
        wm = int(offer.get("want_money") or 0)
        if gm > 0:
            proposer.set(self._money_property, float(proposer.get(self._money_property, 0) or 0) - gm)
            target.set(self._money_property, float(target.get(self._money_property, 0) or 0) + gm)
            moves.append({"money": gm, "from": proposer.id, "to": target.id})
        if wm > 0:
            target.set(self._money_property, float(target.get(self._money_property, 0) or 0) - wm)
            proposer.set(self._money_property, float(proposer.get(self._money_property, 0) or 0) + wm)
            moves.append({"money": wm, "from": target.id, "to": proposer.id})
        return moves

    def _set_property_owner(self, prop_id: Any, new_owner_id: str, state: Any) -> None:
        """Set the owner of a 'property' asset. Default implementation
        looks up an Entity by prop_id (numeric or string) and sets its
        `owner` property. Subclasses / games with different ownership
        models can override the property_owner_property param."""
        ent = state.get_entity(str(prop_id)) if hasattr(state, "get_entity") else None
        if ent is not None and hasattr(ent, "set"):
            ent.set(self._property_owner_property, new_owner_id)

    # ────────────────────────────────────────────────────────────── #
    # Perception
    # ────────────────────────────────────────────────────────────── #

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        incoming = self.pending_for(entity_id)
        if incoming:
            data["incoming_offer"] = _redact(incoming)
        outgoing = self.outgoing_from(entity_id)
        if outgoing:
            recipient_id, offer = outgoing
            data["outgoing_offer"] = {"to": recipient_id, **_redact(offer)}
            data["turn_pinned"] = self._pin_proposer
        if self._log:
            data["recent_trades"] = [_redact(e) for e in self._log[-8:]]
        return data

    # ────────────────────────────────────────────────────────────── #
    # Serialisation
    # ────────────────────────────────────────────────────────────── #

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["state"] = {
            "pending": {k: dict(v) for k, v in self._pending.items()},
            "log": list(self._log),
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "TradeModule":
        mod = cls(name=data.get("name", "trade"), params=data.get("params", {}))
        s = data.get("state", {})
        mod._pending = {k: dict(v) for k, v in (s.get("pending") or {}).items()}
        mod._log = list(s.get("log") or [])
        return mod


# ───────────────────────────────────────────────────────────────────── #
# Helpers
# ───────────────────────────────────────────────────────────────────── #

def _has_any_assets(offer: Dict[str, Any]) -> bool:
    gp = offer.get("give_properties") or []
    wp = offer.get("want_properties") or []
    gm = int(offer.get("give_money") or 0)
    wm = int(offer.get("want_money") or 0)
    return bool(gp or wp or gm or wm)


def _redact(offer: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of the offer safe to show in perception/events."""
    keys = ("proposer", "target", "give_properties", "give_money",
            "want_properties", "want_money", "note", "counter_depth")
    return {k: offer.get(k) for k in keys if k in offer}


# ── Property-validator registry ──
# Each validator is `(prop_id, supposed_owner, state, trade_mod) -> (ok, why)`

def _val_owned_only(prop_id, owner, state, mod) -> Tuple[bool, str]:
    ent = state.get_entity(str(prop_id)) if hasattr(state, "get_entity") else None
    if ent is None:
        return False, "not_found"
    if ent.get(mod._property_owner_property) != owner.id:
        return False, "not_owner"
    return True, ""


def _val_owned_and_unimproved(prop_id, owner, state, mod) -> Tuple[bool, str]:
    ok, why = _val_owned_only(prop_id, owner, state, mod)
    if not ok:
        return ok, why
    ent = state.get_entity(str(prop_id))
    if ent.get("houses", 0) > 0 or ent.get("hotel", False):
        return False, "has_buildings"
    if ent.get("mortgaged", False):
        return False, "mortgaged"
    return True, ""


_PROPERTY_VALIDATORS = {
    "owned_only": _val_owned_only,
    "owned_and_unimproved": _val_owned_and_unimproved,
}


__all__ = ["TradeModule"]
