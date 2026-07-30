"""HandModule — per-player card hand (Tier 5a).

`DeckModule` is a SHARED pile (Chance / Community Chest / event deck).
`HandModule` is PER-PLAYER state: each agent has their own hand plus
optionally a personal draw pile + discard pile. The pair powers every
card game where players collect/play/discard their own cards:
trick-takers, deck-builders, hand-builders, drafting games.

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

    domain_modules:
      - name: hand
        params:
          id: "main"
          card_catalog:                    # all cards that EXIST in the game
            - { id: "knight",   name: "Knight",   cost: 4, effect: [...] }
            - { id: "settler",  name: "Settler",  cost: 2, effect: [...] }
            - { id: "monopoly", name: "Monopoly", cost: 5, effect: [...] }
          starting_hand: 5                  # cards dealt to each agent at setup
          starting_deck:                    # optional per-player draw pile
            - knight × 4
            - settler × 4
          max_hand_size: 10

  effects (in actions):
    - { operation: "draw_to_hand", target: "actor", value: 1 }
    - { operation: "discard_from_hand", target: "actor", field: "knight" }
    - { operation: "play_card",  target: "actor", field: "$params.card_id" }
    - { operation: "pass_card_to", target: "target", field: "$params.card_id" }

═══════════════════════════════════════════════════════════════════════
PUBLIC API
═══════════════════════════════════════════════════════════════════════

  hand_of(entity_id)              → list of card dicts in hand
  draw_pile_of(entity_id)         → list of card dicts in personal deck
  discard_pile_of(entity_id)      → list of card dicts in discard
  draw_to_hand(entity_id, n=1)    → n cards moved deck → hand (reshuffle if empty)
  discard(entity_id, card_id)     → move card hand → discard
  play(entity_id, card_id)        → move card hand → play area; returns card
  pass_to(from_id, to_id, card_id)→ transfer between hands
  catalog_card(card_id)           → lookup by id from card_catalog
"""
from __future__ import annotations

import logging
import random as _random
from typing import Any, Dict, List, Optional, Tuple

from .domain_module import DomainModule

logger = logging.getLogger(__name__)


class HandModule(DomainModule):
    """Per-player hand + draw + discard piles."""

    def __init__(self, name: str = "hand", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params or {})

        self._id: str = self._params.get("id", "hand")
        # Card catalog — `id` -> full card dict (name, cost, effect, …)
        catalog = self._params.get("card_catalog") or []
        self._catalog: Dict[str, Dict[str, Any]] = {
            str(c.get("id", i)): dict(c) for i, c in enumerate(catalog)
            if isinstance(c, dict)
        }
        self._starting_hand: int = int(self._params.get("starting_hand", 0))
        # `starting_deck`: list of card ids OR list of {card_id, count} OR
        # list of "card_id × N" strings.
        self._starting_deck_spec: List[Any] = list(self._params.get("starting_deck") or [])
        self._max_hand_size: Optional[int] = (
            int(self._params["max_hand_size"]) if "max_hand_size" in self._params else None
        )
        seed = self._params.get("seed")
        self._rng = _random.Random(seed) if seed is not None else _random.Random()
        self._initialized: bool = False

        # Per-entity state — populated on first tick
        self._hands: Dict[str, List[Dict[str, Any]]] = {}
        self._draw_piles: Dict[str, List[Dict[str, Any]]] = {}
        self._discards: Dict[str, List[Dict[str, Any]]] = {}

    # ── Contract ──

    @property
    def description(self) -> str:
        return f"Per-player hand '{self._id}' (catalog: {len(self._catalog)} cards)"

    @property
    def id(self) -> str:
        return self._id

    def reseed(self, rng) -> None:
        """Pin per-player deck shuffles to the sim seed (unless the author
        set an explicit ``seed``). Safe to call before the first tick — the
        starting decks are shuffled lazily in ``_initialize_for_agents``, so
        simply swapping the RNG here makes the deal reproducible."""
        if self._params.get("seed") is not None or self._initialized:
            return
        self._rng = rng

    # ── Setup ──

    def _expand_starting_deck(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry in self._starting_deck_spec:
            if isinstance(entry, str):
                # "card_id × N" or just "card_id"
                if "×" in entry:
                    cid, n = entry.split("×", 1)
                elif "x" in entry and entry.count("x") == 1:
                    cid, n = entry.split("x", 1)
                else:
                    cid, n = entry, "1"
                cid = cid.strip()
                try:
                    count = int(n.strip())
                except ValueError:
                    count = 1
                if cid in self._catalog:
                    for _ in range(count):
                        out.append(dict(self._catalog[cid]))
            elif isinstance(entry, dict):
                cid = str(entry.get("card_id") or entry.get("id"))
                count = int(entry.get("count", 1))
                if cid in self._catalog:
                    for _ in range(count):
                        out.append(dict(self._catalog[cid]))
        return out

    def _initialize_for_agents(self, state: Any) -> List[Dict[str, Any]]:
        if self._initialized:
            return []
        self._initialized = True
        events: List[Dict[str, Any]] = []
        for ent in state.get_agent_entities():
            # Personal draw pile
            deck = self._expand_starting_deck()
            self._rng.shuffle(deck)
            self._draw_piles[ent.id] = deck
            self._hands[ent.id] = []
            self._discards[ent.id] = []
            # Deal starting hand
            drawn = self._draw_n(ent.id, self._starting_hand)
            if drawn:
                events.append({
                    "type": "hand_dealt",
                    "entity_id": ent.id,
                    "count": len(drawn),
                    "hand_size": len(self._hands[ent.id]),
                })
        return events

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        if not self._initialized:
            return self._initialize_for_agents(state)
        return []

    # ── Public API ──

    def catalog_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        return self._catalog.get(str(card_id))

    def hand_of(self, entity_id: str) -> List[Dict[str, Any]]:
        return list(self._hands.get(entity_id, []))

    def draw_pile_of(self, entity_id: str) -> List[Dict[str, Any]]:
        return list(self._draw_piles.get(entity_id, []))

    def discard_pile_of(self, entity_id: str) -> List[Dict[str, Any]]:
        return list(self._discards.get(entity_id, []))

    def _draw_n(self, entity_id: str, n: int) -> List[Dict[str, Any]]:
        drawn: List[Dict[str, Any]] = []
        hand = self._hands.setdefault(entity_id, [])
        deck = self._draw_piles.setdefault(entity_id, [])
        for _ in range(int(n)):
            if not deck:
                # Reshuffle discard into deck
                discard = self._discards.get(entity_id, [])
                if not discard:
                    break
                deck = list(discard)
                self._rng.shuffle(deck)
                self._draw_piles[entity_id] = deck
                self._discards[entity_id] = []
            card = deck.pop()
            if (self._max_hand_size is not None
                    and len(hand) >= self._max_hand_size):
                # Hand full — discard immediately
                self._discards.setdefault(entity_id, []).append(card)
                continue
            hand.append(card)
            drawn.append(card)
        return drawn

    def draw_to_hand(self, entity_id: str, n: int = 1) -> List[Dict[str, Any]]:
        return self._draw_n(entity_id, n)

    def discard(self, entity_id: str, card_id: str) -> Optional[Dict[str, Any]]:
        hand = self._hands.get(entity_id) or []
        for i, c in enumerate(hand):
            if str(c.get("id")) == str(card_id):
                removed = hand.pop(i)
                self._discards.setdefault(entity_id, []).append(removed)
                return removed
        return None

    def play(self, entity_id: str, card_id: str) -> Optional[Dict[str, Any]]:
        # Play moves the card to discard (same as discard, but the
        # caller fires the card's effect via DRAW_FROM_DECK pattern).
        return self.discard(entity_id, card_id)

    def pass_to(self, from_id: str, to_id: str, card_id: str) -> Optional[Dict[str, Any]]:
        hand = self._hands.get(from_id) or []
        for i, c in enumerate(hand):
            if str(c.get("id")) == str(card_id):
                removed = hand.pop(i)
                self._hands.setdefault(to_id, []).append(removed)
                return removed
        return None

    # ── Perception ──

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        # Only show the observer's OWN hand. Other players' hand SIZES
        # are public (standard card-game convention); content is private.
        hand = self._hands.get(entity_id, [])
        public_view: Dict[str, int] = {}
        for eid, h in self._hands.items():
            if eid != entity_id:
                public_view[eid] = len(h)
        return {
            "hand_id": self._id,
            "your_hand": [
                {"id": c.get("id"), "name": c.get("name"),
                 "cost": c.get("cost"), "description": c.get("description", "")}
                for c in hand
            ],
            "your_draw_pile_size": len(self._draw_piles.get(entity_id, [])),
            "your_discard_size": len(self._discards.get(entity_id, [])),
            "other_hand_sizes": public_view,
        }

    # ── Serialisation ──

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["state"] = {
            "initialized": self._initialized,
            "hands": {k: list(v) for k, v in self._hands.items()},
            "draw_piles": {k: list(v) for k, v in self._draw_piles.items()},
            "discards": {k: list(v) for k, v in self._discards.items()},
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "HandModule":
        mod = cls(name=data.get("name", "hand"), params=data.get("params", {}))
        s = data.get("state", {})
        mod._initialized = bool(s.get("initialized", False))
        mod._hands = {k: list(v) for k, v in (s.get("hands") or {}).items()}
        mod._draw_piles = {k: list(v) for k, v in (s.get("draw_piles") or {}).items()}
        mod._discards = {k: list(v) for k, v in (s.get("discards") or {}).items()}
        return mod


__all__ = ["HandModule"]
