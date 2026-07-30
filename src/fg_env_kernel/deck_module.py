"""DeckModule — declarative card decks (Phase 3).

A generic shuffled-deck primitive: cards have a `text` + `effect` list
(EffectDSL), drawing pops a card and applies its effects via the same
engine machinery that runs `effects_on_success`. Replaces Chance/
Community-Chest decks in monopoly, poker hole cards, dominoes tiles,
event-card decks in any future env.

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

    domain_modules:
      - name: deck
        params:
          id: "chance"                       # unique deck identifier
          shuffle: true
          reshuffle_on_empty: true
          cards:
            - text: "Advance to GO. Collect $200."
              effect:
                - { operation: set, target: actor, field: position, value: 0 }
                - { operation: add, target: actor, field: money,    value: 200 }
            - text: "Pay $50 each to every other player."
              effect:
                - { operation: subtract, target: actor, field: money, value: 50 }
            - text: "Get out of jail free."
              keep_until_used: true          # card stays with the player
              effect:
                - { operation: grant_token, target: actor, field: jail_free, value: 1 }

  actions:
    - name: draw_chance
      effects_on_success:
        - { operation: draw_from_deck, field: chance }   # → fires this deck

Public API:

    deck.draw(actor)            → { card, applied_changes }
    deck.peek()                 → next card or None
    deck.shuffle()              → re-randomise
    deck.size()                 → cards remaining
    deck.held_by(entity)        → list of cards this entity holds
    deck.use_held(entity, idx)  → consume a held card (apply its effects)
"""
from __future__ import annotations

import logging
import random as _random
from typing import Any, Dict, List, Optional

from .domain_module import DomainModule

logger = logging.getLogger(__name__)


class DeckModule(DomainModule):
    """Generic shuffled-deck primitive — Phase 3."""

    def __init__(self, name: str = "deck", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params or {})

        self._id: str = self._params.get("id", "deck")
        self._reshuffle_on_empty: bool = bool(self._params.get("reshuffle_on_empty", True))
        # `consume_on_draw=True`: drawn cards are GONE (not cycled). Used
        # for poker hole-card dealing, ephemeral event decks, etc.
        # Default False — monopoly Chance / CC style cycling.
        self._consume_on_draw: bool = bool(self._params.get("consume_on_draw", False))

        seed = self._params.get("seed")
        self._rng = _random.Random(seed) if seed is not None else _random.Random()

        # Deep-copy the configured cards so runtime shuffles/discards
        # don't mutate the schema config.
        cards = [dict(c) for c in (self._params.get("cards") or [])]
        # Assign stable indices for reproducible event logs.
        for i, c in enumerate(cards):
            c.setdefault("idx", i)
        self._original_cards: List[Dict[str, Any]] = list(cards)
        self._draw_pile: List[Dict[str, Any]] = list(cards)
        if self._params.get("shuffle", True):
            self._rng.shuffle(self._draw_pile)

        # Cards held outside the deck (kept by entities). Card returns
        # to the bottom of the pile when consumed.
        self._held: Dict[str, List[Dict[str, Any]]] = {}

    # ── Contract ──

    @property
    def description(self) -> str:
        return f"Deck '{self._id}' with {len(self._original_cards)} cards"

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        return []

    # ────────────────────────────────────────────────────────────── #
    # Deck operations
    # ────────────────────────────────────────────────────────────── #

    @property
    def id(self) -> str:
        return self._id

    def reseed(self, rng) -> None:
        """Pin the deck's RNG to the simulation seed for reproducibility.

        Called by the engine at startup. If the world author pinned an
        explicit ``seed`` in the deck params, we honor that instead. Because
        the initial shuffle already happened in ``__init__`` (before the
        engine existed), we rebuild the draw pile from canonical order and
        re-shuffle deterministically.
        """
        if self._params.get("seed") is not None:
            return
        self._rng = rng
        if self._params.get("shuffle", True) and not self._held:
            self._draw_pile = [dict(c) for c in self._original_cards]
            self._rng.shuffle(self._draw_pile)

    def shuffle(self) -> None:
        self._rng.shuffle(self._draw_pile)

    def size(self) -> int:
        return len(self._draw_pile)

    def peek(self) -> Optional[Dict[str, Any]]:
        return self._draw_pile[-1] if self._draw_pile else None

    def draw(self, actor: Any) -> Optional[Dict[str, Any]]:
        """Pop the top card. Returns the card dict, or None if empty
        (and reshuffle_on_empty=False). The engine's PLACE_FROM_DECK /
        DRAW_FROM_DECK effect handler is responsible for applying the
        card's effects — DeckModule only manages the deck itself."""
        if not self._draw_pile:
            if self._reshuffle_on_empty:
                self._reshuffle()
            else:
                return None
        if not self._draw_pile:
            return None
        card = self._draw_pile.pop()
        # If the card is "kept", store it on the entity instead of
        # returning to bottom; the caller applies effects regardless.
        if card.get("keep_until_used"):
            self._held.setdefault(actor.id if actor else "_unowned", []).append(dict(card))
        elif self._consume_on_draw:
            # Truly gone. Nothing more to do — `reshuffle_on_empty` will
            # rebuild from _original_cards (minus held) when empty.
            pass
        else:
            # Cycle: put under the pile so the deck never depletes.
            self._discard_to_bottom(card)
        return card

    def _discard_to_bottom(self, card: Dict[str, Any]) -> None:
        """For non-kept cards, put them at the BOTTOM of the draw pile
        so we cycle naturally; reshuffle_on_empty re-randomises."""
        self._draw_pile.insert(0, dict(card))

    def _reshuffle(self) -> None:
        """Rebuild the draw pile from the original deck, minus any
        cards currently held by entities."""
        held_idxs = {c.get("idx") for cards in self._held.values() for c in cards}
        self._draw_pile = [
            dict(c) for c in self._original_cards
            if c.get("idx") not in held_idxs
        ]
        self._rng.shuffle(self._draw_pile)

    # ── Held cards ──

    def held_by(self, entity_id: str) -> List[Dict[str, Any]]:
        return list(self._held.get(entity_id, []))

    def use_held(self, entity_id: str, idx: int = 0) -> Optional[Dict[str, Any]]:
        """Consume the entity's held card at `idx`. Returns the card so
        the engine can apply its effects. The card returns to the
        bottom of the draw pile."""
        held = self._held.get(entity_id)
        if not held or idx >= len(held):
            return None
        card = held.pop(idx)
        if not held:
            self._held.pop(entity_id, None)
        # Reset keep flag so it cycles normally next time.
        card.pop("keep_until_used", None)
        self._discard_to_bottom(card)
        return card

    # ────────────────────────────────────────────────────────────── #
    # Perception
    # ────────────────────────────────────────────────────────────── #

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "deck_id": self._id,
            "remaining": self.size(),
        }
        held = self.held_by(entity_id)
        if held:
            data["your_held_cards"] = [
                {"text": c.get("text", ""), "idx": c.get("idx")}
                for c in held
            ]
        return data

    # ────────────────────────────────────────────────────────────── #
    # Serialisation
    # ────────────────────────────────────────────────────────────── #

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["state"] = {
            "draw_pile": list(self._draw_pile),
            "held": {k: list(v) for k, v in self._held.items()},
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "DeckModule":
        mod = cls(name=data.get("name", "deck"), params=data.get("params", {}))
        s = data.get("state", {})
        if "draw_pile" in s:
            mod._draw_pile = [dict(c) for c in s["draw_pile"]]
        if "held" in s:
            mod._held = {k: [dict(c) for c in v] for k, v in s["held"].items()}
        return mod


__all__ = ["DeckModule"]
