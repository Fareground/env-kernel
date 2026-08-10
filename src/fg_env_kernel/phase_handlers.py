"""
Phase handlers -- automated game logic that executes during non-agent phases.

Same registry pattern as resolution.py:
- Abstract base class PhaseHandler
- Concrete implementations for each handler type
- PHASE_HANDLER_REGISTRY dict for lookup
- get_phase_handler() accessor function

Handlers mutate WorldState directly and return events for the transcript.
"""
import logging
import random as random_module
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import WorldState

logger = logging.getLogger(__name__)

# Standard 52-card deck
STANDARD_DECK = [
    f"{rank}{suit}"
    for suit in ["♠", "♥", "♦", "♣"]
    for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
]

# Hand rank scoring (simplified)
RANK_VALUES = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def _parse_card_rank(card: str) -> int:
    """Extract numeric rank from a card string like '10♠' or 'A♥'."""
    rank_str = card[:-1]  # Remove suit character
    return RANK_VALUES.get(rank_str, 0)


def _score_hand(hole_cards: List[str], community_cards: List[str]) -> int:
    """
    Simplified poker hand scoring. Returns a numeric score (higher = better).
    Checks for: pairs, two pair, three of a kind, straight, flush, full house, four of a kind.
    """
    all_cards = hole_cards + community_cards
    if not all_cards:
        return 0

    ranks = [_parse_card_rank(c) for c in all_cards]
    suits = [c[-1] for c in all_cards]

    rank_counts: Dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    suit_counts: Dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    counts = sorted(rank_counts.values(), reverse=True)
    has_flush = any(c >= 5 for c in suit_counts.values())

    sorted_ranks = sorted(set(ranks))
    has_straight = False
    for i in range(len(sorted_ranks) - 4):
        if sorted_ranks[i + 4] - sorted_ranks[i] == 4:
            has_straight = True
            break
    # Check A-2-3-4-5 straight
    if {14, 2, 3, 4, 5}.issubset(set(ranks)):
        has_straight = True

    score = 0
    high_card = max(ranks) if ranks else 0

    if has_straight and has_flush:
        score = 8000 + high_card
    elif counts[0] == 4:
        score = 7000 + high_card
    elif counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        score = 6000 + high_card
    elif has_flush:
        score = 5000 + high_card
    elif has_straight:
        score = 4000 + high_card
    elif counts[0] == 3:
        score = 3000 + high_card
    elif counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        score = 2000 + high_card
    elif counts[0] == 2:
        score = 1000 + high_card
    else:
        score = high_card

    return score


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

@dataclass
class PhaseHandlerResult:
    """Outcome of running a phase handler."""
    events: List[Dict[str, Any]] = field(default_factory=list)


class PhaseHandler(ABC):
    """Base class for all phase handlers."""

    @abstractmethod
    def execute(
        self,
        state: "WorldState",
        params: Dict[str, Any],
        round_number: int,
        rng: Optional[random_module.Random] = None,
    ) -> PhaseHandlerResult:
        """Execute the phase handler logic. Mutates state directly. Returns events."""
        ...


# ---------------------------------------------------------------------------
# Entity resolution helpers
# ---------------------------------------------------------------------------

def _resolve_entity(state: "WorldState", entity_ref: str):
    """Resolve an entity reference — try by ID first, then by name."""
    entity = state.get_entity(entity_ref)
    if entity:
        return entity
    # Try by name (entity IDs may differ from the reference in handler params)
    for e in state.entities.values():
        if e.name.lower() == entity_ref.lower():
            return e
    # Substring match on ID (e.g. "deck" matches "deck_1")
    for e in state.entities.values():
        if entity_ref.lower() in e.id.lower():
            return e
    return None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

class ResetDeckHandler(PhaseHandler):
    """Rebuilds full 52-card deck, clears all player hands and community cards."""

    def execute(self, state, params, round_number, rng=None):
        deck_entity = params.get("deck_entity", "deck")
        deck_property = params.get("deck_property", "cards")
        community_entity = params.get("community_entity")
        community_property = params.get("community_property", "community_cards")
        target_type = params.get("target_type")
        hand_property = params.get("hand_property", "hand")

        result = PhaseHandlerResult()

        deck = _resolve_entity(state, deck_entity)
        if deck:
            deck.set(deck_property, list(STANDARD_DECK))
            result.events.append({
                "type": "phase_handler",
                "narrative": "Deck shuffled and reset to 52 cards.",
                "data": {"handler": "reset_deck", "cards": 52},
            })
        else:
            logger.warning(f"reset_deck: deck entity '{deck_entity}' not found")

        # Clear community cards
        if community_entity:
            comm = _resolve_entity(state, community_entity)
            if comm:
                comm.set(community_property, [])

        # Clear player hands
        if target_type:
            recipients = state.get_entities_by_type(target_type)
            for entity in recipients:
                if entity.alive:
                    entity.set(hand_property, [])

        return result


class CollectAntesHandler(PhaseHandler):
    """Deducts a fixed resource amount from all alive entities of a type into a pot."""

    def execute(self, state, params, round_number, rng=None):
        resource_name = params.get("resource", "chips")
        amount = params.get("amount", 10)
        source_type = params.get("source_type")
        pot_entity = params.get("pot_entity")

        result = PhaseHandlerResult()
        pool = state.resources.get(resource_name)
        if not pool:
            return result

        # Resolve pot entity ID
        if pot_entity and not state.get_entity(pot_entity):
            resolved = _resolve_entity(state, pot_entity)
            if resolved:
                pot_entity = resolved.id

        collected = 0
        if source_type:
            entities = state.get_entities_by_type(source_type)
        else:
            entities = list(state.entities.values())

        for entity in entities:
            if not entity.alive:
                continue
            et = state.entity_types.get(entity.entity_type)
            if not et or et.role != "agent":
                continue

            held = pool.get(entity.id)
            actual = min(amount, held)
            if actual > 0:
                pool.remove(entity.id, actual)
                if pot_entity:
                    pool.add(pot_entity, actual) if not pool.resource_type.conservation else None
                    # For conserved resources, transfer instead
                    if pool.resource_type.conservation:
                        pool.holdings[pot_entity] = pool.holdings.get(pot_entity, 0) + actual
                else:
                    pool.unallocated += actual
                collected += actual

        result.events.append({
            "type": "phase_handler",
            "narrative": f"Collected {collected} {resource_name} in antes.",
            "data": {"handler": "collect_antes", "resource": resource_name, "amount": collected},
        })
        return result


class CardDealHandler(PhaseHandler):
    """Shuffles deck and deals cards to each alive agent of a given type."""

    def execute(self, state, params, round_number, rng=None):
        deck_entity = params.get("deck_entity", "deck")
        deck_property = params.get("deck_property", "cards")
        count = params.get("count", 2)
        recipient_type = params.get("recipient_type")
        hand_property = params.get("hand_property", "hand")

        result = PhaseHandlerResult()
        if rng is None:
            rng = random_module.Random()

        deck = _resolve_entity(state, deck_entity)
        if not deck:
            result.events.append({
                "type": "phase_handler_error",
                "narrative": f"Deck entity '{deck_entity}' not found.",
                "data": {"handler": "card_deal", "error": "deck_not_found"},
            })
            return result

        cards = deck.get(deck_property, [])
        if not isinstance(cards, list) or not cards:
            cards = list(STANDARD_DECK)
            rng.shuffle(cards)

        # Shuffle
        rng.shuffle(cards)

        recipients = []
        if recipient_type:
            recipients = [e for e in state.get_entities_by_type(recipient_type) if e.alive]
        if not recipients:
            # Fallback: deal to all alive agents (handles missing type gracefully)
            recipients = [e for e in state.entities.values() if e.alive and state.entity_types.get(e.entity_type, None) and state.entity_types[e.entity_type].role == "agent"]
            if recipient_type:
                logger.warning(f"card_deal: recipient_type '{recipient_type}' matched no entities, falling back to all alive agents")

        for entity in recipients:
            dealt = cards[:count]
            cards = cards[count:]
            current_hand = entity.get(hand_property, [])
            if not isinstance(current_hand, list):
                current_hand = []
            entity.set(hand_property, current_hand + dealt)
            result.events.append({
                "type": "phase_handler",
                "actor_id": entity.id,
                "narrative": f"{entity.name} was dealt {count} cards.",
                "data": {"handler": "card_deal", "entity": entity.id, "cards_dealt": count},
            })

        # Update remaining deck
        deck.set(deck_property, cards)
        return result


class CommunityCardsHandler(PhaseHandler):
    """Reveals community cards from the deck to a shared entity."""

    def execute(self, state, params, round_number, rng=None):
        deck_entity = params.get("deck_entity", "deck")
        deck_property = params.get("deck_property", "cards")
        count = params.get("count", 3)
        community_entity = params.get("community_entity", "table")
        community_property = params.get("community_property", "community_cards")

        result = PhaseHandlerResult()

        deck = _resolve_entity(state, deck_entity)
        if not deck:
            logger.warning(f"community_cards: deck entity '{deck_entity}' not found")
            return result

        cards = deck.get(deck_property, [])
        if not isinstance(cards, list):
            return result

        revealed = cards[:count]
        remaining = cards[count:]
        deck.set(deck_property, remaining)

        comm = _resolve_entity(state, community_entity)
        if comm:
            current = comm.get(community_property, [])
            if not isinstance(current, list):
                current = []
            comm.set(community_property, current + revealed)

        result.events.append({
            "type": "phase_handler",
            "narrative": f"{count} community card(s) revealed: {', '.join(revealed)}.",
            "data": {"handler": "community_cards", "cards": revealed},
        })
        return result


class PokerShowdownHandler(PhaseHandler):
    """Evaluates poker hands and distributes pot to winner(s)."""

    def execute(self, state, params, round_number, rng=None):
        hand_property = params.get("hand_property", "hand")
        pot_resource = params.get("pot_resource", "chips")
        player_type = params.get("player_type")
        community_entity = params.get("community_entity", "table")
        community_property = params.get("community_property", "community_cards")

        result = PhaseHandlerResult()

        # Get community cards
        comm = _resolve_entity(state, community_entity)
        community_cards = []
        if comm:
            community_cards = comm.get(community_property, [])
            if not isinstance(community_cards, list):
                community_cards = []

        # Evaluate each player
        players = []
        if player_type:
            players = [e for e in state.get_entities_by_type(player_type) if e.alive]
        if not players:
            players = state.get_agent_entities()
            if player_type:
                logger.warning(f"poker_showdown: player_type '{player_type}' matched no entities, using all agents")

        scores: List[tuple] = []
        for player in players:
            hand = player.get(hand_property, [])
            if not isinstance(hand, list):
                hand = []
            # Check if player folded (empty hand or folded property)
            if not hand:
                continue
            if player.get("folded", False):
                continue
            score = _score_hand(hand, community_cards)
            scores.append((player, score, hand))

        if not scores:
            result.events.append({
                "type": "phase_handler",
                "narrative": "No players remaining for showdown.",
                "data": {"handler": "poker_showdown", "winner": None},
            })
            return result

        # Find winner(s)
        max_score = max(s[1] for s in scores)
        winners = [(p, h) for p, s, h in scores if s == max_score]

        # Distribute pot
        pool = state.resources.get(pot_resource)
        pot_entity = params.get("pot_entity")
        pot_amount = 0.0

        if pool:
            if pot_entity:
                pot_amount = pool.get(pot_entity)
                pool.set(pot_entity, 0)
            else:
                pot_amount = pool.unallocated
                pool.unallocated = 0

            share = pot_amount / len(winners) if winners else 0
            for winner, hand in winners:
                if pool.resource_type.discrete:
                    pool.holdings[winner.id] = pool.holdings.get(winner.id, 0) + int(share)
                else:
                    pool.holdings[winner.id] = pool.holdings.get(winner.id, 0) + share

        winner_names = [w[0].name for w in winners]
        result.events.append({
            "type": "phase_handler",
            "narrative": f"Showdown! {', '.join(winner_names)} wins {pot_amount} {pot_resource}.",
            "data": {
                "handler": "poker_showdown",
                "winners": [w[0].id for w in winners],
                "pot": pot_amount,
                "hands": {p.id: h for p, h in winners},
            },
        })
        return result


class ResetFieldHandler(PhaseHandler):
    """Resets properties on all entities of a given type."""

    def execute(self, state, params, round_number, rng=None):
        target_type = params.get("target_type")
        resets = params.get("resets", {})

        result = PhaseHandlerResult()
        count = 0

        entities = state.get_entities_by_type(target_type) if target_type else list(state.entities.values())
        for entity in entities:
            if not entity.alive:
                continue
            for field_name, value in resets.items():
                entity.set(field_name, value)
            count += 1

        if resets:
            result.events.append({
                "type": "phase_handler",
                "narrative": f"Reset {list(resets.keys())} on {count} entities.",
                "data": {"handler": "reset_field", "fields": list(resets.keys()), "count": count},
            })
        return result


# ---------------------------------------------------------------------------
# Setup placement (6.I) — randomly drop entity positions before play
# ---------------------------------------------------------------------------

class SetupPlacementHandler(PhaseHandler):
    """Place pieces / units / ships at random or per-spec coordinates
    before the first agent turn. Used by battleship to lay out hidden
    fleets, by any war-game to place starting units.

    Two modes:
      mode="random_grid"  — places N pieces per agent on random grid cells
      mode="from_spec"    — uses per-entity_type `placements: [{entity, position}]`
                            from handler params.

    Params:
      board_id: "main"
      entity_type: "Ship" | "Player" | ...
      pieces_per_agent: 5             # for random_grid
      cells_per_piece: 1              # variable-size pieces (battleship: ships span N cells)
      forbidden_cells: [[0,0], ...]   # cells to avoid (optional)
      placements: [...]               # for from_spec mode
    """

    def execute(
        self,
        state: "WorldState",
        params: Dict[str, Any],
        round_number: int,
        rng: Optional[random_module.Random] = None,
    ) -> PhaseHandlerResult:
        events: List[Dict[str, Any]] = []
        # Fire once via state-attached marker (rounds are 1-indexed; the
        # original `round_number != 0` check meant this never fired).
        marker_key = "_setup_placement_fired"
        if not hasattr(state, marker_key):
            setattr(state, marker_key, set())
        fired: set = getattr(state, marker_key)
        sig = id(params)
        if sig in fired:
            return PhaseHandlerResult(events=events)
        fired.add(sig)

        mode = params.get("mode", "random_grid")
        board_id = params.get("board_id", "main")
        # Find the BoardModule by id
        board = None
        if state.domain_modules:
            from .board_module import BoardModule
            for _m in state.domain_modules._modules.values():
                if isinstance(_m, BoardModule) and _m._id == board_id:
                    board = _m
                    break
        if board is None:
            events.append({"type": "setup_placement_failed",
                           "reason": f"no board with id={board_id!r}"})
            return PhaseHandlerResult(events=events)

        rng = rng or random_module.Random()

        if mode == "from_spec":
            for placement in params.get("placements") or []:
                ent_id = placement.get("entity_id")
                pos = placement.get("position")
                ent = state.get_entity(ent_id) if ent_id else None
                if ent and pos is not None:
                    ent.set(board._position_property, pos)
                    events.append({
                        "type": "piece_placed",
                        "entity_id": ent_id, "position": pos,
                    })
            return PhaseHandlerResult(events=events)

        # random_grid mode
        entity_type = params.get("entity_type")
        pieces_per_agent = int(params.get("pieces_per_agent", 1))
        cells_per_piece = int(params.get("cells_per_piece", 1))
        forbidden = {tuple(c) for c in (params.get("forbidden_cells") or [])}

        agents = [
            e for e in state.get_agent_entities()
            if (not entity_type) or e.entity_type == entity_type
        ]

        # Each agent gets `pieces_per_agent` cells (or runs of cells for
        # ship-style multi-cell pieces). We track per-agent ship coords
        # on a property named `ship_grid` (list of [row, col] tuples).
        for ag in agents:
            placed: List[Tuple[int, int]] = []
            attempts = 0
            while len(placed) < pieces_per_agent and attempts < 500:
                attempts += 1
                r = rng.randint(0, board._rows - 1)
                c = rng.randint(0, board._cols - 1)
                if cells_per_piece == 1:
                    candidate = [(r, c)]
                else:
                    horizontal = rng.random() < 0.5
                    if horizontal:
                        if c + cells_per_piece > board._cols:
                            continue
                        candidate = [(r, c + i) for i in range(cells_per_piece)]
                    else:
                        if r + cells_per_piece > board._rows:
                            continue
                        candidate = [(r + i, c) for i in range(cells_per_piece)]
                if any(cell in placed or cell in forbidden for cell in candidate):
                    continue
                placed.extend(candidate)
            ag.set("ship_grid", [list(c) for c in placed])  # type: ignore[misc]  # PropertyValue is narrower than the runtime property store
            events.append({
                "type": "piece_placed",
                "entity_id": ag.id,
                "positions": [list(c) for c in placed],
                "count": len(placed),
            })
        return PhaseHandlerResult(events=events)


# ---------------------------------------------------------------------------
# Property assignment (assign mark X/O, role, side, etc.) — setup phase
# ---------------------------------------------------------------------------

class AssignPropertiesHandler(PhaseHandler):
    """Round-robin distribute property values across agents at setup.

    Used by every env that needs deterministic seat-based assignment —
    chess sides (white/black), tic-tac-toe marks (X/O), poker positions,
    mafia roles, faction membership.

    Params:
      entity_type: "Player"            # narrow by entity type (optional)
      assignments: [                   # one entry per property to fill
        { property: "mark",  values: ["X", "O"] },
        { property: "side",  values: ["white", "black"] },
        { property: "seat",  values: [0, 1, 2, 3] }    # cycles if shorter than agent count
      ]
      shuffle: false                   # randomise assignment order
      run_once: true                   # only fire on round 0 (default)
    """

    def execute(
        self,
        state: "WorldState",
        params: Dict[str, Any],
        round_number: int,
        rng: Optional[random_module.Random] = None,
    ) -> PhaseHandlerResult:
        events: List[Dict[str, Any]] = []
        # Per-AGENT tracking — arena matches inject the user's agent
        # AFTER the schema's initial entities, so a run-once-per-state
        # check would skip the latecomer. We track which entity ids
        # have already been assigned which property; new agents get
        # the next slot in the values list.
        marker_key = "_assign_properties_done"
        if not hasattr(state, marker_key):
            setattr(state, marker_key, {})
        done: Dict[str, Dict[str, set]] = getattr(state, marker_key)

        entity_type = params.get("entity_type")
        agents = [
            e for e in state.get_agent_entities()
            if (not entity_type) or e.entity_type == entity_type
        ]
        if not agents:
            return PhaseHandlerResult(events=events)

        if params.get("shuffle"):
            rng = rng or random_module.Random()
            agents = list(agents)
            rng.shuffle(agents)

        params_sig = str(id(params))
        per_phase = done.setdefault(params_sig, {})

        assignments = params.get("assignments") or []
        for assignment in assignments:
            if not isinstance(assignment, dict):
                continue
            prop = assignment.get("property")
            values = assignment.get("values") or []
            if not prop or not values:
                continue
            assigned_for_prop: set = per_phase.setdefault(prop, set())
            # Determine the NEXT slot in the values cycle. Already-assigned
            # agents keep their value; new agents get the next slot.
            next_slot = len(assigned_for_prop)
            for ag in agents:
                if ag.id in assigned_for_prop:
                    continue
                v = values[next_slot % len(values)]
                ag.set(prop, v)
                assigned_for_prop.add(ag.id)
                next_slot += 1
                events.append({
                    "type": "property_assigned",
                    "entity_id": ag.id,
                    "property": prop,
                    "value": v,
                })
        return PhaseHandlerResult(events=events)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PHASE_HANDLER_REGISTRY: Dict[str, PhaseHandler] = {
    "reset_deck": ResetDeckHandler(),
    "collect_antes": CollectAntesHandler(),
    "card_deal": CardDealHandler(),
    "community_cards": CommunityCardsHandler(),
    "poker_showdown": PokerShowdownHandler(),
    "reset_field": ResetFieldHandler(),
    "setup_placement": SetupPlacementHandler(),
    "assign_properties": AssignPropertiesHandler(),
}


def get_phase_handler(name: str) -> PhaseHandler:
    """Get a phase handler by name. Raises ValueError if not found."""
    if name not in PHASE_HANDLER_REGISTRY:
        raise ValueError(
            f"Unknown phase handler: '{name}'. "
            f"Available: {list(PHASE_HANDLER_REGISTRY.keys())}"
        )
    return PHASE_HANDLER_REGISTRY[name]
