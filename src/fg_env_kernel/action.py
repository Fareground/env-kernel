"""Action definitions, preconditions, and effects."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Operator(Enum):
    """Precondition operators."""
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    HAS_RESOURCE = "has_resource"
    AT_LOCATION = "at_location"
    IS_ADJACENT = "is_adjacent"
    RELATION_GTE = "relation_gte"
    IS_ALIVE = "is_alive"
    SAME_FACTION = "same_faction"
    DIFFERENT_FACTION = "different_faction"
    HAS_ITEM = "has_item"
    HAS_ITEM_TYPE = "has_item_type"
    INVENTORY_NOT_FULL = "inventory_not_full"
    SKILL_GTE = "skill_gte"
    HAS_RECIPE = "has_recipe"


@dataclass
class Precondition:
    """A single precondition check.

    Two equivalent forms:
        Legacy:   Precondition(subject="actor", operator=Operator.GTE,
                                field="gold", value=100)
        Modern:   Precondition(expr="$actor.gold >= 100")

    When ``expr`` is set it takes precedence and is evaluated through
    the unified predicate engine (``fg_env_kernel.predicates``).
    The modern form is preferred for new games — one grammar covers
    every kind of guard.
    """
    subject: str = "actor"              # "actor", "target", or entity property path
    operator: Operator = Operator.EQ
    value: Any = None                   # The value to compare against
    field: Optional[str] = None         # Property name (for property checks)
    resource: Optional[str] = None      # Resource name (for has_resource)
    relation_type: Optional[str] = None # For relation checks
    description: str = ""
    # New: full expression form. String ("$actor.gold >= 100") or dict.
    # Takes precedence when truthy.
    expr: Any = None


class EffectOperation(Enum):
    """Types of state changes an effect can apply.

    NOTE: New ops should be added via the kernel registry
    (``fg_env_kernel.registry.effect``), not by extending this enum.
    The enum exists for backwards compatibility — built-in ops still
    live here so the engine's legacy switch keeps working, but custom
    games extend the verb set without editing kernel code."""
    SET = "set"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    TRANSFER_RESOURCE = "transfer_resource"
    SET_RELATION = "set_relation"
    MODIFY_RELATION = "modify_relation"
    MOVE_TO = "move_to"
    KILL = "kill"
    EMIT_EVENT = "emit_event"
    APPLY_STATUS = "apply_status"
    REMOVE_STATUS = "remove_status"
    SPAWN_ENTITY = "spawn_entity"
    DESPAWN_ENTITY = "despawn_entity"
    GIVE_ITEM = "give_item"
    TAKE_ITEM = "take_item"
    DROP_ITEM = "drop_item"
    PICKUP_ITEM = "pickup_item"
    AWARD_XP = "award_xp"
    CRAFT_ITEM = "craft_item"
    # Social actions
    POST_CONTENT = "post_content"
    SHARE_CONTENT = "share_content"
    REACT_CONTENT = "react_content"
    FOLLOW_ENTITY = "follow_entity"
    UNFOLLOW_ENTITY = "unfollow_entity"
    # Generic game tokens (Phase 1 EffectDSL — replaces per-env
    # `jail_cards`-style counters with a declarative tokens dict).
    GRANT_TOKEN = "grant_token"
    CONSUME_TOKEN = "consume_token"
    # Increment a numeric property by a value or expression. Same as
    # ADD but exists explicitly for board-track advancement clarity.
    ADVANCE_TRACK = "advance_track"
    # Place a mark on a BoardModule cell (tic-tac-toe, connect-four).
    # `field` = board_id; `value` = mark to place; condition.field == "cell"
    # carries the cell index ($params expression).
    PLACE_ON_BOARD = "place_on_board"
    # Wipe all marks on a named BoardModule. Used for multi-game
    # tic-tac-toe / connect-four / any "best-of-N" format where a game
    # ends and the board resets for the next game.
    # `field` = board_id.
    RESET_BOARD = "reset_board"
    # Draw a card from a named DeckModule and apply its `effect` list.
    # `field` = deck_id. The drawn card's effects fire against this
    # same actor/target/params context — so a Chance card can move the
    # actor, pay the bank, or grant a token.
    DRAW_FROM_DECK = "draw_from_deck"
    # Conditional branching effect. `value` is a dict:
    #   { "if": { subject, field, operator, value }, OR
    #     "if_expr": "$actor.position == 30",  (expression form)
    #     "then": [ effect_dict, ... ],
    #     "else": [ effect_dict, ... ]   (optional) }
    # Lets the schema express "pass-GO collects $200" or "promote pawn
    # at back rank" without any code.
    CONDITIONAL = "conditional"
    # Per-player hand operations (Tier 5a — HandModule).
    # `field` carries the hand_id (defaults to "hand"); `value` carries
    # the number of cards (DRAW_TO_HAND) or the card_id string
    # (DISCARD_FROM_HAND / PLAY_CARD / PASS_CARD).
    DRAW_TO_HAND = "draw_to_hand"
    DISCARD_FROM_HAND = "discard_from_hand"
    PLAY_CARD = "play_card"
    PASS_CARD = "pass_card"
    # Worker placement (Tier 5b — SlotsModule).
    # `field` = slots_id, `value` = slot_id to claim/release.
    CLAIM_SLOT = "claim_slot"
    RELEASE_SLOT = "release_slot"
    # Use one of the entity's held (kept) cards — e.g. "get out of
    # jail free". `field` = deck_id. Optional params['card_idx']
    # selects which held card.
    USE_HELD_CARD = "use_held_card"


@dataclass
class EffectCondition:
    """A runtime condition that must be True for an effect to apply.

    Two equivalent forms:
        Legacy:   EffectCondition(subject="actor", field="gold",
                                   operator="gte", value=100)
        Modern:   EffectCondition(expr="$actor.gold >= 100")

    When ``expr`` is set it takes precedence and is evaluated through
    the unified predicate engine (``fg_env_kernel.predicates``).
    """
    subject: str = "actor"              # "actor", "target"
    field: Optional[str] = None         # Property to check
    operator: str = "gte"               # "gte", "lte", "gt", "lt", "eq", "neq"
    value: Any = None                   # Value to compare against
    check_type: str = "property"        # "property", "alive", "has_status"
    # New: full expression form. Either a string ("$actor.gold >= 100")
    # or a structured dict ({"op": "gte", ...}). Takes precedence when
    # truthy.
    expr: Any = None


@dataclass
class Effect:
    """A single state change to apply after action resolution."""
    target: str                         # "actor", "target", or entity_id
    operation: EffectOperation
    field: Optional[str] = None         # Property to modify
    value: Any = None                   # Value for the operation
    resource: Optional[str] = None      # Resource name
    relation_type: Optional[str] = None
    description: str = ""
    condition: Optional[EffectCondition] = None  # If set, effect only applies when True
    scale_by_magnitude: bool = False    # When True, numeric value is multiplied by success_degree


@dataclass
class ActionDefinition:
    """
    Complete definition of an action that agents can take.

    The LLM sees: name, description, parameters.
    The engine evaluates: preconditions, resolution, effects.
    """
    name: str
    description: str
    actor_type: str                              # EntityType name that can perform this
    target_type: Optional[str] = None            # If the action requires a target
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    # [{name: "amount", type: "int", required: True, min: 1, max: 100}]

    preconditions: List[Precondition] = field(default_factory=list)

    resolution_archetype: str = "deterministic"
    resolution_params: Dict[str, Any] = field(default_factory=dict)

    effects_on_success: List[Effect] = field(default_factory=list)
    effects_on_failure: List[Effect] = field(default_factory=list)
    effects_on_partial: List[Effect] = field(default_factory=list)  # Applied on partial success

    # Visibility: who can see this action was taken
    broadcast: bool = True
    range: Optional[float] = None

    # Action chains
    requires_action: Optional[str] = None       # Must have completed this action first
    requires_action_success: bool = True         # Only if prior action succeeded
    unlocks_actions: List[str] = field(default_factory=list)  # Unlocked on success
    locks_actions: List[str] = field(default_factory=list)    # Locked on success
    cooldown_rounds: int = 0                     # Can't repeat for N rounds

    # Multi-round sequences
    sequence_rounds: int = 0                     # 0 = instant, >0 = takes N rounds
    interruptible: bool = True                   # Can be interrupted by choosing another action

    # Communication
    message_action: bool = False                 # If True, reasoning is posted as a message

    # Continuous-time duration (only used in CONTINUOUS temporal mode)
    duration: float = 0.0                        # How long this action takes in continuous time


@dataclass
class ActionInstance:
    """
    A concrete action attempt by an agent during a turn.
    Created when an agent decides to act; consumed by the engine.
    """
    action_name: str
    actor_id: str
    target_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    speech: str = ""
