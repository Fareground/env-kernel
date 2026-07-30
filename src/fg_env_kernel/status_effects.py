"""Status effects -- temporary or permanent modifiers on entities.

Status effects alter entity capabilities: blocking actions, modifying
properties during resolution, and applying per-round tick effects
(damage-over-time, healing, etc.).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .action import Effect


@dataclass
class StatusEffectDefinition:
    """Blueprint for a status effect type."""
    name: str                                # "poisoned", "inspired", "exhausted"
    description: str = ""
    duration: int = 0                        # 0 = permanent until removed
    tick_effects: List[Effect] = field(default_factory=list)  # Applied each round
    property_modifiers: Dict[str, float] = field(default_factory=dict)  # Additive modifiers
    blocks_actions: List[str] = field(default_factory=list)   # Actions prevented
    stackable: bool = False
    max_stacks: int = 1
    narrative_tag: str = ""                  # e.g., "writhing in pain"


@dataclass
class ActiveStatusEffect:
    """A status effect currently active on an entity."""
    definition: StatusEffectDefinition
    source_entity: Optional[str] = None
    remaining_rounds: int = 0                # 0 = permanent
    stacks: int = 1
    applied_round: int = 0


class StatusEffectTracker:
    """Manages active status effects on all entities."""

    def __init__(self):
        self._effects: Dict[str, List[ActiveStatusEffect]] = {}  # entity_id -> list

    def apply(
        self,
        entity_id: str,
        effect_def: StatusEffectDefinition,
        source: Optional[str] = None,
        round_num: int = 0,
    ):
        """Apply a status effect to an entity."""
        if entity_id not in self._effects:
            self._effects[entity_id] = []

        existing = self._effects[entity_id]

        # Check for existing effect of same type
        for ase in existing:
            if ase.definition.name == effect_def.name:
                if effect_def.stackable:
                    if ase.stacks < effect_def.max_stacks:
                        ase.stacks += 1
                    # Always refresh duration (even at max stacks)
                    ase.remaining_rounds = effect_def.duration
                    return
                else:
                    # Refresh duration
                    ase.remaining_rounds = effect_def.duration
                    return

        # New effect
        active = ActiveStatusEffect(
            definition=effect_def,
            source_entity=source,
            remaining_rounds=effect_def.duration,
            stacks=1,
            applied_round=round_num,
        )
        existing.append(active)

    def tick(self, entity_id: str, round_num: int = 0) -> List[Effect]:
        """Process per-turn effects. Decrements durations, removes expired.

        Returns the list of tick_effects to apply this turn.
        """
        if entity_id not in self._effects:
            return []

        effects_to_apply = []
        still_active = []

        for ase in self._effects[entity_id]:
            # Collect tick effects (multiplied by stacks)
            for effect in ase.definition.tick_effects:
                for _ in range(ase.stacks):
                    effects_to_apply.append(effect)

            # Decrement duration (0 = permanent, never expires)
            if ase.remaining_rounds > 0:
                ase.remaining_rounds -= 1
                if ase.remaining_rounds > 0:
                    still_active.append(ase)
                # else: expired, don't keep
            else:
                # Permanent effect
                still_active.append(ase)

        self._effects[entity_id] = still_active
        return effects_to_apply

    def get_modifiers(self, entity_id: str) -> Dict[str, float]:
        """Get aggregate property modifiers for an entity."""
        if entity_id not in self._effects:
            return {}

        modifiers: Dict[str, float] = {}
        for ase in self._effects[entity_id]:
            for prop, mod_val in ase.definition.property_modifiers.items():
                modifiers[prop] = modifiers.get(prop, 0.0) + (mod_val * ase.stacks)
        return modifiers

    def get_blocked_actions(self, entity_id: str) -> List[str]:
        """Get actions currently blocked for an entity."""
        if entity_id not in self._effects:
            return []

        blocked = set()
        for ase in self._effects[entity_id]:
            for action_name in ase.definition.blocks_actions:
                blocked.add(action_name)
        return list(blocked)

    def remove(self, entity_id: str, effect_name: str):
        """Remove a status effect by name."""
        if entity_id not in self._effects:
            return
        self._effects[entity_id] = [
            ase for ase in self._effects[entity_id]
            if ase.definition.name != effect_name
        ]

    def clear_entity(self, entity_id: str):
        """Remove all status effects for an entity."""
        self._effects.pop(entity_id, None)

    def get_active(self, entity_id: str) -> List[ActiveStatusEffect]:
        """Get all active effects on an entity."""
        return list(self._effects.get(entity_id, []))

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        result = {}
        for entity_id, effects in self._effects.items():
            result[entity_id] = [
                {
                    "name": ase.definition.name,
                    "remaining_rounds": ase.remaining_rounds,
                    "stacks": ase.stacks,
                    "source": ase.source_entity,
                }
                for ase in effects
            ]
        return result
