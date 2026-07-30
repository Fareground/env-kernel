"""Multi-round action sequences.

Some actions take multiple rounds to complete (casting, building, negotiating).
The SequenceTracker manages in-progress sequences per entity.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ActiveSequence:
    """An action sequence currently in progress for an entity."""
    entity_id: str
    action_name: str
    target_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    total_rounds: int = 1
    rounds_completed: int = 0
    started_round: int = 0


class SequenceTracker:
    """Manages in-progress multi-round action sequences."""

    def __init__(self):
        self._active: Dict[str, ActiveSequence] = {}  # entity_id -> active sequence

    def start(
        self,
        entity_id: str,
        action_name: str,
        target_id: Optional[str],
        parameters: Dict[str, Any],
        total_rounds: int,
        round_num: int,
    ):
        """Begin a new sequence for an entity. Cancels any existing sequence."""
        self._active[entity_id] = ActiveSequence(
            entity_id=entity_id,
            action_name=action_name,
            target_id=target_id,
            parameters=dict(parameters),
            total_rounds=total_rounds,
            rounds_completed=0,
            started_round=round_num,
        )

    def get_active(self, entity_id: str) -> Optional[ActiveSequence]:
        """Get the active sequence for an entity, or None."""
        return self._active.get(entity_id)

    def advance(self, entity_id: str) -> bool:
        """Advance the sequence by one round. Returns True if now complete."""
        seq = self._active.get(entity_id)
        if not seq:
            return False
        seq.rounds_completed += 1
        if seq.rounds_completed >= seq.total_rounds:
            del self._active[entity_id]
            return True
        return False

    def cancel(self, entity_id: str) -> Optional[ActiveSequence]:
        """Cancel an active sequence. Returns the cancelled sequence or None."""
        return self._active.pop(entity_id, None)

    def is_in_sequence(self, entity_id: str) -> bool:
        """Check if an entity has an active sequence."""
        return entity_id in self._active

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        return {
            entity_id: {
                "action_name": seq.action_name,
                "target_id": seq.target_id,
                "total_rounds": seq.total_rounds,
                "rounds_completed": seq.rounds_completed,
                "started_round": seq.started_round,
            }
            for entity_id, seq in self._active.items()
        }
