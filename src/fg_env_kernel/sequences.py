"""Multi-round action sequences.

Some actions take multiple rounds to complete (casting, building, negotiating).
The SequenceTracker manages in-progress sequences per entity.
"""
from dataclasses import dataclass, field
import copy
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
    last_advanced_round: Optional[int] = None


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
            parameters=copy.deepcopy(parameters),
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
                "parameters": copy.deepcopy(seq.parameters),
                "total_rounds": seq.total_rounds,
                "rounds_completed": seq.rounds_completed,
                "started_round": seq.started_round,
                "last_advanced_round": seq.last_advanced_round,
            }
            for entity_id, seq in self._active.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceTracker":
        tracker = cls()
        for entity_id, row in data.items():
            if "parameters" not in row:
                raise ValueError("legacy active sequence snapshot lacks its action parameters; cannot safely resume")
            seq = ActiveSequence(entity_id=entity_id, **copy.deepcopy(row))
            if type(seq.total_rounds) is not int or seq.total_rounds < 1:
                raise ValueError("sequence total_rounds must be a positive integer")
            if type(seq.rounds_completed) is not int or not 0 <= seq.rounds_completed < seq.total_rounds:
                raise ValueError("invalid active sequence progress")
            if type(seq.started_round) is not int or seq.started_round < 0 or not isinstance(seq.parameters, dict):
                raise ValueError("invalid sequence start round or parameters")
            if seq.last_advanced_round is not None and (
                type(seq.last_advanced_round) is not int or seq.last_advanced_round < seq.started_round
            ):
                raise ValueError("invalid sequence advancement round")
            tracker._active[entity_id] = seq
        return tracker
