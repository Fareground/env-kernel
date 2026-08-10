"""Agent-to-agent messaging system.

Agents can send messages to each other -- direct, broadcast, or faction-wide.
Messages appear in perception and influence agent decisions.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A message sent between agents."""
    sender_id: str
    sender_name: str
    content: str                                # The message text
    message_type: str = "broadcast"             # "direct" | "broadcast" | "faction"
    recipient_id: Optional[str] = None          # For direct messages
    recipient_faction: Optional[str] = None     # For faction messages
    round_sent: int = 0
    info_payload: Optional[Dict[str, Any]] = None  # Structured info for world model sharing
    negotiation_id: Optional[str] = None            # Links to negotiation state machine
    proposal: Optional[Dict[str, Any]] = None       # Structured proposal terms


class MessageBoard:
    """Manages messages posted during the simulation."""

    def __init__(self):
        self._current_round_messages: List[Message] = []
        self._history: List[Message] = []

    def post(self, message: Message):
        """Post a message. Adds to both current round and full history."""
        self._current_round_messages.append(message)
        self._history.append(message)

    def get_for_entity(
        self,
        entity_id: str,
        faction_id: Optional[str] = None,
    ) -> List[Message]:
        """Get messages visible to an entity for the current round.

        An entity can see:
        - Direct messages addressed to them
        - Broadcast messages
        - Faction messages if they are in the same faction
        """
        visible = []
        for msg in self._current_round_messages:
            if msg.sender_id == entity_id:
                continue  # Don't show own messages
            if msg.message_type == "direct":
                if msg.recipient_id == entity_id:
                    visible.append(msg)
            elif msg.message_type == "broadcast":
                visible.append(msg)
            elif msg.message_type == "faction":
                if faction_id and msg.recipient_faction == faction_id:
                    visible.append(msg)
        return visible

    def start_round(self):
        """Clear current-round messages at the start of each round."""
        self._current_round_messages = []

    def get_history(self, limit: int = 10) -> List[Message]:
        """Get recent message history across all rounds."""
        return self._history[-limit:]

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        return {
            "current_round": [
                {
                    "sender_id": m.sender_id,
                    "sender_name": m.sender_name,
                    "content": m.content,
                    "message_type": m.message_type,
                    "recipient_id": m.recipient_id,
                    "recipient_faction": m.recipient_faction,
                    "round_sent": m.round_sent,
                    "info_payload": m.info_payload,
                    "negotiation_id": m.negotiation_id,
                    "proposal": m.proposal,
                }
                for m in self._current_round_messages
            ],
            "history_count": len(self._history),
        }
