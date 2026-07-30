"""Per-agent persistent world model with spatial memory and confidence decay.

Each agent maintains a mental model of the world -- things they've observed,
where entities were, what resources existed at which locations. Observations
degrade in confidence over time. Agents can share information with each other,
creating secondhand knowledge at lower confidence.

This layer sits *on top* of PerceptionBuilder: it ingests each turn's
perception and accumulates a persistent picture of the world.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObservationEntry:
    """A snapshot of an observed entity at a point in time."""
    subject_id: str                         # Entity observed
    subject_name: str                       # Human name at time of observation
    subject_type: str                       # EntityType name
    properties: Dict[str, Any]              # Property snapshot (visible props only)
    location: Optional[str] = None          # Where the subject was
    round_observed: int = 0                 # When this was last updated
    confidence: float = 1.0                 # 1.0 = just seen, degrades toward 0.0
    source: str = "direct"                  # "direct" | "told" | "inferred"

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "subject_type": self.subject_type,
            "properties": dict(self.properties),
            "location": self.location,
            "round_observed": self.round_observed,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationEntry":
        return cls(
            subject_id=data["subject_id"],
            subject_name=data.get("subject_name", data["subject_id"]),
            subject_type=data.get("subject_type", "unknown"),
            properties=data.get("properties", {}),
            location=data.get("location"),
            round_observed=data.get("round_observed", 0),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "direct"),
        )


@dataclass
class SpatialMemoryEntry:
    """Memory of an entity or resource at a specific location."""
    location_id: str
    entity_id: Optional[str] = None         # Entity seen there (or None for resource)
    entity_name: Optional[str] = None
    resource_name: Optional[str] = None     # Resource observed at this location
    amount: Optional[float] = None          # Resource amount at time of observation
    round_observed: int = 0
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "location_id": self.location_id,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "resource_name": self.resource_name,
            "amount": self.amount,
            "round_observed": self.round_observed,
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpatialMemoryEntry":
        return cls(
            location_id=data["location_id"],
            entity_id=data.get("entity_id"),
            entity_name=data.get("entity_name"),
            resource_name=data.get("resource_name"),
            amount=data.get("amount"),
            round_observed=data.get("round_observed", 0),
            confidence=data.get("confidence", 1.0),
        )


# ---------------------------------------------------------------------------
# AgentWorldModel
# ---------------------------------------------------------------------------

class AgentWorldModel:
    """
    Per-agent persistent world model with staleness tracking.

    Updated each turn from current perception. Entries not refreshed
    degrade in confidence over time. Pruned when confidence < threshold.
    """

    PRUNE_THRESHOLD = 0.1
    DEFAULT_TOLD_CONFIDENCE = 0.5

    def __init__(
        self,
        owner_id: str,
        max_observations: int = 200,
        confidence_decay_rate: float = 0.1,
    ):
        self.owner_id = owner_id
        self.max_observations = max_observations
        self.confidence_decay_rate = confidence_decay_rate

        # subject_id -> ObservationEntry  (most recent observation per entity)
        self._observations: Dict[str, ObservationEntry] = {}

        # location_id -> list of SpatialMemoryEntry
        self._spatial_memories: Dict[str, List[SpatialMemoryEntry]] = {}

    # -- Perception ingestion --

    def update_from_perception(self, perception: Dict[str, Any], round_number: int):
        """Ingest current perception into the world model.

        Entities currently visible get confidence reset to 1.0.
        """
        # Update entity observations
        for ve in perception.get("visible_entities", []):
            entry = ObservationEntry(
                subject_id=ve["id"],
                subject_name=ve.get("name", ve["id"]),
                subject_type=ve.get("type", "unknown"),
                properties=dict(ve.get("properties", {})),
                location=ve.get("location"),
                round_observed=round_number,
                confidence=1.0,
                source="direct",
            )
            self._observations[ve["id"]] = entry

            # Also record spatial memory
            loc = ve.get("location")
            if loc:
                self._record_spatial(
                    location_id=loc,
                    entity_id=ve["id"],
                    entity_name=ve.get("name", ve["id"]),
                    round_observed=round_number,
                )

        # Update self observation
        self_data = perception.get("self", {})
        if self_data:
            self._observations[self.owner_id] = ObservationEntry(
                subject_id=self.owner_id,
                subject_name=self_data.get("name", self.owner_id),
                subject_type=self_data.get("entity_type", "unknown"),
                properties={k: v for k, v in self_data.get("properties", {}).items()},
                location=perception.get("location"),
                round_observed=round_number,
                confidence=1.0,
                source="direct",
            )

        # Update resource observations
        for rname, rdata in perception.get("visible_resources", {}).items():
            own_amount = rdata.get("own", 0)
            loc = perception.get("location")
            if loc:
                self._record_spatial(
                    location_id=loc,
                    resource_name=rname,
                    amount=own_amount,
                    round_observed=round_number,
                )

    def _record_spatial(
        self,
        location_id: str,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        amount: Optional[float] = None,
        round_observed: int = 0,
    ):
        """Add or update a spatial memory entry."""
        if location_id not in self._spatial_memories:
            self._spatial_memories[location_id] = []

        entries = self._spatial_memories[location_id]

        # Update existing entry if present
        for entry in entries:
            if entity_id and entry.entity_id == entity_id:
                entry.round_observed = round_observed
                entry.confidence = 1.0
                entry.entity_name = entity_name
                return
            if resource_name and entry.resource_name == resource_name:
                entry.amount = amount
                entry.round_observed = round_observed
                entry.confidence = 1.0
                return

        # New entry
        entries.append(SpatialMemoryEntry(
            location_id=location_id,
            entity_id=entity_id,
            entity_name=entity_name,
            resource_name=resource_name,
            amount=amount,
            round_observed=round_observed,
            confidence=1.0,
        ))

    # -- Confidence decay --

    def tick(self, current_round: int):
        """Degrade confidence on all entries. Prune below threshold."""
        # Decay entity observations
        to_remove = []
        for sid, entry in self._observations.items():
            if sid == self.owner_id:
                continue  # Self-knowledge doesn't decay
            age = current_round - entry.round_observed
            if age > 0:
                entry.confidence = math.exp(-self.confidence_decay_rate * age)
                if entry.confidence < self.PRUNE_THRESHOLD:
                    to_remove.append(sid)
        for sid in to_remove:
            del self._observations[sid]

        # Decay spatial memories
        for loc_id, entries in list(self._spatial_memories.items()):
            surviving = []
            for entry in entries:
                age = current_round - entry.round_observed
                if age > 0:
                    entry.confidence = math.exp(-self.confidence_decay_rate * age)
                if entry.confidence >= self.PRUNE_THRESHOLD:
                    surviving.append(entry)
            if surviving:
                self._spatial_memories[loc_id] = surviving
            else:
                del self._spatial_memories[loc_id]

        # Enforce max observations limit
        if len(self._observations) > self.max_observations:
            sorted_obs = sorted(
                self._observations.items(),
                key=lambda kv: kv[1].confidence,
            )
            excess = len(self._observations) - self.max_observations
            for sid, _ in sorted_obs[:excess]:
                if sid != self.owner_id:
                    del self._observations[sid]

    # -- Queries --

    def get_last_known(self, entity_id: str) -> Optional[ObservationEntry]:
        """Get last known state of an entity (may be stale)."""
        return self._observations.get(entity_id)

    def get_location_memory(self, location_id: str) -> List[SpatialMemoryEntry]:
        """Get what the agent remembers about a location."""
        return list(self._spatial_memories.get(location_id, []))

    def get_all_observations(self) -> Dict[str, ObservationEntry]:
        """Get all current observations."""
        return dict(self._observations)

    # -- Information sharing --

    def receive_info(
        self,
        from_agent_id: str,
        info: Dict[str, Any],
        round_number: int,
    ):
        """Record information shared by another agent.

        Shared info starts at lower confidence (secondhand knowledge).
        Only updates if the agent doesn't already have fresher direct info.
        """
        subject_id = info.get("subject_id", "")
        if not subject_id:
            return

        existing = self._observations.get(subject_id)
        if existing and existing.source == "direct" and existing.confidence > self.DEFAULT_TOLD_CONFIDENCE:
            return  # Already have better info

        self._observations[subject_id] = ObservationEntry(
            subject_id=subject_id,
            subject_name=info.get("subject_name", subject_id),
            subject_type=info.get("subject_type", "unknown"),
            properties=info.get("properties", {}),
            location=info.get("location"),
            round_observed=round_number,
            confidence=self.DEFAULT_TOLD_CONFIDENCE,
            source="told",
        )

        # Spatial memory from shared info
        loc = info.get("location")
        if loc:
            self._record_spatial(
                location_id=loc,
                entity_id=subject_id,
                entity_name=info.get("subject_name"),
                round_observed=round_number,
            )
            # Lower confidence for shared spatial info
            entries = self._spatial_memories.get(loc, [])
            for entry in entries:
                if entry.entity_id == subject_id and entry.round_observed == round_number:
                    entry.confidence = self.DEFAULT_TOLD_CONFIDENCE

    # -- Prompt generation --

    def summarize_for_prompt(
        self,
        current_perception: Dict[str, Any],
        max_entries: int = 10,
    ) -> str:
        """Generate a concise summary of world model for the LLM prompt.

        Only includes stale memories NOT currently visible in perception.
        Returns empty string if nothing interesting to report.
        """
        # IDs of entities currently visible
        currently_visible = set()
        for ve in current_perception.get("visible_entities", []):
            currently_visible.add(ve["id"])
        currently_visible.add(self.owner_id)

        # Collect stale observations not in current perception
        stale_entries = []
        for sid, entry in self._observations.items():
            if sid in currently_visible:
                continue
            if entry.confidence < self.PRUNE_THRESHOLD:
                continue
            stale_entries.append(entry)

        if not stale_entries:
            return ""

        # Sort by confidence descending (most reliable first), then by round
        stale_entries.sort(key=lambda e: (e.confidence, e.round_observed), reverse=True)
        stale_entries = stale_entries[:max_entries]

        lines = []
        for entry in stale_entries:
            # Format key properties
            props_str = ""
            if entry.properties:
                important = {k: v for k, v in entry.properties.items()
                             if isinstance(v, (int, float, str, bool))}
                if important:
                    props_parts = [f"{k}={v}" for k, v in list(important.items())[:4]]
                    props_str = f", {', '.join(props_parts)}"

            loc_str = f" at {entry.location}" if entry.location else ""
            source_tag = ", told by another" if entry.source == "told" else ""
            lines.append(
                f"- [R{entry.round_observed}, conf:{entry.confidence:.1f}{source_tag}] "
                f"{entry.subject_name} ({entry.subject_type}){loc_str}{props_str}"
            )

        return "\n".join(lines)

    # -- Serialization --

    def to_dict(self) -> dict:
        return {
            "owner_id": self.owner_id,
            "max_observations": self.max_observations,
            "confidence_decay_rate": self.confidence_decay_rate,
            "observations": {
                sid: entry.to_dict()
                for sid, entry in self._observations.items()
            },
            "spatial_memories": {
                loc: [e.to_dict() for e in entries]
                for loc, entries in self._spatial_memories.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentWorldModel":
        model = cls(
            owner_id=data["owner_id"],
            max_observations=data.get("max_observations", 200),
            confidence_decay_rate=data.get("confidence_decay_rate", 0.1),
        )
        for sid, entry_data in data.get("observations", {}).items():
            model._observations[sid] = ObservationEntry.from_dict(entry_data)
        for loc, entries_data in data.get("spatial_memories", {}).items():
            model._spatial_memories[loc] = [
                SpatialMemoryEntry.from_dict(e) for e in entries_data
            ]
        return model
