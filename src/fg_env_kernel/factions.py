"""Factions -- groups of entities that share goals, relations, and identity.

Factions enable coalition gameplay, diplomacy, and collective action.
Entities belong to at most one faction at a time.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Faction:
    """A group of entities with shared identity."""
    id: str
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)  # faction-level stats
    member_ids: List[str] = field(default_factory=list)
    #: Hierarchy: the enclosing faction's id (a division inside a company,
    #: a squad inside an army). None = top-level. Membership stays exclusive
    #: at the leaf; ancestry is what `same_org` and org_root() traverse.
    parent: Optional[str] = None


class FactionManager:
    """Manages faction registration, membership, and queries."""

    def __init__(self):
        self._factions: Dict[str, Faction] = {}       # faction_id -> Faction
        self._membership: Dict[str, str] = {}          # entity_id -> faction_id

    def org_root(self, faction_id: str) -> Optional[str]:
        """The top-most ancestor of a faction (itself when top-level).

        Cycle-safe: a malformed parent loop returns the last unseen node
        rather than hanging."""
        seen: set = set()
        cur = faction_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            f = self._factions.get(cur)
            if f is None or f.parent is None:
                return cur
            cur = f.parent
        return cur if cur is None else faction_id

    def same_org(self, entity_a: str, entity_b: str) -> bool:
        """True when both entities' factions share a top-level ancestor —
        the hierarchical sibling of ``same_faction`` (which stays exact,
        so existing games keep their semantics)."""
        fa = self.get_entity_faction(entity_a)
        fb = self.get_entity_faction(entity_b)
        if not fa or not fb:
            return False
        ra, rb = self.org_root(fa), self.org_root(fb)
        return ra is not None and ra == rb

    def register(self, faction: Faction):
        """Register a faction. Also indexes any pre-set member_ids."""
        self._factions[faction.id] = faction
        for member_id in faction.member_ids:
            self._membership[member_id] = faction.id

    def add_member(self, entity_id: str, faction_id: str):
        """Add an entity to a faction. Removes from previous faction if any."""
        if faction_id not in self._factions:
            return
        # Remove from old faction
        old_faction_id = self._membership.get(entity_id)
        if old_faction_id and old_faction_id in self._factions:
            old_faction = self._factions[old_faction_id]
            if entity_id in old_faction.member_ids:
                old_faction.member_ids.remove(entity_id)

        self._membership[entity_id] = faction_id
        faction = self._factions[faction_id]
        if entity_id not in faction.member_ids:
            faction.member_ids.append(entity_id)

    def remove_member(self, entity_id: str):
        """Remove an entity from its faction."""
        faction_id = self._membership.pop(entity_id, None)
        if faction_id and faction_id in self._factions:
            faction = self._factions[faction_id]
            if entity_id in faction.member_ids:
                faction.member_ids.remove(entity_id)

    def get_faction(self, faction_id: str) -> Optional[Faction]:
        """Get a faction by ID."""
        return self._factions.get(faction_id)

    def faction_of(self, entity_id: str) -> Optional[str]:
        """Alias of :meth:`get_entity_faction` — the predicate layer calls
        ``faction_of`` behind a hasattr guard, so without this alias the
        same_faction/different_faction dict-ops silently returned False."""
        return self.get_entity_faction(entity_id)

    def get_entity_faction(self, entity_id: str) -> Optional[str]:
        """Get the faction ID an entity belongs to, or None."""
        return self._membership.get(entity_id)

    def get_members(self, faction_id: str) -> List[str]:
        """Get all member entity IDs for a faction."""
        faction = self._factions.get(faction_id)
        return list(faction.member_ids) if faction else []

    def are_allies(self, entity_a: str, entity_b: str) -> bool:
        """Check if two entities are in the same faction."""
        fa = self._membership.get(entity_a)
        fb = self._membership.get(entity_b)
        if fa is None or fb is None:
            return False
        return fa == fb

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        return {
            "factions": {
                fid: {
                    "id": f.id,
                    "name": f.name,
                    "description": f.description,
                    "properties": dict(f.properties),
                    "member_ids": list(f.member_ids),
                }
                for fid, f in self._factions.items()
            },
            "membership": dict(self._membership),
        }
