"""Resource types and pool tracking."""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ResourceType:
    """Definition of a resource kind."""
    name: str
    conservation: bool = True   # Total across system must remain constant
    discrete: bool = True       # Integer quantities only
    min_value: float = 0.0
    max_value: Optional[float] = None
    description: str = ""


@dataclass
class ResourcePool:
    """
    Tracks resource quantities held by entities.
    Key: entity_id, Value: quantity.
    Also tracks a system-level 'unallocated' pool for conservation.
    """
    resource_type: ResourceType
    holdings: Dict[str, float] = field(default_factory=dict)
    unallocated: float = 0.0

    @property
    def total(self) -> float:
        """Total resources in the system (holdings + unallocated)."""
        return sum(self.holdings.values()) + self.unallocated

    def get(self, entity_id: str) -> float:
        """Get amount held by an entity."""
        return self.holdings.get(entity_id, 0.0)

    def set(self, entity_id: str, amount: float) -> None:
        """Set absolute amount for an entity.

        IMPORTANT: this is a bootstrap / admin primitive — it does not
        enforce the conservation invariant on its own. For conserved
        resources, prefer `transfer`, `add`, `remove`. `set` is intended
        for initial state hydration (loading a schema's resource_holdings
        at sim start) and for direct admin overrides. It will pull the
        delta out of `unallocated`, which may go negative if you set
        more than the pool currently tracks (treat that as "minting").
        """
        if amount < self.resource_type.min_value:
            amount = self.resource_type.min_value
        if self.resource_type.max_value is not None and amount > self.resource_type.max_value:
            amount = self.resource_type.max_value
        if self.resource_type.discrete:
            amount = int(amount)

        old_amount = self.holdings.get(entity_id, 0.0)
        delta = amount - old_amount

        if self.resource_type.conservation:
            self.unallocated -= delta

        self.holdings[entity_id] = amount

    def transfer(self, from_id: str, to_id: str, amount: float) -> bool:
        """Move resources between entities. Returns True if successful.

        Refuses to take the sender below `min_value` or to overflow the
        recipient past `max_value`. Conservation is preserved trivially
        since the same amount leaves one holder and arrives at the other.
        """
        if amount <= 0 or from_id == to_id:
            return False
        if self.resource_type.discrete:
            amount = int(amount)

        available = self.holdings.get(from_id, 0.0)
        if available - amount < self.resource_type.min_value:
            return False

        recipient = self.holdings.get(to_id, 0.0)
        if self.resource_type.max_value is not None and recipient + amount > self.resource_type.max_value:
            return False

        self.holdings[from_id] = available - amount
        self.holdings[to_id] = recipient + amount
        return True

    def add(self, entity_id: str, amount: float) -> bool:
        """Add resources (from unallocated if conservation, or create if not)."""
        if self.resource_type.discrete:
            amount = int(amount)
        # A negative add would invert conservation accounting (it would
        # *mint* unallocated) and sidestep `remove`'s floor check. Reject it
        # — callers must use `remove` to decrease a holding.
        if amount < 0:
            return False
        new_holding = self.holdings.get(entity_id, 0.0) + amount
        if (self.resource_type.max_value is not None
                and new_holding > self.resource_type.max_value):
            return False
        if self.resource_type.conservation:
            if self.unallocated < amount:
                return False
            self.unallocated -= amount
        self.holdings[entity_id] = new_holding
        return True

    def remove(self, entity_id: str, amount: float) -> bool:
        """Remove resources from entity (return to unallocated if conservation)."""
        if self.resource_type.discrete:
            amount = int(amount)
        # A negative remove would *add* past the max bound and corrupt
        # conservation. Reject it — callers must use `add` to increase.
        if amount < 0:
            return False
        available = self.holdings.get(entity_id, 0.0)
        if available - amount < self.resource_type.min_value:
            return False
        self.holdings[entity_id] = available - amount
        if self.resource_type.conservation:
            self.unallocated += amount
        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "resource": self.resource_type.name,
            "holdings": dict(self.holdings),
            "unallocated": self.unallocated,
            "total": self.total,
        }
