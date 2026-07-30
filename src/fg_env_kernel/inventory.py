"""Inventory system -- items, carrying, dropping, and transferring.

Entities can hold items in their inventory, pick up items from the ground,
drop items, and transfer items to other entities.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Item:
    """A discrete item that can be held by an entity or found on the ground."""
    id: str
    name: str
    item_type: str                          # "weapon", "resource", "tool", etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    stackable: bool = False
    quantity: int = 1


class InventoryManager:
    """Manages entity inventories and ground items at locations."""

    def __init__(self):
        self._inventories: Dict[str, List[Item]] = {}       # entity_id -> items
        self._ground_items: Dict[str, List[Item]] = {}      # location_id -> items
        self._capacity: Dict[str, int] = {}                  # entity_id -> max slots

    def set_capacity(self, entity_id: str, capacity: int):
        """Set the max inventory size for an entity."""
        self._capacity[entity_id] = capacity

    def get_capacity(self, entity_id: str) -> Optional[int]:
        """Get the max inventory size for an entity. None = unlimited."""
        return self._capacity.get(entity_id)

    def add_item(self, entity_id: str, item: Item) -> bool:
        """Add an item to an entity's inventory. Returns False if full.

        If the item is stackable and there's already one of the same ID,
        increases the quantity instead of adding a new slot.
        """
        inv = self._inventories.setdefault(entity_id, [])

        # Check for stackable merge
        if item.stackable:
            for existing in inv:
                if existing.id == item.id:
                    existing.quantity += item.quantity
                    return True

        # Check capacity
        cap = self._capacity.get(entity_id)
        if cap is not None and len(inv) >= cap:
            return False

        inv.append(item)
        return True

    def remove_item(self, entity_id: str, item_id: str) -> Optional[Item]:
        """Remove an item from inventory by ID. Returns the item or None."""
        inv = self._inventories.get(entity_id, [])
        for i, item in enumerate(inv):
            if item.id == item_id:
                return inv.pop(i)
        return None

    def transfer_item(self, from_id: str, to_id: str, item_id: str) -> bool:
        """Transfer an item from one entity to another. Returns True on success."""
        item = self.remove_item(from_id, item_id)
        if item is None:
            return False
        if not self.add_item(to_id, item):
            # Add back if transfer failed (target full)
            self.add_item(from_id, item)
            return False
        return True

    def drop_item(self, entity_id: str, item_id: str, location: str) -> bool:
        """Drop an item from inventory to the ground at a location."""
        item = self.remove_item(entity_id, item_id)
        if item is None:
            return False
        ground = self._ground_items.setdefault(location, [])
        ground.append(item)
        return True

    def pickup_item(self, entity_id: str, item_id: str, location: str) -> bool:
        """Pick up an item from the ground at a location into inventory."""
        ground = self._ground_items.get(location, [])
        for i, item in enumerate(ground):
            if item.id == item_id:
                if self.add_item(entity_id, item):
                    ground.pop(i)
                    return True
                return False  # Inventory full
        return False  # Item not on ground

    def get_inventory(self, entity_id: str) -> List[Item]:
        """Get all items in an entity's inventory."""
        return list(self._inventories.get(entity_id, []))

    def get_ground_items(self, location: str) -> List[Item]:
        """Get all items on the ground at a location."""
        return list(self._ground_items.get(location, []))

    def has_item(self, entity_id: str, item_id: str) -> bool:
        """Check if an entity has a specific item by ID."""
        return any(it.id == item_id for it in self._inventories.get(entity_id, []))

    def has_item_type(self, entity_id: str, item_type: str) -> bool:
        """Check if an entity has any item of a specific type."""
        return any(it.item_type == item_type for it in self._inventories.get(entity_id, []))

    def count_items(self, entity_id: str) -> int:
        """Count the number of distinct item slots in inventory."""
        return len(self._inventories.get(entity_id, []))

    def count_item_type(self, entity_id: str, item_type: str) -> int:
        """Count total quantity of items of a given type (respects stackable quantities)."""
        total = 0
        for item in self._inventories.get(entity_id, []):
            if item.item_type == item_type:
                total += item.quantity
        return total

    def consume_item_type(self, entity_id: str, item_type: str, quantity: int = 1) -> "list[Item]":
        """Remove N items of given type. Returns consumed items, or empty list if not enough.

        For stackable items, reduces quantity. For non-stackable, removes individual items.
        Does NOT partially consume — either all requested quantity is available or nothing is consumed.
        """
        inv = self._inventories.get(entity_id, [])
        available = self.count_item_type(entity_id, item_type)
        if available < quantity:
            return []

        consumed = []
        remaining = quantity

        # Work on a copy of indices to avoid mutation issues
        i = 0
        while i < len(inv) and remaining > 0:
            item = inv[i]
            if item.item_type != item_type:
                i += 1
                continue
            if item.quantity <= remaining:
                # Consume entire item/stack
                remaining -= item.quantity
                consumed.append(inv.pop(i))
            else:
                # Partially consume stack
                item.quantity -= remaining
                partial = Item(
                    id=item.id,
                    name=item.name,
                    item_type=item.item_type,
                    properties=dict(item.properties),
                    stackable=item.stackable,
                    quantity=remaining,
                )
                consumed.append(partial)
                remaining = 0
                i += 1

        return consumed

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        return {
            "inventories": {
                eid: [
                    {
                        "id": it.id,
                        "name": it.name,
                        "item_type": it.item_type,
                        "properties": dict(it.properties),
                        "stackable": it.stackable,
                        "quantity": it.quantity,
                    }
                    for it in items
                ]
                for eid, items in self._inventories.items()
            },
            "ground_items": {
                loc: [
                    {
                        "id": it.id,
                        "name": it.name,
                        "item_type": it.item_type,
                        "properties": dict(it.properties),
                        "stackable": it.stackable,
                        "quantity": it.quantity,
                    }
                    for it in items
                ]
                for loc, items in self._ground_items.items()
            },
            "capacity": dict(self._capacity),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InventoryManager":
        """Restore from a snapshot. Symmetric to ``to_dict``."""
        mgr = cls()
        def _item(d: dict) -> Item:
            return Item(
                id=d["id"],
                name=d.get("name", d["id"]),
                item_type=d.get("item_type", ""),
                properties=dict(d.get("properties", {})),
                stackable=bool(d.get("stackable", False)),
                quantity=int(d.get("quantity", 1)),
            )
        for eid, items in (data.get("inventories") or {}).items():
            mgr._inventories[eid] = [_item(d) for d in items]
        for loc, items in (data.get("ground_items") or {}).items():
            mgr._ground_items[loc] = [_item(d) for d in items]
        for eid, cap in (data.get("capacity") or {}).items():
            mgr._capacity[eid] = int(cap)
        return mgr
