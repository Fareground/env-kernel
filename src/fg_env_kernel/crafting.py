"""Crafting and recipe system -- transform items and resources into new items.

Entities can craft items using recipes that consume ingredients (items and/or
resources) and produce outputs (new items and/or resources). Enables item
production, transformation chains, and economy simulation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RecipeInput:
    """An item input required for a recipe."""
    item_type: str                     # Required item type (consumed)
    quantity: int = 1


@dataclass
class RecipeResourceInput:
    """A resource input required for a recipe."""
    resource: str                      # Resource name
    amount: float = 1.0


@dataclass
class RecipeOutput:
    """An item produced by a recipe."""
    item_id_prefix: str                # Generated item ID = prefix + "_" + counter
    item_name: str
    item_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    stackable: bool = False
    quantity: int = 1


@dataclass
class RecipeResourceOutput:
    """A resource produced by a recipe."""
    resource: str
    amount: float = 1.0


@dataclass
class Recipe:
    """Complete recipe definition."""
    name: str
    description: str = ""
    inputs: List[RecipeInput] = field(default_factory=list)
    resource_inputs: List[RecipeResourceInput] = field(default_factory=list)
    outputs: List[RecipeOutput] = field(default_factory=list)
    resource_outputs: List[RecipeResourceOutput] = field(default_factory=list)
    required_skill: Optional[str] = None       # Skill name
    required_skill_level: float = 0.0
    skill_xp_award: Optional[str] = None       # Skill to grant XP on craft
    skill_xp_amount: float = 1.0
    location_required: Optional[str] = None    # Must be at this location


class RecipeManager:
    """Manages recipe registration, availability checks, and crafting execution."""

    def __init__(self):
        self._recipes: Dict[str, Recipe] = {}  # recipe_name -> Recipe
        self._craft_counter: int = 0           # Auto-increment for unique item IDs

    def register(self, recipe: Recipe):
        """Register a recipe."""
        self._recipes[recipe.name] = recipe

    def get(self, recipe_name: str) -> Optional[Recipe]:
        """Get a recipe by name."""
        return self._recipes.get(recipe_name)

    def get_all(self) -> List[Recipe]:
        """Get all registered recipes."""
        return list(self._recipes.values())

    def can_craft(self, entity_id: str, recipe_name: str, state) -> bool:
        """Check if an entity can craft a recipe right now.

        Checks:
        1. Recipe exists
        2. Entity has all required item inputs (by type + quantity)
        3. Entity has all required resource inputs
        4. Entity meets skill requirement (if any)
        5. Entity is at required location (if any)
        """
        recipe = self._recipes.get(recipe_name)
        if not recipe:
            return False

        # Check item inputs
        for inp in recipe.inputs:
            available = state.inventory.count_item_type(entity_id, inp.item_type)
            if available < inp.quantity:
                return False

        # Check resource inputs
        for rinp in recipe.resource_inputs:
            pool = state.resources.get(rinp.resource)
            if not pool:
                return False
            if pool.get(entity_id) < rinp.amount:
                return False

        # Check skill requirement
        if recipe.required_skill:
            skill_level = state.skills.get_level(entity_id, recipe.required_skill)
            if skill_level < recipe.required_skill_level:
                return False

        # Check location requirement
        if recipe.location_required:
            entity_loc = state.locations.get(entity_id)
            if entity_loc != recipe.location_required:
                return False

        return True

    def get_available(self, entity_id: str, state) -> List[str]:
        """Return names of all recipes the entity can currently craft."""
        return [
            name for name in self._recipes
            if self.can_craft(entity_id, name, state)
        ]

    def craft(self, entity_id: str, recipe_name: str, state) -> Optional[dict]:
        """Execute a recipe: consume inputs, produce outputs.

        Returns event dict with details, or None if cannot craft.
        """
        if not self.can_craft(entity_id, recipe_name, state):
            return None

        recipe = self._recipes[recipe_name]

        # 1. Consume item inputs
        consumed_items = []
        for inp in recipe.inputs:
            consumed = state.inventory.consume_item_type(entity_id, inp.item_type, inp.quantity)
            consumed_items.extend(consumed)

        # 2. Consume resource inputs
        for rinp in recipe.resource_inputs:
            pool = state.resources.get(rinp.resource)
            if pool:
                current = pool.get(entity_id)
                pool.set(entity_id, current - rinp.amount)

        # 3. Create output items
        from .inventory import Item
        produced_items = []
        for output in recipe.outputs:
            self._craft_counter += 1
            item_id = f"{output.item_id_prefix}_{self._craft_counter}"
            item = Item(
                id=item_id,
                name=output.item_name,
                item_type=output.item_type,
                properties=dict(output.properties),
                stackable=output.stackable,
                quantity=output.quantity,
            )
            state.inventory.add_item(entity_id, item)
            produced_items.append({"id": item_id, "name": output.item_name, "type": output.item_type})

        # 4. Award resource outputs
        for rout in recipe.resource_outputs:
            pool = state.resources.get(rout.resource)
            if pool:
                current = pool.get(entity_id)
                pool.set(entity_id, current + rout.amount)

        # 5. Award skill XP
        level_up_info = None
        if recipe.skill_xp_award:
            level_up_info = state.skills.award_xp(entity_id, recipe.skill_xp_award, recipe.skill_xp_amount)

        return {
            "recipe": recipe_name,
            "entity_id": entity_id,
            "consumed_items": [{"type": inp.item_type, "quantity": inp.quantity} for inp in recipe.inputs],
            "consumed_resources": [{"resource": rinp.resource, "amount": rinp.amount} for rinp in recipe.resource_inputs],
            "produced_items": produced_items,
            "produced_resources": [{"resource": rout.resource, "amount": rout.amount} for rout in recipe.resource_outputs],
            "level_up": level_up_info,
        }

    def to_dict(self) -> dict:
        """Serialize for snapshots."""
        return {
            name: {
                "description": r.description,
                "inputs": [{"item_type": i.item_type, "quantity": i.quantity} for i in r.inputs],
                "resource_inputs": [{"resource": ri.resource, "amount": ri.amount} for ri in r.resource_inputs],
                "outputs": [{"item_type": o.item_type, "name": o.item_name} for o in r.outputs],
                "resource_outputs": [{"resource": ro.resource, "amount": ro.amount} for ro in r.resource_outputs],
                "required_skill": r.required_skill,
                "required_skill_level": r.required_skill_level,
            }
            for name, r in self._recipes.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecipeManager":
        """Restore from a to_dict snapshot. Note: ``outputs`` snapshot
        only captures item_type+name — properties/stackable/quantity
        revert to defaults. Use the live schema source-of-truth to
        re-register recipes with full fidelity when available."""
        mgr = cls()
        for name, r in (data or {}).items():
            inputs = [
                RecipeInput(item_type=i["item_type"], quantity=i.get("quantity", 1))
                for i in r.get("inputs", [])
            ]
            resource_inputs = [
                RecipeResourceInput(resource=ri["resource"], amount=ri.get("amount", 1.0))
                for ri in r.get("resource_inputs", [])
            ]
            outputs = [
                RecipeOutput(
                    item_id_prefix=o.get("item_type", "item"),
                    item_name=o.get("name", o.get("item_type", "item")),
                    item_type=o.get("item_type", ""),
                )
                for o in r.get("outputs", [])
            ]
            resource_outputs = [
                RecipeResourceOutput(resource=ro["resource"], amount=ro.get("amount", 1.0))
                for ro in r.get("resource_outputs", [])
            ]
            mgr._recipes[name] = Recipe(
                name=name,
                description=r.get("description", ""),
                inputs=inputs,
                resource_inputs=resource_inputs,
                outputs=outputs,
                resource_outputs=resource_outputs,
                required_skill=r.get("required_skill"),
                required_skill_level=r.get("required_skill_level", 0.0),
            )
        return mgr
