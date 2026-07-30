"""Tests for WorldState.to_dict / apply_snapshot / from_dict symmetry.

The env-builder agent inspects and replays worlds via snapshots, so
round-trip fidelity is critical: ``WorldState.from_dict(state.to_dict())``
must reconstruct semantically equivalent state.
"""
import pytest

from fg_env_kernel.action import ActionDefinition
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.resource import ResourceType
from fg_env_kernel.state import WorldState


@pytest.fixture
def populated_world():
    state = WorldState()
    # Schema
    state.entity_types["player"] = EntityType(name="player", role="agent")
    state.entity_types["resource_node"] = EntityType(name="resource_node", role="object")
    state.register_resource_type(ResourceType(name="gold"))
    state.action_definitions["work"] = ActionDefinition(
        name="work", description="work", actor_type="player",
    )

    # Entities
    state.entities["a"] = Entity(
        id="a", name="Alice", entity_type="player",
        properties={"score": 5, "level": 3, "alive": True},
        location_id="village",
    )
    state.entities["b"] = Entity(
        id="b", name="Bob", entity_type="player",
        properties={"score": 7, "level": 2, "alive": True},
        location_id="forest",
    )
    state.entities["mine"] = Entity(
        id="mine", name="Goldmine", entity_type="resource_node",
        properties={"reserves": 1000},
    )

    # Spatial state
    state.locations = {"a": "village", "b": "forest", "mine": "village"}
    state.spatial_index = {
        "village": {"a", "mine"},
        "forest": {"b"},
    }
    state.adjacency = {"village": ["forest"], "forest": ["village"]}

    # Resources
    state.resources["gold"].holdings = {"a": 100, "b": 50}

    # Temporal
    state.temporal.current_round = 7

    # Inventory
    from fg_env_kernel.inventory import Item
    state.inventory.add_item("a", Item(id="sword", name="Iron Sword", item_type="weapon"))

    # Action history with cooldowns
    state.action_history.record("a", "attack", True, 5)
    state.action_history.record("b", "defend", False, 6)
    state.action_history.set_cooldown("a", "attack", until_round=10)

    return state


class TestBasicRoundTrip:
    def test_temporal_round_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        assert restored.temporal.current_round == 7

    def test_entities_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        assert set(restored.entities.keys()) == {"a", "b", "mine"}
        assert restored.entities["a"].get("score") == 5
        assert restored.entities["b"].get("score") == 7
        assert restored.entities["mine"].get("reserves") == 1000

    def test_entity_alive_status_preserved(self, populated_world):
        populated_world.entities["a"].alive = False
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        assert restored.entities["a"].alive is False
        assert restored.entities["b"].alive is True

    def test_entity_location_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        assert restored.entities["a"].location_id == "village"
        assert restored.entities["b"].location_id == "forest"

    def test_resources_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        gold = restored.resources["gold"]
        assert gold.holdings == {"a": 100, "b": 50}

    def test_spatial_index_preserved(self, populated_world):
        """Critical bug from P1 was spatial_index being lost on snapshot.
        Verify the fix end-to-end."""
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        assert restored.spatial_index == {
            "village": {"a", "mine"},
            "forest": {"b"},
        }

    def test_locations_dict_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        assert restored.locations == {"a": "village", "b": "forest", "mine": "village"}

    def test_adjacency_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        assert restored.adjacency == {"village": ["forest"], "forest": ["village"]}


class TestSubsystemRoundTrip:
    def test_action_history_cooldowns_preserved(self, populated_world):
        """Cooldowns were lost on snapshot before P1 fix."""
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        # Cooldown should still block attack at round 8
        from fg_env_kernel.action import ActionDefinition
        action = ActionDefinition(name="attack", description="", actor_type="player")
        assert restored.action_history.is_available("a", "attack", action, round_num=8) is False
        # And be available at round 10
        assert restored.action_history.is_available("a", "attack", action, round_num=10) is True

    def test_action_history_records_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        recent_a = restored.action_history.get_recent("a", n=5)
        assert len(recent_a) == 1
        assert recent_a[0].action_name == "attack"
        assert recent_a[0].success is True
        recent_b = restored.action_history.get_recent("b", n=5)
        assert len(recent_b) == 1
        assert recent_b[0].success is False

    def test_inventory_preserved(self, populated_world):
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        items = restored.inventory.get_inventory("a")
        assert len(items) == 1
        assert items[0].id == "sword"
        assert items[0].name == "Iron Sword"


class TestPluginModuleRoundTrip:
    def test_plugin_module_state_restored(self, populated_world):
        """Custom subsystems registered via register_module survive
        round-trip when they implement to_dict + from_dict."""
        class CountingModule:
            def __init__(self, count=0):
                self.count = count
            def to_dict(self):
                return {"count": self.count}
            @classmethod
            def from_dict(cls, data):
                return cls(count=data.get("count", 0))

        populated_world.register_module("counter", CountingModule(count=42))
        snap = populated_world.to_dict()
        assert snap["plugin_modules"]["counter"] == {"count": 42}

        # Restore into a fresh state — must register a placeholder first
        # so the dispatcher knows which class to use.
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        restored.register_module("counter", CountingModule(count=0), replace=True)
        # Apply just the plugin slice again so the restorer can update it
        from fg_env_kernel.kernel_module import restore_plugin_modules
        restore_plugin_modules(restored.modules, snap["plugin_modules"])
        assert restored.get_module("counter").count == 42

    def test_plugin_without_from_dict_keeps_live_instance(self, populated_world):
        """A module that only has to_dict but no from_dict is left as-is
        on restore — better than blowing up."""
        class WriteOnly:
            def to_dict(self):
                return {"data": "snapshotted"}

        original = WriteOnly()
        populated_world.register_module("write_only", original)
        snap = populated_world.to_dict()
        assert "write_only" in snap["plugin_modules"]

        # Apply on a fresh state with a live instance
        new_state = WorldState()
        new_instance = WriteOnly()
        new_state.register_module("write_only", new_instance)
        from fg_env_kernel.kernel_module import restore_plugin_modules
        # Should not raise; live instance stays
        restore_plugin_modules(new_state.modules, snap["plugin_modules"])
        assert new_state.get_module("write_only") is new_instance


class TestApplySnapshotInPlace:
    def test_overlay_does_not_reset_schema(self, populated_world):
        """apply_snapshot must preserve existing schema registrations."""
        new_state = WorldState()
        # Pre-register schema (simulates a fresh sim with the same world definition)
        new_state.entity_types["player"] = populated_world.entity_types["player"]
        new_state.register_resource_type(populated_world.resource_types["gold"])
        new_state.entities["a"] = Entity(
            id="a", name="Alice", entity_type="player",
            properties={"score": 0, "level": 1, "alive": True},
        )
        new_state.entities["b"] = Entity(
            id="b", name="Bob", entity_type="player",
            properties={"score": 0, "level": 1, "alive": True},
        )

        snap = populated_world.to_dict()
        new_state.apply_snapshot(snap)

        # Mutable state restored
        assert new_state.entities["a"].get("score") == 5
        # Schema preserved
        assert "player" in new_state.entity_types

    def test_corrupt_section_does_not_abort_restore(self, populated_world):
        """A bad slice in one subsystem must not lose ALL state."""
        snap = populated_world.to_dict()
        # Corrupt the negotiations slice
        snap["negotiations"] = {"this is not valid": "garbage"}
        # Should NOT raise; other state should restore fine
        restored = WorldState.from_dict(snap, schema_provider=populated_world)
        # Entities still restored
        assert restored.entities["a"].get("score") == 5


class TestSchemaRequirement:
    def test_from_dict_without_schema_provider_still_works(self, populated_world):
        """Snapshots without a schema_provider produce a state where
        entities exist but entity_types are empty — caller's responsibility."""
        snap = populated_world.to_dict()
        restored = WorldState.from_dict(snap)
        # Entities reconstructed
        assert "a" in restored.entities
        # Schema empty — can't run the engine on this without schema
        assert restored.entity_types == {}
