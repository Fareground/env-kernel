"""Integration tests for the KernelModule lifecycle protocol.

Proves that an env-builder can plug a custom subsystem into WorldState
and have it participate automatically in:

  - despawn cleanup        (on_entity_despawn)
  - snapshot serialization (to_dict)

without modifying any kernel code.
"""
import pytest

from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.kernel_module import (
    KernelModule,
    collect_snapshots,
    dispatch_despawn,
    dispatch_round_start,
    dispatch_spawn,
)
from fg_env_kernel.state import WorldState


class TrackedReferenceModule:
    """Toy module that holds entity-keyed state and cleans up on despawn."""
    def __init__(self):
        self.refs: dict = {}
        self.despawn_log: list = []

    def add(self, entity_id: str, value: str) -> None:
        self.refs[entity_id] = value

    def on_entity_despawn(self, entity_id: str) -> None:
        self.despawn_log.append(entity_id)
        self.refs.pop(entity_id, None)

    def to_dict(self) -> dict:
        return {"refs": dict(self.refs), "despawn_log": list(self.despawn_log)}


class TickingModule:
    """Toy module that counts on_round_start invocations."""
    def __init__(self):
        self.rounds_seen: list = []

    def on_round_start(self, state, round_number: int) -> None:
        self.rounds_seen.append(round_number)

    def to_dict(self) -> dict:
        return {"rounds_seen": list(self.rounds_seen)}


@pytest.fixture
def world():
    state = WorldState()
    state.entity_types["player"] = EntityType(name="player", role="agent")
    state.entities["a"] = Entity(
        id="a", name="Alice", entity_type="player",
        properties={"gold": 50, "alive": True},
    )
    state.entities["b"] = Entity(
        id="b", name="Bob", entity_type="player",
        properties={"gold": 100, "alive": True},
    )
    return state


class TestBuiltinAutoRegistration:
    def test_named_managers_appear_in_modules_dict(self, world):
        """Every built-in subsystem is reachable via state.modules."""
        for key in ("factions", "inventory", "goals", "skills", "polls",
                    "negotiations", "plans", "roles", "messages",
                    "sequences", "status_effects", "recipes",
                    "location_properties", "action_history", "relations"):
            assert world.get_module(key) is not None, f"missing module: {key}"

    def test_modules_dict_holds_same_instance_as_named_field(self, world):
        """state.factions IS state.modules['factions'] — single instance."""
        assert world.modules["factions"] is world.factions
        assert world.modules["inventory"] is world.inventory


class TestCustomModuleRegistration:
    def test_register_module_makes_it_visible(self, world):
        mod = TrackedReferenceModule()
        world.register_module("my_subsystem", mod)
        assert world.get_module("my_subsystem") is mod

    def test_duplicate_registration_rejected(self, world):
        world.register_module("dup", TrackedReferenceModule())
        with pytest.raises(ValueError, match="already registered"):
            world.register_module("dup", TrackedReferenceModule())

    def test_replace_flag_overrides(self, world):
        a = TrackedReferenceModule()
        b = TrackedReferenceModule()
        world.register_module("repl", a)
        world.register_module("repl", b, replace=True)
        assert world.get_module("repl") is b

    def test_unregister_module(self, world):
        mod = TrackedReferenceModule()
        world.register_module("temp", mod)
        removed = world.unregister_module("temp")
        assert removed is mod
        assert world.get_module("temp") is None


class TestDespawnDispatch:
    def test_custom_module_on_entity_despawn_invoked(self, world):
        """Custom modules clean up automatically when an entity despawns."""
        mod = TrackedReferenceModule()
        mod.add("a", "alice_data")
        mod.add("b", "bob_data")
        world.register_module("tracker", mod)

        world.despawn_entity("a")

        assert mod.despawn_log == ["a"]
        assert "a" not in mod.refs
        assert mod.refs.get("b") == "bob_data"

    def test_buggy_module_does_not_crash_despawn(self, world):
        """An exception in one module's hook must not break the pipeline."""
        class BrokenModule:
            def on_entity_despawn(self, entity_id):
                raise RuntimeError("boom")
            def to_dict(self):
                return {}

        good = TrackedReferenceModule()
        good.add("a", "x")
        world.register_module("broken", BrokenModule())
        world.register_module("good", good)

        # Should not raise
        world.despawn_entity("a")
        # Good module still got the hook
        assert good.despawn_log == ["a"]


class TestSnapshotInclusion:
    def test_plugin_module_appears_in_state_snapshot(self, world):
        mod = TrackedReferenceModule()
        mod.add("a", "alice")
        world.register_module("custom_subsystem", mod)

        snap = world.to_dict()
        assert "plugin_modules" in snap
        assert "custom_subsystem" in snap["plugin_modules"]
        assert snap["plugin_modules"]["custom_subsystem"]["refs"] == {"a": "alice"}

    def test_builtin_modules_not_duplicated_under_plugin_modules(self, world):
        """Factions/inventory etc. live at top level, not also under
        plugin_modules — avoids redundant serialization."""
        snap = world.to_dict()
        assert "factions" in snap
        assert "factions" not in snap.get("plugin_modules", {})

    def test_module_with_no_to_dict_is_skipped(self):
        """collect_snapshots ignores modules that don't implement to_dict."""
        class BareModule:
            pass
        mods = {"bare": BareModule(), "good": TrackedReferenceModule()}
        out = collect_snapshots(mods)
        assert "bare" not in out
        assert "good" in out


class TestDispatcherHelpers:
    def test_dispatch_spawn(self):
        class M:
            def __init__(self):
                self.spawned = []
            def on_entity_spawn(self, eid):
                self.spawned.append(eid)
        m = M()
        dispatch_spawn({"m": m}, "a")
        assert m.spawned == ["a"]

    def test_dispatch_round_start(self):
        m = TickingModule()
        dispatch_round_start({"m": m}, state=None, round_number=3)
        dispatch_round_start({"m": m}, state=None, round_number=4)
        assert m.rounds_seen == [3, 4]

    def test_legacy_method_aliases_recognized(self):
        """Modules that used legacy names like remove_member / clear_entity
        are still picked up by dispatch_despawn for backwards compat."""
        class LegacyA:
            def __init__(self): self.cleared = []
            def remove_member(self, eid): self.cleared.append(eid)
            def to_dict(self): return {}

        class LegacyB:
            def __init__(self): self.cleared = []
            def clear_entity(self, eid): self.cleared.append(eid)
            def to_dict(self): return {}

        a, b = LegacyA(), LegacyB()
        dispatch_despawn({"a": a, "b": b}, "x")
        # Each got their alias method invoked
        assert a.cleared == ["x"]
        assert b.cleared == ["x"]
