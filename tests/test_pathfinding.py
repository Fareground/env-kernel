"""Tests for pathfinding, movement validation, and spatial preconditions.

In-memory tests -- no DB needed.
"""
from fg_env_kernel.pathfinding import Pathfinder
from fg_env_kernel.state import WorldState
from fg_env_kernel.entity import EntityType, Entity
from fg_env_kernel.action import (
    ActionDefinition, ActionInstance, Effect, EffectOperation,
    Precondition, Operator,
)
from fg_env_kernel.engine import SimulationEngine
from fg_env_kernel.temporal import TemporalModel, Phase
from fg_env_kernel.types import PropertySchema, PropertyType


# ---------------------------------------------------------------------------
# Unit tests for Pathfinder
# ---------------------------------------------------------------------------

class TestPathfinder:
    """Unit tests for the Pathfinder class."""

    def _simple_graph(self):
        """A -> B -> C -> D, A -> D (shortcut)."""
        return {
            "A": ["B", "D"],
            "B": ["A", "C"],
            "C": ["B", "D"],
            "D": ["C", "A"],
        }

    def test_find_path_direct(self):
        """find_path should return a direct path for adjacent locations."""
        adj = self._simple_graph()
        path = Pathfinder.find_path(adj, "A", "B")
        assert path == ["A", "B"]

    def test_find_path_multi_hop(self):
        """find_path should return the shortest multi-hop path."""
        adj = self._simple_graph()
        # A->B->C is 2 hops; A->D->C is also 2 hops
        path = Pathfinder.find_path(adj, "A", "C")
        assert path is not None
        assert len(path) == 3  # A -> ? -> C
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_find_path_unreachable(self):
        """find_path should return None for unreachable locations."""
        adj = {"A": ["B"], "B": ["A"], "X": ["Y"], "Y": ["X"]}
        path = Pathfinder.find_path(adj, "A", "X")
        assert path is None

    def test_find_path_with_costs(self):
        """find_path with costs should prefer cheaper routes."""
        # A -> B -> C (cost: 1 + 1 = 2)
        # A -> D -> C (cost: 1 + 10 = 11)
        adj = {
            "A": ["B", "D"],
            "B": ["A", "C"],
            "C": ["B", "D"],
            "D": ["A", "C"],
        }
        costs = {"B": 1.0, "C": 1.0, "D": 10.0}
        path = Pathfinder.find_path(adj, "A", "C", costs=costs)
        assert path == ["A", "B", "C"]

    def test_find_path_same_start_goal(self):
        """find_path with same start and goal should return [start]."""
        adj = self._simple_graph()
        path = Pathfinder.find_path(adj, "A", "A")
        assert path == ["A"]

    def test_find_all_reachable(self):
        """find_all_reachable should return all connected locations with distances."""
        adj = self._simple_graph()
        reachable = Pathfinder.find_all_reachable(adj, "A")
        assert reachable["A"] == 0
        assert reachable["B"] == 1
        assert reachable["D"] == 1
        # C is 2 hops from A via B or D
        assert reachable["C"] == 2

    def test_find_all_reachable_max_distance(self):
        """find_all_reachable with max_distance should limit results."""
        adj = self._simple_graph()
        reachable = Pathfinder.find_all_reachable(adj, "A", max_distance=1)
        assert "A" in reachable
        assert "B" in reachable
        assert "D" in reachable
        assert "C" not in reachable  # 2 hops away

    def test_distance_adjacent(self):
        """distance should return 1 for adjacent locations."""
        adj = self._simple_graph()
        assert Pathfinder.distance(adj, "A", "B") == 1

    def test_distance_multi_hop(self):
        """distance should return the shortest hop count."""
        adj = self._simple_graph()
        assert Pathfinder.distance(adj, "A", "C") == 2

    def test_distance_unreachable(self):
        """distance should return -1 for unreachable locations."""
        adj = {"A": ["B"], "B": ["A"], "X": ["Y"], "Y": ["X"]}
        assert Pathfinder.distance(adj, "A", "X") == -1

    def test_distance_same_location(self):
        """distance from a location to itself should be 0."""
        adj = self._simple_graph()
        assert Pathfinder.distance(adj, "A", "A") == 0

    def test_are_adjacent(self):
        """are_adjacent should check direct neighbor relationship."""
        adj = self._simple_graph()
        assert Pathfinder.are_adjacent(adj, "A", "B") is True
        assert Pathfinder.are_adjacent(adj, "A", "C") is False

    def test_disconnected_graph(self):
        """Pathfinder should handle disconnected graphs correctly."""
        adj = {"island1": ["island2"], "island2": ["island1"], "mainland": []}
        assert Pathfinder.find_path(adj, "island1", "mainland") is None
        reachable = Pathfinder.find_all_reachable(adj, "island1")
        assert "mainland" not in reachable


# ---------------------------------------------------------------------------
# Helper to build a state for engine integration tests
# ---------------------------------------------------------------------------

def _build_spatial_state():
    """Build a world state with locations and adjacency for movement tests."""
    state = WorldState()
    state.temporal = TemporalModel(phases=[Phase(name="action")])

    state.register_entity_type(EntityType(
        name="adventurer",
        role="agent",
        properties=[
            PropertySchema(name="health", type=PropertyType.FLOAT, default=100.0),
        ],
    ))

    alice = Entity(id="alice", name="Alice", entity_type="adventurer",
                   properties={"health": 100.0}, location_id="village")
    bob = Entity(id="bob", name="Bob", entity_type="adventurer",
                 properties={"health": 100.0}, location_id="forest")
    state.spawn_entity(alice)
    state.spawn_entity(bob)

    # Set up spatial tracking
    state.locations["alice"] = "village"
    state.locations["bob"] = "forest"
    state.spatial_index = {
        "village": {"alice"},
        "forest": {"bob"},
        "cave": set(),
        "mountain": set(),
    }

    # Adjacency: village <-> forest <-> cave <-> mountain
    state.adjacency = {
        "village": ["forest"],
        "forest": ["village", "cave"],
        "cave": ["forest", "mountain"],
        "mountain": ["cave"],
    }

    return state


# ---------------------------------------------------------------------------
# Engine integration tests for movement
# ---------------------------------------------------------------------------

class TestMovementEngine:
    """Tests for movement validation in the engine."""

    def test_move_to_adjacent_location(self):
        """MOVE_TO effect should succeed for adjacent locations."""
        state = _build_spatial_state()
        state.register_action(ActionDefinition(
            name="move",
            description="Move to a location",
            actor_type="adventurer",
            resolution_archetype="deterministic",
            effects_on_success=[
                Effect(target="actor", operation=EffectOperation.MOVE_TO, value="forest"),
            ],
        ))

        def decision_fn(eid, perception, valid_actions):
            if eid == "alice" and "move" in valid_actions:
                return ActionInstance(action_name="move", actor_id="alice")
            return None

        engine = SimulationEngine(state=state, decision_fn=decision_fn, max_rounds=1, seed=42)
        engine.run()

        # Alice should have moved from village to forest
        assert state.locations["alice"] == "forest"
        assert "alice" in state.spatial_index["forest"]
        assert "alice" not in state.spatial_index["village"]

    def test_move_to_non_adjacent_blocked(self):
        """MOVE_TO effect should be blocked for non-adjacent locations."""
        state = _build_spatial_state()
        state.register_action(ActionDefinition(
            name="teleport",
            description="Try to jump to cave",
            actor_type="adventurer",
            resolution_archetype="deterministic",
            effects_on_success=[
                Effect(target="actor", operation=EffectOperation.MOVE_TO, value="cave"),
            ],
        ))

        def decision_fn(eid, perception, valid_actions):
            if eid == "alice" and "teleport" in valid_actions:
                return ActionInstance(action_name="teleport", actor_id="alice")
            return None

        engine = SimulationEngine(state=state, decision_fn=decision_fn, max_rounds=1, seed=42)
        engine.run()

        # Alice should NOT have moved (village is not adjacent to cave)
        assert state.locations["alice"] == "village"
        assert "alice" in state.spatial_index["village"]

    def test_move_updates_spatial_index(self):
        """MOVE_TO should correctly update the spatial index."""
        state = _build_spatial_state()
        state.register_action(ActionDefinition(
            name="move",
            description="Move to adjacent location",
            actor_type="adventurer",
            resolution_archetype="deterministic",
            effects_on_success=[
                Effect(target="actor", operation=EffectOperation.MOVE_TO, value="cave"),
            ],
        ))

        def decision_fn(eid, perception, valid_actions):
            if eid == "bob" and "move" in valid_actions:
                return ActionInstance(action_name="move", actor_id="bob")
            return None

        engine = SimulationEngine(state=state, decision_fn=decision_fn, max_rounds=1, seed=42)
        engine.run()

        # Bob moved from forest to cave
        assert state.locations["bob"] == "cave"
        assert "bob" in state.spatial_index["cave"]
        assert "bob" not in state.spatial_index["forest"]

    def test_empty_adjacency_allows_any_movement(self):
        """When no adjacency is defined, MOVE_TO should always succeed (backward compat)."""
        state = _build_spatial_state()
        state.adjacency = {}  # No adjacency constraints

        state.register_action(ActionDefinition(
            name="teleport",
            description="Move anywhere",
            actor_type="adventurer",
            resolution_archetype="deterministic",
            effects_on_success=[
                Effect(target="actor", operation=EffectOperation.MOVE_TO, value="mountain"),
            ],
        ))

        def decision_fn(eid, perception, valid_actions):
            if eid == "alice" and "teleport" in valid_actions:
                return ActionInstance(action_name="teleport", actor_id="alice")
            return None

        engine = SimulationEngine(state=state, decision_fn=decision_fn, max_rounds=1, seed=42)
        engine.run()

        # With no adjacency, movement to any location should work
        assert state.locations["alice"] == "mountain"


# ---------------------------------------------------------------------------
# Precondition tests
# ---------------------------------------------------------------------------

class TestSpatialPreconditions:
    """Tests for AT_LOCATION and IS_ADJACENT preconditions."""

    def test_at_location_passes(self):
        """AT_LOCATION precondition should pass when entity is at the correct location."""
        state = _build_spatial_state()
        state.register_action(ActionDefinition(
            name="mine",
            description="Mine ore (only in cave)",
            actor_type="adventurer",
            resolution_archetype="deterministic",
            preconditions=[
                Precondition(subject="actor", operator=Operator.AT_LOCATION, value="forest"),
            ],
            effects_on_success=[
                Effect(target="actor", operation=EffectOperation.ADD, field="health", value=5.0),
            ],
        ))

        # Bob is at forest, so AT_LOCATION("forest") should pass
        valid = state.get_valid_actions("bob")
        assert "mine" in valid

    def test_at_location_fails(self):
        """AT_LOCATION precondition should fail when entity is at wrong location."""
        state = _build_spatial_state()
        state.register_action(ActionDefinition(
            name="mine",
            description="Mine ore (only in cave)",
            actor_type="adventurer",
            resolution_archetype="deterministic",
            preconditions=[
                Precondition(subject="actor", operator=Operator.AT_LOCATION, value="cave"),
            ],
        ))

        # Alice is at village, not cave
        valid = state.get_valid_actions("alice")
        assert "mine" not in valid

    def test_is_adjacent_passes(self):
        """IS_ADJACENT precondition should pass for adjacent entities."""
        state = _build_spatial_state()
        state.register_action(ActionDefinition(
            name="attack",
            description="Attack nearby target",
            actor_type="adventurer",
            target_type="adventurer",
            resolution_archetype="deterministic",
            preconditions=[
                Precondition(subject="actor", operator=Operator.IS_ADJACENT),
            ],
            effects_on_success=[
                Effect(target="target", operation=EffectOperation.SUBTRACT, field="health", value=10.0),
            ],
        ))

        def decision_fn(eid, perception, valid_actions):
            if eid == "alice" and "attack" in valid_actions:
                return ActionInstance(action_name="attack", actor_id="alice", target_id="bob")
            return None

        engine = SimulationEngine(state=state, decision_fn=decision_fn, max_rounds=1, seed=42)
        engine.run()

        # Alice (village) and Bob (forest) are adjacent, so attack should succeed
        assert state.get_entity("bob").get("health") == 90.0

    def test_is_adjacent_fails(self):
        """IS_ADJACENT precondition should fail for non-adjacent entities."""
        state = _build_spatial_state()
        # Move bob to mountain (not adjacent to village)
        state.locations["bob"] = "mountain"
        state.spatial_index["forest"].discard("bob")
        state.spatial_index["mountain"].add("bob")

        state.register_action(ActionDefinition(
            name="attack",
            description="Attack nearby target",
            actor_type="adventurer",
            target_type="adventurer",
            resolution_archetype="deterministic",
            preconditions=[
                Precondition(subject="actor", operator=Operator.IS_ADJACENT),
            ],
            effects_on_success=[
                Effect(target="target", operation=EffectOperation.SUBTRACT, field="health", value=10.0),
            ],
        ))

        def decision_fn(eid, perception, valid_actions):
            if eid == "alice" and "attack" in valid_actions:
                return ActionInstance(action_name="attack", actor_id="alice", target_id="bob")
            return None

        engine = SimulationEngine(state=state, decision_fn=decision_fn, max_rounds=1, seed=42)
        engine.run()

        # Alice (village) and Bob (mountain) are NOT adjacent
        # The action_failed event should be emitted, bob's health unchanged
        assert state.get_entity("bob").get("health") == 100.0
