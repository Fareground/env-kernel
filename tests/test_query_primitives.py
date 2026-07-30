"""Tests for the state-query + spatial expression functions.

These primitives are what unlocks "any game" from pure JSON. Without
$count / $entities_at / $within_range / $distance, even simple
multi-entity mechanics (AoE attacks, line-of-sight checks, "is there
a piece between me and the target") require Python.
"""
import pytest

from fg_env_kernel.effects import resolve_expression
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.state import WorldState


@pytest.fixture
def world():
    state = WorldState()
    state.entity_types["Player"] = EntityType(name="Player", role="agent")
    state.entity_types["Mob"] = EntityType(name="Mob", role="object")
    # 3 alive players, 1 dead player, 2 mobs
    for i, alive in enumerate([True, True, True, False]):
        state.entities[f"p{i}"] = Entity(
            id=f"p{i}", name=f"P{i}", entity_type="Player",
            properties={"score": i * 10, "ready": i % 2 == 0}, alive=alive,
            location_id=f"loc{i % 2}",
        )
        state.locations[f"p{i}"] = f"loc{i % 2}"
    for i in range(2):
        state.entities[f"m{i}"] = Entity(
            id=f"m{i}", name=f"M{i}", entity_type="Mob",
            properties={}, location_id="loc2",
        )
        state.locations[f"m{i}"] = "loc2"
    # adjacency: loc0 -- loc1 -- loc2
    state.adjacency = {"loc0": ["loc1"], "loc1": ["loc0", "loc2"], "loc2": ["loc1"]}
    return state


class TestStateQueries:
    def test_count_all(self, world):
        assert resolve_expression("$count(Player)", state=world) == 4
        assert resolve_expression("$count(Mob)", state=world) == 2

    def test_count_alive(self, world):
        assert resolve_expression("$count(Player, alive)", state=world) == 3

    def test_count_with_property(self, world):
        # 'ready' is True on p0 and p2
        assert resolve_expression("$count(Player, ready)", state=world) == 2

    def test_entities_of(self, world):
        ids = resolve_expression("$entities_of(Mob)", state=world)
        assert set(ids) == {"m0", "m1"}

    def test_alive_of(self, world):
        ids = resolve_expression("$alive_of(Player)", state=world)
        assert set(ids) == {"p0", "p1", "p2"}
        assert "p3" not in ids  # dead

    def test_find_first(self, world):
        # Find a player with `ready` true — should be p0
        eid = resolve_expression("$find_first(Player, ready)", state=world)
        assert eid == "p0"

    def test_sum_of(self, world):
        # scores: 0, 10, 20, 30 → sum 60
        assert resolve_expression("$sum_of(Player, score)", state=world) == 60

    def test_max_of(self, world):
        assert resolve_expression("$max_of(Player, score)", state=world) == 30

    def test_min_of(self, world):
        assert resolve_expression("$min_of(Player, score)", state=world) == 0


class TestSpatialQueries:
    def test_entities_at(self, world):
        # loc0 has p0, p2
        ids = resolve_expression("$entities_at(loc0)", state=world)
        assert set(ids) == {"p0", "p2"}

    def test_adjacent_entities(self, world):
        # adjacent to loc0 = loc1 = {p1, p3}
        ids = resolve_expression("$adjacent_entities(loc0)", state=world)
        assert set(ids) == {"p1", "p3"}

    def test_within_range_zero_hops(self, world):
        ids = resolve_expression("$within_range(loc0, 0)", state=world)
        assert set(ids) == {"p0", "p2"}  # only loc0 itself

    def test_within_range_one_hop(self, world):
        ids = resolve_expression("$within_range(loc0, 1)", state=world)
        # loc0 + loc1
        assert set(ids) == {"p0", "p2", "p1", "p3"}

    def test_within_range_two_hops(self, world):
        ids = resolve_expression("$within_range(loc0, 2)", state=world)
        # all locations
        assert set(ids) == {"p0", "p1", "p2", "p3", "m0", "m1"}

    def test_distance(self, world):
        assert resolve_expression("$distance(loc0, loc0)", state=world) == 0
        assert resolve_expression("$distance(loc0, loc1)", state=world) == 1
        assert resolve_expression("$distance(loc0, loc2)", state=world) == 2

    def test_distance_unreachable(self, world):
        # Add a disconnected location
        world.adjacency["isolated"] = []
        # loc0 → isolated is unreachable
        assert resolve_expression("$distance(loc0, isolated)", state=world) == -1

    def test_path_exists(self, world):
        assert resolve_expression("$path_exists(loc0, loc2)", state=world) is True

    def test_path_exists_disconnected(self, world):
        world.adjacency["isolated"] = []
        assert resolve_expression("$path_exists(loc0, isolated)", state=world) is False


class TestAggregations:
    def test_avg(self, world):
        assert resolve_expression("$avg(2, 4, 6)") == 4

    def test_first(self, world):
        assert resolve_expression("$first(null, 0, hello)") == 0  # 0 is non-None
        # Test with list
        assert resolve_expression("$first([1, 2, 3])") == 1

    def test_any(self, world):
        # All non-None args
        assert resolve_expression("$any(0, 0, 1)") is True
        assert resolve_expression("$any(0, 0, 0)") is False

    def test_all(self, world):
        assert resolve_expression("$all(1, 2, 3)") is True
        assert resolve_expression("$all(1, 0, 3)") is False

    def test_if_ternary(self, world):
        assert resolve_expression("$if(1, hello, goodbye)") == "hello"
        assert resolve_expression("$if(0, hello, goodbye)") == "goodbye"


class TestQueriesInsidePredicates:
    """The real power: query functions composed with the boolean
    predicate language. This is what makes complex games declarative."""

    def test_last_one_standing_via_query(self, world):
        from fg_env_kernel.predicates import evaluate
        # Predicate: only one alive player remains
        expr = "$count(Player, alive) == 1"
        assert evaluate(expr, state=world) is False
        # Kill p0, p1
        world.entities["p0"].alive = False
        world.entities["p1"].alive = False
        assert evaluate(expr, state=world) is True

    def test_aoe_target_query(self, world):
        """AoE attack hits everyone within range — count them via expr."""
        from fg_env_kernel.predicates import evaluate
        expr = "$len($within_range($actor.location_id, 1)) >= 3"
        # Use p0 as actor (at loc0); within_range 1 = {p0, p2, p1, p3} = 4
        assert evaluate(expr, actor=world.entities["p0"], state=world) is True
