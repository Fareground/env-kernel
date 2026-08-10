"""Tests for the pluggable termination module.

Covers:
  - Each built-in check_type evaluates correctly via the registry
  - Custom @termination("name") registrations work end-to-end
  - Winner resolvers return the expected dict
  - Engine delegates termination to the registry path
"""

import pytest

from fg_env_kernel import termination as term
from fg_env_kernel.registry import termination as term_decorator
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.engine import SimulationEngine, TerminationCondition
from fg_env_kernel.registry import registry
from fg_env_kernel.state import WorldState


@pytest.fixture
def world():
    state = WorldState()
    state.entity_types["player"] = EntityType(name="player", role="agent")
    state.entities["a"] = Entity(
        id="a", name="Alice", entity_type="player",
        properties={"score": 0, "money": 100, "alive": True},
    )
    state.entities["b"] = Entity(
        id="b", name="Bob", entity_type="player",
        properties={"score": 0, "money": 100, "alive": True},
    )
    state.entities["c"] = Entity(
        id="c", name="Charlie", entity_type="player",
        properties={"score": 0, "money": 100, "alive": True},
    )
    return state


class TestBuiltinChecks:
    def test_all_dead(self, world):
        tc = TerminationCondition(
            name="extinction", check_type="all_dead",
            params={"entity_type": "player"},
        )
        assert term.evaluate(world, tc) is False
        for e in world.entities.values():
            e.alive = False
        assert term.evaluate(world, tc) is True

    def test_last_one_standing(self, world):
        tc = TerminationCondition(
            name="solo", check_type="last_one_standing",
            params={"entity_type": "player"},
        )
        assert term.evaluate(world, tc) is False
        world.entities["a"].alive = False
        world.entities["b"].alive = False
        assert term.evaluate(world, tc) is True
        winner = term.resolve_winner(world, tc)
        assert winner["winner_id"] == "c"

    def test_first_to_score(self, world):
        tc = TerminationCondition(
            name="race", check_type="first_to_score",
            params={"entity_type": "player", "property": "score", "target": 10},
        )
        assert term.evaluate(world, tc) is False
        world.entities["b"].set("score", 15)
        assert term.evaluate(world, tc) is True
        winner = term.resolve_winner(world, tc)
        assert winner["winner_id"] == "b"

    def test_property_threshold(self, world):
        tc = TerminationCondition(
            name="wealth", check_type="property_threshold",
            params={"entity_type": "player", "property": "money",
                    "operator": "gte", "value": 500},
        )
        assert term.evaluate(world, tc) is False
        world.entities["a"].set("money", 600)
        assert term.evaluate(world, tc) is True

    def test_bankruptcy(self, world):
        tc = TerminationCondition(
            name="last_solvent", check_type="bankruptcy",
            params={"entity_type": "player", "money_property": "money"},
        )
        # Currently all 3 have money — not triggered
        assert term.evaluate(world, tc) is False
        world.entities["b"].set("money", 0)
        world.entities["c"].set("money", 0)
        assert term.evaluate(world, tc) is True
        winner = term.resolve_winner(world, tc)
        assert winner["winner_id"] == "a"

    def test_count_property(self, world):
        tc = TerminationCondition(
            name="3wealthy", check_type="count_property",
            params={"entity_type": "player", "property": "money",
                    "operator": "gte", "value": 100, "min_count": 3},
        )
        assert term.evaluate(world, tc) is True
        world.entities["a"].set("money", 50)
        assert term.evaluate(world, tc) is False

    def test_expr_check_type(self, world):
        tc = TerminationCondition(
            name="round_check", check_type="expr",
            params={"expr": "1 == 1"},
        )
        assert term.evaluate(world, tc) is True

        tc2 = TerminationCondition(
            name="round_check_false", check_type="expr",
            params={"expr": "1 == 2"},
        )
        assert term.evaluate(world, tc2) is False


class TestPluggability:
    def test_custom_check_type_registration(self, world):
        """An env-builder can register a brand-new termination check."""
        @term_decorator("__test_three_alive", replace=True)
        def _three_alive(state, params, rng):
            alive = sum(1 for e in state.entities.values() if e.alive)
            return alive == 3

        try:
            tc = TerminationCondition(name="custom", check_type="__test_three_alive")
            assert term.evaluate(world, tc) is True
            world.entities["a"].alive = False
            assert term.evaluate(world, tc) is False
        finally:
            registry.terminations.unregister("__test_three_alive")

    def test_custom_check_with_winner_resolver(self, world):
        """Custom check + custom winner resolver works together."""
        @term_decorator("__test_richest_wins", replace=True)
        def _check(state, params, rng):
            return any(e.get("money", 0) >= 300 for e in state.entities.values())

        def _resolve(state, params, rng):
            winner = max(state.entities.values(), key=lambda e: e.get("money", 0))
            return {"winner_id": winner.id, "winner_name": winner.name}
        term.register_winner_resolver("__test_richest_wins", _resolve)

        try:
            world.entities["b"].set("money", 500)
            tc = TerminationCondition(name="rich", check_type="__test_richest_wins")
            assert term.evaluate(world, tc) is True
            winner = term.resolve_winner(world, tc)
            assert winner["winner_id"] == "b"
        finally:
            registry.terminations.unregister("__test_richest_wins")

    def test_unknown_check_type_returns_false(self, world):
        tc = TerminationCondition(name="???", check_type="nonexistent_xyz")
        assert term.evaluate(world, tc) is False

    def test_buggy_check_returns_false_with_log(self, world, caplog):
        @term_decorator("__test_explodes", replace=True)
        def _boom(state, params, rng):
            raise RuntimeError("predicate bug")
        try:
            tc = TerminationCondition(name="boom", check_type="__test_explodes")
            assert term.evaluate(world, tc) is False
        finally:
            registry.terminations.unregister("__test_explodes")


class TestEngineDelegation:
    def test_engine_uses_registry_path(self, world):
        """The engine's _check_termination consults the registry first."""
        tc = TerminationCondition(
            name="bob_won", check_type="first_to_score",
            params={"entity_type": "player", "property": "score", "target": 10},
        )
        engine = SimulationEngine(world, termination_conditions=[tc])

        # Not triggered yet
        assert engine._check_termination() is None
        # Bob scores
        world.entities["b"].set("score", 15)
        fired = engine._check_termination()
        assert fired is tc
        # And the winner resolver matches
        winner = engine._resolve_winner(fired)
        assert winner["winner_id"] == "b"

    def test_engine_compound_and(self, world):
        sub1 = TerminationCondition(
            name="rich", check_type="property_threshold",
            params={"entity_type": "player", "property": "money",
                    "operator": "gte", "value": 50},
        )
        sub2 = TerminationCondition(
            name="late", check_type="expr",
            params={"expr": "$state.temporal.current_round >= 0"},
        )
        tc = TerminationCondition(
            name="both", check_type="compound_and",
            sub_conditions=[sub1, sub2],
        )
        engine = SimulationEngine(world, termination_conditions=[tc])
        assert engine._check_termination() is tc

    def test_check_all_helper(self, world):
        tc1 = TerminationCondition(
            name="never", check_type="first_to_score",
            params={"entity_type": "player", "property": "score", "target": 99999},
        )
        tc2 = TerminationCondition(
            name="last_one_standing",
            check_type="last_one_standing",
            params={"entity_type": "player"},
        )
        world.entities["a"].alive = False
        world.entities["b"].alive = False
        fired = term.check_all(world, [tc1, tc2])
        assert fired is tc2
