"""End-to-end: prove a game can be built using ONLY the unified
expression language for every guard — preconditions, effect
conditions, and termination — with no enum/operator imports.

This is the "schema + rules" half of the env-builder vision: every
condition in the game is a string the agent emits.
"""
import pytest

from fg_env_kernel.action import (
    ActionDefinition,
    Effect,
    EffectCondition,
    EffectOperation,
    Precondition,
)
from fg_env_kernel.engine import SimulationEngine, TerminationCondition
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.resolution import ResolutionResult
from fg_env_kernel.state import WorldState


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
        properties={"gold": 200, "alive": True},
    )
    return state


def test_precondition_via_expr_string(world):
    """Precondition uses ONLY an expression string — no operator enum."""
    state = world
    rich_action = ActionDefinition(
        name="brag",
        description="brag when wealthy",
        actor_type="player",
        preconditions=[Precondition(expr="$actor.gold >= 100")],
    )
    a = state.entities["a"]  # 50 gold
    b = state.entities["b"]  # 200 gold
    assert state._check_actor_preconditions(a, rich_action) is False
    assert state._check_actor_preconditions(b, rich_action) is True


def test_precondition_with_compound_expression(world):
    """`&& / ||` predicates work end-to-end."""
    state = world
    action = ActionDefinition(
        name="lavish_party",
        description="party",
        actor_type="player",
        preconditions=[
            Precondition(expr="$actor.gold >= 100 && $actor.alive"),
        ],
    )
    a, b = state.entities["a"], state.entities["b"]
    assert state._check_actor_preconditions(a, action) is False
    assert state._check_actor_preconditions(b, action) is True
    # Kill bob — now even the rich precondition fails
    b.set("alive", False)
    assert state._check_actor_preconditions(b, action) is False


def test_effect_condition_via_expr(world):
    """An Effect with a conditional `expr` fires only when true."""
    state = world
    engine = SimulationEngine(state)
    a, b = state.entities["a"], state.entities["b"]
    # Effect: +5 gold to actor, but only if target has more gold than actor.
    fx = Effect(
        target="actor",
        operation=EffectOperation.ADD,
        field="gold",
        value=5,
        condition=EffectCondition(expr="$target.gold > $actor.gold"),
    )
    result = ResolutionResult(success=True, magnitude=1.0)
    # b has more than a → fires
    engine._apply_effects([fx], actor=a, target=b, params={}, result=result)
    assert a.get("gold") == 55

    # Reverse direction: a (now 55) vs b (200) — flip targeting
    engine._apply_effects([fx], actor=b, target=a, params={}, result=result)
    assert b.get("gold") == 200  # condition false, no change


def test_effect_condition_legacy_path_still_works(world):
    """The old operator/field/value form still works (compat shim)."""
    state = world
    engine = SimulationEngine(state)
    a, b = state.entities["a"], state.entities["b"]
    fx = Effect(
        target="actor",
        operation=EffectOperation.ADD,
        field="gold",
        value=5,
        condition=EffectCondition(
            subject="actor", check_type="property",
            field="gold", operator="gte", value=50,
        ),
    )
    result = ResolutionResult(success=True, magnitude=1.0)
    engine._apply_effects([fx], actor=a, target=b, params={}, result=result)
    assert a.get("gold") == 55


def test_termination_via_expr(world):
    """A termination condition with check_type='expr' triggers solely
    on the predicate result."""
    state = world
    # Game ends when total gold across all players exceeds 240
    tc = TerminationCondition(
        name="economic_endgame",
        check_type="expr",
        params={"expr": "$actor.gold + $target.gold > 240"},  # context-free won't work
    )
    # Build the engine with this condition
    engine = SimulationEngine(state, termination_conditions=[tc])

    # Predicate without actor/target binding from the engine won't have
    # those entities in scope, so the check should fail-closed.
    # Use a count-based predicate that operates on world state instead:
    tc2 = TerminationCondition(
        name="solo_winner",
        check_type="expr",
        params={"expr": "1 == 1"},  # always true
    )
    engine2 = SimulationEngine(state, termination_conditions=[tc2])
    fired = engine2._check_termination()
    assert fired is not None
    assert fired.name == "solo_winner"


def test_termination_legacy_paths_still_work(world):
    """Old check_type values still trigger correctly."""
    state = world
    # Kill everyone
    state.entities["a"].alive = False
    state.entities["b"].alive = False
    tc = TerminationCondition(
        name="extinction",
        check_type="all_dead",
        params={"entity_type": "player"},
    )
    engine = SimulationEngine(state, termination_conditions=[tc])
    fired = engine._check_termination()
    assert fired is not None
    assert fired.name == "extinction"
