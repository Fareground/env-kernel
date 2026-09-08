import pytest

from fg_env_kernel.action import Effect, EffectCondition, EffectOperation
from fg_env_kernel.engine import SimulationEngine
from fg_env_kernel.entity import Entity
from fg_env_kernel.predicates import evaluate
from fg_env_kernel.runtime.conditions import _evaluate_world_condition
from fg_env_kernel.state import WorldState


@pytest.fixture
def context():
    state = WorldState()
    actor = Entity(id="reviewer", name="Reviewer", entity_type="Reviewer", properties={"tokens": 2})
    target = Entity(id="payment", name="Payment", entity_type="Payment", properties={"is_fraud": True, "amount": 20})
    state.entities = {actor.id: actor, target.id: target}
    engine = SimulationEngine(state)
    return engine, actor, target


@pytest.mark.parametrize("expression,expected", [
    ("$target.is_fraud == true", True),
    ("$target.is_fraud == false", False),
    ("$target.is_fraud", True),
    ("!$target.is_fraud", False),
    ("$target.amount >= 20 && $actor.tokens > 0", True),
    ("$target.amount < 20 || $actor.tokens == 0", False),
    ("!($target.amount < 20)", True),
    ("$target.amount * 2 == 40", True),
    ("$params.flag == false", True),
    ("$target.missing", False),
    ("!$target.missing", False),
    ("$target.missing != true", False),
    ("!($target.missing == true)", False),
    ("$target.missing + 1 > 0", False),
    ("$target.missing == $actor.also_missing", False),
    ("unrecognized text", False),
    ("true", True),
    ("false", False),
])
def test_conditional_branches_match_shared_predicates_and_effect_guards(context, expression, expected):
    engine, actor, target = context
    params = {"flag": False}
    assert evaluate(expression, actor=actor, target=target, state=engine.state, params=params) is expected
    assert engine._evaluate_effect_condition(EffectCondition(expr=expression), actor, target, params) is expected
    effect = Effect(operation=EffectOperation.CONDITIONAL, target="actor", value={
        "if_expr": expression,
        "then": [{"operation": "set", "target": "actor", "field": "branch", "value": "then"}],
        "else": [{"operation": "set", "target": "actor", "field": "branch", "value": "else"}],
    })
    engine._apply_effects([effect], actor, target, params, None)
    assert actor.get("branch") == ("then" if expected else "else")


@pytest.mark.parametrize("flag", [True, False])
def test_target_flag_if_expr_and_if_compare_are_equivalent(context, flag):
    engine, actor, target = context
    target.set("is_fraud", flag)
    for spec in ({"if_expr": "$target.is_fraud == true"}, {"if_compare": ["$target.is_fraud", "==", True]},
                 {"if": {"expr": "$target.is_fraud == true"}}):
        assert engine._evaluate_conditional_clause(
            spec, actor=actor, target=target, params={}, result=None, resolve_val=lambda v: v,
        ) is flag


@pytest.mark.parametrize("spec", [
    {"if_expr": "$state.entities.payment.amount > 10"},
    {"if_compare": ["$state.entities.payment.amount", ">", 10]},
    {"expr": "$state.entities.payment.amount", "operator": "gt", "value": 10},
])
def test_world_conditions_share_numeric_predicate_semantics(context, spec):
    engine, _, _ = context
    assert _evaluate_world_condition(engine, spec) is True


def test_last_event_context_reaches_conditional_predicate(context):
    engine, actor, target = context
    engine._emit_event("observed", data={"is_fraud": True}, narrative="Observed")
    effect = Effect(operation=EffectOperation.CONDITIONAL, target="actor", value={
        "if_expr": "$last_event.is_fraud == true",
        "then": [{"operation": "set", "target": "actor", "field": "observed", "value": True}],
    })
    engine._apply_effects([effect], actor, target, {}, None)
    assert actor.get("observed") is True


def test_explicit_defined_check_allows_missing_and_zero(context):
    _, actor, _ = context
    actor.set("zero", 0)
    assert evaluate({"op": "defined", "value": "$actor.zero"}, actor=actor) is True
    assert evaluate({"op": "not", "child": {"op": "defined", "value": "$actor.missing"}}, actor=actor) is True
    assert evaluate({"op": "not", "child": "$actor.missing == true"}, actor=actor) is False


@pytest.mark.parametrize("clause", [
    {"if_expr": "$target.is_fraud == true"},
    {"if_compare": ["$target.is_fraud", "==", True]},
    {"if": {"expr": "$target.is_fraud == true"}},
])
def test_composed_effects_use_the_canonical_condition_evaluator(context, clause):
    from fg_env_kernel.composition import _shim_engine_for_ctx
    from fg_env_kernel.effect_context import EffectContext
    engine, actor, target = context
    shim = _shim_engine_for_ctx(EffectContext(state=engine.state, actor=actor, target=target))
    assert shim._evaluate_conditional_clause(
        clause, actor=actor, target=target, params={}, result=None, resolve_val=lambda v: v,
    ) is True
