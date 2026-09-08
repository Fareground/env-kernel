import copy
import random

import pytest

from fg_env_kernel import compile_template, smoke_test
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.effect_values import EffectValueError, resolve_value
from fg_env_kernel.pipeline.loader import load_world


WORLD = {
    "name": "Effect value regression",
    "temporal": {"phases": [{"name": "act"}]},
    "entity_types": [{"name": "Manager", "role": "agent", "properties": [
        {"name": "cash", "type": "float", "default": 100},
        {"name": "price", "type": "float", "default": 10},
        {"name": "utility", "type": "int", "default": 0},
        {"name": "active", "type": "bool", "default": False},
    ]}],
    "entities": [{"id": "m", "name": "Manager", "entity_type": "Manager", "properties": {}}],
    "actions": [{"name": "buy", "actor_type": "Manager", "resolution_archetype": "deterministic",
                 "effects_on_success": [{"operation": "subtract", "target": "$actor", "field": "cash", "value": 10}]}],
    "termination_conditions": [{"name": "one_round", "check_type": "round_limit", "params": {"max_rounds": 1}}],
}


def world(op, value, field="cash"):
    data = copy.deepcopy(WORLD)
    data["actions"][0]["effects_on_success"][0].update(operation=op, value=value, field=field)
    return data


def run(data, *, raw=False):
    decide = lambda eid, perception, valid: ActionInstance(action_name="buy", actor_id=eid)
    if raw:
        state, engine = load_world(data, decision_fn=decide)
    else:
        compiled = compile_template(data, decision_fn=decide)
        assert compiled.ok, compiled.errors
        state, engine = compiled.state, compiled.engine
    engine.run()
    dropped = state.event_log.get_by_type("effect_dropped")
    return state.get_entity("m"), dropped


@pytest.mark.parametrize("op,expected", [("subtract", 90), ("add", 110), ("multiply", 1000)])
@pytest.mark.parametrize("value", [10, "$actor.price", {"expr": "$actor.price"},
                                    "$actor.price * 2 - 10", {"expr": "2 * ($actor.price - 5)"}])
def test_numeric_values_execute_exact_arithmetic(op, expected, value):
    entity, dropped = run(world(op, value))
    assert entity.get("cash") == expected
    assert not dropped


@pytest.mark.parametrize("op", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("value", [None, True, "ten", float("nan"), float("inf"), -float("inf"),
                                    {}, {"expr": None}, {"expr": "$actor.price", "extra": 1}])
def test_invalid_explicit_numeric_values_fail_compilation_and_cannot_mutate_raw_worlds(op, value):
    data = world(op, value)
    compiled = compile_template(data)
    assert not compiled.ok
    assert any("value" in issue.path for issue in compiled.errors)
    entity, dropped = run(data, raw=True)
    assert entity.get("cash") == 100
    assert len(dropped) == 1
    assert dropped[0].data["reason"] == "invalid_effect_value"
    assert dropped[0].data["operation"] == op


@pytest.mark.parametrize("op", ["add", "subtract", "multiply", "set"])
@pytest.mark.parametrize("value", ["$actor.missing_price", "$actor.missing_price + 10",
                                    "$actor.price / 0", "$actor.price * 1e309",
                                    "$sum($actor.price, $actor.missing)", "$abs('ten')"])
def test_dynamic_failures_are_diagnostic_not_magnitude_or_zero(op, value):
    entity, dropped = run(world(op, value))
    assert entity.get("cash") == 100
    assert len(dropped) == 1
    assert dropped[0].data["reason"] == "invalid_effect_value"
    assert dropped[0].data["detail"]


@pytest.mark.parametrize("op,expected", [("add", 101), ("subtract", 99), ("multiply", 100)])
def test_only_omitted_operands_retain_magnitude_default_through_compile(op, expected):
    data = world(op, None)
    del data["actions"][0]["effects_on_success"][0]["value"]
    entity, dropped = run(data)
    assert entity.get("cash") == expected
    assert not dropped


@pytest.mark.parametrize("field,value", [("utility", {"invalid": 5}), ("utility", "ten"),
                                        ("utility", None), ("utility", True), ("utility", 1.5),
                                        ("active", "false"), ("active", None), ("active", 1)])
def test_typed_set_rejects_invalid_literals_at_compile_and_runtime(field, value):
    data = world("set", value, field)
    assert not compile_template(data).ok
    entity, dropped = run(data, raw=True)
    assert entity.get(field) == (False if field == "active" else 0)
    assert len(dropped) == 1


def test_set_arithmetic_preserves_integer_and_boolean_types():
    entity, dropped = run(world("set", "$actor.price * 4 - 20", "utility"))
    assert entity.get("utility") == 20
    assert type(entity.get("utility")) is int
    assert not dropped
    entity, dropped = run(world("set", "$entity('m').active", "active"))
    assert entity.get("active") is False
    assert not dropped


def test_missing_boolean_ref_is_never_stored_as_truthy_text():
    entity, dropped = run(world("set", "$entity('absent').active", "active"))
    assert entity.get("active") is False
    assert len(dropped) == 1


def test_derived_numeric_set_and_runtime_lookup_evaluate():
    data = world("set", "$lookup(runtime, tax_rate) * $actor.price", "utility")
    data["tables"] = {"runtime": {"tax_rate": 2}}
    data["derived_rules"] = [{"name": "portfolio", "when": "true", "then": [
        {"operation": "set", "target": "m", "field": "cash",
         "value": {"expr": "$state.entities.m.utility + $state.entities.m.price * 3"}},
    ]}]
    entity, dropped = run(data)
    assert entity.get("utility") == 20
    assert entity.get("cash") == 50
    assert not dropped


@pytest.mark.parametrize("src", ["$actor.price +", "$actor.price garbage", "$actor.price @ 2",
                                 "$actor.price.__class__()", "$lookup(runtime, 'unclosed)",
                                 {"expr": "__import__('os').getcwd()"}])
def test_malformed_expression_is_rejected_not_partially_evaluated(src):
    assert not compile_template(world("subtract", src)).ok


def test_functions_are_seeded_and_if_only_evaluates_selected_branch():
    assert resolve_value("$random(1, 6)", rng=random.Random(3)) == random.Random(3).randint(1, 6)
    assert resolve_value("$if(false, $actor.missing, 7)") == 7
    assert resolve_value({"expr": "$max(2, 3 * 4) + 1"}) == 13
    assert resolve_value("$first($actor.missing, 4)") == 4
    with pytest.raises(EffectValueError):
        resolve_value("$if('false', 7, 8)")


def test_smoke_exposes_invalid_dynamic_mutations_and_blocks_compilation():
    data = world("subtract", "$actor.missing_price")
    compiled = compile_template(data, smoke=True, smoke_rounds=1)
    assert not compiled.ok
    assert any("missing_price" in issue.message and "subtract" in issue.message for issue in compiled.errors)
    report = smoke_test(compile_template(data).engine, rounds=1)
    assert not report.healthy
    assert len(report.invalid_effects) == 1
    assert "missing_price" in report.summary()


@pytest.mark.parametrize("op", ["add", "subtract", "multiply"])
def test_integer_mutations_never_silently_truncate(op):
    data = world(op, 0.5, "utility")
    data["entities"][0]["properties"]["utility"] = 3
    entity, dropped = run(data)
    assert entity.get("utility") == 3
    assert len(dropped) == 1


def test_deeply_nested_expressions_fail_validation_without_recursion_error():
    value = "$abs(" * 70 + "1" + ")" * 70
    assert not compile_template(world("subtract", value)).ok
