"""Tests for the unified predicate evaluator (fg_env_kernel.predicates).

This module is the single source of truth for boolean expressions used
by preconditions, effect conditions, terminations, state-machine
guards, and triggers. Coverage must include:

  - String form        ($actor.gold >= 100, &&/||/!, parentheses)
  - Dict form          ({"op": "gte", "left": ..., "right": ...})
  - List membership    ($x in [1,2,3])
  - Special predicates (has_resource, at_location, is_adjacent, ...)
  - Edge cases         (None, type coercion, missing fields, broken AST)
"""
import pytest

from fg_env_kernel.predicates import evaluate, resolve
from fg_env_kernel.entity import Entity


@pytest.fixture
def actor():
    return Entity(id="a", name="Alice", entity_type="player",
                  properties={"gold": 50, "alive": True, "position": 10})


@pytest.fixture
def target():
    return Entity(id="b", name="Bob", entity_type="player",
                  properties={"gold": 100, "alive": True, "position": 12})


class TestStringForm:
    def test_simple_comparison(self, actor, target):
        assert evaluate("$actor.gold >= 50", actor=actor) is True
        assert evaluate("$actor.gold > 50", actor=actor) is False
        assert evaluate("$actor.gold < 100", actor=actor) is True
        assert evaluate("$actor.gold == 50", actor=actor) is True
        assert evaluate("$actor.gold != 50", actor=actor) is False

    def test_boolean_and(self, actor, target):
        assert evaluate("$actor.alive && $target.alive", actor=actor, target=target) is True
        assert evaluate("$actor.gold > 0 && $target.gold > 0", actor=actor, target=target) is True
        assert evaluate("$actor.gold > 0 && $target.gold > 200", actor=actor, target=target) is False

    def test_boolean_or(self, actor, target):
        assert evaluate("$actor.gold > 200 || $target.gold > 50", actor=actor, target=target) is True
        assert evaluate("$actor.gold > 200 || $target.gold > 200", actor=actor, target=target) is False

    def test_boolean_not(self, actor):
        assert evaluate("!$actor.alive", actor=actor) is False
        assert evaluate("!($actor.gold > 200)", actor=actor) is True

    def test_word_operators_work(self, actor, target):
        assert evaluate("$actor.alive and $target.alive", actor=actor, target=target) is True
        assert evaluate("$actor.gold > 200 or $target.gold > 50", actor=actor, target=target) is True
        assert evaluate("not $actor.alive", actor=actor) is False

    def test_arithmetic_in_comparison(self, actor, target):
        assert evaluate("$actor.gold + 50 == 100", actor=actor) is True
        assert evaluate("$actor.gold * 2 >= $target.gold", actor=actor, target=target) is True

    def test_parentheses_group_correctly(self, actor, target):
        # Without parens, && binds tighter than ||
        assert evaluate(
            "$actor.gold > 100 || $actor.alive && $target.alive",
            actor=actor, target=target,
        ) is True
        assert evaluate(
            "($actor.gold > 100 || $actor.alive) && $target.gold > 200",
            actor=actor, target=target,
        ) is False

    def test_list_membership(self, actor):
        assert evaluate("$actor.position in [10, 20, 30]", actor=actor) is True
        assert evaluate("$actor.position in [1, 2, 3]", actor=actor) is False

    def test_string_literal_equality(self, actor):
        assert evaluate("$actor.entity_type == 'player'", actor=actor) is True
        assert evaluate("$actor.entity_type == \"player\"", actor=actor) is True

    def test_unresolvable_expression_is_false(self, actor):
        # Missing field — predicate fails closed
        assert evaluate("$actor.missing_field > 0", actor=actor) is False

    def test_no_actor_does_not_crash(self):
        assert evaluate("$actor.gold > 0") is False

    def test_empty_predicate_is_false(self):
        assert evaluate("") is False
        assert evaluate(None) is False


class TestDictForm:
    def test_simple_comparison(self, actor):
        assert evaluate({"op": "gte", "left": "$actor.gold", "right": 50}, actor=actor) is True
        assert evaluate({"op": "lt", "left": "$actor.gold", "right": 10}, actor=actor) is False

    def test_boolean_combinators(self, actor, target):
        pred = {
            "op": "and",
            "children": [
                {"op": "gte", "left": "$actor.gold", "right": 50},
                {"op": "gte", "left": "$target.gold", "right": 50},
            ],
        }
        assert evaluate(pred, actor=actor, target=target) is True

    def test_nested_boolean(self, actor, target):
        pred = {
            "op": "or",
            "children": [
                {"op": "gt", "left": "$actor.gold", "right": 1000},
                {"op": "and", "children": [
                    {"op": "gte", "left": "$actor.gold", "right": 50},
                    {"op": "eq", "left": "$target.position", "right": 12},
                ]},
            ],
        }
        assert evaluate(pred, actor=actor, target=target) is True

    def test_not_combinator(self, actor):
        pred = {"op": "not", "child": {"op": "gt", "left": "$actor.gold", "right": 1000}}
        assert evaluate(pred, actor=actor) is True

    def test_empty_op_is_true(self):
        # Matches "no precondition" semantics
        assert evaluate({}) is True


class TestSpecialPredicates:
    def test_is_alive_dict(self, actor):
        assert evaluate({"op": "is_alive", "subject": "actor"}, actor=actor) is True
        actor.set("alive", False)
        assert evaluate({"op": "is_alive", "subject": "actor"}, actor=actor) is False


class TestResolveValue:
    def test_resolve_passes_literal(self):
        assert resolve(42) == 42
        assert resolve("hello") == "hello"

    def test_resolve_dollar_expression(self, actor):
        assert resolve("$actor.gold", actor=actor) == 50

    def test_resolve_arithmetic(self, actor):
        assert resolve("$actor.gold + 25", actor=actor) == 75


class TestSafety:
    def test_exception_during_eval_returns_false(self):
        # Pathological input should not raise.
        assert evaluate("$$$$$$ malformed") is False

    def test_deeply_nested_eval_does_not_blow_stack(self, actor):
        pred = "$actor.gold > 0"
        for _ in range(20):
            pred = f"({pred}) && $actor.gold > 0"
        assert evaluate(pred, actor=actor) is True

    def test_list_form_in_dict_predicate(self, actor):
        pred = {"op": "in", "left": "$actor.position", "right": [10, 20]}
        assert evaluate(pred, actor=actor) is True
