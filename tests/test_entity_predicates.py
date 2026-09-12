"""Documented entity lookup has the same meaning in guards and effects."""
import copy

import pytest

from fg_env_kernel import compile_template
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.effect_values import resolve_value
from fg_env_kernel.predicates import evaluate, resolve
from test_effect_values import WORLD


@pytest.fixture
def state():
    result = compile_template(copy.deepcopy(WORLD))
    assert result.ok
    return result.state


@pytest.mark.parametrize('expression', ["$entity('m').cash", '$entity("m").cash',
                                        '$entity($params.id).cash'])
def test_property_lookup_matches_effect_values(state, expression):
    context = {'state': state, 'params': {'id': 'm'}}
    assert resolve(expression, **context) == resolve_value(expression, **context) == 100
    assert evaluate(expression + ' == 100', **context)
    assert evaluate({'op': '==', 'left': expression, 'right': 100}, **context)
    assert resolve(expression + ' - 20', **context) == 80


def test_false_boolean_is_resolved_not_mistaken_for_missing(state):
    assert evaluate("!$entity('m').active", state=state)
    assert evaluate("$entity('m').active == false", state=state)
    state.get_entity('m').set('active', True)
    assert evaluate("$entity('m').active", state=state)
    assert not evaluate("!$entity('m').active", state=state)


@pytest.mark.parametrize('expression', ["$entity('absent').active", "$entity('m').missing",
    '$entity($params.missing).active', "$entity('m').__class__", '$entity()',
    "$entity('m', 'extra').active"])
def test_unresolved_lookup_never_turns_true_under_negation(state, expression):
    assert not evaluate(expression, state=state)
    assert not evaluate('!' + expression, state=state)
    assert not evaluate(expression + ' != false', state=state)
    assert not evaluate({'op': 'not', 'child': {'op': '==', 'left': expression, 'right': 0}}, state=state)


def test_real_action_guard_and_termination_execute_once():
    schema = copy.deepcopy(WORLD)
    schema['actions'][0]['preconditions'] = [{'expr': "!$entity('m').active"}]
    schema['actions'][0]['effects_on_success'] = [
        {'operation': 'subtract', 'target': 'actor', 'field': 'cash', 'value': 10},
        {'operation': 'set', 'target': 'actor', 'field': 'active', 'value': True}]
    schema['termination_conditions'] = [{'name': 'closed', 'check_type': 'expr',
        'params': {'expr': "$entity('m').active == true"}}]
    calls = []

    def decide(actor_id, perception, valid):
        calls.append(actor_id)
        assert 'buy' in valid
        return ActionInstance(action_name='buy', actor_id=actor_id)

    compiled = compile_template(schema, decision_fn=decide)
    assert compiled.ok, compiled.errors
    compiled.engine.max_rounds = 3
    compiled.engine.run()
    assert calls == ['m']
    assert compiled.state.get_entity('m').get('cash') == 90
    assert compiled.state.get_entity('m').get('active') is True
    assert not compiled.state.event_log.get_by_type('effect_dropped')
