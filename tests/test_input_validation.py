import copy
import math
import pytest
from fg_env_kernel import compile_template
from fg_env_kernel.action import Precondition, Operator
from fg_env_kernel.physics import PhysicsModel, PhysicsExprError

BASE = {
    'name': 'Reservation negotiation',
    'entity_types': [{'name': 'Trader', 'role': 'agent', 'properties': [
        {'name': 'price', 'type': 'float', 'default': 50},
        {'name': 'reservation', 'type': 'float', 'default': 40},
    ]}],
    'entities': [{'id': 'seller', 'name': 'Seller', 'entity_type': 'Trader', 'properties': {'price': 50, 'reservation': 40}}],
    'actions': [{'name': 'accept', 'actor_type': 'Trader', 'resolution_archetype': 'deterministic'}],
    'termination_conditions': [{'name': 'end', 'check_type': 'round_limit', 'params': {'max_rounds': 1}}],
}

@pytest.mark.parametrize('operator', ['gte', 'gt', 'lte', 'lt', 'has_resource', 'skill_gte'])
@pytest.mark.parametrize('value', [{'runtime': 'reservation'}, '$actor.reservation', float('nan'), float('inf'), True])
def test_bad_numeric_guards_fail_compilation_with_repair_guidance(operator, value):
    raw = copy.deepcopy(BASE)
    raw['actions'][0]['preconditions'] = [{'operator': operator, 'field': 'price', 'value': value}]
    result = compile_template(raw)
    assert not result.ok
    assert any('actions[0].preconditions[0]' in e.path and 'expr' in e.message for e in result.errors)
    with pytest.raises(ValueError, match='finite numeric'):
        Precondition(operator=Operator(operator), field='price', value=value)

@pytest.mark.parametrize('expr,price,allowed', [
    ('$actor.price >= $actor.reservation', 50, True),
    ('$actor.price >= $actor.reservation', 30, False),
    ('$actor.price <= $actor.reservation', 30, True),
    ('$actor.price <= $actor.reservation', 50, False),
])
def test_reservation_expressions_work_in_actor_perception(expr, price, allowed):
    raw = copy.deepcopy(BASE)
    raw['entities'][0]['properties']['price'] = price
    raw['actions'][0]['preconditions'] = [{'expr': expr}]
    result = compile_template(raw)
    assert result.ok, result.errors
    assert bool(result.state.get_valid_actions('seller')) is allowed

@pytest.mark.parametrize('value', [-1, 41, float('nan'), float('inf')])
def test_physical_initial_values_outside_bounds_are_rejected(value):
    with pytest.raises(PhysicsExprError):
        PhysicsModel.from_schema({'variables': [{'name': 'water', 'value': value, 'rate': '0', 'min': 0, 'max': 40}]})

@pytest.mark.parametrize('value', [0, 40])
def test_physical_exact_bounds_are_preserved(value):
    model = PhysicsModel.from_schema({'variables': [{'name': 'water', 'value': value, 'rate': '0', 'min': 0, 'max': 40}]})
    model.integrate(1)
    assert model.values['water'] == value

def test_negative_unbounded_physical_state_is_valid():
    model = PhysicsModel.from_schema({'variables': [{'name': 'temperature', 'value': -10, 'rate': '0'}]})
    model.integrate(1)
    assert model.values['temperature'] == -10
