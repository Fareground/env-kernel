"""Run inputs must reach actual state, without mutating reusable definitions."""
import copy
import math
import pytest
from fg_env_kernel import compile_template

COOLING = {
    'name': 'Cooling',
    'runtime_parameters': [{'name': 'start_temp', 'type': 'float', 'default': 30}, {'name': 'cooling_k', 'type': 'float', 'default': 0.2}],
    'entity_types': [{'name': 'Room', 'role': 'object', 'properties': [{'name': 'temperature', 'type': 'float', 'default': 30, 'min_value': -50}]}],
    'entities': [{'id': 'room', 'entity_type': 'Room', 'properties': {}}],
    'physics': {'params': {'k': '$lookup(runtime, cooling_k)'}, 'variables': [{'name': 'temperature', 'value': '$lookup(runtime, start_temp)', 'rate': '-k*(temperature-20)', 'writeback': {'entity_type': 'Room', 'property': 'temperature'}}]},
    'termination_conditions': [{'name': 'end', 'check_type': 'round_limit', 'params': {'max_rounds': 3}}],
}

@pytest.mark.parametrize('start,k', [(30, .2), (40, .4), (-10, .1)])
def test_cooling_inputs_change_initial_state_and_analytical_trajectory(start, k):
    schema = copy.deepcopy(COOLING)
    schema['last_runtime_params'] = {'start_temp': start, 'cooling_k': k}
    before = copy.deepcopy(schema)
    result = compile_template(schema)
    assert result.ok, result.errors
    room = result.state.entities['room']
    assert room.properties['temperature'] == start
    model = result.state.physics
    for _ in range(3):
        model.integrate(1, state=result.state)
    assert room.properties['temperature'] == pytest.approx(20 + (start-20)*math.exp(-k*3), abs=.003)
    assert schema == before


def grant(default='$lookup(runtime, budget)'):
    return {'name': 'Grant budget', 'runtime_parameters': [{'name': 'budget', 'default': 10}],
            'entity_types': [{'name': 'Grant', 'role': 'agent', 'properties': [{'name': 'funds', 'type': 'int', 'default': default, 'min_value': 0}]}],
            'entities': [{'id': 'grant', 'entity_type': 'Grant', 'properties': {}}]}


def test_entity_default_and_override_lookups_resolve_to_numbers():
    schema = grant()
    schema['last_runtime_params'] = {'budget': 5}
    result = compile_template(schema)
    assert result.ok, result.errors
    assert result.state.entities['grant'].properties['funds'] == 5
    schema['entities'][0]['properties']['funds'] = '$lookup("runtime", "budget")'
    assert compile_template(schema).state.entities['grant'].properties['funds'] == 5
    assert schema['entity_types'][0]['properties'][0]['default'] == '$lookup(runtime, budget)'

@pytest.mark.parametrize('budget', [-1, 'five', float('inf')])
def test_bound_entity_values_are_validated_after_resolution(budget):
    schema = grant()
    schema['last_runtime_params'] = {'budget': budget}
    result = compile_template(schema)
    assert not result.ok
    assert 'initial input' in result.errors[0].message.lower() or 'runtime input' in result.errors[0].message.lower()


def test_missing_initial_binding_is_an_actionable_compile_error():
    result = compile_template(grant('$lookup(runtime, misspelled_budget)'))
    assert not result.ok
    assert 'grant.funds' in result.errors[0].message
    assert 'misspelled_budget' in result.errors[0].message


def test_bound_round_limit_is_resolved_without_changing_the_template():
    schema = copy.deepcopy(COOLING)
    schema['runtime_parameters'].append({'name': 'days', 'default': 3})
    schema['termination_conditions'][0]['params']['max_rounds'] = '$lookup(runtime, days)'
    schema['last_runtime_params'] = {'days': 5}
    result = compile_template(schema)
    assert result.ok, result.errors
    assert result.engine.termination_conditions[0].params['max_rounds'] == 5


def test_explicit_runtime_table_is_not_mutated_by_an_override():
    schema = copy.deepcopy(COOLING)
    schema['tables'] = {'runtime': {'cooling_k': .1}}
    before = copy.deepcopy(schema)
    assert compile_template(schema).ok
    assert schema == before
