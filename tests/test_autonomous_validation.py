import copy
import pytest
from fg_env_kernel import compile_template

WORLD = {'name': 'Autonomous counter', 'entity_types': [{'name': 'Counter', 'role': 'object', 'properties': [{'name': 'count', 'type': 'int', 'default': 0}]}], 'entities': [{'id': 'counter', 'entity_type': 'Counter', 'properties': {}}], 'derived_rules': [{'name': 'increment', 'when': True, 'for_each': '$alive_of(Counter)', 'then': [{'operation': 'add', 'target': 'actor', 'field': 'count', 'value': 1}]}], 'termination_conditions': [{'name': 'end', 'check_type': 'round_limit', 'params': {'max_rounds': 3}}]}


def test_round_observers_see_completed_autonomous_updates():
    result = compile_template(WORLD)
    assert result.ok, result.errors
    observed = []
    result.engine.on_round_end = lambda n, state: observed.append((n, state.entities['counter'].get('count')))
    result.engine.run()
    assert observed == [(1, 1), (2, 2), (3, 3)]


@pytest.mark.parametrize('then', [[], [{'operation': 'silently_ignored'}]])
def test_malformed_autonomous_effects_fail_construction(then):
    schema = copy.deepcopy(WORLD)
    schema['derived_rules'][0]['then'] = then
    assert not compile_template(schema).ok


def test_unknown_property_dynamics_shape_is_not_a_working_mechanism():
    schema = copy.deepcopy(WORLD)
    schema.pop('derived_rules')
    schema['property_dynamics'] = {'updates': {'count': 1}}
    result = compile_template(schema)
    assert not result.ok
    assert 'no executable updates' in result.errors[0].message


def test_runtime_rule_failure_is_not_swallowed(monkeypatch):
    import fg_env_kernel.runtime.effect_dispatch as dispatch
    result = compile_template(WORLD)
    def failed(*args, **kwargs):
        raise ValueError('broken rule execution')
    monkeypatch.setattr(dispatch, 'apply_effects', failed)
    with pytest.raises(ValueError, match='broken rule execution'):
        result.engine.run()
