"""Authoring must not receive a successful build for silently ignored triggers."""
import copy

import pytest
from pydantic import ValidationError

from fg_env_kernel import compile_template, export_kernel_contract
from fg_env_kernel.pipeline.loader import load_world
from fg_env_kernel.triggers import TriggerEngine


EFFECT = {'operation': 'add', 'target': 'counter', 'field': 'count', 'value': 1}


def world(trigger):
    return {
        'name': 'Event counter',
        'entity_types': [{'name': 'Counter', 'role': 'object',
                          'properties': [{'name': 'count', 'type': 'int', 'default': 0}]}],
        'entities': [{'id': 'counter', 'entity_type': 'Counter'}],
        'triggers': [copy.deepcopy(trigger)],
        'termination_conditions': [{'name': 'end', 'check_type': 'round_limit',
                                     'params': {'max_rounds': 3}}],
    }


@pytest.mark.parametrize('trigger,path', [
    ({'when': {'expr': 'true'}, 'then': [EFFECT]}, 'when'),
    ({'condition': 'true', 'actions': [EFFECT]}, 'when'),
    ({'when': 'round_end', 'then': [EFFECT]}, 'effect'),
    ({'when': 'round_end', 'effect': []}, 'effect'),
    ({'when': 'round_end', 'effect': [EFFECT], 'once': 'false'}, 'once'),
    ({'when': 'round_end', 'effect': [EFFECT], 'cooldown_rounds': -1}, 'cooldown_rounds'),
    ({'when': 'round_end', 'effect': [EFFECT], 'cooldown_rounds': 1.5}, 'cooldown_rounds'),
    ({'when': ' ', 'effect': [EFFECT]}, 'when'),
    ({'when': 123, 'effect': [EFFECT]}, 'when'),
])
@pytest.mark.parametrize('skip_lint', [False, True])
def test_invalid_trigger_blocks_compilation_even_when_lint_is_skipped(trigger, path, skip_lint):
    schema = world(trigger)
    before = copy.deepcopy(schema)
    result = compile_template(schema, skip_lint=skip_lint, smoke=True)
    assert not result.ok and result.engine is None
    issue = next(issue for issue in result.errors if issue.path == f'triggers[0].{path}')
    assert 'derived_rules' in issue.hint and 'TriggerDefinition' in issue.hint
    assert schema == before


@pytest.mark.parametrize('key', ['effect', 'effects'])
@pytest.mark.parametrize('once,expected', [(False, 3), (True, 1)])
def test_valid_trigger_and_legacy_alias_execute_and_preserve_source(key, once, expected):
    schema = world({'name': 'count', 'when': 'round_end', key: [EFFECT], 'once': once})
    before = copy.deepcopy(schema)
    result = compile_template(schema)
    assert result.ok, result.errors
    assert result.engine.run().entities['counter'].properties['count'] == expected
    assert schema == before


@pytest.mark.parametrize('load', [lambda schema: load_world(schema),
                                  lambda schema: TriggerEngine.from_schema(schema['triggers'])])
def test_direct_loaders_reject_malformed_subscription(load):
    with pytest.raises(ValidationError):
        load(world({'when': {'expr': 'true'}, 'then': [EFFECT]}))


def test_installed_authoring_contract_has_complete_trigger_fields():
    schema = export_kernel_contract()['template_schema']
    assert schema['properties']['triggers']['items'] == {'$ref': '#/$defs/TriggerDefinition'}
    definition = schema['$defs']['TriggerDefinition']
    assert set(definition['required']) == {'when', 'effect'}
    assert definition['properties']['when']['type'] == 'string'
    assert definition['properties']['effect']['items'] == {'$ref': '#/$defs/EffectSpec'}
    assert definition['additionalProperties'] is False


def test_filter_and_cooldown_semantics_are_preserved():
    engine = TriggerEngine.from_schema([{'when': 'sold', 'effect': [EFFECT],
                                       'filter': {'kind': ['cash', 'card']}, 'cooldown_rounds': 2}])
    assert not engine.matching('sold', {'kind': 'credit'}, 'shop', 1)
    trigger, = engine.matching('sold', {'kind': 'cash'}, 'shop', 1)
    engine.record_fired(trigger, 'shop', 1)
    assert not engine.matching('sold', {'kind': 'cash'}, 'shop', 2)
    assert engine.matching('sold', {'kind': 'card'}, 'shop', 3) == [trigger]


def test_unknown_effect_is_a_targeted_compile_error():
    result = compile_template(world({'when': 'round_end', 'effect': [{**EFFECT, 'operation': 'ad'}]}))
    assert not result.ok
    assert any(issue.path == 'triggers[0].effect[0].operation' for issue in result.errors)


def test_trigger_effect_condition_is_not_discarded():
    result = compile_template(world({'when': 'round_end', 'effect': [
        {**EFFECT, 'condition': {'expr': 'false'}}]}))
    assert result.ok, result.errors
    assert result.engine.run().entities['counter'].properties['count'] == 0


def test_failed_trigger_aborts_smoke_instead_of_reporting_a_healthy_world(monkeypatch):
    from fg_env_kernel.pipeline.smoke import smoke_test
    schema = world({'name': 'broken', 'when': 'round_end', 'effect': [EFFECT]})
    result = compile_template(schema)
    assert result.ok, result.errors
    def broken(*args, **kwargs):
        raise ValueError('trigger execution broke')
    monkeypatch.setattr(result.engine, '_apply_effects', broken)
    report = smoke_test(result.engine, rounds=3)
    assert not report.healthy and not report.completed
    assert "Trigger 'broken' failed" in report.crash
    assert result.state.entities['counter'].properties['count'] == 0
    assert not result.engine.triggers._fired_once


def test_registered_trigger_effect_is_not_rejected():
    from fg_env_kernel import registry
    name = '__test_trigger_custom'
    registry.effects.register(name, lambda ctx, spec: None)
    try:
        result = compile_template(world({'when': 'round_end', 'effect': [{'operation': name}]}))
        assert result.ok, result.errors
        result.engine.run()
    finally:
        registry.effects.unregister(name)


def test_legacy_op_alias_remains_executable_without_changing_source():
    effect = {key: value for key, value in EFFECT.items() if key != 'operation'}
    effect['op'] = 'add'
    schema = world({'when': 'round_end', 'effects': [effect]})
    before = copy.deepcopy(schema)
    result = compile_template(schema)
    assert result.ok, result.errors
    assert result.engine.run().entities['counter'].properties['count'] == 3
    assert schema == before
