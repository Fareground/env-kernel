import copy
from dataclasses import fields

from fg_env_kernel import compile_template, export_kernel_contract
from fg_env_kernel.derived_rules import DerivedRule
from fg_env_kernel.temporal import Phase


def test_live_export_resolves_derived_rules_phases_and_effects_without_guesswork():
    schema = export_kernel_contract()['template_schema']
    assert schema['properties']['derived_rules']['items'] == {'$ref': '#/$defs/DerivedRuleSpec'}
    assert schema['properties']['temporal'] == {'$ref': '#/$defs/TemporalSpec'}
    definitions = schema['$defs']
    rule = definitions['DerivedRuleSpec']
    assert set(rule['properties']) == {('as' if f.name == 'as_var' else f.name) for f in fields(DerivedRule)}
    assert rule['properties']['then']['items'] == {'$ref': '#/$defs/EffectSpec'}
    assert rule['required'] == ['then']
    assert definitions['TemporalSpec']['properties']['phases']['items'] == {'$ref': '#/$defs/Phase'}
    assert set(definitions['Phase']['properties']) == {f.name for f in fields(Phase)}
    assert not any('Always include at least one' in note for note in export_kernel_contract()['notes'])


def test_every_exported_local_reference_exists_and_exports_are_independent():
    contract = export_kernel_contract()
    schema = contract['template_schema']
    def visit(value):
        if isinstance(value, dict):
            if '$ref' in value:
                assert value['$ref'].startswith('#/$defs/')
                assert value['$ref'].removeprefix('#/$defs/') in schema['$defs']
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(schema)
    schema['$defs']['DerivedRuleSpec']['properties'].clear()
    assert export_kernel_contract()['template_schema']['$defs']['DerivedRuleSpec']['properties']


def test_documented_autonomous_rule_and_phase_fields_execute_without_rewriting_world():
    schema = {'name': 'Daily stock', 'entity_types': [
        {'name': 'Shop', 'role': 'object', 'properties': [{'name': 'stock', 'type': 'int', 'default': 10}]}],
        'entities': [{'id': 'shop', 'entity_type': 'Shop'}],
        'temporal': {'mode': 'discrete', 'round_duration_seconds': 86400, 'time_unit_label': 'day',
            'sim_start_iso': '2026-01-01T00:00:00Z',
            'phases': [{'name': 'trade', 'active_roles': ['Shop'], 'initiative_type': 'fixed',
                'initiative_property': None, 'initiative_descending': True, 'handler': None,
                'handler_params': {}, 'description': 'Daily activity', 'resolution_mode': 'sequential'}]},
        'derived_rules': [{'name': 'sell', 'description': 'Daily sale', 'when': '$params.store.stock >= 2',
            'for_each': '$entities_of(Shop)', 'as': 'store', 'once_per_entity': False, 'once_global': False,
            'then': [{'operation': 'subtract', 'target': '$params.store', 'field': 'stock', 'value': 2}]}],
        'termination_conditions': [{'name': 'end', 'check_type': 'round_limit', 'params': {'max_rounds': 3}}]}
    before = copy.deepcopy(schema)
    result = compile_template(schema)
    assert result.ok, result.errors
    assert result.engine.run().entities['shop'].get('stock') == 4
    assert result.state.temporal.round_duration_seconds == 86400
    assert result.state.temporal.phases[0].description == 'Daily activity'
    assert schema == before
