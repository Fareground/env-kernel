"""Report actual engine termination evidence, not a stale payload field."""
import pytest

from fg_env_kernel import compile_template, smoke_test


def world(condition):
    compiled = compile_template({
        'name': 'Termination evidence',
        'entity_types': [{'name': 'Shop', 'role': 'object', 'properties': [
            {'name': 'steps', 'type': 'int', 'default': 0}]}],
        'entities': [{'id': 'shop', 'entity_type': 'Shop'}],
        'derived_rules': [{'when': 'true', 'then': [
            {'operation': 'add', 'target': 'shop', 'field': 'steps', 'value': 1}]}],
        'termination_conditions': [condition],
    })
    assert compiled.ok, compiled.errors
    return compiled


@pytest.mark.parametrize('check_type', ['round_limit', 'max_rounds', 'max_rounds_reached'])
def test_smoke_reports_real_horizon_termination_without_a_false_repair_warning(check_type):
    compiled = world({'name': 'Three completed rounds', 'check_type': check_type, 'params': {'max_rounds': 3}})
    report = smoke_test(compiled.engine, rounds=3)
    assert compiled.engine.terminated_by == 'Three completed rounds'
    assert report.terminated_by == compiled.engine.terminated_by
    assert report.rounds_run == 3 and report.healthy
    assert compiled.state.get_entity('shop').get('steps') == 3
    assert not any('termination' in warning for warning in report.warnings)


def test_early_expr_termination_preserves_the_named_condition():
    compiled = world({'name': 'One step', 'check_type': 'expr', 'params': {'expr': '$entity(shop).steps >= 1'}})
    report = smoke_test(compiled.engine, rounds=5)
    assert report.terminated_by == 'One step'
    assert report.rounds_run == 1


def test_unreached_condition_is_not_fabricated_as_successful_termination():
    compiled = world({'name': 'Not reached', 'check_type': 'expr', 'params': {'expr': '$entity(shop).steps >= 10'}})
    report = smoke_test(compiled.engine, rounds=3)
    assert report.terminated_by is None
    assert compiled.engine.terminated_by is None
    assert any('termination' in warning for warning in report.warnings)
