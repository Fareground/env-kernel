"""Structural playtests distinguish dynamics from stalled decision worlds."""
import math

import pytest

from fg_env_kernel.pipeline.loader import load_world
from fg_env_kernel.pipeline.compile import compile_template
from fg_env_kernel.pipeline.smoke import smoke_test


def test_autonomous_physics_is_healthy_without_decisions():
    compiled = compile_template({
        'name': 'Cooling room',
        'entity_types': [{'name': 'Room', 'role': 'object', 'properties': [
            {'name': 'temperature', 'type': 'float', 'default': 30}]}],
        'entities': [{'id': 'room', 'entity_type': 'Room', 'name': 'Room'}],
        'physics': {'params': {'ambient': 20, 'cooling_rate': 0.2}, 'substeps': 16,
            'variables': [{'name': 'temperature', 'value': 30,
                'rate': '-cooling_rate * (temperature - ambient)',
                'writeback': {'entity_type': 'Room', 'property': 'temperature'}}]},
    })
    assert compiled.ok, compiled.errors
    state, engine = compiled.state, compiled.engine
    report = smoke_test(engine, rounds=3)
    assert report.healthy
    assert report.completed
    assert not report.actions_expected
    assert report.actions_taken == {}
    assert not any('no actions' in warning for warning in report.warnings)
    assert state.entities['room'].properties['temperature'] == pytest.approx(
        20 + 10 * math.exp(-0.6), abs=0.001)


@pytest.mark.parametrize('with_agent', [False, True])
def test_declared_actions_that_never_execute_remain_unhealthy(with_agent):
    _, engine = load_world({
        'name': 'Stalled world',
        'entity_types': [{'name': 'Worker', 'role': 'agent', 'properties': []}],
        'entities': [{'id': 'worker', 'entity_type': 'Worker', 'name': 'Worker'}] if with_agent else [],
        'actions': [{'name': 'work', 'actor_type': 'Worker', 'preconditions': [{'expr': '1 == 0'}]}],
    })
    report = smoke_test(engine, rounds=2)
    assert report.actions_expected
    assert not report.healthy
    assert any('no actions' in warning for warning in report.warnings)


def test_agent_without_declared_actions_remains_unhealthy():
    _, engine = load_world({'name': 'Missing decisions',
        'entity_types': [{'name': 'Worker', 'role': 'agent', 'properties': []}],
        'entities': [{'id': 'worker', 'entity_type': 'Worker', 'name': 'Worker'}]})
    report = smoke_test(engine, rounds=2)
    assert report.actions_expected
    assert not report.healthy
