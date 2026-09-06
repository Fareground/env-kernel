import pytest
from fg_env_kernel import Kernel


def template(mode):
    return {
        'name': 'Decision failure',
        'entity_types': [{'name': 'Player', 'role': 'agent', 'properties': [{'name': 'score', 'type': 'int', 'default': 0}]}],
        'entities': [{'id': f'p{i}', 'name': f'Player {i}', 'entity_type': 'Player'} for i in range(4)],
        'actions': [{'name': 'play', 'actor_type': 'Player', 'effects_on_success': [{'target': 'actor', 'operation': 'add', 'field': 'score', 'value': 1}]}],
        'temporal': {'max_rounds': 100, 'phases': [{'name': 'play', 'resolution_mode': mode}]},
    }


@pytest.mark.parametrize('mode,parallel', [('sequential', 0), ('sequential', 2), ('simultaneous', 2)])
def test_decision_dependency_failure_stops_every_execution_mode(mode, parallel):
    calls = []
    def failed(*args):
        calls.append(args[0])
        raise RuntimeError('Selected model unavailable')
    world = Kernel().load(template(mode), decision_fn=failed)
    world.engine.parallel_decisions = parallel
    with pytest.raises(RuntimeError, match='Selected model unavailable'):
        world.run()
    assert world.current_round == 1
    assert not world.engine._running
    assert len(calls) <= max(1, parallel)
    assert all(e.properties['score'] == 0 for e in world.state.get_agent_entities())
    assert any(e.event_type == 'decision_error' for e in world.events)
    assert not any(e.event_type == 'simulation_end' for e in world.events)


def test_intentional_no_action_is_still_a_valid_decision():
    world = Kernel().load(template('sequential'), decision_fn=lambda *args: None, max_rounds=2)
    world.run()
    assert world.current_round == 2
    assert world.finished
