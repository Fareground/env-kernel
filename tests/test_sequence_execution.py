"""Action duration is world time, not an execution-mode implementation detail."""
import json

import pytest

from fg_env_kernel import compile_template
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.engine import SimulationEngine


def make(mode, duration=3, interruptible=True, phases=1, decision=None, checkpoint=None):
    schema = {'name': 'Construction', 'entity_types': [
        {'name': 'Worker', 'role': 'agent'},
        {'name': 'Observer', 'role': 'agent'},
        {'name': 'Site', 'role': 'object', 'properties': [{'name': 'built', 'type': 'int', 'default': 0}]}],
        'entities': [{'id': 'worker', 'entity_type': 'Worker'},
                     {'id': 'observer', 'entity_type': 'Observer'},
                     {'id': 'site', 'entity_type': 'Site'}, {'id': 'other', 'entity_type': 'Site'}],
        'temporal': {'phases': [{'name': f'phase{n}', 'resolution_mode': 'simultaneous' if mode == 'simultaneous' else 'sequential'} for n in range(phases)]},
        'actions': [{'name': 'wait', 'actor_type': 'Worker'},
                    {'name': 'build', 'actor_type': 'Worker', 'target_type': 'Site',
                     'sequence_rounds': duration, 'interruptible': interruptible,
                     'parameters': [{'name': 'amount', 'type': 'int', 'required': True}],
                     'effects_on_success': [{'operation': 'add', 'target': 'target', 'field': 'built', 'value': '$params.amount'}]}]}
    result = compile_template(schema)
    assert result.ok
    def default(actor, perception, valid):
        return ActionInstance('build', actor, 'site', {'amount': 2})
    return SimulationEngine(result.state, decision_fn=decision or default, max_rounds=duration,
                            parallel_decisions=2 if mode == 'parallel' else 0,
                            execution_checkpoint=checkpoint)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
@pytest.mark.parametrize('duration', [1, 2, 3])
def test_effects_apply_once_on_exact_declared_round(mode, duration):
    engine = make(mode, duration)
    for round_number in range(1, duration + 1):
        engine.step()
        assert engine.state.entities['site'].get('built') == (2 if round_number == duration else 0)
    assert len(engine.state.event_log.get_by_type('action_resolved')) == 1
    assert len(engine.state.event_log.get_by_type('sequence_completed')) == 1
    assert not engine.state.sequences.is_in_sequence('worker')


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
def test_continuation_cannot_change_original_target_or_parameters(mode):
    calls = []
    def decide(actor, perception, valid):
        calls.append(perception)
        return ActionInstance('build', actor, 'site' if len(calls) == 1 else 'other',
                              {'amount': 2 if len(calls) == 1 else 99})
    engine = make(mode, decision=decide)
    engine.run()
    assert engine.state.entities['site'].get('built') == 2
    assert engine.state.entities['other'].get('built') == 0
    assert calls[1]['current_sequence']['progress'] == 1


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
@pytest.mark.parametrize('interruptible', [False, True])
@pytest.mark.parametrize('passing', [False, True])
def test_pass_and_replacement_obey_interruptibility(mode, interruptible, passing):
    count = 0
    def decide(actor, perception, valid):
        nonlocal count
        count += 1
        if count == 1:
            return ActionInstance('build', actor, 'site', {'amount': 2})
        return None if passing else ActionInstance('wait', actor)
    engine = make(mode, interruptible=interruptible, decision=decide)
    engine.run()
    assert engine.state.entities['site'].get('built') == (0 if interruptible else 2)
    assert not engine.state.sequences.is_in_sequence('worker')


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
def test_multiple_phases_do_not_accelerate_a_sequence(mode):
    engine = make(mode, duration=3, phases=2)
    engine.step()
    assert engine.state.sequences.get_active('worker').rounds_completed == 1
    engine.step()
    assert engine.state.entities['site'].get('built') == 0
    engine.step()
    assert engine.state.entities['site'].get('built') == 2


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
def test_checkpoint_keeps_progress_and_original_committed_inputs(mode):
    baseline = make(mode)
    baseline.run()
    split = make(mode)
    split.step()
    checkpoint = json.loads(json.dumps(split.checkpoint()))
    resumed = make(mode, checkpoint=checkpoint)
    resumed.run()
    assert resumed.state.to_dict() == baseline.state.to_dict()
    def events(engine):
        return [{k: v for k, v in row.items() if k != 'timestamp'}
                for row in engine.state.event_log.to_transcript()]
    assert events(resumed) == events(baseline)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
def test_invalidated_sequence_cannot_progress_or_complete(mode):
    engine = make(mode)
    engine.step()
    engine.state.action_history.lock('worker', ['build'])
    engine.run()
    assert engine.state.entities['site'].get('built') == 0
    assert not engine.state.sequences.is_in_sequence('worker')
    assert not engine.state.event_log.get_by_type('sequence_completed')


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
def test_noninterruptible_work_needs_only_one_decision_call(mode):
    calls = []
    def decide(actor, perception, valid):
        calls.append(actor)
        assert len(calls) == 1, 'Committed work must not incur another model call'
        return ActionInstance('build', actor, 'site', {'amount': 2})
    engine = make(mode, duration=10, interruptible=False, decision=decide)
    engine.run()
    assert calls == ['worker']
    assert engine.state.entities['site'].get('built') == 2


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
@pytest.mark.parametrize('interruptible', [False, True])
def test_empty_action_menu_does_not_leave_unfinishable_work(mode, interruptible):
    engine = make(mode, interruptible=interruptible)
    engine.step()
    engine.state.action_history.lock('worker', ['build', 'wait'])
    engine.decision_fn = lambda *_: pytest.fail('Do not request decisions with no legal actions')
    engine.run()
    assert not engine.state.sequences.is_in_sequence('worker')
    assert engine.state.entities['site'].get('built') == 0


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
def test_private_sequence_events_do_not_become_public(mode):
    engine = make(mode)
    engine.state.action_definitions['build'].broadcast = False
    engine.run()
    events = [e for e in engine.state.event_log.get_all() if e.event_type.startswith('sequence_')]
    assert events
    assert all(e.data['visible_to'] == ['worker', 'site'] for e in events)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous', 'parallel'])
def test_replacement_multi_round_action_starts_its_own_duration(mode):
    engine = make(mode, duration=3)
    from copy import deepcopy
    replacement = deepcopy(engine.state.action_definitions['build'])
    replacement.name = 'repair'
    replacement.sequence_rounds = 2
    engine.state.register_action(replacement)
    engine.step()
    engine.decision_fn = lambda actor, *_: ActionInstance('repair', actor, 'other', {'amount': 7})
    engine.step()
    assert engine.state.entities['other'].get('built') == 0
    engine.step()
    assert engine.state.entities['site'].get('built') == 0
    assert engine.state.entities['other'].get('built') == 7
