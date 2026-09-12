"""Referenced history retains exact prefixes without repeated history walks."""
import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from fg_env_kernel.action import ActionDefinition
from fg_env_kernel.state import ActionHistory
from test_execution_checkpoint import make_engine, semantic_events


@pytest.mark.parametrize('continuous', [False, True])
def test_referenced_roundtrip_has_identical_actions_rng_and_results(continuous):
    baseline = make_engine(continuous=continuous)
    baseline.run()
    saved = []
    def stop(engine):
        boundary = engine._continuous_time.processed_count >= 7 if continuous else engine.state.temporal.current_round == 3
        if boundary:
            saved.append((json.loads(json.dumps(engine.checkpoint(include_action_history=False))),
                          engine.state.action_history.to_dict()['history']))
            engine.pause()
    source = make_engine(continuous=continuous, callback=stop)
    source.run()
    checkpoint, history = saved[0]
    assert checkpoint['format'] == 'fg-execution-v2'
    assert checkpoint['world']['action_history']['history'] is None
    restored = make_engine(continuous=continuous)
    restored.restore_checkpoint(checkpoint, action_history=history)
    restored.run()
    assert restored.state.to_dict() == baseline.state.to_dict()
    assert semantic_events(restored) == semantic_events(baseline)
    assert restored._rng.getstate() == baseline._rng.getstate()


@pytest.mark.parametrize('corruption', ['missing', 'short', 'long', 'action', 'success', 'round', 'actor', 'inline', 'digest', 'bool_count', 'float_count'])
def test_wrong_history_is_rejected_without_mutating_live_engine(corruption):
    source = make_engine()
    source.step(); source.step()
    checkpoint = source.checkpoint(include_action_history=False)
    history = source.state.action_history.to_dict()['history']
    if corruption == 'missing':
        history = None
    elif corruption == 'short':
        history['a'].pop()
    elif corruption == 'long':
        history['a'].append(copy.deepcopy(history['a'][0]))
    elif corruption == 'actor':
        history['other'] = history.pop('a')
    elif corruption == 'inline':
        checkpoint['world']['action_history']['history'] = history
    elif corruption == 'digest':
        checkpoint['action_history']['entities']['a']['digest'] = '0' * 64
    elif corruption == 'bool_count':
        checkpoint['action_history']['entities']['a']['count'] = True
    elif corruption == 'float_count':
        checkpoint['action_history']['entities']['a']['count'] = float(len(history['a']))
    else:
        record = history['a'][0]
        record[corruption] = {'action': 'other', 'success': not record['success'], 'round': 999}[corruption]
    target = make_engine()
    target.step()
    before = target.checkpoint()
    with pytest.raises(ValueError, match='action history'):
        target.restore_checkpoint(checkpoint, action_history=history)
    assert target.checkpoint() == before


def test_reference_capture_never_iterates_accumulated_records_and_stays_small():
    class NoIteration(list):
        def __iter__(self):
            raise AssertionError('Walked accumulated action history')
    engine = make_engine()
    history = engine.state.action_history
    for i in range(100_000):
        history.record('a', 'try', i % 2 == 0, i)
    history._history['a'] = NoIteration(history._history['a'])
    # Both snapshot and its reference must be O(actors), not O(actions).
    checkpoint = engine.checkpoint(include_events=False, include_action_history=False)
    assert checkpoint['action_history']['entities']['a']['count'] == 100_000
    assert len(json.dumps(checkpoint)) < 20_000
    assert history.is_available('a', 'finish', ActionDefinition('finish', 'Finish', 'player', requires_action='try', requires_action_success=True), 100_001)


def test_action_record_is_immutable_and_restore_rebuilds_reference():
    history = ActionHistory()
    history.record('a', 'try', False, 1)
    history.record('a', 'try', True, 2)
    with pytest.raises(FrozenInstanceError):
        history.get_recent('a')[0].success = True
    reference = history.reference()
    assert ActionHistory.from_dict(history.to_dict()).reference() == reference
    history.record('a', 'later', True, 3)
    assert reference != history.reference()
    assert reference['entities']['a']['count'] == 2


def test_default_export_remains_inline_and_detached():
    engine = make_engine()
    engine.step()
    checkpoint = engine.checkpoint()
    assert checkpoint['format'] == 'fg-execution-v1' and 'action_history' not in checkpoint
    assert checkpoint['world']['action_history']['history']['a']
    restored = make_engine(checkpoint=checkpoint)
    assert restored.checkpoint() == checkpoint


def test_invalid_record_does_not_corrupt_history_or_its_reference():
    history = ActionHistory()
    history.record('a', 'try', True, 1)
    original = history.to_dict(), history.reference()
    with pytest.raises(ValueError):
        history.record('a', 'try', True, float('nan'))
    assert (history.to_dict(), history.reference()) == original


def test_external_event_and_action_histories_are_independently_required():
    engine = make_engine()
    engine.step()
    checkpoint = engine.checkpoint(include_events=False, include_action_history=False)
    history = engine.state.action_history.to_dict()['history']
    target = make_engine()
    with pytest.raises(ValueError, match='event history'):
        target.restore_checkpoint(checkpoint, action_history=history)
    target.restore_checkpoint(checkpoint, action_history=history,
                              event_history=engine.state.event_log.to_transcript())
    assert target.state.to_dict() == engine.state.to_dict()
