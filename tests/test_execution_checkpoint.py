"""Save/reload must be equivalent to uninterrupted execution, not just same seed."""
import copy
import json

import pytest

from fg_env_kernel.action import ActionDefinition, ActionInstance, Effect, EffectOperation
from fg_env_kernel.continuous_time import ContinuousTemporalModel
from fg_env_kernel.engine import SimulationEngine
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.invariants import InvariantChecker
from fg_env_kernel.state import WorldState
from fg_env_kernel.world_events import WorldEventEngine, WorldEventDefinition, WorldDynamicsEngine


def make_state():
    state = WorldState()
    state.register_entity_type(EntityType('player', 'agent'))
    state.spawn_entity(Entity('a', 'Alice', 'player', {'skill': .4, 'score': 0}))
    state.spawn_entity(Entity('b', 'Bob', 'player', {'skill': .7, 'score': 0}))
    state.register_action(ActionDefinition('try', 'Try', 'player', resolution_archetype='skill_check',
        effects_on_success=[Effect('actor', EffectOperation.ADD, field='score', value=1)]))
    state._schema_triggers = [{'name': 'first_success', 'when': 'action_resolved', 'once': True,
        'effect': [{'operation': 'add', 'target': 'a', 'field': 'score', 'value': 5}]},
        {'name': 'periodic', 'when': 'action_resolved', 'cooldown_rounds': 3,
         'effect': [{'operation': 'add', 'target': 'a', 'field': 'score', 'value': 2}]}]
    return state


def decision(entity_id, perception, valid_actions):
    return ActionInstance('try', entity_id)


def world_events():
    return WorldDynamicsEngine(WorldEventEngine([
        WorldEventDefinition('weather', probability=.5, duration=3, cooldown=2,
            cascade_events=['after'], cascade_delay=2,
            effects=[Effect('a', EffectOperation.ADD, field='score', value=3)]),
        WorldEventDefinition('after', trigger_type='scheduled', trigger_rounds=[],
            effects=[Effect('b', EffectOperation.ADD, field='score', value=4)]),
    ]))


def make_engine(*, checkpoint=None, callback=None, continuous=False, event_history=None):
    return SimulationEngine(make_state(), decision_fn=decision, max_rounds=9, seed=189,
        world_event_engine=world_events(), invariant_checker=InvariantChecker(),
        continuous_time=ContinuousTemporalModel(max_time=5, environment_interval=.5) if continuous else None,
        on_checkpoint=callback, execution_checkpoint=checkpoint, checkpoint_event_history=event_history)


def semantic_events(engine):
    return [{key: value for key, value in row.items() if key != 'timestamp'}
            for row in engine.state.event_log.to_transcript()]


@pytest.mark.parametrize('continuous', [False, True])
def test_split_run_matches_uninterrupted_random_resolutions_events_and_outcome(continuous):
    baseline = make_engine(continuous=continuous)
    baseline.run()
    checkpoints = []
    def pause_after_boundary(engine):
        if (engine._continuous_time.processed_count >= 7 if continuous else engine.state.temporal.current_round == 3):
            checkpoints.append(json.loads(json.dumps(engine.checkpoint(), allow_nan=False)))
            engine.pause()
    split = make_engine(continuous=continuous, callback=pause_after_boundary)
    split.run()
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    resumed = make_engine(checkpoint=checkpoint, continuous=continuous)
    resumed.run()
    assert semantic_events(resumed) == semantic_events(baseline)
    assert resumed.state.to_dict() == baseline.state.to_dict()
    assert resumed._rng.getstate() == baseline._rng.getstate()
    assert resumed.triggers._last_fired == baseline.triggers._last_fired
    assert resumed.triggers._fired_once == baseline.triggers._fired_once
    assert resumed.world_event_engine._cascade_queue == baseline.world_event_engine._cascade_queue
    assert resumed.world_event_engine.event_engine._last_triggered == baseline.world_event_engine.event_engine._last_triggered
    assert resumed.world_event_engine.event_engine.rng is resumed._rng
    assert checkpoint == checkpoints[0]


def test_external_event_history_is_required_and_restored_before_next_decision():
    source = make_engine()
    source.step(); source.step()
    cp = source.checkpoint(include_events=False)
    with pytest.raises(ValueError, match='event history'):
        make_engine(checkpoint=cp)
    restored = make_engine(checkpoint=cp, event_history=source.state.event_log.to_transcript())
    assert restored.state.event_log.to_transcript() == source.state.event_log.to_transcript()
    assert restored._trend_analyzer._history == source._trend_analyzer._history
    assert restored._rng.random() == source._rng.random()


def test_failed_restore_does_not_mutate_live_engine_or_world():
    engine = make_engine()
    engine.step()
    state_identity = engine.state
    before = engine.checkpoint()
    corrupt = copy.deepcopy(before)
    corrupt['execution']['rng_state'] = [0, [], None]
    with pytest.raises(ValueError, match='checkpoint'):
        engine.restore_checkpoint(corrupt)
    assert engine.state is state_identity
    assert engine.checkpoint() == before


def test_capture_inside_agent_turn_is_rejected():
    engine = make_engine()
    def during_turn(*args):
        with pytest.raises(ValueError, match='completed round'):
            engine.checkpoint()
        return decision(*args)
    engine.decision_fn = during_turn
    engine.step()
    assert engine.checkpoint()['world']['temporal']['current_round'] == 1


def test_completed_checkpoint_does_not_run_or_emit_end_twice():
    original = make_engine()
    original.run()
    resumed = make_engine(checkpoint=original.checkpoint())
    resumed.run()
    assert resumed.state.event_log.to_transcript() == original.state.event_log.to_transcript()


def test_deck_rng_remains_shared_without_reshuffling_saved_cards():
    from fg_env_kernel.deck_module import DeckModule
    from fg_env_kernel.domain_module import DomainModuleManager
    source = make_state()
    source.domain_modules = DomainModuleManager()
    source.domain_modules.add_module(DeckModule(params={'cards': [{'text': str(i)} for i in range(15)], 'consume_on_draw': True}))
    engine = SimulationEngine(source, decision_fn=decision, seed=189)
    deck = source.domain_modules.get_module('deck')
    deck.draw(source.entities['a'])
    cp = json.loads(json.dumps(engine.checkpoint()))
    restored_state = make_state()
    restored_state.domain_modules = DomainModuleManager()
    restored_state.domain_modules.add_module(DeckModule(params={'cards': []}))
    restored = SimulationEngine(restored_state, decision_fn=decision, execution_checkpoint=cp)
    other = restored.state.domain_modules.get_module('deck')
    assert other._rng is restored._rng
    assert deck.to_dict() == other.to_dict()
    deck.shuffle(); other.shuffle()
    assert deck.to_dict() == other.to_dict()
    assert engine._rng.random() == restored._rng.random()


def test_callback_cannot_capture_mid_event_cascade():
    engine = make_engine()
    def observer(event):
        with pytest.raises(ValueError, match='completed round'):
            engine.checkpoint()
    engine.on_event = observer
    engine.step()
    assert engine.checkpoint()['format'] == 'fg-execution-v1'


def test_scheduled_ties_keep_insertion_order_after_json_reload():
    from fg_env_kernel.continuous_time import EventQueue, ScheduledEvent
    queue = EventQueue()
    queue.schedule(ScheduledEvent(1, id='z', data={'order': 1}))
    queue.schedule(ScheduledEvent(1, id='a', data={'order': 2}))
    restored = EventQueue.from_dict(json.loads(json.dumps(queue.to_dict())))
    for target in (queue, restored):
        target.schedule(ScheduledEvent(1, id='m', data={'order': 3}))
        assert [target.pop().data['order'] for _ in range(3)] == [1, 2, 3]


def test_time_limit_does_not_discard_a_pending_future_event():
    clock = ContinuousTemporalModel(max_time=1)
    event = clock.schedule_agent_turn('a', at_time=2)
    assert clock.pop_next_event() is None
    restored = ContinuousTemporalModel.from_dict(clock.to_dict())
    restored.max_time = 3
    assert restored.pop_next_event().id == event.id


def test_named_scheduled_event_fires_after_reload_and_keeps_agent_turns():
    state = make_state()
    state._schema_triggers = [{'name': 'grant', 'when': 'grant', 'once': True,
        'effect': [{'operation': 'add', 'target': 'a', 'field': 'score', 'value': '$event.amount'}]}]
    clock = ContinuousTemporalModel(max_time=2)
    clock.schedule_custom_event(1.5, 'grant', {'amount': 7})
    engine = SimulationEngine(state, decision_fn=decision, continuous_time=clock, seed=189)
    snapshots = []
    def pause(e):
        if e._continuous_time.current_time == 1:
            snapshots.append(e.checkpoint())
            e.pause()
    engine.on_checkpoint = pause
    engine.run()
    assert snapshots
    resumed = SimulationEngine(make_state(), decision_fn=decision, execution_checkpoint=snapshots[0])
    resumed.run()
    assert len(resumed.state.event_log.get_by_type('grant')) == 1
    assert resumed.state.event_log.get_by_type('action_resolved')
    assert resumed.triggers._fired_once['grant'] is True
    assert resumed.finished


@pytest.mark.parametrize('field,value', [('current_time', -1), ('current_time', float('nan')), ('processed_count', -1)])
def test_malformed_continuous_clock_is_rejected(field, value):
    clock = ContinuousTemporalModel()
    data = clock.to_dict()
    data[field] = value
    with pytest.raises(ValueError, match='checkpoint'):
        ContinuousTemporalModel.from_dict(data)
