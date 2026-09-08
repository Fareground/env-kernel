"""An absent budget runs to a world outcome, without a hidden fallback cap."""
import json
import pytest
from fg_env_kernel.action import ActionDefinition, ActionInstance, Effect, EffectOperation
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.engine import SimulationEngine, TerminationCondition
from fg_env_kernel.state import WorldState


def engine(**kwargs):
    state = WorldState()
    state.register_entity_type(EntityType('player', 'agent'))
    state.spawn_entity(Entity('a', 'Alice', 'player', {'score': 0}))
    state.register_action(ActionDefinition('score', 'Score', 'player', resolution_archetype='deterministic',
        effects_on_success=[Effect('actor', EffectOperation.ADD, field='score', value=1)]))
    return SimulationEngine(state, decision_fn=lambda eid, *_: ActionInstance('score', eid), seed=189,
        max_rounds=None, termination_conditions=[TerminationCondition('goal', check_type='property_threshold',
            params={'entity_type': 'player', 'property': 'score', 'value': 205})], **kwargs)


def test_uncapped_run_ends_on_world_outcome_beyond_old_default():
    sim = engine()
    sim.run()
    assert sim.state.temporal.current_round == 205
    assert sim.terminated_by == 'goal'
    assert sim.finished


def test_uncapped_step_and_checkpoint_continue_to_same_outcome():
    sim = engine()
    for _ in range(202):
        sim.step()
    assert not sim.finished
    checkpoint = json.loads(json.dumps(sim.checkpoint(), allow_nan=False))
    restored = engine(execution_checkpoint=checkpoint)
    assert restored.max_rounds is None
    restored.run()
    assert restored.state.temporal.current_round == 205
    assert restored.terminated_by == 'goal'


def test_uncapped_pause_and_stop_remain_available():
    sim = engine()
    sim.on_round_end = lambda round_number, _: sim.pause() if round_number == 3 else None
    sim.run()
    assert sim.is_paused() and not sim.finished
    sim.on_round_end = lambda round_number, _: sim.stop() if round_number == 7 else None
    sim.resume()
    assert sim.finished and sim.state.temporal.current_round == 7
    assert sim.terminated_by is None


@pytest.mark.parametrize('budget', [-1, True, 2.5, '200'])
def test_invalid_budgets_do_not_become_unbounded(budget):
    with pytest.raises(ValueError, match='max_rounds'):
        SimulationEngine(WorldState(), max_rounds=budget)


def test_continuous_unbounded_clock_and_event_budget_survive_checkpoint():
    from fg_env_kernel.continuous_time import ContinuousTemporalModel
    clock = ContinuousTemporalModel(max_time=None, max_events=None)
    sim = engine(continuous_time=clock)
    sim.on_checkpoint = lambda current: current.pause() if current.state.temporal.current_round == 202 else None
    sim.run()
    assert sim.is_paused()
    checkpoint = json.loads(json.dumps(sim.checkpoint(), allow_nan=False))
    restored = engine(execution_checkpoint=checkpoint)
    assert restored._continuous_time.max_time is None
    assert restored._continuous_time.max_events is None
    restored.run()
    assert restored.terminated_by == 'goal'
    assert restored.state.entities['a'].properties['score'] == 205
