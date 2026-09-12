"""Versioned execution checkpoints, taken only between complete work units."""
import copy
import math
import random
from collections import deque
from dataclasses import asdict
from typing import Any

from .continuous_time import ContinuousTemporalModel
from .event import EventLog, SimEvent
from .invariants import InvariantChecker, WorldInvariant
from .snapshot import SnapshotRestoreError, _preserved
from .status_effects import StatusEffectDefinition
from .triggers import TriggerEngine, TriggerSpec
from .visibility import TrendAnalyzer
from .world_events import ActiveEvent, DynamicsRule, WorldDynamicsEngine, WorldEventDefinition, WorldEventEngine


FORMAT = "fg-execution-v1"
REFERENCED_FORMAT = "fg-execution-v2"


def _world_events_to_dict(manager):
    if manager is None:
        return None
    dynamics = isinstance(manager, WorldDynamicsEngine)
    base = manager.event_engine if dynamics else manager
    if not isinstance(base, WorldEventEngine):
        raise ValueError("world event engine does not support checkpoints")
    definitions = []
    for definition in base.definitions.values():
        row = asdict(definition)
        row['effects'] = StatusEffectDefinition('_', tick_effects=definition.effects).to_dict()['tick_effects']
        definitions.append(row)
    rules = [asdict(rule) for rule in manager.dynamics_rules] if dynamics else []
    for row in rules:
        if row['max_value'] == float('inf'):
            row['max_value'] = None
    return {'kind': 'dynamics' if dynamics else 'events', 'definitions': definitions,
            'dynamics_rules': rules, 'last_triggered': base._last_triggered,
            'active': [{'name': item.definition.name, 'started_round': item.started_round,
                        'remaining_rounds': item.remaining_rounds} for item in base._active_events],
            'cascade_queue': list(manager._cascade_queue) if dynamics else []}


def _world_events_from_dict(data, rng):
    if data is None:
        return None
    from .pipeline.loader import _parse_effects
    definitions = [WorldEventDefinition(**{**row, 'effects': _parse_effects(row['effects'])})
                   for row in data['definitions']]
    base = WorldEventEngine(definitions, rng=rng)
    base._last_triggered = data['last_triggered']
    base._active_events = [ActiveEvent(base.definitions[row['name']], row['started_round'], row['remaining_rounds'])
                           for row in data['active']]
    if data['kind'] == 'events':
        result = base
    elif data['kind'] == 'dynamics':
        result = WorldDynamicsEngine(base, [DynamicsRule(**{**row, 'max_value': float('inf') if row['max_value'] is None else row['max_value']})
                                           for row in data['dynamics_rules']])
        result._cascade_queue = [tuple(row) for row in data['cascade_queue']]
    else:
        raise ValueError('unknown world event checkpoint kind')
    _preserved(data, _world_events_to_dict(result), 'world_events')
    return result


def capture(engine: Any, *, external_state=None, include_events=True, include_action_history=True) -> dict:
    if not engine._checkpoint_ready:
        raise ValueError('Execution checkpoints require a completed round or scheduled event')
    invariant = engine.invariant_checker
    if invariant is not None and not isinstance(invariant, InvariantChecker):
        raise ValueError('Invariant checker does not support checkpoints')
    domain = engine.state.domain_modules
    shared_rng = [[name, attr] for name, module in (domain._modules.items() if domain else [])
                  for attr, value in vars(module).items() if value is engine._rng]
    # WorldState.to_dict already detaches every nested subsystem. Copy the
    # engine/external payload separately so growing world history is not walked
    # and allocated a second time at every checkpoint boundary.
    world = engine.state.to_dict() if include_action_history else engine.state.to_dict(include_action_history=False)
    checkpoint = copy.deepcopy({
        'format': FORMAT if include_action_history else REFERENCED_FORMAT,
        'external_state': external_state,
        'execution': {
            'seed': engine.seed, 'rng_state': engine._rng.getstate(), 'max_rounds': engine.max_rounds,
            'parallel_decisions': engine.parallel_decisions,
            'terminated_by': engine.terminated_by, 'stopped': engine._stopped,
            'start_emitted': engine._start_emitted, 'end_emitted': engine._end_emitted,
            'paused': engine._paused, 'last_env_time': getattr(engine, '_last_env_time', None),
            'continuous_time': engine._continuous_time.to_dict() if engine._continuous_time else None,
            'termination_conditions': [asdict(condition) for condition in engine.termination_conditions],
            'triggers': {'definitions': [asdict(spec) for spec in engine.triggers.triggers],
                         'last_fired': engine.triggers._last_fired, 'fired_once': engine.triggers._fired_once},
            'world_events': _world_events_to_dict(engine.world_event_engine),
            'invariants': None if invariant is None else {
                'definitions': [asdict(rule) for rule in invariant.invariants],
                'initial_resource_totals': invariant._initial_resource_totals, 'violations': invariant.violations},
            'trends': {'max_snapshots': engine._trend_analyzer.max_snapshots,
                       'history': {eid: list(rows) for eid, rows in engine._trend_analyzer._history.items()}},
            'shared_domain_rng': shared_rng,
        },
        'event_log': {'max_events': engine.state.event_log._max_events,
                      'count': len(engine.state.event_log),
                      'events': engine.state.event_log.to_transcript() if include_events else None},
    })
    checkpoint['world'] = world
    if not include_action_history:
        checkpoint['action_history'] = engine.state.action_history.reference()
    return checkpoint


def restore(engine: Any, checkpoint: dict, *, event_history=None, action_history=None) -> None:
    """Validate all components before replacing any live engine/world state.

    Callback implementations and any external clients remain the caller's
    responsibility. `external_state` is returned by capture for that purpose.
    """
    from .runtime.engine import TerminationCondition
    data = copy.deepcopy(checkpoint)
    try:
        if data['format'] not in (FORMAT, REFERENCED_FORMAT):
            raise ValueError('unsupported execution checkpoint format')
        referenced = data['format'] == REFERENCED_FORMAT
        world_data = data['world']
        if referenced:
            from .state import ActionHistory
            if not isinstance(action_history, dict) or world_data['action_history']['history'] is not None:
                raise ValueError('checkpoint action history is missing or not externally referenced')
            # Materialize once on restore, never on every capture. Bind exact
            # actors, counts and record contents before replacing live state.
            world_data = copy.deepcopy(world_data)
            world_data['action_history']['history'] = copy.deepcopy(action_history)
            restored_history = ActionHistory.from_dict(world_data['action_history'])
            if restored_history.reference() != data['action_history']:
                raise ValueError('checkpoint action history does not match its prefix fingerprint')
        execution = data['execution']
        for key in ('seed', 'max_rounds', 'parallel_decisions'):
            if key == 'max_rounds' and execution[key] is None:
                continue
            if type(execution[key]) is not int or (key != 'seed' and execution[key] < 0):
                raise ValueError(f'invalid execution {key}')
        for key in ('stopped', 'start_emitted', 'end_emitted', 'paused'):
            if type(execution[key]) is not bool:
                raise ValueError(f'invalid execution {key}')
        if execution['terminated_by'] is not None and not isinstance(execution['terminated_by'], str):
            raise ValueError('invalid termination marker')
        rng = random.Random()
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value
        rng.setstate(tuples(execution['rng_state']))
        state = copy.copy(engine.state)
        state.modules = dict(state.modules)
        state.apply_snapshot(world_data)
        event_data = data['event_log']
        events = event_data['events'] if event_data['events'] is not None else copy.deepcopy(event_history)
        if events is None or len(events) != event_data['count']:
            raise ValueError('checkpoint event history is missing or has the wrong length')
        state.event_log = EventLog(max_events=event_data['max_events'])
        for row in events:
            state.event_log.emit(SimEvent(**row))
        if len(state.event_log) != event_data['count']:
            raise ValueError('checkpoint event history exceeds its retention limit')
        triggers = execution['triggers']
        trigger_engine = TriggerEngine(triggers=[TriggerSpec(**row) for row in triggers['definitions']],
                                       _last_fired=triggers['last_fired'], _fired_once=triggers['fired_once'])
        world_events = _world_events_from_dict(execution['world_events'], rng)
        invariant_data = execution['invariants']
        invariant = None
        if invariant_data is not None:
            invariant = InvariantChecker([WorldInvariant(**row) for row in invariant_data['definitions']])
            invariant._initial_resource_totals = invariant_data['initial_resource_totals']
            invariant.violations = invariant_data['violations']
        def condition(row):
            return TerminationCondition(**{**row, 'sub_conditions': [condition(child) for child in row['sub_conditions']]})
        termination = [condition(row) for row in execution['termination_conditions']]
        ct_data = execution['continuous_time']
        continuous = ContinuousTemporalModel.from_dict(ct_data) if ct_data is not None else None
        if continuous is not None:
            _preserved(ct_data, continuous.to_dict(), 'continuous_time')
            last_time = execution['last_env_time']
            if execution['start_emitted'] and (isinstance(last_time, bool) or not isinstance(last_time, (int, float))
                    or not math.isfinite(last_time) or not 0 <= last_time <= continuous.current_time):
                raise ValueError('invalid continuous checkpoint physics clock')
        trends = execution['trends']
        trend_analyzer = TrendAnalyzer(trends['max_snapshots'])
        trend_analyzer._history = {eid: deque(rows, maxlen=trends['max_snapshots']) for eid, rows in trends['history'].items()}
        for name, attr in execution['shared_domain_rng']:
            module = state.domain_modules.get_module(name)
            if not isinstance(getattr(module, attr, None), random.Random):
                raise ValueError(f'checkpoint shared RNG target is unavailable: {name}.{attr}')
            setattr(module, attr, rng)
        if state.property_dynamics is not None:
            state.property_dynamics.rng = rng
        fields = {
            '_rng': rng, 'seed': execution['seed'], 'max_rounds': execution['max_rounds'],
            'parallel_decisions': execution['parallel_decisions'],
            'terminated_by': execution['terminated_by'], '_stopped': execution['stopped'],
            '_start_emitted': execution['start_emitted'], '_end_emitted': execution['end_emitted'],
            '_paused': execution['paused'], '_running': False, '_checkpoint_ready': True,
            '_last_env_time': execution['last_env_time'], '_continuous_time': continuous,
            'triggers': trigger_engine, 'world_event_engine': world_events, 'invariant_checker': invariant,
            'termination_conditions': termination, '_trend_analyzer': trend_analyzer,
        }
        candidate = copy.copy(engine)
        candidate.state = state
        candidate.__dict__.update(fields)
        _preserved(data, capture(candidate, external_state=data.get('external_state'),
                                 include_events=event_data['events'] is not None,
                                 include_action_history=not referenced), 'checkpoint')
    except Exception as exc:
        raise SnapshotRestoreError(f'Cannot restore execution checkpoint: {exc}') from exc
    # Preserve the WorldState object's identity: platform callbacks close over it.
    engine.state.__dict__.update(state.__dict__)
    engine.__dict__.update(fields)
