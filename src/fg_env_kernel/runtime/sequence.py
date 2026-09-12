"""Shared discrete action preparation and persisted multi-round lifecycle."""
from __future__ import annotations

from copy import deepcopy

from ..action import ActionInstance


def committed_action(engine, entity_id):
    """Non-interruptible work needs no repeated model decision to continue."""
    active = engine.state.sequences.get_active(entity_id)
    definition = engine.state.action_definitions.get(active.action_name) if active else None
    if active and definition and not definition.interruptible:
        return ActionInstance(active.action_name, entity_id, active.target_id,
                              deepcopy(active.parameters))
    return None


def prepare_action(engine, entity, selected):
    """Return (committed action, definition) only when effects are due now.

    A duration of N counts the starting round and advances once per world
    round. Continuation retains its committed target/parameters. Invalidated
    work is cancelled before progress or completion, in every execution mode.
    """
    tracker = engine.state.sequences
    active = tracker.get_active(entity.id)
    round_num = engine.state.temporal.current_round

    def event(kind, sequence, reason=None):
        definition = engine.state.action_definitions.get(sequence.action_name)
        private = {'visible_to': [eid for eid in (entity.id, sequence.target_id) if eid]} if definition and not definition.broadcast else {}
        engine._emit_event(kind, actor_id=entity.id, action_name=sequence.action_name,
                           target_id=sequence.target_id,
                           data={'rounds_completed': sequence.rounds_completed,
                                 'total_rounds': sequence.total_rounds,
                                 **private,
                                 **({'reason': reason} if reason else {})},
                           narrative=f'{entity.name}: {sequence.action_name} '
                                     f'({sequence.rounds_completed}/{sequence.total_rounds}).')

    if active:
        definition = engine.state.action_definitions.get(active.action_name)
        same = selected is not None and selected.action_name.strip().casefold() == active.action_name.casefold()
        if definition and (same or not definition.interruptible):
            selected = ActionInstance(active.action_name, entity.id, active.target_id,
                                      deepcopy(active.parameters))
        else:
            tracker.cancel(entity.id)
            event('sequence_cancelled', active, 'interrupted' if definition else 'action_removed')
            active = None

    if selected is None:
        engine._emit_event('action_skipped', actor_id=entity.id,
                           narrative=f'{entity.name} passes.')
        return None
    definition = engine._resolve_action_def(entity, selected)
    if definition is None:
        if active:
            tracker.cancel(entity.id)
            event('sequence_cancelled', active, 'action_invalidated')
        return None
    if not active and definition.sequence_rounds <= 0:
        return selected, definition
    starting = active is None
    if starting:
        tracker.start(entity.id, selected.action_name, selected.target_id, selected.parameters,
                      definition.sequence_rounds, round_num)
        active = tracker.get_active(entity.id)
    else:
        # Older snapshots did not record the last tick separately. Their
        # stored progress remains authoritative; avoid repeating that tick.
        last = active.last_advanced_round
        if last is None:
            last = active.started_round + active.rounds_completed
        if round_num <= last:
            return None
    active.last_advanced_round = round_num
    complete = tracker.advance(entity.id)
    if starting:
        event('sequence_started', active)
    if complete:
        event('sequence_completed', active)
        return selected, definition
    if not starting:
        event('sequence_progress', active)
    return None
