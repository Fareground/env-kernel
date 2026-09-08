"""Nonempty snapshot regression: continued behavior, stale state, and atomic failure."""
import json
import random

import pytest

from fg_env_kernel.action import Effect, EffectOperation
from fg_env_kernel.domain.base import DomainModule, DomainModuleManager
from fg_env_kernel.entity import Entity
from fg_env_kernel.factions import Faction
from fg_env_kernel.messaging import Message
from fg_env_kernel.snapshot import SnapshotRestoreError
from fg_env_kernel.state import WorldState
from fg_env_kernel.status_effects import StatusEffectDefinition
from fg_env_kernel.temporal import Phase


def world():
    state = WorldState()
    state.entities['a'] = Entity('a', 'Alice', 'agent', {'health': 10})
    state.factions.register(Faction('org', 'Organization'))
    state.factions.register(Faction('team', 'Team', member_ids=['a', 'b'], parent='org'))
    poison = StatusEffectDefinition('poison', duration=3, tick_effects=[
        Effect(operation=EffectOperation.SUBTRACT, target='actor', field='health', value=2),
    ], blocks_actions=['sprint'], stackable=True, max_stacks=2)
    state.status_effects.apply('a', poison, source='b', round_num=1)
    state.status_effects.apply('a', poison, source='b', round_num=1)
    state.sequences.start('a', 'build', 'tower', {'height': 7}, 3, 1)
    state.sequences.advance('a')
    state.messages.post(Message('b', 'Bob', 'Yesterday', round_sent=1))
    state.messages.start_round()
    state.messages.post(Message('b', 'Bob', 'Private', 'direct', recipient_id='a', round_sent=2))
    state.messages.post(Message('b', 'Bob', 'Team only', 'faction', recipient_faction='team', round_sent=2))
    state.temporal.phases = [Phase('plan'), Phase('act', resolution_mode='simultaneous')]
    state.temporal.current_round = 2
    state.temporal.current_phase_index = 1
    state.temporal.turn_order = ['b', 'a']
    state.temporal.current_turn_index = 1
    return state


@pytest.mark.parametrize('dirty', [False, True])
def test_restore_retains_subsystem_behavior_and_removes_stale_state(dirty):
    source = world()
    snapshot = json.loads(json.dumps(source.to_dict()))
    restored = world() if dirty else WorldState()
    if dirty:
        restored.entities['stale'] = Entity('stale', 'Stale', 'agent')
        restored.factions.register(Faction('stale', 'Stale'))
        restored.messages.post(Message('stale', 'Stale', 'Remove me'))
    restored.apply_snapshot(snapshot)
    assert restored.to_dict() == snapshot
    assert set(restored.entities) == {'a'}
    assert restored.factions.org_root('team') == 'org'
    assert restored.factions.same_org('a', 'b')
    assert restored.temporal.current_phase.resolution_mode == 'simultaneous'
    assert restored.temporal.turn_order == ['b', 'a']
    assert restored.temporal.current_turn_index == 1
    for name in ('factions', 'messages', 'status_effects', 'sequences', 'relations'):
        assert restored.modules[name] is getattr(restored, name)
    assert restored.status_effects.get_blocked_actions('a') == ['sprint']
    assert len(restored.status_effects.tick('a')) == 2
    assert restored.status_effects.to_dict()['a'][0]['remaining_rounds'] == 2
    assert not restored.sequences.advance('a')
    assert restored.sequences.get_active('a').parameters == {'height': 7}
    assert restored.sequences.advance('a')
    assert restored.sequences.get_active('a') is None
    assert [m.content for m in restored.messages.get_for_entity('a', 'team')] == ['Private', 'Team only']
    assert restored.messages.get_for_entity('outsider') == []
    restored.messages.start_round()
    assert restored.messages.get_for_entity('a', 'team') == []
    assert len(restored.messages.to_dict()['history']) == 3
    # Continued mutation of the restored state never touches the source or input.
    assert source.to_dict() == snapshot


@pytest.mark.parametrize('section,value', [
    ('negotiations', {'not_supported': 'garbage'}),
    ('sequences', {'a': {'action_name': 'build'}}),
    ('messages', {'history_count': 10, 'current_round': []}),
    ('factions', {'factions': {'x': {'id': 'x', 'name': 'X', 'parent': 'missing'}}}),
    ('status_effects', {'a': [{'name': 'unknown'}]}),
    ('domain_modules', {'modules': {'unknown': {'name': 'unknown', 'type': 'unknown'}}}),
    ('plugin_modules', {'unknown': {'count': 2}}),
])
def test_bad_snapshot_does_not_partially_mutate_live_world(section, value):
    state = world()
    before = state.to_dict()
    identities = {key: id(value) for key, value in state.modules.items()}
    snapshot = world().to_dict()
    snapshot['entities']['a']['properties']['health'] = 99
    snapshot[section] = value
    with pytest.raises(SnapshotRestoreError, match=section):
        state.apply_snapshot(snapshot)
    assert state.to_dict() == before
    assert {key: id(value) for key, value in state.modules.items()} == identities


class Counter(DomainModule):
    def __init__(self, name='counter', params=None):
        super().__init__(name=name, params=params)
        self.count = 0
        self._rng = random.Random(5)

    def tick(self, state, round_number):
        self.count += 1
        return []

    def to_dict(self):
        return {**super().to_dict(), 'count': self.count}

    @classmethod
    def from_dict(cls, data):
        result = cls(data['name'], data['params'])
        result.count = data['count']
        return result


def test_domain_specific_state_rng_and_alias_are_restored_once():
    source = world()
    manager = DomainModuleManager()
    module = Counter()
    module.count = 42
    module._rng.random()
    manager.add_module(module)
    source.domain_modules = manager
    source.register_module('domain_counter', module)
    snap = json.loads(json.dumps(source.to_dict()))
    assert 'domain_counter' not in snap['plugin_modules']
    restored = WorldState.from_dict(snap, schema_provider=source)
    actual = restored.domain_modules.get_module('counter')
    assert actual.count == 42
    restored.domain_modules.tick_all(restored, 3)
    assert actual.count == 43
    assert module.count == 42
    assert restored.modules['domain_counter'] is actual
    assert actual is not module
    assert actual._rng.random() == module._rng.random()


def test_relations_resume_threshold_latches_and_remove_stale_edges():
    from fg_env_kernel.relations import RelationType, RelationThreshold
    source = world()
    source.relations.register_relation_type(RelationType('trust', thresholds=[RelationThreshold(.5, event_name='trusted')]))
    source.relations.set('a', 'b', 'trust', .8)
    source.relations._edges[('a', 'b', 'trust')].metadata = {'reason': 'helped'}
    assert len(source.relations.tick(2)) == 1
    destination = world()
    destination.relations.set('stale', 'a', 'trust', .3)
    destination.apply_snapshot(json.loads(json.dumps(source.to_dict())))
    assert destination.relations.to_dict() == source.relations.to_dict()
    assert destination.relations.tick(3) == []
    assert list(destination.relations._edges) == [('a', 'b', 'trust')]


def test_derived_once_rule_does_not_fire_again_after_restore():
    from types import SimpleNamespace
    from fg_env_kernel.derived_rules import DerivedRulesEngine
    source = world()
    source._derived_rules = DerivedRulesEngine([{'name': 'once', 'when': True, 'once_global': True,
        'then': [{'operation': 'add', 'target': 'a', 'field': 'health', 'value': 1}]}])
    engine = SimpleNamespace(state=source, _rng=random.Random(1))
    source._derived_rules.tick(engine)
    assert source.entities['a'].get('health') == 11
    restored = WorldState.from_dict(json.loads(json.dumps(source.to_dict())))
    assert restored._derived_rules.tick(SimpleNamespace(state=restored, _rng=random.Random(1))) == []
    assert restored.entities['a'].get('health') == 11


def test_optional_nulls_and_removed_plugins_clear_dirty_state():
    source = world()
    dirty = world()
    dirty.domain_modules = DomainModuleManager()
    dirty.domain_modules.add_module(Counter())
    dirty.register_module('domain_counter', dirty.domain_modules.get_module('counter'))
    dirty.register_module('extra', Counter())
    dirty.apply_snapshot(source.to_dict())
    assert dirty.domain_modules is None
    assert 'extra' not in dirty.modules
    assert 'domain_counter' not in dirty.modules


def test_modern_snapshot_missing_section_is_not_treated_as_legacy_partial():
    snapshot = world().to_dict()
    del snapshot['messages']
    with pytest.raises(SnapshotRestoreError, match='messages'):
        world().apply_snapshot(snapshot)


def test_plugin_serializer_failure_is_not_silently_omitted():
    class Broken:
        def to_dict(self):
            raise RuntimeError('cannot encode state')
    state = world()
    state.register_module('broken', Broken())
    with pytest.raises(ValueError, match='broken'):
        state.to_dict()
