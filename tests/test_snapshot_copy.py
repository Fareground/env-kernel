"""Fast checkpoint copies preserve deepcopy values, aliases and isolation."""
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import random

from fg_env_kernel._snapshot_copy import snapshot_copy


def test_nested_json_is_detached_and_byte_equivalent():
    original = {'entity': {'resources': [1, 2.5, None, True, 'hello'], 'nested': [{'x': 3}]}}
    copied = snapshot_copy(original)
    assert json.dumps(copied, sort_keys=True) == json.dumps(copy.deepcopy(original), sort_keys=True)
    copied['entity']['nested'][0]['x'] = 7
    assert original['entity']['nested'][0]['x'] == 3
    original['entity']['resources'].append('later')
    assert copied['entity']['resources'] == [1, 2.5, None, True, 'hello']


def test_aliases_cycles_and_tuple_fallback_share_one_memo():
    shared = {'values': [1]}
    original = {'left': shared, 'right': [shared], 'tuple': (shared,)}
    original['self'] = original
    shared['root'] = original
    copied = snapshot_copy(original)
    assert copied is copied['self']
    assert copied['left'] is copied['right'][0] is copied['tuple'][0]
    assert copied['left']['root'] is copied
    assert copied['left'] is not shared
    copied['left']['values'].append(2)
    assert shared['values'] == [1]


def test_list_cycle_is_preserved_and_separate_copies_do_not_share_state():
    source = []
    source.append(source)
    first, second = snapshot_copy(source), snapshot_copy(source)
    assert first[0] is first and second[0] is second
    assert first is not second and first is not source


def test_mutually_recursive_tuple_and_list_match_deepcopy():
    items = []
    source = (items,)
    items.append(source)
    result = snapshot_copy(source)
    assert result is not source and result[0] is not items
    assert result[0][0] is result


def test_immutable_rng_tuple_identity_is_preserved_without_generic_copy(monkeypatch):
    state = random.Random(189).getstate()
    assert copy.deepcopy(state) is state
    monkeypatch.setattr(copy, 'deepcopy', lambda *args: (_ for _ in ()).throw(AssertionError('Generic deepcopy not needed')))
    assert snapshot_copy(state) is state


class SpecialDict(dict):
    def __deepcopy__(self, memo):
        return {'custom_dict_hook': True}


class SpecialInt(int):
    def __deepcopy__(self, memo):
        return 'custom_int_hook'


@dataclass
class CustomState:
    value: dict


class Mode(Enum):
    WAIT = 'wait'


def test_custom_objects_subclasses_and_standard_library_protocols_are_preserved():
    shared = {'score': [1]}
    source = {'plain': shared, 'custom': CustomState(shared), 'special': SpecialDict(a=1),
              'number': SpecialInt(2), 'rng': random.Random(189).getstate(),
              'mode': Mode.WAIT, 'when': datetime(2026, 1, 1, tzinfo=timezone.utc),
              'set': {1, 2}, 'tuple_key': {(1, 2): shared}}
    copied = snapshot_copy(source)
    assert copied == copy.deepcopy(source)
    assert copied['plain'] is copied['custom'].value is copied['tuple_key'][(1, 2)]
    assert copied['special'] == {'custom_dict_hook': True}
    assert copied['number'] == 'custom_int_hook'
    assert copied['set'] is not source['set']
    assert copied['custom'] is not source['custom']


def test_random_json_trees_match_deepcopy():
    rng = random.Random(189)
    def tree(depth):
        if not depth or rng.randrange(3) == 0:
            return rng.choice([None, False, True, rng.randrange(100), rng.random(), 'text'])
        if rng.randrange(2):
            return {str(i): tree(depth - 1) for i in range(rng.randrange(6))}
        return [tree(depth - 1) for _ in range(rng.randrange(6))]
    for _ in range(200):
        value = tree(6)
        assert snapshot_copy(value) == copy.deepcopy(value)
