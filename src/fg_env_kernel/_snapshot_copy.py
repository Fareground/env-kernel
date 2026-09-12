"""Detach checkpoint trees without generic deepcopy dispatch for JSON leaves.

Only exact built-in types take the fast path. Subclasses and custom objects
retain their deepcopy protocol. One memo preserves aliases/cycles across both
paths; no mutable value is shared with the live world or another checkpoint.
"""
import copy
from typing import Any

_ATOMIC = frozenset((type(None), bool, int, float, str, bytes))


def snapshot_copy(value: Any, memo: dict | None = None) -> Any:
    kind = type(value)
    if kind in _ATOMIC:
        return value
    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if kind is dict:
        result: Any = {}
        memo[identity] = result
        for key, item in value.items():
            result[snapshot_copy(key, memo)] = snapshot_copy(item, memo)
        return result
    if kind is list:
        result = []
        memo[identity] = result
        result.extend(snapshot_copy(item, memo) for item in value)
        return result
    if kind is tuple:
        # Same recursion rule as deepcopy's tuple handler: a nested mutable
        # child may have completed this tuple while traversing a cycle.
        items = [snapshot_copy(item, memo) for item in value]
        if identity in memo:
            return memo[identity]
        if all(before is after for before, after in zip(value, items)):
            return value  # e.g. immutable RNG state; never live mutable state.
        result = tuple(items)
        memo[identity] = result
        return result
    return copy.deepcopy(value, memo)
