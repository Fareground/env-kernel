"""initial_relations loading — both edge spellings must land in the graph.

Regression: the loader only read {from, to, type}, silently dropping the asset
library's {from_entity, to_entity, relation} edges — every authored social
structure (trust, agreement, dependency) was lost at world build time.
"""
from fg_env_kernel.pipeline.loader import build_world_state


def _schema(edges):
    return {
        "name": "rel-test",
        "entity_types": [
            {"name": "person", "role": "agent", "properties": []},
        ],
        "entities": [
            {"id": "a", "name": "A", "entity_type": "person", "properties": {}},
            {"id": "b", "name": "B", "entity_type": "person", "properties": {}},
        ],
        "actions": [
            {"name": "wave", "actor_type": "person", "resolution_archetype": "deterministic"},
        ],
        "temporal": {"phases": [{"name": "action"}]},
        "initial_relations": edges,
    }


def test_short_spelling_loads():
    state = build_world_state(_schema([
        {"from": "a", "to": "b", "type": "trust", "value": 0.7},
    ]))
    assert state.relations.get("a", "b", "trust") == 0.7


def test_asset_library_spelling_loads():
    state = build_world_state(_schema([
        {"from_entity": "a", "to_entity": "b", "relation": "agreement", "value": 0.6},
    ]))
    assert state.relations.get("a", "b", "agreement") == 0.6


def test_malformed_edge_is_skipped_not_fatal():
    state = build_world_state(_schema([
        {"from_entity": "a", "value": 0.5},                       # missing to/type
        {"from": "a", "to": "b", "type": "trust", "value": 0.9},  # good one after
    ]))
    assert state.relations.get("a", "b", "trust") == 0.9
