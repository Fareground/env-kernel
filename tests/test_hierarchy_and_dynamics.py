"""Faction hierarchy (same_org) + declarative property dynamics."""
from fg_env_kernel.factions import Faction, FactionManager
from fg_env_kernel.pipeline.loader import build_world_state


def _mgr() -> FactionManager:
    m = FactionManager()
    m.register(Faction(id="acme", name="Acme Corp"))
    m.register(Faction(id="sales", name="Sales", parent="acme",
                       member_ids=["alice"]))
    m.register(Faction(id="eng", name="Engineering", parent="acme",
                       member_ids=["bob"]))
    m.register(Faction(id="rival", name="Rival Inc", member_ids=["carol"]))
    return m


class TestHierarchy:
    def test_org_root_walks_parents(self):
        m = _mgr()
        assert m.org_root("sales") == "acme"
        assert m.org_root("acme") == "acme"
        assert m.org_root("rival") == "rival"

    def test_same_org_spans_divisions_same_faction_does_not(self):
        m = _mgr()
        assert m.same_org("alice", "bob")          # sales + eng share acme
        assert not m.same_org("alice", "carol")    # different companies
        # exact membership stays exact — existing semantics untouched
        assert m.faction_of("alice") != m.faction_of("bob")

    def test_cycle_safety(self):
        m = FactionManager()
        m.register(Faction(id="a", name="A", parent="b"))
        m.register(Faction(id="b", name="B", parent="a"))
        assert m.org_root("a") is not None  # returns, never hangs


class TestPropertyDynamicsLoading:
    SCHEMA = {
        "name": "drift world",
        "entity_types": [
            {"name": "Trader", "role": "agent", "properties": [
                {"name": "sentiment", "type": "float", "default": 0.5},
            ]},
        ],
        "entities": [
            {"id": "t1", "entity_type": "Trader", "name": "T1",
             "properties": {"sentiment": 0.5}},
        ],
        "actions": [],
        "property_dynamics": {
            "drift_rules": [
                {"name": "sentiment_decay", "target_type": "Trader",
                 "property_field": "sentiment", "drift_type": "mean_revert",
                 "rate": 0.2, "mean": 0.0, "min_value": 0.0, "max_value": 1.0},
            ],
        },
    }

    def test_the_template_key_loads_the_engine(self):
        state = build_world_state(self.SCHEMA)
        assert state.property_dynamics is not None
        assert state.property_dynamics.drift_rules[0].name == "sentiment_decay"

    def test_drift_actually_moves_the_property(self):
        import random

        state = build_world_state(self.SCHEMA)
        before = state.get_entity("t1").properties["sentiment"]
        state.property_dynamics.tick(state, 1)
        after = state.get_entity("t1").properties["sentiment"]
        assert after < before  # mean-reverting toward 0

    def test_absent_key_stays_none(self):
        schema = {k: v for k, v in self.SCHEMA.items()
                  if k != "property_dynamics"}
        state = build_world_state(schema)
        assert state.property_dynamics is None


class TestTimeMappingLoads:
    def test_temporal_time_fields_load_from_the_schema(self):
        schema = {
            "name": "timed world",
            "entity_types": [{"name": "A", "role": "agent", "properties": []}],
            "entities": [{"id": "a1", "entity_type": "A", "name": "A1",
                          "properties": {}}],
            "actions": [],
            "temporal": {"round_duration_seconds": 86400,
                         "time_unit_label": "day",
                         "sim_start_iso": "2026-08-01T00:00:00Z"},
        }
        state = build_world_state(schema)
        assert state.temporal.round_duration_seconds == 86400
        assert state.temporal.round_to_time_label(30) == "Day 30"

    def test_absent_mapping_stays_legacy(self):
        schema = {
            "name": "plain", "entity_types": [], "entities": [], "actions": [],
        }
        state = build_world_state(schema)
        assert state.temporal.round_duration_seconds is None
