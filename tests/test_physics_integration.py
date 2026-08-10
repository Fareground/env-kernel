"""End-to-end: physics + continuous time wired through the loader and engine.

Verifies the world clock actually advances numeric state between agent turns,
in both discrete and continuous modes, via the public load_world path.
"""
import pytest

from fg_env_kernel import (
    ContinuousTemporalModel,
    TimeMode,
    build_world_state,
    load_world,
)
from fg_env_kernel.action import ActionInstance


def _idle_decision(entity_id, perception, valid_actions):
    """Agents do nothing — isolates the autonomous world dynamics."""
    if valid_actions:
        a = valid_actions[0]
        name = a.name if hasattr(a, "name") else a.get("name")
        return ActionInstance(action_name=name, actor_id=entity_id)
    return None


def _logistic_world(mode="discrete"):
    """A herd whose population grows logistically toward carrying capacity K,
    written back onto the herd entity each step."""
    schema = {
        "name": "ecosystem",
        "entity_types": [
            {"name": "ranger", "role": "agent", "properties": []},
            {"name": "herd", "role": "object",
             "properties": [{"name": "size", "type": "float", "default": 10}]},
        ],
        "entities": [
            {"id": "r1", "name": "Ranger", "entity_type": "ranger", "properties": {}},
            {"id": "h1", "name": "Herd", "entity_type": "herd", "properties": {"size": 10}},
        ],
        "actions": [{"name": "observe", "actor_type": "ranger", "resolution_archetype": "deterministic"}],
        "temporal": {"phases": [{"name": "action"}], "mode": mode},
        "physics": {
            "params": {"r": 0.8, "K": 1000.0},
            "substeps": 8,
            "variables": [
                {"name": "population", "value": 10.0,
                 "rate": "r * population * (1 - population / K)", "min": 0.0,
                 "writeback": {"entity_type": "herd", "property": "size", "mode": "broadcast"}},
            ],
        },
    }
    return schema


def test_physics_attached_from_schema():
    state = build_world_state(_logistic_world())
    assert state.physics is not None
    assert "population" in state.physics.variables
    assert "physics" in state.modules


def test_discrete_physics_evolves_between_turns():
    state, engine = load_world(_logistic_world("discrete"), seed=1, decision_fn=_idle_decision)
    engine.max_rounds = 30
    engine.run()
    pop = state.physics.values["population"]
    assert pop > 100.0  # logistic growth happened
    # Writeback pushed the field onto the herd entity.
    assert state.get_entity("h1").get("size") == pop


def test_physics_step_events_emitted():
    events = []
    state, engine = load_world(
        _logistic_world("discrete"), seed=1,
        decision_fn=_idle_decision, on_event=events.append,
    )
    engine.max_rounds = 10
    engine.run()
    physics_events = [e for e in events if e.get("event_type") == "physics_step"]
    assert physics_events, "expected physics_step events to be emitted"
    assert any(e["data"].get("variable") == "population" for e in physics_events)


def test_continuous_mode_builds_model_and_advances_time():
    schema = _logistic_world("continuous")
    schema["temporal"]["continuous"] = {
        "max_time": 30.0, "environment_interval": 1.0, "default_turn_interval": 1.0,
    }
    state, engine = load_world(schema, seed=1, decision_fn=_idle_decision)
    assert isinstance(engine._continuous_time, ContinuousTemporalModel)
    assert state.temporal.mode == TimeMode.CONTINUOUS
    engine.run()
    # The continuous clock advanced and physics integrated along the way.
    assert engine._continuous_time.current_time > 1.0
    assert state.physics.values["population"] > 10.0


def test_continuous_without_env_ticks_holds_physics():
    # environment_interval=0 disables the world clock — physics shouldn't move
    # even though agents take turns.
    schema = _logistic_world("continuous")
    schema["temporal"]["continuous"] = {"max_time": 10.0, "environment_interval": 0.0}
    state, engine = load_world(schema, seed=1, decision_fn=_idle_decision)
    engine.run()
    assert state.physics.values["population"] == 10.0


def test_continuous_pause_resume_preserves_physics_clock():
    # After a pause mid-run, resume must integrate physics by the TRUE gap since
    # the last environment tick, not under-count it. We compare a paused/resumed
    # run against an uninterrupted one — they must land identically.
    def fresh():
        schema = _logistic_world("continuous")
        schema["temporal"]["continuous"] = {"max_time": 20.0, "environment_interval": 1.0}
        return load_world(schema, seed=1, decision_fn=_idle_decision)

    # Uninterrupted reference.
    state_ref, eng_ref = fresh()
    eng_ref.run()
    ref_pop = state_ref.physics.values["population"]

    # Paused once partway, then resumed.
    state_pr, eng_pr = fresh()

    ticks = {"n": 0}
    orig = eng_pr._tick_physics

    def counting_tick(dt):
        ticks["n"] += 1
        if ticks["n"] == 3:
            eng_pr.pause()
        return orig(dt)

    eng_pr._tick_physics = counting_tick
    eng_pr.run()          # runs until pause flips
    eng_pr._tick_physics = orig
    eng_pr.resume()       # finish the run
    pr_pop = state_pr.physics.values["population"]

    assert pr_pop == pytest.approx(ref_pop, rel=1e-9)


def test_discrete_world_without_physics_is_unaffected():
    schema = _logistic_world("discrete")
    del schema["physics"]
    state, engine = load_world(schema, seed=1, decision_fn=_idle_decision)
    assert state.physics is None
    engine.max_rounds = 5
    engine.run()  # must not raise


def test_state_roundtrip_preserves_physics():
    state, engine = load_world(_logistic_world("discrete"), seed=1, decision_fn=_idle_decision)
    engine.max_rounds = 12
    engine.run()
    snap = state.to_dict()
    restored = build_world_state(_logistic_world("discrete"))
    restored.apply_snapshot(snap) if hasattr(restored, "apply_snapshot") else restored.from_dict(snap)
    assert restored.physics is not None
    assert restored.physics.values["population"] == state.physics.values["population"]
