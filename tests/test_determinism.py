"""Golden-seed determinism guarantee.

The README and engine docstring promise: *"deterministic given the same seed
and LLM responses."* These tests lock that promise in. A run is reproducible
iff two runs with the same seed and the same (deterministic) decision_fn
produce byte-identical event logs — modulo the wall-clock ``timestamp`` field.

If a future change reintroduces nondeterminism (global RNG, set iteration,
completion-order resolution), one of these tests goes red.
"""
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.world_loader import load_world


def _fingerprint(state) -> list:
    """A reproducible fingerprint of the full transcript.

    Excludes ``timestamp`` (wall-clock, never reproducible) but keeps
    everything the simulation logic actually produced.
    """
    out = []
    for ev in state.event_log.get_all():
        d = ev.to_dict()
        d.pop("timestamp", None)
        out.append(d)
    return out


# A world whose outcome depends on the RNG ($random) and on probabilistic
# resolution, so a determinism break actually shows up.
RNG_GAME = {
    "name": "rng_climb",
    "description": "Gamblers race to 100 gold; gamble outcome is RNG-driven.",
    "temporal": {"phases": [{"name": "trade"}]},
    "entity_types": [{
        "name": "Trader",
        "role": "agent",
        "properties": [
            {"name": "gold", "type": "int", "default": 50, "min_value": -1000, "max_value": 1000},
            {"name": "alive", "type": "bool", "default": True},
        ],
    }],
    "actions": [
        {
            "name": "gamble",
            "actor_type": "Trader",
            "resolution_archetype": "skill_check",
            "resolution_params": {"difficulty": 0.5},
            "preconditions": [{"expr": "$actor.alive && $actor.gold >= 10"}],
            "effects_on_success": [
                {"operation": "add", "target": "actor", "field": "gold", "value": 20,
                 "condition": {"expr": "$random(1, 100) > 50"}},
                {"operation": "subtract", "target": "actor", "field": "gold", "value": 5,
                 "condition": {"expr": "$random(1, 100) <= 50"}},
            ],
            "effects_on_failure": [
                {"operation": "subtract", "target": "actor", "field": "gold", "value": 10},
            ],
        },
    ],
    "entities": [
        {"id": "alice", "name": "Alice", "entity_type": "Trader", "properties": {"gold": 50}},
        {"id": "bob", "name": "Bob", "entity_type": "Trader", "properties": {"gold": 50}},
        {"id": "carol", "name": "Carol", "entity_type": "Trader", "properties": {"gold": 50}},
    ],
    "termination_conditions": [
        {"name": "winner", "check_type": "first_to_score",
         "params": {"entity_type": "Trader", "property": "gold", "target": 100}},
    ],
}


def _always_gamble(entity_id, perception, valid_actions):
    if "gamble" in valid_actions:
        return ActionInstance(action_name="gamble", actor_id=entity_id)
    return None


def _run(seed: int):
    state, engine = load_world(RNG_GAME, seed=seed, decision_fn=_always_gamble)
    engine.max_rounds = 60
    engine.run()
    return _fingerprint(state)


def test_same_seed_is_byte_identical():
    """Two runs with the same seed must produce identical transcripts."""
    a = _run(seed=1234)
    b = _run(seed=1234)
    assert a == b, "Same seed produced divergent transcripts — determinism broken."
    assert len(a) > 5, "Sanity: the run should have produced a real transcript."


def test_different_seeds_diverge():
    """Different seeds should (almost always) diverge — proves the RNG is
    actually wired in and the byte-equality test above isn't vacuous."""
    a = _run(seed=1)
    b = _run(seed=99999)
    assert a != b, "Different seeds produced identical transcripts — RNG not threaded?"


def test_repeated_runs_are_stable():
    """Five runs at one seed all agree — guards against rare race-condition
    nondeterminism that a single comparison might miss."""
    runs = [_run(seed=7) for _ in range(5)]
    first = runs[0]
    for i, r in enumerate(runs[1:], start=1):
        assert r == first, f"Run {i} diverged from run 0 at seed 7."


# ---------------------------------------------------------------------------
# Subsystem RNGs must share the engine seed (not their own private RNG).
# These guard the property-dynamics and world-events engines, which used to
# spin up an UNSEEDED random.Random() decoupled from the sim seed.
# ---------------------------------------------------------------------------

def _world_event_run(seed: int):
    """Build a WorldEventEngine with NO rng (the production path) and prove
    the SimulationEngine injects its seeded RNG so the run is reproducible."""
    from fg_env_kernel.state import WorldState
    from fg_env_kernel.temporal import TemporalModel, Phase
    from fg_env_kernel.entity import Entity, EntityType
    from fg_env_kernel.types import PropertySchema, PropertyType
    from fg_env_kernel.action import Effect, EffectOperation
    from fg_env_kernel.world_events import WorldEventDefinition, WorldEventEngine
    from fg_env_kernel.engine import SimulationEngine

    state = WorldState()
    state.temporal = TemporalModel(phases=[Phase(name="action")])
    state.register_entity_type(EntityType(
        name="farmer", role="agent",
        properties=[PropertySchema(name="morale", type=PropertyType.FLOAT,
                                   default=50.0, min_value=0.0, max_value=200.0)],
    ))
    for eid in ("a", "b", "c"):
        state.spawn_entity(Entity(id=eid, name=eid, entity_type="farmer",
                                  properties={"morale": 50.0}))

    # A 50/50 random event — outcome depends entirely on the RNG.
    ev = WorldEventDefinition(
        name="windfall", description="maybe", trigger_type="random",
        probability=0.5, target_type="farmer",
        effects=[Effect(target="actor", operation=EffectOperation.ADD,
                        field="morale", value=5.0)],
    )
    # NOTE: rng intentionally omitted — was the orphaned, unseeded path.
    world_engine = WorldEventEngine([ev])
    engine = SimulationEngine(
        state=state, decision_fn=lambda eid, p, va: None,
        max_rounds=15, seed=seed, world_event_engine=world_engine,
    )
    engine.run()
    return [e.to_dict() | {"timestamp": None}
            for e in state.event_log.get_by_type("world_event")]


def test_world_events_follow_engine_seed():
    a = _world_event_run(seed=2024)
    b = _world_event_run(seed=2024)
    assert a == b, "World events diverged at same seed — subsystem RNG not seeded."


def test_world_events_differ_across_seeds():
    a = _world_event_run(seed=1)
    b = _world_event_run(seed=2)
    # With 15 rounds × 3 agents × p=0.5, two seeds should differ.
    assert a != b, "World events identical across seeds — RNG not actually wired."


def test_unseeded_run_records_a_replayable_seed():
    """An unseeded engine must still be reproducible after the fact: it mints
    and records a seed, and replaying with that seed reproduces the run."""
    from fg_env_kernel.runtime.engine import SimulationEngine
    from fg_env_kernel.state import WorldState

    first = SimulationEngine(WorldState(), max_rounds=3)
    assert isinstance(first.seed, int)
    draws_a = [first._rng.random() for _ in range(5)]

    replay = SimulationEngine(WorldState(), max_rounds=3, seed=first.seed)
    draws_b = [replay._rng.random() for _ in range(5)]
    assert draws_a == draws_b
