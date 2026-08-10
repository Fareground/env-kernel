"""Tests for kernel/engine.py -- SimulationEngine with mock decisions."""
from fg_env_kernel.state import WorldState
from fg_env_kernel.engine import SimulationEngine
from fg_env_kernel.entity import EntityType, Entity
from fg_env_kernel.resource import ResourceType
from fg_env_kernel.action import (
    ActionDefinition, ActionInstance, Precondition, Effect,
    Operator, EffectOperation,
)
from fg_env_kernel.types import PropertySchema, PropertyType


def _build_simple_world() -> WorldState:
    """Build a minimal world with 2 agents for testing."""
    state = WorldState()

    # Entity type
    warrior_type = EntityType(
        name="warrior",
        role="agent",
        properties=[
            PropertySchema(name="health", type=PropertyType.FLOAT, default=100.0, min_value=0, max_value=100),
            PropertySchema(name="strength", type=PropertyType.INT, default=10),
            PropertySchema(name="stamina", type=PropertyType.FLOAT, default=50.0, min_value=0, max_value=100),
        ],
    )
    state.register_entity_type(warrior_type)

    # Resource
    state.register_resource_type(ResourceType(name="gold", conservation=True, discrete=True))
    state.resources["gold"].holdings = {"w1": 50, "w2": 50}

    # Entities
    state.spawn_entity(Entity(id="w1", name="Alice", entity_type="warrior", properties={"health": 100.0, "strength": 15, "stamina": 50.0}))
    state.spawn_entity(Entity(id="w2", name="Bob", entity_type="warrior", properties={"health": 100.0, "strength": 10, "stamina": 50.0}))

    # Actions
    state.register_action(ActionDefinition(
        name="attack",
        description="Attack another warrior",
        actor_type="warrior",
        target_type="warrior",
        preconditions=[
            Precondition(subject="actor", operator=Operator.GTE, field="stamina", value=10),
        ],
        resolution_archetype="contest",
        resolution_params={"attacker_property": "strength", "defender_property": "strength"},
        effects_on_success=[
            Effect(target="target", operation=EffectOperation.SUBTRACT, field="health", value=20),
            Effect(target="actor", operation=EffectOperation.SUBTRACT, field="stamina", value=10),
        ],
        effects_on_failure=[
            Effect(target="actor", operation=EffectOperation.SUBTRACT, field="stamina", value=5),
        ],
    ))

    state.register_action(ActionDefinition(
        name="rest",
        description="Rest to recover stamina",
        actor_type="warrior",
        resolution_archetype="deterministic",
        effects_on_success=[
            Effect(target="actor", operation=EffectOperation.ADD, field="stamina", value=15),
        ],
    ))

    return state


class TestSimulationEngine:
    def test_run_with_no_decision_fn(self):
        """Engine should run without error even without a decision function."""
        state = _build_simple_world()
        engine = SimulationEngine(state=state, max_rounds=3)
        result = engine.run()
        assert result.temporal.current_round == 3
        # No actions taken
        action_events = [e for e in result.event_log.get_all() if e.event_type == "action_resolved"]
        assert len(action_events) == 0

    def test_run_with_mock_always_rest(self):
        """All agents always rest."""
        state = _build_simple_world()

        def always_rest(entity_id, perception, valid_actions):
            if "rest" in valid_actions:
                return ActionInstance(action_name="rest", actor_id=entity_id, reasoning="I need to rest.")
            return None

        engine = SimulationEngine(state=state, decision_fn=always_rest, max_rounds=3, seed=42)
        result = engine.run()

        # Both agents should have rested every round
        action_events = [e for e in result.event_log.get_all() if e.event_type == "action_resolved"]
        assert len(action_events) == 6  # 2 agents * 3 rounds

        # Stamina should have increased (but capped at 100)
        w1 = result.get_entity("w1")
        assert w1.get("stamina") == 95.0  # 50 + (15 * 3) = 95

    def test_run_with_mock_always_attack(self):
        """Agents always attack each other."""
        state = _build_simple_world()
        agents = list(state.entities.keys())

        def always_attack(entity_id, perception, valid_actions):
            if "attack" not in valid_actions:
                return None
            # Target the other warrior
            target = [a for a in agents if a != entity_id][0]
            return ActionInstance(
                action_name="attack",
                actor_id=entity_id,
                target_id=target,
                reasoning="Attack!",
            )

        engine = SimulationEngine(state=state, decision_fn=always_attack, max_rounds=5, seed=42)
        result = engine.run()

        # Actions should have been attempted
        attempted = [e for e in result.event_log.get_all() if e.event_type == "action_attempted"]
        assert len(attempted) > 0

        # Health should have decreased for at least one entity
        w1 = result.get_entity("w1")
        w2 = result.get_entity("w2")
        total_health = w1.get("health") + w2.get("health")
        assert total_health < 200.0  # At least some damage done

    def test_stamina_precondition_blocks_action(self):
        """Agent can't attack when stamina too low."""
        state = _build_simple_world()
        # Set stamina very low
        state.entities["w1"].set("stamina", 5.0)

        valid = state.get_valid_actions("w1")
        assert "attack" not in valid  # Stamina < 10
        assert "rest" in valid

    def test_stop_simulation(self):
        """Engine can be stopped mid-run."""
        state = _build_simple_world()
        rounds_seen = []

        def track_and_stop(entity_id, perception, valid_actions):
            rounds_seen.append(perception.get("round", 0))
            return None

        engine = SimulationEngine(state=state, decision_fn=track_and_stop, max_rounds=100)
        # Stop after round 2 via callback
        engine.on_round_end = lambda r, s: engine.stop() if r >= 2 else None
        result = engine.run()

        assert result.temporal.current_round <= 3

    def test_event_log_populated(self):
        """Event log should contain round_start, round_end, simulation_start, simulation_end."""
        state = _build_simple_world()
        engine = SimulationEngine(state=state, max_rounds=2)
        result = engine.run()

        event_types = [e.event_type for e in result.event_log.get_all()]
        assert "simulation_start" in event_types
        assert "simulation_end" in event_types
        assert "round_start" in event_types
        assert "round_end" in event_types

    def test_seed_reproducibility(self):
        """Same seed should produce same results."""
        def always_attack(entity_id, perception, valid_actions):
            if "attack" not in valid_actions:
                return None
            target = "w2" if entity_id == "w1" else "w1"
            return ActionInstance(action_name="attack", actor_id=entity_id, target_id=target)

        state1 = _build_simple_world()
        engine1 = SimulationEngine(state=state1, decision_fn=always_attack, max_rounds=5, seed=123)
        result1 = engine1.run()
        h1_w1 = result1.get_entity("w1").get("health")

        state2 = _build_simple_world()
        engine2 = SimulationEngine(state=state2, decision_fn=always_attack, max_rounds=5, seed=123)
        result2 = engine2.run()
        h2_w1 = result2.get_entity("w1").get("health")

        assert h1_w1 == h2_w1
