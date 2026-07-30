"""Tests for kernel/action.py and kernel/resolution.py"""
import pytest
from fg_env_kernel.action import ActionDefinition, ActionInstance, Precondition, Effect, Operator, EffectOperation
from fg_env_kernel.resolution import (
    DeterministicResolution,
    ProbabilisticSkillCheck,
    ContestOpposed,
    DeterministicMath,
    get_resolution,
    RESOLUTION_REGISTRY,
)


class TestActionDefinition:
    def test_create_action(self):
        action = ActionDefinition(
            name="attack",
            description="Attack a target",
            actor_type="warrior",
            target_type="enemy",
            preconditions=[
                Precondition(subject="actor", operator=Operator.GTE, field="stamina", value=10),
            ],
            resolution_archetype="contest",
            resolution_params={"attacker_property": "strength", "defender_property": "defense"},
            effects_on_success=[
                Effect(target="target", operation=EffectOperation.SUBTRACT, field="health", value=20),
                Effect(target="actor", operation=EffectOperation.SUBTRACT, field="stamina", value=10),
            ],
            effects_on_failure=[
                Effect(target="actor", operation=EffectOperation.SUBTRACT, field="stamina", value=5),
            ],
        )
        assert action.name == "attack"
        assert len(action.preconditions) == 1
        assert len(action.effects_on_success) == 2

    def test_action_instance(self):
        inst = ActionInstance(
            action_name="attack",
            actor_id="warrior1",
            target_id="goblin1",
            parameters={"power": 0.8},
            reasoning="The goblin is weak, I should attack.",
        )
        assert inst.action_name == "attack"
        assert inst.reasoning != ""


class TestResolutionArchetypes:
    def test_deterministic_always_succeeds(self):
        r = DeterministicResolution()
        result = r.resolve({}, None, {}, {})
        assert result.success is True
        assert result.magnitude == 1.0

    def test_skill_check_basic(self):
        r = ProbabilisticSkillCheck()
        # High skill, low difficulty should usually succeed
        successes = 0
        for _ in range(100):
            result = r.resolve(
                actor_properties={"skill": 0.95},
                target_properties=None,
                params={"skill_property": "skill", "difficulty": 0.1},
                action_params={},
            )
            if result.success:
                successes += 1
        assert successes > 70  # Should succeed most of the time

    def test_skill_check_impossible(self):
        r = ProbabilisticSkillCheck()
        successes = 0
        for _ in range(100):
            result = r.resolve(
                actor_properties={"skill": 0.01},
                target_properties=None,
                params={"skill_property": "skill", "difficulty": 0.99},
                action_params={},
            )
            if result.success:
                successes += 1
        assert successes < 30  # Should fail most of the time

    def test_contest_basic(self):
        r = ContestOpposed()
        result = r.resolve(
            actor_properties={"strength": 80},
            target_properties={"defense": 20},
            params={"attacker_property": "strength", "defender_property": "defense"},
            action_params={},
        )
        # Strong attacker vs weak defender - details should be populated
        assert "attacker_roll" in result.details

    def test_contest_no_defender(self):
        r = ContestOpposed()
        result = r.resolve(
            actor_properties={"strength": 50},
            target_properties=None,
            params={},
            action_params={},
        )
        assert result.success is True

    def test_deterministic_math(self):
        r = DeterministicMath()
        result = r.resolve(
            actor_properties={},
            target_properties=None,
            params={},
            action_params={"amount": 42},
        )
        assert result.success is True
        assert result.magnitude == 42.0

    def test_get_resolution_valid(self):
        r = get_resolution("deterministic")
        assert isinstance(r, DeterministicResolution)

    def test_get_resolution_invalid(self):
        with pytest.raises(ValueError):
            get_resolution("nonexistent")

    def test_registry_completeness(self):
        assert "deterministic" in RESOLUTION_REGISTRY
        assert "skill_check" in RESOLUTION_REGISTRY
        assert "contest" in RESOLUTION_REGISTRY
        assert "deterministic_math" in RESOLUTION_REGISTRY
