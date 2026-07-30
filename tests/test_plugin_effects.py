"""Integration: prove a custom effect op can be registered and dispatched
by the engine without modifying kernel source.

This is the contract underpinning the "configure ANY game" goal — an
env-builder agent should be able to introduce a brand-new verb just by
calling ``@effect("my_op")`` from outside the kernel package.
"""
import pytest

from fg_env_kernel import EffectContext, effect, registry
from fg_env_kernel.action import (
    ActionDefinition,
    Effect,
    EffectOperation,
)
from fg_env_kernel.engine import SimulationEngine, _coerce_effects
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.resolution import ResolutionResult
from fg_env_kernel.state import WorldState


@pytest.fixture
def minimal_world():
    """A 2-entity world with a single 'score' integer property."""
    state = WorldState()
    et = EntityType(name="player", role="agent")
    state.entity_types["player"] = et
    a = Entity(id="a", name="A", entity_type="player", properties={"score": 0})
    b = Entity(id="b", name="B", entity_type="player", properties={"score": 0})
    state.entities["a"] = a
    state.entities["b"] = b
    return state, a, b


def test_custom_op_dispatches_via_registry(minimal_world):
    """A handler registered via @effect is invoked by _apply_effects."""
    state, a, b = minimal_world

    @effect("__test_grant_score", replace=True)
    def _grant_score(ctx: EffectContext, spec: dict):
        amount = int(ctx.resolve(spec.get("value")) or 0)
        target = ctx.target
        if target is None:
            return None
        old = target.get("score")
        target.set("score", (old or 0) + amount)
        return {"entity": target.id, "field": "score", "old": old, "new": target.get("score")}

    try:
        engine = SimulationEngine(state)
        # Build an effect referencing our custom op as a STRING.
        # _coerce_effects must accept the unknown-to-enum op because
        # it's registered. Then _apply_effects routes via the registry.
        coerced = _coerce_effects([{
            "operation": "__test_grant_score",
            "target": "actor",
            "value": 7,
        }])
        assert len(coerced) == 1
        # operation is stored as a lowercase string for registry ops
        assert coerced[0].operation == "__test_grant_score"

        result = ResolutionResult(success=True, magnitude=1.0)
        changes = engine._apply_effects(coerced, actor=a, target=b, params={}, result=result)

        assert a.get("score") == 7
        assert any(c.get("entity") == "a" and c.get("new") == 7 for c in changes)
    finally:
        registry.effects.unregister("__test_grant_score")


def test_unknown_op_string_is_dropped_silently(minimal_world):
    """If an op string isn't in the registry, _coerce_effects skips it
    (matches existing behavior for unknown enum values)."""
    coerced = _coerce_effects([{"operation": "definitely_not_registered_xyz"}])
    assert coerced == []


def test_builtin_ops_still_work_after_registry_addition(minimal_world):
    """Adding a plugin op must not affect built-in EffectOperation paths."""
    state, a, b = minimal_world

    @effect("__test_noop", replace=True)
    def _noop(ctx, spec):
        return None

    try:
        engine = SimulationEngine(state)
        # Use the built-in SET op (enum value, not string).
        effect_obj = Effect(
            target="actor",
            operation=EffectOperation.SET,
            field="score",
            value=42,
        )
        result = ResolutionResult(success=True, magnitude=1.0)
        engine._apply_effects([effect_obj], actor=a, target=b, params={}, result=result)
        assert a.get("score") == 42
    finally:
        registry.effects.unregister("__test_noop")


def test_handler_exception_is_swallowed_with_log(minimal_world, caplog):
    """A buggy plugin must not crash the engine round."""
    state, a, b = minimal_world

    @effect("__test_explodes", replace=True)
    def _boom(ctx, spec):
        raise RuntimeError("boom")

    try:
        engine = SimulationEngine(state)
        coerced = _coerce_effects([{"operation": "__test_explodes", "target": "actor"}])
        result = ResolutionResult(success=True, magnitude=1.0)
        # Should not raise
        changes = engine._apply_effects(coerced, actor=a, target=b, params={}, result=result)
        # And should produce no changes
        assert changes == []
    finally:
        registry.effects.unregister("__test_explodes")


def test_handler_receives_resolved_expression_values(minimal_world):
    """Effect handler sees post-expression values via ctx.resolve."""
    state, a, b = minimal_world
    a.set("score", 10)
    captured = {}

    @effect("__test_capture", replace=True)
    def _cap(ctx, spec):
        captured["value"] = ctx.resolve(spec.get("value"))
        return None

    try:
        engine = SimulationEngine(state)
        coerced = _coerce_effects([{
            "operation": "__test_capture",
            "target": "actor",
            "value": "$actor.score",
        }])
        result = ResolutionResult(success=True, magnitude=1.0)
        engine._apply_effects(coerced, actor=a, target=b, params={}, result=result)
        # value will already be resolved by _apply_effects before
        # handing to the handler — captured value should equal 10.
        assert captured["value"] == 10
    finally:
        registry.effects.unregister("__test_capture")
