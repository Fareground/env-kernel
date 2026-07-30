"""Tests for the KernelRegistry plugin system."""
import pytest

from fg_env_kernel.registry import (
    KernelRegistry,
    registry,
    effect,
    precondition,
    resolution,
)


class TestNamespaceBasics:
    def test_register_and_get(self):
        r = KernelRegistry()
        r.effects.register("test_op", lambda ctx, spec: None)
        assert r.effects.has("test_op")
        assert callable(r.effects.get("test_op"))

    def test_case_insensitive_keys(self):
        r = KernelRegistry()
        r.effects.register("Foo", "bar")
        assert r.effects.get("foo") == "bar"
        assert r.effects.get("FOO") == "bar"

    def test_aliases_resolve(self):
        r = KernelRegistry()
        r.effects.register("post_content", "impl", aliases=["POST_CONTENT", "post"])
        assert r.effects.get("post") == "impl"
        assert r.effects.get("POST_CONTENT") == "impl"

    def test_duplicate_registration_rejected(self):
        r = KernelRegistry()
        r.effects.register("op", "a")
        with pytest.raises(ValueError):
            r.effects.register("op", "b")

    def test_replace_allowed_with_flag(self):
        r = KernelRegistry()
        r.effects.register("op", "a")
        r.effects.register("op", "b", replace=True)
        assert r.effects.get("op") == "b"

    def test_missing_key_raises_keyerror_with_help(self):
        r = KernelRegistry()
        r.effects.register("only_op", "x")
        with pytest.raises(KeyError, match="only_op"):
            r.effects.get("missing")

    def test_try_get_returns_default(self):
        r = KernelRegistry()
        assert r.effects.try_get("missing") is None
        assert r.effects.try_get("missing", "fallback") == "fallback"

    def test_contains_and_iter(self):
        r = KernelRegistry()
        r.effects.register("a", 1)
        r.effects.register("b", 2)
        assert "a" in r.effects
        assert "missing" not in r.effects
        assert set(iter(r.effects)) >= {"a", "b"}


class TestScopedOverrides:
    def test_scoped_falls_back_to_parent(self):
        r = KernelRegistry()
        r.effects.register("global_op", "parent_impl")
        with r.scoped() as child:
            assert child.effects.get("global_op") == "parent_impl"

    def test_scoped_override_does_not_leak(self):
        r = KernelRegistry()
        r.effects.register("op", "parent")
        with r.scoped() as child:
            child.effects.register("op", "child_override")
            assert child.effects.get("op") == "child_override"
            assert r.effects.get("op") == "parent"
        # After scope exits, parent untouched
        assert r.effects.get("op") == "parent"

    def test_scoped_child_local_unregister_does_not_affect_parent(self):
        r = KernelRegistry()
        r.effects.register("op", "parent")
        with r.scoped() as child:
            child.effects.register("op", "child", replace=True)
            child.effects.unregister("op")
            # Falls back to parent
            assert child.effects.get("op") == "parent"


class TestDecorators:
    def test_effect_decorator_registers(self):
        @effect("__test_decorator_op")
        def _h(ctx, spec):
            return {"ok": True}

        assert registry.effects.has("__test_decorator_op")
        # Cleanup so global state stays clean
        registry.effects.unregister("__test_decorator_op")

    def test_decorator_aliases(self):
        @effect("__test_aliased", aliases=["__test_alias_a"])
        def _h(ctx, spec):
            return None

        assert registry.effects.get("__test_alias_a") is _h
        registry.effects.unregister("__test_aliased")

    def test_precondition_decorator(self):
        @precondition("__test_pc")
        def _p(state, actor, target, cond):
            return True

        assert registry.preconditions.has("__test_pc")
        registry.preconditions.unregister("__test_pc")

    def test_resolution_decorator(self):
        @resolution("__test_res")
        class R:
            pass

        assert registry.resolutions.has("__test_res")
        registry.resolutions.unregister("__test_res")


class TestNamespaceIsolation:
    def test_effects_and_preconditions_independent(self):
        r = KernelRegistry()
        r.effects.register("same_name", "effect_impl")
        r.preconditions.register("same_name", "pc_impl")
        assert r.effects.get("same_name") == "effect_impl"
        assert r.preconditions.get("same_name") == "pc_impl"


class TestEffectContext:
    def test_context_constructible(self):
        from fg_env_kernel.effect_context import EffectContext
        ctx = EffectContext(state=None)  # type: ignore[arg-type]
        assert ctx.changes == []
        ctx.record({"op": "test"})
        assert ctx.changes == [{"op": "test"}]

    def test_resolve_passthrough_for_non_expression(self):
        from fg_env_kernel.effect_context import EffectContext
        ctx = EffectContext(state=None)  # type: ignore[arg-type]
        assert ctx.resolve(42) == 42
        assert ctx.resolve("plain_string") == "plain_string"
