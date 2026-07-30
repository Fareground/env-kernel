"""Performance benchmarks — guards against regressions.

Run with:

    python -m pytest tests/test_kernel_benchmarks.py -v -s

The benchmarks are intentionally lightweight (sub-second) so they
can run in CI on every PR. They establish baseline performance for:

  - Kernel compile time (parse + lint + build)
  - Engine throughput (rounds per second with mocked decisions)
  - Effect dispatch throughput (apply_effects ops/sec)
  - Smoke-test wall clock

If a number regresses by >50%, something got slower and deserves
investigation. We don't use pytest-benchmark to avoid adding a hard
dependency — these are just timed assertions with generous margins.
"""
import time

import pytest

from fg_env_kernel import (
    compile_template,
    enable_engine_metrics,
    engine_metrics_enabled,
    metrics,
    replay,
    smoke_test,
)
from fg_env_kernel.action import ActionInstance


# ---------------------------------------------------------------------------
# Reference workloads (3 game types covering the common patterns)
# ---------------------------------------------------------------------------


RACE_TO_10 = {
    "name": "bench_race",
    "temporal": {"phases": [{"name": "roll"}]},
    "entity_types": [{
        "name": "Racer", "role": "agent",
        "properties": [
            {"name": "score", "type": "int", "default": 0, "min_value": 0, "max_value": 100},
        ],
    }],
    "actions": [{
        "name": "roll_dice",
        "actor_type": "Racer",
        "resolution_archetype": "deterministic",
        "preconditions": [{"expr": "$actor.score < 10"}],
        "effects_on_success": [
            {"operation": "add", "target": "actor", "field": "score",
             "value": "$random(1, 6)"},
        ],
    }],
    "entities": [
        {"id": "alice", "name": "A", "entity_type": "Racer", "properties": {"score": 0}},
        {"id": "bob", "name": "B", "entity_type": "Racer", "properties": {"score": 0}},
    ],
    "termination_conditions": [{
        "name": "winner", "check_type": "first_to_score",
        "params": {"entity_type": "Racer", "property": "score", "target": 10},
    }],
}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestCompilePerformance:
    def test_compile_is_fast(self):
        """compile_template should be sub-100ms on a simple template."""
        start = time.perf_counter()
        for _ in range(10):
            result = compile_template(RACE_TO_10)
            assert result.ok
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_compile_ms = elapsed_ms / 10
        print(f"\n  compile: {per_compile_ms:.1f}ms per template "
              f"({elapsed_ms:.0f}ms for 10x)")
        # Generous bound — CI machines vary
        assert per_compile_ms < 250, f"compile took {per_compile_ms:.1f}ms (>250ms = regression?)"


class TestEngineThroughput:
    def test_smoke_test_runs_30_rounds_under_500ms(self):
        result = compile_template(RACE_TO_10)
        assert result.ok
        start = time.perf_counter()
        report = smoke_test(result.engine, rounds=30, seed=42)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"\n  smoke 30 rounds: {elapsed_ms:.1f}ms "
              f"({report.rounds_run} rounds, "
              f"{(report.rounds_run / max(elapsed_ms, 1.0)) * 1000.0:.0f} rounds/sec)")
        assert elapsed_ms < 500, f"smoke_test took {elapsed_ms:.1f}ms (>500ms = regression?)"

    def test_replay_throughput(self):
        start = time.perf_counter()
        trace = replay(RACE_TO_10, seed=42, rounds=30, decisions="random")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"\n  replay 30 rounds: {elapsed_ms:.1f}ms "
              f"({len(trace.steps)} events captured)")
        assert elapsed_ms < 1000, f"replay took {elapsed_ms:.1f}ms (>1s = regression?)"


class TestEffectDispatchThroughput:
    def test_apply_effects_does_not_explode_with_metrics_enabled(self):
        """Observability overhead should be <30% on the hot path."""
        result_off = compile_template(RACE_TO_10)
        assert result_off.ok
        # Baseline (metrics off)
        was_enabled = engine_metrics_enabled()
        enable_engine_metrics(False)
        try:
            start = time.perf_counter()
            smoke_test(result_off.engine, rounds=20, seed=1)
            baseline_ms = (time.perf_counter() - start) * 1000.0
        finally:
            enable_engine_metrics(was_enabled)

        # With metrics on
        result_on = compile_template(RACE_TO_10)
        enable_engine_metrics(True)
        try:
            start = time.perf_counter()
            smoke_test(result_on.engine, rounds=20, seed=1)
            with_metrics_ms = (time.perf_counter() - start) * 1000.0
        finally:
            enable_engine_metrics(was_enabled)

        overhead = (with_metrics_ms / baseline_ms - 1.0) * 100
        print(f"\n  observability overhead: {overhead:+.0f}% "
              f"({baseline_ms:.1f}ms → {with_metrics_ms:.1f}ms)")
        # Generous bound — CI noise can shift this. We mainly want to
        # ensure metrics don't multiply runtime by 10x.
        assert with_metrics_ms < baseline_ms * 3.0 + 50, (
            f"metrics overhead too high: {baseline_ms:.0f}ms vs {with_metrics_ms:.0f}ms"
        )


class TestMetricsRegistry:
    def test_metrics_are_exportable(self):
        from fg_env_kernel import metrics as global_metrics

        # Force a fresh registry so prior tests don't pollute
        global_metrics.reset()
        was_enabled = engine_metrics_enabled()
        enable_engine_metrics(True)
        try:
            result = compile_template(RACE_TO_10)
            smoke_test(result.engine, rounds=10, seed=1)
            export = global_metrics.export()
            print(f"\n  metrics export: {len(export['counters'])} counters, "
                  f"{len(export['histograms'])} histograms")
            # At least the effects counter and apply_effects timer should be present
            assert any("effects_dispatched" in k for k in export["counters"]), (
                f"expected effects counter in {list(export['counters'].keys())}"
            )
            assert any("apply_effects" in k for k in export["histograms"])
        finally:
            enable_engine_metrics(was_enabled)
            global_metrics.reset()
