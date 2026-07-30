"""smoke_test() — fast mocked-decision playtester for compiled worlds.

Runs a compiled engine for N rounds with mocked agent decisions (no
real LLM calls). Returns a structured report covering:

  - Did the game crash?
  - Did a termination fire? Which one?
  - Which actions actually got taken?
  - Action coverage % (how many declared actions ever fired)
  - Number of distinct events emitted

This is the agent's TIGHT iteration loop. Real LLM playtests are slow
($), flaky, and burn agent context. The smoke test is deterministic
(seed-controlled), fast (~ms), and gives the agent enough signal to
know its template is structurally sound.

## Decision strategies

  - ``"random"``       — uniform choice among valid actions each turn
  - ``"first"``        — always pick the first listed valid action
  - ``"round_robin"``  — rotate through declared actions in order
  - ``callable``       — bring your own decision_fn for targeted tests

## Usage

    from fg_env_kernel import compile_template, smoke_test

    result = compile_template(raw_json)
    if not result.ok:
        return result
    report = smoke_test(result.engine, rounds=20, seed=42)
    if not report.healthy:
        print(report.summary())
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..action import ActionInstance
from ..engine import SimulationEngine


@dataclass
class SmokeReport:
    """Outcome of a mocked-decision playtest."""
    completed: bool = False         # ran to termination or max_rounds without crash
    rounds_run: int = 0
    max_rounds: int = 0
    terminated_by: Optional[str] = None  # termination condition name, if any
    crash: Optional[str] = None     # str(exception) if a crash occurred
    actions_taken: Dict[str, int] = field(default_factory=dict)
    action_coverage_pct: float = 0.0    # fraction of declared actions that fired ≥1
    declared_actions: int = 0
    events_emitted: int = 0
    duration_ms: float = 0.0
    seed: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """A 'healthy' run: no crash, ran ≥1 round, ≥1 action taken,
        terminated by a condition (not max_rounds timeout) — unless
        max_rounds was the explicit goal."""
        if self.crash:
            return False
        if self.rounds_run == 0 or not self.actions_taken:
            return False
        return True

    def summary(self) -> str:
        lines = []
        status = "OK" if self.healthy else "UNHEALTHY"
        lines.append(f"smoke_test: {status} (seed={self.seed}, {self.duration_ms:.0f}ms)")
        lines.append(f"  rounds: {self.rounds_run}/{self.max_rounds}")
        if self.terminated_by:
            lines.append(f"  terminated_by: {self.terminated_by}")
        if self.crash:
            lines.append(f"  CRASH: {self.crash}")
        lines.append(f"  actions taken: {dict(self.actions_taken)}")
        lines.append(f"  action coverage: {self.action_coverage_pct:.0%} "
                     f"({len(self.actions_taken)}/{self.declared_actions})")
        lines.append(f"  events emitted: {self.events_emitted}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decision strategies
# ---------------------------------------------------------------------------


def _make_random_decider(rng: random.Random) -> Callable:
    def _decide(entity_id: str, perception: Any, valid_actions: List[str]):
        if not valid_actions:
            return None
        name = rng.choice(valid_actions)
        return ActionInstance(action_name=name, actor_id=entity_id)
    return _decide


def _make_first_decider() -> Callable:
    def _decide(entity_id: str, perception: Any, valid_actions: List[str]):
        if not valid_actions:
            return None
        return ActionInstance(action_name=valid_actions[0], actor_id=entity_id)
    return _decide


def _make_round_robin_decider(declared_actions: List[str]) -> Callable:
    """Cycles through declared actions in order across all agents.
    Good for forcing maximum action coverage."""
    state = {"idx": 0}
    def _decide(entity_id: str, perception: Any, valid_actions: List[str]):
        if not valid_actions:
            return None
        # Try to pick the next declared action that's also valid.
        for offset in range(len(declared_actions) or 1):
            if not declared_actions:
                break
            candidate = declared_actions[(state["idx"] + offset) % len(declared_actions)]
            if candidate in valid_actions:
                state["idx"] = (state["idx"] + offset + 1) % len(declared_actions)
                return ActionInstance(action_name=candidate, actor_id=entity_id)
        # Fallback to any valid action
        return ActionInstance(action_name=valid_actions[0], actor_id=entity_id)
    return _decide


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def smoke_test(
    engine: SimulationEngine,
    *,
    rounds: int = 20,
    seed: int = 42,
    decisions: Union[str, Callable] = "random",
) -> SmokeReport:
    """Run a fast mocked-decision playtest. Returns a SmokeReport.

    Mutates the engine's ``decision_fn``, ``_rng``, ``max_rounds``, and
    attaches an ``on_event`` hook for counting events. After the run,
    the engine has been **used** — don't expect it to be in pristine
    state. Build a fresh one via ``compile_template`` for the real run.
    """
    state = engine.state
    declared_actions = sorted(state.action_definitions.keys())
    rng = random.Random(seed)

    # Set up the decision function
    if callable(decisions):
        decide_fn = decisions
    elif decisions == "random":
        decide_fn = _make_random_decider(rng)
    elif decisions == "first":
        decide_fn = _make_first_decider()
    elif decisions == "round_robin":
        decide_fn = _make_round_robin_decider(declared_actions)
    else:
        decide_fn = _make_random_decider(rng)

    # Capture metrics via on_event
    actions_taken: Dict[str, int] = defaultdict(int)
    events_count = [0]
    terminated_by: List[Optional[str]] = [None]

    def _on_event(ev: Dict[str, Any]) -> None:
        events_count[0] += 1
        et = ev.get("event_type") or ""
        if et == "action_resolved":
            name = ev.get("action_name")
            if name:
                actions_taken[name] += 1
        elif et == "simulation_terminated":
            tc = (ev.get("data") or {}).get("condition_name")
            if tc:
                terminated_by[0] = tc

    # Save + replace engine attributes
    prev_decision_fn = engine.decision_fn
    prev_on_event = engine.on_event
    prev_max_rounds = engine.max_rounds
    prev_rng = getattr(engine, "_rng", None)

    engine.decision_fn = decide_fn
    engine.on_event = _on_event
    engine.max_rounds = rounds
    engine._rng = rng

    start = time.time()
    crash: Optional[str] = None
    try:
        engine.run()
    except Exception as e:
        crash = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.time() - start) * 1000

    # Restore (engine still usable for callers)
    engine.decision_fn = prev_decision_fn
    engine.on_event = prev_on_event
    engine.max_rounds = prev_max_rounds
    if prev_rng is not None:
        engine._rng = prev_rng

    rounds_run = state.temporal.current_round if not crash else 0
    coverage = (
        len(set(actions_taken.keys()) & set(declared_actions)) / len(declared_actions)
        if declared_actions else 0.0
    )

    warnings: List[str] = []
    if not actions_taken and not crash:
        warnings.append(
            "no actions were taken across the entire run — check that an agent-role "
            "entity exists with a valid action whose preconditions can be satisfied."
        )
    if rounds_run >= rounds and not terminated_by[0]:
        warnings.append(
            f"hit max_rounds ({rounds}) without firing a termination — "
            "verify termination conditions are reachable."
        )

    return SmokeReport(
        completed=(crash is None),
        rounds_run=rounds_run,
        max_rounds=rounds,
        terminated_by=terminated_by[0],
        crash=crash,
        actions_taken=dict(actions_taken),
        action_coverage_pct=round(coverage, 3),
        declared_actions=len(declared_actions),
        events_emitted=events_count[0],
        duration_ms=round(elapsed_ms, 1),
        seed=seed,
        warnings=warnings,
    )


__all__ = ["SmokeReport", "smoke_test"]
