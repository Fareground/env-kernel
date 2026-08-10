"""Replay — deterministic step-by-step trace of a simulation run.

When ``smoke_test`` returns ``unhealthy``, the agent (or human dev)
needs to know **why** — what action did each entity take, what events
fired, what state changed. ``replay`` runs the same template with
the same seed and decision strategy, capturing a structured trace.

Determinism guarantee: given the same ``(template, seed, decisions)``
tuple, ``replay`` produces the same ``ReplayTrace`` every time. This
is the foundation for reproducible debugging.

## Usage

    from fg_env_kernel import compile_template, replay

    raw = json.load(open("mygame.json"))
    trace = replay(raw, seed=42, rounds=20, decisions="random")
    print(trace.summary())
    for step in trace.steps:
        print(step)

## Output

``ReplayTrace`` is JSON-serializable so the agent can ingest it,
diff successful vs failed runs, or pipe it to a UI for visualization.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union



@dataclass
class ReplayStep:
    """One event in the run's transcript."""
    round_number: int
    event_type: str
    actor_id: Optional[str] = None
    target_id: Optional[str] = None
    action_name: Optional[str] = None
    narrative: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    state_delta: Dict[str, Any] = field(default_factory=dict)
    timestamp_ms: float = 0.0


@dataclass
class ReplayTrace:
    """Full step-by-step record of a simulation run.

    Use ``to_dict()`` to serialize for storage or transport. Use
    ``summary()`` for a human-readable digest."""
    template_name: str
    seed: int
    decisions_strategy: str
    rounds_run: int
    completed: bool
    terminated_by: Optional[str]
    crash: Optional[str]
    duration_ms: float
    steps: List[ReplayStep] = field(default_factory=list)
    actions_taken: Dict[str, int] = field(default_factory=dict)
    final_entity_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "seed": self.seed,
            "decisions_strategy": self.decisions_strategy,
            "rounds_run": self.rounds_run,
            "completed": self.completed,
            "terminated_by": self.terminated_by,
            "crash": self.crash,
            "duration_ms": self.duration_ms,
            "actions_taken": dict(self.actions_taken),
            "final_entity_state": dict(self.final_entity_state),
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        lines = []
        lines.append(f"replay: {self.template_name} (seed={self.seed}, "
                     f"strategy={self.decisions_strategy})")
        lines.append(f"  ran {self.rounds_run} rounds in {self.duration_ms:.0f}ms")
        if self.crash:
            lines.append(f"  CRASH: {self.crash}")
        elif self.terminated_by:
            lines.append(f"  terminated_by: {self.terminated_by}")
        else:
            lines.append("  reached max_rounds without termination")
        lines.append(f"  actions: {dict(self.actions_taken)}")
        lines.append(f"  events: {len(self.steps)}")
        lines.append("\n  Recent steps:")
        for s in self.steps[-10:]:
            head = f"    r{s.round_number} [{s.event_type}]"
            if s.actor_id:
                head += f" actor={s.actor_id}"
            if s.action_name:
                head += f" action={s.action_name}"
            lines.append(head)
            if s.narrative:
                lines.append(f"      {s.narrative}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replay(
    template: Union[Dict[str, Any], Any],
    *,
    seed: int = 42,
    rounds: int = 50,
    decisions: Union[str, Callable] = "random",
) -> ReplayTrace:
    """Run a template deterministically and capture a full trace.

    The output is reproducible: same inputs → same trace, byte-for-byte
    in the ``to_dict()`` form. Use this to debug failing smoke tests
    or compare two template versions.

    Parameters:
      template — raw dict OR a pre-validated WorldTemplate
      seed     — RNG seed (default 42)
      rounds   — max rounds to run
      decisions — "random" | "first" | "round_robin" | Callable

    Returns a ``ReplayTrace`` with every emitted event captured."""
    # Import lazily to avoid circular deps
    from .compile import compile_template
    from .smoke import (
        _make_random_decider,
        _make_first_decider,
        _make_round_robin_decider,
    )

    raw = template if isinstance(template, dict) else template.model_dump()
    template_name = raw.get("name", "untitled")

    result = compile_template(raw, seed=seed)
    if not result.ok:
        # Can't run — return a trace describing the compile failure
        return ReplayTrace(
            template_name=template_name,
            seed=seed,
            decisions_strategy=str(decisions),
            rounds_run=0,
            completed=False,
            terminated_by=None,
            crash="compile_failed: " + "; ".join(
                f"{e.path}: {e.message}" for e in result.errors
            ),
            duration_ms=0.0,
        )

    engine = result.engine
    state = result.state
    rng = random.Random(seed)
    declared_actions = sorted(state.action_definitions.keys())

    if callable(decisions):
        decide_fn = decisions
        strategy_name = "custom"
    elif decisions == "first":
        decide_fn = _make_first_decider()
        strategy_name = "first"
    elif decisions == "round_robin":
        decide_fn = _make_round_robin_decider(declared_actions)
        strategy_name = "round_robin"
    else:
        decide_fn = _make_random_decider(rng)
        strategy_name = "random"

    steps: List[ReplayStep] = []
    actions_taken: Dict[str, int] = {}
    terminated_by: List[Optional[str]] = [None]
    start_time = time.perf_counter()

    def _on_event(ev: Dict[str, Any]) -> None:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        event_type = ev.get("event_type", "")
        action_name = ev.get("action_name")
        if event_type == "action_resolved" and action_name:
            actions_taken[action_name] = actions_taken.get(action_name, 0) + 1
        if event_type == "simulation_terminated":
            tc = (ev.get("data") or {}).get("condition_name")
            if tc:
                terminated_by[0] = tc
        steps.append(ReplayStep(
            round_number=ev.get("round_number", 0),
            event_type=event_type,
            actor_id=ev.get("actor_id"),
            target_id=ev.get("target_id"),
            action_name=action_name,
            narrative=ev.get("narrative", ""),
            data=dict(ev.get("data") or {}),
            timestamp_ms=round(elapsed, 2),
        ))

    engine.decision_fn = decide_fn
    engine.on_event = _on_event
    engine.max_rounds = rounds
    engine._rng = rng

    crash: Optional[str] = None
    try:
        engine.run()
    except Exception as e:
        crash = f"{type(e).__name__}: {e}"

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Capture final entity state for diffing
    final_entity_state = {
        eid: {
            "alive": e.alive,
            "location_id": e.location_id,
            "properties": dict(e.properties),
        }
        for eid, e in state.entities.items()
    }

    return ReplayTrace(
        template_name=template_name,
        seed=seed,
        decisions_strategy=strategy_name,
        rounds_run=state.temporal.current_round if not crash else 0,
        completed=(crash is None),
        terminated_by=terminated_by[0],
        crash=crash,
        duration_ms=round(duration_ms, 1),
        steps=steps,
        actions_taken=actions_taken,
        final_entity_state=final_entity_state,
    )


__all__ = ["replay", "ReplayStep", "ReplayTrace"]
