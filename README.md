<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg" />
  <img src="assets/wordmark.svg" alt="Fareground" width="320" />
</picture>

# env-kernel

*A deterministic simulation kernel for agent environments — LLMs are brains, code is physics.*

<p>
  <a href="https://github.com/Fareground/env-kernel/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Fareground/env-kernel/ci.yml?branch=main&style=flat-square&label=CI" /></a>
  <a href="https://pypi.org/project/fg-env-kernel/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fg-env-kernel?style=flat-square" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-3b82f6?style=flat-square" />
  <img alt="Dependencies" src="https://img.shields.io/badge/deps-pydantic%20only-2dd4a7?style=flat-square" />
  <img alt="Engine" src="https://img.shields.io/badge/engine-deterministic-9b59b6?style=flat-square" />
</p>

</div>

---

## Overview

**env-kernel** is a continuous-time simulation kernel for agent environments: you describe a world as declarative data, and the kernel compiles it into an executable, deterministic, multi-agent simulation — with RK4 coupled-ODE integration for continuous dynamics and grounded, verifiable outcomes. It is pure Python with a single runtime dependency (`pydantic`) and no coupling to any game, domain, or LLM provider.

env-kernel is stewarded by [Fareground](https://github.com/Fareground) and is one of six open-source building blocks alongside [agent-id](https://github.com/Fareground/agent-id), [agent-messaging](https://github.com/Fareground/agent-messaging), [agent-knowledge](https://github.com/Fareground/agent-knowledge), [agent-memory](https://github.com/Fareground/agent-memory), and [agent-framework](https://github.com/Fareground/agent-framework).

Agents make discrete, turn-based decisions; the kernel is the deterministic rule engine that resolves those decisions and evolves the world around them. It knows nothing about chess or markets or elections — those are just *configurations*. New mechanics plug in through registries and decorators, never by editing the engine.

Between agent turns the world does not have to sit still: an event-driven clock and a coupled-ODE physics integrator can evolve numeric state continuously, so action durations and reaction speed become part of the strategy.

## Install

> **Note:** the distribution name is **`fg-env-kernel`** and the import package is **`fg_env_kernel`**. These are unchanged — downstream projects depend on them, and renaming them would break those imports.

```bash
pip install fg-env-kernel
```

Importing the package never scans the filesystem. Drop-in primitive discovery (`kernel_primitives/*.py`) is opt-in: call `fg_env_kernel.discover()` explicitly, or set the `KERNEL_PRIMITIVES_DIR` environment variable — an explicitly configured directory is honored at import time.

## Quickstart

One line — the built-in seeded random agent plays every turn:

```python
from fg_env_kernel import simulate

world = simulate("path/to/template.json")   # or a template dict
print(world.summary())
```

`simulate(template, *, agent=None, seed=None, max_rounds=None, on_event=None, registry=None)` loads the template (dict, `WorldTemplate`, or path to a JSON file), runs to completion, and returns the finished `World`. With no `agent`, a deterministic random-valid-action policy (`random_policy`) drives every turn — same seed, same run. Pass your own `decision_fn` as `agent` to plug in an LLM.

### Bring your own agent

A world is a plain dict; an agent is a plain function. This is a complete, runnable program:

```python
from fg_env_kernel import ActionInstance, Kernel

template = {
    "name": "Race to 10",
    "description": "Two runners sprint; first to distance 10 wins.",
    "entity_types": [
        {"name": "runner", "role": "agent", "properties": [
            {"name": "distance", "type": "float", "default": 0}
        ]}
    ],
    "entities": [
        {"id": "alice", "entity_type": "runner", "name": "Alice"},
        {"id": "bob", "entity_type": "runner", "name": "Bob"},
    ],
    "actions": [
        {"name": "sprint", "description": "Run forward.", "actor_type": "runner",
         "effects_on_success": [
             {"operation": "add", "target": "actor", "field": "distance",
              "value": "$random(1, 3)"}
         ]}
    ],
    "termination_conditions": [
        {"name": "finish_line", "check_type": "expr",
         "params": {"expr": "$state.entities.alice.distance >= 10 || "
                            "$state.entities.bob.distance >= 10"}}
    ],
    "temporal": {"max_rounds": 20},
}

def decision_fn(entity_id, perception, valid_actions):
    """Called once per agent turn. Swap in an LLM call here."""
    if "sprint" not in valid_actions:
        return None
    return ActionInstance(action_name="sprint", actor_id=entity_id)

world = Kernel(seed=42).load(template, decision_fn=decision_fn)
world.run()                       # or: while not world.finished: world.step()

print(world.terminated_by)        # "finish_line"
print(world.current_round)        # 5
print(world.events[-1].narrative) # "Simulation ended after 5 rounds."
```

Same seed, same template, same `decision_fn` → same run, every time. More in [`examples/`](examples/) — including tic-tac-toe built from a domain module.

## The agent contract

`decision_fn(entity_id, perception, valid_actions) -> ActionInstance | None` is the only interface between your agent (LLM or otherwise) and the kernel:

- **`entity_id`** — id of the agent whose turn it is.
- **`perception`** — a plain dict of what this agent can see, visibility-filtered. Always present: `self` (own id/name/properties), `visible_entities`, `visible_relations`, `visible_resources`, `round`, `phase`, `location`, `faction`. Present when the world provides them: `world_brief` (the template's name/description/rules markdown), `incoming_messages`, `your_recent_actions`, `domain_data` (board layout, hand contents, market state, ...), and more (roles, polls, time context, trade history).
- **`valid_actions`** — names of the actions whose preconditions currently pass. Return an `ActionInstance` whose `action_name` is one of these (with `actor_id=entity_id` and any `parameters` the action declares), or `None` to skip the turn.

The engine is fully decoupled from the LLM — the same world runs with real agents, cheap heuristics, or a deterministic test stub.

`Kernel(seed=..., registry=...)` holds run configuration; `Kernel.load(template, decision_fn=..., on_event=..., seed=..., max_rounds=...)` accepts a template dict, `WorldTemplate`, or path to a JSON file, and returns a `World` with `run()`, `step()`, `finished`, `terminated_by`, `current_round`, `events`, `state`, `seed`, and a readable `summary()`. An `on_event` callback streams each event as it is emitted.

The ladder: `simulate()` for one-shot runs → `Kernel`/`World` for stepwise control → `load_world` for the raw engine.

### Going lower level

The facade is a thin wrapper over `load_world(template, *, seed=0, decision_fn=None, on_event=None, registry=None)`, which returns the raw `(WorldState, SimulationEngine)` pair — use it when you need direct engine or state access. Custom primitives register through the decorator surface (`@effect`, `@resolution`, `@phase`, `@termination_decorator`, `@module`) shown below.

The full template shape is documented in [`docs/template_schema.md`](docs/template_schema.md); the machine-readable contract (including the live list of every registered effect operation, resolution archetype, termination check, and domain module) is [`docs/kernel_contract.json`](docs/kernel_contract.json).

For cash or other conserved numeric properties, use [atomic property transfers](docs/property-transfers.md) instead of independent clamped debit and credit effects.

### Continuous time and physics

A `physics` block on the world definition declares numeric variables and their rates of change. A dt-aware 4th-order Runge–Kutta integrator evolves them between turns — predator/prey, epidemics (SIR), price discovery. Variables can read entity aggregates and write values back onto the world. The result is deterministic and serializable.

```python
"physics": {
    "params": {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4},
    "variables": [
        {"name": "prey", "value": 10, "rate": "alpha*prey - beta*prey*pred", "min": 0},
        {"name": "pred", "value": 5,  "rate": "delta*prey*pred - gamma*pred", "min": 0}
    ]
}
```

### Extending the engine

Register custom verbs, resolution archetypes, phases, and terminations with decorators — the engine looks everything up by string name through the registry. The module-level decorators register process-wide:

```python
from fg_env_kernel import effect, EffectContext

@effect("grant_gold")
def grant_gold(ctx: EffectContext, spec: dict) -> None:
    gold = ctx.actor.properties.get("gold", 0)
    ctx.actor.properties["gold"] = gold + spec.get("value", 1)
```

For per-kernel isolation, fork the registry and register on the fork. A fork sees every built-in primitive (nothing is copied — unknown names fall back to the parent), but its own registrations are invisible to the global registry and to other forks:

```python
from fg_env_kernel import Kernel, registry

mine = registry.fork()

@mine.effect("grant_gold")
def grant_gold(ctx, spec): ...

kernel = Kernel(seed=42, registry=mine)   # worlds resolve against `mine` only
```

Every namespace has an instance decorator (`mine.effect`, `mine.precondition`, `mine.resolution`, `mine.phase`, `mine.termination`, `mine.module`, `mine.target_selector`). Validation/reporting surfaces (`lint_template`, `export_kernel_contract`) read the global registry.

## Concepts

- **Determinism** — given a template, a seed, and a `decision_fn`, a run is fully reproducible. State is serializable end to end, so runs can be replayed step by step.
- **Declarative worlds** — entities, properties, resources, relations, actions, effects, and terminations are all data. A safe expression grammar (`$actor.gold >= 100 && $count(player, alive) > 1`) powers guards, effects, and terminations without per-game Python.
- **Turn-based agents, continuous world** — agents decide in discrete turns; an event-driven clock and the physics integrator evolve the world between those turns.
- **Registry extension points** — custom verbs, resolution archetypes, phases, terminations, and domain modules register by name, keeping the engine core untouched.

## Project Structure

```
src/fg_env_kernel/
  runtime/        the tick loop (discrete + continuous)
  physics.py      coupled-dynamics ODE integrator
  state.py        the world state graph
  action.py …     actions, effects, resolution archetypes
  predicates.py   the expression language
  domain/         optional game-genre modules (markets, boards, …)
  pipeline/       compile · lint · smoke · replay · package
```

See the [`CHANGELOG`](CHANGELOG.md) for what's new.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, running the test suite, and lint/format tooling.

---

<div align="center">
<sub>Stewarded by <b>Fareground</b>.</sub><br />
<sub>Licensed under the <a href="LICENSE">Apache License 2.0</a>.</sub>
</div>
