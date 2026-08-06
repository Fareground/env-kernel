<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg" />
  <img src="assets/wordmark.svg" alt="Fareground" width="320" />
</picture>

# env-kernel

*A deterministic simulation kernel for agent environments — LLMs are brains, code is physics.*

<p>
  <a href="https://github.com/Fareground/env-kernel/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Fareground/env-kernel/ci.yml?branch=main&style=flat-square&label=CI" /></a>
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

The package is not on PyPI — install from GitHub:

```bash
pip install "fg-env-kernel @ git+https://github.com/Fareground/env-kernel.git"
```

## Usage

Build state and an engine from a declarative world definition, then step it:

```python
from fg_env_kernel import load_world

state, engine = load_world(my_world_definition, seed=42, decision_fn=my_agent_brain)
engine.run()
```

`load_world(template, *, seed=0, decision_fn=None, on_event=None)` returns a `(WorldState, SimulationEngine)` tuple. The engine is fully decoupled from the LLM — the same world runs with real agents, cheap heuristics, or a deterministic test stub through the `decision_fn` callback.

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

Register custom verbs, resolution archetypes, phases, and terminations with decorators — the engine looks everything up by string name through the registry:

```python
from fg_env_kernel import effect, EffectContext

@effect("grant_gold")
def grant_gold(ctx: EffectContext, amount: int) -> None:
    ctx.set(ctx.actor, "gold", ctx.get(ctx.actor, "gold") + amount)
```

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
