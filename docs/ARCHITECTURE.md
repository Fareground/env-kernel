# Fareground Env Kernel — Architecture & File Layout

This document is the **single source of truth** for where things live in the kernel package. Before adding new code, read this file. Before reorganizing, update this file.

## Discipline rules

1. **The public API is `packages/kernel/__init__.py`.** External callers import from `fg_env_kernel`. Direct imports of submodules (`from fg_env_kernel.runtime.engine import X`) are tolerated for back-compat but discouraged in new code.
2. **Top-level files are either:**
   - **Canonical implementations** (e.g. `state.py`, `action.py`, `effects.py`) — primary modules
   - **Re-export shims** (e.g. `engine.py`, `world_loader.py`, `domain_module.py`) — 1-line files that point to the canonical location in a subfolder. Marked clearly in their docstring.
3. **Subfolders group by responsibility, not by technology.** `runtime/` holds the tick loop; `pipeline/` holds the env-builder API; `domain/` holds game-genre primitives.
4. **New files belong in a subfolder.** Don't add flat top-level files. If your code doesn't fit a subfolder, you might need a new one — propose it in the PR.
5. **No circular imports.** If `A` needs `B` and `B` needs `A`, one is in the wrong place.

---

## Top-level layout

```
packages/kernel/
├── __init__.py             ← PUBLIC API (the firewall)
├── STRUCTURE.md            ← this file
│
├── runtime/                ← engine + tick loop + dispatch surfaces
├── domain/                 ← DomainModule library (stubs + markets)
├── pipeline/               ← env-builder agent's API (compile/lint/smoke/contract)
│
├── (core data types)
├── action.py               ← ActionDefinition, Effect, Precondition, EffectOperation
├── state.py                ← WorldState (the runtime state graph)
├── entity.py               ← Entity + EntityType
├── resource.py             ← ResourcePool + ResourceType
├── relations.py            ← RelationGraph + RelationType
├── types.py                ← PropertySchema, PropertyType
├── event.py                ← SimEvent + EventLog
├── temporal.py             ← Phase + TemporalModel + TimeMode
├── spatial.py              ← NoSpace, GridSpace, GraphSpace, Continuous2DSpace
├── pathfinding.py          ← Pathfinder
│
├── (expression language)
├── effects.py              ← $-expression resolver ($actor.x, $random(...))
├── predicates.py           ← unified boolean evaluator (the `expr` language)
│
├── (extension points)
├── registry.py             ← KernelRegistry (effects/preconditions/resolutions/...)
├── effect_context.py       ← EffectContext (arg bundle for effect handlers)
├── kernel_module.py        ← KernelModule Protocol + lifecycle dispatchers
│
├── (subsystems — registered in state.modules)
├── factions.py             ← FactionManager
├── inventory.py            ← Item, InventoryManager
├── goals.py                ← GoalTracker
├── skills.py               ← SkillTracker, SkillDefinition
├── crafting.py             ← Recipe, RecipeManager
├── negotiation.py          ← Negotiation, Agreement, Auction
├── planning.py             ← Plan, PlanManager, AgentModel
├── polls.py                ← PollManager
├── roles.py                ← RoleRegistry
├── messaging.py            ← MessageBoard
├── status_effects.py       ← StatusEffectDefinition, StatusEffectTracker
├── sequences.py            ← SequenceTracker (multi-round actions)
├── visibility.py           ← VisibilityRule + PerceptionBuilder
├── location_properties.py  ← LocationDefinition, location modifiers/ticks
├── world_model.py          ← AgentWorldModel (spatial memory)
├── world_events.py         ← world-event engine (random events, cascades)
├── property_dynamics.py    ← property drift, cascades, spawning
├── connectors.py           ← external data integration
├── cognition.py            ← CognitionManager (emotion, bias, bounded rationality)
├── social.py               ← SocialPlatformManager (social graph + viral spread)
├── invariants.py           ← InvariantChecker
├── sim_controller.py       ← SimController (event injection, breakpoints)
│
├── (kernel domain modules — declarative game-genre primitives)
├── board_module.py         ← BoardModule (chess/connect-4/tic-tac-toe/etc.)
├── deck_module.py          ← DeckModule (cards)
├── hand_module.py          ← HandModule (per-player hands)
├── slots_module.py         ← SlotsModule (worker placement)
├── trade_module.py         ← TradeModule
├── hidden_state_module.py  ← HiddenStateModule (asymmetric info)
├── phase_state_machine.py  ← PhaseStateMachineModule
├── phase_handlers.py       ← PHASE_HANDLER_REGISTRY (card_deal, showdown, etc.)
├── turn_manager.py         ← TurnManagerModule
├── continuous_time.py      ← ContinuousTemporalModel
│
├── (mechanics + outcomes)
├── resolution.py           ← ResolutionArchetype + 13 archetypes
├── triggers.py             ← TriggerEngine
├── termination.py          ← pluggable termination check_types (P6 home)
│
└── (back-compat shims — DO NOT add new code here)
    ├── engine.py           → runtime/engine.py
    ├── world_loader.py     → pipeline/loader.py
    ├── compile.py          → pipeline/compile.py
    ├── lint.py             → pipeline/lint.py
    ├── smoke.py            → pipeline/smoke.py
    ├── contract.py         → pipeline/contract.py
    └── domain_module.py    → domain/{base,stubs,markets}.py
```

---

## Subfolder responsibilities

### `runtime/` — the tick loop

Holds the `SimulationEngine` and its dispatch surfaces.

```
runtime/
├── __init__.py
├── engine.py             ← SimulationEngine (3,876 lines — split target)
├── effect_dispatch.py    ← apply_effects(engine, ...) — currently delegates
├── perception.py         ← build_perception(engine, eid) — currently delegates
└── triggers.py           ← emit_event(engine, ...) — currently delegates
```

**Status:** `engine.py` is still a god class. The three sibling files are **stable public surfaces** that currently delegate back to engine methods. The migration path is documented in each file's docstring: future PRs move bodies out one at a time. New callers should import from the sibling files, never directly from `engine.py`.

### `domain/` — game-genre primitives

```
domain/
├── __init__.py
├── base.py     ← DomainModule ABC + DomainConstraint + Manager + Registry
├── stubs.py    ← Economic / Political / Ecological / Health (placeholders)
└── markets.py  ← PredictionMarketModule, SecuritiesTradingModule (full)
```

`DomainModule` subclasses live in `assets/<game>/module.py` and self-register via `DomainModuleRegistry`. The kernel never imports specific games — they're discovered at startup.

### `pipeline/` — env-builder API surface

Everything the studio agent (or any caller building games from JSON) needs.

```
pipeline/
├── __init__.py
├── loader.py    ← WorldTemplate + build_world_state + load_world
├── compile.py   ← compile_template (single entry point)
├── lint.py      ← lint_template (static checks)
├── smoke.py     ← smoke_test (mocked playtest)
└── contract.py  ← export_kernel_contract (JSON Schema + capabilities)
```

The agent's loop: `raw_json → compile_template → smoke_test → real LLM playtest → ship`.

---

## The 5 extension points (plug-in surfaces)

An env-builder agent (or game library author) can extend the kernel with **zero kernel edits**:

| Extension | Decorator / API | File |
|---|---|---|
| Custom verb (effect op) | `@effect("name")` | `registry.py` |
| Custom precondition | `@precondition("name")` | `registry.py` |
| Custom resolution mechanic | `@resolution("name")` | `registry.py` |
| Custom phase handler | `@phase("name")` | `registry.py` |
| Custom termination check_type | `@termination("name")` + `register_winner_resolver` | `termination.py` |
| Custom KernelModule (lifecycle subsystem) | `state.register_module(name, instance)` | `state.py` + `kernel_module.py` |
| Custom DomainModule (game-genre physics) | `DomainModuleRegistry.register(name, cls)` | `domain/base.py` |

---

## Where to put new code

| What you're adding | Where it goes |
|---|---|
| A new effect operation | Register via `@effect` from a domain module file OR a game asset. Don't add to `action.py`. |
| A new termination check_type | Register via `@termination` in a new file under `domain/<game>/` or directly in `termination.py` for built-ins. |
| A new property type | `types.py` — add to `PropertyType` enum + `_PROP_TYPE_MAP` in `pipeline/loader.py`. |
| A new lifecycle hook on KernelModule | `kernel_module.py` — add to Protocol, add dispatcher, document in `STRUCTURE.md`. |
| A new game | `assets/<game>/template.json` + optional `assets/<game>/module.py`. Never edit the kernel. |
| A new env-builder API tool | `pipeline/` — never `runtime/`. |
| A bug fix in `_apply_effects` | `runtime/engine.py` (today) — `runtime/effect_dispatch.py` (when extracted). |

---

## File-size budget

Production-grade code keeps files readable. Targets:

| Size | Budget |
|---|---|
| Standard module | ≤ 400 lines |
| Subsystem manager (`factions`, `inventory`, etc.) | ≤ 600 lines |
| Domain module (chess-style genre primitives) | ≤ 1000 lines |
| God classes (`state.py`, `runtime/engine.py`) | Honest accountability — slated for refactoring |

Currently over budget:
- `runtime/engine.py` (3,876 lines) — extraction in progress (see `runtime/effect_dispatch.py` migration notes)
- `state.py` (974 lines) — `WorldState` god object; deferred
- `board_module.py` (1,009 lines) — single comprehensive board primitive; acceptable for now
- `domain/markets.py` (720 lines) — two complex markets; consider splitting later

---

## Public API surface (`packages/kernel/__init__.py`)

External callers should import from `fg_env_kernel` whenever possible:

```python
from fg_env_kernel import (
    # Pipeline (env-builder API)
    WorldTemplate, compile_template, lint_template, smoke_test,
    export_kernel_contract, load_world,
    # Plugin registration
    effect, precondition, resolution, phase, termination_decorator, module,
    EffectContext, registry,
    # Module lifecycle
    KernelModule,
    # Expression language
    evaluate_predicate, resolve_expression_value,
    # Termination (submodule)
    termination,
)
```

When this list changes, update both this file and `packages/kernel/__init__.py`.

---

## How to verify a refactor

Before merging any cleanup change:

1. `python -m pytest apps/backend/tests/ --ignore=tests/test_agent_brain.py --ignore=tests/test_llm_providers.py --ignore=tests/test_llm_narrative.py`
2. Expect **1077 passing, 18 pre-existing baseline failures** (analytics/snapshots/build_world fixtures). Anything else is a regression.
3. Run `python -c "from fg_env_kernel import compile_template, smoke_test"` — public API must keep working.
4. Update this STRUCTURE.md if file layout changed.
