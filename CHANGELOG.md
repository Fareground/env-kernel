# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Continuous coupled-dynamics ("physics").** New `physics` module: a dt-aware
  system of coupled ODEs (`PhysicsModel`) integrated with 4th-order Runge–Kutta
  and sub-stepping. Variables may be free global scalars, read entity-property
  aggregates (`EntitySource`), and/or write values back onto entities
  (`EntityWriteback`). Rate expressions use a safe whitelisted-AST evaluator (no
  `eval` of arbitrary code). Declare via a `physics` block in the world schema;
  the world evolves between agent turns (predator/prey, SIR, price discovery)
  while agents remain turn-based. Deterministic and serializable.
- **Continuous-time mode wired end-to-end.** `temporal.mode: "continuous"` now
  builds a `ContinuousTemporalModel` (`build_continuous_model`) and drives an
  event-driven clock: agent turns reschedule by per-action `duration`, and a
  recurring environment tick (`environment_interval`) advances physics / property
  dynamics / world events by the real elapsed `dt`. Previously the model existed
  but was never instantiated or run.

### Fixed
- `ContinuousTemporalModel.is_empty()` (the engine's continuous loop called it
  but it was undefined — continuous mode could not start).

## [0.1.0] — 2026-05-27

### Added
- Initial release. Game-agnostic simulation kernel extracted from the Fareground simulation platform.
- Single runtime dependency (`pydantic>=2`).
- Core data types: entities, actions, resources, relations, state.
- Effect / predicate expression language.
- Extension points: verbs, archetypes, phases, primitives.
- Pipeline: compile, lint, smoke, contract.
- Auto-discovery of `kernel_primitives/*.py` extensions.
