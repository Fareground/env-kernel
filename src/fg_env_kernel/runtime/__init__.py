"""Runtime — the engine + tick loop + dispatch.

This subpackage holds the core simulation runtime, one concern per file:

  engine.py          — SimulationEngine: the tick loop, phases, lifecycle
  turn.py            — one agent's turn: perceive, decide, resolve, apply
  actions.py         — action name -> definition, param coercion, resolution
  effect_dispatch.py — applying a list of Effects to the world state
  conditions.py      — condition/predicate evaluation and comparison
  perception.py      — assembling what an agent sees on its turn
  triggers.py        — event emission and the trigger cascade
  termination.py     — when a run ends and who won

``SimulationEngine`` keeps a thin delegating method for each extracted
body, so code calling ``engine._foo(...)`` directly still works. New
callers should import the module-level function instead.
"""
from .engine import (
    SimulationEngine,
    TerminationCondition,
    _coerce_effects,
    _is_multi_target,
    _resolve_cell_for_board,
)
from .effect_dispatch import apply_effects
from .perception import build_perception
from .triggers import emit_event

__all__ = [
    # Core engine
    "SimulationEngine",
    "TerminationCondition",
    # Stable public dispatch surface
    "apply_effects",
    "build_perception",
    "emit_event",
    # Module-level helpers (internal but exposed for back-compat)
    "_coerce_effects",
    "_is_multi_target",
    "_resolve_cell_for_board",
]
