"""Runtime — the engine + tick loop + dispatch.

This subpackage holds the core simulation runtime:

  engine.py          — SimulationEngine god class (TODO: extract methods below)
  effect_dispatch.py — _apply_effects extraction point
  perception.py      — _build_agent_perception extraction point
  triggers.py        — _emit_event + cascade extraction point

The engine.py file is currently still the main 3,876-line class. Future
extractions move specific methods (effect dispatch, perception
assembly, trigger cascade) into focused files alongside engine.py.
Today they re-export from engine for backwards-compat.
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
