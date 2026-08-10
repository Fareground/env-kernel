"""Backwards-compatible re-export.

The canonical engine lives at ``fg_env_kernel.runtime.engine``.
External code can keep importing ``from fg_env_kernel.engine import X`` —
the names below transparently resolve to the new location.
"""
from .runtime.engine import *  # noqa: F401,F403
from .runtime.engine import (  # noqa: F401 — explicit re-exports for IDEs
    SimulationEngine,
    TerminationCondition,
    _coerce_effects,
    _is_multi_target,
    _resolve_cell_for_board,
)
