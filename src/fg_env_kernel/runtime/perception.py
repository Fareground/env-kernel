"""Perception assembly — builds the dict an agent sees on its turn.

## Current state

The canonical implementation lives as
``SimulationEngine._build_agent_perception`` in [engine.py](./engine.py)
(~180 lines). This module exposes a stable public surface so callers
can import from ``runtime.perception``.

## Migration path

When extracting:
  1. Move ``_build_agent_perception`` body into ``build_perception(engine, eid)``
  2. The engine method becomes a 1-line delegation
  3. New callers import from this module directly

The perception payload is intentionally typed as ``dict`` so a game
agent (LLM or otherwise) can read it without any kernel imports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from .engine import SimulationEngine


def build_perception(engine: "SimulationEngine", entity_id: str) -> Tuple[Dict[str, Any], List[str]]:
    """Build the agent's perception payload + list of valid actions.

    Stable public API. Currently delegates to the engine class method;
    future refactor will inline the implementation here without
    changing this signature.
    """
    return engine._build_agent_perception(entity_id)


__all__ = ["build_perception"]
