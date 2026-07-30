"""Backwards-compatible re-export.

The canonical location is ``fg_env_kernel.pipeline.loader``. Keep
importing from ``fg_env_kernel`` (the public API) — direct imports
of this module continue to work as a transition convenience.
"""
from .pipeline.loader import *  # noqa: F401,F403
from .pipeline.loader import (  # explicit re-exports for IDEs
    WorldTemplate,
    EntityTypeSpec,
    ResourceTypeSpec,
    ActionSpec,
    PreconditionSpec,
    EffectSpec,
    EffectConditionSpec,
    EntitySpec,
    TerminationSpec,
    DomainModuleSpec,
    build_world_state,
    load_world,
    load_world_parts,
)
