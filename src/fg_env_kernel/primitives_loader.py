"""Primitive auto-discovery — drop a .py file in a known location,
the kernel registers it at startup.

## Discovery rules

The kernel scans these locations at import time (best-effort, errors
logged but never fatal):

  1. ``<repo_root>/kernel_primitives/*.py``   — repo-shipped library
  2. ``$KERNEL_PRIMITIVES_DIR/*.py``           — env-override directory
  3. Any package importing ``fg_env_kernel.primitives_loader``
     can call ``discover(custom_dir)`` to register from anywhere.

Files are imported as standard Python modules; the @effect /
@termination_decorator / @resolution decorators inside them do the
registration. The loader doesn't track WHAT was registered — that's
visible via ``export_kernel_contract()``.

## File format

Each primitive lives in its own .py file (one primitive per file is
the canonical style, but multiple are allowed). The template that
``python -m fg_env_kernel new-primitive`` writes follows this convention:

    \"\"\"<one-line summary>\"\"\"
    from fg_env_kernel import effect, EffectContext

    @effect("my_op_name")
    def _my_op(ctx: EffectContext, spec: dict):
        # implementation
        return None

## Why this matters

Without auto-discovery, every new primitive requires editing
``src/fg_env_kernel/__init__.py`` to import the module. With it,
contributors drop a file and ship. Production builds + library
authors both win.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _repo_primitives_dir() -> Optional[Path]:
    """Locate the repo-shipped primitives dir, if present.

    The canonical location is <repo_root>/kernel_primitives. Repo root
    is detected by walking up from this file until we find one of:
      - kernel_primitives/        (the conventional location)
      - .git/                     (signals repo root)
      - pyproject.toml            (signals repo root)
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "kernel_primitives"
        if candidate.is_dir():
            return candidate
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            # Stop walking up but no kernel_primitives found
            return None
    return None


def _env_primitives_dir() -> Optional[Path]:
    """Optional environment-override directory."""
    val = os.environ.get("KERNEL_PRIMITIVES_DIR")
    if not val:
        return None
    p = Path(val).expanduser().resolve()
    return p if p.is_dir() else None


def discover(directory: Optional[Path] = None) -> List[str]:
    """Import every .py file in ``directory`` (or the default locations).

    Returns the list of fully-qualified module names that were
    successfully imported. Failed imports are logged but don't raise."""
    discovered: List[str] = []

    if directory is not None:
        discovered.extend(_load_dir(directory))
    else:
        repo_dir = _repo_primitives_dir()
        if repo_dir is not None:
            discovered.extend(_load_dir(repo_dir))
        env_dir = _env_primitives_dir()
        if env_dir is not None:
            discovered.extend(_load_dir(env_dir))

    return discovered


def _load_dir(directory: Path) -> List[str]:
    """Import every .py file in the given directory."""
    out: List[str] = []
    if not directory.is_dir():
        return out
    # Ensure parent is on sys.path so imports work
    parent = str(directory.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    dir_name = directory.name
    # __init__.py first so package-level setup runs
    init_file = directory / "__init__.py"
    if init_file.exists():
        try:
            importlib.import_module(dir_name)
        except Exception:
            logger.exception("kernel primitive __init__ failed: %s", init_file)

    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        mod_name = f"{dir_name}.{path.stem}"
        try:
            importlib.import_module(mod_name)
            out.append(mod_name)
            logger.info("loaded kernel primitive: %s", mod_name)
        except Exception:
            logger.exception("failed loading kernel primitive: %s", mod_name)
    return out


def list_loaded_primitives() -> dict:
    """Inspect the live registry and return what's registered, grouped
    by namespace. Used by ``python -m fg_env_kernel primitives``."""
    from .registry import registry

    return {
        "effects": sorted(registry.effects.keys()),
        "preconditions": sorted(registry.preconditions.keys()),
        "resolutions": sorted(registry.resolutions.keys()),
        "phases": sorted(registry.phases.keys()),
        "terminations": sorted(registry.terminations.keys()),
        "modules": sorted(registry.modules.keys()),
        "target_selectors": sorted(registry.target_selectors.keys()),
        "triggers": sorted(registry.triggers.keys()),
    }


# Templates for each primitive kind — used by `python -m fg_env_kernel new-primitive`
PRIMITIVE_TEMPLATES = {
    "effect": '''"""{description}

Effect handler — fires as part of an action's effects_on_success /
effects_on_failure / effects_on_partial.
"""
from fg_env_kernel import effect, EffectContext


@effect("{name}")
def _{name}(ctx: EffectContext, spec: dict):
    """{description}

    Schema usage:
        {{"operation": "{name}", "target": "actor", "value": ..., "field": ...}}

    Args:
        ctx:  effect context (state, actor, target, params, result, rng, emit)
        spec: parsed effect dict — read .get("value"), .get("field"), etc.
    """
    # value = ctx.resolve(spec.get("value"))
    # target = ctx.target
    # ... do your thing ...
    # return {{"entity": ent.id, "field": ..., "old": ..., "new": ...}}  # optional change record
    return None
''',
    "termination": '''"""{description}

Termination check — registers a custom game-end condition.
"""
from fg_env_kernel.registry import termination
from fg_env_kernel.termination import register_winner_resolver


@termination("{name}")
def _check_{name}(state, params, rng):
    """{description}

    Schema usage:
        {{"name": "...", "check_type": "{name}", "params": {{...}}}}

    Args:
        state:  the WorldState
        params: dict from the termination_conditions[].params field
        rng:    seeded RNG

    Returns True when the game should end.
    """
    return False


# Optional: register a winner resolver — returns {{"winner_id": ..., "winner_name": ...}}
# for the terminated event. Comment out if no specific winner exists.
def _resolve_{name}(state, params, rng):
    """Identify the winning entity when this check_type fires."""
    return {{}}

register_winner_resolver("{name}", _resolve_{name})
''',
    "resolution": '''"""{description}

Resolution archetype — determines action outcomes (success/failure,
magnitude, narrative).
"""
from fg_env_kernel.registry import resolution
from fg_env_kernel.resolution import ResolutionArchetype, ResolutionResult


@resolution("{name}")
class {ClassName}(ResolutionArchetype):
    """{description}

    Schema usage:
        {{"resolution_archetype": "{name}",
          "resolution_params": {{...}}}}
    """

    def resolve(self, actor_properties, target_properties, params, action_params, rng=None) -> ResolutionResult:
        # roll = (rng or random.Random()).random()
        return ResolutionResult(
            success=True,
            magnitude=1.0,
            narrative="...",
            details={{}},
        )
''',
    "precondition": '''"""{description}

Precondition operator — registers a custom action-guard predicate.
For most cases, use the `expr` field on Precondition instead; only
write a custom precondition when the logic doesn't fit an expression.
"""
from fg_env_kernel.registry import precondition


@precondition("{name}")
def _check_{name}(state, actor, target, condition):
    """{description}

    Schema usage (legacy form):
        {{"subject": "actor", "operator": "{name}", "field": ..., "value": ...}}

    Returns True when the precondition passes.
    """
    return True
''',
    "expr_func": '''"""{description}

Expression function — adds a $-function callable from any expr context
(preconditions, conditional effects, terminations).

NOTE: There's no decorator for this — extend the dispatcher in
src/fg_env_kernel/effects.py:_call_function or wrap the function and
add a small adapter. Pure additions to the function library are best
contributed directly to effects.py until a dedicated registry is added.
"""
from typing import Any, List


def _fn_{name}(args: List[Any], *, state=None, rng=None) -> Any:
    """{description}

    Schema usage:
        "${name}(arg1, arg2)"
    """
    return None
''',
}


__all__ = [
    "discover",
    "list_loaded_primitives",
    "PRIMITIVE_TEMPLATES",
]
