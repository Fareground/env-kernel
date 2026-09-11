"""KernelModule — lifecycle protocol for kernel-level subsystems.

A ``KernelModule`` is the unit of pluggable state in WorldState. Every
built-in subsystem (factions, polls, inventory, skills, ...) and every
custom game extension implements this protocol. The engine never
imports modules by name; it iterates ``state.modules`` and dispatches
lifecycle events to each.

This is intentionally narrower than ``DomainModule`` — that class
handles per-round game *physics* (tick, validate_action, perception),
whereas this protocol handles *state-graph hygiene*:

  - on_entity_despawn(entity_id)  — clean up references when an entity dies
  - on_entity_spawn(entity_id)    — seed defaults when a new entity appears
  - to_dict() / from_dict(data)   — snapshot / restore
  - name (property)               — string id used as the registry key

A class can implement BOTH ``KernelModule`` and ``DomainModule`` to get
both lifecycle hooks and game-physics hooks. They are orthogonal.

## Why a Protocol, not an ABC

Existing managers (``FactionManager``, ``PollManager``, etc.) already
have ``to_dict()``/``from_dict()``. Forcing them to inherit a new ABC
would require touching every file. The Protocol form lets the engine
duck-type any object: if it has the right methods, it's a module.

The optional hooks (``on_entity_despawn``, ``on_entity_spawn``,
``on_round_start``) are sniffed with ``getattr(mod, "on_…", None)``
so existing managers don't need to grow no-op stubs to comply.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class KernelModule(Protocol):
    """Structural protocol for kernel-level subsystems.

    A class is a ``KernelModule`` if it provides ``to_dict()``. Other
    methods are optional and only called when present — this lets old
    managers comply without code changes."""

    def to_dict(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Lifecycle dispatch helpers
# ---------------------------------------------------------------------------

def dispatch_despawn(modules: Dict[str, Any], entity_id: str) -> None:
    """Call ``on_entity_despawn(entity_id)`` on every module that
    implements it. Exceptions in a single module are caught and logged
    so one buggy module can't wedge the whole despawn pipeline."""
    import logging
    for name, mod in list(modules.items()):
        hook = getattr(mod, "on_entity_despawn", None)
        if hook is None:
            # Compat aliases — some managers used these legacy names
            for alt in ("remove_entity", "remove_member", "clear_entity"):
                hook = getattr(mod, alt, None)
                if hook is not None:
                    break
        if hook is None:
            continue
        try:
            hook(entity_id)
        except Exception:
            logging.getLogger(__name__).exception(
                "module[%s].on_entity_despawn(%s) raised", name, entity_id,
            )


def dispatch_spawn(modules: Dict[str, Any], entity_id: str) -> None:
    """Call ``on_entity_spawn(entity_id)`` on every module that
    implements it. Exceptions are caught and logged."""
    import logging
    for name, mod in list(modules.items()):
        hook = getattr(mod, "on_entity_spawn", None)
        if hook is None:
            continue
        try:
            hook(entity_id)
        except Exception:
            logging.getLogger(__name__).exception(
                "module[%s].on_entity_spawn(%s) raised", name, entity_id,
            )


def dispatch_round_start(modules: Dict[str, Any], state: Any, round_number: int) -> None:
    """Call ``on_round_start(state, round_number)`` on every module
    that implements it. Used to give plugin modules a deterministic
    hook to update at the start of each round."""
    import logging
    for name, mod in list(modules.items()):
        hook = getattr(mod, "on_round_start", None)
        if hook is None:
            continue
        try:
            hook(state, round_number)
        except Exception:
            logging.getLogger(__name__).exception(
                "module[%s].on_round_start raised", name,
            )


def restore_plugin_modules(
    modules: Dict[str, Any],
    plugin_snapshot: Dict[str, Any],
) -> None:
    """Restore all supplied plugins atomically, rejecting missing implementations."""
    import copy
    from .snapshot import SnapshotRestoreError, _preserved
    replacements = {}
    for name, data in plugin_snapshot.items():
        try:
            existing = modules[name]
            restored = type(existing).from_dict(copy.deepcopy(data))
            _preserved(data, restored.to_dict(), f"plugin_modules.{name}")
            replacements[name] = restored
        except Exception as exc:
            raise SnapshotRestoreError(f"Cannot restore plugin_modules.{name}: {exc}") from exc
    modules.update(replacements)


def collect_snapshots(modules: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot serializable plugins; serializer failures are never hidden.

    Hook-only modules with no serializer are stateless by contract and skipped.
    """
    import copy
    out: Dict[str, Any] = {}
    for name, mod in modules.items():
        td = getattr(mod, "to_dict", None)
        if not callable(td):
            continue
        try:
            out[name] = copy.deepcopy(td())
        except Exception as exc:
            raise ValueError(f"Cannot snapshot plugin_modules.{name}: {exc}") from exc
    return out


__all__ = [
    "KernelModule",
    "dispatch_despawn",
    "dispatch_spawn",
    "dispatch_round_start",
    "collect_snapshots",
    "restore_plugin_modules",
]
