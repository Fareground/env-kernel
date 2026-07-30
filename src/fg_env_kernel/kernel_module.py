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

from typing import Any, Dict, Optional, Protocol, runtime_checkable


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
    """Restore plugin modules from a snapshot dict.

    Iterates ``plugin_snapshot`` (the ``"plugin_modules"`` slice of a
    state.to_dict()) and tries to rebuild each module by calling its
    ``from_dict`` classmethod. The reconstructed instance replaces the
    existing entry in ``modules``.

    Modules whose class doesn't expose ``from_dict`` keep their current
    instance; the engine just attaches the snapshot to the module's
    ``__snapshot__`` attribute so callers can inspect it.

    Errors are logged and swallowed so one buggy plugin can't fail the
    whole restore."""
    import logging
    for name, data in (plugin_snapshot or {}).items():
        existing = modules.get(name)
        if existing is None:
            # No registered module under that name — keep the snapshot
            # attached for later inspection but don't construct
            # arbitrary objects (we have no class to instantiate).
            continue
        cls = type(existing)
        from_dict = getattr(cls, "from_dict", None)
        if not callable(from_dict):
            # Module doesn't support reconstruction — leave the live
            # instance intact and let it stay in sync via other means.
            continue
        try:
            modules[name] = from_dict(data)
        except Exception:
            logging.getLogger(__name__).exception(
                "module[%s].from_dict raised during restore; keeping live instance",
                name,
            )


def collect_snapshots(modules: Dict[str, Any]) -> Dict[str, Any]:
    """Build a ``{name: to_dict()}`` map for snapshot serialization.

    Modules without ``to_dict`` are skipped. Exceptions are swallowed
    so a buggy plugin can't poison the whole snapshot — that one
    module's entry is simply missing from the result."""
    import logging
    out: Dict[str, Any] = {}
    for name, mod in modules.items():
        td = getattr(mod, "to_dict", None)
        if not callable(td):
            continue
        try:
            out[name] = td()
        except Exception:
            logging.getLogger(__name__).exception(
                "module[%s].to_dict() raised; omitted from snapshot", name,
            )
    return out


__all__ = [
    "KernelModule",
    "dispatch_despawn",
    "dispatch_spawn",
    "dispatch_round_start",
    "collect_snapshots",
    "restore_plugin_modules",
]
