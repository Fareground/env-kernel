"""Contract version negotiation for WorldTemplate.

When the kernel's JSON contract changes, old templates need a
migration path. This module:

  1. Defines the current ``CONTRACT_VERSION``
  2. Records ``contract_version`` on every emitted template
  3. Provides a migration registry — register an upgrade function
     under the version it should LIFT ABOVE
  4. ``upgrade_template(raw)`` walks the migration chain until the
     template is at the current version

## Usage

    from fg_env_kernel.pipeline.versioning import (
        CONTRACT_VERSION, register_migration, upgrade_template,
    )

    @register_migration("1.0")
    def _v1_to_v2(raw: dict) -> dict:
        # Move runtime_parameters under settings/
        raw["settings"] = {"runtime_parameters": raw.pop("runtime_parameters", [])}
        raw["contract_version"] = "2.0"
        return raw

A template tagged ``"contract_version": "1.0"`` then flows through
``_v1_to_v2`` before compile.

## Version format

Semver-ish. Compare as strings — the ordering of registered
migrations defines the chain. Don't skip versions; if you go
v1.0 → v3.0, register one migration that takes v1.0 → v2.0 and
another that takes v2.0 → v3.0. ``upgrade_template`` chains them.

## Backwards compatibility

Templates without ``contract_version`` are treated as the **earliest
known version** and walked through every registered migration. This
covers every existing ``assets/<game>/template.json`` file in the
repo today.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


CONTRACT_VERSION = "1.0"
"""Current kernel contract version. Bump when WorldTemplate schema
changes in a backwards-incompatible way. Add a migration via
``@register_migration("<old_version>")`` describing how to upgrade
from that version to the next."""


# In-order list of (from_version, migration_fn) tuples. Order matters —
# this defines the migration chain.
_MIGRATIONS: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = []


def register_migration(from_version: str):
    """Decorator: register a function that upgrades a template from
    ``from_version`` to the next version in the chain.

    The function takes a raw dict and returns the upgraded dict. It
    MUST set ``contract_version`` on the returned dict to the new
    version so the upgrade loop can continue.
    """
    def _decorator(fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        _MIGRATIONS.append((from_version, fn))
        return fn
    return _decorator


def get_template_version(raw: Dict[str, Any]) -> str:
    """Return the declared ``contract_version`` of a template, falling
    back to the EARLIEST known version when omitted. Templates without
    a version are treated as legacy and walked through the full chain.
    """
    v = raw.get("contract_version")
    if isinstance(v, (int, float)):
        v = str(v)
    if not v or not isinstance(v, str):
        return _earliest_known_version()
    return v


def upgrade_template(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Walk the migration chain until ``raw`` is at ``CONTRACT_VERSION``.

    Returns a NEW dict (never mutates the input). If the template is
    already current, returns a shallow copy.

    Logs each migration step at INFO so it's auditable.
    """
    out = dict(raw)  # shallow copy
    if "contract_version" not in out:
        out["contract_version"] = get_template_version(raw)

    current = out["contract_version"]
    # Walk migrations whose `from_version` matches current
    visited: set = set()
    while current != CONTRACT_VERSION:
        if current in visited:
            logger.warning(
                "upgrade_template: cycle detected at version %s — aborting",
                current,
            )
            break
        visited.add(current)

        match = next((fn for v, fn in _MIGRATIONS if v == current), None)
        if match is None:
            logger.warning(
                "upgrade_template: no migration from %s — leaving template at this version",
                current,
            )
            break

        logger.info("upgrade_template: %s → next via %s", current, match.__name__)
        out = match(out)
        new_version = out.get("contract_version")
        if not new_version or new_version == current:
            logger.warning(
                "upgrade_template: migration %s didn't advance version (still %s)",
                match.__name__, current,
            )
            break
        current = new_version

    return out


def _earliest_known_version() -> str:
    """The earliest version we have a migration FOR — or CONTRACT_VERSION
    if no migrations are registered."""
    if _MIGRATIONS:
        return _MIGRATIONS[0][0]
    return CONTRACT_VERSION


def list_known_versions() -> List[str]:
    """All versions in the migration chain, from earliest to current."""
    versions = [v for v, _ in _MIGRATIONS]
    if CONTRACT_VERSION not in versions:
        versions.append(CONTRACT_VERSION)
    return versions


__all__ = [
    "CONTRACT_VERSION",
    "register_migration",
    "get_template_version",
    "upgrade_template",
    "list_known_versions",
]
