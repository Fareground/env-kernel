"""Env package — the .simworld self-contained format.

An env is a directory (or .simworld zip) containing:

  meta.json          — identity (name, version, contract_version, ...)
  overview.md        — natural-language brief
  template.json      — the WorldTemplate
  viz/component.tsx  — visualization code
  primitives/*.py    — env-scoped Python primitives

This module owns the **canonical I/O** for that format:

  load_env_package(path)   — read a dir or .simworld, return EnvPackage
  save_env_package(pkg, path)
  pack(dir, archive_path)  — zip a directory into a .simworld
  unpack(archive, dir)     — extract a .simworld into a directory
  scaffold_env(path, name) — create a minimal skeleton

The package layout is defined by the loader and scaffold in this module.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .compile import CompileResult, compile_template

logger = logging.getLogger(__name__)


PACKAGE_EXTENSION = ".simworld"
META_FILE = "meta.json"
OVERVIEW_FILE = "overview.md"
TEMPLATE_FILE = "template.json"
VIZ_DIR = "viz"
PRIMITIVES_DIR = "primitives"


# ---------------------------------------------------------------------------
# In-memory package representation
# ---------------------------------------------------------------------------


@dataclass
class EnvPackage:
    """In-memory representation of an env package.

    Holds parsed content + the source path so callers can re-save
    after edits. ``compile_result`` is populated when ``load_env_package``
    actually compiles the template; ``None`` if loaded read-only."""
    source_path: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    overview: str = ""
    template: Dict[str, Any] = field(default_factory=dict)
    viz_files: Dict[str, str] = field(default_factory=dict)  # filename → content
    primitive_files: Dict[str, str] = field(default_factory=dict)  # filename → content
    compile_result: Optional[CompileResult] = None

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.source_path.stem

    @property
    def is_archive(self) -> bool:
        return self.source_path.suffix == PACKAGE_EXTENSION


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_env_package(
    path: Union[str, Path],
    *,
    compile: bool = True,
    load_primitives: bool = True,
) -> EnvPackage:
    """Load an env package from a directory or .simworld archive.

    Parameters:
        path             — directory or .simworld file
        compile          — if True, also run compile_template (default True)
        load_primitives  — if True, import the env's primitives/ directory
                           so its @effect decorators register (default True)
    """
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"env package not found: {src}")

    # Archive → unpack into a temp dir, then load
    if src.is_file() and src.suffix == PACKAGE_EXTENSION:
        tmp_dir = Path(tempfile.mkdtemp(prefix="simworld_"))
        try:
            unpack(src, tmp_dir)
            pkg = _load_directory(tmp_dir, load_primitives=load_primitives)
            pkg.source_path = src  # remember origin
        finally:
            # Keep tmp dir around if primitives were loaded — they import
            # from there. Otherwise clean up.
            if not load_primitives or not pkg.primitive_files:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        pkg = _load_directory(src, load_primitives=load_primitives)

    if compile:
        pkg.compile_result = compile_template(pkg.template)
    return pkg


def _load_directory(directory: Path, *, load_primitives: bool) -> EnvPackage:
    if not directory.is_dir():
        raise NotADirectoryError(f"expected env directory: {directory}")

    meta_path = directory / META_FILE
    template_path = directory / TEMPLATE_FILE

    if not template_path.exists():
        raise FileNotFoundError(f"required {TEMPLATE_FILE} missing in {directory}")

    pkg = EnvPackage(source_path=directory)

    if meta_path.exists():
        try:
            pkg.meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid {META_FILE}: {e}") from e

    overview_path = directory / OVERVIEW_FILE
    if overview_path.exists():
        pkg.overview = overview_path.read_text()

    try:
        pkg.template = json.loads(template_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid {TEMPLATE_FILE}: {e}") from e

    # Synchronize contract_version from meta → template if missing
    if pkg.meta.get("contract_version") and "contract_version" not in pkg.template:
        pkg.template["contract_version"] = pkg.meta["contract_version"]

    # Viz files (any file under viz/)
    viz_dir = directory / VIZ_DIR
    if viz_dir.is_dir():
        for vf in viz_dir.iterdir():
            if vf.is_file():
                pkg.viz_files[vf.name] = vf.read_text(errors="replace")

    # Primitives — record file content for snapshots, optionally import them
    prim_dir = directory / PRIMITIVES_DIR
    if prim_dir.is_dir():
        for pf in prim_dir.glob("*.py"):
            pkg.primitive_files[pf.name] = pf.read_text()
        if load_primitives:
            _load_env_primitives(prim_dir)

    return pkg


def _load_env_primitives(prim_dir: Path) -> None:
    """Import every .py file in the env's primitives/ directory so the
    @effect / @termination / etc. decorators fire and register.

    Ensures ``sys.modules['kernel']`` aliases to the active kernel
    instance during loading. Without this, env primitives that
    ``from kernel import effect`` would register in a DIFFERENT
    registry instance than the one the rest of the kernel uses —
    silently invisible.
    """
    from ..primitives_loader import _load_dir
    # Walk up from this file (pipeline/env_package.py) two levels to
    # find the kernel package itself, then ensure ``kernel`` maps to it.
    import fg_env_kernel as _active_kernel
    aliased = False
    if sys.modules.get("kernel") is not _active_kernel:
        # Save the prior value if any; restore after load (best-effort)
        _prior = sys.modules.get("kernel")
        sys.modules["kernel"] = _active_kernel
        aliased = True
    try:
        _load_dir(prim_dir)
    except Exception:
        logger.exception("env primitive load failed: %s", prim_dir)
    finally:
        # Restore the prior alias. If a primitive's import is cached
        # under `kernel.X`, we leave it — those imports already
        # resolved against the active kernel.
        if aliased:
            if _prior is None:
                # We added the alias; leaving it in place is safe
                # (it just points at the right module) and keeps any
                # cached `kernel.X` imports valid.
                pass
            else:
                sys.modules["kernel"] = _prior


# ---------------------------------------------------------------------------
# Saving + scaffolding
# ---------------------------------------------------------------------------


def save_env_package(pkg: EnvPackage, path: Union[str, Path]) -> Path:
    """Write a package to disk in directory form. Path can be a
    directory; missing intermediate dirs are created."""
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if pkg.meta:
        (out / META_FILE).write_text(json.dumps(pkg.meta, indent=2, sort_keys=True))
    if pkg.overview:
        (out / OVERVIEW_FILE).write_text(pkg.overview)
    (out / TEMPLATE_FILE).write_text(json.dumps(pkg.template, indent=2))

    if pkg.viz_files:
        viz_dir = out / VIZ_DIR
        viz_dir.mkdir(exist_ok=True)
        for fname, content in pkg.viz_files.items():
            (viz_dir / fname).write_text(content)

    if pkg.primitive_files:
        prim_dir = out / PRIMITIVES_DIR
        prim_dir.mkdir(exist_ok=True)
        # Always include an __init__.py so it's a package
        init_path = prim_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text('"""Env-scoped primitives — auto-loaded with this env."""\n')
        for fname, content in pkg.primitive_files.items():
            if fname == "__init__.py":
                init_path.write_text(content)
            else:
                (prim_dir / fname).write_text(content)

    return out


def scaffold_env(directory: Union[str, Path], name: str) -> EnvPackage:
    """Create a minimal env scaffold at ``directory``. Returns the
    fresh package object — caller can edit and re-save.

    Skeleton:
        meta.json with {name, contract_version, created}
        overview.md with a short brief
        template.json with one entity_type + one action + one termination
    """
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    from .versioning import CONTRACT_VERSION
    import datetime

    now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    pkg = EnvPackage(source_path=target)
    pkg.meta = {
        "name": name,
        "display_name": name.replace("_", " ").title(),
        "tags": [],
        "version": "0.1.0",
        "contract_version": CONTRACT_VERSION,
        "created": now,
        "updated": now,
    }
    pkg.overview = (
        f"# {pkg.meta['display_name']}\n\n"
        "## Premise\n\n"
        "TODO: Describe what this game/simulation is about.\n\n"
        "## Rules in plain English\n\n"
        "- TODO: list the rules\n"
    )
    pkg.template = {
        "name": name,
        "description": "TODO",
        "contract_version": CONTRACT_VERSION,
        "temporal": {"phases": [{"name": "act"}]},
        "entity_types": [{
            "name": "Player",
            "role": "agent",
            "properties": [
                {"name": "score", "type": "int", "default": 0,
                 "min_value": 0, "max_value": 100},
            ],
        }],
        "actions": [{
            "name": "score",
            "actor_type": "Player",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "add", "target": "actor", "field": "score", "value": 1},
            ],
        }],
        "entities": [
            {"id": "p1", "name": "P1", "entity_type": "Player",
             "properties": {"score": 0}},
        ],
        "termination_conditions": [{
            "name": "winner",
            "check_type": "first_to_score",
            "params": {"entity_type": "Player", "property": "score", "target": 10},
        }],
    }

    save_env_package(pkg, target)
    return pkg


# ---------------------------------------------------------------------------
# Pack / unpack (archive form)
# ---------------------------------------------------------------------------


def pack(directory: Union[str, Path],
         archive_path: Optional[Union[str, Path]] = None) -> Path:
    """Zip a directory env into a .simworld archive.

    If ``archive_path`` is omitted, the archive is written alongside the
    directory with the same name + .simworld extension.
    """
    src = Path(directory).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(f"not a directory: {src}")
    if archive_path is None:
        archive_path = src.with_suffix(PACKAGE_EXTENSION)
    archive_path = Path(archive_path).expanduser().resolve()

    # Validate the directory has at least template.json
    if not (src / TEMPLATE_FILE).exists():
        raise FileNotFoundError(f"{TEMPLATE_FILE} required in {src}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                # Skip cache files
                if "__pycache__" in path.parts or path.name.endswith(".pyc"):
                    continue
                rel = path.relative_to(src)
                zf.write(path, rel)
    return archive_path


def unpack(archive: Union[str, Path],
           directory: Optional[Union[str, Path]] = None) -> Path:
    """Extract a .simworld archive into a directory.

    If ``directory`` is omitted, extracts beside the archive using the
    archive's stem as the directory name.
    """
    arc = Path(archive).expanduser().resolve()
    if not arc.is_file():
        raise FileNotFoundError(f"archive not found: {arc}")
    if arc.suffix != PACKAGE_EXTENSION:
        raise ValueError(f"expected {PACKAGE_EXTENSION} archive, got {arc.suffix}")

    if directory is None:
        directory = arc.with_suffix("")
    out = Path(directory).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(arc, "r") as zf:
        # Safety: reject paths that try to escape (../ in member names)
        for member in zf.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member: {member}")
        zf.extractall(out)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_package(pkg: EnvPackage) -> List[str]:
    """Quick validation of package shape — returns list of issue
    descriptions. Empty list = OK. This is structural validation; the
    template's own lint runs via compile_template separately."""
    issues: List[str] = []

    if not pkg.template:
        issues.append("template.json missing or empty")
    if not pkg.meta.get("name"):
        issues.append("meta.json missing 'name' field")
    if not pkg.meta.get("contract_version") and not pkg.template.get("contract_version"):
        issues.append("contract_version missing from both meta.json and template.json")

    # Cross-check that meta.name matches template.name when both are set
    meta_name = pkg.meta.get("name")
    tpl_name = pkg.template.get("name")
    if meta_name and tpl_name and meta_name != tpl_name:
        issues.append(
            f"meta.name ('{meta_name}') doesn't match template.name ('{tpl_name}') — "
            "these should be the same slug"
        )

    return issues


__all__ = [
    "EnvPackage",
    "load_env_package",
    "save_env_package",
    "scaffold_env",
    "pack",
    "unpack",
    "validate_package",
    "PACKAGE_EXTENSION",
]
