"""Pipeline — the env-builder agent's API surface.

This subpackage holds everything an agent needs to go from raw JSON
to a runnable game:

  loader.py   — WorldTemplate + load_world() + build_world_state()
  compile.py  — compile_template() with structured errors
  lint.py     — lint_template() static checker
  smoke.py    — smoke_test() mocked-decision playtester
  contract.py — export_kernel_contract() JSON Schema + capabilities

The agent's loop:
    raw_json → compile_template → smoke_test → real LLM playtest → ship

Everything here is JSON-in, structured-result-out. No per-game Python.
"""
from .loader import (
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
from .compile import CompileIssue, CompileResult, compile_template
from .lint import lint_template
from .smoke import SmokeReport, smoke_test
from .replay import ReplayStep, ReplayTrace, replay
from .contract import export_kernel_contract
from .env_package import (
    EnvPackage,
    PACKAGE_EXTENSION,
    load_env_package,
    save_env_package,
    scaffold_env,
    pack as pack_env,
    unpack as unpack_env,
    validate_package,
)
from .versioning import (
    CONTRACT_VERSION,
    register_migration,
    upgrade_template,
    get_template_version,
    list_known_versions,
)

__all__ = [
    # Contract
    "WorldTemplate",
    "EntityTypeSpec",
    "ResourceTypeSpec",
    "ActionSpec",
    "PreconditionSpec",
    "EffectSpec",
    "EffectConditionSpec",
    "EntitySpec",
    "TerminationSpec",
    "DomainModuleSpec",
    # Builders
    "build_world_state",
    "load_world",
    "load_world_parts",
    # Single entry point
    "compile_template",
    "CompileIssue",
    "CompileResult",
    # Tools
    "lint_template",
    "smoke_test",
    "SmokeReport",
    "replay",
    "ReplayStep",
    "ReplayTrace",
    # Env packages
    "EnvPackage",
    "PACKAGE_EXTENSION",
    "load_env_package",
    "save_env_package",
    "scaffold_env",
    "pack_env",
    "unpack_env",
    "validate_package",
    "export_kernel_contract",
    # Versioning
    "CONTRACT_VERSION",
    "register_migration",
    "upgrade_template",
    "get_template_version",
    "list_known_versions",
]
