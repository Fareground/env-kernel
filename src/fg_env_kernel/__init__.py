"""Fareground env kernel — the game-agnostic simulation engine.

Public surface for env-builders and integrators:

    from fg_env_kernel import (
        registry,
        effect, precondition, resolution, phase, termination, module,
        EffectContext,
    )

Use the decorators to register custom verbs / archetypes / phases that
extend the engine without modifying its source. The engine looks
everything up by string name through the registry — this is the
foundation of the "configure ANY game from schema + rules + viz" goal.
"""
from .registry import (
    KernelRegistry,
    registry,
    effect,
    precondition,
    resolution,
    phase,
    termination as termination_decorator,  # avoid shadowing .termination submodule
    module,
    target_selector,
)
# Re-export the submodule under its natural name so callers can do
# ``from fg_env_kernel import termination`` and get the module's
# public API (evaluate / check_all / resolve_winner / register_winner_resolver).
from . import termination
from .effect_context import EffectContext
# Import composition primitives so their @effect decorators register
# at kernel startup. Side-effect-only import.
from . import composition  # noqa: F401

# Auto-discover any kernel_primitives/*.py files (repo + env-var dir).
# This lets library authors and downstream apps drop new primitives
# into a known location without editing kernel imports.
from .primitives_loader import discover as _discover_primitives, list_loaded_primitives
try:
    _discover_primitives()
except Exception:
    import logging
    logging.getLogger(__name__).exception("kernel_primitives auto-discovery failed")
# Pipeline — env-builder API surface (loader, compile, lint, smoke, contract)
from .pipeline import (
    WorldTemplate,
    build_world_state,
    load_world,
    load_world_parts,
    CompileIssue,
    CompileResult,
    compile_template,
    lint_template,
    SmokeReport,
    smoke_test,
    ReplayStep,
    ReplayTrace,
    replay,
    EnvPackage,
    PACKAGE_EXTENSION,
    load_env_package,
    save_env_package,
    scaffold_env,
    pack_env,
    unpack_env,
    validate_package,
    CONTRACT_VERSION,
    export_kernel_contract,
)
from .predicates import evaluate as evaluate_predicate
from .predicates import resolve as resolve_expression_value
from .kernel_module import (
    KernelModule,
    dispatch_despawn,
    dispatch_spawn,
    dispatch_round_start,
    collect_snapshots,
)
# Observability — metrics + structured logging
from .observability import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    metrics,
    timed,
    enable_engine_metrics,
    engine_metrics_enabled,
    get_logger,
)
from .temporal import TemporalModel, TimeMode, Phase
from .continuous_time import ContinuousTemporalModel, EventQueue, ScheduledEvent
from .physics import (
    PhysicsModel,
    PhysicsVariable,
    EntitySource,
    EntityWriteback,
    PhysicsExprError,
)

__all__ = [
    "WorldTemplate",
    "build_world_state",
    "load_world",
    "load_world_parts",
    "compile_template",
    "CompileIssue",
    "CompileResult",
    "lint_template",
    "smoke_test",
    "SmokeReport",
    "replay",
    "ReplayStep",
    "ReplayTrace",
    "EnvPackage",
    "PACKAGE_EXTENSION",
    "load_env_package",
    "save_env_package",
    "scaffold_env",
    "pack_env",
    "unpack_env",
    "validate_package",
    "export_kernel_contract",
    "CONTRACT_VERSION",
    "termination",
    "termination_decorator",
    "evaluate_predicate",
    "resolve_expression_value",
    "KernelModule",
    "dispatch_despawn",
    "dispatch_spawn",
    "dispatch_round_start",
    "collect_snapshots",
    # Observability
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "metrics",
    "timed",
    "enable_engine_metrics",
    "engine_metrics_enabled",
    "get_logger",
    # Primitives loader
    "list_loaded_primitives",
    "KernelRegistry",
    "registry",
    "effect",
    "precondition",
    "resolution",
    "phase",
    "termination",
    "module",
    "target_selector",
    "EffectContext",
    # Temporal modes
    "TemporalModel",
    "TimeMode",
    "Phase",
    "ContinuousTemporalModel",
    "EventQueue",
    "ScheduledEvent",
    # Continuous coupled-dynamics ("physics")
    "PhysicsModel",
    "PhysicsVariable",
    "EntitySource",
    "EntityWriteback",
    "PhysicsExprError",
]
