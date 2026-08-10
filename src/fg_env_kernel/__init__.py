"""Fareground env kernel — the game-agnostic simulation engine.

Quickstart (see ``examples/quickstart.py``):

    from fg_env_kernel import Kernel

    kernel = Kernel(seed=42)
    world = kernel.load(template_dict, decision_fn=my_agent)
    world.run()

Extension surface for env-builders:

    from fg_env_kernel import (
        registry,
        effect, precondition, resolution, phase, termination_decorator, module,
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

# Primitive discovery (kernel_primitives/*.py drop-in directories) is
# opt-in: call ``fg_env_kernel.discover()`` explicitly, or set
# ``KERNEL_PRIMITIVES_DIR`` — an explicitly configured directory is a
# clear request for discovery, so we honor it at import time. Importing
# the package no longer scans the filesystem or imports arbitrary .py
# files by default.
from .primitives_loader import discover, list_loaded_primitives
import os as _os
if _os.environ.get("KERNEL_PRIMITIVES_DIR"):
    try:
        discover()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("kernel_primitives discovery failed")
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
# SDK facade + typed agent contract
from .action import ActionInstance
from .event import SimEvent
from .state import WorldState
from .runtime.engine import SimulationEngine
from .kernel import DecisionFn, Kernel, OnEventFn, World
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
    "discover",
    "list_loaded_primitives",
    "KernelRegistry",
    "registry",
    "effect",
    "precondition",
    "resolution",
    "phase",
    "module",
    "target_selector",
    "EffectContext",
    # SDK facade + agent contract
    "Kernel",
    "World",
    "DecisionFn",
    "OnEventFn",
    "ActionInstance",
    "WorldState",
    "SimulationEngine",
    "SimEvent",
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
