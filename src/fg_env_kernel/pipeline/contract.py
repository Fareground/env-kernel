"""Kernel contract export — what the env-builder agent should target.

Returns a single document combining:

  - The JSON Schema of ``WorldTemplate`` (the static shape)
  - Live capabilities (effects/terminations/resolutions/modules
    currently registered in the kernel)
  - The version of the contract (for forward compatibility)

The agent reads this to know:
  - What fields are accepted
  - Which effect ops, termination check_types, resolution archetypes,
    and domain modules are available *right now* (not stale lists)

Expose this via an HTTP endpoint, write it to disk, or call it from
tests — the contract is a pure dict.

## Usage

    from fg_env_kernel import export_kernel_contract

    contract = export_kernel_contract()
    # Hand it to the agent, or write to disk:
    json.dump(contract, open("kernel_contract.json", "w"), indent=2)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..action import EffectOperation
from .loader import WorldTemplate

logger = logging.getLogger(__name__)


CONTRACT_VERSION = "1.0"


def export_kernel_contract() -> Dict[str, Any]:
    """Build the kernel contract document.

    Returns a JSON-serializable dict with three top-level keys:
      - ``version``           — the contract version string
      - ``template_schema``   — JSON Schema for WorldTemplate
      - ``live_capabilities`` — registry-driven name lists
    """
    from .authoring_shapes import enrich_authoring_shapes
    return {
        "version": CONTRACT_VERSION,
        "template_schema": enrich_authoring_shapes(WorldTemplate.model_json_schema()),
        "live_capabilities": _build_live_capabilities(),
        "notes": _contract_notes(),
    }


def _build_live_capabilities() -> Dict[str, List[str]]:
    """Query the live kernel registry for what's available right now."""
    caps: Dict[str, List[str]] = {
        "effect_operations": [],
        "termination_check_types": [],
        "resolution_archetypes": [],
        "phase_handlers": [],
        "domain_modules": [],
        "precondition_operators": [],
    }

    # Effect ops — built-in enum ∪ registry (P3 plug-ins).
    try:
        from ..registry import registry as _kreg
        caps["effect_operations"] = sorted(
            {op.value for op in EffectOperation} | set(_kreg.effects.keys())
        )
        caps["termination_check_types"] = sorted(
            set(_kreg.terminations.keys()) | {"expr", "compound_and", "compound_or"}
        )
        caps["precondition_operators"] = sorted(set(_kreg.preconditions.keys()))
    except Exception:
        logger.exception("contract: failed to read kernel registry")

    try:
        from ..resolution import RESOLUTION_REGISTRY
        from ..registry import registry as _kreg
        caps["resolution_archetypes"] = sorted(
            set(RESOLUTION_REGISTRY.keys()) | set(_kreg.resolutions.keys())
        )
    except Exception:
        logger.exception("contract: failed to read resolution registry")

    try:
        from ..phase_handlers import PHASE_HANDLER_REGISTRY
        from ..registry import registry as _kreg
        caps["phase_handlers"] = sorted(
            set(PHASE_HANDLER_REGISTRY.keys()) | set(_kreg.phases.keys())
        )
    except Exception:
        logger.exception("contract: failed to read phase registry")

    try:
        from ..domain_module import DomainModuleRegistry
        caps["domain_modules"] = sorted(
            DomainModuleRegistry.get_instance().list_available()
        )
    except Exception:
        logger.exception("contract: failed to read domain module registry")

    return caps


def _contract_notes() -> List[str]:
    """Short reminders surfaced to the agent alongside the schema."""
    return [
        "PREFER `expr` strings ('$actor.gold >= 100 && $actor.alive') for any guard "
        "(precondition, effect.condition, termination). Single grammar, fewer pitfalls.",
        "Built-in effect operations and EVERY registered custom op live in "
        "`live_capabilities.effect_operations` — there is no hidden vocabulary.",
        "Use role='agent' only for decision-making participants, with matching actions. "
        "Autonomous systems use role='object' and executable derived rules, triggers, "
        "physics or domain modules; they do not need dummy decision agents.",
        "termination_conditions are optional but recommended; without one the game "
        "only stops at engine.max_rounds.",
        "Domain modules (chess, monopoly, prediction_market, …) are referenced by "
        "name — DO NOT emit Python. Pick from `live_capabilities.domain_modules`.",
        "Visualization is the ONLY thing an env-builder agent generates as code — "
        "and even that is sandboxed Konva. Game logic is always declarative JSON.",
    ]


__all__ = ["export_kernel_contract", "CONTRACT_VERSION"]
