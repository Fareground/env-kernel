"""Backwards-compatible re-export.

Canonical location: ``fg_env_kernel.domain`` (base / stubs / markets).
The 1,169-line monolith was split into three focused files. External
callers can keep importing from this module — the names below resolve
to the new locations.
"""
from .domain import (  # noqa: F401
    DomainConstraint,
    DomainModule,
    DomainModuleManager,
    DomainModuleRegistry,
    EconomicModule,
    PoliticalModule,
    EcologicalModule,
    HealthModule,
    PredictionMarketModule,
    SecuritiesTradingModule,
)
