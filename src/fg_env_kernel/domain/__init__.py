"""Domain modules — game-genre primitives the JSON refers to by name.

Split into:
  base.py    — DomainModule ABC + DomainConstraint + Manager + Registry
  stubs.py   — Economic / Political / Ecological / Health (stubs)
  markets.py — PredictionMarketModule / SecuritiesTradingModule (full)

A game's template references modules via ``domain_modules: [{"name": "..."}]``
which looks up the class in DomainModuleRegistry. Complex per-game logic
(chess legality, monopoly bookkeeping) lives in ``assets/<game>/module.py``
and is auto-registered at startup.
"""
from .base import (
    DomainConstraint,
    DomainModule,
    DomainModuleManager,
    DomainModuleRegistry,
)
from .stubs import (
    EcologicalModule,
    EconomicModule,
    HealthModule,
    PoliticalModule,
)
from .markets import (
    PredictionMarketModule,
    SecuritiesTradingModule,
)

__all__ = [
    "DomainConstraint",
    "DomainModule",
    "DomainModuleManager",
    "DomainModuleRegistry",
    "EconomicModule",
    "PoliticalModule",
    "EcologicalModule",
    "HealthModule",
    "PredictionMarketModule",
    "SecuritiesTradingModule",
]
