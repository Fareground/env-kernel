"""Built-in stub DomainModules — minimal interface implementations.

These are placeholders for domain-specific physics that fully-fledged
games would replace with richer logic. They demonstrate the
DomainModule API and provide hooks the env-builder agent can reference
by name (`"domain_modules": [{"name": "economic"}]`).

For real markets see ``fg_env_kernel.domain.markets``.
"""
from typing import Any, Dict, List, Optional

from .base import DomainModule

class EconomicModule(DomainModule):
    """Stub for economic domain: supply/demand, pricing, trade.

    Params:
      - price_elasticity: float (default 1.0) — how sensitive prices are to supply/demand
      - inflation_rate: float (default 0.0) — per-round price increase
    """

    def __init__(self, name: str = "economic", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)

    @property
    def description(self) -> str:
        return "Economic domain: supply/demand dynamics, pricing, trade mechanics"

    @property
    def required_properties(self) -> List[str]:
        return ["wealth", "price"]

    @property
    def custom_actions(self) -> List[str]:
        return ["trade", "set_price", "invest"]

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        """Apply inflation if configured."""
        changes = []
        inflation = self._params.get("inflation_rate", 0.0)
        if inflation > 0:
            changes.append({
                "type": "economic_tick",
                "inflation_applied": inflation,
                "round": round_number,
            })
        return changes

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        return {
            "module": "economic",
            "price_elasticity": self._params.get("price_elasticity", 1.0),
        }


class PoliticalModule(DomainModule):
    """Stub for political domain: voting, influence, policy."""

    def __init__(self, name: str = "political", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)

    @property
    def description(self) -> str:
        return "Political domain: voting systems, influence mechanics, policy effects"

    @property
    def custom_actions(self) -> List[str]:
        return ["vote", "campaign", "propose_policy"]

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        return []


class EcologicalModule(DomainModule):
    """Stub for ecological domain: resource cycles, carrying capacity."""

    def __init__(self, name: str = "ecological", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)

    @property
    def description(self) -> str:
        return "Ecological domain: resource regeneration cycles, carrying capacity, environmental balance"

    @property
    def required_properties(self) -> List[str]:
        return ["population", "carrying_capacity"]

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        return []


class HealthModule(DomainModule):
    """Stub for health domain: disease spread, recovery, immunity."""

    def __init__(self, name: str = "health", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)

    @property
    def description(self) -> str:
        return "Health domain: disease transmission, recovery mechanics, immunity tracking"

    @property
    def required_properties(self) -> List[str]:
        return ["health", "immunity"]

    @property
    def custom_actions(self) -> List[str]:
        return ["treat", "quarantine", "vaccinate"]

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        return []

