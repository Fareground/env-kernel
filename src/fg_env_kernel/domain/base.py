"""Domain modules -- pluggable domain-specific physics packages.

Domain modules extend the simulation with domain-specific rules, constraints,
and behaviors. Examples: economic supply/demand, political voting, ecological
resource cycles, disease spread. This module defines the plugin architecture;
actual domain implementations are registered separately.
"""
from abc import ABC, abstractmethod
import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class DomainConstraint:
    """A constraint/invariant enforced by a domain module."""
    name: str
    description: str = ""
    check_type: str = "property_range"  # property_range, relation_bound, resource_bound, custom
    params: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"  # warning, error


class DomainModule(ABC):
    """Abstract base class for domain-specific physics modules.
    
    Domain modules hook into the simulation engine at multiple points:
    - tick(): per-round logic (e.g., recalculate market prices)
    - validate_action(): domain-specific precondition checks
    - modify_resolution(): tweak resolution inputs (e.g., economic advantage)
    - post_resolution(): side effects after action resolution
    - get_perception_data(): domain-specific info for agent perception
    - get_constraints(): invariants the domain enforces
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self._name = name
        self._params = params or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def description(self) -> str:
        """Human-readable description of this domain module."""
        return ""

    @property
    def required_properties(self) -> List[str]:
        """Entity properties this module expects to exist."""
        return []

    @property
    def custom_actions(self) -> List[str]:
        """Action names this module provides or enhances."""
        return []

    @abstractmethod
    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        """Per-round domain logic. Returns list of changes/events.
        
        Called at the start of each round, after environment ticks
        but before agent turns.
        """
        ...

    def validate_action(self, action_name: str, actor: Any, target: Any,
                         state: Any) -> Optional[str]:
        """Domain-specific action validation.

        Returns None if valid, or an error message string if invalid.
        Called before resolution for actions this module cares about.
        """
        return None

    def filter_valid_actions(
        self,
        entity_id: str,
        valid_actions: List[str],
        state: Any,
    ) -> List[str]:
        """Optionally narrow the list of valid actions for an agent.

        Called after the generic precondition pass but before the LLM
        is asked to choose. Use this to express domain-specific gating
        that doesn't map cleanly to per-action preconditions — e.g. a
        folded poker player has no legal actions for the rest of the
        hand. Default: pass-through.
        """
        return valid_actions

    def modify_resolution(
        self,
        actor_props: Dict[str, Any],
        target_props: Optional[Dict[str, Any]],
        action_def: Any,
        state: Any,
    ) -> tuple:
        """Modify actor/target properties before resolution.
        
        Returns (modified_actor_props, modified_target_props).
        Default: pass through unchanged.
        """
        return actor_props, target_props

    def post_resolution(
        self,
        actor_id: str,
        action_name: str,
        success: bool,
        result: Any,
        state: Any,
    ) -> List[Dict[str, Any]]:
        """Side effects after action resolution. Returns list of changes.
        
        Called after effects are applied for actions this module enhances.
        """
        return []

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        """Domain-specific data to inject into agent perception.
        
        Returns a dict that will be merged into the perception under
        the key "domain_{module_name}".
        """
        return {}

    def get_constraints(self) -> List[DomainConstraint]:
        """Domain invariants/constraints to enforce."""
        return []

    def to_dict(self) -> dict:
        """Serialize module state."""
        return {
            "name": self._name,
            "params": dict(self._params),
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DomainModule":
        """Deserialize — subclasses should override for custom state."""
        return cls(name=data["name"], params=data.get("params", {}))

# ---------------------------------------------------------------------------
# Registry & Manager
# ---------------------------------------------------------------------------

class DomainModuleRegistry:
    """Global registry of available domain module types.
    
    Maps module names to their classes. New modules register themselves
    at import time or via explicit registration.
    """

    _instance: Optional["DomainModuleRegistry"] = None

    def __init__(self):
        self._modules: Dict[str, Type[DomainModule]] = {}
        # Built-in domain modules — lazy-imported here to keep base.py
        # free of forward dependencies on the stub/market subpackages.
        from .stubs import (
            EconomicModule, PoliticalModule, EcologicalModule, HealthModule,
        )
        from .markets import PredictionMarketModule, SecuritiesTradingModule
        self.register("economic", EconomicModule)
        self.register("political", PoliticalModule)
        self.register("ecological", EcologicalModule)
        self.register("health", HealthModule)
        self.register("prediction_market", PredictionMarketModule)
        self.register("securities_trading", SecuritiesTradingModule)
        # Kernel-generic primitives (Phase 1 of the schema-only initiative).
        # These declarative modules replace 80%+ of bespoke per-env code.
        try:
            from ..turn_manager import TurnManagerModule
            from ..hidden_state_module import HiddenStateModule
            from ..board_module import BoardModule
            from ..deck_module import DeckModule
            from ..trade_module import TradeModule
            from ..phase_state_machine import PhaseStateMachineModule
            from ..hand_module import HandModule
            from ..slots_module import SlotsModule
            self.register("turn_manager", TurnManagerModule)
            self.register("hidden_state", HiddenStateModule)
            self.register("board", BoardModule)
            self.register("deck", DeckModule)
            self.register("trade", TradeModule)
            self.register("state_machine", PhaseStateMachineModule)
            self.register("hand", HandModule)
            self.register("slots", SlotsModule)
        except Exception:  # pragma: no cover — defensive at startup
            import logging
            logging.getLogger(__name__).exception(
                "Failed to register generic kernel modules (turn_manager, hidden_state, board, deck, trade, state_machine)"
            )
        # Auto-discover every DomainModule shipped under `assets/<name>/module.py`.
        # The asset loader handles dynamic imports and registration name resolution.
        try:
            from assets import register_modules as _register_asset_modules
        except ImportError:
            # No `assets` package providing register_modules on the path
            # (a bare assets/ media directory also lands here as a
            # namespace package) — normal for the published library;
            # downstream apps that ship one get it auto-loaded.
            pass
        else:
            try:
                _register_asset_modules(self)
            except Exception:  # pragma: no cover — defensive at startup
                import logging
                logging.getLogger(__name__).exception(
                    "Failed to load asset-side domain modules; only built-in stubs are available."
                )
        # Studio-built environments carry their DomainModule source on
        # the WorldDefinition row and are registered at run time by the
        # sim worker — no disk discovery needed (see
        # services/module_codegen.py:register_module_from_source).

    @classmethod
    def get_instance(cls) -> "DomainModuleRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None

    def register(self, name: str, module_class: Type[DomainModule]):
        """Register a domain module type."""
        self._modules[name] = module_class

    def get(self, name: str) -> Optional[Type[DomainModule]]:
        """Get a module class by name."""
        return self._modules.get(name)

    def create(self, name: str, params: Optional[Dict[str, Any]] = None) -> Optional[DomainModule]:
        """Create a module instance by name."""
        module_class = self._modules.get(name)
        if module_class:
            return module_class(name=name, params=params)
        return None

    def list_available(self) -> List[str]:
        """List all registered module names."""
        return list(self._modules.keys())


class DomainModuleManager:
    """Per-simulation manager for active domain modules.
    
    Orchestrates tick/validate/modify/post hooks across all active modules.
    """

    def __init__(self):
        self._modules: Dict[str, DomainModule] = {}

    def add_module(self, module: DomainModule):
        """Add an active domain module."""
        self._modules[module.name] = module

    def remove_module(self, name: str) -> Optional[DomainModule]:
        """Remove a domain module."""
        return self._modules.pop(name, None)

    def get_module(self, name: str) -> Optional[DomainModule]:
        """Get a module by name."""
        return self._modules.get(name)

    def list_active(self) -> List[str]:
        """List names of active modules."""
        return list(self._modules.keys())

    def tick_all(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        """Tick all modules. Returns aggregated changes."""
        all_changes = []
        for module in self._modules.values():
            changes = module.tick(state, round_number)
            all_changes.extend(changes)
        return all_changes

    def validate_action(self, action_name: str, actor: Any, target: Any,
                         state: Any) -> Optional[str]:
        """Check all modules for action validation. Returns first error or None."""
        for module in self._modules.values():
            if action_name in module.custom_actions:
                error = module.validate_action(action_name, actor, target, state)
                if error:
                    return error
        return None

    def filter_valid_actions(self, entity_id: str, valid_actions: List[str], state: Any) -> List[str]:
        """Chain filter_valid_actions through all modules."""
        for module in self._modules.values():
            valid_actions = module.filter_valid_actions(entity_id, valid_actions, state)
        return valid_actions

    def modify_resolution(
        self,
        actor_props: Dict[str, Any],
        target_props: Optional[Dict[str, Any]],
        action_def: Any,
        state: Any,
    ) -> tuple:
        """Chain modify_resolution through all modules."""
        for module in self._modules.values():
            actor_props, target_props = module.modify_resolution(
                actor_props, target_props, action_def, state,
            )
        return actor_props, target_props

    def post_resolution(
        self,
        actor_id: str,
        action_name: str,
        success: bool,
        result: Any,
        state: Any,
    ) -> List[Dict[str, Any]]:
        """Chain post_resolution through all modules."""
        all_changes = []
        for module in self._modules.values():
            changes = module.post_resolution(actor_id, action_name, success, result, state)
            all_changes.extend(changes)
        return all_changes

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        """Aggregate perception data from all modules."""
        data = {}
        for module in self._modules.values():
            module_data = module.get_perception_data(entity_id, state)
            if module_data:
                data[f"domain_{module.name}"] = module_data
        return data

    def get_all_constraints(self) -> List[DomainConstraint]:
        """Collect all constraints from all modules."""
        constraints = []
        for module in self._modules.values():
            constraints.extend(module.get_constraints())
        return constraints

    def to_dict(self) -> dict:
        return {
            "modules": {
                name: mod.to_dict()
                for name, mod in self._modules.items()
            },
            "rng_states": {
                name: {attr: rng.getstate() for attr, rng in vars(mod).items() if isinstance(rng, random.Random)}
                for name, mod in self._modules.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict, *, existing: Optional["DomainModuleManager"] = None) -> "DomainModuleManager":
        manager = cls()
        registry = DomainModuleRegistry.get_instance()
        for name, mod_data in data.get("modules", {}).items():
            current = existing.get_module(name) if existing else None
            mod_class = type(current) if current is not None else (registry.get(name) or registry.get(mod_data.get("type", name)))
            if mod_class is None:
                raise ValueError(f"cannot restore unregistered domain module {name!r}")
            module = mod_class.from_dict(copy.deepcopy(mod_data))
            if module.name != name:
                raise ValueError(f"domain module {name!r} restored with a different name")
            # Constructors sometimes inject defaults; parameters belong to the snapshot.
            module._params = copy.deepcopy(mod_data.get("params", {}))
            manager._modules[name] = module
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value
        for name, states in data.get("rng_states", {}).items():
            if name not in manager._modules:
                raise ValueError(f"random state names unknown domain module {name!r}")
            for attr, rng_state in states.items():
                rng = getattr(manager._modules[name], attr, None)
                if not isinstance(rng, random.Random):
                    raise ValueError(f"cannot restore domain random generator {name}.{attr}")
                rng.setstate(tuples(rng_state))
        return manager
