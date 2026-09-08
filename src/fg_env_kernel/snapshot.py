"""Atomic restoration of world snapshots; engine checkpoints are a separate concern."""
import copy
import importlib
from typing import Any, Dict

from .entity import Entity
from .resource import ResourcePool, ResourceType
from .temporal import Phase, TemporalModel, TimeMode


class SnapshotRestoreError(ValueError):
    """A snapshot cannot be restored without losing supplied state."""


def _preserved(before: Any, after: Any, path: str) -> None:
    """Permit added defaults for legacy snapshots, but never discarded input."""
    if isinstance(before, dict) and isinstance(after, dict):
        for key, value in before.items():
            if key not in after:
                raise SnapshotRestoreError(f"{path}.{key}: unsupported snapshot field")
            _preserved(value, after[key], f"{path}.{key}")
    elif isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        if len(before) != len(after):
            raise SnapshotRestoreError(f"{path}: restored collection length differs")
        for index, (left, right) in enumerate(zip(before, after)):
            _preserved(left, right, f"{path}[{index}]")
    elif before != after:
        raise SnapshotRestoreError(f"{path}: restored value differs")


def apply_snapshot(state: Any, data: Dict[str, Any]) -> None:
    """Stage replacements before committing, preserving schema and module aliases."""
    data = copy.deepcopy(data)
    staged: Dict[str, Any] = {}
    modules = dict(state.modules)
    section = "snapshot"
    try:
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        if type(data.get("snapshot_version", 1)) is not int or data.get("snapshot_version", 1) not in (1, 2):
            raise ValueError("unsupported snapshot version")
        for section in ("adjacency", "locations", "properties", "tables"):
            if section in data:
                if not isinstance(data[section], dict):
                    raise ValueError("expected an object")
                staged[section] = data[section]
        section = "entities"
        if section in data:
            staged[section] = {}
            for eid, row in data[section].items():
                values = {"id": eid, "name": eid, "entity_type": "", **row}
                entity = Entity(**values)
                if entity.id != eid:
                    raise ValueError("entity key does not match id")
                staged[section][eid] = entity
        section = "resource_definitions"
        resource_types = copy.deepcopy(state.resource_types)
        if section in data:
            resource_types = {name: ResourceType(**row) for name, row in data[section].items()}
            if any(name != resource.name for name, resource in resource_types.items()):
                raise ValueError("resource key does not match name")
            staged["resource_types"] = resource_types
        section = "resources"
        if section in data:
            staged[section] = {}
            for name, row in data[section].items():
                pool = ResourcePool(resource_types[name], row.get("holdings", {}), row.get("unallocated", 0))
                _preserved(row, pool.to_dict(), f"resources.{name}")
                staged[section][name] = pool
        section = "spatial_index"
        if section in data:
            staged[section] = {loc: set(ids) for loc, ids in data[section].items()}
        elif "locations" in data:
            staged[section] = {}
            for eid, loc in data["locations"].items():
                staged[section].setdefault(loc, set()).add(eid)
        section = "temporal"
        if section in data:
            row = data[section]
            phases = copy.deepcopy(state.temporal.phases)
            if "phase_definitions" in row:
                phases = [Phase(**phase) for phase in row["phase_definitions"]]
            elif "phases" in row:
                definitions = {phase.name: phase for phase in phases}
                phases = [definitions.get(name, Phase(name=name)) for name in row["phases"]]
            phase_index = row.get("current_phase_index")
            if phase_index is None:
                names = [phase.name for phase in phases]
                phase_index = names.index(row["current_phase"]) if row.get("current_phase") in names else 0
            temporal = TemporalModel(
                mode=TimeMode(row.get("mode", state.temporal.mode.value)), phases=phases,
                current_round=row.get("current_round", 0), current_phase_index=phase_index,
                current_turn_index=row.get("current_turn_index", 0), turn_order=row.get("turn_order", []),
                round_duration_seconds=row.get("round_duration_seconds"),
                sim_start_iso=row.get("sim_start_iso"), time_unit_label=row.get("time_unit_label"),
            )
            for value in (temporal.current_round, phase_index, temporal.current_turn_index):
                if type(value) is not int or value < 0:
                    raise ValueError("invalid temporal counter")
            _preserved(row, temporal.to_dict(), section)
            staged[section] = temporal
        restorers = {
            "action_history": ("state", "ActionHistory"),
            "status_effects": ("status_effects", "StatusEffectTracker"),
            "relations": ("relations", "RelationGraph"),
            "factions": ("factions", "FactionManager"),
            "sequences": ("sequences", "SequenceTracker"),
            "messages": ("messaging", "MessageBoard"),
            "location_properties": ("location_properties", "LocationPropertyManager"),
            "inventory": ("inventory", "InventoryManager"),
            "goals": ("goals", "GoalTracker"), "skills": ("skills", "SkillTracker"),
            "recipes": ("crafting", "RecipeManager"), "negotiations": ("negotiation", "NegotiationManager"),
            "plans": ("planning", "PlanManager"), "roles": ("roles", "RoleRegistry"),
            "polls": ("polls", "PollManager"),
            "property_dynamics": ("property_dynamics", "PropertyDynamicsEngine"),
            "physics": ("physics", "PhysicsModel"), "connectors": ("connectors", "ConnectorManager"),
            "domain_modules": ("domain_module", "DomainModuleManager"),
            "controller": ("sim_controller", "SimController"), "cognition": ("cognition", "CognitionManager"),
            "social": ("social", "SocialPlatformManager"),
        }
        if data.get("snapshot_version") == 2:
            expected = set(restorers) | {
                "snapshot_version", "resource_definitions", "tables", "entities", "resources",
                "temporal", "entity_types", "action_definitions", "adjacency", "locations",
                "spatial_index", "properties", "world_models", "crowd_agents", "plugin_modules",
                "derived_rules", "domain_module_aliases",
            }
            missing, extra = expected - data.keys(), data.keys() - expected
            if missing or extra:
                raise ValueError(f"incomplete or unsupported v2 snapshot: missing={sorted(missing)}, extra={sorted(extra)}")
        optional = {"property_dynamics", "physics", "connectors", "domain_modules", "controller", "cognition", "social", "crowd_agents"}
        for section in [*restorers, "crowd_agents"]:
            if section not in data:
                continue
            row = data[section]
            if row is None:
                if section not in optional:
                    raise ValueError("required subsystem cannot be null")
                staged[section] = None
                modules.pop(section, None)
                continue
            if section == "crowd_agents":
                cls = type(state.crowd_agents)
            else:
                module, name = restorers[section]
                cls = getattr(importlib.import_module(f"fg_env_kernel.{module}"), name)
            kwargs: Dict[str, Any] = {}
            if section == "status_effects":
                kwargs["definitions"] = state.status_effect_defs
            elif section == "relations":
                kwargs["definitions"] = state.relations.relation_types
            elif section == "domain_modules":
                kwargs["existing"] = state.domain_modules
            restored = cls.from_dict(row, **kwargs)
            _preserved(row, restored.to_dict(), section)
            staged[section] = restored
            modules[section] = restored
        section = "derived_rules"
        if section in data:
            from .derived_rules import DerivedRulesEngine
            restored_rules = DerivedRulesEngine.from_dict(data[section]) if data[section] is not None else None
            if restored_rules is not None:
                _preserved(data[section], restored_rules.to_dict(), section)
            staged["_derived_rules"] = restored_rules
        section = "world_models"
        if section in data:
            from .world_model import AgentWorldModel
            staged[section] = {eid: AgentWorldModel.from_dict(row) for eid, row in data[section].items()}
            _preserved(data[section], {eid: wm.to_dict() for eid, wm in staged[section].items()}, section)
        # Domain aliases must refer to the restored manager's actual modules.
        old_domain = state.domain_modules
        new_domain = staged.get("domain_modules", old_domain)
        aliases = {}
        if old_domain is not None:
            for alias, obj in modules.items():
                for name, old in old_domain._modules.items():
                    if obj is old:
                        aliases[alias] = name
        if "domain_module_aliases" in data:
            for alias in aliases:
                modules.pop(alias, None)
            aliases = data["domain_module_aliases"]
            if not isinstance(aliases, dict):
                raise ValueError("domain module aliases must be an object")
            if any(alias in restorers or alias == "crowd_agents" for alias in aliases):
                raise ValueError("domain alias collides with a built-in subsystem")
        section = "domain_module_aliases"
        for alias, name in aliases.items():
            if new_domain is not None and name in new_domain._modules:
                modules[alias] = new_domain._modules[name]
            elif "domain_module_aliases" in data:
                raise ValueError(f"alias {alias!r} points to a missing domain module")
            else:
                modules.pop(alias, None)
        section = "plugin_modules"
        if section in data:
            for name in list(modules):
                if (name not in restorers and name != "crowd_agents" and name not in aliases
                        and name not in data[section] and callable(getattr(modules[name], "to_dict", None))):
                    del modules[name]
        for name, row in data.get(section, {}).items():
            section = f"plugin_modules.{name}"
            if name in restorers or name == "crowd_agents":
                if data.get("snapshot_version", 1) == 2:
                    raise ValueError("plugin name collides with a built-in subsystem")
                _preserved(row, modules[name].to_dict(), section)
                continue
            if name in aliases:
                _preserved(row, modules[name].to_dict(), section)
                continue
            from .kernel_module import restore_plugin_modules
            restore_plugin_modules(modules, {name: row})
    except Exception as exc:
        raise SnapshotRestoreError(f"Cannot restore {section}: {exc}") from exc
    for attr, value in staged.items():
        setattr(state, attr, value)
    state.modules = modules
