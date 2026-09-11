"""WorldLoader — THE canonical JSON-to-runnable-engine pipeline.

This is the single source of truth for turning a declarative JSON
template into a working ``WorldState`` + ``SimulationEngine``. Every
game in the system — built-in `assets/<game>/template.json` files,
env-builder-agent output, snapshot restores — flows through
``build_world_state(schema)``.

There is no other builder. The previous service-level builder (445
lines) has been folded in here.

## Contract

An env-builder agent emits a dict matching ``WorldTemplate``. Three
logical layers:

  schema  — type definitions (entities, resources, relations,
             spatial/temporal structure, visibility rules)
  rules   — behavior (actions with preconditions/effects, triggers,
             termination conditions, modules to instantiate)
  viz     — rendering hints (kernel never reads them; viz package owns)

Everything is JSON. Per-game Python is only ever a pre-registered
``DomainModule`` in the library (chess, monopoly, etc.) — the JSON
references them by name. The agent never emits Python.

## Public API

  build_world_state(schema)   — full canonical builder
  load_world(template)        — convenience: build + create engine
  load_world_parts(s, r, v)   — three-document form
  WorldTemplate(...)          — Pydantic model documenting every field
"""
from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..continuous_time import ContinuousTemporalModel

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator, model_serializer, model_validator

from ..action import (
    ActionDefinition,
    Effect,
    EffectCondition,
    EffectOperation,
    Operator,
    Precondition,
    validate_numeric_precondition,
)
from ..engine import SimulationEngine, TerminationCondition
from ..entity import Entity, EntityType
from ..factions import Faction
from ..resource import ResourceType
from ..registry import KernelRegistry, registry as _global_registry
from ..relations import RelationType
from ..spatial import Continuous2DSpace, GraphSpace, GridSpace, NoSpace
from ..state import WorldState
from ..temporal import Phase, TemporalModel, TimeMode
from ..types import PropertySchema, PropertyType
from ..visibility import VisibilityRule

logger = logging.getLogger(__name__)


# ===========================================================================
# Pydantic models — the JSON contract surface (documentation = code)
# ===========================================================================


class PropertySpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: str = "float"          # float | int | string | bool | enum | list
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    hidden: bool = False
    enum_values: Optional[List[Any]] = None


class EntityTypeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    role: str = "agent"          # agent | object | location | abstract
    description: str = ""
    properties: List[PropertySpec] = Field(default_factory=list)


class ResourceTypeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str = ""
    conservation: bool = True
    discrete: bool = True
    min_value: float = 0.0
    max_value: Optional[float] = None
    initial_supply: float = 0


class PreconditionSpec(BaseModel):
    """Either an expression string (``expr``) OR legacy structured form."""
    model_config = ConfigDict(extra="allow")
    expr: Optional[Any] = None
    subject: str = "actor"
    operator: str = "gte"
    value: Any = None
    field: Optional[str] = None
    resource: Optional[str] = None
    relation_type: Optional[str] = None
    description: str = ""

    @model_validator(mode="after")
    def validate_bound(self):
        validate_numeric_precondition(self.operator, self.value, self.expr)
        return self


class EffectConditionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    expr: Optional[Any] = None
    subject: str = "actor"
    field: Optional[str] = None
    operator: str = "gte"
    value: Any = None
    check_type: str = "property"


class EffectSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    operation: str
    target: str = "actor"
    field: Optional[str] = None
    value: Any = None
    resource: Optional[str] = None
    relation_type: Optional[str] = None
    description: str = ""
    condition: Optional[EffectConditionSpec] = None
    scale_by_magnitude: bool = False

    @field_validator("value")
    @classmethod
    def validate_value(cls, value, info):
        from ..effect_values import validate_operand
        operation = info.data.get("operation")
        if operation in {"set", "add", "subtract", "multiply"}:
            validate_operand(value, numeric=operation != "set")
        return value

    @model_serializer(mode="wrap")
    def preserve_missing_value(self, handler):
        data = handler(self)
        if "value" not in self.model_fields_set:
            data.pop("value", None)
        return data


class TriggerDefinition(BaseModel):
    """Event subscription, not a predicate rule. Exported to authoring tools.

    Predicate-based ``when`` / ``then`` belongs in derived_rules. Previously
    arbitrary dictionaries here could compile and silently do nothing.
    Keep the historically supported ``effects`` alias, normalized to effect.
    """
    model_config = ConfigDict(extra="forbid")
    when: StrictStr = Field(min_length=1, description="Exact emitted event type, e.g. round_end; not an expression or condition object.")
    effect: List[EffectSpec] = Field(min_length=1, validation_alias=AliasChoices("effect", "effects"))
    filter: Optional[Dict[str, Any]] = None
    cooldown_rounds: StrictInt = Field(default=0, ge=0)
    once: StrictBool = False
    name: Optional[str] = None
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def legacy_effect_operations(cls, value):
        if not isinstance(value, dict):
            return value
        value = copy.deepcopy(value)
        for key in ("effect", "effects"):
            effects = value.get(key)
            if not isinstance(effects, list):
                continue
            for effect in effects:
                if isinstance(effect, dict) and "op" in effect:
                    if "operation" in effect and effect["op"] != effect["operation"]:
                        raise ValueError("Trigger effect has conflicting operation and op values")
                    effect["operation"] = effect.pop("op")
        return value

    @field_validator("when")
    @classmethod
    def nonempty_event(cls, value):
        if not value.strip():
            raise ValueError("Trigger when must name an emitted event; use derived_rules for predicates")
        return value


class ActionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str = ""
    actor_type: str = ""
    target_type: Optional[str] = None
    preconditions: List[PreconditionSpec] = Field(default_factory=list)
    resolution_archetype: str = "deterministic"
    resolution_params: Dict[str, Any] = Field(default_factory=dict)
    effects_on_success: List[EffectSpec] = Field(default_factory=list)
    effects_on_failure: List[EffectSpec] = Field(default_factory=list)
    effects_on_partial: List[EffectSpec] = Field(default_factory=list)
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    requires_action: Optional[str] = None
    requires_action_success: bool = True
    cooldown_rounds: int = 0
    sequence_rounds: int = 0
    interruptible: bool = True
    message_action: bool = False
    #: Visibility — hidden-information games set broadcast=False so an action
    #: (a night kill, a secret vote) isn't announced to every agent.
    broadcast: bool = True
    range: Optional[float] = None
    unlocks_actions: List[str] = Field(default_factory=list)
    locks_actions: List[str] = Field(default_factory=list)


class EntitySpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str = ""
    entity_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    location_id: Optional[str] = None
    alive: bool = True


class TerminationSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str = ""
    check_type: str = "expr"
    params: Dict[str, Any] = Field(default_factory=dict)
    sub_conditions: List["TerminationSpec"] = Field(default_factory=list)


TerminationSpec.model_rebuild()


class DomainModuleSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class WorldTemplate(BaseModel):
    """The canonical contract for a runnable game — agent target.

    Every game in the system, including the built-in
    ``assets/<game>/template.json`` files, conforms to this shape."""
    model_config = ConfigDict(extra="allow")

    # Metadata
    name: str = "untitled_world"
    description: str = ""
    # `rules` is the rich natural-language brief shown on the
    # Overview tab AND injected into every playing agent's perception
    # under `world_brief`. Markdown is supported. Use for: premise,
    # how-to-play, win conditions in plain English. The agent reads
    # this to decide what to do.
    rules: str = ""
    # Contract version — set by the env-builder; auto-upgraded by
    # compile_template via the versioning module. Templates without
    # this field are treated as the earliest known version.
    contract_version: str = "1.0"

    # Schema layer
    entity_types: List[EntityTypeSpec] = Field(default_factory=list)
    resource_types: List[ResourceTypeSpec] = Field(default_factory=list)
    relation_types: List[Dict[str, Any]] = Field(default_factory=list)
    visibility_rules: List[Dict[str, Any]] = Field(default_factory=list)
    spatial: Dict[str, Any] = Field(default_factory=lambda: {"type": "none"})
    temporal: Dict[str, Any] = Field(default_factory=lambda: {"phases": []})
    tables: Dict[str, Any] = Field(default_factory=dict)

    # Rules layer
    actions: List[ActionSpec] = Field(default_factory=list)
    triggers: List[TriggerDefinition] = Field(default_factory=list)
    derived_rules: List[Dict[str, Any]] = Field(default_factory=list)
    termination_conditions: List[TerminationSpec] = Field(default_factory=list)
    domain_modules: List[DomainModuleSpec] = Field(default_factory=list)

    # Instance layer
    entities: List[EntitySpec] = Field(default_factory=list)
    resource_holdings: List[Dict[str, Any]] = Field(default_factory=list)
    resource_unallocated: List[Dict[str, Any]] = Field(default_factory=list)
    initial_relations: List[Dict[str, Any]] = Field(default_factory=list)
    factions: List[Dict[str, Any]] = Field(default_factory=list)
    adjacency: List[Dict[str, Any]] = Field(default_factory=list)

    # Optional subsystem data
    location_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    goals: List[Dict[str, Any]] = Field(default_factory=list)
    skill_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    initial_skills: List[Dict[str, Any]] = Field(default_factory=list)
    recipes: List[Dict[str, Any]] = Field(default_factory=list)
    connectors: List[Dict[str, Any]] = Field(default_factory=list)
    runtime_parameters: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(default_factory=list)
    last_runtime_params: Dict[str, Any] = Field(default_factory=dict)

    # Subsystem toggles (config only — actual subsystems wire up when enabled)
    cognitive_config: Dict[str, Any] = Field(default_factory=dict)
    social_config: Dict[str, Any] = Field(default_factory=dict)
    crowd_config: Dict[str, Any] = Field(default_factory=dict)

    # Viz layer (opaque to kernel)
    viz: Dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# Lookup maps (module-level — built once)
# ===========================================================================

_PROP_TYPE_MAP: Dict[str, PropertyType] = {
    "float": PropertyType.FLOAT,
    "int": PropertyType.INT,
    "string": PropertyType.STRING,
    "bool": PropertyType.BOOL,
    "enum": PropertyType.ENUM,
    "list": PropertyType.LIST,
}

_OPERATOR_MAP: Dict[str, Operator] = {
    "eq": Operator.EQ, "neq": Operator.NEQ,
    "gt": Operator.GT, "gte": Operator.GTE,
    "lt": Operator.LT, "lte": Operator.LTE,
    "in": Operator.IN,
    "has_resource": Operator.HAS_RESOURCE,
    "at_location": Operator.AT_LOCATION,
    "is_adjacent": Operator.IS_ADJACENT,
    "relation_gte": Operator.RELATION_GTE,
    "is_alive": Operator.IS_ALIVE,
    "same_faction": Operator.SAME_FACTION,
    "different_faction": Operator.DIFFERENT_FACTION,
    "has_item": Operator.HAS_ITEM,
    "has_item_type": Operator.HAS_ITEM_TYPE,
    "inventory_not_full": Operator.INVENTORY_NOT_FULL,
    "skill_gte": Operator.SKILL_GTE,
    "has_recipe": Operator.HAS_RECIPE,
}

# Auto-built so every new EffectOperation is automatically schema-loadable.
_EFFECT_OP_MAP: Dict[str, EffectOperation] = {op.value: op for op in EffectOperation}


# ===========================================================================
# Public API
# ===========================================================================


def build_world_state(
    schema: Dict[str, Any],
    *,
    registry: Optional[KernelRegistry] = None,
) -> WorldState:
    """Build a fully-wired WorldState from a JSON schema dict.

    This is the canonical builder — the single source of truth for
    turning declarative JSON into a runtime state graph. ``registry``
    scopes custom effect-op validation; ``None`` = process-global."""
    state = WorldState()

    _apply_tables_and_runtime(state, schema)
    state._schema_triggers = schema.get("triggers") or []  # type: ignore[attr-defined]  # loader-injected runtime attr, read via getattr()
    # World brief — name + description + rules markdown. Injected into
    # every agent's perception under `world_brief` so playing LLMs read
    # the rules. The agent's chain-of-thought references these for any
    # action choice. Stored as a plain dict on state for easy perception
    # assembly.
    private_information = any(prop.get("hidden") is True
        for entity_type in schema.get("entity_types") or []
        for prop in entity_type.get("properties") or [])
    participant_rules = schema.get("participant_briefing")
    if participant_rules is None and not private_information:
        participant_rules = schema.get("rules", "")
    state._world_brief = {  # type: ignore[attr-defined]  # loader-injected runtime attr, read via getattr()
        "name": schema.get("name", ""),
        "description": "" if private_information else schema.get("description", ""),
        "rules": participant_rules if isinstance(participant_rules, str) else "",
    }
    # Derived rules — the unified inference layer. Stored on state so
    # the engine's tick loop can run them after agent turns.
    derived = schema.get("derived_rules") or []
    if derived:
        from ..derived_rules import DerivedRulesEngine
        state._derived_rules = DerivedRulesEngine(derived)  # type: ignore[attr-defined]  # loader-injected runtime attr, read via getattr()
    else:
        state._derived_rules = None  # type: ignore[attr-defined]  # loader-injected runtime attr, read via getattr()

    _apply_spatial(state, schema.get("spatial") or {})
    _apply_temporal(state, schema.get("temporal") or {})

    _apply_entity_types(state, schema.get("entity_types") or [])
    _apply_resource_types(state, schema.get("resource_types") or [])
    _apply_relation_types(state, schema.get("relation_types") or [])
    _apply_visibility_rules(state, schema.get("visibility_rules") or [])
    _apply_actions(state, schema.get("actions") or [], registry=registry)
    _apply_entities(state, schema.get("entities") or [])
    _apply_resource_holdings(state, schema)
    _apply_initial_relations(state, schema.get("initial_relations") or [])
    _apply_factions(state, schema.get("factions") or [])
    _apply_adjacency(state, schema.get("adjacency") or [])
    _apply_location_definitions(state, schema.get("location_definitions") or [])
    _apply_goals(state, schema.get("goals") or [])
    _apply_skills(state, schema)
    _apply_recipes(state, schema.get("recipes") or [])
    _apply_connectors(state, schema.get("connectors") or [])
    _apply_domain_modules(state, schema.get("domain_modules") or [])

    _apply_physics(state, schema.get("physics") or {})
    _apply_property_dynamics(state, schema.get("property_dynamics") or {})

    _apply_controller(state)
    _apply_cognitive(state, schema.get("cognitive_config") or {})
    _apply_social(state, schema.get("social_config") or {})
    _apply_crowd(state, schema.get("crowd_config") or {})

    # Re-wire module dict to include every manager constructed above.
    state._wire_builtin_modules()
    return state


def load_world(
    template: Union[Dict[str, Any], WorldTemplate],
    *,
    seed: int = 0,
    decision_fn: Any = None,
    on_event: Any = None,
    registry: Optional["KernelRegistry"] = None,
) -> Tuple[WorldState, SimulationEngine]:
    """Build state + engine from a template dict.

    Accepts either a raw dict (as emitted by an env-builder agent) or
    a pre-validated ``WorldTemplate`` instance.

    ``registry`` scopes custom-primitive resolution (effect ops,
    termination checks) to the given ``KernelRegistry``; ``None`` uses
    the process-global registry, preserving existing behavior."""
    schema: Dict[str, Any] = (
        template.model_dump() if isinstance(template, WorldTemplate) else template
    )
    state = build_world_state(schema, registry=registry)
    engine_kwargs: Dict[str, Any] = {}
    # temporal.max_rounds is the template's round budget — honor it here
    # so every load_world caller gets it, not just the SDK facade.
    max_rounds = (schema.get("temporal") or {}).get("max_rounds")
    if max_rounds is not None:
        engine_kwargs["max_rounds"] = int(max_rounds)
    engine = SimulationEngine(
        state,
        seed=seed,
        decision_fn=decision_fn,
        on_event=on_event,
        termination_conditions=build_termination_conditions(schema, state),
        continuous_time=build_continuous_model(schema),
        registry=registry,
        **engine_kwargs,
    )
    return state, engine


def load_world_parts(
    schema: Dict[str, Any],
    rules: Dict[str, Any],
    viz: Optional[Dict[str, Any]] = None,
    *,
    seed: int = 0,
    decision_fn: Any = None,
    on_event: Any = None,
    registry: Optional[KernelRegistry] = None,
) -> Tuple[WorldState, SimulationEngine]:
    """Three-document form: schema + rules + viz as separate dicts."""
    merged = {**schema, **rules}
    if viz:
        merged["viz"] = viz
    return load_world(merged, seed=seed, decision_fn=decision_fn, on_event=on_event,
                      registry=registry)


# ===========================================================================
# Section appliers — each is small, focused, and unit-testable
# ===========================================================================


def _apply_tables_and_runtime(state: WorldState, schema: Dict[str, Any]) -> None:
    """Schema tables for $lookup + runtime_parameters as a layer."""
    tables_raw = schema.get("tables") or {}
    if isinstance(tables_raw, dict):
        state.tables = copy.deepcopy(tables_raw)
    raw_runtime = schema.get("runtime_parameters") or []
    runtime_table: Dict[str, Any] = dict(raw_runtime) if isinstance(raw_runtime, dict) else {}
    if isinstance(raw_runtime, list):
        for rp in raw_runtime:
            if not isinstance(rp, dict):
                continue
            name = rp.get("name")
            if name and "default" in rp:
                runtime_table[str(name)] = rp["default"]
    runtime_snap = schema.get("last_runtime_params") or {}
    if isinstance(runtime_snap, dict):
        for k, v in runtime_snap.items():
            runtime_table[str(k)] = v
    if runtime_table:
        state.tables.setdefault("runtime", {}).update(runtime_table)


def _apply_spatial(state: WorldState, spec: Dict[str, Any]) -> None:
    """Build a SpatialModel from spec.

    Accepts both old (`width/height`, `nodes`) and new
    (`rows/cols`, `locations`) shapes for backwards compat."""
    spatial_type = (spec.get("type") or "none").lower()
    if spatial_type == "grid":
        width = spec.get("width") or spec.get("cols") or 10
        height = spec.get("height") or spec.get("rows") or 10
        state.spatial = GridSpace(width=int(width), height=int(height))
    elif spatial_type == "graph":
        gs = GraphSpace()
        nodes = spec.get("nodes") or spec.get("locations") or []
        for node in nodes:
            gs.add_node(node)
        for edge in spec.get("edges", []):
            if isinstance(edge, dict):
                gs.add_edge(edge["from"], edge["to"], edge.get("weight", 1.0))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                gs.add_edge(edge[0], edge[1], 1.0)
        state.spatial = gs
        # Mirror adjacency for $is_adjacent expressions
        for src, neighbors in gs.edges.items():
            state.adjacency.setdefault(src, []).extend(n for n, _ in neighbors)
    elif spatial_type == "continuous_2d":
        state.spatial = Continuous2DSpace(
            width=float(spec.get("width", 100.0)),
            height=float(spec.get("height", 100.0)),
        )
    else:
        state.spatial = NoSpace()


def _apply_physics(state: WorldState, spec: Dict[str, Any]) -> None:
    """Attach the continuous coupled-dynamics ("physics") system from the
    ``physics`` block, if present. No-op when absent or variable-less."""
    from ..physics import PhysicsModel
    # Initial bindings are evaluated once per fresh world, never written back
    # into the reusable template. Differential equations remain dynamic.
    resolved = {**spec, "params": _resolve_initial(spec.get("params") or {}, state, "physics.params"),
                "variables": [{**v, "value": _resolve_initial(v.get("value", 0), state, f"physics.{v['name']}.value")}
                              for v in spec.get("variables") or []]}
    model = PhysicsModel.from_schema(resolved)
    if model is not None:
        state.physics = model
        state.modules["physics"] = model
        model.integrate(0, state=state)


def build_continuous_model(schema: Dict[str, Any]) -> Optional["ContinuousTemporalModel"]:
    """Build the continuous-time model from a schema, or None for discrete envs.

    Reads ``temporal.mode == 'continuous'`` and a ``temporal.continuous`` config
    block plus per-action ``duration`` fields. Action durations let a slow action
    (a 2-day negotiation) and a fast one (a 1-second quote) coexist on one clock.
    """
    from ..continuous_time import ContinuousTemporalModel

    temporal = schema.get("temporal") or {}
    if temporal.get("mode") != "continuous":
        return None
    cfg = temporal.get("continuous") or {}

    # Per-action durations: action def may declare ``duration`` (sim-time units).
    action_durations: Dict[str, float] = dict(cfg.get("action_durations") or {})
    for a in schema.get("actions") or []:
        if isinstance(a, dict) and a.get("duration") is not None and a.get("name"):
            action_durations[a["name"]] = float(a["duration"])

    return ContinuousTemporalModel(
        action_durations=action_durations,
        default_turn_interval=float(cfg.get("default_turn_interval", 1.0)),
        max_time=float(cfg.get("max_time", 100.0)),
        environment_interval=float(cfg.get("environment_interval", 1.0)),
    )


def _apply_temporal(state: WorldState, spec: Dict[str, Any]) -> None:
    phases = [
        Phase(
            name=p.get("name", f"phase_{i}"),
            description=p.get("description", ""),
            active_roles=list(p.get("active_roles", [])),
            initiative_type=p.get("initiative_type", "fixed"),
            initiative_property=p.get("initiative_property"),
            initiative_descending=p.get("initiative_descending", True),
            handler=p.get("handler"),
            handler_params=p.get("handler_params", {}),
            resolution_mode=p.get("resolution_mode", "sequential"),
        )
        for i, p in enumerate(spec.get("phases") or [{"name": "action"}])
    ]
    mode = TimeMode.CONTINUOUS if spec.get("mode") == "continuous" else TimeMode.DISCRETE
    state.temporal = TemporalModel(
        phases=phases,
        mode=mode,
        # Time mapping — always supported by TemporalModel, never loaded
        # until now: with it, agents get time context in perception and
        # results can speak calendar units instead of abstract rounds.
        round_duration_seconds=spec.get("round_duration_seconds"),
        sim_start_iso=spec.get("sim_start_iso"),
        time_unit_label=spec.get("time_unit_label"),
    )


def _apply_entity_types(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    for et in specs:
        props = [
            PropertySchema(
                name=p["name"],
                type=_PROP_TYPE_MAP.get(p.get("type", "float"), PropertyType.FLOAT),
                default=p.get("default"),
                min_value=p.get("min_value"),
                max_value=p.get("max_value"),
                enum_values=p.get("enum_values"),
                description=p.get("description", ""),
                hidden=p.get("hidden", False),
            )
            for p in et.get("properties", [])
        ]
        state.register_entity_type(EntityType(
            name=et["name"],
            role=et.get("role", "agent"),
            properties=props,
            description=et.get("description", ""),
        ))


def _apply_resource_types(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    for rt in specs:
        state.register_resource_type(ResourceType(
            name=rt["name"],
            conservation=rt.get("conservation", True),
            discrete=rt.get("discrete", True),
            min_value=rt.get("min_value", 0.0),
            max_value=rt.get("max_value"),
            description=rt.get("description", ""),
        ))
        if rt.get("initial_supply"):
            state.resources[rt["name"]].unallocated = float(rt["initial_supply"])


def _apply_relation_types(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    for rel in specs:
        state.register_relation_type(RelationType(
            name=rel["name"],
            symmetric=rel.get("symmetric", False),
            default_value=rel.get("default_value", 0.0),
            min_value=rel.get("min_value", -1.0),
            max_value=rel.get("max_value", 1.0),
            description=rel.get("description", ""),
        ))


def _apply_visibility_rules(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    for vr in specs:
        state.visibility_rules.append(VisibilityRule(
            observer_type=vr.get("observer_type", "*"),
            visible_entity_types=vr.get("visible_entity_types", []),
            visible_properties=vr.get("visible_properties", {}),
            max_range=vr.get("max_range"),
            requires_same_location=vr.get("requires_same_location", False),
            see_relations_involving_self=vr.get("see_relations_involving_self", True),
            see_all_relations=vr.get("see_all_relations", False),
            visible_resources=vr.get("visible_resources", []),
        ))


def _apply_actions(
    state: WorldState,
    specs: List[Dict[str, Any]],
    *,
    registry: Optional[KernelRegistry] = None,
) -> None:
    for ad in specs:
        state.register_action(ActionDefinition(
            name=ad["name"],
            description=ad.get("description", ""),
            actor_type=ad.get("actor_type", ""),
            target_type=ad.get("target_type"),
            parameters=ad.get("parameters", []),
            preconditions=_parse_preconditions(ad.get("preconditions", [])),
            resolution_archetype=ad.get("resolution_archetype", "deterministic"),
            resolution_params=ad.get("resolution_params", {}),
            effects_on_success=_parse_effects(ad.get("effects_on_success", []), registry=registry),
            effects_on_failure=_parse_effects(ad.get("effects_on_failure", []), registry=registry),
            effects_on_partial=_parse_effects(ad.get("effects_on_partial", []), registry=registry),
            requires_action=ad.get("requires_action"),
            requires_action_success=ad.get("requires_action_success", True),
            cooldown_rounds=ad.get("cooldown_rounds", 0),
            sequence_rounds=ad.get("sequence_rounds", 0),
            interruptible=ad.get("interruptible", True),
            message_action=ad.get("message_action", False),
            broadcast=ad.get("broadcast", True),
            range=ad.get("range"),
            unlocks_actions=ad.get("unlocks_actions", []),
            locks_actions=ad.get("locks_actions", []),
        ))


def _parse_preconditions(data: List[Dict[str, Any]]) -> List[Precondition]:
    out: List[Precondition] = []
    for pc in data:
        # Modern: expr form takes precedence
        if pc.get("expr"):
            out.append(Precondition(expr=pc["expr"], description=pc.get("description", "")))
            continue
        op_str = pc.get("operator", "gte")
        operator = _OPERATOR_MAP.get(op_str)
        if operator is None:
            # A silently-defaulted operator INVERTS the gate ("lte" typo →
            # GTE). Loud failure is the only honest translation.
            raise ValueError(
                f"unknown precondition operator {op_str!r} "
                f"(known: {sorted(_OPERATOR_MAP)})"
            )
        out.append(Precondition(
            subject=pc.get("subject", "actor"),
            operator=operator,
            value=pc.get("value"),
            field=pc.get("field"),
            resource=pc.get("resource"),
            relation_type=pc.get("relation_type"),
            description=pc.get("description", ""),
        ))
    return out


def _parse_effects(
    data: List[Dict[str, Any]],
    *,
    registry: Optional[KernelRegistry] = None,
) -> List[Effect]:
    out: List[Effect] = []
    kreg = registry if registry is not None else _global_registry
    for eff in data:
        op_str = eff.get("operation", "set")
        operation: Any = _EFFECT_OP_MAP.get(op_str)
        if operation is None:
            # Check kernel registry (P3 custom ops)
            try:
                if kreg.effects.has(op_str):
                    operation = op_str.lower()
                else:
                    raise ValueError(
                        f"unknown effect operation {op_str!r} — a dropped "
                        "effect is a silently dead action"
                    )
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    f"unknown effect operation {op_str!r} (registry "
                    f"unavailable: {exc})"
                ) from exc

        condition = None
        if eff.get("condition"):
            c = eff["condition"]
            if c.get("expr"):
                condition = EffectCondition(expr=c["expr"])
            else:
                condition = EffectCondition(
                    subject=c.get("subject", "actor"),
                    field=c.get("field"),
                    operator=c.get("operator", "gte"),
                    value=c.get("value"),
                    check_type=c.get("check_type", "property"),
                )

        out.append(Effect(
            target=eff.get("target", "actor"),
            operation=operation,
            field=eff.get("field"),
            value=eff.get("value"),
            value_supplied="value" in eff,
            resource=eff.get("resource"),
            relation_type=eff.get("relation_type"),
            description=eff.get("description", ""),
            condition=condition,
            scale_by_magnitude=eff.get("scale_by_magnitude", False),
        ))
    return out


def _has_initial_lookup(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().startswith("$lookup(")
    if isinstance(value, dict):
        return any(_has_initial_lookup(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_initial_lookup(v) for v in value)
    return False


def _resolve_initial(value: Any, state: WorldState, path: str) -> Any:
    """Resolve explicit table lookups in initial fields with the kernel grammar.

    Action expressions remain dynamic. Prose and ordinary strings are literals;
    missing bindings fail construction instead of becoming text or zero inputs.
    """
    from ..effects import resolve_expression
    if isinstance(value, dict):
        return {k: _resolve_initial(v, state, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_initial(v, state, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str) and _has_initial_lookup(value):
        resolved = resolve_expression(value.strip(), state=state)
        if resolved is None or _has_initial_lookup(resolved):
            raise ValueError(f"Unresolved initial input at {path}: {value}; define the referenced runtime parameter or table key")
        if isinstance(resolved, (float, int)) and not math.isfinite(resolved):
            raise ValueError(f"Nonfinite initial input at {path}")
        return resolved
    return value


def _apply_entities(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    from .initial_values import INITIAL_VALUE_HINT, initial_scalar_error

    for ent in specs:
        # Merge EntityType defaults under explicit per-entity overrides.
        # Without this, declaring an entity with partial properties
        # (e.g. {"color": "red"}) leaves all OTHER declared properties
        # as None — and predicates against them silently fail closed.
        et = state.entity_types.get(ent["entity_type"])
        props: Dict[str, Any] = {}
        if et is not None:
            for pschema in et.properties:
                if pschema.default is not None:
                    props[pschema.name] = pschema.default
        props.update(ent.get("properties") or {})
        bound = {name for name, value in props.items() if _has_initial_lookup(value)}
        props = _resolve_initial(props, state, ent["id"])
        if et is not None:
            for pschema in et.properties:
                error = initial_scalar_error(props.get(pschema.name), pschema.type.value)
                if error:
                    raise ValueError(
                        f"Invalid initial input at {ent['id']}.{pschema.name}: {error}. "
                        f"{INITIAL_VALUE_HINT}"
                    )
                if pschema.name in bound and not pschema.validate(props.get(pschema.name)):
                    raise ValueError(f"Invalid runtime input for {ent['id']}.{pschema.name}; check its type and bounds")
        if et is not None:
            for pschema in et.properties:
                if pschema.enum_values and not pschema.validate(props.get(pschema.name)):
                    raise ValueError(
                        f"Invalid initial value for {ent['id']}.{pschema.name}; "
                        f"choose one of {pschema.enum_values}"
                    )
        entity = Entity(
            id=ent["id"],
            name=ent.get("name", ent["id"]),
            entity_type=ent["entity_type"],
            properties=props,
            location_id=ent.get("location_id"),
            alive=ent.get("alive", True),
        )
        state.spawn_entity(entity)


def _apply_resource_holdings(state: WorldState, schema: Dict[str, Any]) -> None:
    for rh in schema.get("resource_holdings") or []:
        res = rh.get("resource") or rh.get("name")
        entity_id = rh.get("entity_id") or rh.get("entity")
        amount = rh.get("amount", 0)
        pool = state.resources.get(res) if res else None
        if pool and entity_id is not None:
            pool.set(entity_id, amount)
    for ru in schema.get("resource_unallocated") or []:
        res = ru.get("resource") or ru.get("name")
        pool = state.resources.get(res) if res else None
        if pool:
            pool.unallocated = ru.get("amount", 0)


def _apply_initial_relations(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    # Two edge spellings exist in the wild: {from, to, type} and the asset
    # library's {from_entity, to_entity, relation}. Accept both.
    for rel in specs:
        try:
            src = rel.get("from") or rel.get("from_entity")
            dst = rel.get("to") or rel.get("to_entity")
            kind = rel.get("type") or rel.get("relation")
            if not (src and dst and kind):
                raise KeyError("needs from/from_entity, to/to_entity, type/relation")
            state.relations.set(src, dst, kind, rel.get("value", 0.0))
        except Exception:
            logger.exception("initial_relations: bad edge %r", rel)


def _apply_factions(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    for f in specs:
        faction = Faction(
            id=f["id"],
            name=f.get("name", f["id"]),
            description=f.get("description", ""),
            properties=f.get("properties", {}),
            member_ids=list(f.get("member_ids", [])),
            parent=f.get("parent"),
        )
        state.register_faction(faction)


def _apply_property_dynamics(state: WorldState, spec: Dict[str, Any]) -> None:
    """Autonomous per-entity dynamics — drift/spawn/cascade rules, loaded
    declaratively. The engine (PropertyDynamicsEngine) always existed and
    was ticked every round; this is the missing template key that makes
    'prices drift', 'morale decays toward its mean', or 'reinforcements
    spawn when the line breaks' pure JSON."""
    if not spec:
        return
    if not any(spec.get(k) for k in ("drift_rules", "spawn_rules", "cascade_rules")):
        raise ValueError("property_dynamics needs drift_rules, spawn_rules or cascade_rules; no executable updates were defined")
    from ..property_dynamics import PropertyDynamicsEngine

    state.property_dynamics = PropertyDynamicsEngine.from_dict(spec)


def _apply_adjacency(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    for adj in specs:
        loc_id = adj.get("location")
        if loc_id:
            state.adjacency[loc_id] = list(adj.get("neighbors", []))


def _apply_location_definitions(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    if not specs:
        return
    from ..location_properties import LocationDefinition, LocationModifier, LocationTick
    for loc in specs:
        modifiers = [
            LocationModifier(
                property_name=m["property_name"],
                operation=m.get("operation", "add"),
                value=m.get("value", 0),
                entity_type=m.get("entity_type"),
            )
            for m in loc.get("modifiers", [])
        ]
        ticks = [
            LocationTick(
                property_name=t["property_name"],
                operation=t.get("operation", "add"),
                value=t.get("value", 0),
                entity_type=t.get("entity_type"),
            )
            for t in loc.get("tick_effects", [])
        ]
        state.location_properties.register(LocationDefinition(
            id=loc["id"],
            name=loc.get("name", loc["id"]),
            description=loc.get("description", ""),
            properties=loc.get("properties", {}),
            modifiers=modifiers,
            tick_effects=ticks,
            entry_requirements=loc.get("entry_requirements", []),
        ))


def _apply_goals(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    if not specs:
        return
    from ..goals import Goal, GoalCondition
    for gd in specs:
        conditions = [
            GoalCondition(
                check_type=c["check_type"],
                subject=c.get("subject", "self"),
                field=c.get("field"),
                value=c.get("value"),
            )
            for c in gd.get("conditions", [])
        ]
        goal = Goal(
            id=gd["id"],
            description=gd.get("description", ""),
            owner_id=gd["owner_id"],
            conditions=conditions,
            priority=gd.get("priority", 1),
        )
        state.goals.add_goal(gd["owner_id"], goal)


def _apply_skills(state: WorldState, schema: Dict[str, Any]) -> None:
    defs = schema.get("skill_definitions") or []
    if defs:
        from ..skills import SkillDefinition
        for sd in defs:
            state.register_skill_definition(SkillDefinition(
                name=sd["name"],
                description=sd.get("description", ""),
                max_level=sd.get("max_level", 100.0),
                default_level=sd.get("default_level", 0.0),
            ))
    for init in schema.get("initial_skills") or []:
        entity_id = init["entity_id"]
        skill_name = init["skill"]
        level = init.get("level", 0.0)
        xp = init.get("xp", 0.0)
        state.skills.set_level(entity_id, skill_name, level)
        if xp > 0:
            state.skills.award_xp(entity_id, skill_name, xp)


def _apply_recipes(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    if not specs:
        return
    from ..crafting import (
        Recipe, RecipeInput, RecipeResourceInput,
        RecipeOutput, RecipeResourceOutput,
    )
    for r in specs:
        inputs = [
            RecipeInput(item_type=inp["item_type"], quantity=inp.get("quantity", 1))
            for inp in r.get("inputs", [])
        ]
        resource_inputs = [
            RecipeResourceInput(resource=ri["resource"], amount=ri.get("amount", 1.0))
            for ri in r.get("resource_inputs", [])
        ]
        outputs = [
            RecipeOutput(
                item_id_prefix=o.get("item_id_prefix", o.get("item_type", "item")),
                item_name=o.get("item_name", o.get("item_type", "item")),
                item_type=o["item_type"],
                properties=o.get("properties", {}),
                stackable=o.get("stackable", False),
                quantity=o.get("quantity", 1),
            )
            for o in r.get("outputs", [])
        ]
        resource_outputs = [
            RecipeResourceOutput(resource=ro["resource"], amount=ro.get("amount", 1.0))
            for ro in r.get("resource_outputs", [])
        ]
        state.register_recipe(Recipe(
            name=r["name"],
            description=r.get("description", ""),
            inputs=inputs,
            resource_inputs=resource_inputs,
            outputs=outputs,
            resource_outputs=resource_outputs,
            required_skill=r.get("required_skill"),
            required_skill_level=r.get("required_skill_level", 0.0),
            skill_xp_award=r.get("skill_xp_award"),
            skill_xp_amount=r.get("skill_xp_amount", 1.0),
            location_required=r.get("location_required"),
        ))


def _apply_connectors(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    if not specs:
        return
    try:
        from ..connectors import (
            ConnectorConfig, ConnectorInstance, ConnectorManager, ConnectorType,
        )
    except Exception:
        return
    mgr = ConnectorManager()
    for spec in specs:
        try:
            ctype = (
                ConnectorType(spec.get("connector_type", "api").upper())
                if spec.get("connector_type") else ConnectorType.API
            )
        except Exception:
            ctype = ConnectorType.API
        config = ConnectorConfig(
            name=spec["name"],
            connector_type=ctype,
            endpoint=spec.get("config", {}).get("endpoint", ""),
        )
        mgr.add_instance(ConnectorInstance(config=config))
    state.connectors = mgr


def _apply_domain_modules(state: WorldState, specs: List[Dict[str, Any]]) -> None:
    if not specs:
        return
    from ..domain_module import DomainModuleManager, DomainModuleRegistry
    registry = DomainModuleRegistry.get_instance()
    mgr = state.domain_modules or DomainModuleManager()
    for spec in specs:
        logger.info("Creating domain module '%s' with params: %s", spec["name"], spec.get("params", {}))
        module = registry.create(spec["name"], params=_resolve_initial(spec.get("params", {}), state, f"domain_modules.{spec['name']}.params"))
        if module is not None:
            mgr.add_module(module)
    state.domain_modules = mgr


def _apply_controller(state: WorldState) -> None:
    """Always wire SimController so runtime control endpoints work."""
    from ..sim_controller import SimController
    state.controller = SimController()


def _apply_cognitive(state: WorldState, config: Dict[str, Any]) -> None:
    if not config or not config.get("enabled"):
        return
    from ..cognition import CognitionManager
    state.cognition = CognitionManager()


def _apply_social(state: WorldState, config: Dict[str, Any]) -> None:
    if not config or not config.get("enabled"):
        return
    from ..social import SocialPlatformManager, ViralSpreadModel
    viral = ViralSpreadModel(
        model_type=config.get("viral_model", "simple_cascade"),
        spread_probability=config.get("spread_probability", 0.3),
    )
    state.social = SocialPlatformManager(
        viral_model=viral,
        reputation_enabled=config.get("reputation_enabled", True),
    )


def _apply_crowd(state: WorldState, config: Dict[str, Any]) -> None:
    """Always init crowd manager so per-archetype NPC selection works
    even when crowd_ratio is 0."""
    try:
        from agents.crowd_agent import CrowdAgentManager
    except Exception:
        return
    mgr = CrowdAgentManager()
    state.crowd_agents = mgr
    if config and config.get("enabled"):
        state._crowd_config = config  # type: ignore[attr-defined]  # loader-injected runtime attr, read via getattr()
    else:
        state._crowd_config = config or {  # type: ignore[attr-defined]  # loader-injected runtime attr, read via getattr()
            "enabled": True,
            "crowd_ratio": 0,
            "behavior_mapping": {},
            "default_behavior": "noise_trader",
        }


def build_termination_conditions(schema: Dict[str, Any], state: Optional[WorldState] = None) -> List[TerminationCondition]:
    if state is None:
        state = WorldState()
        _apply_tables_and_runtime(state, schema)
    out: List[TerminationCondition] = []
    for tc in schema.get("termination_conditions") or []:
        out.append(_build_termination(_resolve_initial(tc, state, "termination_conditions")))
    return out


# Compatibility for existing integrations; new callers use the public name.
_build_termination_conditions = build_termination_conditions


def _build_termination(spec: Dict[str, Any]) -> TerminationCondition:
    return TerminationCondition(
        name=spec.get("name", ""),
        description=spec.get("description", ""),
        check_type=spec.get("check_type", "expr"),
        params=dict(spec.get("params") or {}),
        sub_conditions=[_build_termination(s) for s in spec.get("sub_conditions") or []],
    )


__all__ = [
    "WorldTemplate",
    "build_termination_conditions",
    "EntityTypeSpec",
    "ResourceTypeSpec",
    "ActionSpec",
    "PreconditionSpec",
    "EffectSpec",
    "EffectConditionSpec",
    "EntitySpec",
    "TerminationSpec",
    "DomainModuleSpec",
    "build_world_state",
    "load_world",
    "load_world_parts",
]
