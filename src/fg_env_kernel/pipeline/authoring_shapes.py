"""Complete authoring references for structures loaded from legacy dictionaries.

These schemas document the loader; they do not change checkpoint loading or
coerce existing worlds. Tests exercise every field against the runtime and
compare phase/rule field coverage so additions cannot silently drift.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..temporal import Phase
from .loader import EffectSpec


class DerivedRuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    when: Any = Field(default=None, description="Predicate expression; evaluated per entity after agent turns each round.")
    then: List[EffectSpec] = Field(min_length=1)
    for_each: Optional[str] = Field(default=None, description="Expression returning entities, e.g. $entities_of(Shop). Omit for a global rule.")
    as_var: str = Field(default="it", alias="as", description="Iteration entity is available as $params.<as>; use it as the effect target.")
    once_per_entity: bool = False
    once_global: bool = False
    description: str = ""


class TemporalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["discrete", "continuous"] = "discrete"
    phases: List[Phase] = Field(default_factory=lambda: [Phase(name="action")],
        description="Each eligible agent acts once per phase per round. active_roles names entity types.")
    round_duration_seconds: Optional[int] = Field(default=None, description="Real-world time represented by one simulation round.")
    sim_start_iso: Optional[str] = None
    time_unit_label: Optional[str] = None


def enrich_authoring_shapes(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a fresh WorldTemplate schema, preserving all native definitions."""
    definitions = schema.setdefault("$defs", {})
    shapes: list[tuple[str, type[BaseModel], bool]] = [
        ("derived_rules", DerivedRuleSpec, True),
        ("temporal", TemporalSpec, False),
    ]
    for field, model, is_list in shapes:
        definition = model.model_json_schema(by_alias=True)
        for name, value in definition.pop("$defs", {}).items():
            if name in definitions and definitions[name] != value:
                raise ValueError(f"Conflicting authoring schema definition: {name}")
            definitions[name] = value
        definitions[model.__name__] = definition
        ref = {"$ref": f"#/$defs/{model.__name__}"}
        schema["properties"][field] = {"type": "array", "items": ref} if is_list else ref
    return schema
