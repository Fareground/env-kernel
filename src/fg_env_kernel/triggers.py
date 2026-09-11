"""TriggerEngine — declarative "when X happens, also Y" framework (Tier 5a).

Subscribes to engine events and fires effect chains when patterns match.
Replaces the bespoke `post_resolution` work many envs do for passive /
reactive abilities.

═══════════════════════════════════════════════════════════════════════
USAGE (schema-level)
═══════════════════════════════════════════════════════════════════════

    triggers:
      - when: "board_mark_placed"           # event_type to match
        filter:                              # optional — narrow the event
          mark: "X"
        effect:                              # effect list to fire
          - { operation: "add", target: "$event.actor", field: "score", value: 1 }
      - when: "monopoly_landed"
        filter: { space: 30 }                # "Go to Jail" space
        effect:
          - { operation: "set", target: "$event.actor", field: "in_jail", value: true }
      - when: "deck_card_drawn"
        filter: { deck_id: "chance" }
        cooldown_rounds: 2                   # don't fire more than once per N rounds per actor
        effect:
          - { operation: "emit_event", value: "chance_card_drawn_reaction" }

The trigger's effect list uses the standard EffectDSL. Expressions
reference the original event via `$event.field` (the event payload),
plus the usual `$actor` / `$target` / `$state` sources.

═══════════════════════════════════════════════════════════════════════
PUBLIC API
═══════════════════════════════════════════════════════════════════════

  engine.triggers.evaluate(event)  → list of (trigger, effects) ready to fire
  engine.triggers.on_fire(actor)   → record cooldown bookkeeping

The engine integrates by calling `_evaluate_triggers(event)` after
each `_emit_event` call. Matching trigger effects are applied through
`_apply_effects` so the same machinery handles them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TriggerSpec:
    when: str                                       # event_type to match
    effect: List[Dict[str, Any]] = field(default_factory=list)
    filter: Optional[Dict[str, Any]] = None         # subset of event.data to match
    cooldown_rounds: int = 0
    once: bool = False                              # fires at most once per sim
    name: Optional[str] = None


@dataclass
class TriggerEngine:
    """Owns the list of triggers + per-trigger cooldown / once state."""
    triggers: List[TriggerSpec] = field(default_factory=list)
    # trigger_name -> last_fired_round_per_actor
    _last_fired: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # trigger_name -> already fired (for `once`)
    _fired_once: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_schema(cls, schema_triggers: List[Dict[str, Any]]) -> "TriggerEngine":
        # Validate direct loader callers too; do not coerce a condition object
        # to an event-name string or silently skip a malformed subscription.
        from .pipeline.loader import TriggerDefinition

        specs: List[TriggerSpec] = []
        for i, t in enumerate(schema_triggers or []):
            declaration = TriggerDefinition.model_validate(t)
            specs.append(TriggerSpec(
                when=declaration.when,
                effect=[effect.model_dump() for effect in declaration.effect],
                filter=declaration.filter,
                cooldown_rounds=declaration.cooldown_rounds,
                once=declaration.once,
                name=declaration.name or f"trigger_{i}",
            ))
        return cls(triggers=specs)

    def matching(self, event_type: str, event_data: Dict[str, Any],
                 actor_id: Optional[str], current_round: int) -> List[TriggerSpec]:
        """Return triggers that should fire for this event."""
        out: List[TriggerSpec] = []
        for spec in self.triggers:
            if spec.when != event_type:
                continue
            # once-guard
            if spec.once and self._fired_once.get(spec.name or "", False):
                continue
            # cooldown
            if spec.cooldown_rounds > 0 and actor_id:
                last = self._last_fired.get(spec.name or "", {}).get(actor_id)
                if last is not None and current_round - last < spec.cooldown_rounds:
                    continue
            # filter
            if spec.filter and not _matches_filter(event_data, spec.filter):
                continue
            out.append(spec)
        return out

    def record_fired(self, spec: TriggerSpec, actor_id: Optional[str], current_round: int) -> None:
        if spec.once:
            self._fired_once[spec.name or ""] = True
        if spec.cooldown_rounds > 0 and actor_id:
            self._last_fired.setdefault(spec.name or "", {})[actor_id] = current_round


def _matches_filter(event_data: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """Check that every key in `flt` matches the value in event_data."""
    for k, v in flt.items():
        # nested dict filter? walk in
        if isinstance(v, dict):
            sub = event_data.get(k)
            if not isinstance(sub, dict) or not _matches_filter(sub, v):
                return False
            continue
        # list filter — event value must be IN the list
        if isinstance(v, list):
            if event_data.get(k) not in v:
                return False
            continue
        if event_data.get(k) != v:
            return False
    return True


__all__ = ["TriggerEngine", "TriggerSpec"]
