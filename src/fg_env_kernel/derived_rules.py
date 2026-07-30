"""Derived rules — the unified inference layer.

## What problem this solves

Today, "when X then Y" lives in four different schema shapes:
  1. ``triggers``         — event-driven (when event of type T fires)
  2. ``effects.condition`` — inline conditional on a specific effect
  3. state-machine ``transitions[].when`` — phase-state guards
  4. ``termination_conditions`` — when game ends

All four use the same predicate language (P4), but the SHAPES differ.
A game author thinking in terms of rules has to mentally translate.

## What derived_rules is

A SINGLE rule format that runs at the world level, every round, with
forward-chaining semantics:

    derived_rules: [
      {
        "name": "death_when_zero_hp",
        "when": "$actor.hp <= 0 && $actor.alive",
        "for_each": "$alive_of(Unit)",
        "as": "actor",
        "then": [
          {"operation": "set", "target": "$params.actor",
           "field": "alive", "value": false},
          {"operation": "emit_event", "value": "unit_died",
           "actor_id": "$params.actor"}
        ],
        "once_per_entity": true
      }
    ]

Each round (after agent turns + before termination check), the rule
walks every entity in ``for_each``, evaluates ``when`` with that
entity bound, and if true → fires ``then``. ``once_per_entity`` keeps
the rule from re-firing for the same entity (e.g. a unit dies once).

## Why this matters

This IS forward-chaining inference, exposed as JSON. A user thinking
"whenever X is true, also Y is true" writes one rule, not four
different shapes. It composes with everything else — the `then`
effects can use `invoke_action`, `for_each`, expressions, etc.

## Edge cases

- ``when`` is a standard predicate expression — same grammar as
  preconditions and termination expressions
- ``for_each`` is optional; if omitted, the rule fires globally (no
  per-entity scope), useful for world-level facts like "if total gold
  exceeds 1000, trigger inflation"
- ``once_per_entity`` (or ``once_global``) prevents storm-firing
- Rules fire AFTER agent turns, BEFORE termination check — so they
  can change state that terminations then see (e.g. mark someone
  dead → ``last_one_standing`` fires the same round)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DerivedRule:
    """Compiled form of a derived rule from the schema."""
    name: str
    when: Any                              # expression (string or dict)
    then: List[Any]                        # list of effect specs
    for_each: Optional[str] = None         # expression resolving to a list
    as_var: str = "it"                     # name under $params for per-entity scope
    once_per_entity: bool = False
    once_global: bool = False
    description: str = ""


class DerivedRulesEngine:
    """Holds derived rules + the once-fired tracker. Ticks each round
    after agent turns to apply forward-chaining inference."""

    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        self._rules: List[DerivedRule] = []
        # (rule_name, entity_id_or_global) → bool firing-history
        self._fired: Set[tuple] = set()
        for spec in (rules or []):
            self.add_rule(spec)

    def add_rule(self, spec: Dict[str, Any]) -> None:
        """Compile a JSON rule spec into a DerivedRule."""
        rule = DerivedRule(
            name=str(spec.get("name") or f"rule_{len(self._rules)}"),
            when=spec.get("when"),
            then=spec.get("then") or [],
            for_each=spec.get("for_each"),
            as_var=str(spec.get("as", "it")),
            once_per_entity=bool(spec.get("once_per_entity", False)),
            once_global=bool(spec.get("once_global", False)),
            description=str(spec.get("description", "")),
        )
        self._rules.append(rule)

    def tick(self, engine: Any) -> List[Dict[str, Any]]:
        """Walk every rule, fire matches, return aggregated changes.

        Called by the engine once per round, after agent turns and
        before the termination check. Returns the list of state-change
        records emitted by the fired effects so the engine can log them.
        """
        from .predicates import evaluate as _eval
        from .runtime.effect_dispatch import apply_effects
        from .pipeline.loader import _parse_effects

        all_changes: List[Dict[str, Any]] = []
        for rule in self._rules:
            if rule.once_global and (rule.name, "__global__") in self._fired:
                continue

            # Determine the iteration set
            iter_targets: List[Any]
            if rule.for_each:
                from .effects import resolve_expression
                collection = resolve_expression(
                    rule.for_each, state=engine.state, rng=engine._rng,
                )
                iter_targets = collection if isinstance(collection, list) else []
            else:
                # Global rule — fire once if `when` is true at world scope
                iter_targets = [None]

            for entity_id in iter_targets:
                # Bind entity as $params.<as_var> if iterating.
                # Store the ENTITY OBJECT so dotted access works
                # (`$params.it.hp`). The effect-dispatch resolver knows
                # to pull `.id` when targeting.
                params: Dict[str, Any] = {}
                actor = None
                if entity_id is not None:
                    if isinstance(entity_id, str):
                        actor = engine.state.entities.get(entity_id)
                        params[rule.as_var] = actor if actor is not None else entity_id
                    else:
                        # Already an entity object
                        params[rule.as_var] = entity_id
                        actor = entity_id if hasattr(entity_id, "id") else None

                # Once-per-entity bookkeeping
                bookkeep_key = (rule.name, entity_id if entity_id is not None else "__global__")
                if rule.once_per_entity and bookkeep_key in self._fired:
                    continue
                if rule.once_global and bookkeep_key in self._fired:
                    continue

                # Evaluate the predicate
                ok = _eval(
                    rule.when,
                    actor=actor,
                    params=params,
                    state=engine.state,
                    rng=engine._rng,
                )
                if not ok:
                    continue

                # Fire the effects
                try:
                    parsed = _parse_effects(rule.then)
                    sub_changes = apply_effects(
                        engine, parsed,
                        actor=actor, target=None,
                        params=params,
                        result=None,
                    )
                    if isinstance(sub_changes, list):
                        all_changes.extend(sub_changes)
                    if rule.once_per_entity or rule.once_global:
                        self._fired.add(bookkeep_key)
                except Exception:
                    logger.exception("derived_rule '%s' fire failed", rule.name)

        return all_changes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [
                {
                    "name": r.name,
                    "when": r.when,
                    "then": r.then,
                    "for_each": r.for_each,
                    "as": r.as_var,
                    "once_per_entity": r.once_per_entity,
                    "once_global": r.once_global,
                    "description": r.description,
                }
                for r in self._rules
            ],
            "fired": [list(k) for k in self._fired],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DerivedRulesEngine":
        eng = cls()
        for rspec in (data.get("rules") or []):
            eng.add_rule(rspec)
        for k in (data.get("fired") or []):
            if isinstance(k, list) and len(k) == 2:
                eng._fired.add(tuple(k))
        return eng


__all__ = ["DerivedRule", "DerivedRulesEngine"]
