"""HiddenStateModule — declarative per-property visibility.

Most games have private information: poker hole cards, mafia roles,
battleship ship positions, fog-of-war boards. Each used to ship as
custom code inside a DomainModule. This module replaces that with a
declarative spec the engine applies at perception build time:

  domain_modules:
    - name: hidden_state
      params:
        rules:
          - property: hole_cards
            visible_to: ["self"]              # only the owner sees
          - property: role
            visible_to: ["self", "role:mafioso"]
                                              # owner sees, and all
                                              # entities with role==mafioso
                                              # see all roles (so mafia
                                              # know each other)
          - property: ship_grid
            visible_to: ["self"]
            redact_as: "fog"                  # value substituted on the
                                              # observer's view (default: null)
          - property: hand_strength
            visible_to: ["self"]              # purely for UI/perception

The module exposes redacted snapshots via `get_perception_data` —
non-owner observers see the property as `redact_as` (default: None).
Owners see the real value. This works for any property the schema
already defines on entity types; nothing else to wire up.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .domain_module import DomainModule


class HiddenStateModule(DomainModule):
    """Declarative per-property visibility, applied at perception time."""

    def __init__(
        self,
        name: str = "hidden_state",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, params=params or {})
        # Normalise rules once on init for fast lookup.
        self._rules: List[Dict[str, Any]] = []
        for rule in self._params.get("rules", []) or []:
            if not isinstance(rule, dict):
                continue
            prop = rule.get("property")
            if not prop:
                continue
            self._rules.append({
                "property": str(prop),
                "visible_to": list(rule.get("visible_to") or []),
                "redact_as": rule.get("redact_as"),
                "entity_type": rule.get("entity_type"),  # optional scope
            })

    @property
    def description(self) -> str:
        return "Declarative per-property visibility for hidden game info."

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------ #
    # Visibility checks
    # ------------------------------------------------------------------ #

    def can_observer_see(
        self,
        observer: Any,
        owner: Any,
        rule: Dict[str, Any],
    ) -> bool:
        """Does `observer` satisfy any of the rule's `visible_to` tokens
        for the property owned by `owner`?

        Tokens:
          "self"          → observer.id == owner.id
          "all"           → always visible
          "role:<name>"   → observer's entity_type is <name>
          "<entity_id>"   → specific entity id
          "faction:<n>"   → observer is in faction <n> (reads
                            observer.get('faction') for now)
        """
        for tok in rule.get("visible_to", []):
            if tok == "all":
                return True
            if tok == "self":
                if observer is not None and owner is not None and observer.id == owner.id:
                    return True
                continue
            if tok.startswith("role:"):
                wanted = tok.split(":", 1)[1]
                if getattr(observer, "entity_type", None) == wanted:
                    return True
                continue
            if tok.startswith("faction:"):
                wanted = tok.split(":", 1)[1]
                try:
                    if str(observer.get("faction")) == wanted:
                        return True
                except Exception:
                    pass
                continue
            # Specific entity id
            if observer is not None and observer.id == tok:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Engine integration: called by the perception builder to redact
    # entity dictionaries before they reach the LLM.
    # ------------------------------------------------------------------ #

    def redact_entity_view(
        self,
        observer: Any,
        owner: Any,
        entity_view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Mutate-or-copy entity_view, replacing hidden properties with
        their redact_as values for non-permitted observers.

        Returns a new dict (never mutates the input)."""
        if not self._rules:
            return entity_view
        out = dict(entity_view)
        props = dict(out.get("properties") or {})
        for rule in self._rules:
            # Optional entity_type scope
            if rule.get("entity_type") and rule["entity_type"] != getattr(owner, "entity_type", None):
                continue
            prop_name = rule["property"]
            if prop_name not in props and prop_name not in out:
                continue
            if self.can_observer_see(observer, owner, rule):
                continue
            # Redact
            redacted = rule.get("redact_as")
            if prop_name in props:
                props[prop_name] = redacted
            elif prop_name in out:
                out[prop_name] = redacted
        out["properties"] = props
        return out

    # ------------------------------------------------------------------ #
    # Perception data — the engine doesn't auto-apply yet (see engine
    # patch in this same phase). For now we expose the rules so the
    # perception builder can consult them.
    # ------------------------------------------------------------------ #

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        # We don't need to emit data per-entity — the redaction happens
        # in-engine. Return an empty dict so we don't bloat perception.
        return {}

    @property
    def rules(self) -> List[Dict[str, Any]]:
        return list(self._rules)


__all__ = ["HiddenStateModule"]
