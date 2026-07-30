"""EffectContext — the argument bundle passed to every effect handler.

Every registered effect handler receives a single ``EffectContext``
plus the parsed effect spec. This keeps the handler signature stable
even as the engine grows new arguments, and makes handlers trivially
testable: construct a context, call the handler, assert on the result.

Handlers MUST treat the context as read-only for fields that aren't
explicitly meant for mutation. Mutations go through the helpers
(``apply_property_delta``, ``move_entity``, ``emit_event``) so the
engine can keep its bookkeeping (event log, narrative hooks, snapshot
diff) consistent.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import WorldState
    from .entity import Entity
    from .resolution import ResolutionResult


@dataclass
class EffectContext:
    """All state a handler needs to apply one effect.

    Fields:
        state:       the WorldState being mutated
        actor:       the entity taking the action (may be None for triggers)
        target:      the target entity (may be None)
        params:      action parameters from the ActionInstance
        result:      the ResolutionResult that produced this effect
        rng:         engine's seeded Random (handlers MUST use this for
                     randomness, never `random` module directly)
        emit:        callable to emit an event (type, data) → None
        record:      append a state-change record for narrative/replay
    """
    state: "WorldState"
    actor: Optional["Entity"] = None
    target: Optional["Entity"] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional["ResolutionResult"] = None
    rng: Optional[random.Random] = None
    emit: Optional[Callable[..., None]] = None
    # Sink for state-change descriptions appended during effect application.
    # Engine reads this back to build the round transcript.
    changes: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, change: Dict[str, Any]) -> None:
        """Append a state-change record for the engine's transcript."""
        self.changes.append(change)

    # ---- Expression resolution -----------------------------------------

    def resolve(self, value: Any) -> Any:
        """Resolve a value through the kernel's expression language.

        Strings starting with ``$`` are evaluated against (actor, target,
        params, state). Everything else passes through unchanged."""
        from .effects import resolve_expression, is_expression
        if isinstance(value, str) and is_expression(value):
            return resolve_expression(
                value,
                actor=self.actor,
                target=self.target,
                params=self.params,
                state=self.state,
                rng=self.rng,
            )
        return value


__all__ = ["EffectContext"]
