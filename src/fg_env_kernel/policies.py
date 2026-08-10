"""Built-in agent policies — zero-config brains for ``simulate()``.

The kernel's agent contract is a plain function
(``decision_fn(entity_id, perception, valid_actions) -> ActionInstance | None``),
so a "policy" here is just a factory that returns such a function.
"""
from __future__ import annotations

import hashlib
import random
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .action import ActionInstance

if TYPE_CHECKING:
    from .state import WorldState


def _derive_rng(seed: int, entity_id: str) -> random.Random:
    """A ``random.Random`` deterministically derived from (seed, entity_id).

    Uses SHA-256 over the string form — stable across processes and runs
    (never Python's salted ``hash()``).
    """
    digest = hashlib.sha256(f"{seed}:{entity_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def random_policy(seed: int = 0, state: Optional["WorldState"] = None):
    """Return a seeded random-valid-action ``decision_fn``.

    Each turn it picks a uniformly random action name from
    ``valid_actions`` and returns ``ActionInstance(action_name, actor_id)``.

    Deterministic — including under concurrency: every entity gets its own
    RNG stream, derived stably from ``(seed, entity_id)`` via SHA-256, so
    the same seed replays the same choices even when the engine collects
    decisions from thread-pool workers (simultaneous/parallel phases), and
    one entity's rolls never shift another's. Global random state is never
    touched.

    What it does and doesn't handle:

    - **Target-less actions** — always constructable; this is the core case.
    - **Targeted actions** (``target_type`` set) — handled only when
      ``state`` is bound (``simulate()`` binds it automatically): a random
      alive entity of the declared target type (other than the actor) is
      chosen. If no valid target exists, or no ``state`` is bound, the
      action is skipped as un-constructable.
    - **Actions with required parameters** — never attempted when ``state``
      is bound (the policy cannot invent meaningful values); without
      ``state`` it cannot see parameter specs, so such actions are
      attempted bare and may fail resolution.

    Returns ``None`` (skip the turn) when nothing constructable remains.
    """
    rngs: Dict[str, random.Random] = {}
    rngs_lock = threading.Lock()

    def _rng_for(entity_id: str) -> random.Random:
        with rngs_lock:
            rng = rngs.get(entity_id)
            if rng is None:
                rng = rngs[entity_id] = _derive_rng(seed, entity_id)
            return rng

    def decision_fn(
        entity_id: str, perception: Dict[str, Any], valid_actions: List[str]
    ) -> Optional[ActionInstance]:
        rng = _rng_for(entity_id)
        candidates: List[tuple] = []  # (action_name, target_id | None)
        for name in valid_actions:
            action_def = state.action_definitions.get(name) if state else None
            if action_def is None:
                # No definition visible — attempt the action bare.
                candidates.append((name, None))
                continue
            if any(p.get("required") for p in action_def.parameters):
                continue  # cannot invent required parameter values
            if action_def.target_type:
                targets = [
                    e.id
                    for e in state.entities.values()
                    if e.alive
                    and e.entity_type == action_def.target_type
                    and e.id != entity_id
                ]
                if not targets:
                    continue
                candidates.append((name, rng.choice(sorted(targets))))
            else:
                candidates.append((name, None))
        if not candidates:
            return None
        name, target_id = rng.choice(candidates)
        return ActionInstance(action_name=name, actor_id=entity_id, target_id=target_id)

    return decision_fn
