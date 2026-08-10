"""Termination evaluator — pluggable win/end conditions.

Every termination ``check_type`` is a registered function in
``registry.terminations``. The engine delegates to ``evaluate`` which
looks up the check_type and dispatches. Custom games add new win
conditions with::

    from fg_env_kernel import termination

    @termination("dominion")
    def _dominion(state, params, rng):
        # Triggers when one faction owns 75%+ of tiles
        return ...

    @termination("dominion", aliases=["empire"])  # also discoverable as "empire"
    ...

## Contract for check functions

A check function has the signature::

    (state, params: dict, rng: random.Random) -> bool

It must be pure with respect to its return value — calling it twice
in the same state yields the same answer. Side effects on state are
forbidden; the predicate is for *detection*, not *application*.

For determining the winning entity, register a *winner resolver*
under the same name via :func:`register_winner_resolver`::

    @termination("dominion", aliases=...)
    def _check(...): ...

    def _resolve_dominion(state, params, rng):
        # Return {"winner_id": ..., "winner_name": ...}
        return {...}
    register_winner_resolver("dominion", _resolve_dominion)
"""
from __future__ import annotations

import logging
import random as _random
from typing import Any, Callable, Dict, List, Optional

from .registry import KernelRegistry, registry as _global_registry
from .registry import termination as _termination_decorator
from .predicates import evaluate as evaluate_predicate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Parallel registry of winner resolvers, keyed by check_type name.
_WINNER_RESOLVERS: Dict[str, Callable[..., Dict[str, Any]]] = {}


def register_winner_resolver(
    name: str,
    fn: Callable[[Any, Dict[str, Any], _random.Random], Dict[str, Any]],
) -> None:
    """Register a winner-resolver function paired with a termination
    check_type. Called when the game ends to enrich the termination
    event with `winner_id`, `winning_faction`, etc."""
    _WINNER_RESOLVERS[name.strip().lower()] = fn


def evaluate(
    state: Any,
    condition: Any,
    rng: Optional[_random.Random] = None,
    *,
    registry: Optional["KernelRegistry"] = None,
) -> bool:
    """Evaluate a single termination condition against current state.

    Returns ``True`` if the condition is met. Returns ``False`` when
    the check_type is unknown — callers may fall back to legacy in-
    engine handling for built-ins still living there during migration.

    ``registry`` scopes custom check_type lookup; ``None`` = the
    process-global registry.
    """
    rng = rng or _random.Random(0)
    kreg = registry if registry is not None else _global_registry
    check_type = (getattr(condition, "check_type", None) or "").strip().lower()
    params = getattr(condition, "params", None) or {}

    # Modern: full-expression form. A schema can write
    #   { "check_type": "expr", "params": { "expr": "$count(player,alive)==1" } }
    # or sit ``expr`` alongside another check_type as an AND-guard.
    params_expr = params.get("expr") if isinstance(params, dict) else None
    if check_type == "expr" or params_expr:
        expr = params_expr or params.get("predicate")
        if not expr:
            return False
        ok = evaluate_predicate(expr, state=state, rng=rng)
        if check_type == "expr":
            return ok
        if not ok:
            return False
        # fall through to AND with the named check

    # Compound conditions recurse through this same evaluator
    if check_type == "compound_and":
        subs = getattr(condition, "sub_conditions", None) or []
        return bool(subs) and all(
            evaluate(state, sub, rng, registry=registry) for sub in subs)
    if check_type == "compound_or":
        subs = getattr(condition, "sub_conditions", None) or []
        return bool(subs) and any(
            evaluate(state, sub, rng, registry=registry) for sub in subs)

    # Registry lookup
    fn = kreg.terminations.try_get(check_type)
    if fn is None:
        return False
    try:
        return bool(fn(state, params, rng))
    except Exception:
        logger.exception("termination check '%s' raised", check_type)
        return False


def check_all(
    state: Any,
    conditions: List[Any],
    rng: Optional[_random.Random] = None,
    *,
    registry: Optional["KernelRegistry"] = None,
) -> Optional[Any]:
    """Walk a list of conditions; return the first one that fires, or None."""
    for tc in conditions:
        if evaluate(state, tc, rng, registry=registry):
            return tc
    return None


def resolve_winner(
    state: Any,
    condition: Any,
    rng: Optional[_random.Random] = None,
) -> Dict[str, Any]:
    """Identify the winning entity/faction for a fired termination.

    Returns a dict like ``{"winner_id": ..., "winner_name": ...}`` or
    ``{}`` if no specific winner can be determined (e.g. a draw)."""
    check_type = (getattr(condition, "check_type", None) or "").strip().lower()
    params = getattr(condition, "params", None) or {}
    fn = _WINNER_RESOLVERS.get(check_type)
    if fn is None:
        return {}
    try:
        out = fn(state, params, rng or _random.Random(0))
        return out if isinstance(out, dict) else {}
    except Exception:
        logger.exception("winner resolver '%s' raised", check_type)
        return {}


# ---------------------------------------------------------------------------
# Built-in check_types — registered at import time
# ---------------------------------------------------------------------------

@_termination_decorator("all_dead")
def _check_all_dead(state, params, rng):
    entity_type = params.get("entity_type")
    if not entity_type:
        return False
    entities = state.get_entities_by_type(entity_type)
    return len(entities) > 0 and all(not e.alive for e in entities)


@_termination_decorator("resource_exhausted")
def _check_resource_exhausted(state, params, rng):
    name = params.get("resource")
    if not name:
        return False
    pool = state.resources.get(name)
    if not pool:
        return False
    total = sum(pool.holdings.values()) + pool.unallocated
    return total <= 0


@_termination_decorator("round_limit")
def _check_round_limit(state, params, rng):
    """End the game when ``state.temporal.current_round`` reaches the
    declared ``max_rounds``. The engine's outer loop also enforces a
    hard cap via its ``max_rounds`` constructor arg; this check_type
    lets the schema express the cap declaratively."""
    cap = params.get("max_rounds") or params.get("rounds")
    if cap is None:
        return False
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        return False
    return state.temporal.current_round >= cap


#: ``max_rounds`` / ``max_rounds_reached`` — dialect aliases of round_limit
#: found across the asset library. They were silently DEAD (unregistered)
#: until the lint sweep caught them; register them for real.
_termination_decorator("max_rounds")(_check_round_limit)
_termination_decorator("max_rounds_reached")(_check_round_limit)


@_termination_decorator("entity_count")
def _check_entity_count(state, params, rng):
    """End when the number of qualifying entities reaches ``count``.

    Qualifying = alive entities of ``entity_type``; when ``min_property`` is
    given, only those with ``property >= min_value`` (poker: players with
    chips left)."""
    etype = params.get("entity_type")
    if not etype:
        return False
    target = int(params.get("count", 1))
    prop = params.get("min_property")
    min_value = params.get("min_value", 1)
    n = 0
    for e in state.get_entities_by_type(etype):
        if not e.alive:
            continue
        if prop is not None and not (e.get(prop, 0) >= min_value):
            continue
        n += 1
    return n <= target


@_termination_decorator("entity_property_count")
def _check_entity_property_count(state, params, rng):
    """End when the count of entities whose ``property == expected_value``
    lands inside [min_count, max_count] (monopoly: exactly one solvent
    player left)."""
    etype = params.get("entity_type")
    prop = params.get("property")
    if not etype or not prop:
        return False
    expected = params.get("expected_value")
    n = sum(
        1 for e in state.get_entities_by_type(etype)
        if e.alive and e.get(prop, None) == expected
    )
    lo = params.get("min_count")
    hi = params.get("max_count")
    if lo is not None and n < int(lo):
        return False
    if hi is not None and n > int(hi):
        return False
    return lo is not None or hi is not None


@_termination_decorator("property_stable")
def _check_property_stable(state, params, rng):
    """End when a numeric property stops moving: its round-to-round relative
    change stays under ``threshold`` for ``window`` consecutive rounds.

    History rides the world-properties bag (survives snapshots), keyed per
    condition, so the checker stays a pure function of state."""
    prop = params.get("property")
    if not prop:
        return False
    window = max(1, int(params.get("window", 3)))
    threshold = float(params.get("threshold", 0.01))
    etype = params.get("entity_type")

    # Current value: world property, or the mean across entities carrying it.
    bag = getattr(state, "properties", None)
    value = None
    if bag is not None and prop in bag:
        value = bag.get(prop)
    else:
        entities = (
            state.get_entities_by_type(etype) if etype
            else list(getattr(state, "entities", {}).values())
        )
        vals = [e.get(prop, None) for e in entities if e.alive]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            value = sum(vals) / len(vals)
    if not isinstance(value, (int, float)):
        return False

    if bag is None:
        return False
    hist_key = f"_stability:{prop}"
    hist = bag.get(hist_key)
    if not isinstance(hist, list):
        hist = []
    hist = (hist + [float(value)])[-(window + 1):]
    bag[hist_key] = hist
    if len(hist) < window + 1:
        return False
    for prev, cur in zip(hist, hist[1:]):
        denom = abs(prev) if abs(prev) > 1e-12 else 1.0
        if abs(cur - prev) / denom > threshold:
            return False
    return True


@_termination_decorator("rounds_idle")
def _check_rounds_idle(state, params, rng):
    max_idle = params.get("max_idle_rounds", 3)
    current = state.temporal.current_round
    for r in range(max(1, current - max_idle + 1), current + 1):
        events = state.event_log.get_round(r)
        if any(e.event_type == "action_resolved" for e in events):
            return False
    return current >= max_idle


#: Symbolic aliases authors naturally write ("&gt;=") map onto the named ops.
_THRESHOLD_OPS = {
    "gte": ">=", ">=": ">=",
    "lte": "<=", "<=": "<=",
    "gt": ">", ">": ">",
    "lt": "<", "<": "<",
    "eq": "==", "==": "==", "=": "==",
}


def _threshold_hit(v, operator, value) -> bool:
    op = _THRESHOLD_OPS.get(str(operator))
    if op == ">=": return v >= value
    if op == "<=": return v <= value
    if op == ">":  return v > value
    if op == "<":  return v < value
    if op == "==": return v == value
    return False


@_termination_decorator("property_threshold")
def _check_property_threshold(state, params, rng):
    prop = params.get("property")
    operator = params.get("operator", "gte")
    value = params.get("value", 0)
    if not prop:
        return False
    # scope: "world" (EXPLICIT) reads the world/state properties bag (domain
    # modules record global facts there — e.g. verdict_recorded). A missing
    # entity_type without scope:"world" is a spec bug — fail closed (the lint
    # flags it) rather than silently guessing world scope.
    if params.get("scope") == "world":
        bag = getattr(state, "properties", None) or {}
        if prop in bag:
            return _threshold_hit(bag[prop], operator, value)
        # Legacy modules set world facts as bare attributes on the state.
        legacy = getattr(state, prop, None)
        if isinstance(legacy, (int, float, bool)):
            return _threshold_hit(legacy, operator, value)
        return False
    if not params.get("entity_type"):
        return False
    for e in state.get_entities_by_type(params["entity_type"]):
        if not e.alive:
            continue
        if _threshold_hit(e.get(prop, 0), operator, value):
            return True
    return False


@_termination_decorator("all_goals_complete")
def _check_all_goals_complete(state, params, rng):
    entity_type = params.get("entity_type")
    if entity_type:
        entities = state.get_entities_by_type(entity_type)
    else:
        entities = state.get_agent_entities()
    if not entities:
        return False
    return all(
        state.goals.all_goals_complete(e.id)
        for e in entities if e.alive
    )


@_termination_decorator("event_triggered")
def _check_event_triggered(state, params, rng):
    event_type = params.get("event_type")
    min_count = params.get("count", 1)
    if not event_type:
        return False
    total = 0
    for r in range(1, state.temporal.current_round + 1):
        events = state.event_log.get_round(r)
        total += sum(1 for e in events if e.event_type == event_type)
    return total >= min_count


@_termination_decorator("last_one_standing")
def _check_last_one_standing(state, params, rng):
    entity_type = params.get("entity_type")
    exclude_when = params.get("exclude_when", "eliminated")
    if not entity_type:
        return False
    survivors = [
        e for e in state.get_entities_by_type(entity_type)
        if e.alive and not bool(e.get(exclude_when))
    ]
    return len(survivors) == 1


def _resolve_last_one_standing(state, params, rng):
    entity_type = params.get("entity_type")
    exclude_when = params.get("exclude_when", "eliminated")
    if not entity_type:
        return {}
    for e in state.get_entities_by_type(entity_type):
        if e.alive and not bool(e.get(exclude_when)):
            return {"winner_id": e.id, "winner_name": e.name}
    return {}
register_winner_resolver("last_one_standing", _resolve_last_one_standing)


@_termination_decorator("first_to_score")
def _check_first_to_score(state, params, rng):
    from .effects import resolve_expression, is_expression
    entity_type = params.get("entity_type")
    prop = params.get("property", "score")
    target_raw = params.get("target")
    target = (resolve_expression(target_raw, state=state, rng=rng)
              if is_expression(target_raw) else target_raw)
    if not entity_type or target is None:
        return False
    try:
        target_num = float(target)
    except (TypeError, ValueError):
        return False
    for e in state.get_entities_by_type(entity_type):
        if not e.alive:
            continue
        try:
            if float(e.get(prop, 0) or 0) >= target_num:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _resolve_first_to_score(state, params, rng):
    from .effects import resolve_expression, is_expression
    entity_type = params.get("entity_type")
    prop = params.get("property", "score")
    target_raw = params.get("target")
    target = (resolve_expression(target_raw, state=state, rng=rng)
              if is_expression(target_raw) else target_raw)
    if not entity_type or target is None:
        return {}
    try:
        target_num = float(target)
    except (TypeError, ValueError):
        return {}
    best, best_v = None, float("-inf")
    for e in state.get_entities_by_type(entity_type):
        if not e.alive:
            continue
        try:
            v = float(e.get(prop, 0) or 0)
        except (TypeError, ValueError):
            continue
        if v >= target_num and v > best_v:
            best, best_v = e, v
    if best is not None:
        return {"winner_id": best.id, "winner_name": best.name, "winner_score": best_v}
    return {}
register_winner_resolver("first_to_score", _resolve_first_to_score)


@_termination_decorator("score_after_n_rounds")
def _check_score_after_n_rounds(state, params, rng):
    after = int(params.get("after_rounds", 0))
    return state.temporal.current_round >= after


def _resolve_score_after_n_rounds(state, params, rng):
    entity_type = params.get("entity_type")
    prop = params.get("property", "score")
    if not entity_type:
        return {}
    best, best_v = None, float("-inf")
    for e in state.get_entities_by_type(entity_type):
        try:
            v = float(e.get(prop, 0) or 0)
        except (TypeError, ValueError):
            continue
        if v > best_v:
            best, best_v = e, v
    if best is not None:
        return {"winner_id": best.id, "winner_name": best.name, "winner_score": best_v}
    return {}
register_winner_resolver("score_after_n_rounds", _resolve_score_after_n_rounds)


@_termination_decorator("count_property")
def _check_count_property(state, params, rng):
    entity_type = params.get("entity_type")
    prop = params.get("property")
    operator = params.get("operator", "gte")
    value = params.get("value", 0)
    min_count = params.get("min_count")
    max_count = params.get("max_count")
    if not entity_type or not prop:
        return False
    matches = 0
    for e in state.get_entities_by_type(entity_type):
        if not e.alive:
            continue
        try:
            v = float(e.get(prop, 0) or 0)
        except (TypeError, ValueError):
            continue
        hit = False
        if operator == "gte" and v >= value: hit = True
        elif operator == "lte" and v <= value: hit = True
        elif operator == "gt"  and v >  value: hit = True
        elif operator == "lt"  and v <  value: hit = True
        elif operator == "eq"  and v == value: hit = True
        elif operator == "neq" and v != value: hit = True
        if hit:
            matches += 1
    if min_count is None and max_count is None:
        return matches > 0
    if min_count is not None and matches < int(min_count):
        return False
    if max_count is not None and matches > int(max_count):
        return False
    return True


@_termination_decorator("bankruptcy")
def _check_bankruptcy(state, params, rng):
    entity_type = params.get("entity_type")
    money_prop = params.get("money_property", "money")
    min_money = float(params.get("min_money", 0))
    if not entity_type:
        return False
    entities = state.get_entities_by_type(entity_type)
    solvent = [
        e for e in entities
        if e.alive and float(e.get(money_prop, 0) or 0) > min_money
    ]
    return len(solvent) == 1 and len(entities) > 1


def _resolve_bankruptcy(state, params, rng):
    entity_type = params.get("entity_type")
    money_prop = params.get("money_property", "money")
    if not entity_type:
        return {}
    for e in state.get_entities_by_type(entity_type):
        if e.alive and float(e.get(money_prop, 0) or 0) > 0:
            return {"winner_id": e.id, "winner_name": e.name}
    return {}
register_winner_resolver("bankruptcy", _resolve_bankruptcy)


@_termination_decorator("faction_win")
def _check_faction_win(state, params, rng):
    rule = params.get("rule", "last_faction_standing")
    threshold_pct = float(params.get("threshold_pct", 50.0))
    mgr = state.factions
    if mgr is None:
        return False
    counts: Dict[str, int] = {}
    for e in state.get_agent_entities():
        if not e.alive:
            continue
        fac = mgr.get_entity_faction(e.id)
        if not fac:
            continue
        counts[fac] = counts.get(fac, 0) + 1
    if not counts:
        return False
    if rule == "last_faction_standing":
        return len([f for f, n in counts.items() if n > 0]) == 1
    if rule == "majority":
        total = sum(counts.values())
        if total == 0:
            return False
        top = max(counts.values())
        return (top / total) * 100.0 > threshold_pct
    return False


def _resolve_faction_win(state, params, rng):
    mgr = state.factions
    if mgr is None:
        return {}
    counts: Dict[str, List[Any]] = {}
    for e in state.get_agent_entities():
        if not e.alive:
            continue
        fac = mgr.get_entity_faction(e.id)
        if not fac:
            continue
        counts.setdefault(fac, []).append(e)
    if not counts:
        return {}
    winning_faction = max(counts.keys(), key=lambda f: len(counts[f]))
    members = counts[winning_faction]
    return {
        "winning_faction": winning_faction,
        "winner_ids": [e.id for e in members],
        "winner_names": [e.name for e in members],
    }
register_winner_resolver("faction_win", _resolve_faction_win)


@_termination_decorator("vote_threshold")
def _check_vote_threshold(state, params, rng):
    entity_type = params.get("entity_type")
    prop = params.get("vote_property", "votes_received")
    threshold_pct = float(params.get("threshold_pct", 50.0))
    min_votes = int(params.get("min_total_votes", 1))
    require_unique = bool(params.get("require_unique_max", False))
    if not entity_type:
        return False
    counts = []
    for e in state.get_entities_by_type(entity_type):
        if not e.alive:
            continue
        try:
            counts.append(float(e.get(prop, 0) or 0))
        except (TypeError, ValueError):
            counts.append(0.0)
    total = sum(counts)
    if total < min_votes:
        return False
    top = max(counts)
    if (top / total) * 100.0 <= threshold_pct:
        return False
    if require_unique and counts.count(top) > 1:
        return params.get("tie_break") is not None
    return True


def _resolve_vote_threshold(state, params, rng):
    entity_type = params.get("entity_type")
    prop = params.get("vote_property", "votes_received")
    tie_break = params.get("tie_break")
    tie_property = params.get("tie_break_property")
    if not entity_type:
        return {}
    ranked = []
    for e in state.get_entities_by_type(entity_type):
        if not e.alive:
            continue
        try:
            v = float(e.get(prop, 0) or 0)
        except (TypeError, ValueError):
            continue
        ranked.append((v, e))
    if not ranked:
        return {}
    ranked.sort(key=lambda x: x[0], reverse=True)
    top_v = ranked[0][0]
    top_tier = [e for v, e in ranked if v == top_v]
    if len(top_tier) == 1:
        best = top_tier[0]
        return {"winner_id": best.id, "winner_name": best.name, "winner_votes": top_v}
    if tie_break == "highest_property" and tie_property:
        top_tier.sort(key=lambda e: float(e.get(tie_property, 0) or 0), reverse=True)
        best = top_tier[0]
    elif tie_break == "random":
        best = rng.choice(top_tier)
    elif tie_break == "first_alphabetical":
        top_tier.sort(key=lambda e: e.name.lower())
        best = top_tier[0]
    else:
        return {"tied": True,
                "tied_ids": [e.id for e in top_tier],
                "winner_votes": top_v}
    return {"winner_id": best.id, "winner_name": best.name,
            "winner_votes": top_v, "tie_broken_by": tie_break}
register_winner_resolver("vote_threshold", _resolve_vote_threshold)


__all__ = [
    "evaluate",
    "check_all",
    "resolve_winner",
    "register_winner_resolver",
]
