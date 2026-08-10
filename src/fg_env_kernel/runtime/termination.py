"""Termination + winner resolution — decides when a run ends and who won.

The canonical implementations of ``SimulationEngine._check_termination``,
``_evaluate_condition``, ``_resolve_winner`` and the board-pattern helpers,
extracted here so the engine class stays focused on the tick loop. The
engine's methods are now 1-line delegations into this module.

Every import the original bodies relied on is function-local (registry,
board_module, effects DSL, the ``termination`` predicate registry), so this
module stays cheap to import and free of cycles.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .engine import TerminationCondition

logger = logging.getLogger(__name__)


def _check_termination(engine) -> Optional[TerminationCondition]:
    """Evaluate all termination conditions. Returns first triggered, or None."""
    for tc in engine.termination_conditions:
        if engine._evaluate_condition(tc):
            return tc
    return None

def _evaluate_condition(engine, tc: TerminationCondition) -> bool:
    """Evaluate a single termination condition against current state.

    First consults the registry-driven ``termination`` module which
    owns the canonical implementations of most check_types. The
    legacy ``if/elif`` chain below remains for genre-coupled
    check_types (checkmate, board_pattern) that still reference
    engine-internal state directly."""
    # Plugin/registry path — covers all_dead, resource_exhausted,
    # rounds_idle, property_threshold, all_goals_complete,
    # event_triggered, last_one_standing, first_to_score,
    # score_after_n_rounds, count_property, bankruptcy,
    # faction_win, vote_threshold, expr, compound_and/or, plus any
    # custom @termination(...) registration.
    from .. import termination as _term
    check_type = (tc.check_type or "").lower()
    if (check_type in ("expr", "compound_and", "compound_or")
            or engine.registry.terminations.has(check_type)
            or ((tc.params or {}).get("expr") if tc.params else None)):
        return _term.evaluate(engine.state, tc, engine._rng, registry=engine.registry)

    # Legacy fall-through for check_types still living in-engine.
    if tc.check_type == "all_dead":
        entity_type = tc.params.get("entity_type")
        if not entity_type:
            return False
        entities = engine.state.get_entities_by_type(entity_type)
        return len(entities) > 0 and all(not e.alive for e in entities)

    elif tc.check_type == "resource_exhausted":
        resource_name = tc.params.get("resource")
        if not resource_name:
            return False
        pool = engine.state.resources.get(resource_name)
        if not pool:
            return False
        total = sum(pool.holdings.values()) + pool.unallocated
        return total <= 0

    elif tc.check_type == "rounds_idle":
        max_idle = tc.params.get("max_idle_rounds", 3)
        current = engine.state.temporal.current_round
        # Check last N rounds for any resolved actions
        for r in range(max(1, current - max_idle + 1), current + 1):
            events = engine.state.event_log.get_round(r)
            if any(e.event_type == "action_resolved" for e in events):
                return False
        # Only trigger if we've had enough rounds
        return current >= max_idle

    elif tc.check_type == "property_threshold":
        # Single source of truth — the registered checker also handles
        # scope:"world" properties and symbolic operators (">=", …).
        from ..termination import _check_property_threshold
        return _check_property_threshold(engine.state, tc.params, None)

    elif tc.check_type == "all_goals_complete":
        entity_type = tc.params.get("entity_type")
        if entity_type:
            entities = engine.state.get_entities_by_type(entity_type)
        else:
            entities = engine.state.get_agent_entities()
        if not entities:
            return False
        return all(
            engine.state.goals.all_goals_complete(e.id)
            for e in entities if e.alive
        )

    elif tc.check_type == "event_triggered":
        # Check if a specific event type occurred at least N times
        event_type = tc.params.get("event_type")
        min_count = tc.params.get("count", 1)
        if not event_type:
            return False
        total_count = 0
        for r in range(1, engine.state.temporal.current_round + 1):
            events = engine.state.event_log.get_round(r)
            total_count += sum(1 for e in events if e.event_type == event_type)
        return total_count >= min_count

    elif tc.check_type == "compound_and":
        # All sub-conditions must be true
        if not tc.sub_conditions:
            return False
        return all(engine._evaluate_condition(sub) for sub in tc.sub_conditions)

    elif tc.check_type == "compound_or":
        # Any sub-condition must be true
        if not tc.sub_conditions:
            return False
        return any(engine._evaluate_condition(sub) for sub in tc.sub_conditions)

    # ── Phase-1 declarative win predicates ──
    # These replace ~150 lines of per-env win detection. Schema:
    #   termination_conditions:
    #     - { check_type: last_one_standing,
    #         params: { entity_type: MonopolyPlayer, exclude_when: bankrupt } }
    elif tc.check_type == "last_one_standing":
        entity_type = tc.params.get("entity_type")
        exclude_when = tc.params.get("exclude_when", "eliminated")
        if not entity_type:
            return False
        entities = engine.state.get_entities_by_type(entity_type)
        survivors = [
            e for e in entities
            if e.alive and not bool(e.get(exclude_when))
        ]
        return len(survivors) == 1

    elif tc.check_type == "first_to_score":
        # First entity to reach `target` on `property` ends the game.
        # `target` accepts expressions: `$lookup(runtime, wins_needed)`,
        # `$state.X`, etc. — resolved through the standard EffectDSL
        # resolver so runtime-configured thresholds work.
        entity_type = tc.params.get("entity_type")
        prop = tc.params.get("property", "score")
        target_raw = tc.params.get("target")
        from ..effects import resolve_expression, is_expression
        target = (resolve_expression(target_raw, state=engine.state, rng=engine._rng)
                  if is_expression(target_raw) else target_raw)
        if not entity_type or target is None:
            return False
        try:
            target_num = float(target)
        except (TypeError, ValueError):
            return False
        entities = engine.state.get_entities_by_type(entity_type)
        for e in entities:
            if not e.alive:
                continue
            try:
                if float(e.get(prop, 0) or 0) >= target_num:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    elif tc.check_type == "score_after_n_rounds":
        # End after N rounds — winner = highest scorer. We only test
        # the gate here; the engine picks the winner from final state.
        after = int(tc.params.get("after_rounds", 0))
        return engine.state.temporal.current_round >= after

    elif tc.check_type == "board_pattern":
        # N-in-a-row on a board property. Generic enough for
        # tic-tac-toe (3-in-row on 3x3), connect-four (4-in-row on
        # 7x6), gomoku (5-in-row on 19x19).
        #
        # Params:
        #   board_key: entity property carrying a 2-D list (rows of cols)
        #   patterns: ["row_3", "col_3", "diag_3"]  (n inferred from suffix)
        #   by: "any_actor" | "specific_value"
        #   value: when by==specific_value, the cell value to match
        #   entity_type: entity whose property carries the board
        return engine._evaluate_board_pattern(tc.params)

    elif tc.check_type == "vote_threshold":
        # Final-state tally — caller is responsible for setting a
        # vote-count property on entities. Triggers when the highest
        # tally exceeds `threshold_pct` of total cast votes.
        # Tie-break (Tier 2): when `require_unique_max=True`, a tie
        # at the top doesn't trigger the win unless a tie-break rule
        # is configured (`tie_break` ∈ "highest_property" | "random"
        # | "first_alphabetical").
        entity_type = tc.params.get("entity_type")
        prop = tc.params.get("vote_property", "votes_received")
        threshold_pct = float(tc.params.get("threshold_pct", 50.0))
        min_votes = int(tc.params.get("min_total_votes", 1))
        require_unique = bool(tc.params.get("require_unique_max", False))
        if not entity_type:
            return False
        entities = engine.state.get_entities_by_type(entity_type)
        counts = []
        for e in entities:
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
            # Tie at the top — needs a tie-break rule (handled in
            # _resolve_winner). The predicate still fires so the
            # game ends; the winner field will reflect the break.
            return tc.params.get("tie_break") is not None
        return True

    elif tc.check_type == "count_property":
        # 6.L — count entities whose `property` is ≥ / ≤ / == value.
        #   { check_type: count_property,
        #     params: { entity_type, property, operator, value,
        #               min_count?, max_count? } }
        # If `min_count`/`max_count` are present, triggers when the
        # number of MATCHING entities is in [min_count, max_count].
        # Otherwise, triggers when ANY entity matches.
        entity_type = tc.params.get("entity_type")
        prop = tc.params.get("property")
        operator = tc.params.get("operator", "gte")
        value = tc.params.get("value", 0)
        min_count = tc.params.get("min_count")
        max_count = tc.params.get("max_count")
        if not entity_type or not prop:
            return False
        matches = 0
        for e in engine.state.get_entities_by_type(entity_type):
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

    elif tc.check_type == "cooperative_win":
        # Tier 5a — Pandemic-style coop. Game ends in a SHARED win
        # when the world-level condition is met. The predicate
        # evaluates against an arbitrary expression that reads
        # world state (e.g. `$state.tables.diseases_cured == 4`).
        # Params:
        #   if_expr | if_compare | if  (same as CONDITIONAL effect)
        from ..effects import resolve_expression
        spec = tc.params or {}
        return engine._evaluate_world_condition(spec)

    elif tc.check_type == "cooperative_loss":
        # Pandemic-style shared loss. Same predicate forms.
        return engine._evaluate_world_condition(tc.params or {})

    elif tc.check_type == "world_property_threshold":
        # End when an arbitrary world-level expression crosses a
        # threshold. Useful for "X% of map controlled" / "the
        # rebel pool hit 100".
        #   { expr: "$lookup(map, controlled_count)",
        #     operator: "gte", value: 100 }
        return engine._evaluate_world_condition(tc.params or {})

    elif tc.check_type == "checkmate":
        # 6.M — `side` is in checkmate ⇒ the OTHER side wins.
        #   { check_type: checkmate,
        #     params: { board_id, side: "white"|"black" } }
        board_id = tc.params.get("board_id")
        side_param = tc.params.get("side")
        if not engine.state.domain_modules:
            return False
        from ..board_module import BoardModule
        for _m in engine.state.domain_modules._modules.values():
            if not isinstance(_m, BoardModule):
                continue
            if board_id and _m._id != board_id:
                continue
            # If `side` not specified, check both
            sides = [side_param] if side_param else ["white", "black"]
            for s in sides:
                if _m.is_in_checkmate(s, engine.state):
                    return True
        return False

    elif tc.check_type == "faction_win":
        # 6.F — One faction wins when it's the only one with any
        # alive members remaining, OR when its alive count exceeds
        # a configurable threshold of the total alive population.
        #   { check_type: faction_win,
        #     params: { rule: "last_faction_standing" | "majority",
        #               threshold_pct: 50.0,         # for majority rule
        #               required_factions: ["mafia","town"]?  # optional
        #             } }
        rule = tc.params.get("rule", "last_faction_standing")
        threshold_pct = float(tc.params.get("threshold_pct", 50.0))
        mgr = engine.state.factions
        if mgr is None:
            return False
        fac_counts: Dict[str, int] = {}
        for e in engine.state.get_agent_entities():
            if not e.alive:
                continue
            fac = mgr.get_entity_faction(e.id)
            if not fac:
                continue
            fac_counts[fac] = fac_counts.get(fac, 0) + 1
        if not fac_counts:
            return False
        if rule == "last_faction_standing":
            surviving = [f for f, n in fac_counts.items() if n > 0]
            return len(surviving) == 1
        if rule == "majority":
            total = sum(fac_counts.values())
            if total == 0:
                return False
            top = max(fac_counts.values())
            return (top / total) * 100.0 > threshold_pct
        return False

    elif tc.check_type == "bankruptcy":
        # End when only one solvent (money > 0) player remains.
        entity_type = tc.params.get("entity_type")
        money_prop = tc.params.get("money_property", "money")
        min_money = float(tc.params.get("min_money", 0))
        if not entity_type:
            return False
        entities = engine.state.get_entities_by_type(entity_type)
        solvent = [
            e for e in entities
            if e.alive and float(e.get(money_prop, 0) or 0) > min_money
        ]
        return len(solvent) == 1 and len(entities) > 1

    return False

def _resolve_winner(engine, tc) -> Dict[str, Any]:
    """For Phase-1 declarative win predicates, identify WHO won so
    the terminated event carries a winner_id. Returns {} when the
    predicate doesn't have a single-entity winner (e.g. round_limit
    timeouts) — the caller falls back to "draw"."""
    # Plugin/registry path — covers last_one_standing, first_to_score,
    # score_after_n_rounds, count_property, bankruptcy, faction_win,
    # vote_threshold, plus any custom resolver registered via
    # termination.register_winner_resolver(name, fn).
    try:
        from .. import termination as _term
        registered = _term.resolve_winner(engine.state, tc, engine._rng)
        if registered:
            return registered
    except Exception:
        logger.exception("registry winner resolver failed; falling back to legacy")
    try:
        params = tc.params or {}
    except Exception:
        return {}

    if tc.check_type == "last_one_standing":
        entity_type = params.get("entity_type")
        exclude_when = params.get("exclude_when", "eliminated")
        if not entity_type:
            return {}
        for e in engine.state.get_entities_by_type(entity_type):
            if e.alive and not bool(e.get(exclude_when)):
                return {"winner_id": e.id, "winner_name": e.name}
        return {}

    if tc.check_type == "first_to_score":
        entity_type = params.get("entity_type")
        prop = params.get("property", "score")
        target_raw = params.get("target")
        # Mirror _evaluate_condition: target can be a $-expression.
        from ..effects import resolve_expression, is_expression
        target = (resolve_expression(target_raw, state=engine.state, rng=engine._rng)
                  if is_expression(target_raw) else target_raw)
        if not entity_type or target is None:
            return {}
        try:
            target_num = float(target)
        except (TypeError, ValueError):
            return {}
        best, best_v = None, float("-inf")
        # id-sorted iteration → ties resolve to the lowest id
        # deterministically, independent of entity insertion order
        # (which can differ after a snapshot/fork rebuild).
        for e in sorted(engine.state.get_entities_by_type(entity_type), key=lambda e: e.id):
            if not e.alive:
                continue
            try:
                v = float(e.get(prop, 0) or 0)
            except (TypeError, ValueError):
                continue
            if v >= target_num and v > best_v:
                best, best_v = e, v
        if best is not None:
            return {"winner_id": best.id, "winner_name": best.name,
                    "winner_score": best_v}
        return {}

    if tc.check_type == "score_after_n_rounds":
        entity_type = params.get("entity_type")
        prop = params.get("property", "score")
        if not entity_type:
            return {}
        best, best_v = None, float("-inf")
        for e in sorted(engine.state.get_entities_by_type(entity_type), key=lambda e: e.id):
            try:
                v = float(e.get(prop, 0) or 0)
            except (TypeError, ValueError):
                continue
            if v > best_v:
                best, best_v = e, v
        if best is not None:
            return {"winner_id": best.id, "winner_name": best.name,
                    "winner_score": best_v}
        return {}

    if tc.check_type == "vote_threshold":
        entity_type = params.get("entity_type")
        prop = params.get("vote_property", "votes_received")
        tie_break = params.get("tie_break")  # 6.G
        tie_property = params.get("tie_break_property")
        if not entity_type:
            return {}
        ranked = []
        for e in engine.state.get_entities_by_type(entity_type):
            if not e.alive:
                continue
            try:
                v = float(e.get(prop, 0) or 0)
            except (TypeError, ValueError):
                continue
            ranked.append((v, e))
        if not ranked:
            return {}
        # Sort by score, then id — a stable, insertion-order-independent
        # ranking (float votes make a bare score sort non-unique).
        ranked.sort(key=lambda x: (-x[0], x[1].id))
        top_v = ranked[0][0]
        # Float-tolerant top-tier membership: exact == can drop a true
        # co-leader when votes are fractional.
        top_tier = [e for v, e in ranked if abs(v - top_v) < 1e-9]
        if len(top_tier) == 1:
            best = top_tier[0]
            return {"winner_id": best.id, "winner_name": best.name,
                    "winner_votes": top_v}
        # Tie at the top
        if tie_break == "highest_property" and tie_property:
            top_tier.sort(key=lambda e: (-float(e.get(tie_property, 0) or 0), e.id))
            best = top_tier[0]
        elif tie_break == "random":
            # top_tier is already id-sorted above, so _rng.choice indexes
            # into a stable order → reproducible given the seed.
            best = engine._rng.choice(top_tier)
        elif tie_break == "first_alphabetical":
            top_tier.sort(key=lambda e: e.name.lower())
            best = top_tier[0]
        else:
            return {"tied": True,
                    "tied_ids": [e.id for e in top_tier],
                    "winner_votes": top_v}
        return {"winner_id": best.id, "winner_name": best.name,
                "winner_votes": top_v, "tie_broken_by": tie_break}

    if tc.check_type == "count_property":
        # Count-property win is collective — no single entity wins
        # unless the schema names one. Return the count for record.
        entity_type = params.get("entity_type")
        prop = params.get("property")
        if not entity_type or not prop:
            return {}
        matches = []
        for e in engine.state.get_entities_by_type(entity_type):
            if not e.alive:
                continue
            try:
                if float(e.get(prop, 0) or 0) >= float(params.get("value", 0)):
                    matches.append(e)
            except (TypeError, ValueError):
                continue
        if len(matches) == 1:
            m = matches[0]
            return {"winner_id": m.id, "winner_name": m.name}
        return {"match_count": len(matches),
                "match_ids": [e.id for e in matches]}

    if tc.check_type == "cooperative_win":
        # Everyone wins together.
        return {
            "outcome": "cooperative_win",
            "winner_ids": [e.id for e in engine.state.get_agent_entities() if e.alive],
            "winner_names": [e.name for e in engine.state.get_agent_entities() if e.alive],
        }

    if tc.check_type == "cooperative_loss":
        return {"outcome": "cooperative_loss"}

    if tc.check_type == "world_property_threshold":
        return {"outcome": "world_threshold",
                "condition": (tc.params or {}).get("expr") or (tc.params or {}).get("if_compare")}

    if tc.check_type == "checkmate":
        board_id = params.get("board_id")
        if not engine.state.domain_modules:
            return {}
        from ..board_module import BoardModule
        for _m in engine.state.domain_modules._modules.values():
            if not isinstance(_m, BoardModule):
                continue
            if board_id and _m._id != board_id:
                continue
            for s in ("white", "black"):
                if _m.is_in_checkmate(s, engine.state):
                    winner_side = _m._opposite_side_of(s)
                    # Find any entity of winning side to surface a winner_id
                    for ent in engine.state.get_agent_entities():
                        try:
                            if ent.get(_m._side_property) == winner_side:
                                return {"winner_id": ent.id, "winner_name": ent.name,
                                        "winning_side": winner_side,
                                        "checkmated_side": s}
                        except Exception:
                            continue
                    return {"winning_side": winner_side, "checkmated_side": s}
        return {}

    if tc.check_type == "faction_win":
        mgr = engine.state.factions
        if mgr is None:
            return {}
        faction_members: Dict[str, List[Any]] = {}
        for e in engine.state.get_agent_entities():
            if not e.alive:
                continue
            fac = mgr.get_entity_faction(e.id)
            if not fac:
                continue
            faction_members.setdefault(fac, []).append(e)
        if not faction_members:
            return {}
        # Winner = faction with the most alive members
        winning_faction = max(faction_members.keys(), key=lambda f: len(faction_members[f]))
        members = faction_members[winning_faction]
        return {
            "winning_faction": winning_faction,
            "winner_ids": [e.id for e in members],
            "winner_names": [e.name for e in members],
        }

    if tc.check_type == "bankruptcy":
        entity_type = params.get("entity_type")
        money_prop = params.get("money_property", "money")
        if not entity_type:
            return {}
        for e in engine.state.get_entities_by_type(entity_type):
            if e.alive and float(e.get(money_prop, 0) or 0) > 0:
                return {"winner_id": e.id, "winner_name": e.name}
        return {}

    if tc.check_type == "board_pattern":
        # Walk the BoardModule snapshot, find which mark completed
        # the pattern, and look up the entity whose `mark`/`side`
        # matches.
        try:
            from ..board_module import BoardModule
            board_id = params.get("board_id")
            for _mod in (engine.state.domain_modules._modules.values()
                         if engine.state.domain_modules else []):
                if not isinstance(_mod, BoardModule):
                    continue
                if board_id and _mod._id != board_id:
                    continue
                snap = _mod.snapshot_grid(engine.state)
                winning_mark = engine._winning_mark_in_snapshot(
                    snap, params.get("patterns", ["row_3", "col_3", "diag_3"])
                )
                if winning_mark is not None:
                    # Find entity with this mark
                    for ent in engine.state.get_agent_entities():
                        if ent.get("mark") == winning_mark or ent.get("side") == winning_mark:
                            return {"winner_id": ent.id, "winner_name": ent.name,
                                    "winning_mark": winning_mark}
                    return {"winning_mark": winning_mark}
        except Exception:
            logger.exception("_resolve_winner board_pattern failed")
        return {}

    return {}

def _winning_mark_in_snapshot(board, patterns):
    if not board or not isinstance(board, list):
        return None
    rows = len(board)
    cols = len(board[0]) if isinstance(board[0], list) else 0
    for pat in patterns:
        try:
            kind, n_s = pat.rsplit("_", 1)
            n = int(n_s)
        except ValueError:
            continue

        def scan_line(seq):
            for i in range(len(seq) - n + 1):
                window = seq[i:i+n]
                if any(v in (None, "", 0) for v in window):
                    continue
                if all(v == window[0] for v in window):
                    return window[0]
            return None

        if kind == "row":
            for r in range(rows):
                w = scan_line(list(board[r]))
                if w is not None:
                    return w
        elif kind == "col":
            for c in range(cols):
                w = scan_line([board[r][c] for r in range(rows)])
                if w is not None:
                    return w
        elif kind == "diag":
            for r0 in range(rows - n + 1):
                for c0 in range(cols - n + 1):
                    w = scan_line([board[r0+i][c0+i] for i in range(n)])
                    if w is not None:
                        return w
                    w = scan_line([board[r0+i][c0+n-1-i] for i in range(n)])
                    if w is not None:
                        return w
    return None

def _evaluate_board_pattern(engine, params: dict) -> bool:
    """Generic N-in-a-row detector. Reads a 2-D board from either:
      (a) a BoardModule snapshot (preferred if a 'board' domain
          module is registered), OR
      (b) an entity property `board_key` on entities of `entity_type`.
    """
    entity_type = params.get("entity_type")
    board_key = params.get("board_key", "board")
    patterns: List[str] = list(params.get("patterns") or ["row_3", "col_3", "diag_3"])
    mode = params.get("by", "any_actor")
    target_value = params.get("value")

    # ── Path (a): BoardModule snapshot ──
    if engine.state.domain_modules:
        try:
            from ..board_module import BoardModule
            target_board_id = params.get("board_id")
            for _name, _mod in engine.state.domain_modules._modules.items():
                if not isinstance(_mod, BoardModule):
                    continue
                if target_board_id and _mod._id != target_board_id:
                    continue
                snapshot = _mod.snapshot_grid(engine.state)
                if not snapshot:
                    continue
                rows = len(snapshot)
                cols = len(snapshot[0]) if snapshot[0] else 0
                for pat in patterns:
                    try:
                        kind, n_s = pat.rsplit("_", 1)
                        n = int(n_s)
                    except ValueError:
                        continue
                    if engine._scan_board_pattern(snapshot, rows, cols, kind, n, mode, target_value):
                        return True
        except Exception:
            logger.exception("board_pattern WinPredicate: BoardModule snapshot failed")

    # ── Path (b): entity-property board ──
    entities = (
        engine.state.get_entities_by_type(entity_type)
        if entity_type else list(engine.state.get_agent_entities())
    )
    for ent in entities:
        board = ent.get(board_key)
        if not isinstance(board, list) or not board:
            continue
        rows = len(board)
        cols = len(board[0]) if isinstance(board[0], list) else 0
        for pat in patterns:
            try:
                kind, n_s = pat.rsplit("_", 1)
                n = int(n_s)
            except ValueError:
                continue
            if engine._scan_board_pattern(board, rows, cols, kind, n, mode, target_value):
                return True
    return False

def _scan_board_pattern(
    board: List[List[Any]],
    rows: int,
    cols: int,
    kind: str,
    n: int,
    mode: str,
    target_value: Any,
) -> bool:
    def _match(seq: List[Any]) -> bool:
        if len(seq) < n:
            return False
        for i in range(len(seq) - n + 1):
            window = seq[i : i + n]
            if any(v in (None, "", 0) for v in window):
                continue
            if mode == "specific_value":
                if all(v == target_value for v in window):
                    return True
            else:  # any_actor — uniform non-empty run
                first = window[0]
                if all(v == first for v in window):
                    return True
        return False

    if kind == "row":
        return any(_match(list(board[r])) for r in range(rows))
    if kind == "col":
        return any(
            _match([board[r][c] for r in range(rows)])
            for c in range(cols)
        )
    if kind == "diag":
        # main diagonals (\) and anti-diagonals (/)
        for r0 in range(rows - n + 1):
            for c0 in range(cols - n + 1):
                if _match([board[r0 + i][c0 + i] for i in range(n)]):
                    return True
                if _match([board[r0 + i][c0 + n - 1 - i] for i in range(n)]):
                    return True
        return False
    return False

# -------------------------------------------------------------------
# World events
# -------------------------------------------------------------------


__all__ = [
    "_check_termination",
    "_evaluate_condition",
    "_resolve_winner",
    "_winning_mark_in_snapshot",
    "_evaluate_board_pattern",
    "_scan_board_pattern",
]
