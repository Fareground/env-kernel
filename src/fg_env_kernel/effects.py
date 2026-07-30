"""Effect DSL — expression resolution and new effect operations.

This file gives the existing schema-level Effect system the dynamism
that custom DomainModules used to provide. With these extensions an
action like "buy_property" can declaratively express:

  effects_on_success: [
    { target: actor, operation: subtract, field: money,
      value: "$actor.standing_on_space.price" },
    { target: actor, operation: set_owner,
      value: "$actor.standing_on_space" },
    { target: actor, operation: emit_event,
      event_type: "monopoly_bought",
      payload: { space: "$actor.position", price: "$actor.standing_on_space.price" } }
  ]

The kernel resolves the `$…` expressions against actor / target /
params / state / last_event at the moment the effect fires. No
DomainModule code required.

═══════════════════════════════════════════════════════════════════════
EXPRESSION FORMS
═══════════════════════════════════════════════════════════════════════

  $actor               — the entity performing the action
  $actor.money         — entity property access (dotted, list[idx] OK)
  $target              — entity being acted on
  $params              — action parameters dict
  $params.amount       — same, with path
  $last_event          — most recent emitted event
  $last_event.dice_sum — payload field of most recent event
  $result              — ResolutionResult object
  $state               — world state (use sparingly)

  $random(1, 6)        — uniform integer in [1, 6] inclusive
  $random_float(0, 1)  — uniform float
  $random_choice([a, b, c])  — pick one element uniformly
  $dice(2, 6)          — sum of 2 d6
  $lookup("rent", $actor.position)
                       — read state.tables['rent'][key]; key auto-stringified
"""
from __future__ import annotations

import json
import random as _random
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Expression resolver
# ---------------------------------------------------------------------------

# Path-access form: $source.path  or  $source
_PATH_EXPR_PATTERN = re.compile(r"^\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.(.+))?$")
# Function-call form: $name(args)
_FUNC_EXPR_PATTERN = re.compile(r"^\$([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", re.DOTALL)


def is_expression(value: Any) -> bool:
    """True if `value` is a string starting with `$` — i.e. a dynamic
    reference rather than a literal."""
    if not isinstance(value, str) or not value.startswith("$"):
        return False
    return bool(_FUNC_EXPR_PATTERN.match(value) or _PATH_EXPR_PATTERN.match(value))


def resolve_expression(
    expr: Any,
    *,
    actor: Any = None,
    target: Any = None,
    params: Optional[Dict[str, Any]] = None,
    state: Any = None,
    last_event: Optional[Dict[str, Any]] = None,
    result: Any = None,
    rng: Optional[_random.Random] = None,
) -> Any:
    """Resolve a `$…` expression against the current effect context.

    Path-access forms:
      $actor       → the entity performing the action
      $target      → the entity being acted on
      $params      → action parameters dict (`$params.amount`)
      $last_event  → most recent emitted event (`$last_event.dice_sum`)
      $result      → ResolutionResult (`$result.magnitude`, `$result.success_degree`)
      $state       → world state (`$state.temporal.current_round`)

    Function-call forms (Tier 1):
      $random(min, max)       — uniform int in [min, max] inclusive
      $random_float(min, max) — uniform float
      $random_choice([…])     — pick one element uniformly
      $dice(n, sides)         — sum of n d-sided dice
      $lookup(table, key)     — state.tables[table][stringified_key]

    Returns the resolved value, or the original expression string if
    unresolvable (so misconfigurations are visible in event logs
    rather than silently zeroed).
    """
    if not is_expression(expr):
        return expr

    # Function-call form first — `$name(args)` could otherwise be matched
    # as `$name` with no path.
    func_m = _FUNC_EXPR_PATTERN.match(expr)
    if func_m:
        fn_name, args_str = func_m.group(1), func_m.group(2)
        args = _parse_args(
            args_str,
            actor=actor, target=target, params=params, state=state,
            last_event=last_event, result=result, rng=rng,
        )
        return _call_function(
            fn_name, args, state=state, rng=rng,
        )

    m = _PATH_EXPR_PATTERN.match(expr)
    if not m:
        return expr
    source, path = m.group(1), m.group(2) or ""

    sources = {
        "actor": actor,
        "target": target,
        "params": params or {},
        "last_event": last_event or {},
        "result": result,
        "state": state,
    }

    base = sources.get(source)
    if base is None:
        return expr  # unresolved — let the caller log

    if not path:
        return base
    return _walk_path(base, path) if path else base


def _walk_path(obj: Any, path: str) -> Any:
    """Walk a dotted path through an object/dict/entity, supporting:
      foo.bar            → attribute / key / Entity property
      foo.bar[0]         → list / tuple indexing
      foo.bar.baz        → nested
    Returns None if the path is unreachable.
    """
    cur = obj
    for segment in _split_path(path):
        if cur is None:
            return None
        # List / tuple index
        if isinstance(segment, int):
            try:
                cur = cur[segment]
            except (IndexError, TypeError, KeyError):
                return None
            continue
        # Entity-style: prefer .get(prop) if available
        if hasattr(cur, "get") and callable(cur.get):
            try:
                val = cur.get(segment)
                if val is not None:
                    cur = val
                    continue
            except (TypeError, KeyError):
                pass
        # Dict
        if isinstance(cur, dict):
            cur = cur.get(segment)
            continue
        # Attribute fallback
        cur = getattr(cur, segment, None)
    return cur


def _split_path(path: str) -> Tuple[Any, ...]:
    """Tokenize `a.b[0].c` → ('a', 'b', 0, 'c')."""
    parts: list = []
    for chunk in path.split("."):
        # peel any [N] suffixes off
        while "[" in chunk and chunk.endswith("]"):
            head, idx = chunk[:-1].rsplit("[", 1)
            if head:
                parts.append(head)
            try:
                parts.append(int(idx))
            except ValueError:
                parts.append(idx)
            chunk = ""
        if chunk:
            parts.append(chunk)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Token registry — declarative "game tokens" carried by entities.
# Replaces ad-hoc int counters like `jail_cards`, `get_out_of_jail_free`,
# `immunity_until_round`, etc. that custom modules maintain.
# Stored on the entity itself under the property `_tokens` (dict of
# token_name → count) so it auto-serialises with the entity state.
# ---------------------------------------------------------------------------

_TOKENS_PROP = "_tokens"


def get_tokens(entity: Any) -> Dict[str, int]:
    if entity is None:
        return {}
    try:
        existing = entity.get(_TOKENS_PROP)
    except Exception:
        existing = None
    if isinstance(existing, dict):
        return existing
    fresh: Dict[str, int] = {}
    try:
        entity.set(_TOKENS_PROP, fresh)
    except Exception:
        pass
    return fresh


def grant_token(entity: Any, name: str, count: int = 1) -> int:
    tokens = get_tokens(entity)
    tokens[name] = int(tokens.get(name, 0)) + int(count)
    try:
        entity.set(_TOKENS_PROP, tokens)
    except Exception:
        pass
    return tokens[name]


def consume_token(entity: Any, name: str, count: int = 1) -> bool:
    """Spend `count` of token `name`. Returns True if the entity had
    enough, False otherwise."""
    tokens = get_tokens(entity)
    have = int(tokens.get(name, 0))
    if have < count:
        return False
    tokens[name] = have - count
    if tokens[name] <= 0:
        tokens.pop(name, None)
    try:
        entity.set(_TOKENS_PROP, tokens)
    except Exception:
        pass
    return True


def has_token(entity: Any, name: str, min_count: int = 1) -> bool:
    return int(get_tokens(entity).get(name, 0)) >= int(min_count)


# ---------------------------------------------------------------------------
# Function-call expressions ($random, $dice, $random_choice, $lookup)
# ---------------------------------------------------------------------------

def _parse_args(
    args_str: str,
    **ctx: Any,
) -> List[Any]:
    """Tokenise the inside of `$func(...)` into a list of resolved args.

    Each top-level comma-separated item is either:
      - a `$…` expression (recursively resolved with the same context)
      - a JSON literal (`1`, `"red"`, `[1, 2, 3]`, `true`, `null`)

    Top-level means brackets / quotes are respected when splitting.
    """
    s = args_str.strip()
    if not s:
        return []
    tokens = _split_args_top_level(s)
    out: List[Any] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t.startswith("$"):
            out.append(resolve_expression(t, **ctx))
            continue
        # Try JSON
        try:
            out.append(json.loads(t))
            continue
        except ValueError:
            pass
        # Single-quoted string — expressions live inside JSON documents,
        # so 'contact' is the natural spelling of a string arg there.
        if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
            out.append(t[1:-1])
            continue
        # Bare identifier — treat as a string token (lets users write
        # `$lookup(rent, 12)` without quoting the table name).
        out.append(t)
    return out


def _split_args_top_level(s: str) -> List[str]:
    """Split on commas at bracket/paren/quote depth 0."""
    out: List[str] = []
    depth = 0
    in_quote: Optional[str] = None
    buf: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if in_quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch in ")]}":
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf))
    return out


def _call_function(
    name: str,
    args: List[Any],
    *,
    state: Any = None,
    rng: Optional[_random.Random] = None,
) -> Any:
    """Dispatch a `$func(...)` expression. Falls back to None on
    unknown function or bad args — the caller can log."""
    r = rng or _DEFAULT_RNG

    if name == "random":
        # $random(min, max) → uniform int inclusive
        if len(args) >= 2:
            try:
                lo, hi = int(args[0]), int(args[1])
                if lo > hi:
                    lo, hi = hi, lo
                return r.randint(lo, hi)
            except (TypeError, ValueError):
                return None
        return None

    if name == "random_float":
        if len(args) >= 2:
            try:
                return r.uniform(float(args[0]), float(args[1]))
            except (TypeError, ValueError):
                return None
        return None

    if name == "random_choice":
        if not args:
            return None
        # Either $random_choice(a, b, c) or $random_choice([a, b, c])
        pool = args[0] if (len(args) == 1 and isinstance(args[0], list)) else list(args)
        if not pool:
            return None
        return r.choice(pool)

    if name == "dice":
        # $dice(n, sides) — sum of n d-sided rolls
        n = int(args[0]) if args else 1
        sides = int(args[1]) if len(args) > 1 else 6
        n = max(0, n); sides = max(1, sides)
        return sum(r.randint(1, sides) for _ in range(n))

    if name == "lookup":
        # $lookup(table_name, key, [sub_key]) — read state.tables[table][key]
        if len(args) < 2 or state is None:
            return None
        tables = getattr(state, "tables", None)
        if not tables:
            return None
        table = tables.get(str(args[0]))
        if table is None:
            return None
        key = args[1]
        # Allow int → string lookup transparently
        val = table.get(key) if isinstance(table, dict) else None
        if val is None and isinstance(table, dict):
            val = table.get(str(key))
        if val is not None and len(args) >= 3:
            # Optional sub-key: $lookup("rent", "1", 3) → rent[1][3]
            sub = args[2]
            if isinstance(val, list):
                try:
                    val = val[int(sub)]
                except (IndexError, ValueError, TypeError):
                    return None
            elif isinstance(val, dict):
                val = val.get(sub) if sub in val else val.get(str(sub))
        return val

    if name == "min":
        return min(args) if args else None
    if name == "max":
        return max(args) if args else None
    if name == "sum":
        return sum(a for a in args if isinstance(a, (int, float))) if args else 0
    if name == "abs":
        try:
            return abs(args[0]) if args else 0
        except (TypeError, ValueError):
            return 0

    if name == "poker_score":
        # 6.Q — Poker hand evaluation as an expression.
        #   $poker_score($actor.hole_cards, $state.community_cards)
        # Returns an integer where higher = stronger hand. Use in
        # CONDITIONAL effects to determine showdown winners.
        from .phase_handlers import _score_hand as _ps
        hole = args[0] if args and isinstance(args[0], list) else []
        comm = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        try:
            return _ps(hole, comm)
        except Exception:
            return 0

    if name == "len":
        try:
            return len(args[0]) if args else 0
        except (TypeError, ValueError):
            return 0

    # ── Collection / aggregation helpers ────────────────────────────
    if name == "avg":
        nums = [a for a in args if isinstance(a, (int, float))]
        return sum(nums) / len(nums) if nums else 0

    if name == "first":
        # Return first non-None element of a list or first non-None arg.
        if not args:
            return None
        if len(args) == 1 and isinstance(args[0], list):
            seq = args[0]
        else:
            seq = list(args)
        for v in seq:
            if v is not None:
                return v
        return None

    if name == "any":
        if not args:
            return False
        seq = args[0] if (len(args) == 1 and isinstance(args[0], list)) else list(args)
        return any(bool(v) for v in seq)

    if name == "all":
        if not args:
            return True
        seq = args[0] if (len(args) == 1 and isinstance(args[0], list)) else list(args)
        return all(bool(v) for v in seq)

    if name == "in":
        # $in(needle, [haystack...]) — membership test for expressions
        if len(args) < 2:
            return False
        needle = args[0]
        hay = args[1] if isinstance(args[1], list) else list(args[1:])
        return needle in hay

    if name == "if":
        # Inline ternary: $if(cond, then, else)
        if len(args) < 2:
            return None
        return args[1] if args[0] else (args[2] if len(args) >= 3 else None)

    # ── State queries (require `state` to be threaded) ──────────────
    if name == "entities_of":
        # $entities_of(EntityType) → list of entity ids
        if not args or state is None:
            return []
        et_name = str(args[0])
        return [e.id for e in state.entities.values() if e.entity_type == et_name]

    if name == "count":
        # $count(EntityType)             — count all
        # $count(EntityType, field_name) — count with truthy field
        if not args or state is None:
            return 0
        et_name = str(args[0])
        if len(args) == 1:
            return sum(1 for e in state.entities.values() if e.entity_type == et_name)
        field = str(args[1])
        cnt = 0
        for e in state.entities.values():
            if e.entity_type != et_name:
                continue
            # Special-case `alive` to read the dataclass attr first
            if field == "alive":
                if getattr(e, "alive", True):
                    cnt += 1
                continue
            if e.properties.get(field):
                cnt += 1
        return cnt

    if name == "alive_of":
        # $alive_of(EntityType) → list of alive entity ids
        if not args or state is None:
            return []
        et_name = str(args[0])
        return [
            e.id for e in state.entities.values()
            if e.entity_type == et_name and getattr(e, "alive", True)
        ]

    if name == "find_first":
        # $find_first(EntityType, field_name) → first entity id with truthy field
        if len(args) < 2 or state is None:
            return None
        et_name, field = str(args[0]), str(args[1])
        for e in state.entities.values():
            if e.entity_type != et_name:
                continue
            if (field == "alive" and getattr(e, "alive", True)) or e.properties.get(field):
                return e.id
        return None

    if name == "sum_of":
        # $sum_of(EntityType, property_name) → sum of property across entities
        if len(args) < 2 or state is None:
            return 0
        et_name, prop = str(args[0]), str(args[1])
        total = 0
        for e in state.entities.values():
            if e.entity_type != et_name:
                continue
            try:
                total += float(e.properties.get(prop, 0) or 0)
            except (TypeError, ValueError):
                continue
        return total

    if name == "max_of":
        if len(args) < 2 or state is None:
            return None
        et_name, prop = str(args[0]), str(args[1])
        vals = []
        for e in state.entities.values():
            if e.entity_type != et_name:
                continue
            try:
                vals.append(float(e.properties.get(prop, 0) or 0))
            except (TypeError, ValueError):
                continue
        return max(vals) if vals else None

    if name == "min_of":
        if len(args) < 2 or state is None:
            return None
        et_name, prop = str(args[0]), str(args[1])
        vals = []
        for e in state.entities.values():
            if e.entity_type != et_name:
                continue
            try:
                vals.append(float(e.properties.get(prop, 0) or 0))
            except (TypeError, ValueError):
                continue
        return min(vals) if vals else None

    # ── Relation-graph neighborhood queries ──
    # The missing primitive for network dynamics: without these, "infect
    # me if a graph NEIGHBOR is infected" was inexpressible — relations
    # were readable only as an actor→target boolean.
    def _eid(v: Any) -> str:
        # Accept an entity object ($params.it in a for_each rule) or an id.
        return str(getattr(v, "id", v))

    if name == "neighbors":
        # $neighbors(entity_id, relation) → neighbor ids (either direction)
        if len(args) < 2 or state is None:
            return []
        eid, rel = _eid(args[0]), str(args[1])
        out = {e.to_entity for e in state.relations.get_outgoing(eid, rel)}
        out |= {e.from_entity for e in state.relations.get_incoming(eid, rel)}
        out.discard(eid)
        return sorted(out)

    if name in ("neighbor_count", "neighbor_sum"):
        # $neighbor_count(entity_id, relation)        — degree
        # $neighbor_count(entity_id, relation, prop)  — neighbors w/ truthy prop
        # $neighbor_sum(entity_id, relation, prop)    — sum of prop
        if len(args) < 2 or state is None:
            return 0
        eid, rel = _eid(args[0]), str(args[1])
        prop = str(args[2]) if len(args) > 2 else None
        ids = {e.to_entity for e in state.relations.get_outgoing(eid, rel)}
        ids |= {e.from_entity for e in state.relations.get_incoming(eid, rel)}
        ids.discard(eid)
        if name == "neighbor_count":
            if prop is None:
                return len(ids)
            return sum(
                1 for nid in ids
                if (ent := state.entities.get(nid)) is not None
                and ent.properties.get(prop)
            )
        if prop is None:
            return 0
        total = 0.0
        for nid in ids:
            ent = state.entities.get(nid)
            if ent is None:
                continue
            try:
                total += float(ent.properties.get(prop, 0) or 0)
            except (TypeError, ValueError):
                continue
        return total

    # ── Spatial queries (require state.locations / state.adjacency) ──
    if name == "entities_at":
        # $entities_at(location_id) → list of entity ids at that location
        if not args or state is None:
            return []
        loc = str(args[0])
        return [eid for eid, l in state.locations.items() if l == loc]

    if name == "adjacent_entities":
        # $adjacent_entities(location_id) → entities at any adjacent location
        if not args or state is None:
            return []
        loc = str(args[0])
        neighbors = state.adjacency.get(loc, [])
        out: List[str] = []
        for n in neighbors:
            out.extend(eid for eid, l in state.locations.items() if l == n)
        return out

    if name == "within_range":
        # $within_range(location_id, N) — BFS over adjacency for entities
        # within N hops. Hop 0 = same location.
        if len(args) < 2 or state is None:
            return []
        try:
            origin, hops = str(args[0]), int(args[1])
        except (TypeError, ValueError):
            return []
        visited = {origin}
        frontier = [origin]
        for _ in range(max(0, hops)):
            new_frontier = []
            for node in frontier:
                for nb in state.adjacency.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        new_frontier.append(nb)
            frontier = new_frontier
            if not frontier:
                break
        return [eid for eid, l in state.locations.items() if l in visited]

    if name == "distance":
        # $distance(loc_a, loc_b) — BFS shortest-path in adjacency graph.
        # Returns -1 if unreachable.
        if len(args) < 2 or state is None:
            return -1
        a, b = str(args[0]), str(args[1])
        if a == b:
            return 0
        seen = {a}
        frontier = [(a, 0)]
        while frontier:
            node, d = frontier.pop(0)
            for nb in state.adjacency.get(node, []):
                if nb == b:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    frontier.append((nb, d + 1))
        return -1

    if name == "path_exists":
        # $path_exists(loc_a, loc_b) — boolean
        if len(args) < 2 or state is None:
            return False
        a, b = str(args[0]), str(args[1])
        if a == b:
            return True
        seen = {a}
        frontier = [a]
        while frontier:
            node = frontier.pop(0)
            for nb in state.adjacency.get(node, []):
                if nb == b:
                    return True
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        return False

    # ── Board pattern queries (delegate to BoardModule) ─────────────
    # Together with the existing board move primitives in board_module,
    # these are the foundation of the spatial-game pattern DSL.
    # They make chess-class games expressible in pure JSON: the agent
    # references piece patterns by name and queries threat/check status
    # via expressions.

    if name in ("legal_moves", "is_threatened", "is_in_check",
                "is_in_checkmate", "line_clear", "piece_at", "square_empty"):
        board = _get_board_module(state)
        if board is None:
            # No board configured — return safe defaults
            if name in ("is_threatened", "is_in_check", "is_in_checkmate"):
                return False
            if name == "line_clear" or name == "square_empty":
                return True
            if name == "piece_at":
                return None
            if name == "legal_moves":
                return []

        if name == "legal_moves":
            # $legal_moves(piece_type, from_square, [side])
            if len(args) < 2:
                return []
            piece_type = str(args[0])
            from_sq = board._parse_position(args[1])
            side = str(args[2]) if len(args) >= 3 else None
            if from_sq is None:
                return []
            try:
                moves = board.valid_moves(piece_type, from_sq, side, state)
                return [list(m) if isinstance(m, tuple) else m for m in moves]
            except Exception:
                return []

        if name == "is_threatened":
            # $is_threatened(square, by_side)
            if len(args) < 2:
                return False
            sq = board._parse_position(args[0])
            by_side = str(args[1])
            if sq is None:
                return False
            try:
                return bool(board.is_square_attacked(sq, by_side, state))
            except Exception:
                return False

        if name == "is_in_check":
            # $is_in_check(side)
            if not args:
                return False
            side = str(args[0])
            try:
                return bool(board.is_in_check(side, state))
            except Exception:
                return False

        if name == "is_in_checkmate":
            # $is_in_checkmate(side)
            if not args:
                return False
            side = str(args[0])
            try:
                return bool(board.is_in_checkmate(side, state))
            except Exception:
                return False

        if name == "line_clear":
            # $line_clear(from_square, to_square)
            if len(args) < 2:
                return True
            a = board._parse_position(args[0])
            b = board._parse_position(args[1])
            if a is None or b is None:
                return False
            try:
                return bool(board.line_clear(a, b, state))
            except Exception:
                return False

        if name == "piece_at":
            # $piece_at(square) → mark/piece string or None
            if not args:
                return None
            sq = board._parse_position(args[0])
            if sq is None:
                return None
            # Grid boards use mark_at(); linear-ring boards use cell_at()
            if hasattr(board, "mark_at"):
                try:
                    val = board.mark_at(sq)
                    if val is not None and val != "":
                        return val
                except Exception:
                    pass
            if hasattr(board, "cell_at"):
                cell = board.cell_at(sq)
                if isinstance(cell, dict):
                    return cell.get("piece") or cell.get("mark")
                return cell
            return None

        if name == "square_empty":
            # $square_empty(square)
            if not args:
                return True
            sq = board._parse_position(args[0])
            if sq is None:
                return False
            if hasattr(board, "mark_at"):
                try:
                    val = board.mark_at(sq)
                    return val is None or val == "" or val == 0
                except Exception:
                    pass
            if hasattr(board, "cell_at"):
                cell = board.cell_at(sq)
                if cell is None:
                    return True
                if isinstance(cell, dict):
                    return not (cell.get("piece") or cell.get("mark"))
                return not cell
            return True

    # ── Graph algorithm primitives (general spatial / network games) ──

    if name == "shortest_path":
        # $shortest_path(from, to) — return list of nodes from→to inclusive,
        # or [] if no path. Uses state.adjacency.
        if len(args) < 2 or state is None:
            return []
        src, dst = str(args[0]), str(args[1])
        if src == dst:
            return [src]
        # BFS with parent pointers
        parents = {src: None}
        frontier = [src]
        found = False
        while frontier and not found:
            new_frontier = []
            for node in frontier:
                for nb in state.adjacency.get(node, []):
                    if nb in parents:
                        continue
                    parents[nb] = node
                    if nb == dst:
                        found = True
                        break
                    new_frontier.append(nb)
                if found:
                    break
            frontier = new_frontier
        if not found:
            return []
        # Reconstruct
        path = [dst]
        cur = dst
        while parents[cur] is not None:
            cur = parents[cur]
            path.append(cur)
        return list(reversed(path))

    if name == "connected_component":
        # $connected_component(node) — all nodes reachable from `node`
        # via state.adjacency. Useful for Catan-style "longest road"
        # via successive calls + $len.
        if not args or state is None:
            return []
        node = str(args[0])
        seen = {node}
        frontier = [node]
        while frontier:
            current = frontier.pop(0)
            for nb in state.adjacency.get(current, []):
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        return list(seen)

    return None


def _get_board_module(state: Any):
    """Find the (first) BoardModule on the state. Returns None when no
    board is configured. Used by the board-pattern query functions."""
    if state is None:
        return None
    dm = getattr(state, "domain_modules", None)
    if dm is None:
        return None
    modules = getattr(dm, "_modules", None) or {}
    try:
        from .board_module import BoardModule
    except Exception:
        return None
    for m in modules.values():
        if isinstance(m, BoardModule):
            return m
    return None


# Deterministic fallback RNG. Seeded with 0 so any caller that fails
# to thread the engine RNG still produces reproducible output. Production
# callers MUST pass the engine's own seeded Random instance.
_DEFAULT_RNG = _random.Random(0)


__all__ = [
    "is_expression",
    "resolve_expression",
    "get_tokens",
    "grant_token",
    "consume_token",
    "has_token",
]
