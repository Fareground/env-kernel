"""Unified predicate / boolean-expression evaluator.

This module supplies the SINGLE expression language used by:

  - Action preconditions    (was: Operator enum + Precondition struct)
  - Effect conditions       (was: EffectCondition struct with string operator)
  - Termination conditions  (was: TerminationCondition.check_type switch)
  - State-machine guards    (was: ad-hoc parsing in phase_state_machine)
  - Trigger predicates      (was: ad-hoc in triggers.py)

## Two equivalent forms

A predicate can be expressed as:

  (1) An **expression string**:

      "$actor.gold >= 100"
      "$actor.alive && $count(player, alive) > 1"
      "$actor.position == 0 || $actor.position == 39"

  (2) A **structured dict** (for tools that prefer JSON tree form):

      { "op": "gte",  "left": "$actor.gold", "right": 100 }
      { "op": "and",  "children": [ ..., ... ] }
      { "op": "not",  "child": { ... } }

Both forms compile to the same AST and evaluate identically.

## Supported operators

  Comparison:    ==  !=  <  <=  >  >=  in  not_in
  Boolean:       &&  ||  !   (or: and / or / not)
  Membership:    contains
  Existence:     defined?  (truthy and not None/0/'')
  Special:       has_resource(name, amount), at_location(loc),
                 is_alive, same_faction(target)

The whole right-hand side of any comparison is itself an expression,
so chained access like ``$actor.inventory.gold`` works everywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .effects import is_expression, resolve_expression

logger = logging.getLogger(__name__)

#: Predicates that raised during evaluation, logged once each (bounded) — a
#: silently fail-closed predicate is how a typo'd rule becomes an action that
#: never fires with no trace. Fail-closed stays; silence does not.
_LOGGED_FAILURES: set = set()
_MAX_LOGGED_FAILURES = 512


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    predicate: Any,
    *,
    actor: Any = None,
    target: Any = None,
    params: Optional[Dict[str, Any]] = None,
    state: Any = None,
    last_event: Optional[Dict[str, Any]] = None,
    result: Any = None,
    rng: Any = None,
) -> bool:
    """Evaluate a predicate (string OR dict) against the given context.

    Returns a strict bool. Unresolvable expressions are treated as
    ``False`` so misconfigured rules fail closed rather than firing
    unexpectedly. Any exception during evaluation is caught and logged
    to the result as False — predicates must never crash the engine."""
    ctx = _Ctx(actor=actor, target=target, params=params or {}, state=state,
               last_event=last_event or {}, result=result, rng=rng)
    try:
        if predicate is None:
            return False
        if isinstance(predicate, bool):
            return predicate
        if isinstance(predicate, dict):
            answer = _eval_dict(predicate, ctx)
            return answer and not ctx.unresolved
        if isinstance(predicate, str):
            answer = _eval_string(predicate, ctx)
            return answer and not ctx.unresolved
        return bool(predicate)
    except Exception as exc:
        # Fail closed — but VISIBLY. Each failing predicate is logged once per
        # process so a typo'd rule ("$actor.golde") surfaces in the logs
        # instead of silently never firing.
        key = repr(predicate)[:200]
        if key not in _LOGGED_FAILURES and len(_LOGGED_FAILURES) < _MAX_LOGGED_FAILURES:
            _LOGGED_FAILURES.add(key)
            logger.warning(
                "predicate failed and evaluated as False (fail-closed): %s — %s: %s",
                key, type(exc).__name__, exc,
            )
        return False


def resolve(
    value: Any,
    *,
    actor: Any = None,
    target: Any = None,
    params: Optional[Dict[str, Any]] = None,
    state: Any = None,
    last_event: Optional[Dict[str, Any]] = None,
    result: Any = None,
    rng: Any = None,
) -> Any:
    """Resolve any value through the expression language.

    - ``$…`` path/function expressions resolve via ``resolve_expression``
    - Strings containing operators (e.g. ``"$a + $b"``) are evaluated
      as an arithmetic/boolean expression
    - Everything else passes through unchanged
    """
    ctx = _Ctx(actor=actor, target=target, params=params or {}, state=state,
               last_event=last_event or {}, result=result, rng=rng)
    return _eval_value(value, ctx)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class _Ctx:
    actor: Any
    target: Any
    params: Dict[str, Any]
    state: Any
    last_event: Dict[str, Any]
    result: Any
    rng: Any
    # Missing data must not become True after negating a comparison or
    # combining it with arithmetic. Shared by every predicate entry point.
    unresolved: bool = False


# ---------------------------------------------------------------------------
# Dict form
# ---------------------------------------------------------------------------

_BOOL_OPS = {"and", "or", "&&", "||"}
_NOT_OPS = {"not", "!"}
_CMP_OPS = {"==", "eq", "!=", "neq", "ne", "<", "lt", "<=", "lte", ">", "gt", ">=", "gte",
            "in", "not_in", "contains"}
_SPECIAL_OPS = {"has_resource", "at_location", "is_adjacent", "is_alive",
                "same_faction", "same_org", "different_faction", "relation_gte",
                "has_item", "has_item_type", "inventory_not_full",
                "skill_gte", "has_recipe", "defined"}


def _eval_dict(node: Dict[str, Any], ctx: _Ctx) -> bool:
    op = str(node.get("op") or node.get("operator") or "").strip().lower()
    if not op:
        # Empty predicate is True (matches "no precondition" semantics)
        return True

    if op in _BOOL_OPS:
        children = node.get("children") or []
        if op in {"and", "&&"}:
            return all(_eval_dict(c, ctx) if isinstance(c, dict) else _eval_string(str(c), ctx)
                       for c in children)
        else:
            return any(_eval_dict(c, ctx) if isinstance(c, dict) else _eval_string(str(c), ctx)
                       for c in children)

    if op in _NOT_OPS:
        child = node.get("child")
        if isinstance(child, dict):
            # Fail CLOSED symmetry with the string `!` form: a child
            # comparison whose $-operand doesn't resolve fails closed to
            # False — negating that into True would fire rules on typos.
            if _dict_cmp_operand_unresolved(child, ctx):
                _log_unresolved(child)
                return False
            return not _eval_dict(child, ctx)
        return not _eval_string(str(child), ctx)

    if op in _CMP_OPS:
        left = _eval_value(node.get("left"), ctx)
        right = _eval_value(node.get("right"), ctx)
        return _cmp(op, left, right)

    if op in _SPECIAL_OPS:
        return _eval_special(op, node, ctx)

    return False


# ---------------------------------------------------------------------------
# String form  —  small Pratt-style expression parser
# ---------------------------------------------------------------------------

# Token kinds
_T_NUM, _T_STR, _T_EXPR, _T_IDENT, _T_OP, _T_LPAREN, _T_RPAREN, _T_LBRACK, _T_RBRACK, _T_COMMA, _T_LBRACE, _T_RBRACE = range(12)

# Tokens recognised by the lexer. Order matters — longer operators first.
_OPS_BY_LEN = [
    "&&", "||", "==", "!=", "<=", ">=", "**",
    "<", ">", "!", "+", "-", "*", "/", "%", "^",
]
_WORD_OPS = {"and", "or", "not", "in"}


def _tokenize(src: str) -> List[Tuple[int, Any]]:
    """Lex an expression string into tokens. ``$foo.bar`` and
    ``$func(args)`` are passed through as single EXPR tokens so the
    existing ``resolve_expression`` can handle them unchanged."""
    out: List[Tuple[int, Any]] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue

        # $-expression: greedily consume up to balanced parens or whitespace/punct
        if ch == "$":
            j = i + 1
            # identifier portion
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            # function call?  $name(...)
            if j < n and src[j] == "(":
                depth = 1
                j += 1
                while j < n and depth > 0:
                    if src[j] == "(":
                        depth += 1
                    elif src[j] == ")":
                        depth -= 1
                    j += 1
            # dotted path:  $foo.bar.baz[0].x
            while j < n and (src[j] == "." or src[j] == "[" or src[j] == "]"
                             or src[j].isalnum() or src[j] == "_"):
                j += 1
            out.append((_T_EXPR, src[i:j]))
            i = j
            continue

        # Number (incl. scientific notation: `1e-06` used to tokenize as
        # `1`, turning a one-in-a-million shock roll into a certainty)
        if ch.isdigit() or (ch == "-" and i + 1 < n and src[i + 1].isdigit()
                            and (not out or out[-1][0] in (_T_OP, _T_LPAREN, _T_LBRACK, _T_COMMA))):
            j = i + 1
            saw_dot = saw_exp = False
            while j < n:
                c = src[j]
                if c.isdigit():
                    j += 1
                elif c == "." and not saw_dot and not saw_exp:
                    saw_dot = True
                    j += 1
                elif c in "eE" and not saw_exp and j + 1 < n and (
                        src[j + 1].isdigit()
                        or (src[j + 1] in "+-" and j + 2 < n
                            and src[j + 2].isdigit())):
                    saw_exp = True
                    j += 2 if src[j + 1] in "+-" else 1
                else:
                    break
            num_str = src[i:j]
            num: Any
            try:
                num = float(num_str) if (saw_dot or saw_exp) else int(num_str)
            except ValueError:
                num = num_str
            out.append((_T_NUM, num))
            i = j
            continue

        # Quoted string
        if ch in ('"', "'"):
            quote = ch
            j = i + 1
            buf = []
            while j < n and src[j] != quote:
                if src[j] == "\\" and j + 1 < n:
                    buf.append(src[j + 1])
                    j += 2
                else:
                    buf.append(src[j])
                    j += 1
            out.append((_T_STR, "".join(buf)))
            i = j + 1  # skip closing quote
            continue

        if ch == "(":
            out.append((_T_LPAREN, "("))
            i += 1
            continue
        if ch == ")":
            out.append((_T_RPAREN, ")"))
            i += 1
            continue
        if ch == "[":
            out.append((_T_LBRACK, "["))
            i += 1
            continue
        if ch == "]":
            out.append((_T_RBRACK, "]"))
            i += 1
            continue
        if ch == ",":
            out.append((_T_COMMA, ","))
            i += 1
            continue
        if ch == "{":
            out.append((_T_LBRACE, "{"))
            i += 1
            continue
        if ch == "}":
            out.append((_T_RBRACE, "}"))
            i += 1
            continue

        # Multi-char operator?
        matched = False
        for op in _OPS_BY_LEN:
            if src.startswith(op, i):
                out.append((_T_OP, op))
                i += len(op)
                matched = True
                break
        if matched:
            continue

        # Identifier (true/false/null/word ops). Dots stay part of the
        # token: a bare `actor.trust` must reach parse_atom as ONE dotted
        # identifier (which fails closed there), never as `actor` + a
        # silently-dropped dot + `trust`.
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] in "._"):
                j += 1
            ident = src[i:j].rstrip(".")
            low = ident.lower()
            if low in _WORD_OPS:
                # Normalize word ops to their symbolic form
                mapped = {"and": "&&", "or": "||", "not": "!", "in": "in"}[low]
                out.append((_T_OP, mapped))
            else:
                out.append((_T_IDENT, ident))
            i = j
            continue

        # Unknown char — skip it
        i += 1
    return out


# Precedence (higher binds tighter)
_PRECEDENCE = {
    "||": 1, "&&": 2,
    "==": 3, "!=": 3, "<": 3, "<=": 3, ">": 3, ">=": 3, "in": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5, "%": 5,
    "**": 6, "^": 6,
}
_UNARY = {"!", "-"}


class _Parser:
    def __init__(self, tokens: List[Tuple[int, Any]]):
        self.toks = tokens
        self.pos = 0

    def peek(self) -> Optional[Tuple[int, Any]]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def take(self) -> Optional[Tuple[int, Any]]:
        if self.pos >= len(self.toks):
            return None
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        node = self.parse_expr(0)
        return node

    def parse_expr(self, min_prec: int):
        left = self.parse_atom()
        while True:
            tok = self.peek()
            if tok is None or tok[0] != _T_OP:
                break
            op = tok[1]
            prec = _PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break
            self.take()
            right = self.parse_expr(prec + 1)
            left = ("binop", op, left, right)
        return left

    def parse_atom(self):
        tok = self.take()
        if tok is None:
            return ("lit", None)
        kind, val = tok
        if kind == _T_OP and val in _UNARY:
            child = self.parse_atom()
            return ("unop", val, child)
        if kind == _T_NUM:
            return ("lit", val)
        if kind == _T_STR:
            return ("lit", val)
        if kind == _T_EXPR:
            return ("expr", val)
        if kind == _T_IDENT:
            low = val.lower()
            if low in ("true", "yes", "on"):
                return ("lit", True)
            if low in ("false", "no", "off"):
                return ("lit", False)
            if low in ("null", "none", "nil"):
                return ("lit", None)
            if "." in val:
                # A bare DOTTED identifier (`actor.trust`) is a mistyped
                # path, not an enum literal — the author forgot the `$`.
                # Evaluate as missing data (None → comparisons False),
                # never as a string that might win a coercion.
                return ("lit", None)
            return ("lit", val)
        if kind == _T_LPAREN:
            inner = self.parse_expr(0)
            # consume RPAREN if present
            tok2 = self.peek()
            if tok2 and tok2[0] == _T_RPAREN:
                self.take()
            return inner
        if kind == _T_LBRACK:
            # List literal — useful for `$x in [1, 2, 3]`
            items = []
            while True:
                tok2 = self.peek()
                if tok2 is None or tok2[0] == _T_RBRACK:
                    if tok2:
                        self.take()
                    break
                items.append(self.parse_expr(0))
                tok2 = self.peek()
                if tok2 and tok2[0] == _T_COMMA:
                    self.take()
            return ("list", items)
        return ("lit", val)


def _eval_string(src: str, ctx: _Ctx) -> bool:
    src = src.strip()
    if not src:
        return False
    # If the string has no boolean/comparison structure and no $-ref,
    # it's not a meaningful predicate — fail closed rather than
    # treating arbitrary text as truthy.
    if "$" not in src and not _has_comparison_op(src) and src.lower() not in {"true", "yes"}:
        return False
    val = _eval_string_value(src, ctx)
    # An expression like "$actor.foo" without a comparison yields the
    # raw value; truthy on missing data must be False.
    if val is None:
        return False
    if isinstance(val, str) and val.startswith("$"):
        # Unresolved expression — treat as missing
        return False
    return _truthy(val)


def _has_comparison_op(src: str) -> bool:
    """Quick check: does the string contain any boolean/comparison op?"""
    return any(op in src for op in ("==", "!=", "<=", ">=", "<", ">", "&&", "||", " and ", " or ", " in ", "!"))


def _eval_string_value(src: str, ctx: _Ctx) -> Any:
    src = src.strip()
    if not src:
        return None
    # Pure $-expression short-circuit (no operators) — let resolve_expression
    # handle it directly so existing behaviour is preserved bit-for-bit.
    if is_expression(src) and not _contains_operator(src):
        return _resolve_dollar(src, ctx)

    toks = _tokenize(src)
    if not toks:
        return None
    ast = _Parser(toks).parse()
    return _eval_ast(ast, ctx)


def _contains_operator(src: str) -> bool:
    """True if the source has top-level operators outside $ expressions."""
    depth = 0
    i = 0
    n = len(src)
    in_expr = False
    while i < n:
        ch = src[i]
        if ch == "$":
            in_expr = True
            i += 1
            continue
        if in_expr:
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth -= 1
                i += 1
                if depth <= 0:
                    in_expr = False
                continue
            if ch.isalnum() or ch in "_.[]":
                i += 1
                continue
            in_expr = False
            # fall through to operator check
        for op in _OPS_BY_LEN:
            if src.startswith(op, i):
                return True
        if src[i:i + 3].lower() == "and" or src[i:i + 2].lower() == "or" or src[i:i + 4].lower() == " in ":
            return True
        i += 1
    return False


def _eval_ast(node: Tuple, ctx: _Ctx) -> Any:
    if not node:
        return None
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "expr":
        return _resolve_dollar(node[1], ctx)
    if kind == "list":
        return [_eval_ast(item, ctx) for item in node[1]]
    if kind == "unop":
        _, op, child = node
        cv = _eval_ast(child, ctx)
        if op == "!":
            # Fail CLOSED on unresolvable refs: `!$actor.golde` must not
            # become True because the typo'd path resolved to nothing.
            if cv is None and child and child[0] == "expr":
                _log_unresolved(child[1])
                return False
            return not _truthy(cv)
        if op == "-":
            try:
                return -float(cv) if isinstance(cv, float) else -int(cv)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return 0
    if kind == "binop":
        _, op, l, r = node
        lv = _eval_ast(l, ctx)
        rv = _eval_ast(r, ctx)
        if op == "&&":
            return _truthy(lv) and _truthy(rv)
        if op == "||":
            return _truthy(lv) or _truthy(rv)
        if op in {"==", "!=", "<", "<=", ">", ">=", "in"}:
            return _cmp(op, lv, rv)
        if op in {"+", "-", "*", "/", "%", "**", "^"}:
            return _arith(op, lv, rv)
    return None


def _eval_value(value: Any, ctx: _Ctx) -> Any:
    if isinstance(value, str):
        if is_expression(value) and not _contains_operator(value):
            return _resolve_dollar(value, ctx)
        if _looks_like_expression(value):
            return _eval_string_value(value, ctx)
        return value
    if isinstance(value, list):
        return [_eval_value(v, ctx) for v in value]
    if isinstance(value, dict):
        # In comparison position, a dict on left/right is a nested predicate.
        return _eval_dict(value, ctx)
    return value


def _looks_like_expression(src: str) -> bool:
    """Heuristic: does this string contain $-refs or operators worth
    parsing? Plain strings short-circuit for speed."""
    return ("$" in src) or any(op in src for op in _OPS_BY_LEN)


def _resolve_dollar(src: str, ctx: _Ctx) -> Any:
    if src.startswith("$entity("):
        # Entity lookup plus a property path is already supported by effects.
        # The legacy resolver only recognizes calls ending in ')', so routing
        # '$entity(id).field' through it made documented action guards inert.
        from .effect_values import EffectValueError, resolve_value

        try:
            val = resolve_value(src, actor=ctx.actor, target=ctx.target, params=ctx.params,
                                state=ctx.state, last_event=ctx.last_event,
                                result=ctx.result, rng=ctx.rng)
        except EffectValueError:
            val = None
    else:
        val = resolve_expression(
            src,
            actor=ctx.actor, target=ctx.target, params=ctx.params,
            state=ctx.state, last_event=ctx.last_event, result=ctx.result,
            rng=ctx.rng,
        )
    # resolve_expression returns the original string when unresolvable —
    # for predicate purposes that's a missing value, not the literal
    # string. Map it back to None so comparisons fail closed.
    if isinstance(val, str) and val == src:
        ctx.unresolved = True
        return None
    if val is None:
        ctx.unresolved = True
    return val


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _dict_cmp_operand_unresolved(node: Dict[str, Any], ctx: "_Ctx") -> bool:
    """True when any comparison in this dict-form subtree has a $-expression
    operand that resolves to None (path doesn't exist) — the fail-closed
    trigger for `not`. Recurses through and/or/not so nesting can't launder
    a typo'd ref back into True."""
    op = node.get("op")
    if op in _BOOL_OPS:
        return any(
            _dict_cmp_operand_unresolved(c, ctx)
            for c in (node.get("children") or []) if isinstance(c, dict)
        )
    if op in _NOT_OPS:
        child = node.get("child")
        return isinstance(child, dict) and _dict_cmp_operand_unresolved(child, ctx)
    if op in _CMP_OPS:
        for side in ("left", "right"):
            raw = node.get(side)
            if isinstance(raw, str) and raw.startswith("$"):
                if _eval_value(raw, ctx) is None:
                    return True
    return False


def _log_unresolved(what: Any) -> None:
    """Once-per-process note that a negative predicate failed closed on an
    unresolvable operand — the silent inverse would fire rules on typos."""
    key = f"unresolved:{what!r}"[:200]
    if key not in _LOGGED_FAILURES and len(_LOGGED_FAILURES) < _MAX_LOGGED_FAILURES:
        _LOGGED_FAILURES.add(key)
        logger.warning(
            "negative predicate operand unresolved (%r) — failing closed", what
        )


def _cmp(op: str, l: Any, r: Any) -> bool:
    op = op.lower()
    if op in ("==", "eq"):
        return _coerce_eq(l, r)
    if op in ("!=", "neq", "ne"):
        # Negative comparison against an UNRESOLVED operand fails closed:
        # `$actor.golde != 0` is True for every typo under naive semantics,
        # which silently fires the rule forever.
        if l is None or r is None:
            if not (l is None and r is None):
                _log_unresolved((l, op, r))
                return False
        return not _coerce_eq(l, r)
    if op in ("<", "lt"):
        return _coerce_lt(l, r)
    if op in ("<=", "lte"):
        return _coerce_eq(l, r) or _coerce_lt(l, r)
    if op in (">", "gt"):
        return _coerce_lt(r, l)
    if op in (">=", "gte"):
        return _coerce_eq(l, r) or _coerce_lt(r, l)
    if op == "in":
        try:
            return l in r  # type: ignore[operator]
        except TypeError:
            return False
    if op == "not_in":
        if l is None or r is None:
            _log_unresolved((l, op, r))
            return False
        try:
            return l not in r  # type: ignore[operator]
        except TypeError:
            return False
    if op == "contains":
        try:
            return r in l  # type: ignore[operator]
        except TypeError:
            return False
    return False


def _arith(op: str, l: Any, r: Any) -> Any:
    try:
        a = float(l) if l is not None else 0.0
        b = float(r) if r is not None else 0.0
    except (TypeError, ValueError):
        return 0
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op in ("**", "^"):
        # Both spellings mean POWER: `^` used to fall through the parser
        # and evaluate as truthiness — a fail-OPEN gate inversion.
        try:
            return a ** b
        except (OverflowError, ZeroDivisionError):
            return 0
    if op == "/":
        return a / b if b != 0 else 0
    if op == "%":
        return a % b if b != 0 else 0
    return 0


def _coerce_eq(a: Any, b: Any) -> bool:
    if a == b:
        return True
    # Numeric coercion: "5" == 5
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _coerce_lt(a: Any, b: Any) -> bool:
    # If either side is None, comparisons are False (fail-closed for
    # missing data — predicates should not silently succeed on undefined
    # fields).
    if a is None or b is None:
        return False
    try:
        return float(a) < float(b)
    except (TypeError, ValueError):
        # Mixed numeric-vs-string NEVER compares lexicographically — that
        # made `99.0 < "actor.trust"` True, silently inverting a gate the
        # author merely misspelled (forgot the `$`). Ordering is only
        # meaningful between two strings.
        if isinstance(a, str) and isinstance(b, str):
            return a < b
        return False


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        # Common false-y string forms
        return v.strip().lower() not in ("", "false", "no", "0", "off", "null", "none")
    return bool(v)


# ---------------------------------------------------------------------------
# Special predicates  (legacy Operator enum compatibility shims)
# ---------------------------------------------------------------------------

def _eval_special(op: str, node: Dict[str, Any], ctx: _Ctx) -> bool:
    """Evaluate the kernel's domain-specific predicates that resist a
    pure expression form — has_resource, at_location, is_adjacent etc.

    These exist as escape hatches for behaviour that the path/operator
    grammar can't express concisely. Over time more of these can be
    replaced by ``$has_resource(name, amount)``-style functions."""
    state = ctx.state
    actor = ctx.actor
    target = ctx.target
    subject = (node.get("subject") or "actor").lower()
    ent = actor if subject == "actor" else target

    if op == "is_alive":
        if ent is None:
            return False
        # Properties override the dataclass attribute so tests/games can
        # toggle alive via `entity.set("alive", False)` without forcing
        # callers to mutate the attribute directly.
        if hasattr(ent, "get"):
            prop_val = ent.get("alive")
            if prop_val is not None:
                return bool(prop_val)
        return bool(getattr(ent, "alive", True))

    if op == "has_resource":
        name = node.get("field") or node.get("resource")
        amount = node.get("value", 0)
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = 0
        if not (state and name and ent):
            return False
        pool = state.resources.get(name)
        if pool is None:
            return False
        bal = pool.get(ent.id) if hasattr(pool, "get") else 0
        try:
            return float(bal) >= amount
        except (TypeError, ValueError):
            return False

    if op == "at_location":
        if not (state and ent):
            return False
        want = node.get("value")
        return state.locations.get(ent.id) == want

    if op == "is_adjacent":
        if not (state and actor and target):
            return False
        la = state.locations.get(actor.id)
        lb = state.locations.get(target.id)
        if la is None or lb is None:
            return False
        if la == lb:
            return True
        return lb in (state.adjacency.get(la) or [])

    if op == "same_faction":
        if not (state and actor and target):
            return False
        fa = state.factions.faction_of(actor.id) if hasattr(state.factions, "faction_of") else None
        fb = state.factions.faction_of(target.id) if hasattr(state.factions, "faction_of") else None
        return bool(fa) and fa == fb

    if op == "same_org":
        # Hierarchical: shared top-level ancestor. same_faction stays exact.
        if not (state and actor and target):
            return False
        mgr = state.factions
        if hasattr(mgr, "same_org"):
            return bool(mgr.same_org(actor.id, target.id))
        return False

    if op == "different_faction":
        if not (state and actor and target):
            return False
        fa = state.factions.faction_of(actor.id) if hasattr(state.factions, "faction_of") else None
        fb = state.factions.faction_of(target.id) if hasattr(state.factions, "faction_of") else None
        return bool(fa) and bool(fb) and fa != fb

    if op == "relation_gte":
        if not (state and actor and target):
            return False
        rel_type = node.get("relation_type") or "trust"
        try:
            threshold = float(node.get("value", 0))
        except (TypeError, ValueError):
            threshold = 0
        val = state.relations.get_value(actor.id, target.id, rel_type) if hasattr(state.relations, "get_value") else None
        if val is None:
            return False
        try:
            return float(val) >= threshold
        except (TypeError, ValueError):
            return False

    if op == "has_item":
        if not (state and ent):
            return False
        want_id = node.get("value")
        items = state.inventory.get_inventory(ent.id) if hasattr(state.inventory, "get_inventory") else []
        return any(getattr(it, "id", None) == want_id for it in items)

    if op == "has_item_type":
        if not (state and ent):
            return False
        want_type = node.get("value")
        items = state.inventory.get_inventory(ent.id) if hasattr(state.inventory, "get_inventory") else []
        return any(getattr(it, "item_type", None) == want_type for it in items)

    if op == "inventory_not_full":
        if not (state and ent):
            return False
        capacity = state.inventory.get_capacity(ent.id) if hasattr(state.inventory, "get_capacity") else None
        items = state.inventory.get_inventory(ent.id) if hasattr(state.inventory, "get_inventory") else []
        if capacity is None:
            return True
        return len(items) < capacity

    if op == "skill_gte":
        if not (state and ent):
            return False
        name = node.get("field")
        try:
            threshold = float(node.get("value", 0))
        except (TypeError, ValueError):
            threshold = 0
        if not (name and hasattr(state.skills, "get_level")):
            return False
        return state.skills.get_level(ent.id, name) >= threshold

    if op == "has_recipe":
        if not (state and ent and hasattr(state.recipes, "can_craft")):
            return False
        return state.recipes.can_craft(ent.id, node.get("value"), state)

    if op == "defined":
        # An explicit presence check is allowed to observe missing data;
        # present zero/False values are defined as well.
        was_unresolved = ctx.unresolved
        value = _eval_value(node.get("left", node.get("value")), ctx)
        ctx.unresolved = was_unresolved
        return value is not None

    return False


__all__ = ["evaluate", "resolve"]
