"""Strict values for state mutations; never turn invalid arithmetic into zero.

The predicate evaluator intentionally fails closed. Mutations instead need a
diagnostic: a missing price must not become a free (or one-credit) purchase.
This evaluator accepts the kernel's dollar references/functions and arithmetic,
including the builder's explicit ``{"expr": "..."}`` form. Python AST is only
used to parse operators; no Python code, calls, or attributes are executed.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from .effects import _call_function, _split_args_top_level, _walk_path


class EffectValueError(ValueError):
    """A supplied effect value cannot safely be applied."""


def number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EffectValueError(f"expected a finite number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise EffectValueError(f"expected a finite number, got {value!r}")
    return value


def expression_source(value: Any) -> str | None:
    if isinstance(value, dict) and "expr" in value:
        if set(value) != {"expr"} or not isinstance(value["expr"], str) or not value["expr"].strip():
            raise EffectValueError("an expression value must be exactly {'expr': '<expression>'}")
        return value["expr"].strip()
    if isinstance(value, str):
        src = value.strip()
        if src.startswith("$") or (src.startswith(("(", "+", "-")) and "$" in src):
            return src
    return None


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATH = re.compile(r"(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*")
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


def _reference(src: str, start: int) -> tuple[tuple[str, str | None, str], int]:
    match = _NAME.match(src, start + 1)
    if not match:
        raise EffectValueError("expected a reference name after '$'")
    name, end = match.group(), match.end()
    args = None
    if end < len(src) and src[end] == "(":
        begin = end + 1
        end, depth, quote = begin, 1, None
        while end < len(src) and depth:
            char = src[end]
            if quote:
                if char == "\\":
                    end += 2
                    continue
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            end += 1
        if depth or quote:
            raise EffectValueError(f"unclosed ${name}(...) expression")
        args = src[begin:end - 1]
    path_match = _PATH.match(src, end)
    assert path_match is not None
    path = path_match.group()
    return (name, args, path.lstrip(".")), path_match.end()


def _parse(src: str, depth: int = 0) -> tuple[ast.AST, dict[str, tuple[str, str | None, str]]]:
    if len(src) > 16_384 or depth > 64:
        raise EffectValueError("effect expression is too long or deeply nested")
    refs: dict[str, tuple[str, str | None, str]] = {}
    parts: list[str] = []
    i, quote = 0, None
    while i < len(src):
        char = src[i]
        if quote:
            parts.append(char)
            if char == "\\" and i + 1 < len(src):
                i += 1
                parts.append(src[i])
            elif char == quote:
                quote = None
            i += 1
            continue
        if char in "\"'":
            quote = char
        if char == "$":
            ref, i = _reference(src, i)
            key = f"__effect_ref_{len(refs)}"
            refs[key] = ref
            parts.append(key)
            continue
        parts.append(char)
        i += 1
    try:
        tree = ast.parse("".join(parts), mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise EffectValueError(f"invalid effect expression {src!r}") from exc
    # Validate the whole tree, even branches that will not be evaluated.
    allowed = (ast.Constant, ast.Name, ast.Load, ast.List, ast.Tuple,
               ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
               *tuple(_BINARY), *tuple(_COMPARE), ast.USub, ast.UAdd,
               ast.Not, ast.And, ast.Or)
    nodes = list(ast.walk(tree))
    if len(nodes) > 1024 or any(not isinstance(node, allowed) for node in nodes):
        raise EffectValueError("unsupported effect expression syntax")
    for _, args, _ in refs.values():
        if args is not None:
            for arg in _split_args_top_level(args):
                _parse(arg.strip(), depth + 1)
    return tree, refs


def validate_operand(value: Any, *, numeric: bool = False) -> None:
    """Validate explicit literals and expression syntax without reading state."""
    src = expression_source(value)
    if src is not None:
        _parse(src)
    elif numeric:
        number(value)


def resolve_value(value: Any, **context: Any) -> Any:
    src = expression_source(value)
    if src is None:
        return value
    try:
        return _evaluate(src, context)
    except EffectValueError:
        raise
    except (ArithmeticError, TypeError, ValueError, KeyError, RecursionError) as exc:
        raise EffectValueError(f"cannot evaluate {src!r}: {exc}") from exc


def _evaluate(src: str, context: dict[str, Any], *, allow_missing: bool = False) -> Any:
    tree, refs = _parse(src)

    def reference(ref: tuple[str, str | None, str]) -> Any:
        name, args_src, path = ref
        if args_src is None:
            if name not in context or context[name] is None:
                if allow_missing:
                    return None
                raise EffectValueError(f"unresolved reference ${name}")
            value = context[name]
        else:
            args = [part.strip() for part in _split_args_top_level(args_src)]
            if name == "if":
                if len(args) != 3:
                    raise EffectValueError("$if requires condition, true value, false value")
                condition = _evaluate(args[0], context)
                if not isinstance(condition, bool):
                    raise EffectValueError("$if condition must resolve to a boolean")
                value = _evaluate(args[1 if condition else 2], context)
            else:
                resolved = [_evaluate(arg, context, allow_missing=name == "first") for arg in args]
                if name == "entity":
                    if len(resolved) != 1 or context.get("state") is None:
                        raise EffectValueError("$entity requires an entity id and world state")
                    value = context["state"].get_entity(resolved[0])
                else:
                    # Numeric helpers must not silently discard invalid arguments.
                    if name in {"min", "max", "sum", "avg", "abs", "random", "random_float", "dice"}:
                        for arg in resolved:
                            number(arg)
                    value = _call_function(name, resolved, state=context.get("state"), rng=context.get("rng"))
        if path:
            if any(part.startswith("_") for part in path.split(".")):
                raise EffectValueError("private attribute paths are not effect values")
            value = _walk_path(value, path)
        if value is None and not allow_missing:
            raise EffectValueError(f"unresolved ${name}{'(...)' if args_src is not None else ''}{'.' + path if path else ''}")
        return value

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in refs:
                return reference(refs[node.id])
            return {"true": True, "false": False, "null": None}.get(node.id, node.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(item) for item in node.elts]
        if isinstance(node, ast.BinOp):
            left, right = number(walk(node.left)), number(walk(node.right))
            if isinstance(node.op, ast.Pow) and abs(right) > 1024:
                raise EffectValueError("effect exponent is too large")
            return number(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp):
            item = walk(node.operand)
            if isinstance(node.op, ast.Not):
                if not isinstance(item, bool):
                    raise EffectValueError("not requires a boolean")
                return not item
            return number(item) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                result = walk(item)
                if not isinstance(result, bool):
                    raise EffectValueError("boolean operators require booleans")
                if isinstance(node.op, ast.And) and not result:
                    return False
                if isinstance(node.op, ast.Or) and result:
                    return True
            return isinstance(node.op, ast.And)
        if isinstance(node, ast.Compare):
            left = walk(node.left)
            for op, rhs in zip(node.ops, node.comparators):
                right = walk(rhs)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        raise EffectValueError("unsupported effect value")

    return walk(tree)
