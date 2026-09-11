"""Validate scalar starting values without treating stored objects as code."""
from __future__ import annotations

import math
from typing import Any


INITIAL_VALUE_HINT = (
    "Use a correctly typed literal or the string '$lookup(runtime, parameter_name)' "
    "for a declared runtime input. Initial properties do not evaluate {'expr': ...} "
    "objects or bare '$parameter' references."
)


def initial_scalar_error(value: Any, kind: str, *, allow_binding: bool = False) -> str | None:
    # Unset properties are supported. Object/list payloads and prose are not code.
    if value is None or kind not in {"int", "float", "bool"}:
        return None
    if allow_binding and isinstance(value, str) and value.strip().startswith("$lookup("):
        return None
    if kind == "bool":
        return None if isinstance(value, bool) else "initial boolean property requires true or false"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"initial {kind} property requires a number, not {type(value).__name__}"
    if isinstance(value, float) and not math.isfinite(value):
        return "initial numeric property must be finite"
    if kind == "int" and not isinstance(value, int):
        return "initial integer property requires an integer literal"
    return None
