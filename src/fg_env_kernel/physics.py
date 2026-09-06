"""Continuous coupled-dynamics — the "code is physics" layer.

While agents (LLMs) make discrete turn-based decisions, the *world* between
those decisions does not have to sit frozen. This module advances numeric state
forward in time by integrating a system of **coupled** ordinary differential
equations (ODEs):

    d(var)/dt = f(all vars, params, t)

Each variable's rate may reference every other variable, so genuinely coupled
systems — predator/prey (Lotka–Volterra), epidemics (SIR), supply/demand price
discovery, resource depletion — are first-class, not a bag of independent
drifts (which is what :mod:`property_dynamics` provides).

Design goals:

* **dt-aware.** The same model integrates by a real time delta. In discrete
  mode that delta is one round (dt=1, ticked once per round before agents act);
  in continuous mode it is the gap between successive environment ticks
  (``ContinuousTemporalModel.environment_interval``). Either way the *clock*, not
  a round counter, drives evolution — and agent turns interleave freely between
  ticks without advancing physics themselves.
* **Accurate & stable.** Classic 4th-order Runge–Kutta with configurable
  sub-stepping, so a large dt doesn't blow up a stiff system.
* **Turn-based compatible.** Agents still act in discrete turns; physics simply
  runs *between* turns and hands the agent the evolved world. Nothing here calls
  an LLM.
* **Bound to the world.** A variable can be a free global scalar, or read an
  aggregate of an entity property (sum/avg/…), and/or write its value back onto
  entities (broadcast/distribute). That is how a `price` field becomes every
  trader's observed price, or an `infected` count drives per-agent state.
* **Deterministic & serializable.** No hidden RNG; full ``to_dict``/``from_dict``
  for replay and fork.

The expression sub-language is a *safe* numeric evaluator (whitelisted AST), not
``eval`` — it cannot touch attributes, names, or builtins outside the math
namespace.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Safe numeric expression evaluator
# ---------------------------------------------------------------------------

#: Math functions exposed to rate expressions. Pure, side-effect-free.
_FUNCS: Dict[str, Any] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs, "min": min, "max": max, "pow": pow,
    "floor": math.floor, "ceil": math.ceil,
    "tanh": math.tanh, "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    "sign": lambda x: (x > 0) - (x < 0),
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
}

#: Constants available by name.
_CONSTS: Dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

#: AST node types permitted in a rate expression. Anything else is rejected at
#: compile time — no attribute access, no comprehensions, no lambdas, no names
#: outside the supplied namespace.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.BoolOp, ast.And, ast.Or,
)


class PhysicsExprError(ValueError):
    """A rate expression is malformed or uses a forbidden construct."""


class _CompiledExpr:
    """A rate expression compiled once and evaluated against a namespace.

    Comparisons and boolean ops evaluate to 1.0/0.0 so they compose into
    rates (e.g. ``(S > 0) * beta * S * I`` gates a term on positivity)."""

    __slots__ = ("source", "_code", "_names")

    def __init__(self, source: str):
        self.source = source
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:  # noqa: TRY003
            raise PhysicsExprError(f"cannot parse rate expr {source!r}: {exc}") from exc
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise PhysicsExprError(
                    f"forbidden construct {type(node).__name__} in rate expr {source!r}"
                )
            if isinstance(node, ast.Name):
                names.add(node.id)
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise PhysicsExprError(f"only direct function calls allowed in {source!r}")
            # Only numeric literals — reject strings/bytes (string-repetition DoS)
            # and other constant types. bool is an int subclass and is allowed.
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise PhysicsExprError(
                    f"non-numeric constant {node.value!r} in rate expr {source!r}"
                )
        self._names = names
        self._code = compile(tree, "<physics-rate>", "eval")

    def eval(self, namespace: Mapping[str, Any]) -> float:
        # Booleans (from Compare/BoolOp) collapse to 1.0/0.0 so they're usable
        # as numeric multipliers/terms.
        val = eval(self._code, {"__builtins__": {}}, namespace)  # noqa: S307 — AST-whitelisted
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        return float(val)


# ---------------------------------------------------------------------------
# Entity binding — read aggregates / write values back to the entity graph
# ---------------------------------------------------------------------------

_REDUCERS = {
    "sum": lambda xs: float(sum(xs)),
    "avg": lambda xs: float(sum(xs) / len(xs)) if xs else 0.0,
    "mean": lambda xs: float(sum(xs) / len(xs)) if xs else 0.0,
    "min": lambda xs: float(min(xs)) if xs else 0.0,
    "max": lambda xs: float(max(xs)) if xs else 0.0,
    "count": lambda xs: float(len(xs)),
}


@dataclass
class EntitySource:
    """Read an aggregate of an entity property into a physics variable.

    e.g. ``{entity_type: 'firm', property: 'inventory', reduce: 'sum'}`` makes a
    variable track total inventory across all live firms each step."""
    entity_type: str
    property: str
    reduce: str = "sum"

    def read(self, state: Any) -> float:
        ents = [e for e in state.get_entities_by_type(self.entity_type) if getattr(e, "alive", True)]
        if self.reduce == "count":
            return float(len(ents))
        # A NaN/inf property is a float and would pass a bare isinstance
        # filter, then poison the whole integration step (every derivative
        # eval goes non-finite and the isfinite guard freezes physics with no
        # surfaced cause). Drop non-finite reads at the source instead.
        vals = [
            float(raw)
            for e in ents
            for raw in (e.get(self.property, 0.0),)
            if isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(raw)
        ]
        return _REDUCERS.get(self.reduce, _REDUCERS["sum"])(vals)

    def to_dict(self) -> dict:
        return {"entity_type": self.entity_type, "property": self.property, "reduce": self.reduce}

    @classmethod
    def from_dict(cls, d: dict) -> "EntitySource":
        return cls(entity_type=d["entity_type"], property=d["property"], reduce=d.get("reduce", "sum"))


@dataclass
class EntityWriteback:
    """Write a physics variable's value back onto entities.

    * ``broadcast`` — set every matching entity's property to the variable value
      (e.g. a single market ``price`` observed by all traders).
    * ``distribute`` — split the value equally across matching entities.
    """
    entity_type: str
    property: str
    mode: str = "broadcast"

    def write(self, state: Any, value: float) -> List[Dict[str, Any]]:
        ents = [e for e in state.get_entities_by_type(self.entity_type) if getattr(e, "alive", True)]
        if not ents:
            return []
        per = value / len(ents) if self.mode == "distribute" else value
        changes: List[Dict[str, Any]] = []
        for e in ents:
            old = e.get(self.property, 0.0)
            e.set(self.property, per)
            changes.append({"entity_id": e.id, "field": self.property, "old": old, "new": per})
        return changes

    def to_dict(self) -> dict:
        return {"entity_type": self.entity_type, "property": self.property, "mode": self.mode}

    @classmethod
    def from_dict(cls, d: dict) -> "EntityWriteback":
        return cls(entity_type=d["entity_type"], property=d["property"], mode=d.get("mode", "broadcast"))


# ---------------------------------------------------------------------------
# Variables & equations
# ---------------------------------------------------------------------------

@dataclass
class PhysicsVariable:
    """One continuous state variable in the system.

    A variable with a ``rate`` is *integrated* (its ODE). A variable with only a
    ``source`` is *algebraic* — refreshed from the entity graph each step and
    available to other equations (read-only). ``min``/``max`` clamp the value
    after each integration step (e.g. a population or price can't go negative).
    """
    name: str
    value: float = 0.0
    rate: Optional[str] = None          # d(value)/dt expression; None = not integrated
    min: Optional[float] = None
    max: Optional[float] = None
    source: Optional[EntitySource] = None
    writeback: Optional[EntityWriteback] = None

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {"name": self.name, "value": self.value}
        if self.rate is not None:
            d["rate"] = self.rate
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.source is not None:
            d["source"] = self.source.to_dict()
        if self.writeback is not None:
            d["writeback"] = self.writeback.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicsVariable":
        return cls(
            name=d["name"],
            value=float(d.get("value", 0.0)),
            rate=d.get("rate"),
            min=d.get("min"),
            max=d.get("max"),
            source=EntitySource.from_dict(d["source"]) if d.get("source") else None,
            writeback=EntityWriteback.from_dict(d["writeback"]) if d.get("writeback") else None,
        )


class PhysicsModel:
    """A coupled-ODE system advanced by RK4 with sub-stepping.

    Usage::

        m = PhysicsModel.from_dict({
            "params": {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4},
            "variables": [
                {"name": "prey", "value": 10, "rate": "alpha*prey - beta*prey*pred", "min": 0},
                {"name": "pred", "value": 5,  "rate": "delta*prey*pred - gamma*pred", "min": 0},
            ],
            "substeps": 8,
        })
        m.integrate(dt=1.0)            # advance the world one time unit
        m.values  # -> {"prey": ..., "pred": ...}

    Bound to entities (a market price every trader sees)::

        {"name": "price", "value": 100, "rate": "k*(demand - supply)",
         "writeback": {"entity_type": "trader", "property": "observed_price"}}
    """

    # Ceiling on RK4 sub-steps per integrate() call — a config-DoS guard
    # (see __init__). 4096 is far above any physically-motivated value.
    MAX_SUBSTEPS = 4096

    def __init__(
        self,
        variables: Optional[List[PhysicsVariable]] = None,
        params: Optional[Dict[str, float]] = None,
        substeps: int = 4,
        time: float = 0.0,
    ):
        self.variables: Dict[str, PhysicsVariable] = {v.name: v for v in (variables or [])}
        self.params: Dict[str, float] = dict(params or {})
        # Clamp substeps to a sane ceiling: each integrate() call loops
        # `substeps` times doing four dict-allocating derivative evals, so an
        # unbounded config value (`substeps: 100000000`) is a per-tick DoS.
        self.substeps: int = max(1, min(int(substeps), self.MAX_SUBSTEPS))
        self.time: float = float(time)
        # Compile each integrated variable's rate expression once.
        self._compiled: Dict[str, _CompiledExpr] = {
            name: _CompiledExpr(v.rate)
            for name, v in self.variables.items()
            if v.rate is not None
        }
        self._validate_names()
        # A variable can be integrated (rate) OR algebraically bound to the
        # entity graph (source), not both — a source refresh each step would
        # silently clobber the integrated value. Fail loud at build time.
        for name, v in self.variables.items():
            if v.rate is not None and v.source is not None:
                raise PhysicsExprError(
                    f"variable {name!r} has both a rate and a source — a source is "
                    "read-only (algebraic); drop one."
                )
            for label, value in (("value", v.value), ("min", v.min), ("max", v.max)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise PhysicsExprError(f"variable {name!r} {label} must be finite")
            if v.source is None and ((v.min is not None and v.value < v.min) or (v.max is not None and v.value > v.max)):
                raise PhysicsExprError(f"variable {name!r} initial value {v.value} is outside its bounds ({v.min}, {v.max})")
            if v.min is not None and v.max is not None and v.min > v.max:
                raise PhysicsExprError(
                    f"variable {name!r} has min ({v.min}) > max ({v.max})"
                )

    # -- introspection ----------------------------------------------------

    @property
    def values(self) -> Dict[str, float]:
        """Current value of every variable, keyed by name."""
        return {name: v.value for name, v in self.variables.items()}

    @property
    def integrated(self) -> List[str]:
        """Names of variables that have an ODE (are integrated)."""
        return [n for n, v in self.variables.items() if v.rate is not None]

    def is_empty(self) -> bool:
        return not self.variables

    def _validate_names(self) -> None:
        """Every name referenced by a rate must resolve to a variable, param,
        constant, the time symbol, or a whitelisted function. Fail loud at build
        time, not silently to zero at run time."""
        known = set(self.variables) | set(self.params) | set(_CONSTS) | set(_FUNCS) | {"t"}
        for name, expr in self._compiled.items():
            unknown = expr._names - known
            if unknown:
                raise PhysicsExprError(
                    f"rate for {name!r} references unknown symbol(s): {sorted(unknown)}"
                )

    # -- integration ------------------------------------------------------

    def _namespace(self, values: Mapping[str, float], t: float) -> Dict[str, Any]:
        ns: Dict[str, Any] = {}
        ns.update(_FUNCS)
        ns.update(_CONSTS)
        ns.update(self.params)
        ns.update(values)
        ns["t"] = t
        return ns

    def _derivatives(self, values: Mapping[str, float], t: float) -> Dict[str, float]:
        """Evaluate every ODE's right-hand side at variable values ``values`` and
        time ``t``.

        ``values`` carries the (possibly RK4-perturbed) *integrated* variables;
        algebraic / source-bound variables are read at their current value so
        coupled rates can reference them. Integrated perturbations win."""
        full: Dict[str, float] = {n: v.value for n, v in self.variables.items()}
        full.update(values)
        ns = self._namespace(full, t)
        return {name: expr.eval(ns) for name, expr in self._compiled.items()}

    def integrate(self, dt: float, state: Any = None) -> List[Dict[str, Any]]:
        """Advance the system forward by ``dt`` time units.

        If ``state`` is given, source-bound variables are first refreshed from
        the entity graph, and writeback variables are pushed back afterwards.
        Returns a list of change dicts (for event emission). ``dt <= 0`` is a
        no-op for integration (time never runs backwards here) but still syncs
        source/writeback bindings.

        Numeric blow-ups (division by zero, math-domain errors, overflow, or a
        non-finite result) do NOT crash the simulation: the step is skipped, the
        variables keep their prior values, and a ``physics_error`` change is
        returned so the condition is visible. Physics is an enhancement layer —
        a bad equation degrades gracefully rather than killing the run."""
        # Always refresh algebraic source bindings so reads are current.
        if state is not None:
            self._refresh_sources(state)

        if dt <= 0 or not self._compiled:
            return self._apply_writebacks(state) if state is not None else []

        before = {n: self.variables[n].value for n in self._compiled}

        # RK4 with sub-stepping: split dt into `substeps` slices for stability.
        # k-stages are evaluated at the correct intermediate TIMES (t, t+h/2,
        # t+h) — required for time-dependent rates like sin(t).
        h = dt / self.substeps
        y = dict(before)
        t = self.time
        try:
            for _ in range(self.substeps):
                k1 = self._derivatives(y, t)
                k2 = self._derivatives({n: y[n] + 0.5 * h * k1[n] for n in y}, t + 0.5 * h)
                k3 = self._derivatives({n: y[n] + 0.5 * h * k2[n] for n in y}, t + 0.5 * h)
                k4 = self._derivatives({n: y[n] + h * k3[n] for n in y}, t + h)
                y = {n: y[n] + (h / 6.0) * (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]) for n in y}
                t += h
        except (ArithmeticError, ValueError) as exc:
            return [{
                "type": "physics_error", "error": str(exc), "dt": dt,
                "narrative": f"Physics step skipped (numeric error): {exc}",
            }]
        if any(not math.isfinite(v) for v in y.values()):
            return [{
                "type": "physics_error", "error": "non-finite result", "dt": dt,
                "narrative": "Physics step skipped: a variable became non-finite.",
            }]

        self.time = self.time + dt  # commit time exactly (no float drift)

        changes: List[Dict[str, Any]] = []
        for n in self._compiled:
            var = self.variables[n]
            new_val = y[n]
            if var.min is not None:
                new_val = max(var.min, new_val)
            if var.max is not None:
                new_val = min(var.max, new_val)
            if abs(new_val - before[n]) > 1e-12:
                changes.append({
                    "type": "physics_step",
                    "variable": n,
                    "old_value": round(before[n], 6),
                    "new_value": round(new_val, 6),
                    "dt": dt,
                    "narrative": (
                        f"{n} {'rose' if new_val > before[n] else 'fell'} "
                        f"from {before[n]:.3g} to {new_val:.3g}"
                    ),
                })
            var.value = new_val

        if state is not None:
            changes.extend(self._apply_writebacks(state))
        return changes

    def _refresh_sources(self, state: Any) -> None:
        for v in self.variables.values():
            if v.source is not None:
                v.value = v.source.read(state)

    def _apply_writebacks(self, state: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for v in self.variables.values():
            if v.writeback is not None:
                wrote = v.writeback.write(state, v.value)
                if wrote:
                    out.append({
                        "type": "physics_writeback",
                        "variable": v.name,
                        "field": v.writeback.property,
                        "entity_type": v.writeback.entity_type,
                        "value": round(v.value, 6),
                        "count": len(wrote),
                        "narrative": (
                            f"{v.writeback.entity_type}.{v.writeback.property} "
                            f"set to {v.value:.3g} ({v.name})"
                        ),
                    })
        return out

    # -- tick (engine-facing, mirrors PropertyDynamicsEngine.tick) --------

    def tick(self, state: Any, dt: float) -> List[Dict[str, Any]]:
        """Engine entry point: integrate ``dt`` against ``state`` and return
        change events. Mirrors :meth:`PropertyDynamicsEngine.tick` so the engine
        treats both dynamics layers uniformly."""
        return self.integrate(dt, state=state)

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "variables": [v.to_dict() for v in self.variables.values()],
            "params": dict(self.params),
            "substeps": self.substeps,
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PhysicsModel":
        return cls(
            variables=[PhysicsVariable.from_dict(v) for v in (d.get("variables") or [])],
            params={k: float(val) for k, val in (d.get("params") or {}).items()},
            substeps=int(d.get("substeps", 4)),
            time=float(d.get("time", 0.0)),
        )

    @classmethod
    def from_schema(cls, spec: Optional[Mapping[str, Any]]) -> Optional["PhysicsModel"]:
        """Build from a world-definition ``physics`` block, or None if absent.

        Schema shape::

            "physics": {
                "params": {"r": 0.5, "K": 1000},
                "substeps": 8,
                "variables": [
                    {"name": "population", "value": 10,
                     "rate": "r*population*(1 - population/K)", "min": 0,
                     "writeback": {"entity_type": "herd", "property": "size"}}
                ]
            }
        """
        if not spec or not (spec.get("variables")):
            return None
        return cls.from_dict(spec)


__all__ = [
    "PhysicsModel",
    "PhysicsVariable",
    "EntitySource",
    "EntityWriteback",
    "PhysicsExprError",
]
