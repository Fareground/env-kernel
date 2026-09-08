"""compile() — the single entry point for env-builder pipelines.

The agent (or any caller) hands raw JSON to ``compile(raw)`` and gets
back a structured ``CompileResult``:

  - On success: ``ok == True``, ``state`` + ``engine`` ready to run
  - On failure: ``errors`` lists what went wrong with JSON-pointer
    paths, severities, and fix hints — actionable by the agent

The pipeline:

    raw_json → Pydantic validate → lint_template → load_world → engine

Each stage emits ``CompileIssue`` records that aggregate into the
result. Errors short-circuit (later stages don't run if earlier ones
already failed); warnings are advisory and don't block the build.

## Why a single entry point

Before: the agent's edit/save loop went through several services
which surfaced errors as Pydantic stack traces. This made iteration
expensive — the agent burned context understanding what was wrong.

After: one call returns a structured report. The agent can read
``result.errors[0].hint`` and patch its JSON deterministically.

## Usage

    from fg_env_kernel import compile_template

    result = compile_template(raw_json)
    if not result.ok:
        for issue in result.errors:
            print(f"[{issue.severity}] {issue.path}: {issue.message}")
            if issue.hint:
                print(f"  hint: {issue.hint}")
        return
    result.engine.run()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import ValidationError

from ..engine import SimulationEngine
from ..state import WorldState
from .loader import WorldTemplate, load_world


@dataclass
class CompileIssue:
    """One problem found during compilation.

    Severity:
      - ``"error"``   — blocks the build (returned in result.errors)
      - ``"warning"`` — advisory, build still proceeds (result.warnings)

    Path:
      - JSON-pointer-like dotted path into the template
        (e.g. ``"actions[2].preconditions[0].expr"``)

    Hint:
      - Optional one-line suggested fix
    """
    severity: str
    path: str
    message: str
    hint: str = ""

    def __str__(self) -> str:
        head = f"[{self.severity}] {self.path or '<root>'}: {self.message}"
        return head + (f"\n  hint: {self.hint}" if self.hint else "")


@dataclass
class CompileResult:
    """Outcome of compiling a template dict."""
    template: Optional[WorldTemplate] = None
    state: Optional[WorldState] = None
    engine: Optional[SimulationEngine] = None
    errors: List[CompileIssue] = field(default_factory=list)
    warnings: List[CompileIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.engine is not None

    def report(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.ok:
            lines.append("compile: OK")
        else:
            lines.append(f"compile: FAILED ({len(self.errors)} error(s))")
        for issue in self.errors:
            lines.append(str(issue))
        for issue in self.warnings:
            lines.append(str(issue))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def compile_template(
    raw: Dict[str, Any],
    *,
    seed: int = 0,
    decision_fn: Optional[Callable] = None,
    on_event: Optional[Callable] = None,
    skip_lint: bool = False,
    smoke: bool = False,
    smoke_rounds: int = 20,
) -> CompileResult:
    """Compile a raw JSON template into a runnable engine.

    Pipeline:
      0. ``upgrade_template`` — apply any registered migrations to
         lift the template to the current ``CONTRACT_VERSION``
      1. Pydantic ``WorldTemplate.model_validate`` — structural checks
      2. ``lint_template`` — static logic checks (skippable)
      3. ``load_world`` — build state + engine
      4. ``smoke_test`` — optional mocked playtest (``smoke=True``)

    Any errors at any stage short-circuit the pipeline. Warnings are
    collected and surfaced but don't block the build.

    When ``smoke=True``, a throwaway engine (NOT ``result.engine``) is run
    for ``smoke_rounds`` with mocked random decisions. A *crash* during the
    playtest is a real kernel-level defect and surfaces as an **error**
    (blocks ``result.ok``); softer findings ("no actions were ever taken",
    "hit max_rounds without terminating") surface as **warnings** — they're
    often legitimate for open-ended worlds, so they don't block the build.

    Returns a ``CompileResult``. Inspect ``result.ok`` to know whether
    ``result.engine`` is ready to run.
    """
    result = CompileResult()

    # ---- Stage 0: Pre-Pydantic shape check + version upgrade --------
    if not isinstance(raw, dict):
        result.errors.append(CompileIssue(
            severity="error",
            path="",
            message=f"template must be a dict, got {type(raw).__name__}",
            hint="Wrap your template in `{...}` JSON object syntax.",
        ))
        return result

    # Apply version migrations BEFORE Pydantic validation. Legacy
    # templates (no contract_version, or older versions) get lifted
    # transparently to the current shape.
    try:
        from .versioning import upgrade_template
        raw = upgrade_template(raw)
    except Exception as e:  # pragma: no cover
        result.warnings.append(CompileIssue(
            severity="warning",
            path="contract_version",
            message=f"version upgrade failed: {e}",
            hint="Template will be validated as-is.",
        ))

    # ---- Stage 1: Pydantic structural validation --------------------
    try:
        template = WorldTemplate.model_validate(raw)
    except ValidationError as e:
        for err in e.errors():
            result.errors.append(CompileIssue(
                severity="error",
                path=_jsonpath_from_loc(err.get("loc", ())),
                message=err.get("msg", "validation error"),
                hint=_hint_for_validation_error(err),
            ))
        return result

    result.template = template

    # ---- Stage 2: Static lint ---------------------------------------
    if not skip_lint:
        from ..lint import lint_template
        issues = lint_template(template)
        for iss in issues:
            if iss.severity == "error":
                result.errors.append(iss)
            else:
                result.warnings.append(iss)
        if result.errors:
            return result

    # ---- Stage 3: Build engine --------------------------------------
    try:
        state, engine = load_world(
            template,
            seed=seed,
            decision_fn=decision_fn,
            on_event=on_event,
        )
    except Exception as e:
        result.errors.append(CompileIssue(
            severity="error",
            path="",
            message=f"engine construction failed: {type(e).__name__}: {e}",
            hint=(
                "Most often this means a referenced kernel resource "
                "(DomainModule, ResolutionArchetype, …) is not registered. "
                "Call kernel_capabilities() to verify the names you used."
            ),
        ))
        return result

    result.state = state
    result.engine = engine

    # ---- Stage 4: Optional smoke playtest ---------------------------
    # Runs on a SEPARATE throwaway engine so result.engine stays pristine
    # for the caller's real run.
    if smoke:
        try:
            from ..smoke import smoke_test
            probe_state, probe_engine = load_world(template, seed=seed)
            report = smoke_test(probe_engine, rounds=smoke_rounds, seed=seed)
            if report.crash:
                result.errors.append(CompileIssue(
                    severity="error",
                    path="",
                    message=f"smoke playtest crashed: {report.crash}",
                    hint=(
                        "The world builds but crashes when run. This is a "
                        "kernel-level defect in the world definition — check "
                        "the failing action's effects/preconditions."
                    ),
                ))
            for effect in report.invalid_effects:
                result.errors.append(CompileIssue(
                    severity="error", path="",
                    message=(f"smoke: invalid {effect.get('operation')} effect on "
                             f"{effect.get('target')}.{effect.get('field')}: {effect.get('detail')}"),
                    hint="Fix the value expression and its referenced data before running this world.",
                ))
            for w in report.warnings:
                result.warnings.append(CompileIssue(
                    severity="warning", path="", message=f"smoke: {w}",
                ))
        except Exception as e:  # pragma: no cover — smoke must never hard-fail compile
            result.warnings.append(CompileIssue(
                severity="warning",
                path="",
                message=f"smoke playtest could not run: {type(e).__name__}: {e}",
            ))

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonpath_from_loc(loc: tuple) -> str:
    """Convert a Pydantic error `loc` tuple into a dotted/indexed path.

    Pydantic emits tuples like ``("actions", 2, "preconditions", 0,
    "operator")``. We render that as ``actions[2].preconditions[0].operator``
    — the convention an agent or human can paste into a JSON path tool."""
    parts: List[str] = []
    for item in loc:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
        else:
            parts.append(str(item))
    return ".".join(parts)


def _hint_for_validation_error(err: Mapping[str, Any]) -> str:
    """Try to suggest a fix for common Pydantic errors."""
    msg = (err.get("msg") or "").lower()
    loc = err.get("loc") or ()
    field = loc[-1] if loc else ""

    if "field required" in msg:
        return f"Add the missing '{field}' field."
    if "extra inputs" in msg or "extra fields" in msg:
        # Models use extra="allow" so this is rare; still helpful.
        return f"Field '{field}' isn't recognized; check spelling."
    if "input should be" in msg:
        return f"Wrong type for '{field}'. See the WorldTemplate spec."
    if "value is not a valid" in msg:
        return f"Value for '{field}' has the wrong shape."
    return ""


__all__ = [
    "CompileIssue",
    "CompileResult",
    "compile_template",
]
