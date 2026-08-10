"""KernelRegistry — central plugin dispatch for the simulation kernel.

The registry is the foundation of the "configure ANY game" vision. Every
extension point in the kernel — effect operations, preconditions,
resolution archetypes, phase handlers, termination predicates, domain
modules, target selectors — is reachable by string name through this
single object.

A game author (or an env-builder agent) writes JSON schema/rules that
reference *names*; modules register *implementations*; the engine looks
up implementations by name at runtime. The engine itself imports nothing
from the domain layer.

## Namespaces

- ``effects``         — ``EffectHandler``: applies a single effect op
- ``preconditions``   — ``PreconditionFn``: returns bool given (state, actor, target, cond)
- ``resolutions``     — ``ResolutionArchetype``: outcome mechanic
- ``phases``          — ``PhaseHandler``: drives a phase tick
- ``terminations``    — ``TerminationCheck``: returns bool|winner-spec
- ``modules``         — ``KernelModule`` factory: builds a domain module from config
- ``target_selectors``— callable resolving a target expression to entity ids
- ``triggers``        — declarative triggers (handled via ``triggers.py``)

## Usage

    from fg_env_kernel.registry import registry, effect

    @effect("post_content")
    def _post_content(ctx, effect_spec):
        ...

    handler = registry.effects.get("post_content")

The registry is a process-wide singleton. For deterministic test
isolation use ``with registry.scoped(): ...`` which creates a child
registry that falls back to the global one for unknown names.
"""
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class _Namespace:
    """A single registry namespace (e.g. ``effects``). Holds string→object
    bindings. ``parent`` allows scoped overrides during tests."""

    __slots__ = ("_name", "_items", "_aliases", "_parent")

    def __init__(self, name: str, parent: Optional["_Namespace"] = None):
        self._name = name
        self._items: Dict[str, Any] = {}
        self._aliases: Dict[str, str] = {}  # alias → canonical
        self._parent = parent

    def register(
        self,
        key: str,
        value: Any,
        *,
        aliases: Optional[List[str]] = None,
        replace: bool = False,
    ) -> None:
        key = self._normalize(key)
        if not replace and key in self._items:
            raise ValueError(
                f"registry[{self._name}]: '{key}' already registered. "
                "Pass replace=True to override."
            )
        self._items[key] = value
        for alias in aliases or []:
            self._aliases[self._normalize(alias)] = key

    def get(self, key: str) -> Any:
        canonical = self._aliases.get(self._normalize(key), self._normalize(key))
        if canonical in self._items:
            return self._items[canonical]
        if self._parent is not None:
            return self._parent.get(key)
        raise KeyError(
            f"registry[{self._name}]: '{key}' not registered. "
            f"Known: {sorted(self.keys())}"
        )

    def try_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.get(key)
        except KeyError:
            return default

    def has(self, key: str) -> bool:
        canonical = self._aliases.get(self._normalize(key), self._normalize(key))
        if canonical in self._items:
            return True
        return self._parent.has(key) if self._parent else False

    def keys(self) -> List[str]:
        out = set(self._items.keys()) | set(self._aliases.keys())
        if self._parent is not None:
            out |= set(self._parent.keys())
        return sorted(out)

    def unregister(self, key: str) -> None:
        """Local-only removal. Does not affect parent."""
        self._items.pop(self._normalize(key), None)

    @staticmethod
    def _normalize(key: str) -> str:
        return str(key).strip().lower()

    def __contains__(self, key: str) -> bool:  # type: ignore[override]
        return self.has(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())


@dataclass
class KernelRegistry:
    """Process-wide registry of pluggable kernel pieces.

    All namespaces are public attributes — use them with their
    ``register`` / ``get`` / ``has`` API. The registry itself is
    intentionally lock-free for reads (CPython dict reads are GIL-safe);
    registrations happen at import time so contention is minimal.
    """
    effects: _Namespace = field(init=False)
    preconditions: _Namespace = field(init=False)
    resolutions: _Namespace = field(init=False)
    phases: _Namespace = field(init=False)
    terminations: _Namespace = field(init=False)
    modules: _Namespace = field(init=False)
    target_selectors: _Namespace = field(init=False)
    triggers: _Namespace = field(init=False)

    _parent: Optional["KernelRegistry"] = None

    def __post_init__(self) -> None:
        parents = (
            self._parent.effects if self._parent else None,
            self._parent.preconditions if self._parent else None,
            self._parent.resolutions if self._parent else None,
            self._parent.phases if self._parent else None,
            self._parent.terminations if self._parent else None,
            self._parent.modules if self._parent else None,
            self._parent.target_selectors if self._parent else None,
            self._parent.triggers if self._parent else None,
        )
        self.effects = _Namespace("effects", parents[0])
        self.preconditions = _Namespace("preconditions", parents[1])
        self.resolutions = _Namespace("resolutions", parents[2])
        self.phases = _Namespace("phases", parents[3])
        self.terminations = _Namespace("terminations", parents[4])
        self.modules = _Namespace("modules", parents[5])
        self.target_selectors = _Namespace("target_selectors", parents[6])
        self.triggers = _Namespace("triggers", parents[7])

    @contextlib.contextmanager
    def scoped(self) -> Iterator["KernelRegistry"]:
        """Yield a child registry that falls back to ``self`` for unknown
        keys. Useful in tests to register temp handlers without polluting
        global state."""
        yield self.fork()

    def fork(self) -> "KernelRegistry":
        """Return an isolated child registry.

        The child resolves unknown names through ``self`` (so built-in
        primitives stay visible — nothing is copied), while its own
        registrations stay local: they are invisible to the parent and
        to sibling forks. This is the isolation unit ``Kernel``
        consumes: ``Kernel(registry=registry.fork())``.
        """
        return KernelRegistry(_parent=self)

    # -- registry-bound decorators ------------------------------------------
    # Instance twins of the module-level @effect / @termination / ...
    # decorators: they register into THIS registry instead of the global
    # one, e.g. ``@my_registry.effect("my_op")``.

    def _decorator(self, namespace: _Namespace):
        def deco(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
            def _wrap(obj: Any) -> Any:
                with _REGISTRATION_LOCK:
                    namespace.register(name, obj, aliases=aliases, replace=replace)
                return obj
            return _wrap
        return deco

    def effect(self, name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
        """Decorator: register an effect handler in this registry."""
        return self._decorator(self.effects)(name, aliases=aliases, replace=replace)

    def precondition(self, name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
        return self._decorator(self.preconditions)(name, aliases=aliases, replace=replace)

    def resolution(self, name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
        return self._decorator(self.resolutions)(name, aliases=aliases, replace=replace)

    def phase(self, name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
        return self._decorator(self.phases)(name, aliases=aliases, replace=replace)

    def termination(self, name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
        return self._decorator(self.terminations)(name, aliases=aliases, replace=replace)

    def module(self, name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
        return self._decorator(self.modules)(name, aliases=aliases, replace=replace)

    def target_selector(self, name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
        return self._decorator(self.target_selectors)(name, aliases=aliases, replace=replace)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

registry = KernelRegistry()
"""The global registry. Domain modules register their effects /
preconditions / phases against this object at import time."""


_REGISTRATION_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Decorator helpers — sugar for the common case of registering a function
# ---------------------------------------------------------------------------

def effect(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
    """Decorator: register an effect handler under ``name`` in the
    GLOBAL registry. For per-kernel isolation use the instance twin:
    ``@my_registry.effect(name)``.

    The handler signature is::

        def handler(ctx: EffectContext, spec: dict) -> Optional[dict]
    """
    return registry.effect(name, aliases=aliases, replace=replace)


def precondition(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
    return registry.precondition(name, aliases=aliases, replace=replace)


def resolution(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
    return registry.resolution(name, aliases=aliases, replace=replace)


def phase(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
    return registry.phase(name, aliases=aliases, replace=replace)


def termination(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
    return registry.termination(name, aliases=aliases, replace=replace)


def module(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
    return registry.module(name, aliases=aliases, replace=replace)


def target_selector(name: str, *, aliases: Optional[List[str]] = None, replace: bool = False):
    return registry.target_selector(name, aliases=aliases, replace=replace)


__all__ = [
    "KernelRegistry",
    "registry",
    "effect",
    "precondition",
    "resolution",
    "phase",
    "termination",
    "module",
    "target_selector",
]
