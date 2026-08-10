"""SDK facade — the one-line and two-object entry points for integrators.

One line (built-in random agent, deterministic given the seed):

    from fg_env_kernel import simulate

    world = simulate(template)
    print(world.summary())

Two objects, full control:

    from fg_env_kernel import Kernel

    kernel = Kernel(seed=42)
    world = kernel.load(template_dict, decision_fn=my_agent)
    world.run()                # or: while not world.finished: world.step()
    print(world.terminated_by, world.events[-1].narrative)

``Kernel`` holds run configuration (seed, registry); ``World`` wraps the
``(WorldState, SimulationEngine)`` pair produced by the canonical
``pipeline.loader.load_world`` and delegates to the engine — it adds no
behavior of its own. Power users can keep using ``load_world`` directly.

Templates may be passed as a dict, a ``WorldTemplate``, or a str/Path to
a JSON file — ``simulate()`` and ``Kernel.load`` accept all three.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .pipeline.loader import WorldTemplate

from .action import ActionInstance
from .event import SimEvent
from .registry import KernelRegistry, registry as _global_registry
from .runtime.engine import SimulationEngine
from .state import WorldState

# The agent callback contract. Called once per agent turn:
#
#     decision_fn(entity_id, perception, valid_actions) -> ActionInstance | None
#
#   entity_id      — id of the agent whose turn it is.
#   perception     — dict of what the agent can see (visibility-filtered).
#                    Always present: "self" (own id/name/properties),
#                    "visible_entities", "visible_relations",
#                    "visible_resources", "round", "phase", "location",
#                    "faction". Present when the world provides them:
#                    "world_brief" (name/description/rules markdown),
#                    "incoming_messages", "your_recent_actions",
#                    "domain_data" (module-contributed sections), and
#                    others (roles, polls, time_context, trade_history).
#   valid_actions  — action names whose preconditions currently pass.
#
# Return an ``ActionInstance`` (``action_name`` must be one of
# ``valid_actions``; ``actor_id`` should be ``entity_id``), or ``None``
# to skip the turn.
DecisionFn = Callable[[str, Dict[str, Any], List[str]], Optional[ActionInstance]]

# Real-time event stream callback: called with each event dict as it is
# emitted (same payloads that accumulate in ``World.events``).
OnEventFn = Callable[[Dict[str, Any]], None]

# A template argument anywhere in the SDK: a raw dict, a pre-validated
# WorldTemplate, or a str/Path to a JSON file on disk.
TemplateLike = Union[Dict[str, Any], "WorldTemplate", str, "os.PathLike[str]"]


def _coerce_template(template: TemplateLike) -> Union[Dict[str, Any], "WorldTemplate"]:
    """Normalize a template argument to a dict/WorldTemplate.

    str/Path inputs are treated as a path to a JSON file and loaded,
    with friendly errors for missing files and invalid JSON.
    """
    if isinstance(template, (str, os.PathLike)):
        path = Path(template)
        if not path.exists():
            raise FileNotFoundError(
                f"Template file not found: {path} — pass a dict, a WorldTemplate, "
                f"or a path to an existing JSON template file."
            )
        try:
            text = path.read_text()
        except OSError as exc:
            raise OSError(f"Could not read template file {path}: {exc}") from exc
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Template file {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Template file {path} must contain a JSON object at the top "
                f"level, got {type(loaded).__name__}."
            )
        return loaded
    return template


class World:
    """A loaded, runnable world — thin typed wrapper over the engine.

    Construct via ``Kernel.load``; direct construction from an existing
    ``(WorldState, SimulationEngine)`` pair also works.
    """

    def __init__(self, state: WorldState, engine: SimulationEngine):
        self._state = state
        self.engine = engine

    # -- state & transcript -------------------------------------------------

    @property
    def state(self) -> WorldState:
        """The live world state (entities, resources, event log...)."""
        return self._state

    @property
    def events(self) -> List[SimEvent]:
        """All events emitted so far (copy of the append-only log)."""
        return self._state.event_log.get_all()

    @property
    def current_round(self) -> int:
        return self._state.temporal.current_round

    @property
    def terminated_by(self) -> Optional[str]:
        """Name of the termination condition that ended the sim, if any."""
        return self.engine.terminated_by

    @property
    def finished(self) -> bool:
        """True once the sim has ended (termination condition, stop(),
        or the round budget ran out)."""
        return self.engine.finished

    @property
    def seed(self) -> int:
        """The seed this run uses (readable even for unseeded runs)."""
        return self.engine.seed

    @property
    def name(self) -> str:
        """The template's world name ('' when the template omits it)."""
        brief = getattr(self._state, "_world_brief", None) or {}
        return str(brief.get("name") or "")

    # -- readable results ---------------------------------------------------

    def summary(self) -> str:
        """A small human-readable wrap-up of the run so far."""
        name = self.name or "world"
        status = "finished" if self.finished else "running"
        lines = [
            f"{name}: {status} after {self.current_round} round(s) "
            f"(budget {self.engine.max_rounds}).",
            f"Terminated by: {self.terminated_by or ('round budget' if self.finished else '—')}",
        ]
        events = self.events
        if events:
            lines.append(f"Final event:   {events[-1].narrative}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "finished" if self.finished else "running"
        return (
            f"<World {self.name or 'world'!r}: round "
            f"{self.current_round}/{self.engine.max_rounds}, {status}, "
            f"terminated_by={self.terminated_by!r}>"
        )

    # -- execution ----------------------------------------------------------

    def step(self) -> WorldState:
        """Advance exactly one round (discrete mode only). No-op once
        finished — check ``world.finished`` in your loop."""
        return self.engine.step()

    def run(self) -> WorldState:
        """Run to completion (termination condition or round budget)."""
        return self.engine.run()


class Kernel:
    """Entry point holding run configuration.

    Args:
        seed:     default RNG seed for worlds loaded by this kernel.
        registry: primitive registry that worlds loaded by this kernel
                  resolve custom effect ops and termination checks
                  against. Defaults to the process-global registry.
                  Build an isolated one with ``registry.fork()`` — the
                  fork sees all built-ins but keeps its own
                  registrations private to this kernel::

                      from fg_env_kernel import Kernel, registry

                      mine = registry.fork()

                      @mine.effect("my_op")
                      def _my_op(ctx, spec): ...

                      kernel = Kernel(seed=42, registry=mine)

                  Validation/reporting surfaces (``lint_template``,
                  ``export_kernel_contract``) still read the global
                  registry.
    """

    def __init__(self, seed: int = 0, registry: Optional[KernelRegistry] = None):
        self.seed = seed
        self.registry = registry if registry is not None else _global_registry

    def load(
        self,
        template: TemplateLike,
        *,
        decision_fn: Optional[DecisionFn] = None,
        on_event: Optional[OnEventFn] = None,
        seed: Optional[int] = None,
        max_rounds: Optional[int] = None,
    ) -> World:
        """Build a runnable ``World`` from a template dict, a
        pre-validated ``WorldTemplate``, or a str/Path to a JSON file.

        The loader honors the template's ``temporal.max_rounds``; an
        explicit ``max_rounds`` argument overrides it.
        """
        from .pipeline.loader import load_world

        state, engine = load_world(
            _coerce_template(template),
            seed=self.seed if seed is None else seed,
            decision_fn=decision_fn,
            on_event=on_event,
            registry=self.registry,
        )
        if max_rounds is not None:
            engine.max_rounds = int(max_rounds)
        return World(state, engine)


def simulate(
    template: TemplateLike,
    *,
    agent: Optional[DecisionFn] = None,
    seed: Optional[int] = None,
    max_rounds: Optional[int] = None,
    on_event: Optional[OnEventFn] = None,
    registry: Optional[KernelRegistry] = None,
) -> World:
    """One-shot simulation: load a template, run to completion, return
    the finished ``World``.

        from fg_env_kernel import simulate

        world = simulate(template)
        print(world.summary())

    Args:
        template:   template dict, ``WorldTemplate``, or str/Path to a
                    JSON template file.
        agent:      ``decision_fn(entity_id, perception, valid_actions)``
                    called for every agent turn. Defaults to the built-in
                    seeded random-valid-action policy (``random_policy``)
                    — deterministic given ``seed``, never touches global
                    random state.
        seed:       RNG seed for the run (engine + default agent).
                    Defaults to 0.
        max_rounds: overrides the template's ``temporal.max_rounds``.
        on_event:   callback receiving each event as it is emitted.
        registry:   ``KernelRegistry`` scoping custom primitives;
                    defaults to the process-global registry.
    """
    from .policies import random_policy

    effective_seed = 0 if seed is None else seed
    kernel = Kernel(seed=effective_seed, registry=registry)
    world = kernel.load(template, on_event=on_event, max_rounds=max_rounds)
    world.engine.decision_fn = (
        agent if agent is not None
        else random_policy(seed=effective_seed, state=world.state)
    )
    world.run()
    return world
