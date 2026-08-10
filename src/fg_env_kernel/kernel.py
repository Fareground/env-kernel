"""SDK facade — the two-object entry point for integrators.

    from fg_env_kernel import Kernel

    kernel = Kernel(seed=42)
    world = kernel.load(template_dict, decision_fn=my_agent)
    world.run()                # or: while not world.finished: world.step()
    print(world.terminated_by, world.events[-1].narrative)

``Kernel`` holds run configuration (seed, registry); ``World`` wraps the
``(WorldState, SimulationEngine)`` pair produced by the canonical
``pipeline.loader.load_world`` and delegates to the engine — it adds no
behavior of its own. Power users can keep using ``load_world`` directly.
"""
from __future__ import annotations

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
        registry: primitive registry to resolve effects / terminations /
                  modules against. Defaults to the process-global
                  registry; a custom ``KernelRegistry`` is accepted now
                  so callers can prepare for per-kernel isolation, but
                  the loader currently resolves against the global
                  registry — isolated resolution is a planned follow-up.
    """

    def __init__(self, seed: int = 0, registry: Optional[KernelRegistry] = None):
        self.seed = seed
        self.registry = registry if registry is not None else _global_registry

    def load(
        self,
        template: Union[Dict[str, Any], "WorldTemplate"],
        *,
        decision_fn: Optional[DecisionFn] = None,
        on_event: Optional[OnEventFn] = None,
        seed: Optional[int] = None,
        max_rounds: Optional[int] = None,
    ) -> World:
        """Build a runnable ``World`` from a template dict (or a
        pre-validated ``WorldTemplate``).

        The loader honors the template's ``temporal.max_rounds``; an
        explicit ``max_rounds`` argument overrides it.
        """
        from .pipeline.loader import load_world

        state, engine = load_world(
            template,
            seed=self.seed if seed is None else seed,
            decision_fn=decision_fn,
            on_event=on_event,
        )
        if max_rounds is not None:
            engine.max_rounds = int(max_rounds)
        return World(state, engine)
