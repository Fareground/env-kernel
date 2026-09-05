"""
Simulation engine -- the tick loop that drives the world forward.

Responsibilities:
1. For each round, for each phase, determine turn order
2. For each agent's turn: build perception, call decision_fn, validate, resolve, apply effects
3. Emit events for the transcript
4. Process world events, status effects, action chains, and relation decay
5. Check termination conditions and world invariants
"""
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ..registry import KernelRegistry
    from ..triggers import TriggerSpec

from ..state import WorldState
from ..action import ActionInstance, Effect, EffectOperation
from ..resolution import ResolutionResult
from ..visibility import PerceptionBuilder, TrendAnalyzer
from ..temporal import TurnOrderResolver
from ..phase_handlers import get_phase_handler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effect helpers
# ---------------------------------------------------------------------------

def _is_multi_target(token: str) -> bool:
    """True if `token` is a multi-target EffectDSL target — `all`,
    `all_others`, `role:X`, `faction:Y`."""
    if not isinstance(token, str):
        return False
    if token in ("all", "all_others"):
        return True
    return token.startswith("role:") or token.startswith("faction:")


def _action_suppresses_chat(state: Any, action_name: str) -> bool:
    """Check whether any domain module servicing `action_name` opts out of
    public chat / speech leakage. Modules signal this by setting the
    class attribute `suppress_chat = True` (or returning True from a
    property of the same name).

    Used by the action resolver to scrub `speech` / `reasoning` from
    action_attempted / action_resolved event payloads and to skip the
    `agent_message` emission entirely — making the action's
    in-character text invisible to opponents. Wordle Duel uses this to
    prevent competitors from leaking their reasoning (e.g. "I'm trying
    CRANE to test C/R/N") and gifting their deductions to the other
    side.
    """
    modules = getattr(state, "domain_modules", None)
    if not modules:
        return False
    mod_dict = getattr(modules, "_modules", None) or {}
    for module in mod_dict.values():
        custom = getattr(module, "custom_actions", []) or []
        if action_name in custom and getattr(module, "suppress_chat", False):
            return True
    return False


def _coerce_effects(raw: Any, registry: Optional["KernelRegistry"] = None) -> List["Effect"]:
    """Turn schema-style effect dicts into Effect dataclass instances.

    Used by features that store effects in JSON (deck cards, phase-
    state-machine transitions, ad-hoc world events). The shape mirrors
    `effects_on_success`:
      { operation: "add", target: "actor", field: "money", value: 50 }
    """
    from ..action import Effect
    out: List[Effect] = []
    if not raw:
        return out
    if not isinstance(raw, list):
        raw = [raw]
    for d in raw:
        if isinstance(d, Effect):
            out.append(d)
            continue
        if not isinstance(d, dict):
            continue
        op_raw = d.get("operation") or d.get("op")
        if not op_raw:
            continue
        op: Any
        if isinstance(op_raw, EffectOperation):
            op = op_raw
        else:
            try:
                op = EffectOperation(op_raw)
            except ValueError:
                # Not a built-in op — check the plugin registry. If
                # registered, keep the raw string as the operation;
                # _apply_effects will route it via registry dispatch.
                if registry is None:
                    from ..registry import registry
                if registry.effects.has(str(op_raw)):
                    op = str(op_raw).lower()
                else:
                    continue
        out.append(Effect(
            target=str(d.get("target", "actor")),
            operation=op,
            field=d.get("field"),
            value=d.get("value"),
            resource=d.get("resource"),
            relation_type=d.get("relation_type"),
            description=str(d.get("description", "")),
            scale_by_magnitude=bool(d.get("scale_by_magnitude", False)),
        ))
    return out


def _resolve_cell_for_board(board_mod, raw):
    """Translate a user-friendly cell value into a board position.

    Grid boards accept:
      - 1-based linear index  ("1"..rows*cols, top-left → bottom-right)
      - 0-based linear index  (0..rows*cols-1)
      - "row,col" string       ("0,2")
      - [row, col] list/tuple
    Linear-ring boards accept:
      - int (mod spaces)

    Returns None on parse failure.
    """
    if raw is None:
        return None
    kind = getattr(board_mod, "_kind", None)
    if kind == "linear_ring":
        try:
            n = int(raw)
            return n % board_mod._spaces
        except (TypeError, ValueError):
            return None
    # Grid
    rows, cols = board_mod._rows, board_mod._cols
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str) and "," in raw:
        try:
            r, c = raw.split(",", 1)
            return (int(r.strip()), int(c.strip()))
        except ValueError:
            return None
    # Numeric — try 1-based first (more agent-friendly), fall back to 0-based
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return None
    if 1 <= idx <= rows * cols:
        idx0 = idx - 1
        return (idx0 // cols, idx0 % cols)
    if 0 <= idx < rows * cols:
        return (idx // cols, idx % cols)
    return None


# ---------------------------------------------------------------------------
# Termination Conditions
# ---------------------------------------------------------------------------

@dataclass
class TerminationCondition:
    """A condition that ends the simulation early when met."""
    name: str
    description: str = ""
    check_type: str = "all_dead"  # all_dead, resource_exhausted, rounds_idle, property_threshold,
                                  # all_goals_complete, event_triggered, compound_and, compound_or
    params: dict = field(default_factory=dict)
    sub_conditions: List['TerminationCondition'] = field(default_factory=list)  # For compound types


class SimulationEngine:
    """
    The core simulation loop.

    The engine is deterministic given the same random seed and LLM responses.
    LLMs are injected as a decision_fn callback.
    """

    def __init__(
        self,
        state: WorldState,
        decision_fn: Optional[Callable] = None,
        outcome_fn: Optional[Callable] = None,
        narrative_fn: Optional[Callable] = None,
        max_rounds: int = 100,
        seed: Optional[int] = None,
        on_round_start: Optional[Callable] = None,
        on_round_end: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
        termination_conditions: Optional[List[TerminationCondition]] = None,
        invariant_checker: Optional[Any] = None,
        world_event_engine: Optional[Any] = None,
        continuous_time: Optional[Any] = None,
        parallel_decisions: int = 0,
        emit_state_snapshots: bool = False,
        registry: Optional["KernelRegistry"] = None,
    ):
        self.state = state
        # Primitive registry this engine resolves custom effect ops and
        # termination checks against. None = process-global registry.
        from ..registry import registry as _global_registry
        self.registry: "KernelRegistry" = registry if registry is not None else _global_registry
        # Tier 5a — Triggered effects loaded from schema. Engine
        # evaluates these on every _emit_event call.
        from ..triggers import TriggerEngine
        self.triggers = TriggerEngine.from_schema(
            getattr(state, "_schema_triggers", None) or [],
        )
        # Thread-local cascade depth — parallel-phase threads each get
        # their own counter so trigger fan-out doesn't race.
        import threading as _threading
        self._cascade_tls = _threading.local()
        self.decision_fn = decision_fn    # fn(entity_id, perception, valid_actions) -> ActionInstance
        self.parallel_decisions = parallel_decisions  # 0 = sequential, N = max concurrent LLM calls
        self.outcome_fn = outcome_fn      # fn(entity_id, action_name, success, narrative, details) -> None
        self.narrative_fn = narrative_fn   # fn(actor, target, action_def, action_instance, result, state_changes) -> str
        self.max_rounds = max_rounds
        self._running = False
        self._paused = False
        self._stopped = False
        # step() and run() share these so mixing them never double-emits
        # the simulation_start / simulation_end bracket events.
        self._start_emitted = False
        self._end_emitted = False
        self._perception_builder = PerceptionBuilder()
        self._trend_analyzer = TrendAnalyzer(max_snapshots=5)
        self.on_round_start = on_round_start
        self.on_round_end = on_round_end
        self.on_event = on_event          # fn(event_dict) -> None  -- real-time event streaming
        # Emit a state_snapshot event after every action_resolved /
        # phase_handler — the contract every live visualization reads
        # (entity properties + resources per round). Off by default: it
        # multiplies event volume, so only viewers that render state ask.
        self.emit_state_snapshots = emit_state_snapshots
        self.termination_conditions = termination_conditions or []
        self.invariant_checker = invariant_checker  # InvariantChecker instance (or None)
        self.world_event_engine = world_event_engine  # WorldEventEngine instance (or None)
        self._continuous_time = continuous_time       # ContinuousTemporalModel instance (or None)
        self.terminated_by: Optional[str] = None  # Name of condition that ended the sim
        # Every run is reproducible, seeded or not. When no seed is given we
        # mint one from system entropy and RECORD it as `self.seed` rather than
        # letting `random.Random()` swallow it — an unseeded run still varies
        # run-to-run, but the seed it used is readable afterwards, so any run
        # (including a surprising one) can be replayed exactly by passing
        # `seed=engine.seed` back in. The docstring's determinism promise held
        # only for the explicit-seed path before; now it holds for both.
        self.seed: int = seed if seed is not None else random.SystemRandom().getrandbits(64)
        self._rng = random.Random(self.seed)

        # Single source of randomness. Subsystems that own a private RNG
        # (property dynamics, world events) must share THIS one, or their
        # draws are decoupled from the seed and the sim stops being
        # reproducible. We deliberately do NOT touch the global `random`
        # module — seeding it would (a) not help, since nothing in the
        # kernel reads global random, and (b) clobber global state shared
        # by in-process batch/fork runs.
        if self.world_event_engine is not None:
            self.world_event_engine.rng = self._rng
        pd = getattr(self.state, "property_dynamics", None)
        if pd is not None:
            pd.rng = self._rng
        # Domain modules that own an RNG (card decks, per-player hands) expose
        # `reseed(rng)` to pin their shuffles to the sim seed. Without this a
        # card game deals differently on every same-seed run.
        dm = getattr(self.state, "domain_modules", None)
        if dm is not None:
            for mod in getattr(dm, "_modules", {}).values():
                reseed = getattr(mod, "reseed", None)
                if callable(reseed):
                    reseed(self._rng)

    @property
    def finished(self) -> bool:
        """True once the sim has ended — a termination condition fired,
        ``stop()`` was called, or the round budget ran out."""
        return (
            self.terminated_by is not None
            or self._stopped
            or self.state.temporal.current_round >= self.max_rounds
        )

    def step(self) -> WorldState:
        """Advance the simulation by exactly one discrete round.

        Emits ``simulation_start`` on the first call and
        ``simulation_end`` after the final round (termination condition,
        ``stop()``, or the round budget). Calling again after the end is
        a no-op. Continuous-time sims are event-driven and have no round
        granularity — use ``run()`` for those.
        """
        if self._continuous_time is not None:
            raise RuntimeError(
                "step() supports discrete (turn-based) mode only; "
                "continuous-time simulations must use run()"
            )
        if self._end_emitted or self.finished:
            return self.state
        self._running = True
        self._emit_start()
        self.state.temporal.advance_round()
        self._run_round()
        if (self.finished or not self._running) and not self._paused:
            self._emit_end()
        return self.state

    def _emit_start(self) -> None:
        """Emit simulation_start exactly once per engine lifetime."""
        if self._start_emitted:
            return
        self._start_emitted = True
        agents = self.state.get_agent_entities()
        self._emit_event(
            "simulation_start",
            narrative=f"Simulation begins. {len(agents)} agents active.",
        )

    def _emit_end(self) -> None:
        """Emit simulation_end exactly once and stop the loop."""
        if not self._end_emitted:
            self._end_emitted = True
            self._emit_event(
                "simulation_end",
                narrative=f"Simulation ended after {self.state.temporal.current_round} rounds.",
            )
        self._running = False

    def run(self) -> WorldState:
        """Run the full simulation. Returns final state.

        If the state's current_round > 0 (e.g. restored from snapshot),
        the engine continues from that round rather than resetting.
        If paused, returns early with state intact for later resume.

        Dispatches to _run_discrete() or _run_continuous() based on
        the temporal model's mode.
        """
        self._running = True
        self._paused = False

        # Check if continuous time model is configured
        if self._continuous_time is not None:
            return self._run_continuous()

        return self._run_discrete()

    def _run_discrete(self) -> WorldState:
        """Run the simulation in discrete (turn-based) mode.

        Resumable: honors rounds already played via step() (or a restored
        snapshot) — it emits simulation_start only if step() hasn't, runs
        only the REMAINING round budget, and no-ops if already ended.
        """
        if self._end_emitted:
            self._running = False
            return self.state
        self._emit_start()

        remaining = self.max_rounds - self.state.temporal.current_round
        for _ in range(max(0, remaining)):
            if not self._running or self._paused:
                break
            self.state.temporal.advance_round()
            self._run_round()
            if self._paused:
                break

        if not self._paused:
            self._emit_end()
        return self.state

    def _run_continuous(self) -> WorldState:
        """Run the simulation in continuous (event-driven) mode.

        Pops events from the priority queue, advances time, and processes
        each event. Agent turns are rescheduled after their action duration.
        """
        ct = self._continuous_time
        agents = self.state.get_agent_entities()

        # Only bootstrap on a FRESH start. On resume (pause→resume), the
        # event queue already holds the in-flight schedule — re-initializing
        # would double-schedule turns and re-emit the start event.
        if ct.is_empty():
            self._emit_event(
                "simulation_start",
                narrative=f"Continuous simulation begins. {len(agents)} agents active.",
            )
            ct.initialize_agents([a.id for a in agents], stagger=0.1)
            # Kick off the recurring environment clock so physics / property
            # dynamics / world events actually advance between agent turns.
            ct.schedule_environment_tick()
            # Anchor the physics dt clock at the fresh start.
            self._last_env_time = ct.current_time
        elif not hasattr(self, "_last_env_time"):
            # Defensive: a resume on an engine that never bootstrapped here.
            # On a normal pause→resume `_last_env_time` PERSISTS on self so the
            # next physics dt spans the true gap since the last env tick.
            self._last_env_time = ct.current_time

        while self._running and not self._paused:
            event = ct.pop_next_event()
            if event is None:
                break  # No more events or past max_time

            # Map continuous time to a pseudo-round for event logging
            pseudo_round = int(ct.current_time)
            self.state.temporal.current_round = pseudo_round

            if event.event_type == "agent_turn":
                entity_id = event.entity_id
                if not entity_id:
                    continue
                entity = self.state.get_entity(entity_id)
                if not entity or not entity.alive:
                    continue

                # Snapshot the latest action BEFORE the turn so we can
                # detect whether this turn actually recorded a new one.
                before = self.state.action_history.get_recent(entity_id, 1)
                before_id = id(before[0]) if before else None

                # Run the agent's turn (reuses existing logic)
                self._run_agent_turn(entity_id)

                # Schedule next turn based on the action just taken.
                # If no new action was recorded (skip/sequence/error), fall
                # back to the configured default — using the previous
                # action's duration would mis-schedule the next tick.
                after = self.state.action_history.get_recent(entity_id, 1)
                if after and (before_id is None or id(after[0]) != before_id):
                    action_name = after[0].action_name
                else:
                    action_name = "default"
                ct.schedule_next_turn_after_action(entity_id, action_name)

            elif event.event_type == "environment":
                # Process environment events
                if self.world_event_engine:
                    self._process_world_events(pseudo_round)
                if self.state.property_dynamics:
                    dynamics_changes = self.state.property_dynamics.tick(self.state, pseudo_round)
                    for change in dynamics_changes:
                        self._emit_event(
                            "environment_change",
                            actor_id=change.get("entity_id"),
                            data=change,
                            narrative=change.get("narrative", "The environment shifts."),
                        )
                # Integrate physics by the REAL elapsed time since the last
                # environment tick (true continuous dt), then reschedule the
                # next recurring tick. This is the continuous world clock.
                dt = ct.current_time - getattr(self, "_last_env_time", ct.current_time)
                if dt > 0:
                    self._tick_physics(dt)
                self._last_env_time = ct.current_time
                if event.data.get("recurring"):
                    ct.schedule_environment_tick()

            # Check termination
            triggered = self._check_termination()
            if triggered:
                self.terminated_by = triggered.name
                self._emit_event(
                    "simulation_terminated",
                    data={"condition": triggered.name, "description": triggered.description},
                    narrative=f"Simulation terminated: {triggered.name} — {triggered.description}",
                )
                self._running = False
                break

        if not self._paused:
            self._emit_event(
                "simulation_end",
                narrative=f"Continuous simulation ended at time {ct.current_time:.2f}.",
            )
            self._running = False
        return self.state

    def pause(self):
        """Pause the simulation after the current turn completes."""
        self._paused = True

    def resume(self) -> WorldState:
        """Resume a paused simulation. Continues the run loop."""
        if not self._paused:
            return self.state
        self._paused = False
        # Continuous-time sims must resume on the continuous event loop, not
        # the discrete round loop below — otherwise a paused continuous sim
        # silently switches to discrete semantics on resume.
        if self._continuous_time is not None:
            self._running = True
            return self._run_continuous()
        # Continue from the authoritative round counter — _run_discrete is
        # resume-aware (remaining budget only, bracket events emitted once).
        return self._run_discrete()

    def is_paused(self) -> bool:
        """Check if the simulation is currently paused."""
        return self._paused

    def _run_round(self):
        """Execute a single round with all phases."""
        try:
            self._run_round_inner()
        except TypeError:
            import traceback
            logger.error(f"TypeError in round execution:\n{traceback.format_exc()}")
            raise

    def _run_round_inner(self):
        """Inner round execution logic."""
        round_num = self.state.temporal.current_round
        self._emit_event("round_start", narrative=f"Round {round_num} begins.")

        if self.on_round_start:
            self.on_round_start(round_num, self.state)

        # Clear per-round messages
        self.state.messages.start_round()

        # Tick world models (confidence decay) at round start
        for agent in self.state.get_agent_entities():
            wm = self.state.world_models.get(agent.id)
            if wm:
                wm.tick(round_num)

        # Tick cognitive architecture (emotional decay, stress updates)
        if self.state.cognition:
            self.state.cognition.tick_all(round_num)

        # Tick data connectors (fetch external data, apply mappings)
        if self.state.connectors:
            self.state.connectors.tick(self.state, round_num)

        # Tick domain modules (domain-specific per-round logic)
        if self.state.domain_modules:
            domain_changes = self.state.domain_modules.tick_all(self.state, round_num)
            for change in domain_changes:
                # Domain modules can name their own event type via
                # change["event_type"] so termination conditions (and
                # the UI) can react to game-specific events like
                # "mafia_victory" or "phase_revealed". Default stays
                # "domain_tick" for backward compat. When a module returns
                # change={"event_type", "data": {...}, ...}, we use the
                # inner data as the event's payload — that keeps the saved
                # event shape canonical (data IS the payload, not a wrapper).
                # Many DomainModules (Monopoly, Chess, Securities Trading)
                # return dicts keyed by `"type"`, not `"event_type"` — fall
                # back to that so the canonical event type reaches the FE
                # instead of getting hidden behind a generic "domain_tick".
                evt_type = (
                    change.get("event_type")
                    or change.get("type")
                    or "domain_tick"
                )
                payload = change.get("data") if isinstance(change.get("data"), dict) else change
                self._emit_event(
                    evt_type,
                    actor_id=change.get("actor_id"),
                    target_id=change.get("target_id"),
                    data=payload,
                    narrative=change.get("narrative", f"Domain module tick: {change.get('type', 'unknown')}"),
                )

        # Process mid-simulation controller: pending injections and narrative directives
        if self.state.controller:
            # Process event injections
            ready_injections = self.state.controller.process_pending_injections(round_num)
            for inj in ready_injections:
                self._emit_event(
                    inj.event_type,
                    data={"injection_id": inj.id, "description": inj.description, **inj.data},
                    narrative=inj.description or f"Injected event: {inj.event_type}",
                )
                # Apply injection effects to target entities (or all agents if global)
                if inj.effects:
                    targets = inj.target_entities or [e.id for e in self.state.get_agent_entities()]
                    for target_id in targets:
                        target_ent = self.state.get_entity(target_id)
                        if target_ent:
                            self._apply_effects(
                                [Effect(**eff) if isinstance(eff, dict) else eff for eff in inj.effects],
                                target_ent, None, {}, None,
                            )

            # Check narrative directives
            fired_directives = self.state.controller.check_directives(self.state, round_num)
            for directive in fired_directives:
                self._emit_event(
                    "narrative_directive",
                    data={
                        "directive_id": directive.id,
                        "directive_name": directive.name,
                        **directive.event_data,
                    },
                    narrative=directive.narrative_event or f"Narrative: {directive.name}",
                )

        # Tick social platform (process content spread, feeds, reputation)
        if self.state.social:
            self.state.social.tick(self.state, round_num, self._rng)

        # Process world events before agent turns
        if self.world_event_engine:
            self._process_world_events(round_num)

        # Process autonomous property dynamics
        if self.state.property_dynamics:
            dynamics_changes = self.state.property_dynamics.tick(self.state, round_num)
            for change in dynamics_changes:
                self._emit_event(
                    "environment_change",
                    actor_id=change.get("entity_id"),
                    data=change,
                    narrative=change.get("narrative", "The environment shifts."),
                )

        # Advance continuous coupled dynamics ("physics"). In discrete mode each
        # round is one unit of physics time (dt=1.0): the world evolves between
        # agent turns via the ODE system, the LLM agents then react to the
        # evolved state on their turn.
        self._tick_physics(1.0)

        # An env that hasn't declared any phases yet (e.g., a partially-
        # built draft being test-run during the Studio build session) has
        # an empty `phases` list. Accessing `current_phase` on an empty
        # list raises IndexError and bubbles up as a hard crash. Skip the
        # inner phase loop in that case — there's literally nothing to run.
        if not self.state.temporal.phases:
            logger.warning(
                "round %d: no phases declared in schema — skipping phase loop "
                "(this run will produce no actions)", round_num,
            )
        else:
            while True:
                if self._paused:
                    break
                phase = self.state.temporal.current_phase
                self._run_phase(phase)
                if not self.state.temporal.advance_phase():
                    break

        # Apply relation decay and check thresholds (skip if paused mid-round)
        if not self._paused:
            relation_events = self.state.relations.tick(round_num)
            for rel_event in relation_events:
                self._emit_event(
                    "relation_threshold",
                    data=rel_event,
                    narrative=f"Relation threshold: {rel_event.get('event_name', '')} — {rel_event.get('description', '')}",
                )

        # Tick negotiations (expire old, close auctions, check agreements)
        if not self._paused:
            neg_events = self.state.negotiations.tick(round_num)
            for ne in neg_events:
                self._emit_event(
                    ne.get("type", "negotiation_event"),
                    data=ne,
                    narrative=ne.get("narrative", ""),
                )
                # Execute auction resource transfers when auctions close with a winner
                if ne.get("type") == "auction_closed" and ne.get("winner_id"):
                    auction = self.state.negotiations.get_auction(ne.get("auction_id", ""))
                    if auction:
                        transfer_events = self.state.negotiations.execute_auction_award(auction, self.state)
                        for te in transfer_events:
                            self._emit_event(te["type"], data=te, narrative=te.get("narrative", ""))

            # Execute any newly created agreement transfers
            agr_events = self.state.negotiations.flush_pending_agreements(self.state)
            for ae in agr_events:
                self._emit_event(ae.get("type", "agreement_transfer"), data=ae, narrative=ae.get("narrative", ""))

            # Check agreement violations
            violations = self.state.negotiations.check_agreement_violations(self.state, round_num)
            for v in violations:
                self._emit_event(
                    "agreement_violated",
                    data=v,
                    narrative=v.get("narrative", "An agreement was violated."),
                )

        # Evaluate goals for all entities
        if not self._paused:
            for entity in self.state.get_agent_entities():
                goal_events = self.state.goals.evaluate_goals(entity.id, self.state, round_num)
                for ge in goal_events:
                    self._emit_event(
                        "goal_completed",
                        actor_id=ge["entity_id"],
                        data=ge,
                        narrative=f"{entity.name} completed goal: {ge['goal_description']}",
                    )

        # Tick controller takeovers (decrement remaining turns, expire finished ones)
        if not self._paused and self.state.controller:
            self.state.controller.tick_takeovers()

        # Check controller breakpoints
        if not self._paused and self.state.controller:
            triggered_bps = self.state.controller.check_breakpoints(self.state, round_num)
            for bp in triggered_bps:
                self._emit_event(
                    "breakpoint_triggered",
                    data={"breakpoint_id": bp.id, "breakpoint_name": bp.name, "action": bp.action},
                    narrative=f"Breakpoint triggered: {bp.name}",
                )
                if bp.action == "pause":
                    self._paused = True

        if self.on_round_end:
            self.on_round_end(round_num, self.state)

        # Derived rules — forward-chaining inference. Runs AFTER agent
        # turns and BEFORE termination check so newly-derived facts
        # (e.g. "hp <= 0 → alive = false") are visible to terminations.
        derived = getattr(self.state, "_derived_rules", None)
        if derived is not None:
            try:
                derived_changes = derived.tick(self)
                if derived_changes:
                    for ch in derived_changes:
                        self._emit_event(
                            "derived_fact",
                            data=ch if isinstance(ch, dict) else {"change": ch},
                            narrative=f"Derived: {ch}",
                        )
            except Exception:
                logger.exception("derived_rules tick failed")

        # Check termination conditions
        triggered = self._check_termination()
        if triggered:
            self.terminated_by = triggered.name
            # Resolve the winner so vizualisations + result screens can
            # show "X wins" without having to re-evaluate the predicate.
            winner_info = self._resolve_winner(triggered)
            self._emit_event(
                "simulation_terminated",
                data={
                    "condition": triggered.name,
                    "description": triggered.description,
                    **winner_info,
                },
                narrative=(
                    f"Simulation terminated: {triggered.name} — {triggered.description}"
                    + (f" — winner: {winner_info.get('winner_name')}"
                       if winner_info.get('winner_name') else "")
                ),
            )
            self._running = False

        # Check world invariants
        if self.invariant_checker and not self._paused:
            violations = self.invariant_checker.check_all(self.state, round_num)
            for v in violations:
                self._emit_event(
                    "invariant_violation",
                    data=v,
                    narrative=f"Invariant violation: {v.get('invariant', 'unknown')} — {v.get('message', '')}",
                )
                if v.get("severity") == "error":
                    self._running = False

        self._emit_event("round_end", narrative=f"Round {round_num} ends.")

    def _run_phase(self, phase):
        """Execute a single phase -- handler first, then eligible agents act in order."""
        # Execute phase handler if one is configured
        if phase.handler:
            try:
                handler = get_phase_handler(phase.handler)
                handler_result = handler.execute(
                    state=self.state,
                    params=phase.handler_params,
                    round_number=self.state.temporal.current_round,
                    rng=self._rng,
                )
                for evt in handler_result.events:
                    self._emit_event(
                        event_type=evt.get("type", "phase_handler"),
                        actor_id=evt.get("actor_id"),
                        target_id=evt.get("target_id"),
                        data=evt.get("data", {}),
                        narrative=evt.get("narrative", ""),
                    )
            except Exception as e:
                logger.error(f"Phase handler '{phase.handler}' failed: {e}")
                self._emit_event(
                    "phase_handler_error",
                    data={"handler": phase.handler, "error": str(e)},
                    narrative=f"Phase handler error: {e}",
                )

        agents = self.state.get_agent_entities()

        # Filter by phase active_roles
        if phase.active_roles:
            agents = [a for a in agents if a.entity_type in phase.active_roles]

        # Resolve turn order based on phase initiative settings
        round_num = self.state.temporal.current_round
        turn_order = TurnOrderResolver.resolve(agents, phase, round_num, self._rng)
        self.state.temporal.set_turn_order(turn_order)

        if getattr(phase, "resolution_mode", "sequential") == "simultaneous":
            self._run_phase_simultaneous(turn_order)
        elif self.parallel_decisions > 0 and len(turn_order) > 1:
            self._run_phase_parallel(turn_order)
        else:
            for entity_id in turn_order:
                if not self._running or self._paused:
                    break
                entity = self.state.get_entity(entity_id)
                if not entity or not entity.alive:
                    continue
                self._run_agent_turn(entity_id)

    def _run_phase_simultaneous(self, turn_order: List[str]):
        """Commit-then-reveal phase.

        Every eligible agent picks an action against the SAME perception
        snapshot (no in-phase state updates leak between them). Decisions
        run in parallel (no order coupling). Once ALL decisions return,
        their actions are applied in turn_order under the resolve lock —
        so the "reveal" step is deterministic and atomic.

        This is the primitive for RPS, sealed-bid auctions, blind voting,
        simultaneous role moves in social deduction games. Resolution
        order within the batch follows turn_order so games can still
        define tie-break ordering.
        """
        import threading

        # Step 1: snapshot perceptions for everyone BEFORE any LLM call
        # fires. Each agent sees identical pre-phase state.
        agent_tasks: List[dict] = []
        for entity_id in turn_order:
            if not self._running or self._paused:
                break
            entity = self.state.get_entity(entity_id)
            if not entity or not entity.alive:
                continue
            if self.state.crowd_agents and self.state.crowd_agents.is_crowd(entity_id):
                # Crowd behaviors are cheap (no LLM) — keep them in the
                # batch but mark them so we can dispatch via crowd path.
                agent_tasks.append({"entity_id": entity_id, "crowd": True})
                continue
            perception, valid_actions = self._build_agent_perception(entity_id)
            if not valid_actions:
                continue
            agent_tasks.append({
                "entity_id": entity_id,
                "perception": perception,
                "valid_actions": valid_actions,
            })

        if not agent_tasks:
            return

        # Step 2: collect decisions in parallel. Crowd agents resolve
        # locally; LLM agents fire concurrently.
        submissions: Dict[str, ActionInstance] = {}
        sub_lock = threading.Lock()

        # Crowd agents run serially first (cheap, no LLM). They were
        # previously filtered out of the parallel dispatch AND never run
        # anywhere else — so crowd traders in a simultaneous phase silently
        # never acted. Run them here, in turn_order, before the LLM batch.
        for task in agent_tasks:
            if not self._running or self._paused:
                break
            if task.get("crowd"):
                self._run_agent_turn(task["entity_id"])

        def _collect(task):
            eid = task["entity_id"]
            try:
                action = self.decision_fn(eid, task["perception"], task["valid_actions"]) if self.decision_fn else None
            except Exception as e:
                logger.error(f"Simultaneous-phase decision failed for {eid}: {e}")
                with sub_lock:
                    self._emit_event(
                        "decision_error",
                        actor_id=eid,
                        data={"error": str(e)},
                        narrative=f"Decision error for {eid}: {e}",
                    )
                return
            if action is not None:
                with sub_lock:
                    submissions[eid] = action

        workers = max(1, self.parallel_decisions or len(agent_tasks))
        # Propagate the caller's contextvars (notably the wallet usage
        # context) into each worker thread. ThreadPoolExecutor does NOT
        # inherit context by default. Important: each task gets its OWN
        # context COPY — a single Context object can only be `.run()`
        # once, so sharing across N submits raises "already entered".
        import contextvars as _ctxvars
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_ctxvars.copy_context().run, _collect, t)
                for t in agent_tasks if not t.get("crowd")
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Simultaneous task error: {e}")

        # Step 3: reveal — apply submissions in turn_order, serially.
        # All agents committed against the same perception; resolution
        # order is deterministic and visible to subsequent phases.
        for entity_id in turn_order:
            if not self._running or self._paused:
                break
            action = submissions.get(entity_id)
            if action is None:
                continue
            self._resolve_and_apply(entity_id, action)

    def _run_phase_parallel(self, turn_order: List[str]):
        """Run agent decisions in parallel micro-batches with fresh perceptions.

        Uses micro-batches (sized to parallel_decisions) so each batch sees
        updated market state from the previous batch's trades.  This prevents
        the "stale perception" problem where 100 agents all see the same price
        and pile into the same direction.

        Within each micro-batch:
          1. Build perceptions (sees live price from previous batch's trades)
          2. Fire LLM calls in parallel (the expensive part)
          3. Resolve in deterministic batch order (NOT completion order) so
             the price path is reproducible. All agents in the batch already
             decided against the same pre-batch snapshot, so resolution order
             changes nothing they saw.

        Crowd agents run first (no LLM needed).
        """
        import threading

        # Step 0: Run crowd agents first (instant, no LLM)
        llm_agents: List[str] = []
        for entity_id in turn_order:
            if not self._running or self._paused:
                break
            entity = self.state.get_entity(entity_id)
            if not entity or not entity.alive:
                continue
            if self.state.crowd_agents and self.state.crowd_agents.is_crowd(entity_id):
                self._run_agent_turn(entity_id)
            else:
                llm_agents.append(entity_id)

        if not llm_agents:
            return

        # Step 1: Process LLM agents in micro-batches
        batch_size = max(1, self.parallel_decisions)
        resolve_lock = threading.Lock()

        for batch_start in range(0, len(llm_agents), batch_size):
            if not self._running or self._paused:
                break

            batch = llm_agents[batch_start:batch_start + batch_size]

            # Build fresh perceptions for THIS batch (sees latest prices)
            agent_tasks: List[dict] = []
            for entity_id in batch:
                entity = self.state.get_entity(entity_id)
                if not entity or not entity.alive:
                    continue
                perception, valid_actions = self._build_agent_perception(entity_id)
                if not valid_actions:
                    continue
                agent_tasks.append({
                    "entity_id": entity_id,
                    "perception": perception,
                    "valid_actions": valid_actions,
                })

            if not agent_tasks:
                continue

            # Fire LLM calls in parallel (the expensive part), but only
            # COLLECT decisions here — do not resolve yet. All agents in this
            # micro-batch already decided against the same pre-batch
            # perception snapshot, so resolving them in LLM-completion order
            # would make the price path depend on network timing (a
            # reproducibility hole). We instead resolve in deterministic
            # ``batch`` order below, under the lock.
            submissions: Dict[str, ActionInstance] = {}
            sub_lock = threading.Lock()

            def _decide(task):
                eid = task["entity_id"]
                try:
                    action = self.decision_fn(eid, task["perception"], task["valid_actions"]) if self.decision_fn else None
                except Exception as e:
                    logger.error(f"Parallel decision failed for {eid}: {e}")
                    # Surface the failure in the event log so the UI /
                    # transcript shows that an agent was unable to act,
                    # rather than silently doing nothing.
                    with sub_lock:
                        self._emit_event(
                            "decision_error",
                            actor_id=eid,
                            data={"error": str(e)},
                            narrative=f"Decision error for {eid}: {e}",
                        )
                    return
                if action is not None:
                    with sub_lock:
                        submissions[eid] = action

            import contextvars as _ctxvars
            with ThreadPoolExecutor(max_workers=self.parallel_decisions) as pool:
                futures = [
                    pool.submit(_ctxvars.copy_context().run, _decide, t)
                    for t in agent_tasks
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Parallel agent turn error: {e}")

            # Resolve in deterministic batch order (live price updates still
            # happen here, just in a reproducible sequence).
            with resolve_lock:
                for task in agent_tasks:
                    action = submissions.get(task["entity_id"])
                    if action is not None:
                        self._resolve_and_apply(task["entity_id"], action)

    def _build_agent_perception(self, entity_id: str):
        """Build perception and valid actions for an agent. Returns
        (perception, valid_actions). The canonical implementation lives in
        ``runtime/perception.py``; this method is a thin delegation
        preserved for backwards compatibility with code that calls
        ``engine._build_agent_perception(...)`` directly."""
        from .perception import build_perception
        return build_perception(self, entity_id)

    def _resolve_action_def(self, entity, action_instance: "ActionInstance"):
        """Canonical implementation in ``runtime/actions.py``."""
        from .actions import _resolve_action_def
        return _resolve_action_def(self, entity, action_instance)

    def _coerce_action_params(self, entity, action_def, action_instance) -> bool:
        """Canonical implementation in ``runtime/actions.py``."""
        from .actions import _coerce_action_params
        return _coerce_action_params(self, entity, action_def, action_instance)

    def _resolve_and_apply(self, entity_id: str, action_instance: ActionInstance):
        """Canonical implementation in ``runtime/actions.py``."""
        from .actions import _resolve_and_apply
        return _resolve_and_apply(self, entity_id, action_instance)

    def _run_agent_turn(self, entity_id: str):
        """Execute a single agent's turn: perceive -> decide -> resolve -> apply."""
        try:
            self._run_agent_turn_inner(entity_id)
        except TypeError:
            import traceback
            logger.error(f"TypeError in agent turn for {entity_id}:\n{traceback.format_exc()}")
            raise

    def _run_agent_turn_inner(self, entity_id: str):
        """Canonical implementation in ``runtime/turn.py``."""
        from .turn import run_agent_turn
        return run_agent_turn(self, entity_id)

    def _apply_effects(
        self,
        effects: List[Effect],
        actor: Any,
        target: Any,
        params: dict,
        result: ResolutionResult,
    ) -> List[dict]:
        """Apply a list of effects. The canonical implementation lives in
        ``runtime/effect_dispatch.py``; this method is a thin delegation
        preserved for backwards compatibility with code that calls
        ``engine._apply_effects(...)`` directly."""
        from .effect_dispatch import apply_effects
        return apply_effects(self, effects, actor, target, params, result)

    def _evaluate_conditional_clause(
        self, spec, *, actor, target, params, result, resolve_val,
    ) -> bool:
        """Canonical implementation in ``runtime/conditions.py``."""
        from .conditions import _evaluate_conditional_clause
        return _evaluate_conditional_clause(
            self, spec, actor=actor, target=target, params=params,
            result=result, resolve_val=resolve_val,
        )

    def _evaluate_world_condition(self, spec: dict) -> bool:
        """Canonical implementation in ``runtime/conditions.py``."""
        from .conditions import _evaluate_world_condition
        return _evaluate_world_condition(self, spec)

    def _resolve_multi_target(self, target_token: str, actor, target):
        """Canonical implementation in ``runtime/conditions.py``."""
        from .conditions import _resolve_multi_target
        return _resolve_multi_target(self, target_token, actor, target)

    def _evaluate_effect_condition(self, condition, actor, target, params=None) -> bool:
        """Canonical implementation in ``runtime/conditions.py``."""
        from .conditions import _evaluate_effect_condition
        return _evaluate_effect_condition(self, condition, actor, target, params)

    def _check_target_preconditions(self, actor, target, action_def) -> bool:
        """Canonical implementation in ``runtime/conditions.py``."""
        from .conditions import _check_target_preconditions
        return _check_target_preconditions(self, actor, target, action_def)

    @staticmethod
    def _compare(val, operator: str, target_val) -> bool:
        """Canonical implementation in ``runtime/conditions.py``."""
        from .conditions import _compare
        return _compare(val, operator, target_val)

    @staticmethod
    def _compare_inner(val, operator: str, target_val) -> bool:
        """Canonical implementation in ``runtime/conditions.py``."""
        from .conditions import _compare_inner
        return _compare_inner(val, operator, target_val)

    def _entities_snapshot(self) -> Dict[str, Any]:
        """Canonical implementation in ``runtime/triggers.py``."""
        from .triggers import _entities_snapshot
        return _entities_snapshot(self)

    def _emit_event(
        self,
        event_type: str,
        actor_id: str = None,
        target_id: str = None,
        action_name: str = None,
        data: dict = None,
        narrative: str = "",
    ):
        """Canonical implementation in ``runtime/triggers.py``."""
        from .triggers import _emit_event
        return _emit_event(
            self, event_type, actor_id, target_id, action_name, data, narrative,
        )

    def _fire_trigger(
        self,
        spec: "TriggerSpec",
        actor_id: Optional[str],
        target_id: Optional[str],
        event_data: Dict[str, Any],
    ) -> None:
        """Canonical implementation in ``runtime/triggers.py``."""
        from .triggers import _fire_trigger
        return _fire_trigger(self, spec, actor_id, target_id, event_data)

    def stop(self):
        """Stop the simulation after the current turn completes."""
        self._running = False
        self._stopped = True

    # -------------------------------------------------------------------
    # Termination condition evaluation
    # -------------------------------------------------------------------

    def _check_termination(self) -> Optional[TerminationCondition]:
        """Canonical implementation in ``runtime/termination.py``."""
        from .termination import _check_termination
        return _check_termination(self)

    def _evaluate_condition(self, tc: TerminationCondition) -> bool:
        """Canonical implementation in ``runtime/termination.py``."""
        from .termination import _evaluate_condition
        return _evaluate_condition(self, tc)

    def _resolve_winner(self, tc) -> Dict[str, Any]:
        """Canonical implementation in ``runtime/termination.py``."""
        from .termination import _resolve_winner
        return _resolve_winner(self, tc)

    @staticmethod
    def _winning_mark_in_snapshot(board, patterns):
        """Canonical implementation in ``runtime/termination.py``."""
        from .termination import _winning_mark_in_snapshot
        return _winning_mark_in_snapshot(board, patterns)

    def _evaluate_board_pattern(self, params: dict) -> bool:
        """Canonical implementation in ``runtime/termination.py``."""
        from .termination import _evaluate_board_pattern
        return _evaluate_board_pattern(self, params)

    @staticmethod
    def _scan_board_pattern(
        board: List[List[Any]],
        rows: int,
        cols: int,
        kind: str,
        n: int,
        mode: str,
        target_value: Any,
    ) -> bool:
        """Canonical implementation in ``runtime/termination.py``."""
        from .termination import _scan_board_pattern
        return _scan_board_pattern(board, rows, cols, kind, n, mode, target_value)

    def _tick_physics(self, dt: float) -> None:
        """Advance the continuous coupled-dynamics ("physics") system by ``dt``
        time units and emit a change event per affected variable. No-op when the
        env declares no physics. Works identically in discrete (dt=1 per round)
        and continuous (dt=real elapsed gap) modes."""
        physics = getattr(self.state, "physics", None)
        if physics is None or physics.is_empty():
            return
        try:
            changes = physics.tick(self.state, dt)
        except Exception:  # noqa: BLE001 — physics is an enhancement; never crash the sim
            logger.exception("physics tick failed (dt=%s) — skipping this step", dt)
            return
        for change in changes:
            self._emit_event(
                "physics_step",
                actor_id=change.get("entity_id"),
                data=change,
                narrative=change.get("narrative", "The world evolves."),
            )

    def _process_world_events(self, round_number: int):
        """Evaluate and apply world events for this round."""
        triggered = self.world_event_engine.evaluate(self.state, round_number)
        for te in triggered:
            defn = te.definition
            # Apply effects to each affected entity
            for entity_id in te.affected_entities:
                entity = self.state.get_entity(entity_id)
                if entity:
                    self._apply_effects(defn.effects, entity, None, {}, None)

            self._emit_event(
                "world_event",
                data={
                    "event_name": defn.name,
                    "affected_entities": te.affected_entities,
                    "duration": defn.duration,
                },
                narrative=f"World event: {defn.name} — {defn.description}",
            )
