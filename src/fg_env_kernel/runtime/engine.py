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
from typing import Any, Callable, Dict, List, Optional

from ..state import WorldState
from ..action import ActionInstance, ActionDefinition, Effect, EffectOperation
from ..messaging import Message
from ..resolution import get_resolution, ResolutionResult
from ..visibility import PerceptionBuilder, TrendAnalyzer
from ..event import SimEvent
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


def _coerce_effects(raw: Any) -> List["Effect"]:
    """Turn schema-style effect dicts into Effect dataclass instances.

    Used by features that store effects in JSON (deck cards, phase-
    state-machine transitions, ad-hoc world events). The shape mirrors
    `effects_on_success`:
      { operation: "add", target: "actor", field: "money", value: 50 }
    """
    from ..action import Effect, EffectOperation
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
                from ..registry import registry as _kreg
                if _kreg.effects.has(str(op_raw)):
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
    ):
        self.state = state
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
        """Run the simulation in discrete (turn-based) mode."""
        agents = self.state.get_agent_entities()
        self._emit_event(
            "simulation_start",
            narrative=f"Simulation begins. {len(agents)} agents active.",
        )

        for round_num in range(self.max_rounds):
            if not self._running or self._paused:
                break
            self.state.temporal.advance_round()
            self._run_round()
            if self._paused:
                break

        if not self._paused:
            self._emit_event(
                "simulation_end",
                narrative=f"Simulation ended after {self.state.temporal.current_round} rounds.",
            )
            self._running = False
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
        # Continue the loop from where we left off using the authoritative round counter
        remaining = self.max_rounds - self.state.temporal.current_round
        for _ in range(max(0, remaining)):
            if not self._running or self._paused:
                break
            self.state.temporal.advance_round()
            self._run_round()
            if self._paused:
                break
        if not self._paused:
            self._emit_event(
                "simulation_end",
                narrative=f"Simulation ended after {self.state.temporal.current_round} rounds.",
            )
            self._running = False
        return self.state

    def is_paused(self) -> bool:
        """Check if the simulation is currently paused."""
        return self._paused

    def _run_round(self):
        """Execute a single round with all phases."""
        try:
            self._run_round_inner()
        except TypeError as e:
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
        """Build perception and valid actions for an agent. Returns (perception, valid_actions)."""
        entity = self.state.get_entity(entity_id)
        if not entity:
            return {}, []

        round_num = self.state.temporal.current_round

        # Tick status effects
        tick_effects = self.state.status_effects.tick(entity_id, round_num)
        if tick_effects:
            self._apply_effects(tick_effects, entity, None, {}, None)

        # Location effects
        entity_loc = self.state.locations.get(entity_id)
        if entity_loc:
            loc_changes = self.state.location_properties.apply_tick_effects(
                entity, entity_loc, self.state.entity_types,
            )
            for lc in loc_changes:
                self._emit_event(
                    "location_effect",
                    actor_id=entity_id,
                    data=lc,
                    narrative=f"{entity.name} affected by {entity_loc}: {lc.get('field')} {lc.get('effect')} {(lc.get('new') or 0) - (lc.get('old') or 0):.1f}",
                )

        # Build perception
        active_events = None
        if self.world_event_engine:
            active_events = [
                {"name": ae.definition.name, "description": ae.definition.description,
                 "remaining_rounds": ae.remaining_rounds}
                for ae in self.world_event_engine.get_active_events()
            ]
        perception = self._perception_builder.build_perception(
            observer_id=entity_id,
            observer_type=entity.entity_type,
            rules=self.state.visibility_rules,
            entities=self.state.entities,
            entity_types=self.state.entity_types,
            resources=self.state.resources,
            relations=self.state.relations,
            spatial=self.state.spatial,
            temporal=self.state.temporal,
            active_world_events=active_events,
            faction_manager=self.state.factions,
        )

        # World brief — name + description + rules from the env's
        # template. The agent reads this to understand how to play
        # the game. Includes the natural-language rules markdown.
        world_brief = getattr(self.state, "_world_brief", None)
        if world_brief:
            perception["world_brief"] = dict(world_brief)

        # Messages
        entity_faction = self.state.factions.get_entity_faction(entity_id)
        incoming = self.state.messages.get_for_entity(entity_id, entity_faction)
        if incoming:
            perception["incoming_messages"] = [
                {"sender": m.sender_name, "sender_id": m.sender_id,
                 "content": m.content, "type": m.message_type}
                for m in incoming
            ]

        # Domain data
        if self.state.domain_modules:
            domain_data = self.state.domain_modules.get_perception_data(entity_id, self.state)
            if domain_data:
                perception["domain_data"] = domain_data
            for _mod_name, _module in self.state.domain_modules._modules.items():
                if hasattr(_module, 'get_visibility_overrides'):
                    overrides = _module.get_visibility_overrides()
                    if overrides.get("hide_agent_identities"):
                        perception["visible_entities"] = []
                        perception.pop("agent_models", None)

        # Crowd trends
        if self.state.crowd_agents:
            crowd_trends = self.state.crowd_agents.get_crowd_trends()
            if crowd_trends:
                perception["crowd_trends"] = crowd_trends

        # Cognitive processing
        if self.state.cognition:
            overlay = self.state.cognition.get_prompt_overlay(entity_id)
            if overlay:
                perception["cognitive_state"] = overlay
            perception = self.state.cognition.process_perception_for(entity_id, perception)

        # Trade history — structured blotter from this agent's past trades
        try:
            agent_events = self.state.event_log.get_by_actor(entity_id)
            trade_actions = {"buy", "sell", "buy_yes", "buy_no", "sell_yes", "sell_no"}
            trades = []
            for ev in agent_events:
                if ev.event_type != "action_resolved" or ev.action_name not in trade_actions:
                    continue
                details = ev.data.get("details", {}) if isinstance(ev.data, dict) else {}
                if not details.get("shares") and not details.get("amount"):
                    continue
                trades.append({
                    "round": ev.round_number,
                    "action": ev.action_name,
                    "amount": details.get("amount", 0),
                    "shares": details.get("shares", 0),
                    "price_at_trade": details.get("price_after") or details.get("execution_price") or details.get("new_price", 0),
                })
            if trades:
                # Summarize: last 10 trades + running P&L
                total_spent = sum(t["amount"] for t in trades if t["action"].startswith("buy"))
                total_received = sum(t["amount"] for t in trades if t["action"].startswith("sell"))
                perception["trade_history"] = {
                    "total_trades": len(trades),
                    "total_spent": round(total_spent, 2),
                    "total_received": round(total_received, 2),
                    "realized_pnl": round(total_received - total_spent, 2),
                    "recent_trades": trades[-10:],
                }
        except Exception:
            pass

        # Time context (when simulation has time config)
        time_ctx = self.state.temporal.time_context(self.max_rounds)
        if time_ctx:
            perception["time_context"] = time_ctx

        # Role + asymmetric info. The agent always knows their own role.
        # Teammates / extra_visible_roles are revealed per the registry's
        # rules. Everyone else's role is hidden.
        if self.state.roles and self.state.roles.assignments:
            own = self.state.roles.get_role(entity_id)
            if own:
                perception["your_role"] = {
                    "name": own.name,
                    "team": own.team,
                    "description": own.description,
                }
            visible = self.state.roles.visible_role_map(entity_id)
            visible.pop(entity_id, None)
            if visible:
                perception["visible_roles"] = visible
            mates = self.state.roles.teammates(entity_id)
            if mates:
                perception["teammates"] = sorted(mates)

        # Open polls the agent is eligible to vote in. Domain modules
        # decide when/how to open polls; the kernel just surfaces them.
        if self.state.polls:
            my_polls = self.state.polls.list_for_voter(entity_id)
            if my_polls:
                perception["open_polls"] = [
                    {
                        "poll_id": p.poll_id,
                        "description": p.description,
                        "options": list(p.options),
                        "rule": p.rule,
                        "you_voted": p.votes.get(entity_id),
                        "allow_abstain": p.allow_abstain,
                    }
                    for p in my_polls
                ]

        # Recap the agent's own recent decisions so they can build on
        # past behavior instead of repeating mistakes. Pulled from the
        # action history; private to this agent.
        recent = self.state.action_history.get_recent(entity_id, 5)
        if recent:
            perception["your_recent_actions"] = [
                {
                    "round": r.round_number,
                    "action": r.action_name,
                    "success": r.success,
                }
                for r in recent
            ]

        valid_actions = self.state.get_valid_actions(entity_id)
        # Domain modules can narrow the action list (e.g. a folded poker
        # player has nothing legal to do for the rest of the hand).
        if self.state.domain_modules:
            valid_actions = self.state.domain_modules.filter_valid_actions(
                entity_id, valid_actions, self.state,
            )
        return perception, valid_actions

    def _resolve_action_def(self, entity, action_instance: "ActionInstance"):
        """Resolve an action name to its definition, normalizing it in place.

        Returns the ActionDefinition, or None if the name can't be resolved
        (in which case the appropriate event has already been emitted).

        Used by BOTH the sequential turn path and the parallel/simultaneous
        ``_resolve_and_apply`` path so action-name handling is identical
        everywhere. We auto-apply a correction ONLY for an unambiguous
        case/whitespace match or a high-confidence typo; weaker matches are
        rejected (with suggestions) so we never silently substitute a
        mechanically different action the agent didn't choose.
        """
        action_name = action_instance.action_name
        action_def = self.state.action_definitions.get(action_name)
        if action_def is not None:
            self._coerce_action_params(entity, action_def, action_instance)
            return action_def

        from difflib import get_close_matches
        valid = list(self.state.action_definitions.keys())
        suggestions = get_close_matches(action_name, valid, n=3, cutoff=0.6)
        # Case/whitespace-only mismatches are unambiguous — match them first
        # (difflib is case-sensitive, so "Buy"→"buy" scores only 0.67 and
        # would otherwise be rejected, forfeiting a turn that should plainly
        # execute).
        _norm = action_name.casefold().strip()
        strong = [k for k in valid if k.casefold().strip() == _norm]
        if not strong:
            HIGH_CONFIDENCE = 0.88
            strong = get_close_matches(action_name, valid, n=1, cutoff=HIGH_CONFIDENCE)
        if strong:
            fallback = strong[0]
            self._emit_event(
                "action_corrected",
                actor_id=entity.id,
                action_name=action_name,
                data={"reason": "typo_autocorrect", "details": {
                    "action_name": action_name, "corrected_to": fallback,
                    "suggestions": suggestions}},
                narrative=(f"{entity.name} called '{action_name}' — auto-corrected "
                           f"to near-identical '{fallback}'."),
            )
            action_instance.action_name = fallback
            corrected = self.state.action_definitions[fallback]
            self._coerce_action_params(entity, corrected, action_instance)
            return corrected

        self._emit_event(
            "action_failed",
            actor_id=entity.id,
            action_name=action_name,
            data={"reason": "unknown_action", "details": {
                "action_name": action_name, "suggestions": suggestions,
                "valid_actions": valid}},
            narrative=(
                f"{entity.name}'s action '{action_name}' is not defined."
                + (f" Did you mean: {', '.join(suggestions)}?" if suggestions else "")
                + f" Valid actions: {', '.join(valid) or '(none)'}."
            ),
        )
        return None

    def _coerce_action_params(self, entity, action_def, action_instance) -> None:
        """Enforce declared parameter types and bounds on submitted values.

        Declared ``parameters`` on an ActionDefinition were previously
        prompt decoration only — an out-of-range or wrong-typed LLM value
        flowed raw into resolution and effects. Coerce numerics to their
        declared type, clamp to [min, max], and fill a missing value from
        the declaration's default. Every correction is announced with an
        ``action_corrected`` event so the transcript stays honest.
        """
        if not action_def.parameters:
            return
        submitted = action_instance.parameters or {}
        fixes: dict[str, Any] = {}
        for decl in action_def.parameters:
            if not isinstance(decl, dict) or not decl.get("name"):
                continue
            name = str(decl["name"])
            ptype = str(decl.get("type") or "").lower()
            lo, hi = decl.get("min"), decl.get("max")
            if lo is None:
                lo = decl.get("min_value")
            if hi is None:
                hi = decl.get("max_value")
            value = submitted.get(name)
            if value is None:
                if decl.get("default") is not None:
                    fixes[name] = decl["default"]
                continue
            if ptype in ("int", "float", "number") or lo is not None or hi is not None:
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    if decl.get("default") is not None:
                        fixes[name] = decl["default"]
                    continue
                clamped = num
                if lo is not None:
                    clamped = max(float(lo), clamped)
                if hi is not None:
                    clamped = min(float(hi), clamped)
                if ptype == "int":
                    clamped = int(round(clamped))
                out = clamped
                if out != value or type(out) is not type(value):
                    fixes[name] = out
        if not fixes:
            return
        merged = dict(submitted)
        changed = {
            k: {"submitted": submitted.get(k), "coerced": v}
            for k, v in fixes.items() if submitted.get(k) != v
        }
        merged.update(fixes)
        action_instance.parameters = merged
        if changed:
            self._emit_event(
                "action_corrected",
                actor_id=entity.id,
                action_name=action_instance.action_name,
                data={"reason": "parameter_coerced", "details": changed},
                narrative=(
                    f"{entity.name}'s '{action_instance.action_name}' "
                    f"parameters were coerced to their declared bounds: "
                    + ", ".join(
                        f"{k}: {v['submitted']!r} → {v['coerced']!r}"
                        for k, v in changed.items())
                ),
            )

    def _resolve_and_apply(self, entity_id: str, action_instance: ActionInstance):
        """Resolve an action and apply its effects to the world state."""
        entity = self.state.get_entity(entity_id)
        if not entity:
            return

        action_def = self._resolve_action_def(entity, action_instance)
        if action_def is None:
            return
        action_name = action_instance.action_name

        target = None
        if action_instance.target_id:
            target = self.state.get_entity(action_instance.target_id)
            if target is None:
                # If the action doesn't declare a target_type, silently drop
                # the stray target instead of failing the whole turn. LLMs
                # sometimes hallucinate target_id for target-less actions
                # (e.g. chess `make_move` with target_id='d7'). Treat that
                # as harmless: clear the target and proceed.
                if not action_def.target_type:
                    action_instance.target_id = None
                else:
                    self._emit_event(
                        "action_failed",
                        actor_id=entity_id,
                        action_name=action_name,
                        target_id=action_instance.target_id,
                        data={"reason": "unknown_target", "details": {"target_id": action_instance.target_id}},
                        narrative=f"{entity.name} targeted a nonexistent entity ({action_instance.target_id}).",
                    )
                    return
        # Targeted action with no target supplied -> reject before resolution.
        if action_def.target_type and target is None:
            self._emit_event(
                "action_failed",
                actor_id=entity_id,
                action_name=action_name,
                data={"reason": "missing_target"},
                narrative=f"{entity.name} attempted {action_name} without a target.",
            )
            return

        # Some domain modules (e.g. Wordle / Hangman / Sudoku Duel) are
        # sealed-tick deduction races and must suppress public chat /
        # speech / action parameters between competitors. We check that
        # intent ONCE here and apply the flag downstream that would
        # otherwise echo these fields.
        #
        # Critically: `reasoning` is NOT shared between agents — it's
        # a UI-only field shown to the spectator (the user watching the
        # match). Suppressing it removes diagnostic insight without
        # adding any privacy. Only `speech` (the chat-panel content)
        # and `parameters` (the secret guess / letter / cell) need to
        # be wiped from broadcast events.
        suppress_chat = _action_suppresses_chat(self.state, action_name)
        evt_reasoning = action_instance.reasoning   # always exposed
        evt_speech = "" if suppress_chat else action_instance.speech
        evt_parameters = {} if suppress_chat else action_instance.parameters

        # Emit action_attempted event
        attempted_data = {"parameters": evt_parameters, "reasoning": evt_reasoning, "speech": evt_speech}
        if not action_def.broadcast:
            attempted_data["visible_to"] = [
                p for p in (entity_id, action_instance.target_id) if p
            ]
        self._emit_event(
            "action_attempted",
            actor_id=entity_id,
            target_id=action_instance.target_id,
            action_name=action_name,
            data=attempted_data,
            narrative=f"{entity.name} attempts to {action_name}" +
                      (f" targeting {target.name}" if target else ""),
        )

        # Handle message actions. A non-broadcast message action is a
        # private channel — deliver to the target only, never the room.
        if action_def.message_action and not suppress_chat:
            content = (action_instance.speech or action_instance.reasoning or action_instance.parameters.get("message", ""))
            if content:
                msg = Message(
                    sender_id=entity_id,
                    sender_name=entity.name,
                    content=content,
                    message_type="broadcast" if action_def.broadcast else "direct",
                    recipient_id=action_instance.target_id,
                )
                self.state.messages.post(msg)

        # Resolve action — mirror the sequential path's calling convention
        resolution = get_resolution(action_def.resolution_archetype)

        # Build modified properties with status effect bonuses/penalties
        actor_props = dict(entity.properties)
        actor_props["_entity_id"] = entity_id
        actor_mods = self.state.status_effects.get_modifiers(entity_id)
        for prop, mod_val in actor_mods.items():
            if prop in actor_props and isinstance(actor_props[prop], (int, float)):
                actor_props[prop] = actor_props[prop] + mod_val

        target_props = None
        if target:
            target_props = dict(target.properties)
            target_mods = self.state.status_effects.get_modifiers(target.id)
            for prop, mod_val in target_mods.items():
                if prop in target_props and isinstance(target_props[prop], (int, float)):
                    target_props[prop] = target_props[prop] + mod_val

        # Domain module hooks: modify properties/params before resolution
        resolution_params = dict(action_def.resolution_params) if action_def.resolution_params else {}
        resolution_params["action_name"] = action_instance.action_name  # So resolution can infer direction
        if self.state.domain_modules:
            actor_props, target_props = self.state.domain_modules.modify_resolution(
                actor_props, target_props, action_def, self.state,
            )
            dm_keys = [k for k in actor_props if k.startswith("_")]
            for k in dm_keys:
                resolution_params[k[1:]] = actor_props.pop(k)

        result = resolution.resolve(
            actor_properties=actor_props,
            target_properties=target_props,
            params=resolution_params,
            action_params=action_instance.parameters,
            rng=self._rng,
        )

        # Apply effects
        effects = action_def.effects_on_success if result.success else action_def.effects_on_failure
        state_changes = self._apply_effects(effects, entity, target, action_instance.parameters, result)

        # Notify domain modules
        if self.state.domain_modules:
            # Make action parameters available to the domain module (e.g. bet amount).
            if result and getattr(result, "details", None) is not None:
                result.details["_action_params"] = dict(action_instance.parameters or {})
                # Forward action_instance's target_id and speech so domain
                # modules see what the LLM chose (not just the resolved
                # target entity, which may be filtered). Speech is wiped
                # for chat-suppressed modules — even the action_resolved
                # event's `details` should carry no in-character text.
                result.details["target_id"] = action_instance.target_id
                result.details["speech"] = "" if suppress_chat else action_instance.speech
            domain_changes = self.state.domain_modules.post_resolution(
                entity_id, action_instance.action_name, result.success, result, self.state,
            )
            # Scrub parameter leak from result.details so the downstream
            # action_resolved event broadcast doesn't carry the secret
            # input (e.g. the guess word) for chat-suppressed modules.
            if suppress_chat and getattr(result, "details", None) is not None:
                result.details.pop("_action_params", None)
            # Mirror the sequential path: surface each domain change as an
            # event so the live UI + replay see board updates, captures,
            # check/checkmate, etc. Without this, games like chess emit
            # internal state updates that never reach the viz.
            for change in domain_changes or []:
                # Same shape as the tick_all path: prefer event_type,
                # fall back to type, default to domain_update.
                evt_type = (
                    change.get("event_type")
                    or change.get("type")
                    or "domain_update"
                )
                payload = change.get("data") if isinstance(change.get("data"), dict) else change
                self._emit_event(
                    evt_type,
                    actor_id=change.get("actor_id") or entity_id,
                    target_id=change.get("target_id"),
                    action_name=action_instance.action_name,
                    data=payload,
                    narrative=change.get("narrative", f"Domain update: {change.get('type', 'unknown')}"),
                )

        # Generate narrative
        narrative = ""
        if self.narrative_fn:
            try:
                narrative = self.narrative_fn(entity, target, action_def, action_instance, result, state_changes)
            except Exception:
                narrative = f"{entity.name} {'successfully' if result.success else 'failed to'} {action_name}"
        if not narrative:
            narrative = result.narrative or f"{entity.name} {'successfully' if result.success else 'failed to'} {action_name}"

        # Notify brain
        if self.outcome_fn:
            self.outcome_fn(entity_id, action_name, result.success, narrative, result.details)

        # Emit action_resolved event. Non-broadcast actions (hidden votes,
        # night kills) carry visible_to so every downstream consumer —
        # perception, transcripts, analysis — can restrict who sees them.
        resolved_data = {
            "success": result.success,
            "magnitude": result.magnitude,
            "details": result.details,
            "state_changes": state_changes,
        }
        if not action_def.broadcast:
            resolved_data["visible_to"] = [
                p for p in (entity_id, action_instance.target_id) if p
            ]
        self._emit_event(
            "action_resolved",
            actor_id=entity_id,
            target_id=action_instance.target_id,
            action_name=action_name,
            data=resolved_data,
            narrative=narrative,
        )

        # Surface public speech as a dedicated agent_message event ONLY
        # for actions explicitly declared as `message_action: true`.
        #
        # Previously any action could leak a `speech` field into the
        # chat panel, which meant LLMs spontaneously chatted during
        # non-social games (Battleship, Connect Five, …) by populating
        # the optional speech field on their move actions. The rule now
        # is strict: if the schema doesn't mark the action as a message
        # action, no chat event is emitted — period. Domain modules can
        # additionally opt out via `suppress_chat = True`.
        spoken_pub = ""
        if action_def.message_action and not suppress_chat:
            spoken_pub = (action_instance.speech or "").strip()
            if not spoken_pub:
                spoken_pub = (action_instance.reasoning or "").strip()
        if spoken_pub:
            target_name = target.name if target else None
            self._emit_event(
                "agent_message",
                actor_id=entity_id,
                target_id=action_instance.target_id,
                action_name=action_name,
                data={
                    "sender_name": entity.name,
                    "content": spoken_pub,
                    "message_type": action_instance.parameters.get("message_type", "broadcast"),
                    "recipient_name": target_name,
                },
                narrative=f'{entity.name} says: "{spoken_pub[:200]}"',
            )

        # Track action for crowd agent promotion/demotion
        if self.state.crowd_agents:
            self.state.crowd_agents.record_action(entity_id, action_instance.action_name)

        # Record action history
        round_num = self.state.temporal.current_round
        self.state.action_history.record(entity_id, action_instance.action_name, result.success, round_num)

    def _run_agent_turn(self, entity_id: str):
        """Execute a single agent's turn: perceive -> decide -> resolve -> apply."""
        try:
            self._run_agent_turn_inner(entity_id)
        except TypeError as e:
            import traceback
            logger.error(f"TypeError in agent turn for {entity_id}:\n{traceback.format_exc()}")
            raise

    def _run_agent_turn_inner(self, entity_id: str):
        """Inner agent turn logic."""
        entity = self.state.get_entity(entity_id)
        if not entity:
            return

        # 0. Tick status effects (damage-over-time, healing, etc.)
        round_num = self.state.temporal.current_round
        tick_effects = self.state.status_effects.tick(entity_id, round_num)
        if tick_effects:
            self._apply_effects(tick_effects, entity, None, {}, None)

        # 0b. Tick location effects
        entity_loc = self.state.locations.get(entity_id)
        if entity_loc:
            loc_changes = self.state.location_properties.apply_tick_effects(
                entity, entity_loc, self.state.entity_types,
            )
            for lc in loc_changes:
                self._emit_event(
                    "location_effect",
                    actor_id=entity_id,
                    data=lc,
                    narrative=f"{entity.name} affected by {entity_loc}: {lc.get('field')} {lc.get('effect')} {(lc.get('new') or 0) - (lc.get('old') or 0):.1f}",
                )

        # 0c. Check for active sequence
        active_seq = self.state.sequences.get_active(entity_id)

        # 1. Build perception
        active_events = None
        if self.world_event_engine:
            active_events = [
                {"name": ae.definition.name, "description": ae.definition.description,
                 "remaining_rounds": ae.remaining_rounds}
                for ae in self.world_event_engine.get_active_events()
            ]
        perception = self._perception_builder.build_perception(
            observer_id=entity_id,
            observer_type=entity.entity_type,
            rules=self.state.visibility_rules,
            entities=self.state.entities,
            entity_types=self.state.entity_types,
            resources=self.state.resources,
            relations=self.state.relations,
            spatial=self.state.spatial,
            temporal=self.state.temporal,
            active_world_events=active_events,
            faction_manager=self.state.factions,
        )

        # Add sequence info to perception
        if active_seq:
            perception["current_sequence"] = {
                "action": active_seq.action_name,
                "progress": active_seq.rounds_completed,
                "total": active_seq.total_rounds,
            }

        # Add incoming messages to perception
        entity_faction = self.state.factions.get_entity_faction(entity_id)
        incoming = self.state.messages.get_for_entity(entity_id, entity_faction)
        if incoming:
            perception["incoming_messages"] = [
                {
                    "sender": m.sender_name,
                    "sender_id": m.sender_id,
                    "content": m.content,
                    "type": m.message_type,
                }
                for m in incoming
            ]
            # Route info_payloads from messages into the agent's world model
            for msg in incoming:
                if msg.info_payload:
                    wm = self.state.get_or_create_world_model(entity_id)
                    wm.receive_info(msg.sender_id, msg.info_payload, round_num)

        # Add location info to perception
        if entity_loc:
            loc_def = self.state.location_properties.get(entity_loc)
            if loc_def:
                perception["location_info"] = {
                    "id": loc_def.id,
                    "name": loc_def.name,
                    "description": loc_def.description,
                    "properties": dict(loc_def.properties),
                }

        # Add inventory to perception
        inv_items = self.state.inventory.get_inventory(entity_id)
        if inv_items:
            perception["inventory"] = [
                {"id": it.id, "name": it.name, "type": it.item_type, "quantity": it.quantity}
                for it in inv_items
            ]
        if entity_loc:
            ground = self.state.inventory.get_ground_items(entity_loc)
            if ground:
                perception["ground_items"] = [
                    {"id": it.id, "name": it.name, "type": it.item_type, "quantity": it.quantity}
                    for it in ground
                ]

        # Add goals to perception
        active_goals = self.state.goals.get_active_goals(entity_id)
        if active_goals:
            perception["goals"] = [
                {"id": g.id, "description": g.description, "priority": g.priority}
                for g in active_goals
            ]

        # Add skills to perception
        all_skills = self.state.skills.get_all_skills(entity_id)
        if all_skills:
            perception["skills"] = {
                name: {"level": entry.level, "xp": entry.xp}
                for name, entry in all_skills.items()
            }

        # Add available recipes to perception
        available_recipes = self.state.recipes.get_available(entity_id, self.state)
        if available_recipes:
            perception["available_recipes"] = available_recipes

        # 1b. Trend analysis — store snapshot and inject trends into perception
        self._trend_analyzer.update(entity_id, perception)
        trends = self._trend_analyzer.get_trends(entity_id)
        if trends:
            perception["trends"] = trends

        # 1c. Update persistent world model from current perception
        world_model = self.state.get_or_create_world_model(entity_id)
        world_model.update_from_perception(perception, round_num)
        wm_summary = world_model.summarize_for_prompt(perception)
        if wm_summary:
            perception["world_model_memories"] = wm_summary

        # 1d. Add active negotiations / agreements / auctions to perception
        neg_info = self.state.negotiations.get_active_for_entity(entity_id, current_round=round_num)
        if any(neg_info.values()):
            perception["active_negotiations"] = neg_info

        # 1e. Add plan & theory-of-mind to perception
        plan_info = self.state.plans.summarize_for_prompt(entity_id)
        if plan_info.get("plan"):
            perception["current_plan"] = plan_info["plan"]
        if plan_info.get("agent_models"):
            perception["agent_models"] = plan_info["agent_models"]

        # 1g. Add external data from connectors
        if self.state.connectors:
            ext_data = self.state.connectors.inject_context(entity_id)
            if ext_data:
                perception["external_data"] = ext_data

        # 1h. Add domain-specific perception data
        if self.state.domain_modules:
            domain_data = self.state.domain_modules.get_perception_data(entity_id, self.state)
            if domain_data:
                perception["domain_data"] = domain_data

            # Apply domain visibility overrides (e.g., anonymous markets)
            for _mod_name, _module in self.state.domain_modules._modules.items():
                if hasattr(_module, 'get_visibility_overrides'):
                    overrides = _module.get_visibility_overrides()
                    if overrides.get("hide_agent_identities"):
                        # Remove individual agent details — agents only see aggregate market data
                        perception["visible_entities"] = []
                        perception.pop("agent_models", None)

        # 1i. Add cognitive state to perception
        if self.state.cognition:
            overlay = self.state.cognition.get_prompt_overlay(entity_id)
            if overlay:
                perception["cognitive_state"] = overlay

        # 1j. Add social data to perception
        if self.state.social:
            social_data = self.state.social.get_perception_data(entity_id)
            if social_data:
                perception["social_data"] = social_data

        # 1k. Add crowd trends to perception
        if self.state.crowd_agents:
            crowd_trends = self.state.crowd_agents.get_crowd_trends()
            if crowd_trends:
                perception["crowd_trends"] = crowd_trends

        # 1f. Validate plan against currently available actions
        valid_actions = self.state.get_valid_actions(entity_id)
        if self.state.domain_modules:
            valid_actions = self.state.domain_modules.filter_valid_actions(
                entity_id, valid_actions, self.state,
            )
        if valid_actions:
            self.state.plans.validate_plan(entity_id, valid_actions, round_num)

        # 2. Get valid actions
        if not valid_actions:
            return

        # 2b. Apply bounded rationality to filter perception
        if self.state.cognition:
            perception = self.state.cognition.process_perception_for(entity_id, perception)

        # 3. Get decision from callback (or crowd behavior or human takeover)
        if self.decision_fn is None:
            return

        # 3a. Check if this is a crowd agent (skip LLM entirely)
        action_instance = None
        if self.state.crowd_agents and self.state.crowd_agents.is_crowd(entity_id):
            behavior = self.state.crowd_agents.get_behavior(entity_id)
            if behavior:
                # Crowd behaviors mutate state (buy/sell); pin their RNG to the
                # sim seed so two same-seed runs produce identical crowd trades.
                behavior._rng = self._rng
                # Build a lightweight perception summary for the crowd behavior
                own_props = entity.properties if entity else {}
                crowd_perception = {
                    "own_properties": own_props,
                    "crowd_trends": perception.get("crowd_trends", {}),
                }
                # Inject domain module perception data (e.g. prediction market prices)
                if self.state.domain_modules:
                    for mod_name, module in self.state.domain_modules._modules.items():
                        dm_data = module.get_perception_data(entity_id, self.state)
                        if dm_data:
                            crowd_perception[f"domain_{mod_name}"] = dm_data
                decision = behavior.decide(entity_id, crowd_perception, valid_actions)
                if decision:
                    action_instance = ActionInstance(
                        action_name=decision["action_name"],
                        actor_id=entity_id,
                        target_id=decision.get("target_id"),
                        parameters=decision.get("parameters", {}),
                        reasoning=decision.get("reasoning", ""),
                        speech=(decision.get("speech") or "").strip(),
                    )

        # 3b. Check if this agent is under human takeover via SimController
        if action_instance is None and self.state.controller:
            takeover = self.state.controller.is_taken_over(entity_id)
            if takeover and takeover._human_decision_fn:
                action_instance = takeover._human_decision_fn(entity_id, perception, valid_actions)

        # 3c. Fall back to LLM decision_fn
        if action_instance is None:
            try:
                action_instance = self.decision_fn(entity_id, perception, valid_actions)
            except Exception as e:
                logger.error(f"Sequential decision failed for {entity_id}: {e}")
                self._emit_event(
                    "decision_error",
                    actor_id=entity_id,
                    data={"error": str(e)},
                    narrative=f"Decision error for {entity_id}: {e}",
                )
                action_instance = None
        if action_instance is None:
            # If in a non-interruptible sequence, force continuation
            if active_seq:
                seq_def = self.state.action_definitions.get(active_seq.action_name)
                if seq_def and not seq_def.interruptible:
                    action_instance = ActionInstance(
                        action_name=active_seq.action_name,
                        actor_id=entity_id,
                        target_id=active_seq.target_id,
                        parameters=active_seq.parameters,
                    )
                else:
                    # Interruptible sequence cancelled by passing
                    self.state.sequences.cancel(entity_id)
                    self._emit_event(
                        "sequence_cancelled",
                        actor_id=entity_id,
                        action_name=active_seq.action_name,
                        narrative=f"{entity.name} abandons {active_seq.action_name}.",
                    )

            if action_instance is None:
                self._emit_event(
                    "action_skipped",
                    actor_id=entity_id,
                    narrative=f"{entity.name} passes.",
                )
                return

        # 3b. Handle sequence continuation or interruption
        if active_seq:
            if action_instance.action_name == active_seq.action_name:
                # Continue the sequence
                completed = self.state.sequences.advance(entity_id)
                if not completed:
                    # Still in progress — emit progress event, skip resolution
                    seq = self.state.sequences.get_active(entity_id)
                    self._emit_event(
                        "sequence_progress",
                        actor_id=entity_id,
                        action_name=active_seq.action_name,
                        data={
                            "rounds_completed": seq.rounds_completed if seq else active_seq.rounds_completed + 1,
                            "total_rounds": active_seq.total_rounds,
                        },
                        narrative=f"{entity.name} continues {active_seq.action_name} ({active_seq.rounds_completed + 1}/{active_seq.total_rounds}).",
                    )
                    return
                else:
                    # Sequence complete — emit completion event and proceed to normal resolution
                    self._emit_event(
                        "sequence_completed",
                        actor_id=entity_id,
                        action_name=active_seq.action_name,
                        narrative=f"{entity.name} completes {active_seq.action_name}!",
                    )
                    # Fall through to normal resolution below
            else:
                # Chose a different action
                seq_def = self.state.action_definitions.get(active_seq.action_name)
                if seq_def and not seq_def.interruptible:
                    # Force continuation — override the agent's choice
                    action_instance = ActionInstance(
                        action_name=active_seq.action_name,
                        actor_id=entity_id,
                        target_id=active_seq.target_id,
                        parameters=active_seq.parameters,
                    )
                    completed = self.state.sequences.advance(entity_id)
                    if not completed:
                        seq = self.state.sequences.get_active(entity_id)
                        self._emit_event(
                            "sequence_progress",
                            actor_id=entity_id,
                            action_name=active_seq.action_name,
                            data={
                                "rounds_completed": seq.rounds_completed if seq else active_seq.rounds_completed + 1,
                                "total_rounds": active_seq.total_rounds,
                            },
                            narrative=f"{entity.name} must continue {active_seq.action_name} ({active_seq.rounds_completed + 1}/{active_seq.total_rounds}).",
                        )
                        return
                    else:
                        self._emit_event(
                            "sequence_completed",
                            actor_id=entity_id,
                            action_name=active_seq.action_name,
                            narrative=f"{entity.name} completes {active_seq.action_name}!",
                        )
                else:
                    # Interruptible — cancel and proceed with new action
                    self.state.sequences.cancel(entity_id)
                    self._emit_event(
                        "sequence_cancelled",
                        actor_id=entity_id,
                        action_name=active_seq.action_name,
                        narrative=f"{entity.name} interrupts {active_seq.action_name}.",
                    )

        # 4. Validate the chosen action (normalizes case/typo in place and
        # emits action_corrected / action_failed — same logic as the
        # parallel resolve path).
        action_def = self._resolve_action_def(entity, action_instance)
        if action_def is None:
            return

        # 4b. Check if this is a new multi-round sequence start
        if action_def.sequence_rounds > 0 and not active_seq:
            # Start a new sequence — don't resolve yet
            self.state.sequences.start(
                entity_id=entity_id,
                action_name=action_instance.action_name,
                target_id=action_instance.target_id,
                parameters=action_instance.parameters,
                total_rounds=action_def.sequence_rounds,
                round_num=round_num,
            )
            self._emit_event(
                "sequence_started",
                actor_id=entity_id,
                action_name=action_instance.action_name,
                data={"total_rounds": action_def.sequence_rounds},
                narrative=f"{entity.name} begins {action_instance.action_name} (1/{action_def.sequence_rounds} rounds).",
            )
            return

        # See the docstring in the parallel resolve path: chat-suppressed
        # modules (Wordle Duel) wipe speech/reasoning from every event
        # downstream, so opponents see no in-character text at all.
        # Reasoning is UI-only (spectator view), never fed to other
        # agents — keep it visible. Speech + parameters are the actual
        # secret-leak channels and stay suppressed.
        suppress_chat = _action_suppresses_chat(self.state, action_instance.action_name)
        evt_reasoning = action_instance.reasoning   # always exposed
        evt_speech = "" if suppress_chat else action_instance.speech
        evt_parameters = {} if suppress_chat else action_instance.parameters

        attempted_data = {"parameters": evt_parameters, "reasoning": evt_reasoning, "speech": evt_speech}
        _adef = self.state.action_definitions.get(action_instance.action_name)
        if _adef is not None and not _adef.broadcast:
            attempted_data["visible_to"] = [
                p for p in (entity_id, action_instance.target_id) if p
            ]
        self._emit_event(
            "action_attempted",
            actor_id=entity_id,
            target_id=action_instance.target_id,
            action_name=action_instance.action_name,
            data=attempted_data,
            narrative=f"{entity.name} attempts {action_instance.action_name}.",
        )

        # 5. Resolve (apply status effect modifiers to properties for resolution)
        target = self.state.get_entity(action_instance.target_id) if action_instance.target_id else None

        # 5a. Domain module validation (e.g. budget checks for prediction markets)
        if self.state.domain_modules:
            validation_error = self.state.domain_modules.validate_action(
                action_instance.action_name, entity, target, self.state,
            )
            if validation_error:
                self._emit_event(
                    "action_failed",
                    actor_id=entity_id,
                    target_id=action_instance.target_id,
                    action_name=action_instance.action_name,
                    data={"success": False, "details": {"reason": validation_error}},
                    narrative=f"{entity.name} cannot {action_instance.action_name}: {validation_error}",
                )
                return

        # 5b. Check target-side preconditions (IS_ADJACENT, SAME_FACTION, etc.)
        if not self._check_target_preconditions(entity, target, action_def):
            self._emit_event(
                "action_failed",
                actor_id=entity_id,
                target_id=action_instance.target_id,
                action_name=action_instance.action_name,
                narrative=f"{entity.name} cannot perform {action_instance.action_name}: preconditions not met.",
            )
            return

        archetype = get_resolution(action_def.resolution_archetype)

        # Build modified properties with status effect bonuses/penalties
        actor_props = dict(entity.properties)
        actor_props["_entity_id"] = entity_id  # Inject entity ID for domain modules
        actor_mods = self.state.status_effects.get_modifiers(entity_id)
        for prop, mod_val in actor_mods.items():
            if prop in actor_props and isinstance(actor_props[prop], (int, float)):
                actor_props[prop] = actor_props[prop] + mod_val

        target_props = None
        if target:
            target_props = dict(target.properties)
            target_mods = self.state.status_effects.get_modifiers(target.id)
            for prop, mod_val in target_mods.items():
                if prop in target_props and isinstance(target_props[prop], (int, float)):
                    target_props[prop] = target_props[prop] + mod_val

        # Domain module hooks: modify properties/params before resolution
        resolution_params = dict(action_def.resolution_params) if action_def.resolution_params else {}
        resolution_params["action_name"] = action_instance.action_name  # So resolution can infer direction
        if self.state.domain_modules:
            actor_props, target_props = self.state.domain_modules.modify_resolution(
                actor_props, target_props, action_def, self.state,
            )
            # Domain modules can inject params via actor_props prefixed with "_"
            dm_keys = [k for k in actor_props if k.startswith("_")]
            for k in dm_keys:
                resolution_params[k[1:]] = actor_props.pop(k)

        result = archetype.resolve(
            actor_properties=actor_props,
            target_properties=target_props,
            params=resolution_params,
            action_params=action_instance.parameters,
            rng=self._rng,
        )

        # 6. Apply effects (with partial success support)
        if result.success:
            effects = action_def.effects_on_success
        elif result.partial and action_def.effects_on_partial:
            effects = action_def.effects_on_partial
        else:
            effects = action_def.effects_on_failure
        state_changes = self._apply_effects(effects, entity, target, action_instance.parameters, result)

        # 7. Generate narrative (rich if narrative_fn provided, else mechanical fallback)
        if self.narrative_fn:
            narrative = self.narrative_fn(entity, target, action_def, action_instance, result, state_changes)
        else:
            narrative = result.narrative

        resolved_data = {
            "success": result.success,
            "magnitude": result.magnitude,
            "state_changes": state_changes,
            "details": result.details,
        }
        if not action_def.broadcast:
            resolved_data["visible_to"] = [
                p for p in (entity_id, action_instance.target_id) if p
            ]
        self._emit_event(
            "action_resolved",
            actor_id=entity_id,
            target_id=action_instance.target_id,
            action_name=action_instance.action_name,
            data=resolved_data,
            narrative=narrative,
        )

        # 7a. Domain module post-resolution hooks (update pools, positions, etc.)
        if self.state.domain_modules:
            if result and getattr(result, "details", None) is not None:
                result.details["_action_params"] = dict(action_instance.parameters or {})
                # Forward action_instance's target_id and speech so domain
                # modules see what the LLM chose (not just the resolved
                # target entity, which may be filtered). Speech wiped for
                # chat-suppressed modules.
                result.details["target_id"] = action_instance.target_id
                result.details["speech"] = "" if suppress_chat else action_instance.speech
            domain_changes = self.state.domain_modules.post_resolution(
                actor_id=entity_id,
                action_name=action_instance.action_name,
                success=result.success,
                result=result,
                state=self.state,
            )
            if suppress_chat and getattr(result, "details", None) is not None:
                result.details.pop("_action_params", None)
            for change in domain_changes:
                evt_type = (
                    change.get("event_type")
                    or change.get("type")
                    or "domain_update"
                )
                payload = change.get("data") if isinstance(change.get("data"), dict) else change
                self._emit_event(
                    evt_type,
                    actor_id=change.get("actor_id") or entity_id,
                    target_id=change.get("target_id"),
                    action_name=action_instance.action_name,
                    data=payload,
                    narrative=change.get("narrative", f"Domain update: {change.get('type', 'unknown')}"),
                )

        # 7b. Trigger emotions based on action outcome
        if self.state.cognition:
            if result.success:
                self.state.cognition.trigger_emotion_for(entity_id, "joy", 0.2 * result.success_degree)
            else:
                self.state.cognition.trigger_emotion_for(entity_id, "anger", 0.1)
                self.state.cognition.trigger_emotion_for(entity_id, "fear", 0.05)

        # 7c. Track action for crowd agent promotion/demotion
        if self.state.crowd_agents:
            self.state.crowd_agents.record_action(entity_id, action_instance.action_name)

        # 8. Record action and process chains
        self.state.action_history.record(entity_id, action_instance.action_name, result.success, round_num)
        if result.success:
            if action_def.unlocks_actions:
                self.state.action_history.unlock(entity_id, action_def.unlocks_actions)
            if action_def.locks_actions:
                self.state.action_history.lock(entity_id, action_def.locks_actions)
        if action_def.cooldown_rounds > 0:
            self.state.action_history.set_cooldown(
                entity_id, action_instance.action_name,
                round_num + action_def.cooldown_rounds + 1,
            )

        # 8b. Advance plan if the agent took the planned action
        active_plan = self.state.plans.get_active_plan(entity_id)
        if active_plan:
            current_step = active_plan.get_current_step()
            if current_step and current_step.action == action_instance.action_name:
                outcome_str = narrative[:120] if narrative else ""
                plan_done = self.state.plans.advance_plan(entity_id, result.success, outcome_str)
                if plan_done:
                    self._emit_event(
                        "plan_completed",
                        actor_id=entity_id,
                        data={"plan_goal": active_plan.goal_description},
                        narrative=f"{entity.name} completed their plan: {active_plan.goal_description}",
                    )

        # 8c. Update theory-of-mind for all agents who can observe this action
        for observer in self.state.get_agent_entities():
            if observer.id == entity_id:
                continue
            self.state.plans.update_agent_model(
                observer_id=observer.id,
                target_id=entity_id,
                target_name=entity.name,
                action_name=action_instance.action_name,
                action_target_id=action_instance.target_id,
                success=result.success,
                round_num=round_num,
            )

        # 9. Notify outcome callback (for agent feedback loop)
        if self.outcome_fn:
            # Include target_id in details so the callback can track entity interactions
            outcome_details = dict(result.details) if result.details else {}
            if action_instance.target_id and "target_id" not in outcome_details:
                outcome_details["target_id"] = action_instance.target_id
            self.outcome_fn(entity_id, action_instance.action_name, result.success, narrative, outcome_details)

        # 10. Post message / speech.
        # Chat is strictly opt-in: only actions explicitly declared as
        # `message_action: true` can produce an agent_message event. This
        # prevents LLMs from spontaneously chatting during non-social
        # games (Battleship, Connect Five, …) by filling the optional
        # `speech` field on move actions. Domain modules marked
        # chat-suppressed (e.g. Wordle Duel) skip this too.
        # Reasoning is PRIVATE inner thought and never goes to chat
        # for non-message actions.
        spoken = ""
        if action_def.message_action and not suppress_chat:
            spoken = (action_instance.speech or "").strip()
            if not spoken:
                spoken = (action_instance.reasoning or "").strip()

        if spoken:
            msg_type = action_instance.parameters.get("message_type", "broadcast")
            if action_def.message_action:
                msg = Message(
                    sender_id=entity_id,
                    sender_name=entity.name,
                    content=spoken,
                    message_type=msg_type,
                    round_sent=round_num,
                )
                if msg_type == "direct":
                    msg.recipient_id = action_instance.parameters.get("recipient_id") or action_instance.target_id
                elif msg_type == "faction":
                    msg.recipient_faction = action_instance.parameters.get("recipient_faction") or self.state.factions.get_entity_faction(entity_id)
                self.state.messages.post(msg)

            target_name = target.name if target else None
            self._emit_event(
                "agent_message",
                actor_id=entity_id,
                target_id=action_instance.target_id,
                action_name=action_instance.action_name,
                data={
                    "sender_name": entity.name,
                    "content": spoken,
                    "message_type": msg_type,
                    "recipient_name": target_name,
                },
                narrative=f'{entity.name} says: "{spoken[:200]}"',
            )

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
        """Evaluate a CONDITIONAL effect's `if` / `if_expr` clause.

        Supports two forms:
          { "if": { subject, field?, operator, value, check_type? } }
            — uses the existing EffectCondition machinery.
          { "if_expr": "<expression>" }
            — resolves the expression; truthy ⇒ branch fires.
          { "if_compare": [lhs, "op", rhs] }
            — both sides may be expressions; supports == != > >= < <=
        """
        if not isinstance(spec, dict):
            return False

        if "if_expr" in spec:
            return bool(resolve_val(spec["if_expr"]))

        if "if_compare" in spec:
            cmp = spec["if_compare"]
            if not (isinstance(cmp, (list, tuple)) and len(cmp) == 3):
                return False
            lhs = resolve_val(cmp[0])
            op = str(cmp[1])
            rhs = resolve_val(cmp[2])
            try:
                if op == "==": return lhs == rhs
                if op == "!=": return lhs != rhs
                if op == ">":  return lhs > rhs
                if op == ">=": return lhs >= rhs
                if op == "<":  return lhs < rhs
                if op == "<=": return lhs <= rhs
                if op == "in":  return lhs in rhs
                if op == "not_in": return lhs not in rhs
            except TypeError:
                return False
            return False

        if "if" in spec and isinstance(spec["if"], dict):
            from ..action import EffectCondition
            d = spec["if"]
            try:
                cond = EffectCondition(
                    subject=str(d.get("subject", "actor")),
                    field=d.get("field"),
                    operator=str(d.get("operator", "eq")),
                    value=resolve_val(d.get("value")),
                    check_type=str(d.get("check_type", "property")),
                )
            except Exception:
                return False
            return self._evaluate_effect_condition(cond, actor, target, params)

        return False

    def _evaluate_world_condition(self, spec: dict) -> bool:
        """Generic world-state condition evaluator shared by
        `cooperative_win` / `cooperative_loss` / `world_property_threshold`.

        Supports the same clause forms as CONDITIONAL effects:
          `if_expr`, `if_compare: [lhs, op, rhs]`, or
          `expr` + `operator` + `value` shorthand.
        """
        from ..effects import resolve_expression
        if not isinstance(spec, dict):
            return False
        if "if_expr" in spec:
            return bool(resolve_expression(spec["if_expr"], state=self.state, rng=self._rng))
        if "if_compare" in spec:
            cmp = spec["if_compare"]
            if not (isinstance(cmp, (list, tuple)) and len(cmp) == 3):
                return False
            lhs = resolve_expression(cmp[0], state=self.state, rng=self._rng)
            op = str(cmp[1])
            rhs = resolve_expression(cmp[2], state=self.state, rng=self._rng)
            try:
                if op == "==": return lhs == rhs
                if op == "!=": return lhs != rhs
                if op == ">":  return lhs > rhs
                if op == ">=": return lhs >= rhs
                if op == "<":  return lhs < rhs
                if op == "<=": return lhs <= rhs
                if op == "in":  return lhs in rhs
            except TypeError:
                return False
            return False
        if "expr" in spec:
            lhs = resolve_expression(spec["expr"], state=self.state, rng=self._rng)
            op = spec.get("operator", "gte")
            rhs = spec.get("value", 0)
            try:
                if op == "gte": return lhs >= rhs
                if op == "lte": return lhs <= rhs
                if op == "gt":  return lhs > rhs
                if op == "lt":  return lhs < rhs
                if op == "eq":  return lhs == rhs
                if op == "neq": return lhs != rhs
            except TypeError:
                return False
        return False

    def _resolve_multi_target(self, target_token: str, actor, target):
        """Expand a multi-target token into a list of entities.

        Tokens (Tier-1 EffectDSL):
          "all"          — every alive agent
          "all_others"   — every alive agent except the actor
          "role:X"       — every entity whose entity_type == X (alive)
          "faction:Y"    — every entity in faction Y (alive)

        Returns [] if nothing matches. Single-target tokens
        (`actor`, `target`, or a specific id) should NOT be routed here.
        """
        if target_token == "all":
            return [e for e in self.state.get_agent_entities() if e.alive]
        if target_token == "all_others":
            actor_id = actor.id if actor else None
            return [
                e for e in self.state.get_agent_entities()
                if e.alive and e.id != actor_id
            ]
        if target_token.startswith("role:"):
            wanted = target_token.split(":", 1)[1]
            return [
                e for e in self.state.get_agent_entities()
                if e.alive and getattr(e, "entity_type", None) == wanted
            ]
        if target_token.startswith("faction:"):
            wanted = target_token.split(":", 1)[1]
            mgr = self.state.factions
            if mgr is None:
                return []
            return [
                e for e in self.state.get_agent_entities()
                if e.alive and mgr.get_entity_faction(e.id) == wanted
            ]
        return []

    def _evaluate_effect_condition(self, condition, actor, target, params=None) -> bool:
        """Evaluate a runtime condition for a conditional effect.

        ``params`` is threaded so expressions inside the condition can
        reference effect-time variables — critical for ``for_each``
        sub-effects where the iteration variable lives in
        ``$params.<as>``. Without this, conditions referencing
        ``$params.X`` always evaluated False.
        """
        # Modern: full-expression form takes precedence over the legacy
        # subject/operator/field structure when set.
        expr = getattr(condition, "expr", None)
        if expr:
            from ..predicates import evaluate as _predicate_eval
            return _predicate_eval(
                expr,
                actor=actor, target=target,
                params=params or {}, state=self.state, rng=self._rng,
            )

        # Resolve the subject entity
        if condition.subject == "actor":
            ent = actor
        elif condition.subject == "target":
            ent = target
        else:
            ent = self.state.get_entity(condition.subject)

        if ent is None:
            return False

        if condition.check_type == "property":
            val = ent.get(condition.field, 0) if condition.field else 0
            return self._compare(val, condition.operator, condition.value)

        elif condition.check_type == "alive":
            expected = condition.value if condition.value is not None else True
            return ent.alive == expected

        elif condition.check_type == "has_status":
            status_name = condition.value
            active = self.state.status_effects.get_active(ent.id)
            has_it = any(ase.definition.name == status_name for ase in active)
            return has_it

        logger.warning(f"Unknown effect condition check_type: '{condition.check_type}'")
        return False  # Unknown check_type: fail safe

    def _check_target_preconditions(self, actor, target, action_def) -> bool:
        """Check preconditions that require both actor and target (IS_ADJACENT, faction checks)."""
        from ..action import Operator
        for pc in action_def.preconditions:
            # Modern expression form supersedes the legacy operator switch.
            # If it referenced $target, the unified evaluator handles it.
            expr = getattr(pc, "expr", None)
            if expr:
                from ..predicates import evaluate as _predicate_eval
                if not _predicate_eval(expr, actor=actor, target=target, state=self.state, rng=self._rng):
                    return False
                continue
            if pc.operator == Operator.IS_ADJACENT:
                if not target:
                    return False
                actor_loc = self.state.locations.get(actor.id) or actor.location_id
                target_loc = self.state.locations.get(target.id) or target.location_id
                if not actor_loc or not target_loc:
                    return False
                if not self.state.adjacency:
                    # No adjacency defined — consider all locations adjacent
                    continue
                from ..pathfinding import Pathfinder
                if not Pathfinder.are_adjacent(self.state.adjacency, actor_loc, target_loc):
                    return False
            elif pc.operator == Operator.SAME_FACTION:
                if not target:
                    return False
                if not self.state.factions.are_allies(actor.id, target.id):
                    return False
            elif pc.operator == Operator.DIFFERENT_FACTION:
                if not target:
                    return False
                if self.state.factions.are_allies(actor.id, target.id):
                    return False
        return True

    @staticmethod
    def _compare(val, operator: str, target_val) -> bool:
        """Compare a value using a string operator. Type-mismatched operands
        (string property vs numeric target) fail CLOSED, never crash the run."""
        try:
            return SimulationEngine._compare_inner(val, operator, target_val)
        except TypeError:
            logger.warning(
                "comparison %r %s %r raised TypeError — condition treated as False",
                val, operator, target_val,
            )
            return False

    @staticmethod
    def _compare_inner(val, operator: str, target_val) -> bool:
        if operator == "gte":
            return val >= target_val
        elif operator == "lte":
            return val <= target_val
        elif operator == "gt":
            return val > target_val
        elif operator == "lt":
            return val < target_val
        elif operator == "eq":
            return val == target_val
        elif operator == "neq" or operator == "ne":
            return val != target_val
        # Unknown operator → fail CLOSED. Returning True made a typo'd
        # condition fire the effect unconditionally — the one fail-open in
        # an otherwise fail-closed layer.
        logger.warning("unknown comparison operator %r — condition treated as False", operator)
        return False

    def _entities_snapshot(self) -> Dict[str, Any]:
        """JSON-safe {entity_id: {name,type,alive,properties[,resources]}} —
        the shape every visualization component reads."""
        import json as _json

        snap: Dict[str, Any] = {}
        for eid, entity in self.state.entities.items():
            ent: Dict[str, Any] = {
                "name": entity.name,
                "type": entity.entity_type,
                "alive": entity.alive,
                "properties": {},
            }
            for pname, pval in entity.properties.items():
                try:
                    _json.dumps(pval)
                    ent["properties"][pname] = pval
                except (TypeError, ValueError):
                    ent["properties"][pname] = str(pval)
            resources = {}
            for res_name, pool in self.state.resources.items():
                amt = pool.get(eid)
                if amt is not None and amt != 0:
                    resources[res_name] = amt
            if resources:
                ent["resources"] = resources
            snap[eid] = ent
        return snap

    def _emit_event(
        self,
        event_type: str,
        actor_id: str = None,
        target_id: str = None,
        action_name: str = None,
        data: dict = None,
        narrative: str = "",
    ):
        """Emit an event to the transcript and optional streaming callback."""
        # Safely get current phase name (index may be past end after advance_phase)
        temporal = self.state.temporal
        if temporal.phases and 0 <= temporal.current_phase_index < len(temporal.phases):
            phase_name = temporal.phases[temporal.current_phase_index].name
        elif temporal.phases:
            phase_name = temporal.phases[-1].name  # Use last phase as fallback
        else:
            phase_name = ""

        event = SimEvent(
            event_type=event_type,
            round_number=self.state.temporal.current_round,
            phase=phase_name,
            actor_id=actor_id,
            target_id=target_id,
            action_name=action_name,
            data=data or {},
            narrative=narrative,
        )
        self.state.event_log.emit(event)

        # Stream to real-time callback if set
        if self.on_event:
            try:
                self.on_event(event.to_dict())
            except Exception as e:
                logger.debug(f"on_event callback error (non-fatal): {e}")
            # Live-state contract for visualizations: entity properties +
            # resources after every resolved action / phase step.
            if self.emit_state_snapshots and event_type in ("action_resolved", "phase_handler"):
                try:
                    self.on_event({
                        "event_type": "state_snapshot",
                        "round_number": event.round_number,
                        "phase": event.phase,
                        "actor_id": None,
                        "target_id": None,
                        "action_name": None,
                        "data": {"entities": self._entities_snapshot()},
                        "narrative": "",
                        "timestamp": "",
                    })
                except Exception as e:  # noqa: BLE001 — viz feed must never break the sim
                    logger.debug(f"state_snapshot emission error (non-fatal): {e}")

        # Tier 5a — Triggered effects. Walk schema-declared triggers
        # and fire matching ones. Guarded by `_in_trigger_cascade` to
        # prevent infinite recursion (a trigger that emits the same
        # event it listens to). Cascading triggers are allowed but
        # capped at MAX_DEPTH.
        triggers = getattr(self, "triggers", None)
        depth = getattr(self._cascade_tls, "depth", 0)
        if triggers and event_type and depth < 4:
            try:
                matching = triggers.matching(
                    event_type, data or {}, actor_id,
                    self.state.temporal.current_round,
                )
                if matching:
                    self._cascade_tls.depth = depth + 1
                    try:
                        for spec in matching:
                            self._fire_trigger(spec, actor_id, target_id, data or {})
                    finally:
                        self._cascade_tls.depth = depth
            except Exception:
                logger.exception("trigger evaluation failed for event %s", event_type)

    def _fire_trigger(
        self,
        spec: "TriggerSpec",
        actor_id: Optional[str],
        target_id: Optional[str],
        event_data: Dict[str, Any],
    ) -> None:
        """Apply a triggered effect chain. The triggering event's
        actor/target/data become the new effect context — so effects
        can reference `$event.field` (mapped to params here)."""
        actor = self.state.get_entity(actor_id) if actor_id else None
        target = self.state.get_entity(target_id) if target_id else None
        effects = _coerce_effects(spec.effect)
        if not effects:
            return
        # We pass `event_data` through the `params` channel so $params.X
        # in trigger effects can read the triggering event's payload.
        # Also stash actor_id into params['actor'] for convenience.
        params = dict(event_data)
        params.setdefault("_trigger", spec.name or "")
        try:
            self._apply_effects(effects, actor, target, params, None)
            self.triggers.record_fired(
                spec, actor_id, self.state.temporal.current_round,
            )
        except Exception:
            logger.exception("trigger fire failed: %s", spec.name)

    def stop(self):
        """Stop the simulation after the current turn completes."""
        self._running = False
        self._stopped = True

    # -------------------------------------------------------------------
    # Termination condition evaluation
    # -------------------------------------------------------------------

    def _check_termination(self) -> Optional[TerminationCondition]:
        """Evaluate all termination conditions. Returns first triggered, or None."""
        for tc in self.termination_conditions:
            if self._evaluate_condition(tc):
                return tc
        return None

    def _evaluate_condition(self, tc: TerminationCondition) -> bool:
        """Evaluate a single termination condition against current state.

        First consults the registry-driven ``termination`` module which
        owns the canonical implementations of most check_types. The
        legacy ``if/elif`` chain below remains for genre-coupled
        check_types (checkmate, board_pattern) that still reference
        engine-internal state directly."""
        # Plugin/registry path — covers all_dead, resource_exhausted,
        # rounds_idle, property_threshold, all_goals_complete,
        # event_triggered, last_one_standing, first_to_score,
        # score_after_n_rounds, count_property, bankruptcy,
        # faction_win, vote_threshold, expr, compound_and/or, plus any
        # custom @termination(...) registration.
        from .. import termination as _term
        from ..registry import registry as _kreg
        check_type = (tc.check_type or "").lower()
        if (check_type in ("expr", "compound_and", "compound_or")
                or _kreg.terminations.has(check_type)
                or ((tc.params or {}).get("expr") if tc.params else None)):
            return _term.evaluate(self.state, tc, self._rng)

        # Legacy fall-through for check_types still living in-engine.
        if tc.check_type == "all_dead":
            entity_type = tc.params.get("entity_type")
            if not entity_type:
                return False
            entities = self.state.get_entities_by_type(entity_type)
            return len(entities) > 0 and all(not e.alive for e in entities)

        elif tc.check_type == "resource_exhausted":
            resource_name = tc.params.get("resource")
            if not resource_name:
                return False
            pool = self.state.resources.get(resource_name)
            if not pool:
                return False
            total = sum(pool.holdings.values()) + pool.unallocated
            return total <= 0

        elif tc.check_type == "rounds_idle":
            max_idle = tc.params.get("max_idle_rounds", 3)
            current = self.state.temporal.current_round
            # Check last N rounds for any resolved actions
            for r in range(max(1, current - max_idle + 1), current + 1):
                events = self.state.event_log.get_round(r)
                if any(e.event_type == "action_resolved" for e in events):
                    return False
            # Only trigger if we've had enough rounds
            return current >= max_idle

        elif tc.check_type == "property_threshold":
            # Single source of truth — the registered checker also handles
            # scope:"world" properties and symbolic operators (">=", …).
            from ..termination import _check_property_threshold
            return _check_property_threshold(self.state, tc.params, None)

        elif tc.check_type == "all_goals_complete":
            entity_type = tc.params.get("entity_type")
            if entity_type:
                entities = self.state.get_entities_by_type(entity_type)
            else:
                entities = self.state.get_agent_entities()
            if not entities:
                return False
            return all(
                self.state.goals.all_goals_complete(e.id)
                for e in entities if e.alive
            )

        elif tc.check_type == "event_triggered":
            # Check if a specific event type occurred at least N times
            event_type = tc.params.get("event_type")
            min_count = tc.params.get("count", 1)
            if not event_type:
                return False
            total_count = 0
            for r in range(1, self.state.temporal.current_round + 1):
                events = self.state.event_log.get_round(r)
                total_count += sum(1 for e in events if e.event_type == event_type)
            return total_count >= min_count

        elif tc.check_type == "compound_and":
            # All sub-conditions must be true
            if not tc.sub_conditions:
                return False
            return all(self._evaluate_condition(sub) for sub in tc.sub_conditions)

        elif tc.check_type == "compound_or":
            # Any sub-condition must be true
            if not tc.sub_conditions:
                return False
            return any(self._evaluate_condition(sub) for sub in tc.sub_conditions)

        # ── Phase-1 declarative win predicates ──
        # These replace ~150 lines of per-env win detection. Schema:
        #   termination_conditions:
        #     - { check_type: last_one_standing,
        #         params: { entity_type: MonopolyPlayer, exclude_when: bankrupt } }
        elif tc.check_type == "last_one_standing":
            entity_type = tc.params.get("entity_type")
            exclude_when = tc.params.get("exclude_when", "eliminated")
            if not entity_type:
                return False
            entities = self.state.get_entities_by_type(entity_type)
            survivors = [
                e for e in entities
                if e.alive and not bool(e.get(exclude_when))
            ]
            return len(survivors) == 1

        elif tc.check_type == "first_to_score":
            # First entity to reach `target` on `property` ends the game.
            # `target` accepts expressions: `$lookup(runtime, wins_needed)`,
            # `$state.X`, etc. — resolved through the standard EffectDSL
            # resolver so runtime-configured thresholds work.
            entity_type = tc.params.get("entity_type")
            prop = tc.params.get("property", "score")
            target_raw = tc.params.get("target")
            from ..effects import resolve_expression, is_expression
            target = (resolve_expression(target_raw, state=self.state, rng=self._rng)
                      if is_expression(target_raw) else target_raw)
            if not entity_type or target is None:
                return False
            try:
                target_num = float(target)
            except (TypeError, ValueError):
                return False
            entities = self.state.get_entities_by_type(entity_type)
            for e in entities:
                if not e.alive:
                    continue
                try:
                    if float(e.get(prop, 0) or 0) >= target_num:
                        return True
                except (TypeError, ValueError):
                    continue
            return False

        elif tc.check_type == "score_after_n_rounds":
            # End after N rounds — winner = highest scorer. We only test
            # the gate here; the engine picks the winner from final state.
            after = int(tc.params.get("after_rounds", 0))
            return self.state.temporal.current_round >= after

        elif tc.check_type == "board_pattern":
            # N-in-a-row on a board property. Generic enough for
            # tic-tac-toe (3-in-row on 3x3), connect-four (4-in-row on
            # 7x6), gomoku (5-in-row on 19x19).
            #
            # Params:
            #   board_key: entity property carrying a 2-D list (rows of cols)
            #   patterns: ["row_3", "col_3", "diag_3"]  (n inferred from suffix)
            #   by: "any_actor" | "specific_value"
            #   value: when by==specific_value, the cell value to match
            #   entity_type: entity whose property carries the board
            return self._evaluate_board_pattern(tc.params)

        elif tc.check_type == "vote_threshold":
            # Final-state tally — caller is responsible for setting a
            # vote-count property on entities. Triggers when the highest
            # tally exceeds `threshold_pct` of total cast votes.
            # Tie-break (Tier 2): when `require_unique_max=True`, a tie
            # at the top doesn't trigger the win unless a tie-break rule
            # is configured (`tie_break` ∈ "highest_property" | "random"
            # | "first_alphabetical").
            entity_type = tc.params.get("entity_type")
            prop = tc.params.get("vote_property", "votes_received")
            threshold_pct = float(tc.params.get("threshold_pct", 50.0))
            min_votes = int(tc.params.get("min_total_votes", 1))
            require_unique = bool(tc.params.get("require_unique_max", False))
            if not entity_type:
                return False
            entities = self.state.get_entities_by_type(entity_type)
            counts = []
            for e in entities:
                if not e.alive:
                    continue
                try:
                    counts.append(float(e.get(prop, 0) or 0))
                except (TypeError, ValueError):
                    counts.append(0.0)
            total = sum(counts)
            if total < min_votes:
                return False
            top = max(counts)
            if (top / total) * 100.0 <= threshold_pct:
                return False
            if require_unique and counts.count(top) > 1:
                # Tie at the top — needs a tie-break rule (handled in
                # _resolve_winner). The predicate still fires so the
                # game ends; the winner field will reflect the break.
                return tc.params.get("tie_break") is not None
            return True

        elif tc.check_type == "count_property":
            # 6.L — count entities whose `property` is ≥ / ≤ / == value.
            #   { check_type: count_property,
            #     params: { entity_type, property, operator, value,
            #               min_count?, max_count? } }
            # If `min_count`/`max_count` are present, triggers when the
            # number of MATCHING entities is in [min_count, max_count].
            # Otherwise, triggers when ANY entity matches.
            entity_type = tc.params.get("entity_type")
            prop = tc.params.get("property")
            operator = tc.params.get("operator", "gte")
            value = tc.params.get("value", 0)
            min_count = tc.params.get("min_count")
            max_count = tc.params.get("max_count")
            if not entity_type or not prop:
                return False
            matches = 0
            for e in self.state.get_entities_by_type(entity_type):
                if not e.alive:
                    continue
                try:
                    v = float(e.get(prop, 0) or 0)
                except (TypeError, ValueError):
                    continue
                hit = False
                if operator == "gte" and v >= value: hit = True
                elif operator == "lte" and v <= value: hit = True
                elif operator == "gt"  and v >  value: hit = True
                elif operator == "lt"  and v <  value: hit = True
                elif operator == "eq"  and v == value: hit = True
                elif operator == "neq" and v != value: hit = True
                if hit:
                    matches += 1
            if min_count is None and max_count is None:
                return matches > 0
            if min_count is not None and matches < int(min_count):
                return False
            if max_count is not None and matches > int(max_count):
                return False
            return True

        elif tc.check_type == "cooperative_win":
            # Tier 5a — Pandemic-style coop. Game ends in a SHARED win
            # when the world-level condition is met. The predicate
            # evaluates against an arbitrary expression that reads
            # world state (e.g. `$state.tables.diseases_cured == 4`).
            # Params:
            #   if_expr | if_compare | if  (same as CONDITIONAL effect)
            from ..effects import resolve_expression
            spec = tc.params or {}
            return self._evaluate_world_condition(spec)

        elif tc.check_type == "cooperative_loss":
            # Pandemic-style shared loss. Same predicate forms.
            return self._evaluate_world_condition(tc.params or {})

        elif tc.check_type == "world_property_threshold":
            # End when an arbitrary world-level expression crosses a
            # threshold. Useful for "X% of map controlled" / "the
            # rebel pool hit 100".
            #   { expr: "$lookup(map, controlled_count)",
            #     operator: "gte", value: 100 }
            return self._evaluate_world_condition(tc.params or {})

        elif tc.check_type == "checkmate":
            # 6.M — `side` is in checkmate ⇒ the OTHER side wins.
            #   { check_type: checkmate,
            #     params: { board_id, side: "white"|"black" } }
            board_id = tc.params.get("board_id")
            side_param = tc.params.get("side")
            if not self.state.domain_modules:
                return False
            from ..board_module import BoardModule
            for _m in self.state.domain_modules._modules.values():
                if not isinstance(_m, BoardModule):
                    continue
                if board_id and _m._id != board_id:
                    continue
                # If `side` not specified, check both
                sides = [side_param] if side_param else ["white", "black"]
                for s in sides:
                    if _m.is_in_checkmate(s, self.state):
                        return True
            return False

        elif tc.check_type == "faction_win":
            # 6.F — One faction wins when it's the only one with any
            # alive members remaining, OR when its alive count exceeds
            # a configurable threshold of the total alive population.
            #   { check_type: faction_win,
            #     params: { rule: "last_faction_standing" | "majority",
            #               threshold_pct: 50.0,         # for majority rule
            #               required_factions: ["mafia","town"]?  # optional
            #             } }
            rule = tc.params.get("rule", "last_faction_standing")
            threshold_pct = float(tc.params.get("threshold_pct", 50.0))
            mgr = self.state.factions
            if mgr is None:
                return False
            counts: Dict[str, int] = {}
            for e in self.state.get_agent_entities():
                if not e.alive:
                    continue
                fac = mgr.get_entity_faction(e.id)
                if not fac:
                    continue
                counts[fac] = counts.get(fac, 0) + 1
            if not counts:
                return False
            if rule == "last_faction_standing":
                surviving = [f for f, n in counts.items() if n > 0]
                return len(surviving) == 1
            if rule == "majority":
                total = sum(counts.values())
                if total == 0:
                    return False
                top = max(counts.values())
                return (top / total) * 100.0 > threshold_pct
            return False

        elif tc.check_type == "bankruptcy":
            # End when only one solvent (money > 0) player remains.
            entity_type = tc.params.get("entity_type")
            money_prop = tc.params.get("money_property", "money")
            min_money = float(tc.params.get("min_money", 0))
            if not entity_type:
                return False
            entities = self.state.get_entities_by_type(entity_type)
            solvent = [
                e for e in entities
                if e.alive and float(e.get(money_prop, 0) or 0) > min_money
            ]
            return len(solvent) == 1 and len(entities) > 1

        return False

    def _resolve_winner(self, tc) -> Dict[str, Any]:
        """For Phase-1 declarative win predicates, identify WHO won so
        the terminated event carries a winner_id. Returns {} when the
        predicate doesn't have a single-entity winner (e.g. round_limit
        timeouts) — the caller falls back to "draw"."""
        # Plugin/registry path — covers last_one_standing, first_to_score,
        # score_after_n_rounds, count_property, bankruptcy, faction_win,
        # vote_threshold, plus any custom resolver registered via
        # termination.register_winner_resolver(name, fn).
        try:
            from .. import termination as _term
            registered = _term.resolve_winner(self.state, tc, self._rng)
            if registered:
                return registered
        except Exception:
            logger.exception("registry winner resolver failed; falling back to legacy")
        try:
            params = tc.params or {}
        except Exception:
            return {}

        if tc.check_type == "last_one_standing":
            entity_type = params.get("entity_type")
            exclude_when = params.get("exclude_when", "eliminated")
            if not entity_type:
                return {}
            for e in self.state.get_entities_by_type(entity_type):
                if e.alive and not bool(e.get(exclude_when)):
                    return {"winner_id": e.id, "winner_name": e.name}
            return {}

        if tc.check_type == "first_to_score":
            entity_type = params.get("entity_type")
            prop = params.get("property", "score")
            target_raw = params.get("target")
            # Mirror _evaluate_condition: target can be a $-expression.
            from ..effects import resolve_expression, is_expression
            target = (resolve_expression(target_raw, state=self.state, rng=self._rng)
                      if is_expression(target_raw) else target_raw)
            if not entity_type or target is None:
                return {}
            try:
                target_num = float(target)
            except (TypeError, ValueError):
                return {}
            best, best_v = None, float("-inf")
            # id-sorted iteration → ties resolve to the lowest id
            # deterministically, independent of entity insertion order
            # (which can differ after a snapshot/fork rebuild).
            for e in sorted(self.state.get_entities_by_type(entity_type), key=lambda e: e.id):
                if not e.alive:
                    continue
                try:
                    v = float(e.get(prop, 0) or 0)
                except (TypeError, ValueError):
                    continue
                if v >= target_num and v > best_v:
                    best, best_v = e, v
            if best is not None:
                return {"winner_id": best.id, "winner_name": best.name,
                        "winner_score": best_v}
            return {}

        if tc.check_type == "score_after_n_rounds":
            entity_type = params.get("entity_type")
            prop = params.get("property", "score")
            if not entity_type:
                return {}
            best, best_v = None, float("-inf")
            for e in sorted(self.state.get_entities_by_type(entity_type), key=lambda e: e.id):
                try:
                    v = float(e.get(prop, 0) or 0)
                except (TypeError, ValueError):
                    continue
                if v > best_v:
                    best, best_v = e, v
            if best is not None:
                return {"winner_id": best.id, "winner_name": best.name,
                        "winner_score": best_v}
            return {}

        if tc.check_type == "vote_threshold":
            entity_type = params.get("entity_type")
            prop = params.get("vote_property", "votes_received")
            tie_break = params.get("tie_break")  # 6.G
            tie_property = params.get("tie_break_property")
            if not entity_type:
                return {}
            ranked = []
            for e in self.state.get_entities_by_type(entity_type):
                if not e.alive:
                    continue
                try:
                    v = float(e.get(prop, 0) or 0)
                except (TypeError, ValueError):
                    continue
                ranked.append((v, e))
            if not ranked:
                return {}
            # Sort by score, then id — a stable, insertion-order-independent
            # ranking (float votes make a bare score sort non-unique).
            ranked.sort(key=lambda x: (-x[0], x[1].id))
            top_v = ranked[0][0]
            # Float-tolerant top-tier membership: exact == can drop a true
            # co-leader when votes are fractional.
            top_tier = [e for v, e in ranked if abs(v - top_v) < 1e-9]
            if len(top_tier) == 1:
                best = top_tier[0]
                return {"winner_id": best.id, "winner_name": best.name,
                        "winner_votes": top_v}
            # Tie at the top
            if tie_break == "highest_property" and tie_property:
                top_tier.sort(key=lambda e: (-float(e.get(tie_property, 0) or 0), e.id))
                best = top_tier[0]
            elif tie_break == "random":
                # top_tier is already id-sorted above, so _rng.choice indexes
                # into a stable order → reproducible given the seed.
                best = self._rng.choice(top_tier)
            elif tie_break == "first_alphabetical":
                top_tier.sort(key=lambda e: e.name.lower())
                best = top_tier[0]
            else:
                return {"tied": True,
                        "tied_ids": [e.id for e in top_tier],
                        "winner_votes": top_v}
            return {"winner_id": best.id, "winner_name": best.name,
                    "winner_votes": top_v, "tie_broken_by": tie_break}

        if tc.check_type == "count_property":
            # Count-property win is collective — no single entity wins
            # unless the schema names one. Return the count for record.
            entity_type = params.get("entity_type")
            prop = params.get("property")
            if not entity_type or not prop:
                return {}
            matches = []
            for e in self.state.get_entities_by_type(entity_type):
                if not e.alive:
                    continue
                try:
                    if float(e.get(prop, 0) or 0) >= float(params.get("value", 0)):
                        matches.append(e)
                except (TypeError, ValueError):
                    continue
            if len(matches) == 1:
                m = matches[0]
                return {"winner_id": m.id, "winner_name": m.name}
            return {"match_count": len(matches),
                    "match_ids": [e.id for e in matches]}

        if tc.check_type == "cooperative_win":
            # Everyone wins together.
            return {
                "outcome": "cooperative_win",
                "winner_ids": [e.id for e in self.state.get_agent_entities() if e.alive],
                "winner_names": [e.name for e in self.state.get_agent_entities() if e.alive],
            }

        if tc.check_type == "cooperative_loss":
            return {"outcome": "cooperative_loss"}

        if tc.check_type == "world_property_threshold":
            return {"outcome": "world_threshold",
                    "condition": (tc.params or {}).get("expr") or (tc.params or {}).get("if_compare")}

        if tc.check_type == "checkmate":
            board_id = params.get("board_id")
            if not self.state.domain_modules:
                return {}
            from ..board_module import BoardModule
            for _m in self.state.domain_modules._modules.values():
                if not isinstance(_m, BoardModule):
                    continue
                if board_id and _m._id != board_id:
                    continue
                for s in ("white", "black"):
                    if _m.is_in_checkmate(s, self.state):
                        winner_side = _m._opposite_side_of(s)
                        # Find any entity of winning side to surface a winner_id
                        for ent in self.state.get_agent_entities():
                            try:
                                if ent.get(_m._side_property) == winner_side:
                                    return {"winner_id": ent.id, "winner_name": ent.name,
                                            "winning_side": winner_side,
                                            "checkmated_side": s}
                            except Exception:
                                continue
                        return {"winning_side": winner_side, "checkmated_side": s}
            return {}

        if tc.check_type == "faction_win":
            mgr = self.state.factions
            if mgr is None:
                return {}
            counts: Dict[str, List[Any]] = {}
            for e in self.state.get_agent_entities():
                if not e.alive:
                    continue
                fac = mgr.get_entity_faction(e.id)
                if not fac:
                    continue
                counts.setdefault(fac, []).append(e)
            if not counts:
                return {}
            # Winner = faction with the most alive members
            winning_faction = max(counts.keys(), key=lambda f: len(counts[f]))
            members = counts[winning_faction]
            return {
                "winning_faction": winning_faction,
                "winner_ids": [e.id for e in members],
                "winner_names": [e.name for e in members],
            }

        if tc.check_type == "bankruptcy":
            entity_type = params.get("entity_type")
            money_prop = params.get("money_property", "money")
            if not entity_type:
                return {}
            for e in self.state.get_entities_by_type(entity_type):
                if e.alive and float(e.get(money_prop, 0) or 0) > 0:
                    return {"winner_id": e.id, "winner_name": e.name}
            return {}

        if tc.check_type == "board_pattern":
            # Walk the BoardModule snapshot, find which mark completed
            # the pattern, and look up the entity whose `mark`/`side`
            # matches.
            try:
                from ..board_module import BoardModule
                board_id = params.get("board_id")
                for _mod in (self.state.domain_modules._modules.values()
                             if self.state.domain_modules else []):
                    if not isinstance(_mod, BoardModule):
                        continue
                    if board_id and _mod._id != board_id:
                        continue
                    snap = _mod.snapshot_grid(self.state)
                    winning_mark = self._winning_mark_in_snapshot(
                        snap, params.get("patterns", ["row_3", "col_3", "diag_3"])
                    )
                    if winning_mark is not None:
                        # Find entity with this mark
                        for ent in self.state.get_agent_entities():
                            if ent.get("mark") == winning_mark or ent.get("side") == winning_mark:
                                return {"winner_id": ent.id, "winner_name": ent.name,
                                        "winning_mark": winning_mark}
                        return {"winning_mark": winning_mark}
            except Exception:
                logger.exception("_resolve_winner board_pattern failed")
            return {}

        return {}

    @staticmethod
    def _winning_mark_in_snapshot(board, patterns):
        if not board or not isinstance(board, list):
            return None
        rows = len(board)
        cols = len(board[0]) if isinstance(board[0], list) else 0
        for pat in patterns:
            try:
                kind, n_s = pat.rsplit("_", 1)
                n = int(n_s)
            except ValueError:
                continue

            def scan_line(seq):
                for i in range(len(seq) - n + 1):
                    window = seq[i:i+n]
                    if any(v in (None, "", 0) for v in window):
                        continue
                    if all(v == window[0] for v in window):
                        return window[0]
                return None

            if kind == "row":
                for r in range(rows):
                    w = scan_line(list(board[r]))
                    if w is not None:
                        return w
            elif kind == "col":
                for c in range(cols):
                    w = scan_line([board[r][c] for r in range(rows)])
                    if w is not None:
                        return w
            elif kind == "diag":
                for r0 in range(rows - n + 1):
                    for c0 in range(cols - n + 1):
                        w = scan_line([board[r0+i][c0+i] for i in range(n)])
                        if w is not None:
                            return w
                        w = scan_line([board[r0+i][c0+n-1-i] for i in range(n)])
                        if w is not None:
                            return w
        return None

    def _evaluate_board_pattern(self, params: dict) -> bool:
        """Generic N-in-a-row detector. Reads a 2-D board from either:
          (a) a BoardModule snapshot (preferred if a 'board' domain
              module is registered), OR
          (b) an entity property `board_key` on entities of `entity_type`.
        """
        entity_type = params.get("entity_type")
        board_key = params.get("board_key", "board")
        patterns: List[str] = list(params.get("patterns") or ["row_3", "col_3", "diag_3"])
        mode = params.get("by", "any_actor")
        target_value = params.get("value")

        # ── Path (a): BoardModule snapshot ──
        if self.state.domain_modules:
            try:
                from ..board_module import BoardModule
                target_board_id = params.get("board_id")
                for _name, _mod in self.state.domain_modules._modules.items():
                    if not isinstance(_mod, BoardModule):
                        continue
                    if target_board_id and _mod._id != target_board_id:
                        continue
                    snapshot = _mod.snapshot_grid(self.state)
                    if not snapshot:
                        continue
                    rows = len(snapshot)
                    cols = len(snapshot[0]) if snapshot[0] else 0
                    for pat in patterns:
                        try:
                            kind, n_s = pat.rsplit("_", 1)
                            n = int(n_s)
                        except ValueError:
                            continue
                        if self._scan_board_pattern(snapshot, rows, cols, kind, n, mode, target_value):
                            return True
            except Exception:
                logger.exception("board_pattern WinPredicate: BoardModule snapshot failed")

        # ── Path (b): entity-property board ──
        entities = (
            self.state.get_entities_by_type(entity_type)
            if entity_type else list(self.state.get_agent_entities())
        )
        for ent in entities:
            board = ent.get(board_key)
            if not isinstance(board, list) or not board:
                continue
            rows = len(board)
            cols = len(board[0]) if isinstance(board[0], list) else 0
            for pat in patterns:
                try:
                    kind, n_s = pat.rsplit("_", 1)
                    n = int(n_s)
                except ValueError:
                    continue
                if self._scan_board_pattern(board, rows, cols, kind, n, mode, target_value):
                    return True
        return False

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
        def _match(seq: List[Any]) -> bool:
            if len(seq) < n:
                return False
            for i in range(len(seq) - n + 1):
                window = seq[i : i + n]
                if any(v in (None, "", 0) for v in window):
                    continue
                if mode == "specific_value":
                    if all(v == target_value for v in window):
                        return True
                else:  # any_actor — uniform non-empty run
                    first = window[0]
                    if all(v == first for v in window):
                        return True
            return False

        if kind == "row":
            return any(_match(list(board[r])) for r in range(rows))
        if kind == "col":
            return any(
                _match([board[r][c] for r in range(rows)])
                for c in range(cols)
            )
        if kind == "diag":
            # main diagonals (\) and anti-diagonals (/)
            for r0 in range(rows - n + 1):
                for c0 in range(cols - n + 1):
                    if _match([board[r0 + i][c0 + i] for i in range(n)]):
                        return True
                    if _match([board[r0 + i][c0 + n - 1 - i] for i in range(n)]):
                        return True
            return False
        return False

    # -------------------------------------------------------------------
    # World events
    # -------------------------------------------------------------------

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
