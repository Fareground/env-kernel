"""Perception assembly — builds the dict an agent sees on its turn.

The canonical implementation of ``SimulationEngine._build_agent_perception``,
extracted here so the engine class stays focused on the tick loop. The
engine's method is now a 1-line delegation to ``build_perception()``.

The perception payload is intentionally typed as ``dict`` so a game
agent (LLM or otherwise) can read it without any kernel imports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from .engine import SimulationEngine

# Trade-shaped actions whose resolved events feed the agent's blotter.
_TRADE_ACTIONS = frozenset(
    {"buy", "sell", "buy_yes", "buy_no", "sell_yes", "sell_no"}
)
_RECENT_TRADE_LIMIT = 10
_RECENT_ACTION_LIMIT = 5


def build_perception(
    engine: "SimulationEngine", entity_id: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Build perception and valid actions for an agent.

    Returns ``(perception, valid_actions)``. An unknown ``entity_id``
    yields ``({}, [])`` rather than raising.
    """
    state = engine.state
    entity = state.get_entity(entity_id)
    if not entity:
        return {}, []

    round_num = state.temporal.current_round

    # Tick status effects
    tick_effects = state.status_effects.tick(entity_id, round_num)
    if tick_effects:
        engine._apply_effects(tick_effects, entity, None, {}, None)

    # Location effects
    entity_loc = state.locations.get(entity_id)
    if entity_loc:
        loc_changes = state.location_properties.apply_tick_effects(
            entity, entity_loc, state.entity_types,
        )
        for lc in loc_changes:
            delta = (lc.get("new") or 0) - (lc.get("old") or 0)
            engine._emit_event(
                "location_effect",
                actor_id=entity_id,
                data=lc,
                narrative=(
                    f"{entity.name} affected by {entity_loc}: "
                    f"{lc.get('field')} {lc.get('effect')} {delta:.1f}"
                ),
            )

    # Build perception
    active_events = None
    if engine.world_event_engine:
        active_events = [
            {"name": ae.definition.name, "description": ae.definition.description,
             "remaining_rounds": ae.remaining_rounds}
            for ae in engine.world_event_engine.get_active_events()
        ]
    perception = engine._perception_builder.build_perception(
        observer_id=entity_id,
        observer_type=entity.entity_type,
        rules=state.visibility_rules,
        entities=state.entities,
        entity_types=state.entity_types,
        resources=state.resources,
        relations=state.relations,
        spatial=state.spatial,
        temporal=state.temporal,
        active_world_events=active_events,
        faction_manager=state.factions,
    )

    _add_world_brief(state, perception)
    _add_messages(state, entity_id, perception)
    _add_domain_data(state, entity_id, perception)
    _add_crowd_trends(state, perception)
    perception = _apply_cognition(state, entity_id, perception)
    _add_trade_history(state, entity_id, perception)
    _add_time_context(engine, state, perception)
    _add_roles(state, entity_id, perception)
    _add_open_polls(state, entity_id, perception)
    _add_recent_actions(state, entity_id, perception)

    valid_actions = state.get_valid_actions(entity_id)
    # Domain modules can narrow the action list (e.g. a folded poker
    # player has nothing legal to do for the rest of the hand).
    if state.domain_modules:
        valid_actions = state.domain_modules.filter_valid_actions(
            entity_id, valid_actions, state,
        )
    return perception, valid_actions


# ---------------------------------------------------------------------------
# Section builders. Each mutates ``perception`` in place and is a no-op when
# the corresponding subsystem is absent, so a minimal world still perceives.
# ---------------------------------------------------------------------------


def _add_world_brief(state: Any, perception: Dict[str, Any]) -> None:
    """Name + description + rules from the env's template. The agent reads
    this to understand how to play the game, including rules markdown."""
    world_brief = getattr(state, "_world_brief", None)
    if world_brief:
        perception["world_brief"] = dict(world_brief)


def _add_messages(state: Any, entity_id: str, perception: Dict[str, Any]) -> None:
    entity_faction = state.factions.get_entity_faction(entity_id)
    incoming = state.messages.get_for_entity(entity_id, entity_faction)
    if incoming:
        perception["incoming_messages"] = [
            {"sender": m.sender_name, "sender_id": m.sender_id,
             "content": m.content, "type": m.message_type}
            for m in incoming
        ]


def _add_domain_data(state: Any, entity_id: str, perception: Dict[str, Any]) -> None:
    if not state.domain_modules:
        return
    domain_data = state.domain_modules.get_perception_data(entity_id, state)
    if domain_data:
        perception["domain_data"] = domain_data
    for _mod_name, module in state.domain_modules._modules.items():
        if hasattr(module, "get_visibility_overrides"):
            overrides = module.get_visibility_overrides()
            if overrides.get("hide_agent_identities"):
                perception["visible_entities"] = []
                perception.pop("agent_models", None)


def _add_crowd_trends(state: Any, perception: Dict[str, Any]) -> None:
    if not state.crowd_agents:
        return
    crowd_trends = state.crowd_agents.get_crowd_trends()
    if crowd_trends:
        perception["crowd_trends"] = crowd_trends


def _apply_cognition(
    state: Any, entity_id: str, perception: Dict[str, Any]
) -> Dict[str, Any]:
    """Cognitive processing. Unlike its siblings this one *replaces* the
    perception dict, so it returns the (possibly new) payload."""
    if not state.cognition:
        return perception
    overlay = state.cognition.get_prompt_overlay(entity_id)
    if overlay:
        perception["cognitive_state"] = overlay
    return state.cognition.process_perception_for(entity_id, perception)


def _add_trade_history(state: Any, entity_id: str, perception: Dict[str, Any]) -> None:
    """Structured blotter from this agent's past trades."""
    try:
        agent_events = state.event_log.get_by_actor(entity_id)
        trades = []
        for ev in agent_events:
            if ev.event_type != "action_resolved" or ev.action_name not in _TRADE_ACTIONS:
                continue
            details = ev.data.get("details", {}) if isinstance(ev.data, dict) else {}
            if not details.get("shares") and not details.get("amount"):
                continue
            trades.append({
                "round": ev.round_number,
                "action": ev.action_name,
                "amount": details.get("amount", 0),
                "shares": details.get("shares", 0),
                "price_at_trade": (
                    details.get("price_after")
                    or details.get("execution_price")
                    or details.get("new_price", 0)
                ),
            })
        if trades:
            # Summarize: last N trades + running P&L
            total_spent = sum(t["amount"] for t in trades if t["action"].startswith("buy"))
            total_received = sum(t["amount"] for t in trades if t["action"].startswith("sell"))
            perception["trade_history"] = {
                "total_trades": len(trades),
                "total_spent": round(total_spent, 2),
                "total_received": round(total_received, 2),
                "realized_pnl": round(total_received - total_spent, 2),
                "recent_trades": trades[-_RECENT_TRADE_LIMIT:],
            }
    except Exception:
        pass


def _add_time_context(
    engine: "SimulationEngine", state: Any, perception: Dict[str, Any]
) -> None:
    """Time context, present only when the simulation has time config."""
    time_ctx = state.temporal.time_context(engine.max_rounds)
    if time_ctx:
        perception["time_context"] = time_ctx


def _add_roles(state: Any, entity_id: str, perception: Dict[str, Any]) -> None:
    """Role + asymmetric info. The agent always knows their own role.
    Teammates / extra_visible_roles are revealed per the registry's rules.
    Everyone else's role is hidden."""
    if not (state.roles and state.roles.assignments):
        return
    own = state.roles.get_role(entity_id)
    if own:
        perception["your_role"] = {
            "name": own.name,
            "team": own.team,
            "description": own.description,
        }
    visible = state.roles.visible_role_map(entity_id)
    visible.pop(entity_id, None)
    if visible:
        perception["visible_roles"] = visible
    mates = state.roles.teammates(entity_id)
    if mates:
        perception["teammates"] = sorted(mates)


def _add_open_polls(state: Any, entity_id: str, perception: Dict[str, Any]) -> None:
    """Open polls the agent is eligible to vote in. Domain modules decide
    when/how to open polls; the kernel just surfaces them."""
    if not state.polls:
        return
    my_polls = state.polls.list_for_voter(entity_id)
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


def _add_recent_actions(state: Any, entity_id: str, perception: Dict[str, Any]) -> None:
    """Recap the agent's own recent decisions so they can build on past
    behavior instead of repeating mistakes. Private to this agent."""
    recent = state.action_history.get_recent(entity_id, _RECENT_ACTION_LIMIT)
    if recent:
        perception["your_recent_actions"] = [
            {
                "round": r.round_number,
                "action": r.action_name,
                "success": r.success,
            }
            for r in recent
        ]


__all__ = ["build_perception"]
