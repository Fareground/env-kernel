"""Agent turn execution — perceive, decide, resolve, apply.

The canonical implementation of ``SimulationEngine._run_agent_turn_inner``,
extracted here so the engine class stays focused on the tick loop. The
engine's method is now a 1-line delegation to ``run_agent_turn()``.

This is the hot path of a discrete run: one entity is given its perception,
its decision callback is invoked, and whatever it chose is resolved against
the world and applied.
"""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..action import ActionInstance, ActionDefinition, Effect, EffectOperation
from ..messaging import Message
from ..resolution import get_resolution, ResolutionResult
from .engine import _action_suppresses_chat, _coerce_effects, _is_multi_target

if TYPE_CHECKING:
    from .engine import SimulationEngine

logger = logging.getLogger(__name__)


def run_agent_turn(engine, entity_id: str):
    """Execute one agent's turn: perceive -> decide -> resolve -> apply."""
    entity = engine.state.get_entity(entity_id)
    if not entity:
        return

    # 0. Tick status effects (damage-over-time, healing, etc.)
    round_num = engine.state.temporal.current_round
    tick_effects = engine.state.status_effects.tick(entity_id, round_num)
    if tick_effects:
        engine._apply_effects(tick_effects, entity, None, {}, None)

    # 0b. Tick location effects
    entity_loc = engine.state.locations.get(entity_id)
    if entity_loc:
        loc_changes = engine.state.location_properties.apply_tick_effects(
            entity, entity_loc, engine.state.entity_types,
        )
        for lc in loc_changes:
            engine._emit_event(
                "location_effect",
                actor_id=entity_id,
                data=lc,
                narrative=f"{entity.name} affected by {entity_loc}: {lc.get('field')} {lc.get('effect')} {(lc.get('new') or 0) - (lc.get('old') or 0):.1f}",
            )

    # 0c. Check for active sequence
    active_seq = engine.state.sequences.get_active(entity_id)

    # 1. Build perception
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
        rules=engine.state.visibility_rules,
        entities=engine.state.entities,
        entity_types=engine.state.entity_types,
        resources=engine.state.resources,
        relations=engine.state.relations,
        spatial=engine.state.spatial,
        temporal=engine.state.temporal,
        active_world_events=active_events,
        faction_manager=engine.state.factions,
    )

    # Add sequence info to perception
    if active_seq:
        perception["current_sequence"] = {
            "action": active_seq.action_name,
            "progress": active_seq.rounds_completed,
            "total": active_seq.total_rounds,
        }

    # Add incoming messages to perception
    entity_faction = engine.state.factions.get_entity_faction(entity_id)
    incoming = engine.state.messages.get_for_entity(entity_id, entity_faction)
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
                wm = engine.state.get_or_create_world_model(entity_id)
                wm.receive_info(msg.sender_id, msg.info_payload, round_num)

    # Add location info to perception
    if entity_loc:
        loc_def = engine.state.location_properties.get(entity_loc)
        if loc_def:
            perception["location_info"] = {
                "id": loc_def.id,
                "name": loc_def.name,
                "description": loc_def.description,
                "properties": dict(loc_def.properties),
            }

    # Add inventory to perception
    inv_items = engine.state.inventory.get_inventory(entity_id)
    if inv_items:
        perception["inventory"] = [
            {"id": it.id, "name": it.name, "type": it.item_type, "quantity": it.quantity}
            for it in inv_items
        ]
    if entity_loc:
        ground = engine.state.inventory.get_ground_items(entity_loc)
        if ground:
            perception["ground_items"] = [
                {"id": it.id, "name": it.name, "type": it.item_type, "quantity": it.quantity}
                for it in ground
            ]

    # Add goals to perception
    active_goals = engine.state.goals.get_active_goals(entity_id)
    if active_goals:
        perception["goals"] = [
            {"id": g.id, "description": g.description, "priority": g.priority}
            for g in active_goals
        ]

    # Add skills to perception
    all_skills = engine.state.skills.get_all_skills(entity_id)
    if all_skills:
        perception["skills"] = {
            name: {"level": entry.level, "xp": entry.xp}
            for name, entry in all_skills.items()
        }

    # Add available recipes to perception
    available_recipes = engine.state.recipes.get_available(entity_id, engine.state)
    if available_recipes:
        perception["available_recipes"] = available_recipes

    # 1b. Trend analysis — store snapshot and inject trends into perception
    engine._trend_analyzer.update(entity_id, perception)
    trends = engine._trend_analyzer.get_trends(entity_id)
    if trends:
        perception["trends"] = trends

    # 1c. Update persistent world model from current perception
    world_model = engine.state.get_or_create_world_model(entity_id)
    world_model.update_from_perception(perception, round_num)
    wm_summary = world_model.summarize_for_prompt(perception)
    if wm_summary:
        perception["world_model_memories"] = wm_summary

    # 1d. Add active negotiations / agreements / auctions to perception
    neg_info = engine.state.negotiations.get_active_for_entity(entity_id, current_round=round_num)
    if any(neg_info.values()):
        perception["active_negotiations"] = neg_info

    # 1e. Add plan & theory-of-mind to perception
    plan_info = engine.state.plans.summarize_for_prompt(entity_id)
    if plan_info.get("plan"):
        perception["current_plan"] = plan_info["plan"]
    if plan_info.get("agent_models"):
        perception["agent_models"] = plan_info["agent_models"]

    # 1g. Add external data from connectors
    if engine.state.connectors:
        ext_data = engine.state.connectors.inject_context(entity_id)
        if ext_data:
            perception["external_data"] = ext_data

    # 1h. Add domain-specific perception data
    if engine.state.domain_modules:
        domain_data = engine.state.domain_modules.get_perception_data(entity_id, engine.state)
        if domain_data:
            perception["domain_data"] = domain_data

        # Apply domain visibility overrides (e.g., anonymous markets)
        for _mod_name, _module in engine.state.domain_modules._modules.items():
            if hasattr(_module, 'get_visibility_overrides'):
                overrides = _module.get_visibility_overrides()
                if overrides.get("hide_agent_identities"):
                    # Remove individual agent details — agents only see aggregate market data
                    perception["visible_entities"] = []
                    perception.pop("agent_models", None)

    # 1i. Add cognitive state to perception
    if engine.state.cognition:
        overlay = engine.state.cognition.get_prompt_overlay(entity_id)
        if overlay:
            perception["cognitive_state"] = overlay

    # 1j. Add social data to perception
    if engine.state.social:
        social_data = engine.state.social.get_perception_data(entity_id)
        if social_data:
            perception["social_data"] = social_data

    # 1k. Add crowd trends to perception
    if engine.state.crowd_agents:
        crowd_trends = engine.state.crowd_agents.get_crowd_trends()
        if crowd_trends:
            perception["crowd_trends"] = crowd_trends

    # 1f. Validate plan against currently available actions
    valid_actions = engine.state.get_valid_actions(entity_id)
    if engine.state.domain_modules:
        valid_actions = engine.state.domain_modules.filter_valid_actions(
            entity_id, valid_actions, engine.state,
        )
    if valid_actions:
        engine.state.plans.validate_plan(entity_id, valid_actions, round_num)

    # 2. Get valid actions
    if not valid_actions:
        return

    # 2b. Apply bounded rationality to filter perception
    if engine.state.cognition:
        perception = engine.state.cognition.process_perception_for(entity_id, perception)

    # 3. Get decision from callback (or crowd behavior or human takeover)
    if engine.decision_fn is None:
        return

    # 3a. Check if this is a crowd agent (skip LLM entirely)
    action_instance = None
    if engine.state.crowd_agents and engine.state.crowd_agents.is_crowd(entity_id):
        behavior = engine.state.crowd_agents.get_behavior(entity_id)
        if behavior:
            # Crowd behaviors mutate state (buy/sell); pin their RNG to the
            # sim seed so two same-seed runs produce identical crowd trades.
            behavior._rng = engine._rng
            # Build a lightweight perception summary for the crowd behavior
            own_props = entity.properties if entity else {}
            crowd_perception = {
                "own_properties": own_props,
                "crowd_trends": perception.get("crowd_trends", {}),
            }
            # Inject domain module perception data (e.g. prediction market prices)
            if engine.state.domain_modules:
                for mod_name, module in engine.state.domain_modules._modules.items():
                    dm_data = module.get_perception_data(entity_id, engine.state)
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
    if action_instance is None and engine.state.controller:
        takeover = engine.state.controller.is_taken_over(entity_id)
        if takeover and takeover._human_decision_fn:
            action_instance = takeover._human_decision_fn(entity_id, perception, valid_actions)

    # 3c. Fall back to LLM decision_fn
    if action_instance is None:
        try:
            action_instance = engine.decision_fn(entity_id, perception, valid_actions)
        except Exception as e:
            logger.error(f"Sequential decision failed for {entity_id}: {e}")
            engine._emit_event(
                "decision_error",
                actor_id=entity_id,
                data={"error": str(e)},
                narrative=f"Decision error for {entity_id}: {e}",
            )
            action_instance = None
    if action_instance is None:
        # If in a non-interruptible sequence, force continuation
        if active_seq:
            seq_def = engine.state.action_definitions.get(active_seq.action_name)
            if seq_def and not seq_def.interruptible:
                action_instance = ActionInstance(
                    action_name=active_seq.action_name,
                    actor_id=entity_id,
                    target_id=active_seq.target_id,
                    parameters=active_seq.parameters,
                )
            else:
                # Interruptible sequence cancelled by passing
                engine.state.sequences.cancel(entity_id)
                engine._emit_event(
                    "sequence_cancelled",
                    actor_id=entity_id,
                    action_name=active_seq.action_name,
                    narrative=f"{entity.name} abandons {active_seq.action_name}.",
                )

        if action_instance is None:
            engine._emit_event(
                "action_skipped",
                actor_id=entity_id,
                narrative=f"{entity.name} passes.",
            )
            return

    # 3b. Handle sequence continuation or interruption
    if active_seq:
        if action_instance.action_name == active_seq.action_name:
            # Continue the sequence
            completed = engine.state.sequences.advance(entity_id)
            if not completed:
                # Still in progress — emit progress event, skip resolution
                seq = engine.state.sequences.get_active(entity_id)
                engine._emit_event(
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
                engine._emit_event(
                    "sequence_completed",
                    actor_id=entity_id,
                    action_name=active_seq.action_name,
                    narrative=f"{entity.name} completes {active_seq.action_name}!",
                )
                # Fall through to normal resolution below
        else:
            # Chose a different action
            seq_def = engine.state.action_definitions.get(active_seq.action_name)
            if seq_def and not seq_def.interruptible:
                # Force continuation — override the agent's choice
                action_instance = ActionInstance(
                    action_name=active_seq.action_name,
                    actor_id=entity_id,
                    target_id=active_seq.target_id,
                    parameters=active_seq.parameters,
                )
                completed = engine.state.sequences.advance(entity_id)
                if not completed:
                    seq = engine.state.sequences.get_active(entity_id)
                    engine._emit_event(
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
                    engine._emit_event(
                        "sequence_completed",
                        actor_id=entity_id,
                        action_name=active_seq.action_name,
                        narrative=f"{entity.name} completes {active_seq.action_name}!",
                    )
            else:
                # Interruptible — cancel and proceed with new action
                engine.state.sequences.cancel(entity_id)
                engine._emit_event(
                    "sequence_cancelled",
                    actor_id=entity_id,
                    action_name=active_seq.action_name,
                    narrative=f"{entity.name} interrupts {active_seq.action_name}.",
                )

    # 4. Validate the chosen action (normalizes case/typo in place and
    # emits action_corrected / action_failed — same logic as the
    # parallel resolve path).
    action_def = engine._resolve_action_def(entity, action_instance)
    if action_def is None:
        return

    # 4b. Check if this is a new multi-round sequence start
    if action_def.sequence_rounds > 0 and not active_seq:
        # Start a new sequence — don't resolve yet
        engine.state.sequences.start(
            entity_id=entity_id,
            action_name=action_instance.action_name,
            target_id=action_instance.target_id,
            parameters=action_instance.parameters,
            total_rounds=action_def.sequence_rounds,
            round_num=round_num,
        )
        engine._emit_event(
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
    suppress_chat = _action_suppresses_chat(engine.state, action_instance.action_name)
    evt_reasoning = action_instance.reasoning   # always exposed
    evt_speech = "" if suppress_chat else action_instance.speech
    evt_parameters = {} if suppress_chat else action_instance.parameters

    attempted_data = {"parameters": evt_parameters, "reasoning": evt_reasoning, "speech": evt_speech}
    _adef = engine.state.action_definitions.get(action_instance.action_name)
    if _adef is not None and not _adef.broadcast:
        attempted_data["visible_to"] = [
            p for p in (entity_id, action_instance.target_id) if p
        ]
    engine._emit_event(
        "action_attempted",
        actor_id=entity_id,
        target_id=action_instance.target_id,
        action_name=action_instance.action_name,
        data=attempted_data,
        narrative=f"{entity.name} attempts {action_instance.action_name}.",
    )

    # 5. Resolve (apply status effect modifiers to properties for resolution)
    target = engine.state.get_entity(action_instance.target_id) if action_instance.target_id else None

    # 5a. Domain module validation (e.g. budget checks for prediction markets)
    if engine.state.domain_modules:
        validation_error = engine.state.domain_modules.validate_action(
            action_instance.action_name, entity, target, engine.state,
        )
        if validation_error:
            engine._emit_event(
                "action_failed",
                actor_id=entity_id,
                target_id=action_instance.target_id,
                action_name=action_instance.action_name,
                data={"success": False, "details": {"reason": validation_error}},
                narrative=f"{entity.name} cannot {action_instance.action_name}: {validation_error}",
            )
            return

    # 5b. Check target-side preconditions (IS_ADJACENT, SAME_FACTION, etc.)
    if not engine._check_target_preconditions(entity, target, action_def):
        engine._emit_event(
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
    actor_mods = engine.state.status_effects.get_modifiers(entity_id)
    for prop, mod_val in actor_mods.items():
        if prop in actor_props and isinstance(actor_props[prop], (int, float)):
            actor_props[prop] = actor_props[prop] + mod_val

    target_props = None
    if target:
        target_props = dict(target.properties)
        target_mods = engine.state.status_effects.get_modifiers(target.id)
        for prop, mod_val in target_mods.items():
            if prop in target_props and isinstance(target_props[prop], (int, float)):
                target_props[prop] = target_props[prop] + mod_val

    # Domain module hooks: modify properties/params before resolution
    resolution_params = dict(action_def.resolution_params) if action_def.resolution_params else {}
    resolution_params["action_name"] = action_instance.action_name  # So resolution can infer direction
    if engine.state.domain_modules:
        actor_props, target_props = engine.state.domain_modules.modify_resolution(
            actor_props, target_props, action_def, engine.state,
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
        rng=engine._rng,
    )

    # 6. Apply effects (with partial success support)
    if result.success:
        effects = action_def.effects_on_success
    elif result.partial and action_def.effects_on_partial:
        effects = action_def.effects_on_partial
    else:
        effects = action_def.effects_on_failure
    state_changes = engine._apply_effects(effects, entity, target, action_instance.parameters, result)

    # 7. Generate narrative (rich if narrative_fn provided, else mechanical fallback)
    if engine.narrative_fn:
        narrative = engine.narrative_fn(entity, target, action_def, action_instance, result, state_changes)
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
    engine._emit_event(
        "action_resolved",
        actor_id=entity_id,
        target_id=action_instance.target_id,
        action_name=action_instance.action_name,
        data=resolved_data,
        narrative=narrative,
    )

    # 7a. Domain module post-resolution hooks (update pools, positions, etc.)
    if engine.state.domain_modules:
        if result and getattr(result, "details", None) is not None:
            result.details["_action_params"] = dict(action_instance.parameters or {})
            # Forward action_instance's target_id and speech so domain
            # modules see what the LLM chose (not just the resolved
            # target entity, which may be filtered). Speech wiped for
            # chat-suppressed modules.
            result.details["target_id"] = action_instance.target_id
            result.details["speech"] = "" if suppress_chat else action_instance.speech
        domain_changes = engine.state.domain_modules.post_resolution(
            actor_id=entity_id,
            action_name=action_instance.action_name,
            success=result.success,
            result=result,
            state=engine.state,
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
            engine._emit_event(
                evt_type,
                actor_id=change.get("actor_id") or entity_id,
                target_id=change.get("target_id"),
                action_name=action_instance.action_name,
                data=payload,
                narrative=change.get("narrative", f"Domain update: {change.get('type', 'unknown')}"),
            )

    # 7b. Trigger emotions based on action outcome
    if engine.state.cognition:
        if result.success:
            engine.state.cognition.trigger_emotion_for(entity_id, "joy", 0.2 * result.success_degree)
        else:
            engine.state.cognition.trigger_emotion_for(entity_id, "anger", 0.1)
            engine.state.cognition.trigger_emotion_for(entity_id, "fear", 0.05)

    # 7c. Track action for crowd agent promotion/demotion
    if engine.state.crowd_agents:
        engine.state.crowd_agents.record_action(entity_id, action_instance.action_name)

    # 8. Record action and process chains
    engine.state.action_history.record(entity_id, action_instance.action_name, result.success, round_num)
    if result.success:
        if action_def.unlocks_actions:
            engine.state.action_history.unlock(entity_id, action_def.unlocks_actions)
        if action_def.locks_actions:
            engine.state.action_history.lock(entity_id, action_def.locks_actions)
    if action_def.cooldown_rounds > 0:
        engine.state.action_history.set_cooldown(
            entity_id, action_instance.action_name,
            round_num + action_def.cooldown_rounds + 1,
        )

    # 8b. Advance plan if the agent took the planned action
    active_plan = engine.state.plans.get_active_plan(entity_id)
    if active_plan:
        current_step = active_plan.get_current_step()
        if current_step and current_step.action == action_instance.action_name:
            outcome_str = narrative[:120] if narrative else ""
            plan_done = engine.state.plans.advance_plan(entity_id, result.success, outcome_str)
            if plan_done:
                engine._emit_event(
                    "plan_completed",
                    actor_id=entity_id,
                    data={"plan_goal": active_plan.goal_description},
                    narrative=f"{entity.name} completed their plan: {active_plan.goal_description}",
                )

    # 8c. Update theory-of-mind for all agents who can observe this action
    for observer in engine.state.get_agent_entities():
        if observer.id == entity_id:
            continue
        engine.state.plans.update_agent_model(
            observer_id=observer.id,
            target_id=entity_id,
            target_name=entity.name,
            action_name=action_instance.action_name,
            action_target_id=action_instance.target_id,
            success=result.success,
            round_num=round_num,
        )

    # 9. Notify outcome callback (for agent feedback loop)
    if engine.outcome_fn:
        # Include target_id in details so the callback can track entity interactions
        outcome_details = dict(result.details) if result.details else {}
        if action_instance.target_id and "target_id" not in outcome_details:
            outcome_details["target_id"] = action_instance.target_id
        engine.outcome_fn(entity_id, action_instance.action_name, result.success, narrative, outcome_details)

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
                msg.recipient_faction = action_instance.parameters.get("recipient_faction") or engine.state.factions.get_entity_faction(entity_id)
            engine.state.messages.post(msg)

        target_name = target.name if target else None
        engine._emit_event(
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


__all__ = ["run_agent_turn"]
