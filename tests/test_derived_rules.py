"""Tests for derived_rules — the unified inference layer.

A single JSON shape that subsumes triggers + conditional effects +
state machine transitions for the common "when X then Y" pattern.
This is the framework's forward-chaining inference primitive.
"""

from fg_env_kernel import compile_template
from fg_env_kernel.action import ActionInstance


# ---------------------------------------------------------------------------
# Death rule — the canonical use case
# ---------------------------------------------------------------------------

DEATH_GAME = {
    "name": "death_rule",
    "description": "Units die when hp hits 0 — derived rule, not per-action.",
    "temporal": {"phases": [{"name": "act"}]},
    "entity_types": [{
        "name": "Unit", "role": "agent",
        "properties": [
            {"name": "hp", "type": "int", "default": 10, "min_value": 0, "max_value": 100},
        ],
    }],
    "actions": [{
        "name": "hurt_self",
        "actor_type": "Unit",
        "resolution_archetype": "deterministic",
        "preconditions": [{"expr": "$actor.alive"}],
        "effects_on_success": [
            {"operation": "subtract", "target": "actor", "field": "hp", "value": 11}
        ],
    }],
    "entities": [
        {"id": "u", "name": "U", "entity_type": "Unit", "properties": {"hp": 10}}
    ],
    # The KEY POINT: death-on-zero-hp is a global derived rule, not a
    # per-action effect. Any future action that reduces hp causes the
    # rule to fire automatically.
    "derived_rules": [{
        "name": "death",
        "when": "$params.it.hp <= 0",
        "for_each": "$entities_of(Unit)",
        "as": "it",
        "then": [
            {"operation": "kill", "target": "$params.it"}
        ],
        "once_per_entity": True,
    }],
    "termination_conditions": [
        {"name": "all_dead", "check_type": "all_dead",
         "params": {"entity_type": "Unit"}}
    ],
}


def test_derived_rule_marks_unit_dead_when_hp_zero():
    """The derived rule fires after the action that brings hp ≤ 0.
    No per-action effect for death is needed."""
    def decide(eid, perc, va):
        return ActionInstance(action_name="hurt_self", actor_id=eid) if "hurt_self" in va else None

    result = compile_template(DEATH_GAME, seed=1, decision_fn=decide)
    assert result.ok
    state = result.state
    assert state.entities["u"].alive is True

    result.engine.max_rounds = 5
    result.engine.run()

    # Unit should be dead — derived_rule fired
    assert state.entities["u"].alive is False
    # And termination should have triggered
    assert result.engine.terminated_by == "all_dead"


def test_once_per_entity_prevents_re_firing():
    """Without once_per_entity, a death rule would re-fire every round
    while hp stays ≤ 0. With it, the rule fires exactly once."""
    def decide(eid, perc, va):
        return ActionInstance(action_name="hurt_self", actor_id=eid) if "hurt_self" in va else None

    # Capture every event the engine emits — count `derived_fact`s for our unit
    captured = []
    def on_event(ev):
        if ev.get("event_type") == "derived_fact":
            captured.append(ev)

    result = compile_template(DEATH_GAME, seed=1, decision_fn=decide, on_event=on_event)
    assert result.ok
    result.engine.max_rounds = 10
    result.engine.run()

    # Death rule should have fired exactly once for "u"
    assert len(captured) == 1, f"expected 1 derived_fact, got {len(captured)}: {captured}"


# ---------------------------------------------------------------------------
# Global rule — no for_each, fires once when world condition is true
# ---------------------------------------------------------------------------

GLOBAL_RULE_GAME = {
    "name": "global_inference",
    "description": "Game enters 'endgame' phase when total resource is low.",
    "temporal": {"phases": [{"name": "act"}]},
    "entity_types": [{
        "name": "P", "role": "agent",
        "properties": [
            {"name": "gold", "type": "int", "default": 5, "min_value": 0, "max_value": 100},
            {"name": "endgame_flag", "type": "bool", "default": False},
        ],
    }],
    "actions": [{
        "name": "spend",
        "actor_type": "P",
        "resolution_archetype": "deterministic",
        "preconditions": [{"expr": "$actor.gold > 0"}],
        "effects_on_success": [
            {"operation": "subtract", "target": "actor", "field": "gold", "value": 1}
        ],
    }],
    "entities": [
        {"id": f"p{i}", "name": f"P{i}", "entity_type": "P",
         "properties": {"gold": 5, "endgame_flag": False}}
        for i in range(3)
    ],
    # World-level inference: when total gold drops below 3, mark
    # everyone with endgame_flag = true. No for_each → global scope.
    # We use for_each here to apply the flag to every player.
    "derived_rules": [{
        "name": "endgame_trigger",
        "when": "$sum_of(P, gold) <= 5",
        "for_each": "$entities_of(P)",
        "as": "p",
        "then": [
            {"operation": "set", "target": "$params.p",
             "field": "endgame_flag", "value": True}
        ],
        "once_per_entity": True,
    }],
    "termination_conditions": [{
        "name": "round_cap", "check_type": "round_limit",
        "params": {"max_rounds": 20},
    }],
}


def test_world_condition_inference():
    def decide(eid, perc, va):
        return ActionInstance(action_name="spend", actor_id=eid) if "spend" in va else None

    result = compile_template(GLOBAL_RULE_GAME, seed=1, decision_fn=decide)
    assert result.ok
    result.engine.max_rounds = 15
    result.engine.run()

    # After enough spending, total gold ≤ 5 → everyone should have endgame_flag
    flags = [e.get("endgame_flag") for e in result.state.entities.values()]
    assert all(flags), f"expected all endgame_flag=True; got {flags}"


# ---------------------------------------------------------------------------
# Inference compose: derived rule fires an invoke_action
# ---------------------------------------------------------------------------

COMPOSE_GAME = {
    "name": "compose_inference",
    "description": "Inference rule triggers an invoke_action chain.",
    "temporal": {"phases": [{"name": "act"}]},
    "entity_types": [{
        "name": "U", "role": "agent",
        "properties": [
            {"name": "energy", "type": "int", "default": 5,
             "min_value": 0, "max_value": 100},
            {"name": "rest_count", "type": "int", "default": 0,
             "min_value": 0, "max_value": 100},
        ],
    }],
    "actions": [
        {
            "name": "do_rest",
            "actor_type": "U",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "add", "target": "actor", "field": "energy", "value": 5},
                {"operation": "add", "target": "actor", "field": "rest_count", "value": 1},
            ],
        },
        {
            "name": "drain",
            "actor_type": "U",
            "resolution_archetype": "deterministic",
            "preconditions": [{"expr": "$actor.energy > 0"}],
            "effects_on_success": [
                {"operation": "subtract", "target": "actor", "field": "energy", "value": 2}
            ],
        },
    ],
    "entities": [
        {"id": "u", "name": "U", "entity_type": "U",
         "properties": {"energy": 5, "rest_count": 0}}
    ],
    # When energy hits 0 → auto-rest. Inference fires an action.
    "derived_rules": [{
        "name": "auto_rest_when_drained",
        "when": "$params.it.energy <= 0",
        "for_each": "$entities_of(U)",
        "as": "it",
        "then": [
            {"operation": "invoke_action",
             "value": {"action_name": "do_rest",
                       "actor": "$params.it"}}
        ],
    }],
    "termination_conditions": [{
        "name": "rested_enough", "check_type": "expr",
        "params": {"expr": "$max_of(U, rest_count) >= 2"},
    }],
}


def test_derived_rule_can_invoke_action():
    """Drain energy until 0 → derived rule auto-rests → unit can drain again."""
    def decide(eid, perc, va):
        # Always drain
        return ActionInstance(action_name="drain", actor_id=eid) if "drain" in va else None

    result = compile_template(COMPOSE_GAME, seed=1, decision_fn=decide)
    assert result.ok
    result.engine.max_rounds = 30
    result.engine.run()

    # The unit should have rested at least twice (each rest fires when
    # energy → 0, and once_per_entity is FALSE for this rule)
    assert result.state.entities["u"].get("rest_count") >= 2, (
        f"expected rest_count ≥ 2; got {result.state.entities['u'].get('rest_count')}"
    )
