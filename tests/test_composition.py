"""Tests for action composition primitives (invoke_action, for_each, sequence).

These are the kernel-registered effect ops that let JSON express
multi-step / aggregate mechanics. Together with the state-query
functions, they're what makes the engine truly genre-agnostic.
"""
import pytest

from fg_env_kernel import compile_template
from fg_env_kernel.action import ActionInstance


# ---------------------------------------------------------------------------
# for_each — "AoE attack hits all enemies in range"
# ---------------------------------------------------------------------------

AOE_GAME = {
    "name": "aoe_test",
    "temporal": {"phases": [{"name": "act"}]},
    "entity_types": [{
        "name": "Unit", "role": "agent",
        "properties": [
            {"name": "health", "type": "int", "default": 10,
             "min_value": 0, "max_value": 100},
            {"name": "team", "type": "string", "default": "blue"},
        ],
    }],
    "actions": [{
        "name": "fireball",
        "actor_type": "Unit",
        "resolution_archetype": "deterministic",
        "effects_on_success": [{
            "operation": "for_each",
            "value": {
                "in": "$alive_of(Unit)",
                "as": "victim",
                "effects": [
                    {"operation": "subtract", "target": "$params.victim",
                     "field": "health", "value": 3}
                ]
            }
        }],
    }],
    "entities": [
        {"id": f"u{i}", "name": f"U{i}", "entity_type": "Unit",
         "properties": {"health": 10, "team": "blue" if i < 2 else "red"}}
        for i in range(4)
    ],
    "termination_conditions": [{
        "name": "everyone_hurt",
        "check_type": "expr",
        "params": {"expr": "$max_of(Unit, health) <= 4"},
    }],
}


def test_for_each_hits_every_entity():
    """A single fireball action should damage all 4 units. Verified by
    counting how many took damage after exactly ONE action fires —
    we let only one unit act to isolate the for_each behavior."""
    cast_count = {"n": 0}

    def decide(eid, perc, va):
        # Only let u0 cast fireball; everyone else passes
        if eid == "u0" and "fireball" in va and cast_count["n"] == 0:
            cast_count["n"] += 1
            return ActionInstance(action_name="fireball", actor_id=eid)
        return None

    result = compile_template(AOE_GAME, seed=1, decision_fn=decide)
    assert result.ok
    state = result.state
    assert all(e.get("health") == 10 for e in state.entities.values())

    result.engine.max_rounds = 1
    result.engine.run()

    # Exactly one fireball fired — every unit should be at 7
    healths_after = {e.id: e.get("health") for e in state.entities.values()}
    assert all(h == 7 for h in healths_after.values()), (
        f"expected for_each to damage every unit by 3: {healths_after}"
    )


# ---------------------------------------------------------------------------
# invoke_action — "castling = move_king + move_rook atomically"
# ---------------------------------------------------------------------------

INVOKE_GAME = {
    "name": "invoke_test",
    "temporal": {"phases": [{"name": "act"}]},
    "entity_types": [{
        "name": "Player", "role": "agent",
        "properties": [
            {"name": "primary", "type": "int", "default": 0,
             "min_value": 0, "max_value": 100},
            {"name": "secondary", "type": "int", "default": 0,
             "min_value": 0, "max_value": 100},
        ],
    }],
    "actions": [
        {
            "name": "bump_secondary",
            "actor_type": "Player",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "add", "target": "actor", "field": "secondary", "value": 5}
            ],
        },
        {
            "name": "combo",
            "actor_type": "Player",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "add", "target": "actor", "field": "primary", "value": 3},
                {"operation": "invoke_action",
                 "value": {"action_name": "bump_secondary"}},
            ],
        }
    ],
    "entities": [
        {"id": "a", "name": "A", "entity_type": "Player",
         "properties": {"primary": 0, "secondary": 0}},
    ],
    "termination_conditions": [{
        "name": "done", "check_type": "round_limit",
        "params": {"max_rounds": 1},
    }],
}


def test_invoke_action_fires_sub_action_atomically():
    """A `combo` action should bump primary AND invoke bump_secondary."""
    def decide(eid, perc, va):
        if "combo" in va:
            return ActionInstance(action_name="combo", actor_id=eid)
        return None

    result = compile_template(INVOKE_GAME, seed=42, decision_fn=decide)
    assert result.ok
    result.engine.max_rounds = 1
    result.engine.run()

    a = result.state.entities["a"]
    assert a.get("primary") == 3, f"combo's own effect didn't fire: primary={a.get('primary')}"
    assert a.get("secondary") == 5, (
        f"invoke_action didn't fire bump_secondary: secondary={a.get('secondary')}"
    )


# ---------------------------------------------------------------------------
# Combined: a realistic genre — RPG attack with AoE + invoke
# ---------------------------------------------------------------------------

RPG_GAME = {
    "name": "rpg_combat",
    "description": "Attack actions and a chain-lightning move.",
    "temporal": {"phases": [{"name": "act"}]},
    "entity_types": [{
        "name": "Fighter", "role": "agent",
        "properties": [
            {"name": "hp", "type": "int", "default": 20, "min_value": 0, "max_value": 100},
            {"name": "mp", "type": "int", "default": 10, "min_value": 0, "max_value": 100},
        ],
    }],
    "actions": [
        {
            "name": "shock_one",
            "actor_type": "Fighter",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "subtract", "target": "$params.target_id",
                 "field": "hp", "value": 5}
            ],
        },
        {
            "name": "chain_lightning",
            "actor_type": "Fighter",
            "resolution_archetype": "deterministic",
            "preconditions": [{"expr": "$actor.mp >= 5"}],
            "effects_on_success": [
                {"operation": "subtract", "target": "actor", "field": "mp", "value": 5},
                {"operation": "for_each",
                 "value": {
                     "in": "$alive_of(Fighter)",
                     "as": "victim",
                     "effects": [
                         {"operation": "subtract", "target": "$params.victim",
                          "field": "hp", "value": 4}
                     ]
                 }},
            ],
        }
    ],
    "entities": [
        {"id": f"f{i}", "name": f"F{i}", "entity_type": "Fighter",
         "properties": {"hp": 20, "mp": 10}}
        for i in range(3)
    ],
    "termination_conditions": [{
        "name": "anyone_low", "check_type": "expr",
        "params": {"expr": "$min_of(Fighter, hp) <= 5"},
    }],
}


def test_chain_lightning_rpg_pattern():
    """Real RPG mechanic in pure JSON: chain lightning costs mana,
    damages everyone alive. Uses for_each + state queries."""
    def decide(eid, perc, va):
        return (ActionInstance(action_name="chain_lightning", actor_id=eid)
                if "chain_lightning" in va else None)

    result = compile_template(RPG_GAME, seed=7, decision_fn=decide)
    assert result.ok
    result.engine.max_rounds = 2
    result.engine.run()

    # First fighter to act burned 5 mp and dealt 4 to everyone
    state = result.state
    fighters = list(state.entities.values())
    # At least one should have spent mana
    assert any(f.get("mp") < 10 for f in fighters), (
        f"expected mp drain: {[(f.id, f.get('mp')) for f in fighters]}"
    )
    # Everyone should have taken some damage
    assert all(f.get("hp") < 20 for f in fighters), (
        f"expected damage to all: {[(f.id, f.get('hp')) for f in fighters]}"
    )


# ---------------------------------------------------------------------------
# Capability proof: 4 wildly different genres in pure JSON
# ---------------------------------------------------------------------------


def test_kernel_capabilities_expose_new_ops():
    """The new effect ops must show up in the live registry export."""
    from fg_env_kernel import export_kernel_contract

    caps = export_kernel_contract()["live_capabilities"]
    ops = caps["effect_operations"]
    assert "for_each" in ops
    assert "invoke_action" in ops
    assert "sequence" in ops
