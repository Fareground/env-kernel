"""Effects targeting the physics model — shocks reach the ODE world."""
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.pipeline.loader import load_world


def _world():
    _state, engine = load_world({
        "name": "ode shock world",
        "entity_types": [{"name": "Operator", "role": "agent",
                          "properties": []}],
        "entities": [{"id": "op1", "entity_type": "Operator", "name": "Op"}],
        "physics": {
            "params": {"growth": 0.1},
            "variables": [
                {"name": "price", "value": 100.0, "rate": "growth * price",
                 "min": 0.0},
            ],
        },
        "actions": [{
            "name": "undercut",
            "description": "Competitor cuts prices",
            "actor_type": "Operator",
            "effects_on_success": [
                {"target": "physics", "operation": "multiply",
                 "field": "price", "value": 0.8},
                {"target": "physics", "operation": "set",
                 "field": "growth", "value": 0.02},
            ],
        }],
        "temporal": {"max_rounds": 3},
    })
    return _state, engine


class TestPhysicsEffects:
    def test_effects_mutate_variable_and_param(self):
        state, engine = _world()
        engine._resolve_and_apply("op1", ActionInstance(
            action_name="undercut", actor_id="op1"))
        assert state.physics.variables["price"].value == 80.0
        assert state.physics.params["growth"] == 0.02

    def test_unknown_physics_name_is_dropped_with_diagnostic(self):
        state, engine = _world()
        state.action_definitions["undercut"].effects_on_success[0].field = "ghost"
        engine._resolve_and_apply("op1", ActionInstance(
            action_name="undercut", actor_id="op1"))
        dropped = [e for e in state.event_log.get_all()
                   if e.event_type == "effect_dropped"]
        assert dropped and dropped[0].data["reason"] == "unknown_physics_name"

    def test_variable_clamps_apply(self):
        state, engine = _world()
        state.action_definitions["undercut"].effects_on_success[0] = \
            state.action_definitions["undercut"].effects_on_success[0]
        eff = state.action_definitions["undercut"].effects_on_success[0]
        eff.operation = type(eff.operation)("subtract")
        eff.value = 500.0
        engine._resolve_and_apply("op1", ActionInstance(
            action_name="undercut", actor_id="op1"))
        assert state.physics.variables["price"].value == 0.0  # min clamp
