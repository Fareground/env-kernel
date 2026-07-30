"""Declared action-parameter bounds are enforced, not decorative."""
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.pipeline.loader import load_world


def _world():
    _state, engine = load_world({
        "name": "coerce world",
        "entity_types": [{
            "name": "Bidder", "role": "agent",
            "properties": [{"name": "offer", "type": "float", "default": 0.0}],
        }],
        "entities": [{"id": "b1", "entity_type": "Bidder", "name": "B1"}],
        "actions": [{
            "name": "bid",
            "description": "Place a bid",
            "actor_type": "Bidder",
            "parameters": [{"name": "amount", "type": "float",
                            "min": 1.0, "max": 100.0}],
            "effects_on_success": [{"target": "actor", "operation": "set",
                                    "field": "offer",
                                    "value": "$params.amount"}],
        }],
        "temporal": {"max_rounds": 3},
    })
    return engine


class TestParamCoercion:
    def test_out_of_range_value_is_clamped(self):
        world = _world()
        world._resolve_and_apply("b1", ActionInstance(
            action_name="bid", actor_id="b1",
            parameters={"amount": 5000.0}))
        assert world.state.get_entity("b1").properties["offer"] == 100.0
        events = [e for e in world.state.event_log.get_all()
                  if e.event_type == "action_corrected"]
        assert events and events[0].data["reason"] == "parameter_coerced"

    def test_below_min_is_clamped_up(self):
        world = _world()
        world._resolve_and_apply("b1", ActionInstance(
            action_name="bid", actor_id="b1",
            parameters={"amount": -3.0}))
        assert world.state.get_entity("b1").properties["offer"] == 1.0

    def test_in_range_value_is_untouched(self):
        world = _world()
        world._resolve_and_apply("b1", ActionInstance(
            action_name="bid", actor_id="b1",
            parameters={"amount": 42.5}))
        assert world.state.get_entity("b1").properties["offer"] == 42.5
        assert not [e for e in world.state.event_log.get_all()
                    if e.event_type == "action_corrected"]

    def test_uncoercible_string_falls_back_to_default(self):
        world = _world()
        world.state.action_definitions["bid"].parameters[0]["default"] = 10.0
        world._resolve_and_apply("b1", ActionInstance(
            action_name="bid", actor_id="b1",
            parameters={"amount": "a lot"}))
        assert world.state.get_entity("b1").properties["offer"] == 10.0

    def test_int_type_rounds(self):
        world = _world()
        world.state.action_definitions["bid"].parameters[0]["type"] = "int"
        world._resolve_and_apply("b1", ActionInstance(
            action_name="bid", actor_id="b1",
            parameters={"amount": 7.6}))
        assert world.state.get_entity("b1").properties["offer"] == 8
