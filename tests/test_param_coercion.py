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
        "temporal": {"max_rounds": 3, "phases": [{"name": "act", "initiative_type": "fixed"}]},
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


class TestParameterContract:
    def test_missing_required_value_does_not_change_state(self):
        world = _world()
        world.state.action_definitions['bid'].parameters[0]['required'] = True
        world._resolve_and_apply('b1', ActionInstance(action_name='bid', actor_id='b1'))
        assert world.state.get_entity('b1').properties['offer'] == 0.0
        failures = [e for e in world.state.event_log.get_all() if e.event_type == 'action_failed']
        assert failures[0].data == {'reason': 'invalid_parameters', 'details': {'amount': 'is required'}}
        assert not [e for e in world.state.event_log.get_all() if e.event_type == 'action_resolved']

    def test_explicit_default_is_validated_and_clamped(self):
        world = _world()
        world.state.action_definitions['bid'].parameters[0].update(required=True, default=500)
        world._resolve_and_apply('b1', ActionInstance(action_name='bid', actor_id='b1'))
        assert world.state.get_entity('b1').properties['offer'] == 100

    def test_invalid_values_never_reach_numeric_effect(self):
        for value in ('not a score', float('nan'), float('inf'), True, [], {}):
            world = _world()
            world._resolve_and_apply('b1', ActionInstance(action_name='bid', actor_id='b1', parameters={'amount': value}))
            assert world.state.get_entity('b1').properties['offer'] == 0.0, value
            assert any(e.data.get('reason') == 'invalid_parameters' for e in world.state.event_log.get_all())

    def test_all_fields_validate_before_any_effect_or_public_message(self):
        world = _world()
        action = world.state.action_definitions['bid']
        action.message_action = True
        action.parameters.append({'name': 'message', 'type': 'string', 'required': True})
        world._resolve_and_apply('b1', ActionInstance(action_name='bid', actor_id='b1', parameters={'amount': 20, 'message': []}))
        assert world.state.get_entity('b1').properties['offer'] == 0
        assert not [e for e in world.state.event_log.get_all() if e.event_type in ('action_resolved', 'agent_message')]

    def test_boolean_collection_and_enum_contracts(self):
        cases = [('bool', 'false'), ('list', '[]'), ('dict', '{}'), ('enum', 'other')]
        for kind, invalid in cases:
            world = _world()
            declaration = {'name': 'extra', 'type': kind, 'required': True}
            if kind == 'enum':
                declaration['enum_values'] = ['yes', 'no']
            world.state.action_definitions['bid'].parameters.append(declaration)
            world._resolve_and_apply('b1', ActionInstance(action_name='bid', actor_id='b1', parameters={'amount': 20, 'extra': invalid}))
            assert world.state.get_entity('b1').properties['offer'] == 0


def test_smoke_exercises_required_parameters_with_seeded_samples():
    from fg_env_kernel.pipeline.smoke import smoke_test
    results = []
    for _ in range(2):
        world = _world()
        world.state.action_definitions['bid'].parameters[0]['required'] = True
        report = smoke_test(world, rounds=3, seed=13)
        assert report.healthy
        assert report.actions_taken == {'bid': 3}
        score = world.state.get_entity('b1').properties['offer']
        assert 1 <= score <= 100
        results.append(score)
    assert results[0] == results[1]


def test_explicit_smoke_decisions_can_still_test_rejected_inputs():
    from fg_env_kernel.pipeline.smoke import smoke_test
    world = _world()
    world.state.action_definitions['bid'].parameters[0]['required'] = True
    report = smoke_test(world, rounds=2, decisions=lambda entity, perception, actions:
                        ActionInstance(action_name='bid', actor_id=entity))
    assert not report.healthy
    assert world.state.get_entity('b1').properties['offer'] == 0
