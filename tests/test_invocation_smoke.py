"""Autonomous settlement is real execution, not a missing decision cast."""
from decimal import Decimal

import pytest

from fg_env_kernel import compile_template
from fg_env_kernel.pipeline.smoke import smoke_test


def checkout(*, cash=0.3, enabled=True):
    return {
        'name': 'Autonomous checkout',
        'entity_types': [{'name': 'Account', 'role': 'object', 'properties': [
            {'name': 'cash', 'type': 'float', 'default': 0, 'min_value': 0},
            {'name': 'stock', 'type': 'int', 'default': 0, 'min_value': 0},
            {'name': 'sold', 'type': 'int', 'default': 0},
        ]}],
        'entities': [
            {'id': 'shop', 'entity_type': 'Account', 'properties': {'stock': 10}},
            {'id': 'customer', 'entity_type': 'Account', 'properties': {'cash': cash}},
        ],
        'actions': [{'name': 'pay',
            'transfers': [{'source': 'customer', 'target': 'shop', 'field': 'cash', 'amount': 0.1}],
            'effects_on_success': [
                {'operation': 'subtract', 'target': 'shop', 'field': 'stock', 'value': 1},
                {'operation': 'add', 'target': 'shop', 'field': 'sold', 'value': 1},
            ]}],
        'derived_rules': [{'name': 'checkout', 'when': enabled,
            'then': [{'operation': 'invoke_action', 'value': {'action_name': 'pay'}}]}],
        'termination_conditions': [{'name': 'end', 'check_type': 'round_limit', 'params': {'max_rounds': 3}}],
    }


def test_smoke_counts_successful_autonomous_invocations_without_adding_agents():
    compiled = compile_template(checkout())
    assert compiled.ok, compiled.errors
    report = smoke_test(compiled.engine, rounds=3)
    assert report.healthy, report
    assert report.actions_taken == {'pay': 3}
    assert report.action_coverage_pct == 1
    assert not compiled.state.get_agent_entities()
    assert compiled.state.get_entity('shop').get('stock') == 7
    assert Decimal(str(compiled.state.get_entity('shop').get('cash'))) == Decimal('0.3')
    assert compiled.state.get_entity('customer').get('cash') == 0


@pytest.mark.parametrize('cash,enabled', [(0, True), (0.3, False)])
def test_failed_or_unreached_invocations_do_not_fake_smoke_action_coverage(cash, enabled):
    compiled = compile_template(checkout(cash=cash, enabled=enabled))
    assert compiled.ok, compiled.errors
    report = smoke_test(compiled.engine, rounds=3)
    assert not report.healthy
    assert report.actions_taken == {}
    assert report.action_coverage_pct == 0
    assert compiled.state.get_entity('shop').get('stock') == 10
    assert compiled.state.get_entity('shop').get('cash') == 0


def test_success_events_are_distinct_from_participant_decisions_and_keep_visibility():
    schema = checkout()
    schema['derived_rules'][0]['for_each'] = '$alive_of(Account)'
    schema['derived_rules'][0]['when'] = '$actor.id == "customer"'
    compiled = compile_template(schema)
    assert compiled.ok, compiled.errors
    events = []
    compiled.engine.on_event = events.append
    compiled.engine.max_rounds = 3
    compiled.engine.run()
    invoked = [event for event in events if event['event_type'] == 'action_invoked']
    assert len(invoked) == 3
    assert not any(event['event_type'] == 'action_resolved' for event in events)
    assert all(event['actor_id'] == 'customer' and event['action_name'] == 'pay' for event in invoked)
    assert all(event['data']['visible_to'] == ['customer'] for event in invoked)
