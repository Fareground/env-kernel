"""A model's selected action is untrusted, even if its menu was correct."""
import pytest

from fg_env_kernel import compile_template
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.domain.base import DomainModule, DomainModuleManager
from fg_env_kernel.status_effects import StatusEffectDefinition


def build(mode, chosen='spend', target=None, sequence=0):
    schema = {'name': 'Action authorization', 'entity_types': [
        {'name': kind, 'role': role, 'properties': [{'name': 'cash', 'type': 'int', 'default': 5}]}
        for kind, role in [('Buyer', 'agent'), ('Seller', 'agent'), ('Stock', 'object')]],
        'entities': [{'id': name, 'entity_type': kind} for name, kind in
                     [('buyer', 'Buyer'), ('seller', 'Seller'), ('stock', 'Stock')]],
        'temporal': {'phases': [{'name': 'act', 'active_roles': ['Buyer'], 'resolution_mode': mode}]},
        'actions': [{'name': 'wait', 'actor_type': 'Buyer'},
                    {'name': 'spend', 'actor_type': 'Buyer', 'sequence_rounds': sequence,
                     'message_action': True, 'effects_on_success': [
                         {'operation': 'subtract', 'target': 'actor', 'field': 'cash', 'value': 1}]},
                    {'name': 'seller_only', 'actor_type': 'Seller', 'effects_on_success': [
                         {'operation': 'subtract', 'target': 'actor', 'field': 'cash', 'value': 1}]}]}
    result = compile_template(schema, decision_fn=lambda actor, *_: ActionInstance(
        action_name=chosen, actor_id=actor, target_id=target, speech='Must not broadcast on rejection'))
    assert result.ok, result.errors
    return result


def run_and_assert_rejected(result):
    result.engine.max_rounds = 1
    result.engine.run()
    assert result.state.entities['buyer'].get('cash') == 5
    assert not result.state.event_log.get_by_type('action_resolved')
    assert not result.state.event_log.get_by_type('sequence_started')
    assert not result.state.event_log.get_by_type('action_attempted')
    assert result.state.event_log.get_by_type('action_failed')
    assert not result.state.messages.get_for_entity('seller')


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
@pytest.mark.parametrize('chosen', ['seller_only', ' SELLER_ONLY ', 'seller_onl'])
def test_foreign_role_cannot_execute_even_with_name_correction(mode, chosen):
    result = build(mode, chosen)
    assert result.state.get_valid_actions('buyer') == ['wait', 'spend']
    run_and_assert_rejected(result)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
@pytest.mark.parametrize('restriction', ['cooldown', 'locked', 'prerequisite', 'status'])
@pytest.mark.parametrize('sequence', [0, 3])
def test_unavailable_choice_cannot_execute_or_start_sequence(mode, restriction, sequence):
    result = build(mode, sequence=sequence)
    state = result.state
    if restriction == 'cooldown':
        state.action_history.set_cooldown('buyer', 'spend', 10)
    elif restriction == 'locked':
        state.action_history.lock('buyer', ['spend'])
    elif restriction == 'prerequisite':
        state.action_definitions['spend'].requires_action = 'wait'
    else:
        state.status_effects.apply('buyer', StatusEffectDefinition('blocked', duration=10,
                                   blocks_actions=['spend']), round_num=0)
    assert state.get_valid_actions('buyer') == ['wait']
    run_and_assert_rejected(result)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
@pytest.mark.parametrize('target', [None, 'missing', 'seller'])
@pytest.mark.parametrize('sequence', [0, 3])
def test_required_target_is_checked_before_sequence_and_effects(mode, target, sequence):
    result = build(mode, target=target, sequence=sequence)
    result.state.action_definitions['spend'].target_type = 'Stock'
    run_and_assert_rejected(result)


class RejectSpend(DomainModule):
    @property
    def custom_actions(self):
        return ['spend']

    def tick(self, state, round_number):
        return []

    def validate_action(self, action_name, actor, target, state):
        return 'Budget unavailable' if action_name == 'spend' else None


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
@pytest.mark.parametrize('sequence', [0, 3])
def test_domain_validation_cannot_be_bypassed(mode, sequence):
    result = build(mode, sequence=sequence)
    result.state.domain_modules = DomainModuleManager()
    result.state.domain_modules.add_module(RejectSpend('budget'))
    run_and_assert_rejected(result)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
@pytest.mark.parametrize('chosen', ['spend', ' SPEND ', 'spen'])
def test_valid_choices_and_unambiguous_corrections_still_execute(mode, chosen):
    result = build(mode, chosen, target='stock')
    result.state.action_definitions['spend'].target_type = 'Stock'
    result.engine.max_rounds = 1
    result.engine.run()
    assert result.state.entities['buyer'].get('cash') == 4
    assert len(result.state.event_log.get_by_type('action_resolved')) == 1


class FilterSpend(RejectSpend):
    def filter_valid_actions(self, entity_id, valid_actions, state):
        return [name for name in valid_actions if name != 'spend']

    def validate_action(self, action_name, actor, target, state):
        return None


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
def test_domain_menu_restriction_is_enforced_at_execution(mode):
    result = build(mode)
    result.state.domain_modules = DomainModuleManager()
    result.state.domain_modules.add_module(FilterSpend('filter'))
    run_and_assert_rejected(result)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
@pytest.mark.parametrize('change', ['dead', 'locked'])
def test_choice_is_revalidated_after_decision_collection(mode, change):
    result = build(mode)
    def decide(actor, perception, valid):
        assert 'spend' in valid
        if change == 'dead':
            result.state.entities[actor].alive = False
        else:
            result.state.action_history.lock(actor, ['spend'])
        return ActionInstance(action_name='spend', actor_id=actor)
    result.engine.decision_fn = decide
    run_and_assert_rejected(result)


def test_valid_sequence_completes_and_cooldown_still_recovers():
    result = build('sequential', sequence=2)
    result.state.action_definitions['spend'].cooldown_rounds = 2
    result.engine.max_rounds = 8
    result.engine.run()
    assert len(result.state.event_log.get_by_type('sequence_started')) == 2
    assert len(result.state.event_log.get_by_type('action_resolved')) == 2
    assert result.state.entities['buyer'].get('cash') == 3
