import pytest
from fg_env_kernel import compile_template
from fg_env_kernel.action import ActionInstance


@pytest.mark.parametrize('quantity,accepted', [(2, True), (8, True), (9, False), (11, False)])
def test_target_preconditions_receive_submitted_parameters(quantity, accepted):
    schema = {'name': 'Parameterized purchase', 'entity_types': [
        {'name':'Buyer','role':'agent','properties':[{'name':'cash','type':'int','default':40}]},
        {'name':'Stock','role':'object','properties':[{'name':'quantity','type':'int','default':8},
                                                  {'name':'price','type':'int','default':5}]}],
        'entities':[{'id':'buyer','name':'Buyer','entity_type':'Buyer'}, {'id':'stock','name':'Stock','entity_type':'Stock'}],
        'actions':[{'name':'buy','actor_type':'Buyer','target_type':'Stock','resolution_archetype':'deterministic',
            'parameters':[{'name':'quantity','type':'int','min':1,'max':20}],
            'preconditions':[{'expr':'$params.quantity <= $target.quantity'},
                             {'expr':'$params.quantity * $target.price <= $actor.cash'}],
            'effects_on_success':[{'operation':'subtract','target':'actor','field':'cash',
                                   'value':{'expr':'$params.quantity * $target.price'}},
                                  {'operation':'subtract','target':'target','field':'quantity','value':'$params.quantity'}]}]}
    result = compile_template(schema, decision_fn=lambda *args: ActionInstance(
        action_name='buy',actor_id='buyer',target_id='stock',parameters={'quantity':quantity}))
    assert result.ok
    result.engine.max_rounds=1
    result.engine.run()
    assert result.state.entities['buyer'].get('cash') == (40-quantity*5 if accepted else 40)
    assert result.state.entities['stock'].get('quantity') == (8-quantity if accepted else 8)
    assert len(result.state.event_log.get_by_type('action_resolved')) == int(accepted)


@pytest.mark.parametrize('mode', ['sequential', 'simultaneous'])
@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('params,accepted', [({'quantity': 2}, True), ({'quantity': 8}, True),
                                           ({'quantity': 9}, False), ({'quantity': -1}, False), ({}, False)])
def test_parameter_only_guards_wait_for_the_decision_but_still_gate_resolution(mode, structured, params, accepted):
    guard = ({'op': 'and', 'children': [
        {'op': 'gte', 'left': '$params.quantity', 'right': 1},
        {'op': 'lte', 'left': '$params.quantity', 'right': '$actor.cash'}]}
        if structured else '$params.quantity >= 1 && $params.quantity <= $actor.cash')
    schema = {'name': 'Parameterized spend', 'entity_types': [
        {'name': 'Buyer', 'role': 'agent', 'properties': [{'name': 'cash', 'type': 'int', 'default': 8}]}],
        'entities': [{'id': 'buyer', 'entity_type': 'Buyer'}],
        'temporal': {'phases': [{'name': 'spend', 'resolution_mode': mode}]},
        'actions': [{'name': 'spend', 'actor_type': 'Buyer', 'parameters': [
            {'name': 'quantity', 'type': 'int', 'min_value': -20, 'max_value': 20}],
            'preconditions': [{'expr': '$actor.cash > 0'}, {'expr': guard}],
            'effects_on_success': [{'operation': 'subtract', 'target': 'actor', 'field': 'cash', 'value': '$params.quantity'}]}]}
    selected = []
    def decide(actor_id, perception, valid):
        selected.append(valid)
        return ActionInstance(action_name='spend', actor_id=actor_id, parameters=params)
    result = compile_template(schema, decision_fn=decide)
    assert result.ok
    assert result.state.get_valid_actions('buyer') == ['spend']
    result.engine.max_rounds = 1
    result.engine.run()
    assert selected == [['spend']]
    assert result.state.entities['buyer'].get('cash') == (8 - params['quantity'] if accepted else 8)
    assert len(result.state.event_log.get_by_type('action_resolved')) == int(accepted)
    result.state.entities['buyer'].set('cash', 0)
    assert result.state.get_valid_actions('buyer') == []  # actor-only guard still filters
