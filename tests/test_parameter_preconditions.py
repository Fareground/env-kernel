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
