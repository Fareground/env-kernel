"""Observation APIs never fall back to private owner setup text."""
import pytest
from fg_env_kernel import compile_template
from fg_env_kernel.runtime.perception import build_perception


@pytest.mark.parametrize('briefing', [None, 'Inspect to learn the concealed status.'])
def test_private_world_observation_uses_only_participant_briefing(briefing):
    schema = {'name':'Investigation', 'description':'Payroll is compromised.',
        'rules':'The hidden answer is payroll.',
        'entity_types':[{'name':'Analyst','role':'agent','properties':[
            {'name':'secret','type':'bool','default':True,'hidden':True}]}],
        'entities':[{'id':'analyst','name':'Analyst','entity_type':'Analyst'}]}
    if briefing is not None:
        schema['participant_briefing'] = briefing
    result = compile_template(schema)
    assert result.ok
    perception, _ = build_perception(result.engine, 'analyst')
    assert perception['world_brief']['rules'] == (briefing or '')
    assert perception['world_brief']['description'] == ''
    assert 'payroll' not in str(perception['world_brief']).lower()


def test_public_legacy_world_keeps_rules_and_description():
    result = compile_template({'name':'Public','description':'A queue.', 'rules':'Serve in order.', 'entity_types':[{'name':'Item','role':'agent','properties':[]}], 'entities':[{'id':'item','name':'Item','entity_type':'Item'}], 'termination_conditions':[{'name':'end','check_type':'round_limit','params':{'max_rounds':1}}]})
    assert result.ok
    assert result.state._world_brief == {'name':'Public','description':'A queue.','rules':'Serve in order.'}
