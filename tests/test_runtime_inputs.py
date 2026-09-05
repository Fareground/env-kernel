import copy

import pytest

from fg_env_kernel.pipeline.loader import WorldTemplate, build_world_state


@pytest.mark.parametrize('runtime', [
    {'capacity': 7, 'closed': False, 'label': 'A'},
    [{'name': 'capacity', 'default': 7}, {'name': 'closed', 'default': False}, {'name': 'label', 'default': 'A'}],
])
def test_runtime_inputs_accept_values_or_descriptors_without_losing_saved_overrides(runtime):
    schema = {'name': 'Runtime inputs', 'runtime_parameters': runtime,
              'last_runtime_params': {'capacity': 9}, 'tables': {'unchanged': {'x': 1}}}
    original = copy.deepcopy(schema)
    validated = WorldTemplate(**schema).model_dump()
    state = build_world_state(validated)
    assert state.tables['runtime'] == {'capacity': 9, 'closed': False, 'label': 'A'}
    assert state.tables['unchanged'] == {'x': 1}
    assert schema == original
