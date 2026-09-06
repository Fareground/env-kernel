import pytest
from fg_env_kernel.pipeline.compile import compile_template


@pytest.mark.parametrize("value,ok", [("cool", True), ("heat", True), ("invalid", False), ("", False)])
def test_compile_enforces_initial_enum_values(value, ok):
    schema = {"name": "Modes", "entity_types": [{"name": "Room", "role": "agent", "properties": [
        {"name": "mode", "type": "enum", "default": "cool", "enum_values": ["heat", "cool"]}]}],
        "entities": [{"id": "room", "entity_type": "Room", "properties": {"mode": value}}],
        "actions": [{"name": "wait", "actor_type": "Room"}],
        "termination_conditions": [{"name": "end", "check_type": "round_limit", "params": {"max_rounds": 1}}]}
    result = compile_template(schema)
    assert result.ok is ok
    if not ok:
        assert any("room.mode" in issue.message for issue in result.errors)
