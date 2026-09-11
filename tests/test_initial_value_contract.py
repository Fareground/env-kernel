"""Starting-state expressions must not silently become inert numeric objects."""
import copy

import pytest

from fg_env_kernel import compile_template
from fg_env_kernel.pipeline.loader import load_world


def world(kind="float", value=10, *, default=False):
    return {
        "name": "Initial values",
        "runtime_parameters": [{"name": "starting_stock", "type": "int", "default": 10}],
        "entity_types": [{"name": "Store", "role": "agent", "properties": [
            {"name": "stock", "type": kind, "default": value if default else None},
        ]}],
        "entities": [{"id": "store", "entity_type": "Store",
                      "properties": {} if default else {"stock": value}}],
        "termination_conditions": [{"name": "end", "check_type": "round_limit",
                                    "params": {"max_rounds": 1}}],
    }


@pytest.mark.parametrize("kind,value", [
    ("float", {"expr": "$starting_stock"}),
    ("int", {"expr": "$lookup(runtime, starting_stock)"}),
    ("float", "$starting_stock"), ("int", "ten"), ("float", []),
    ("float", True), ("int", 1.5), ("float", float("inf")),
    ("bool", {"expr": "true"}), ("bool", "false"), ("bool", 0),
])
@pytest.mark.parametrize("default", [False, True])
def test_compile_reports_exact_initial_field_and_repair_without_mutation(kind, value, default):
    schema = world(kind, value, default=default)
    before = copy.deepcopy(schema)
    result = compile_template(schema)
    assert not result.ok
    path = "entity_types[0].properties[0].default" if default else "entities[0].properties.stock"
    issue = next(issue for issue in result.errors if issue.path == path)
    assert "$lookup(runtime, parameter_name)" in issue.hint
    assert schema == before


@pytest.mark.parametrize("default", [False, True])
def test_skipped_lint_and_direct_loader_cannot_bypass_scalar_validation(default):
    schema = world("float", {"expr": "$starting_stock"}, default=default)
    result = compile_template(schema, skip_lint=True)
    assert not result.ok
    assert "store.stock" in result.errors[0].message
    with pytest.raises(ValueError, match="store.stock"):
        load_world(schema)


@pytest.mark.parametrize("kind,value", [
    ("int", 0), ("int", -1), ("float", 2.5), ("float", 10),
    ("bool", False), ("int", None), ("string", "literal $starting_stock"),
    ("list", [{"expr": "stored data, not executable code"}]),
    ("json", {"expr": "stored data, not executable code"}), ("json", []),
])
def test_valid_literals_and_literal_payloads_are_preserved(kind, value):
    result = compile_template(world(kind, value))
    assert result.ok, result.errors
    assert result.state.entities["store"].properties["stock"] == value


@pytest.mark.parametrize("default", [False, True])
def test_canonical_runtime_binding_remains_editable_and_numeric(default):
    schema = world("int", "$lookup(runtime, starting_stock)", default=default)
    schema["last_runtime_params"] = {"starting_stock": 7}
    before = copy.deepcopy(schema)
    result = compile_template(schema)
    assert result.ok, result.errors
    assert result.state.entities["store"].properties["stock"] == 7
    assert schema == before


def test_nan_is_rejected():
    result = compile_template(world("float", float("nan")))
    assert not result.ok
    assert "finite" in result.errors[0].message
