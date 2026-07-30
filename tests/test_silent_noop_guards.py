"""Silent no-op guards: failing predicates log; typo'd spec fields lint."""
from __future__ import annotations

import logging

from fg_env_kernel import predicates
from fg_env_kernel.pipeline.lint import lint_template


class _Boom:
    """A predicate whose evaluation genuinely raises."""

    def __bool__(self):  # pragma: no cover - trivially exercised below
        raise RuntimeError("boom")

    def __repr__(self):
        return "<boom-predicate>"


def test_failing_predicate_logs_once(caplog):
    predicates._LOGGED_FAILURES.clear()
    bad = _Boom()
    with caplog.at_level(logging.WARNING, logger="fg_env_kernel.predicates"):
        assert predicates.evaluate(bad) is False
        assert predicates.evaluate(bad) is False
    hits = [r for r in caplog.records if "fail-closed" in r.getMessage()]
    assert len(hits) == 1  # logged once, not per evaluation


def _template(**over) -> dict:
    base = {
        "entity_types": [{
            "name": "player", "role": "agent",
            "properties": [{"name": "gold", "type": "float"}],
        }],
        "entities": [{"id": "p1", "entity_type": "player"}],
        "actions": [{
            "name": "mine",
            "actor_type": "player",
            "effects_on_success": [{"operation": "add", "field": "gold",
                                    "value": 1}],
        }],
        "termination_conditions": [{"check_type": "max_rounds"}],
    }
    base.update(over)
    return base


def test_clean_template_has_no_unknown_field_warnings():
    issues = lint_template(_template())
    assert not [i for i in issues if "unknown field" in i.message]


def test_typoed_action_field_is_flagged():
    t = _template()
    t["actions"][0]["effcts_on_success"] = t["actions"][0].pop("effects_on_success")
    issues = lint_template(t)
    flagged = [i for i in issues if "unknown field 'effcts_on_success'" in i.message]
    assert flagged and flagged[0].severity == "warning"
    assert "actions[0]" in flagged[0].path


def test_typoed_nested_property_field_is_flagged():
    t = _template()
    t["entity_types"][0]["properties"][0]["typ"] = "float"
    issues = lint_template(t)
    assert any("unknown field 'typ'" in i.message for i in issues)


def test_typoed_subcondition_field_is_flagged():
    t = _template(termination_conditions=[{
        "check_type": "compound_and",
        "sub_conditions": [{"check_typ": "max_rounds"}],
    }])
    issues = lint_template(t)
    assert any(
        "unknown field 'check_typ'" in i.message
        and "sub_conditions[0]" in i.path
        for i in issues
    )


def test_broadcast_false_reaches_the_runtime_action():
    """The loader must pass visibility through — a hidden night-kill that
    defaults back to broadcast=True is a silent information leak."""
    from fg_env_kernel.pipeline.loader import build_world_state

    t = _template()
    t["actions"][0]["broadcast"] = False
    t["actions"][0]["unlocks_actions"] = ["celebrate"]
    state = build_world_state(t)
    action = state.action_definitions["mine"]
    assert action.broadcast is False
    assert action.unlocks_actions == ["celebrate"]


def test_world_scope_property_threshold_fires():
    from fg_env_kernel.termination import _check_property_threshold
    from fg_env_kernel.pipeline.loader import build_world_state

    state = build_world_state(_template())
    params = {"property": "verdict_recorded", "scope": "world",
              "operator": ">=", "value": 1}
    assert _check_property_threshold(state, params, None) is False
    state.properties["verdict_recorded"] = 1
    assert _check_property_threshold(state, params, None) is True


def test_world_scope_threshold_reads_legacy_attribute():
    from fg_env_kernel.termination import _check_property_threshold
    from fg_env_kernel.pipeline.loader import build_world_state

    state = build_world_state(_template())
    state.verdict_recorded = 1
    params = {"property": "verdict_recorded", "scope": "world",
              "operator": ">=", "value": 1}
    assert _check_property_threshold(state, params, None) is True


def test_world_properties_survive_snapshot_roundtrip():
    from fg_env_kernel.pipeline.loader import build_world_state

    state = build_world_state(_template())
    state.properties["market_halted"] = True
    snap = state.to_dict()
    restored = build_world_state(_template())
    restored.apply_snapshot(snap)
    assert restored.properties.get("market_halted") is True


def test_non_broadcast_action_events_carry_visible_to():
    from fg_env_kernel.pipeline.loader import build_world_state
    from fg_env_kernel.runtime.engine import SimulationEngine

    t = _template()
    t["actions"][0]["broadcast"] = False
    state = build_world_state(t)

    def decide(entity_id, perception, valid_actions):
        from fg_env_kernel.action import ActionInstance
        return ActionInstance(action_name="mine", actor_id=entity_id)

    engine = SimulationEngine(state=state, decision_fn=decide,
                              max_rounds=1, seed=1)
    engine.run()
    hidden = [e for e in state.event_log.get_by_type("action_resolved")
              if e.action_name == "mine"]
    assert hidden, "action should have resolved"
    assert hidden[0].data.get("visible_to") == ["p1"]


def test_negative_predicates_fail_closed_on_unresolved():
    predicates._LOGGED_FAILURES.clear()
    ctx_entity = type("E", (), {"properties": {"gold": 5}, "get": lambda s, k, d=None: s.properties.get(k, d)})()
    # Typo'd path resolves to None → negative ops must NOT fire.
    assert predicates.evaluate("$actor.golde != 0", actor=ctx_entity) is False
    assert predicates.evaluate("!$actor.golde", actor=ctx_entity) is False
    # Legit negatives still work.
    assert predicates.evaluate("$actor.gold != 0", actor=ctx_entity) is True


def test_unknown_precondition_operator_raises_at_load():
    import pytest
    from fg_env_kernel.pipeline.loader import build_world_state

    t = _template()
    t["actions"][0]["preconditions"] = [
        {"field": "gold", "operator": "gte_typo", "value": 1}]
    with pytest.raises(ValueError, match="unknown precondition operator"):
        build_world_state(t)


def test_unknown_effect_operation_raises_at_load():
    import pytest
    from fg_env_kernel.pipeline.loader import build_world_state

    t = _template()
    t["actions"][0]["effects_on_success"] = [
        {"operation": "increment_typo", "field": "gold", "value": 1}]
    with pytest.raises(ValueError, match="unknown effect operation"):
        build_world_state(t)


def test_lint_flags_unknown_operator_and_effect_op():
    t = _template()
    t["actions"][0]["preconditions"] = [
        {"field": "gold", "operator": "gte_typo", "value": 1}]
    t["actions"][0]["effects_on_success"] = [
        {"operation": "increment_typo", "field": "gold", "value": 1}]
    issues = lint_template(t)
    assert any("unknown precondition operator" in i.message and i.severity == "error"
               for i in issues)
    assert any("unknown effect operation" in i.message and i.severity == "error"
               for i in issues)


def test_lint_recurses_subconditions_and_root_fields():
    t = _template(termination_conditions=[{
        "name": "x", "check_type": "compound_or",
        "sub_conditions": [{"name": "y", "check_type": "not_a_real_type"}],
    }])
    t["termination_conditons"] = []  # root-level typo
    issues = lint_template(t)
    assert any("sub_conditions[0].check_type" in i.path for i in issues)
    assert any("termination_conditons" in i.path for i in issues)


def test_new_termination_checkers_fire():
    from fg_env_kernel.pipeline.loader import build_world_state
    from fg_env_kernel.registry import registry

    state = build_world_state(_template())
    state.temporal.current_round = 7
    assert registry.terminations.try_get("max_rounds")(state, {"max_rounds": 5}, None) is True
    assert registry.terminations.try_get("max_rounds_reached")(state, {"max_rounds": 9}, None) is False

    # property_stable: converges after `window` stable deltas.
    check = registry.terminations.try_get("property_stable")
    state.properties["price"] = 100.0
    params = {"property": "price", "window": 2, "threshold": 0.01}
    assert check(state, params, None) is False
    state.properties["price"] = 100.2
    assert check(state, params, None) is False
    state.properties["price"] = 100.3
    assert check(state, params, None) is True


def test_dict_not_fails_closed_on_unresolved_operand():
    predicates._LOGGED_FAILURES.clear()
    entity = type("E", (), {"properties": {"gold": 5},
                            "get": lambda s, k, d=None: s.properties.get(k, d)})()
    bad = {"op": "not", "child": {"op": "gte", "left": "$actor.golde", "right": 1}}
    assert predicates.evaluate(bad, actor=entity) is False
    good = {"op": "not", "child": {"op": "gte", "left": "$actor.gold", "right": 100}}
    assert predicates.evaluate(good, actor=entity) is True


def test_engine_compare_fails_closed_on_type_mismatch_and_unknown_op():
    from fg_env_kernel.runtime.engine import SimulationEngine

    assert SimulationEngine._compare("high", "gte", 5) is False
    assert SimulationEngine._compare(5, "ltee", 3) is False
    assert SimulationEngine._compare(5, "gte", 3) is True
    assert SimulationEngine._compare(5, "ne", 3) is True


def test_nested_not_still_fails_closed():
    entity = type("E", (), {"properties": {"gold": 5},
                            "get": lambda s, k, d=None: s.properties.get(k, d)})()
    nested = {"op": "not", "child": {"op": "and", "children": [
        {"op": "gte", "left": "$actor.golde", "right": 1}]}}
    assert predicates.evaluate(nested, actor=entity) is False


def test_state_snapshots_emitted_when_enabled():
    from fg_env_kernel.pipeline.loader import build_world_state
    from fg_env_kernel.runtime.engine import SimulationEngine

    events = []
    state = build_world_state(_template())

    def decide(entity_id, perception, valid_actions):
        from fg_env_kernel.action import ActionInstance
        return ActionInstance(action_name="mine", actor_id=entity_id)

    engine = SimulationEngine(state=state, decision_fn=decide, max_rounds=2,
                              seed=1, on_event=events.append,
                              emit_state_snapshots=True)
    engine.run()
    snaps = [e for e in events if e.get("event_type") == "state_snapshot"]
    assert snaps, "expected state_snapshot events"
    ent = snaps[-1]["data"]["entities"]["p1"]
    assert ent["type"] == "player" and "gold" in ent["properties"]

    # Off by default — no snapshots without the flag.
    events2 = []
    state2 = build_world_state(_template())
    SimulationEngine(state=state2, decision_fn=decide, max_rounds=1, seed=1,
                     on_event=events2.append).run()
    assert not [e for e in events2 if e.get("event_type") == "state_snapshot"]
