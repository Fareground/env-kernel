import copy

import pytest

from fg_env_kernel import compile_template, export_kernel_contract
from fg_env_kernel.action import ActionInstance
from fg_env_kernel.resolution import ResolutionResult
from fg_env_kernel.transfers import TransferError, apply_action_effects, settle_transfers


def world(balance=100, *, amount=90, maximum=None, mode="sequential"):
    return {
        "name": "Conserved payment",
        "entity_types": [
            {"name": "Buyer", "role": "agent", "properties": [
                {"name": "cash", "type": "float", "default": balance, "min_value": 0},
                {"name": "paid", "type": "bool", "default": False}]},
            {"name": "Seller", "role": "object", "properties": [
                {"name": "cash", "type": "float", "default": 1000, "max_value": maximum}]}],
        "entities": [{"id": "buyer", "entity_type": "Buyer"},
                     {"id": "seller", "entity_type": "Seller"}],
        "temporal": {"phases": [{"name": "act", "resolution_mode": mode}]},
        "actions": [{"name": "pay", "actor_type": "Buyer", "target_type": "Seller",
            "transfers": [{"source": "actor", "target": "target", "field": "cash", "amount": amount}],
            "effects_on_success": [{"operation": "set", "target": "actor", "field": "paid", "value": True}]}],
        "termination_conditions": [{"name": "one", "check_type": "round_limit", "params": {"max_rounds": 1}}],
    }


def compiled(data):
    result = compile_template(data, decision_fn=lambda eid, *_:
        ActionInstance(action_name="pay", actor_id=eid, target_id="seller"))
    assert result.ok, result.errors
    return result


def balances(state):
    return {eid: entity.get("cash") for eid, entity in state.entities.items()}


@pytest.mark.parametrize("mode", ["sequential", "simultaneous"])
@pytest.mark.parametrize("balance,expected,success", [(100, {"buyer": 10, "seller": 1090}, True),
    (10, {"buyer": 10, "seller": 1000}, False), (90, {"buyer": 0, "seller": 1090}, True)])
def test_payment_never_creates_cash_and_success_effects_follow_settlement(mode, balance, expected, success):
    result = compiled(world(balance, mode=mode))
    result.engine.run()
    assert balances(result.state) == expected
    assert sum(expected.values()) == balance + 1000
    assert result.state.entities["buyer"].get("paid") is success
    events = result.state.event_log.get_by_type("action_resolved")
    assert len(events) == 1 and events[0].data["success"] is success
    if not success:
        assert events[0].data["details"]["transfer_error"] == "insufficient_transfer_balance"
        assert events[0].data["state_changes"] == []


def test_recipient_capacity_rejects_whole_payment():
    result = compiled(world(maximum=1050))
    result.engine.run()
    assert balances(result.state) == {"buyer": 100, "seller": 1000}
    assert not result.state.entities["buyer"].get("paid")


@pytest.mark.parametrize("selector", ["$entity(buyer)", "$params.payer"])
def test_dynamic_account_selectors(selector):
    result = compiled(world())
    changes = settle_transfers(result.state, [{"source": selector, "target": "seller",
        "field": "cash", "amount": 90}], result.state.entities["buyer"], None, {"payer": "buyer"})
    assert len(changes) == 2
    assert balances(result.state) == {"buyer": 10, "seller": 1090}


def test_explicit_credit_floor_and_default_zero_floor():
    data = world(10)
    data["entity_types"][0]["properties"][0]["min_value"] = -100
    result = compiled(data)
    result.engine.run()
    assert balances(result.state) == {"buyer": -80, "seller": 1090}
    data["entity_types"][0]["properties"][0].pop("min_value")
    result = compiled(data)
    result.engine.run()
    assert balances(result.state) == {"buyer": 10, "seller": 1000}


def test_integer_balances_above_float_precision_remain_exact():
    initial = 10**30
    data = world(initial, amount=1)
    for kind in data["entity_types"]:
        kind["properties"][0]["type"] = "int"
    result = compiled(data)
    result.engine.run()
    assert balances(result.state) == {"buyer": initial - 1, "seller": 1001}
    assert sum(balances(result.state).values()) == initial + 1000


def test_all_amounts_resolve_once_before_any_account_changes():
    data = world(amount={"expr": "$actor.cash / 2"})
    data["actions"][0]["transfers"] *= 2
    result = compiled(data)
    result.engine.run()
    assert balances(result.state) == {"buyer": 0, "seller": 1100}


def test_aggregate_debits_are_checked_not_individual_transfers():
    data = world(100, amount=60)
    data["actions"][0]["transfers"] *= 2
    result = compiled(data)
    result.engine.run()
    assert balances(result.state) == {"buyer": 100, "seller": 1000}
    assert not result.state.entities["buyer"].get("paid")


def test_simultaneous_contenders_cannot_spend_the_same_balance_twice():
    data = world(mode="simultaneous")
    data["entities"].extend([{"id": "buyer2", "entity_type": "Buyer"},
                             {"id": "bank", "entity_type": "Seller", "properties": {"cash": 100}}])
    data["actions"][0]["transfers"][0]["source"] = "bank"
    result = compiled(data)
    result.engine.run()
    assert result.state.entities["bank"].get("cash") == 10
    assert result.state.entities["seller"].get("cash") == 1090
    outcomes = result.state.event_log.get_by_type("action_resolved")
    assert sorted(event.data["success"] for event in outcomes) == [False, True]


@pytest.mark.parametrize("amount", [None, True, -1, "ninety", float("inf"),
    {"expr": "__import__('os')"}, {"expr": "$actor.cash", "extra": 1}])
def test_invalid_transfer_amounts_fail_compilation(amount):
    result = compile_template(world(amount=amount))
    assert not result.ok
    assert any("transfers" in issue.path or "transfers" in issue.message for issue in result.errors)


@pytest.mark.parametrize("change", [{"source": "missing"}, {"field": "misspelled_cash"}, {"field": "paid"}])
def test_static_transfer_references_are_checked(change):
    data = world()
    data["actions"][0]["transfers"][0].update(change)
    result = compile_template(data)
    assert not result.ok
    assert any("transfers" in issue.path for issue in result.errors)


@pytest.mark.parametrize("amount,code", [("$actor.missing", "invalid_transfer_value"),
    ("-$actor.cash", "negative_transfer_amount"), ("$actor.cash / 0", "invalid_transfer_value")])
def test_dynamic_value_failure_changes_nothing_and_does_not_expose_balances(amount, code):
    result = compiled(world(amount=amount))
    result.engine.run()
    assert balances(result.state) == {"buyer": 100, "seller": 1000}
    event = result.state.event_log.get_by_type("action_resolved")[0]
    assert event.data["details"]["transfer_error"] == code
    assert "1000" not in str(event.data["details"])


@pytest.mark.parametrize("amount", [0, 1 / 3, 0.1])
def test_zero_and_fractional_transfers(amount):
    result = compiled(world(0.3, amount=amount))
    result.state.entities["seller"].set("cash", 0)
    if amount > 0.3:
        result.state.entities["buyer"].set("cash", 1)
    before = sum(balances(result.state).values())
    result.engine.run()
    assert result.state.entities["buyer"].get("paid")
    assert sum(balances(result.state).values()) == pytest.approx(before, abs=1e-14)


def test_decimal_boundary_does_not_spuriously_overdraw():
    data = world(0.3, amount=0.1)
    data["actions"][0]["transfers"].append({"source": "actor", "target": "target", "field": "cash", "amount": 0.2})
    result = compiled(data)
    result.state.entities["seller"].set("cash", 0)
    result.engine.run()
    assert balances(result.state) == {"buyer": 0, "seller": 0.3}


def test_transfer_too_small_for_float_precision_is_rejected():
    result = compiled(world(1e20, amount=0.01))
    result.engine.run()
    assert balances(result.state) == {"buyer": 1e20, "seller": 1000}
    assert not result.state.entities["buyer"].get("paid")


def test_integer_accounts_reject_fractional_amounts():
    data = world(amount=0.5)
    data["entity_types"][0]["properties"][0]["type"] = "int"
    result = compiled(data)
    result.engine.run()
    assert balances(result.state) == {"buyer": 100, "seller": 1000}
    assert not result.state.entities["buyer"].get("paid")


@pytest.mark.parametrize("partial", [False, True])
def test_unsuccessful_resolution_does_not_transfer(partial):
    result = compiled(world())
    outcome = ResolutionResult(success=False, partial=partial)
    changes = apply_action_effects(result.engine, result.state.action_definitions["pay"],
        result.state.entities["buyer"], result.state.entities["seller"], {}, outcome)
    assert changes == []
    assert balances(result.state) == {"buyer": 100, "seller": 1000}


@pytest.mark.parametrize("balance,success", [(100, True), (10, False)])
def test_invoked_action_cannot_bypass_transfer_constraints(balance, success):
    data = world(balance)
    data["actions"][0]["name"] = "settle"
    data["actions"].append({"name": "pay", "actor_type": "Buyer", "target_type": "Seller",
        "effects_on_success": [{"operation": "invoke_action", "target": "target", "value": {"action_name": "settle"}}]})
    result = compiled(data)
    result.engine.run()
    assert result.state.entities["buyer"].get("paid") is success
    assert sum(balances(result.state).values()) == balance + 1000
    if not success:
        assert balances(result.state) == {"buyer": 10, "seller": 1000}
        assert result.state.event_log.get_by_type("action_failed")[0].data["reason"] == "transfer_rejected"


def test_failed_batch_has_no_partial_writes():
    result = compiled(world())
    before = copy.deepcopy(balances(result.state))
    transfers = [{"source": "buyer", "target": "seller", "field": "cash", "amount": 10},
                 {"source": "buyer", "target": "seller", "field": "cash", "amount": "$params.missing"}]
    with pytest.raises(TransferError):
        settle_transfers(result.state, transfers, None, None, {})
    assert balances(result.state) == before


def test_installed_contract_exposes_typed_transfers():
    schema = export_kernel_contract()["template_schema"]
    assert schema["$defs"]["ActionSpec"]["properties"]["transfers"]["items"]["$ref"].endswith("PropertyTransferSpec")
    assert schema["$defs"]["PropertyTransferSpec"]["required"] == ["field", "amount"]
