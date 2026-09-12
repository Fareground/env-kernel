"""Conserved property transfers: validate a small ledger, then commit it once.

Work is O(entries + touched accounts), independent of world/history size.
This does not clone the engine or reinterpret ordinary clamped properties.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Any

from .effect_values import EffectValueError, number, resolve_value
from .types import PropertyType

if TYPE_CHECKING:
    from .action import ActionDefinition
    from .entity import Entity, PropertySchema
    from .resolution import ResolutionResult
    from .runtime.engine import SimulationEngine
    from .state import WorldState


class TransferError(ValueError):
    """Stable, value-free diagnostic suitable for an actor's outcome."""


def settle_transfers(state: WorldState, transfers: list[dict[str, Any]],
                     actor: Entity | None, target: Entity | None,
                     params: dict[str, Any]) -> list[dict]:
    if not transfers:
        return []
    if not isinstance(transfers, list) or len(transfers) > 100:
        raise TransferError("invalid_transfer_list")
    context = dict(state=state, actor=actor, target=target, params=params)
    accounts: dict[tuple[str, str], tuple[Entity, PropertySchema, int | float]] = {}
    deltas: dict[tuple[str, str], Decimal] = {}

    def entity(ref):
        if ref == "actor":
            found = actor
        elif ref == "target":
            found = target
        elif isinstance(ref, str) and ref.startswith("$"):
            resolved = resolve_value(ref, **context)
            found = state.get_entity(resolved if isinstance(resolved, str) else getattr(resolved, "id", None))
        else:
            found = state.get_entity(ref) if isinstance(ref, str) else None
        if found is None or not found.alive:
            raise TransferError("invalid_transfer_entity")
        return found

    def account(ref, field):
        owner = entity(ref)
        key = (owner.id, field)
        if key not in accounts:
            kind = state.entity_types.get(owner.entity_type)
            prop = kind.get_property_schema(field) if kind else None
            if prop is None or prop.type not in (PropertyType.INT, PropertyType.FLOAT):
                raise TransferError("invalid_transfer_property")
            balance = number(owner.get(field))
            if prop.type == PropertyType.INT and balance != int(balance):
                raise TransferError("fractional_integer_balance")
            accounts[key] = (owner, prop, balance)
            deltas[key] = Decimal(0)
        return key

    try:
        # Decimal arithmetic avoids 0.3 - 0.1 - 0.2 falsely crossing a zero
        # floor. Existing kernel state stays int/float, never Decimal objects.
        with localcontext() as decimal_context:
            decimal_context.prec = 700
            for transfer in transfers:
                if not isinstance(transfer, dict) or set(transfer) - {"source", "target", "field", "amount"}:
                    raise TransferError("invalid_transfer")
                field = transfer.get("field")
                if not isinstance(field, str) or not field:
                    raise TransferError("invalid_transfer_property")
                source = account(transfer.get("source", "actor"), field)
                destination = account(transfer.get("target", "target"), field)
                if source == destination:
                    raise TransferError("same_transfer_account")
                amount = number(resolve_value(transfer.get("amount"), **context))
                if amount < 0:
                    raise TransferError("negative_transfer_amount")
                if any(accounts[key][1].type == PropertyType.INT for key in (source, destination)) and amount != int(amount):
                    raise TransferError("fractional_integer_transfer")
                debit = Decimal(str(amount))
                deltas[source] -= debit
                deltas[destination] += debit

            pending = []
            for key, (owner, prop, balance) in accounts.items():
                projected = Decimal(str(balance)) + deltas[key]
                minimum = Decimal(str(number(prop.min_value))) if prop.min_value is not None else Decimal(0)
                maximum = Decimal(str(number(prop.max_value))) if prop.max_value is not None else None
                if projected < minimum:
                    raise TransferError("insufficient_transfer_balance")
                if maximum is not None and projected > maximum:
                    raise TransferError("transfer_capacity_exceeded")
                value = int(projected) if prop.type == PropertyType.INT else float(projected)
                number(value)
                # Allow ordinary floating roundoff, but not payments swallowed
                # by a huge balance's precision. Error is bounded relative to
                # the transferred delta, NOT the much larger account balance.
                tolerance = abs(deltas[key]) * Decimal("1e-12")
                if abs(Decimal(str(value)) - projected) > tolerance:
                    raise TransferError("transfer_precision_loss")
                pending.append((owner, key[1], balance, value))
    except TransferError:
        raise
    except (EffectValueError, ArithmeticError, TypeError, ValueError) as exc:
        raise TransferError("invalid_transfer_value") from exc

    # No mutation or events until EVERY transfer and final balance is valid.
    changes = []
    for owner, field, old, new in pending:
        owner.set(field, new)
        changes.append({"entity": owner.id, "field": field, "old": old, "new": new})
    return changes


def apply_action_effects(engine: SimulationEngine, action: ActionDefinition,
                         actor: Entity | None, target: Entity | None,
                         params: dict[str, Any], result: ResolutionResult) -> list[dict]:
    """Shared settlement path for sequential, simultaneous and invoked actions."""
    changes = []
    if result.success and action.transfers:
        try:
            changes = settle_transfers(engine.state, action.transfers, actor, target, params)
        except TransferError as exc:
            result.success = result.partial = False
            result.magnitude = result.success_degree = 0.0
            result.narrative = "Transfer rejected; no balances or success effects changed."
            result.details = {**(result.details or {}), "reason": "transfer_rejected", "transfer_error": str(exc)}
            return []
    if result.success:
        effects = action.effects_on_success
    elif result.partial and action.effects_on_partial:
        effects = action.effects_on_partial
    else:
        effects = action.effects_on_failure
    return changes + engine._apply_effects(effects, actor, target, params, result)
