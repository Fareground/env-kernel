# Conserved property transfers

Use an action's `transfers` for cash, inventory, or other conserved numeric
properties. Independent `subtract` and `add` effects clamp separately; bounds
alone do not make those edits a valid payment.

```json
{
  "name": "accept_offer",
  "actor_type": "Seller",
  "target_type": "Buyer",
  "preconditions": [{"expr": "$entity(negotiation).deal_reached == false"}],
  "transfers": [{
    "source": "target",
    "target": "actor",
    "field": "cash",
    "amount": {"expr": "$entity(negotiation).current_offer"}
  }],
  "effects_on_success": [{
    "operation": "set", "target": "negotiation",
    "field": "deal_reached", "value": true
  }]
}
```

Declare the same numeric property on each account. Do not duplicate its debit
or credit in `effects_on_success`. Those effects are for subsequent state
changes such as closing a deal or recording payoffs. Ordinary properties and
effects keep their existing behavior; this is an explicit action contract.

All amounts resolve once against the pre-transfer state. Up to 100 transfers
settle together. The kernel aggregates account deltas and validates every final
balance before writing any of them. Incoming amounts may fund outgoing amounts
in this atomic settlement. There are no intermediate balances or clamps.

The default lower bound is zero; an explicitly declared negative `min_value`
permits modeled credit. Declared upper bounds enforce account capacity. Invalid
references, missing/nonfinite/negative amounts, fractional transfers involving
integer accounts, and self-transfers reject the settlement. Zero amounts are
valid. Precision loss greater than `1e-12` of an account's net transfer is also
rejected; ordinary floating-point roundoff is permitted. State remains ordinary
JSON-compatible integer/float data.

Transfers execute only on full resolution success, before success effects.
When settlement fails, no transfer or subsequent success/failure effect runs;
the action reports `success: false` and a value-free `transfer_error` code.
An ordinary failed or partial resolution does not transfer and retains its
existing failure/partial effects. A success effect cannot retroactively change
the already evaluated transfer amount.

Sequential and simultaneous actions use the same settlement path. Competing
simultaneous actions cannot each spend the same balance: settlement checks the
current state, not the earlier decision snapshot. `invoke_action` also enforces
transfer constraints, emits an actor-visible failure for a rejected invoked
action, and skips that invoked action's success effects. This does not turn
an entire chain of separate actions into one transaction or roll back unrelated
effects that the caller already performed.

Boundary tests should cover empty and exact balances, recipient capacity,
multiple debits against one account, and both settlement and no-settlement
paths. A passing default scenario is not sufficient evidence of conservation.
