# State mutation values

`set`, `add`, `subtract`, and `multiply` accept literal values, dollar
references, and expressions. Numeric properties require finite numbers
(booleans and numeric strings are not numbers). Boolean properties require
booleans. Integer properties require integer results; fractions are not
silently truncated.

Examples:

```json
{"operation": "subtract", "target": "actor", "field": "cash", "value": "$actor.price"}
{"operation": "set", "target": "actor", "field": "profit", "value": {"expr": "$actor.cash + $actor.holdings * $actor.price - $lookup(runtime, initial_cash)"}}
{"operation": "set", "target": "actor", "field": "believes", "value": "$entity($params.predecessor_id).believes"}
```

Arithmetic supports parentheses, `+`, `-`, `*`, `/`, `%`, and `**`.
Expressions can call the existing dollar functions and access dotted paths.
The explicit `{"expr": "..."}` wrapper must contain exactly that one key.
`$if(condition, yes, no)` evaluates only the selected branch and requires a
boolean condition. `$first($actor.optional_value, 0)` is an explicit fallback
for missing data. A missing operand in arithmetic never becomes zero.

Invalid literals and malformed syntax fail compilation with a value path.
Invalid dynamic values emit `effect_dropped` with reason
`invalid_effect_value`, the operation, target, field, and failure detail. The
invalid mutation leaves its field unchanged. This is not transaction rollback
of earlier effects in the action. Smoke tests expose these diagnostics in
`invalid_effects`, report unhealthy, and block `compile_template(smoke=True)`.

For compatibility, an **omitted** ADD/SUBTRACT operand still uses the
resolution magnitude, and an omitted MULTIPLY operand uses 1. Explicit JSON
`null` is invalid; it does not request that default. The JSON loader preserves
this distinction through typed template serialization and effect expansion.
Direct Python `Effect` callers can set `value_supplied=True` to distinguish an
explicit null from the legacy `value=None` default.
