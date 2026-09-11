# Conditional effects

Conditional effects, expression-based effect guards, and action preconditions
use the same predicate evaluator. `if_expr` evaluates comparisons and boolean
operators; it does not treat an unresolved expression string as truthy.

```json
{
  "operation": "conditional",
  "value": {
    "if_expr": "$target.is_fraud == true && $target.amount > 0",
    "then": [{"operation": "add", "target": "ledger", "field": "fraud_loss", "value": "$target.amount"}],
    "else": [{"operation": "add", "target": "ledger", "field": "legitimate_revenue", "value": "$target.amount"}]
  }
}
```

The effect context includes actor, target, parameters, resolution result,
the latest event payload, world state, and the seeded random generator.
`if_compare: [left, operator, right]` and nested `if: {expr: ...}` use the
same comparison semantics. World-level conditions and composed effects also
delegate to this evaluator.

Missing references make a predicate false, including under negation or inside
arithmetic. To intentionally test whether optional data exists, use the
explicit `defined` predicate; present `0` and `false` values count as defined.
