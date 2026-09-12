# Referenced action history

Ordinary `engine.checkpoint()` remains an inline `fg-execution-v1` checkpoint.
Full `WorldState.to_dict()` exports retain all action records.

Hosts that store action records separately can opt into `fg-execution-v2`:

```python
checkpoint = engine.checkpoint(include_action_history=False)
# Persist the records once (or append new records to a host-owned journal).
history = engine.state.action_history.to_dict()['history']

# Supply the exact per-entity prefixes described by checkpoint['action_history'].
restored.restore_checkpoint(checkpoint, action_history=history)
```

Each entity's manifest contains its record count and a chained SHA-256 digest
of the ordered action name, success flag and round. The manifest is captured
in O(entities), without serializing or traversing previous actions. Missing,
extra, changed, reordered or differently attributed records fail restoration
before any live engine state changes. External event history is a separate
requirement when `include_events=False`.

Records are append-only and `ActionRecord` is immutable. Use `record()` and
`from_dict()`; private `_history` mutation is unsupported. Action prerequisite
membership is indexed without discarding records, so checking a long-lived
prerequisite no longer scans the whole history. Locks, unlocks and cooldowns
remain in every checkpoint.

This is a serialization primitive, not a durability claim. A host must commit
the referenced records with, or before, the checkpoint; retain the exact
prefix; bind storage reads to the correct run and owner; and fence stale
writers. A reference whose records were not durably retained is not a usable
checkpoint. It must not be silently replaced with an empty or recent-only
history. The host also remains responsible for participant memory, caches,
events, schema/version compatibility and ambiguous external calls.
