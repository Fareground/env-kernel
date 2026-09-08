# World snapshots

`WorldState.to_dict()` emits version 2 snapshots. `apply_snapshot()` stages every
supplied section and commits only after all restorers succeed. It replaces
collections, so entities, edges, optional managers, and plugins absent from the
saved state cannot survive from an unrelated live world. Named manager attributes
and `state.modules` refer to the same restored instances; domain aliases refer
to the manager’s restored child modules.

Snapshots include active status definitions and counters, sequence parameters,
faction hierarchy/membership, current and historical messages, temporal phases
and turn order, resource definitions, relation metadata and threshold latches,
lookup tables, once-fired derived rules, and module-specific state and RNGs.
Returned data is detached from the live world. Domain restorers use the registered
module’s own `from_dict`, not just its constructor parameters.

Pass the compiled world as `schema_provider` when using `WorldState.from_dict`
to execute new actions: entity/action definitions, visibility rules, and plugin
classes are not executable code embedded in a snapshot. A missing plugin class,
failed serializer, invalid section, or discarded supplied value raises an error.
Hook-only plugins without `to_dict` are stateless by contract.

Legacy unversioned partial snapshots can replace individual supplied sections.
Missing sections remain unchanged. Lost historical messages or active sequence
parameters cannot be inferred: these legacy snapshots require starting a fresh
run. Version 2 requires all top-level sections. Redacted spectator views are not
restorable snapshots.

World snapshots are not complete engine checkpoints. Use `engine.checkpoint()`
and `execution_checkpoint=` when restoring the execution itself; see
[execution checkpoints](checkpoints.md). Persisted snapshots contain private state
and random-generator state; filter these from live spectator payloads.
