# Execution checkpoints

`engine.checkpoint()` returns a JSON-compatible `fg-execution-v1` checkpoint.
Pass it as `execution_checkpoint=` to `SimulationEngine`, supplying a fresh world
compiled from the same schema and the original callback implementations. Calling
`run()` continues the saved budget. A completed checkpoint stays completed.

Checkpoints preserve the world, engine RNG position, shared domain RNG bindings,
continuous-time queue and physics clock, trigger once/cooldown bookkeeping,
world-event cooldowns/active events/cascades, invariant baselines, termination
configuration, perception trends, event history, and lifecycle flags. Restoration
validates and stages components before committing, retaining the WorldState
object's identity so callbacks can safely close over it. It does not reseed or
reshuffle restored decks.

Capture between completed discrete rounds or continuous scheduled events. The
`on_checkpoint(engine)` callback runs at those boundaries, after event/trigger
processing. Capture from inside a turn or event cascade is rejected. A partial
round paused mid-turn is not a supported checkpoint boundary.

Arbitrary callbacks and live external clients cannot be serialized by the kernel.
Use `external_state=` to capture application-managed agent memory, policies,
response caches, and configuration, then restore those before creating callbacks.
Identical future LLM replies/external inputs are still required for deterministic
results; a checkpoint cannot predict a future model reply or market data update.

By default event history is included. Hosts persisting the transcript separately
may use `include_events=False` and supply the exact retained event prefix through
`checkpoint_event_history=` on reconstruction. Missing or truncated history is an
error. Checkpoints are private; do not send them in spectator or public replay
payloads.

The returned checkpoint is detached from both the live world and caller-owned
`external_state`: changing either cannot rewrite a saved boundary, and editing a
checkpoint cannot change a running engine. Capture reuses the already-detached
`WorldState.to_dict()` result rather than deep-copying it again. This avoids a
redundant full-history traversal; it does not bound checkpoint size or make
capture independent of accumulated action history. Full action history remains
present, and the `fg-execution-v1` format and restore contract are unchanged.

Fareground stores checkpoints at saved round boundaries, with ordinal-addressed
transcript records and deduplicated shared agent caches. A fork without a seed
override continues the saved RNG stream and extends the budget by the requested
additional rounds (time units for continuous runs). An explicit seed override
starts a new RNG stream while retaining world state. Old world-only snapshots
cannot promise execution continuation and require a new simulation. Terminal
conditions require choosing an earlier checkpoint.

An engine created with `max_rounds=None` has no discrete round budget. It runs
until a world termination condition or an explicit stop, while pause and step
remain available. The null budget survives JSON checkpoint restoration; hosts
must not replace it with a numeric fallback. Explicit finite budgets remain
available for bounded experiments, and must not imply a natural game verdict.
