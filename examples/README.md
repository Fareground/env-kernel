# Examples

Runnable, end-to-end examples of the SDK facade. From a clone of the repo:

```bash
PYTHONPATH=src python3 examples/00_simulate.py
PYTHONPATH=src python3 examples/quickstart.py
```

(With the package pip-installed, drop the `PYTHONPATH=src`.)

## 00_simulate.py

The one-liner: `simulate(path_to_template)` loads the tic-tac-toe template, lets the built-in seeded random policy play every turn, and returns the finished `World` — then `world.summary()` reads the outcome. Zero configuration; deterministic given the seed.

## quickstart.py

The whole loop in ~35 lines: load `tic_tac_toe/template.json`, plug in a toy `decision_fn` (a random-cell picker — the slot where an LLM call goes), step until `world.finished`, then read `world.terminated_by` and the final event narrative. A seeded `Kernel` plus a deterministic `decision_fn` makes the run reproducible.

## tic_tac_toe/template.json

A complete world as pure JSON — no Python anywhere in the template:

- one `entity_types` entry (`Player`, `role: "agent"`) and two entity instances carrying an `X`/`O` mark;
- the `board` **domain module** providing the 3x3 grid, occupancy rules, and per-agent board perception (`perception["domain_data"]`);
- one action, `place_mark`, whose effect is the registered `place_on_board` operation;
- a `board_pattern` termination (`row_3`/`col_3`/`diag_3`) plus `temporal.max_rounds: 9` as the draw backstop.

## From here to your own template

1. Start from the inline template in the [README quickstart](../README.md#quickstart) (no domain module — plain properties, an `add` effect, an `expr` termination) or copy `tic_tac_toe/template.json`.
2. Describe your world: `entity_types` (at least one with `role: "agent"`), `entities`, `actions` whose `actor_type` matches, `termination_conditions`, and `temporal.max_rounds` as a budget. Every field is documented in [`docs/template_schema.md`](../docs/template_schema.md); valid effect operations / archetypes / check types / module names are listed live in [`docs/kernel_contract.json`](../docs/kernel_contract.json).
3. Replace the toy `decision_fn` with your agent. It receives `(entity_id, perception, valid_actions)` and returns an `ActionInstance` or `None` — the perception dict is designed to be serialized straight into an LLM prompt.
4. Validate before running: `lint_template(template)` flags contract violations, and `smoke_test(engine)` (on the engine from `load_world`) plays a short scripted run.
