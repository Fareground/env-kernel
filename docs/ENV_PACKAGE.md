# Env Package Format

An environment is a **self-contained package** holding everything needed to load, play, share, and publish a game or simulation. Library, marketplace, fork, and version-control operations all move this whole package.

## Layout

```
<env_name>/                          # directory form (preferred for editing)
├── meta.json                        # required — env identity
├── overview.md                      # required — natural-language brief
├── template.json                    # required — WorldTemplate (schema/rules)
├── viz/                             # optional — visualization
│   └── component.tsx                #   sandboxed React component
└── primitives/                      # optional — env-scoped Python primitives
    ├── __init__.py
    └── *.py                         #   auto-discovered when env loads
```

The same content can be packed into a single `.simworld` zip for transport.

## File-by-file contract

### `meta.json` — env identity

Single source of truth for "what is this env at a glance." Used by library list views, search, version control.

```json
{
  "name": "race_to_10",
  "display_name": "Race to 10",
  "tags": ["dice", "race", "2-player"],
  "author": "alice@example.com",
  "version": "0.3.0",
  "contract_version": "1.0",
  "created": "2026-05-23T12:00:00Z",
  "updated": "2026-05-23T15:30:00Z",
  "license": "CC-BY-4.0",
  "min_kernel_version": "1.0"
}
```

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Slug — lowercase, snake_case, unique within a library |
| `display_name` | ⬜ | Human-friendly title (defaults to name) |
| `tags` | ⬜ | Searchable labels |
| `author` | ⬜ | Anything — name, email, handle |
| `version` | ⬜ | Semver string |
| `contract_version` | ✅ | Kernel contract version this env targets (see `python -m kernel versions`) |
| `created` / `updated` | ⬜ | ISO 8601 timestamps |
| `license` | ⬜ | License identifier |
| `min_kernel_version` | ⬜ | Reject load if the running kernel is older |

### `overview.md` — natural-language brief

Markdown. The "what is this game" content the **Overview** tab renders. Written by the env-builder agent (or human) to capture the *intent* of the env, not its mechanics:

```markdown
# Race to 10

Two players take turns rolling a die. First to reach 10 points wins.

## Premise
A friendly race between friends. No special rules, no luck-skewing —
pure dice racing.

## Rules in plain English
- Each turn, the active player rolls 1d6
- The roll is added to their score
- First player to reach 10 wins
- If both reach 10 in the same round, higher score wins
```

The kernel does not parse this file; it's UI content. Treat it as the env's elevator pitch.

### `template.json` — WorldTemplate

The full declarative game definition. See [`kernel_contract.json`](./kernel_contract.json) for the schema.

The **Build** tab pretty-prints this. The **Settings** tab reads `runtime_parameters` from it.

The kernel validates this through `compile_template()`; structural or lint errors block the env from loading.

### `viz/component.tsx` — visualization

Sandboxed React component. Receives events from the engine and renders the game state.

| Constraint | Why |
|---|---|
| No `eval` / `Function(...)` | Code injection |
| No `fetch` / network calls | Data exfiltration |
| No `dangerouslySetInnerHTML` | XSS |
| No third-party URLs | Supply-chain attacks |
| No filesystem access | Sandbox break |

Failure to render falls back to a generic event log so the game is never blank.

### `primitives/` — env-scoped extensions

Optional directory of Python files that register custom effects / terminations / resolutions specific to THIS env. Auto-discovered when the env loads. Examples:

- A game-specific scoring algorithm
- An exotic resolution mechanic the standard kernel doesn't ship
- A custom termination check_type

The kernel's existing `kernel_primitives/` directory is the **global library** (shared across all envs). The `primitives/` directory inside an env package is **scoped** — its registrations only fire when this env is active.

Files follow the same convention as global primitives (see [FRAMEWORK.md](./FRAMEWORK.md#3-primitives-format)).

## The `.simworld` archive format

For sharing / publishing / library storage, the directory is zipped into a `.simworld` file:

```bash
python -m kernel pack my_env/                 # → my_env.simworld
python -m kernel unpack my_env.simworld       # → my_env/
python -m kernel compile my_env.simworld --smoke 30   # works on .simworld directly
```

The archive is a standard ZIP file with the same layout as the directory form. Any tool that handles zips can introspect it.

## Loading flow

```
.simworld OR directory
        │
        ▼
load_env_package(path)
        │
        ├── parse meta.json          → version checks, identity
        ├── read overview.md         → cached for UI
        ├── parse template.json      → WorldTemplate
        ├── discover primitives/     → @effect / @termination decorators fire
        ├── compile_template(...)    → linted + validated
        └── return EnvPackage object with engine, viz path, meta
```

The engine starts in a known good state; viz code is shipped to the frontend; primitives are registered for this env's lifetime.

## Versioning

`meta.contract_version` declares which kernel contract version this env was built for. The `upgrade_template()` pipeline in `pipeline/versioning.py` migrates old envs forward when the kernel is upgraded. Envs with no `contract_version` are treated as legacy and walked through the full migration chain.

## What's outside the package

These are RUNTIME state, not part of the env definition:

- Live playtest reports (smoke / lint / replay results — re-runnable any time)
- Active simulation snapshots (the engine's running state)
- User-specific data (preferences, history)

These belong in app-level storage, not in the env package.

## What the package guarantees

- **Self-contained.** Drop a `.simworld` file on any kernel of compatible contract_version → it runs.
- **Inspectable.** A non-technical user can open `overview.md` and `template.json` to see what they're getting.
- **Versioned.** Forward-compatible migration via `contract_version`.
- **Pluggable.** Per-env primitives ride along; no need to install global library updates.
- **Sandboxed.** Viz code is sandboxed React; primitives run in the kernel sandbox.

## Reference

- [`FRAMEWORK.md`](./FRAMEWORK.md) — the three formats (declarative / inference / primitives)
- [`HOW_TO_WRITE_A_GAME.md`](./HOW_TO_WRITE_A_GAME.md) — author's guide
- [`kernel_contract.json`](./kernel_contract.json) — full WorldTemplate JSON Schema
- `python -m kernel pack --help` — CLI reference
