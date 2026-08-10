# World template schema

A human-readable reference for the template dict that `Kernel.load()` / `load_world()` accepts. The authoritative sources are the pydantic models in `src/fg_env_kernel/pipeline/loader.py` (`WorldTemplate` and its sub-specs) and the machine-readable export in [`kernel_contract.json`](kernel_contract.json) — which also carries the **live registries**: every valid `operation`, `resolution_archetype`, `check_type`, phase handler, and domain module name. When in doubt, that file wins.

All sections are optional (every field has a default), and unknown extra keys are allowed — but a playable world needs at least one `entity_types` entry with `role: "agent"`, one entity of that type, and one action whose `actor_type` matches it.

Expressions: any field named `expr` (and many `value` fields) accept the safe expression grammar — `$actor.gold >= 100 && $count(player, alive) > 1`, with roots `$actor`, `$target`, `$params`, `$state`, `$last_event`, `$result` and functions like `$random(min, max)`, `$dice(n, sides)`, `$lookup(table, key)`. Prefer `expr` strings over the legacy structured operator form.

## Top-level layout

| Key | Type | Purpose |
| --- | --- | --- |
| `name`, `description` | str | Metadata; both surface to agents via `world_brief`. |
| `rules` | str | Natural-language how-to-play brief (markdown). Injected into every agent's perception as `world_brief.rules`. |
| `contract_version` | str | Contract version, default `"1.0"`. Auto-upgraded by `compile_template`. |
| `entity_types`, `resource_types`, `relation_types`, `visibility_rules`, `spatial`, `temporal`, `tables` | — | Schema layer (type definitions). |
| `actions`, `triggers`, `derived_rules`, `termination_conditions`, `domain_modules` | — | Rules layer (behavior). |
| `entities`, `resource_holdings`, `resource_unallocated`, `initial_relations`, `factions`, `adjacency` | — | Instance layer (starting world). |
| `physics`, `property_dynamics` | dict | Continuous dynamics (below). |
| `location_definitions`, `goals`, `skill_definitions`, `initial_skills`, `recipes`, `connectors`, `runtime_parameters` | list | Optional subsystems. |
| `cognitive_config`, `social_config`, `crowd_config` | dict | Subsystem toggles (wire up when `enabled: true`). |
| `viz` | dict | Rendering hints. Opaque to the kernel. |

## entity_types

Type definitions for everything that exists in the world.

```json
{"entity_types": [
  {"name": "Player", "role": "agent", "description": "A player.",
   "properties": [
     {"name": "gold", "type": "int", "default": 100, "min_value": 0},
     {"name": "mark", "type": "string", "default": ""}
   ]}
]}
```

- `role` — `agent` (gets turns) | `object` | `location` | `abstract`.
- Property `type` — `float` (default) | `int` | `string` | `bool` | `enum` | `list`. Extras: `default`, `min_value`, `max_value`, `enum_values`, `description`, `hidden` (excluded from other agents' perception).

## entities

Starting instances. Type defaults fill in any property you don't override.

```json
{"entities": [
  {"id": "player_x", "entity_type": "Player", "name": "Player X",
   "properties": {"mark": "X"}, "location_id": null, "alive": true}
]}
```

Required: `id`, `entity_type`.

## actions

What agents can do. Only `name` is required.

```json
{"actions": [
  {"name": "attack",
   "description": "Strike an adjacent enemy.",
   "actor_type": "Warrior",
   "target_type": "Warrior",
   "parameters": [
     {"name": "power", "type": "int", "min": 1, "max": 10, "description": "Effort."}
   ],
   "preconditions": [
     {"expr": "$actor.stamina >= 2", "description": "Needs stamina."}
   ],
   "resolution_archetype": "skill_check",
   "resolution_params": {"skill_property": "strength"},
   "effects_on_success": [
     {"operation": "subtract", "target": "target", "field": "hp", "value": "$params.power"}
   ],
   "effects_on_failure": [
     {"operation": "subtract", "target": "actor", "field": "stamina", "value": 1}
   ]}
]}
```

Field notes:

- `actor_type` — entity type allowed to perform it; `target_type` — entity type of the (optional) target.
- `parameters` — free-form dicts describing per-call arguments (surfaced to agents; passed as `ActionInstance.parameters` and readable in effects via `$params.<name>`).
- `resolution_archetype` — how success is decided. `deterministic` (default, always succeeds) or any registered archetype: `skill_check`, `contest`, `voting`, `order_book`, `cpmm`, the auction family, etc. — live list in `kernel_contract.json`. `resolution_params` configures it.
- `effects_on_success` / `effects_on_failure` / `effects_on_partial` — effect lists (below).
- Sequencing: `requires_action` (+ `requires_action_success`), `cooldown_rounds`, `sequence_rounds` (multi-round actions), `interruptible`, `unlocks_actions`, `locks_actions`.
- `message_action: true` — the action carries a message payload.
- `broadcast: false` — the resolved action is not announced to every agent (night kills, secret votes).
- `range` — max spatial distance to the target.
- `duration` — sim-time length of the action in continuous mode (see temporal).

### Preconditions

Prefer the expression form; the structured form (`subject`/`operator`/`value`/`field`/`resource`/`relation_type`) is legacy. Unknown operators fail loudly at load.

```json
{"expr": "$actor.gold >= 100 && $actor.alive"}
```

### Effects

```json
{"operation": "transfer_resource", "target": "target", "resource": "gold",
 "value": "$params.amount",
 "condition": {"expr": "$target.alive"},
 "scale_by_magnitude": true}
```

- `operation` (required) — one of the built-in + registered ops (`set`, `add`, `subtract`, `multiply`, `transfer_resource`, `move_to`, `kill`, `spawn_entity`, `emit_event`, `conditional`, `for_each`, `sequence`, board/card/token ops, ...). Live list: `kernel_contract.json → live_capabilities.effect_operations`. Unknown ops fail loudly at load.
- `target` — `actor` | `target` | an entity id | an expression.
- `field` / `value` — property to change and the (possibly `$`-expression) value.
- `resource`, `relation_type` — for resource/relation ops.
- `condition` — optional guard (`expr` preferred); the effect is skipped when false.
- `scale_by_magnitude` — scale numeric value by the resolution result's magnitude.

## relation_types / initial_relations

Typed, directed (or symmetric) weighted edges between entities.

```json
{"relation_types": [
   {"name": "trust", "symmetric": false, "default_value": 0.0,
    "min_value": -1.0, "max_value": 1.0}
 ],
 "initial_relations": [
   {"from": "alice", "to": "bob", "type": "trust", "value": 0.5}
 ]}
```

Edges also accept the `{from_entity, to_entity, relation}` spelling.

## resource_types / holdings

Conserved (or not) quantities held by entities.

```json
{"resource_types": [
   {"name": "gold", "conservation": true, "discrete": true,
    "min_value": 0, "max_value": null, "initial_supply": 1000}
 ],
 "resource_holdings": [
   {"resource": "gold", "entity_id": "alice", "amount": 100}
 ],
 "resource_unallocated": [
   {"resource": "gold", "amount": 900}
 ]}
```

## visibility_rules

Who sees what. Without rules, defaults apply; `hidden` properties are never shown to others.

```json
{"visibility_rules": [
  {"observer_type": "Player",
   "visible_entity_types": ["Player", "Item"],
   "visible_properties": {"Player": ["hp"]},
   "max_range": 5.0,
   "requires_same_location": false,
   "see_relations_involving_self": true,
   "see_all_relations": false,
   "visible_resources": ["gold"]}
]}
```

## spatial

```json
{"spatial": {"type": "grid", "rows": 10, "cols": 10}}
```

- `type: "none"` (default) — no space.
- `type: "grid"` — `rows`/`cols` (or legacy `width`/`height`).
- `type: "graph"` — `locations` (or `nodes`) list + `edges` (`{"from": "a", "to": "b", "weight": 1.0}` or `["a", "b"]`). Edges mirror into adjacency for `$is_adjacent`.
- `type: "continuous_2d"` — `width`/`height` floats.

## temporal

```json
{"temporal": {
  "max_rounds": 50,
  "phases": [
    {"name": "night", "active_roles": ["Werewolf"], "resolution_mode": "sequential"},
    {"name": "day", "handler": "assign_properties", "handler_params": {}}
  ],
  "round_duration_seconds": 86400,
  "sim_start_iso": "2026-01-01T00:00:00",
  "time_unit_label": "day"
}}
```

- `max_rounds` — the round budget. Honored by `load_world` and the facade; without a termination condition it is the only stop. `Kernel.load(max_rounds=...)` overrides it.
- `phases` — ordered per-round phases; default is a single `action` phase. Per-phase: `active_roles`, `initiative_type` (`fixed` | property-driven via `initiative_property` + `initiative_descending`), `handler` + `handler_params` (registered phase handlers — live list in `kernel_contract.json`), `resolution_mode` (`sequential` | simultaneous).
- Time mapping (`round_duration_seconds`, `sim_start_iso`, `time_unit_label`) — optional; gives agents calendar context in perception.
- **Continuous mode** — `"mode": "continuous"` plus a `continuous` block; per-action `duration` fields let slow and fast actions share one clock:

```json
{"temporal": {"mode": "continuous",
  "continuous": {"default_turn_interval": 1.0, "max_time": 100.0,
                 "environment_interval": 1.0, "action_durations": {"negotiate": 48.0}}}}
```

## termination_conditions

When the sim ends. `World.terminated_by` reports the `name` of the condition that fired. Optional but recommended.

```json
{"termination_conditions": [
  {"name": "last_survivor", "check_type": "expr",
   "params": {"expr": "$count(Player, alive) <= 1"}},
  {"name": "either", "check_type": "compound_or", "sub_conditions": [
    {"name": "rich", "check_type": "expr", "params": {"expr": "$state.entities.alice.gold >= 1000"}},
    {"name": "board_win", "check_type": "board_pattern",
     "params": {"board_id": "main", "patterns": ["row_3", "col_3", "diag_3"]}}
  ]}
]}
```

- `check_type` — `expr` (preferred) or any registered check: `last_one_standing`, `first_to_score`, `faction_win`, `resource_exhausted`, `round_limit`, `compound_and`/`compound_or` (with `sub_conditions`), etc. Live list: `kernel_contract.json → live_capabilities.termination_check_types`.
- An `expr` in `params` alongside another `check_type` acts as an AND-guard.

## physics

Continuous coupled dynamics, integrated with RK4 between turns. Deterministic and serializable.

```json
{"physics": {
  "params": {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4},
  "variables": [
    {"name": "prey", "value": 10, "rate": "alpha*prey - beta*prey*pred", "min": 0},
    {"name": "pred", "value": 5,  "rate": "delta*prey*pred - gamma*pred", "min": 0}
  ]
}}
```

Variables can read entity aggregates and write values back onto entities — see `src/fg_env_kernel/physics.py` (`EntitySource`, `EntityWriteback`) for those blocks.

### property_dynamics

Autonomous per-entity drift/spawn/cascade rules, ticked every round: `{"property_dynamics": {"drift_rules": [...], "spawn_rules": [...], "cascade_rules": [...]}}` — see `src/fg_env_kernel/property_dynamics.py` for rule shapes.

## domain_modules

Pre-registered Python modules referenced by name — the template never contains Python. Each takes free-form `params`.

```json
{"domain_modules": [
  {"name": "board", "params": {"id": "main", "kind": "grid", "rows": 3, "cols": 3,
                               "auto_marks": ["X", "O"]}}
]}
```

Available names (chess, poker, monopoly, mafia, prediction_market, securities_trading, turn_manager, ...): `kernel_contract.json → live_capabilities.domain_modules`. Modules contribute perception (`domain_data`), effects, phase handlers, and can narrow the valid-action list.

## Everything else

- `tables` — named lookup tables for `$lookup(table, key)`.
- `runtime_parameters` / `last_runtime_params` — named tunables exposed under the `runtime` table.
- `triggers`, `derived_rules` — reactive rules and the inference layer; free-form dicts consumed by their engines.
- `factions` — `{id, name, member_ids, parent, properties}` teams.
- `adjacency` — `{location, neighbors}` edges when not using a graph space.
- `location_definitions` — per-location properties, entry requirements, modifiers, and tick effects.
- `goals`, `skill_definitions` + `initial_skills`, `recipes` (crafting), `connectors` — optional subsystems; see their modules under `src/fg_env_kernel/`.
