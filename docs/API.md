# CR-Native-Sandbox JSON-line API

This is the authoritative wire protocol for the headless `libg.so` battle
service (`serve-direct`). It is a machine-facing specification: every operation
is one UTF-8 JSON object terminated by a newline, sent over a persistent TCP
connection, and answered by exactly one newline-terminated JSON object.

> 本文是外部 JSON-line 协议的正式规范，是 `native_core.client` 与
> `native_core.env` 所实现的同一套 wire 格式。运行时不变量、版本保护与
> fail-closed 条件见
> [`SANDBOX_RUNTIME_TECHNICAL.zh-CN.md`](SANDBOX_RUNTIME_TECHNICAL.zh-CN.md)。

- Protocol version: `schema_version = 1`
- Game / runtime: `15.535.29` / `150535029` / `x86_64`
- Transport: TCP, one request → one response per line, `TCP_NODELAY`
- Limits: request ≤ 32 MiB, response ≤ 64 MiB, trace 1..64 ticks

---

## 1. Envelope

### 1.1 Request

```json
{"op": "<operation>", "…operation-specific fields…"}
```

Every request carries a string `op`. Unknown operations fail closed with
`ok=false`; they are never silently mapped to another operation.

### 1.2 Response

Success:

```json
{"schema_version": 1, "ok": true, "op": "<operation>", "…fields…"}
```

Failure:

```json
{
  "schema_version": 1,
  "ok": false,
  "error_type": "java.lang.IllegalArgumentException",
  "error": "human-readable reason"
}
```

`error_type` is the Java exception class name; `error` is its message. A
non-`ok` response means the operation had no effect on battle state (except
where an operation is explicitly not atomic; see §8).

---

## 2. Result-code table

Native commands (play and ability) return a `result_code` alongside `accepted`.

| `result_code` | hex | meaning |
| ---: | --- | --- |
| `0` | `0x0` | accepted and applied |
| `1014` | `0x3F6` | ability charges exhausted |
| `1050` | `0x41A` | not enough elixir |
| other | — | `native_rejected` (evidence not yet named; fail closed) |

`accepted` is `true` only when the original native command was constructed and
executed. The Python layer adds `ability_state_name` for display but never uses
it to override the native verdict.

---

## 3. Coordinate system, sides and grid

- The arena is an **18-column × 32-row** grid. One cell is **1000** native
  coordinate units on each axis.
- `x` ranges over `[0, 18000)` (columns), `y` over `[0, 32000)` (rows).
- Cell `(column, row)` has center `(column*1000 + 500, row*1000 + 500)`.
- `side` is `0` or `1`. Side `0` is the home (blue) half, rows `0..14`;
  side `1` is the away (red) half, rows `17..31`. Row `15..16` is the river.
- Red/blue is positional, not player identity: an action's ownership is
  resolved through `account_hi` / `account_lo`, not through `side`.

`probe_grid` returns the raw native 18×32 binary mask (each row is 18 `0`/`1`
characters). `deployment_grid(adjusted=True)` additionally applies the arena
boundary rules in `native_core.deployment` (symmetric mirroring, King 4×4 and
Princess 3×3 footprints, friendly-half restriction, five-cell pocket after a
Princess tower falls, spells keep full targeting). The adjustment is a client
convenience, not a `libg` single-function output.

---

## 4. `entity_id` lifecycle

- `entity_id` is the public **5,000,000-series generation key**; it equals the
  entity's `category` field.
- `creation_ordinal = category - 5_000_000` and is `>= 0`.
- It is unique among live entities of one battle and stable for the lifetime of
  that entity.
- A reset starts a new battle, so entity ids are **not** preserved across
  `reset`; always re-read ids from the current observation before `ability`.
- The raw process pointer (`id` in entities, and similar pointer fields) is a
  diagnostic value only. It must not be sent back as an action handle.

---

## 5. Operations

### 5.1 `ping`

```json
{"op": "ping"}
```

Response (no extra payload):

```json
{"schema_version": 1, "ok": true, "op": "ping"}
```

### 5.2 `status`

```json
{"op": "status"}
```

Response carries `state` from the native runtime probe (manager/state/battle
readiness, current state type, tick, replay data pointer, …). Read-only.

### 5.3 `observe`

```json
{"op": "observe"}
```

Response: `state` is the **full** public observation (§9).

### 5.4 `observe_compact_v1`

```json
{"op": "observe_compact_v1"}
```

Response: `state` is the **compact** public observation. Its envelope is
`{"schema_version":1,"kind":"libg_native_compact_state_v1","coherent":true,...}`
and it omits paths, collisions and per-effect detail (§9).

### 5.5 `reset` / `restart_replay`

```json
{"op": "reset", "replay": {"battle": {"deck0": {"sp": [...]}, "deck1": {"sp": [...]}, "avatar0": {...}, "avatar1": {...}}, "rndSeed": 42}}
```

In-process BattleGameState `4 → 4` replacement: the current `libg.so` is kept
loaded, the old state is detached, the new replay is set via `0xCE7C40`, the
manager executes the replacement via `0xCE7810`, and the new battle is warmed to
the requested tick. Response contains `result` (`lifecycle_restarted`, `load`,
`state`) and the resulting `state`. `reset` clears the terminal latch (§8).

### 5.6 `load_replay`

```json
{"op": "load_replay", "replay": {"battle": {...}, "rndSeed": 42}}
```

Legacy load/adopt entry. If the canonical bootstrap replay is still adoptable
and the current state type is `4`, the service adopts the already-paused
bootstrap without re-invoking the native loader (`result.adopted_bootstrap`).
Otherwise it runs the outer replay loader. **Rejected after a terminal episode
latch** (see §8): recycle the host process or use `reset`.

### 5.7 `step`

```json
{"op": "step", "steps": 1}
```

Advances the native core `steps` times (one native update per tick at 20 Hz).
Response `result`:

```json
{
  "tick_before": 100,
  "tick_after": 101,
  "stepped": 1,
  "episode": {"terminated": false, "truncated": false, "crowns": [0,0], "...": "..."}
}
```

`stepped` is the actually-completed tick count (the tiebreak clock can pause, so
`stepped` may be less than `steps`). The terminal latch is set when
`episode.terminated` is `true`.

### 5.8 `step_trace`

```json
{
  "op": "step_trace",
  "steps": 1,
  "trace_schema_version": 1,
  "max_response_bytes": 33554432
}
```

Advances and returns an initial frame plus one frame per tick. `result` is a
`libg_native_tick_trace` object:

```json
{
  "schema_version": 1,
  "trace_schema_version": 1,
  "kind": "libg_native_tick_trace",
  "encoding": "full-v1",
  "requested_steps": 1,
  "max_response_bytes": 33554432,
  "stepped": 1,
  "final_frame_index": 1,
  "terminal": false,
  "initial_frame": {"frame_index": 0, "advanced_steps": 0, "observation_complete": true, "state": {...}},
  "frames": [{"frame_index": 1, "advanced_steps": 1, "observation_complete": true, "step": {...}, "state": {...}}]
}
```

Constraints: `trace_schema_version` must be `1`, `steps` in `1..64`,
`max_response_bytes` in `65536..33554432`.

### 5.9 `probe_grid`

```json
{
  "op": "probe_grid",
  "action": {"side": 0, "deck_index": 2, "account_hi": 1, "account_lo": 1}
}
```

Returns the raw native deployment mask for one current hand card (read-only;
does not mutate). `result` includes `rows` (18 binary strings), `cell_size`
(1000), and the card/account identity.

### 5.10 `act`

```json
{
  "op": "act",
  "action": {
    "type": "play",
    "side": 0,
    "deck_index": 2,
    "x": 9000,
    "y": 10000,
    "account_hi": 1,
    "account_lo": 1,
    "dry_run": false
  }
}
```

Runs the original `DoSpellCommand` path. `dry_run: true` validates only (returns
the same `result` shape without mutating state). `account_hi`/`account_lo`
default to `side + 1` when omitted. `result` includes `accepted`, `result_code`,
`resolved_data_id` (the resolved base/evolution/hero form), and any rejection
reason.

### 5.11 `ability`

```json
{
  "op": "ability",
  "action": {
    "type": "ability",
    "side": 0,
    "entity_id": 5000011,
    "account_hi": 1,
    "account_lo": 1
  }
}
```

Runs the native active-ability command (`0x5A`) for a live entity identified by
`entity_id`. `result` includes `accepted`, `result_code`, `native_mana_cost`,
elixir before/after, and ability state/charges/cooldown/pending fields.

### 5.12 `joint_act`

```json
{
  "op": "joint_act",
  "actions": [
    {"type": "play", "side": 0, "deck_index": 2, "x": 9000, "y": 10000, "account_hi": 1, "account_lo": 1, "dry_run": false},
    {"type": "ability", "side": 1, "entity_id": 5000012, "account_hi": 2, "account_lo": 2}
  ]
}
```

At most one action per side; the server enforces unique sides and applies them
in canonical `side 0 → side 1` order. `result`:

```json
{
  "canonical_order": "side_0_then_side_1",
  "actions": [
    {"side": 0, "result": {"accepted": true, "result_code": 0, "...": "..."}},
    {"side": 1, "result": {"accepted": true, "result_code": 0, "...": "..."}}
  ]
}
```

### 5.13 `joint_transition`

```json
{
  "op": "joint_transition",
  "actions": [
    {"type": "play", "side": 0, "deck_index": 2, "x": 9000, "y": 10000, "account_hi": 1, "account_lo": 1, "dry_run": false}
  ],
  "steps": 1
}
```

Applies the joint actions, advances `steps`, and returns
`result = {joint_action, step, state?}`. `state` is omitted when the episode is
terminal or truncated (callers must read the terminal episode instead).

### 5.14 `joint_transition_trace`

```json
{
  "op": "joint_transition_trace",
  "actions": [
    {"type": "play", "side": 0, "deck_index": 2, "x": 9000, "y": 10000, "account_hi": 1, "account_lo": 1, "dry_run": false}
  ],
  "steps": 1,
  "trace_schema_version": 1,
  "max_response_bytes": 33554432
}
```

Applies joint actions, then collects a per-tick trace. `result` =
`{joint_action, trace, episode}` where `episode` must equal the final trace
frame's episode (the Python client enforces this consistency).

### 5.15 `shutdown`

```json
{"op": "shutdown"}
```

Stops the service. The connection closes after the response.

---

## 6. Action objects

### 6.1 `play`

| field | type | meaning |
| --- | --- | --- |
| `type` | string | `"play"` |
| `side` | int | `0` or `1` |
| `deck_index` | int | index into the side's configured deck |
| `x`, `y` | int | native arena coordinates |
| `account_hi`, `account_lo` | int | account id (defaults to `side+1`) |
| `dry_run` | bool | validate only when `true` |

### 6.2 `ability`

| field | type | meaning |
| --- | --- | --- |
| `type` | string | `"ability"` |
| `side` | int | `0` or `1` |
| `entity_id` | int | live entity generation key (§4) |
| `account_hi`, `account_lo` | int | account id (defaults to `side+1`) |

---

## 7. Replay object

The `replay` accepted by `reset` / `load_replay` is a JSON object containing a
`battle` object and optional `rndSeed`. Each deck entry is
`{"d": <base card id>, "l": <zero-based level>, "el": <form mask>}`:

| `el` | form |
| ---: | --- |
| `0`/omitted | base |
| `1` | evolution |
| `2` | hero |
| `3` | both (catalog permitting) |

Missing `battle`, `deck0`/`deck1`, or `avatar0`/`avatar1` is fatal (fail closed).

---

## 8. Reset and terminal-latch semantics

- The service holds exactly one in-process BattleGameState. `reset` replaces it
  (`4 → 4`) without reloading `libg.so` or rebooting Android.
- When a `step`, `step_trace`, `joint_transition` or `joint_transition_trace`
  reaches a **terminal** episode, the host sets a terminal latch.
- After the latch is set, `load_replay` is **rejected**
  (`native terminal is latched; recycle the host process before reset`).
  `reset`/`restart_replay` **is** allowed and clears the latch.
- `joint_transition` and `joint_transition_trace` omit the post-step `state`
  once terminal/truncated; read the returned `episode` instead.
- The host observes native crown towers and latches the result because there is
  no in-game result screen; the adaptation layer never fabricates an outcome.

---

## 9. TCP disconnect and no-replay rules

- A single connection may carry many requests; the server processes them
  strictly one line at a time (the client serializes on the same connection).
- **Mutating** operations — `reset`, `load_replay`, `step`, `step_trace`, `act`,
  `ability`, `joint_act`, `joint_transition`, `joint_transition_trace`,
  `shutdown` — are **never automatically replayed** after an ambiguous I/O
  failure (connect/send/receive error). Replaying could double-play a card or
  double-press an ability.
- **Read-only** operations — `ping`, `status`, `observe`,
  `observe_compact_v1`, `probe_grid` — may reconnect once.
- The next explicit request always opens a fresh connection after a failure or a
  worker replacement.

---

## 10. Full vs compact observation

Both share `schema_version`, `kind`, `coherent`, `players`, `entities`,
`episode`, `rng_state`, `state_hash` and `state_hash_scope`
(`public-observe-v6`). The compact form's `kind` is
`libg_native_compact_state_v1`.

| field group | full `observe` | `observe_compact_v1` |
| --- | --- | --- |
| `players` (elixir, refill, hand, cycle) | yes | yes |
| `entities` (position, hp, ability) | yes | yes |
| per-entity path nodes | yes (≤ 115) | **omitted** |
| collision / avoidance accumulators | yes | **omitted** |
| `effects` detail | yes | reduced/omitted |
| `projectiles` | yes | reduced/omitted |

Top-level full observation fields:

```text
schema_version / kind
tick / tick_after / applied_replay_tick
coherent
players / entities / effects / projectiles
entity_count / effect_count / projectile_count
rng_algorithm / rng_state
state_hash / state_hash_scope
episode
```

`state_hash` covers tick, normalized entities, ability fields, players, paths,
effects and RNG. It detects state/schema drift and is not a full-process memory
hash.

---

## 11. Limits and fail-closed conditions

Hard limits: request ≤ 32 MiB, response ≤ 64 MiB, trace `1..64` ticks, trace
response `65536..33554432` bytes, entities ≤ 2048, path nodes ≤ 115.

The host fails closed on: `libg.so` hash/RVA mismatch; unreadable
manager/state/battle/logic; current state type `!= 4`; replay missing
`battle`/`deck`/`account`; card not in hand; account not mappable to a player;
selection parse failure; native deployment validator rejection; ability on a
dead or wrong-side entity; entity/path/trace over-limit; response schema or
frame mismatch; remote JAR/Bridge hash drift.
