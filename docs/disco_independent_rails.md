# DiSCO rails protocol v2: independent main/cache rails

**Audience:** whoever edits the DiSCO control code
(`DiSCO_Control/Full_System_Integration`: `communication.py`, with
`composition_receiver`, `objective_sender` and `write_objective`, plus `archerfish.py`).
**Status:** the ZoMBI side is implemented (2026-11-12). The DiSCO side described here
is **not** implemented yet. ZoMBI's *two-head mode* needs it.

---

## 1. Why

DiSCO has two rails, called `main` and `cache` in every composition packet. Each
transporter prints from one slot, and each printed slide comes back as its own
objective packet some time later. Print and measurement times vary, so the results
come back in an unpredictable order.

Protocol v1 cannot cope with that order:

| v1 behaviour | Consequence |
|---|---|
| One timestamp (`ts`) covers both slots. A new `ts` makes `composition_receiver` **replace both slots** and re-arm both. | ZoMBI cannot command one rail without re-commanding the other. The other rail's line is either printed twice or overwritten before it is printed. |
| An objective packet carries only `values`, `comps` and `ts`. | ZoMBI cannot tell which rail, or which commanded line, a result belongs to. The single-head loop credits whatever arrives to the line it just sent, which can be the cache rail or a slide from two lines ago. This causes the "dropped points" and mislabelled data. |
| `objective.db` holds one result, and the sender re-sends only the latest. | Two results finishing close together can hide one another. |

Two-head mode runs two independent optimisers ("heads"), one per rail:
head 0 → `main`, head 1 → `cache`. Each head commands only its own rail, whenever
it is ready, and must get back exactly the results of its own lines.

## 2. What ZoMBI now sends (composition packet, ZoMBI → DiSCO)

Unchanged fields, still sent: `type: "composition"`, `ts`, `start`, `end`, `comps`,
`start_cache`, `end_cache`, `compositions_cache`.

New fields, present whenever ZoMBI's `compositions.db` has the `slot_ts` table
(always, from now on):

| Field | Type | Meaning |
|---|---|---|
| `protocol` | int | `2` |
| `ts_main` | float or null | Per-slot timestamp of the line in the `main` slot. It changes **only** when ZoMBI commands a new main line. `null` means the slot has never been commanded: idle, do not print. |
| `ts_cache` | float or null | The same for the `cache` slot. |
| `ack_main` | float or null | `ts` of the latest `main` objective packet ZoMBI has ingested. |
| `ack_cache` | float or null | The same for `cache`. |

Notes:

- An idle slot (its `ts_<slot>` is null) may come with empty `start`/`end`/`comps`
  lists. Do not print it.
- Legacy `ts` still changes on **every** ZoMBI write, whether one slot or both. A v2
  DiSCO must ignore it for arming slots when `protocol == 2`.
- Both-slot writes still exist. ZoMBI's axial initialization and its single-head
  mode write both slots at once with `ts_main == ts_cache == ts`. Under v2 that
  simply means both slots are new, which matches v1 behaviour.
- ZoMBI-side code: `scripts/communication.py`:
  - `write_slot` writes one slot.
  - `write_compositions` writes both slots.
  - `composition_sender` builds the packet.

## 3. Changes needed in DiSCO

### 3.1 `composition_receiver`: arm each slot on its own timestamp

- Keep the last-seen `ts_main` and `ts_cache` separately, and persist them next to
  the mirrored compositions, so a DiSCO restart does not re-print a slot.
- When `protocol == 2`:
  - If `ts_main` changed (and is not null), replace **only** the `main` slot
    (start/end/compositions) and mark it unread.
  - If `ts_cache` changed (and is not null), replace **only** the `cache` slot.
  - **Never touch the other slot.** In particular, do not DELETE and re-insert
    it, and do not re-mark it unread.
  - Ignore the legacy `ts` for arming.
- If a slot gets a new `ts_<slot>` while its previous line is still unread (no
  transporter has taken it yet), replace that line. ZoMBI superseded it (a re-send
  or a new plan), so the old one is intentionally dropped. This is the **only**
  overwrite allowed, and only within the same slot.
- A slot is printed **at most once** per `ts_<slot>` value.
- When `protocol` is absent, keep the v1 behaviour (an old ZoMBI).

### 3.2 Carry the slot identity through the measurement pipeline (`archerfish.py` → stability → `write_objective`)

When a transporter takes a slot, record on that slide, and keep it until its
result is written:

- `slot`: `"main"` or `"cache"`, meaning the slot the composition was taken from.
  Which transporter took it does not matter.
- `slot_ts`: the `ts_<slot>` value of the line that was taken.

Transporter assignment can stay as it is: main to whichever transporter is free
first, else cache, or a fixed front=main / back=cache. ZoMBI routes results by
slot, never by transporter.

### 3.3 `write_objective` and `objective_sender`: one result per slide, tagged, re-sent until acknowledged

Objective packet (DiSCO → ZoMBI). New fields are marked **new**:

| Field | Meaning |
|---|---|
| `type` | `"objective"` (unchanged) |
| `values` | per-row objective, as now |
| `comps` | per-row measured 10-column composition, as now |
| `ts` | **Unique** per result: a fresh value on every `write_objective` (already the case). |
| `slot` **new** | `"main"` or `"cache"`: the slot this slide was printed from |
| `slot_ts` **new** | The `ts_<slot>` of the commanded line this slide printed. ZoMBI routes the result to its line by this value. Echo it exactly as received: same float, no rounding beyond 1e-6. |

Delivery rules:

1. **Never lose a result because another one arrived.** Keep, per slot, the latest
   result that has not been acknowledged; a small outbox table keyed by
   `(slot, ts)` works. A main result must not overwrite a pending cache result,
   and vice versa.
2. **Re-send until acknowledged.** Keep re-sending each slot's latest
   unacknowledged result at the current ~1 Hz, interleaving slots. Stop re-sending
   it once a composition packet arrives with `ack_<slot> == ts` of that result.
   ZoMBI deduplicates by `(slot, ts)`, so re-sends are harmless.
3. **A completely failed slide** (every row NaN) should still send a packet with
   `slot`, `slot_ts` and all-NaN `values`. Today nothing is sent, so ZoMBI waits out
   a stall timeout (up to hours) before re-planning. ZoMBI v2 accepts an all-NaN
   *tagged* packet as "this line produced no usable points" and moves on at once.
   Untagged all-NaN packets are still ignored, as before.
4. Partial NaN rows and the other existing conventions are unchanged.

## 4. How ZoMBI uses this (for reference)

- **`scripts/communication.py` → `objective_receiver`**
  - Dedupes v2 packets by `(slot, ts)`.
  - Appends every ingested packet to an append-only `objective_queue` table in
    `objective.db`. It also still writes the legacy single-packet mailbox and
    handshake for single-head mode and initialization.
  - Records `ack_<slot>`.
- **`scripts/rail_broker.py` → `RailBroker`**
  - Each head's `measure()` writes its own slot with a fresh `slot_ts`.
  - Routes queue rows to lines by `slot_ts`.
  - If a result is missing after the stall timeout, re-sends the line with a new
    `slot_ts`, up to 3 sends.
  - A result that arrives for a line its head is no longer waiting on is still
    used: both heads get it as GP data.
- **v1 detection:** if results arrive without `slot`/`slot_ts`, ZoMBI logs a
  warning once and falls back to matching results to lines by composition. That
  works, but a v1 DiSCO still re-prints the other rail's line on every write, so
  **two-head mode should only be run against a v2 DiSCO.**
- **`src/core/two_head.py`:** the two heads, sharing GP data, with penalty regions
  adopted only between activations.

## 5. Acceptance checklist for the DiSCO change

Run each scenario with the real DiSCO receiver/sender code and fake transporters, in
the style of `tests/test_init_disco_pipeline.py` on the lab machine.

1. **Main-only write.** ZoMBI writes main with a new `ts_main` while a cache line
   is still unread or printing. The cache line is printed exactly once, and the
   main line exactly once.
2. **Same-slot supersede.** Main written twice before any transporter takes it.
   Only the second main line is printed.
3. **Results returned out of order.** The cache slide finishes before the main
   slide. Both results reach ZoMBI, each tagged with the correct `slot` and
   `slot_ts`.
4. **Close finishes.** Main and cache results are written within 1 s of each
   other. Both are delivered, and both are re-sent until acked; neither hides the
   other.
5. **Ack stops re-sends.** After ZoMBI acks `ack_main == ts`, that main result is
   no longer sent.
6. **Failed slide.** A tagged all-NaN packet is sent.
7. **Both-slot write** (`ts_main == ts_cache`, as in initialization). Both slots
   are printed once each, as in v1.
8. **Idle slot.** `ts_cache` is null and the cache lists are empty. Nothing is
   printed from cache.
9. **DiSCO restart.** Restart DiSCO mid-run with unchanged `ts_main`/`ts_cache`.
   Nothing is re-printed.
10. **Old ZoMBI** (no `protocol` field). DiSCO behaves exactly as v1.
