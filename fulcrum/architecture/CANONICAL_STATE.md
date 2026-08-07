---
title: "Fulcrum Canonical State — the frozen ABI"
tags: [fulcrum, platform, canonical-state, abi, contract]
status: SPEC v1.0 — FROZEN CONTRACT. Everything above the CORE layer depends on this. Change = new version + deprecation.
---

# Canonical State — the platform ABI

The single most load-bearing artifact in Fulcrum. **Not the encoder — this.** Raw data → **Canonical State** →
Foundation Services → Applications → Products. If a data source changes, ingestion adapters absorb it and *nothing
above this layer changes*. That property is the whole platform. It only holds if this contract is **frozen and
versioned** and every adapter passes the **conformance test**.

## The invariant (why this is the crown jewel)
The Canonical State is the **agnosticism law made into a data structure**: anonymous geometry in canonical units,
identity/attributes attached only downstream as labels. This is what makes Fulcrum simultaneously **refreshable**
(swap the data → current players), **multi-source** (StatsBomb 360 · SkillCorner · Metrica · Sofascore · broadcast
GSR all normalise to it), and **transferable across sports** (the same structure describes basketball; see
`SPORT2_MAPPING.md`).

## Schema — `CanonicalState` (one frame)
Nodes = `ball + players`. Each node carries geometry only. **No identity, no attributes** (agnosticism law).

| field | type | units / frame | notes |
|---|---|---|---|
| `pos` | float32 `[N,2]` | **metres**, pitch `105 × 68`, origin **bottom-left**, x along length | canonical coordinate frame |
| `vel` | float32 `[N,2]` | **m/s** | 0 where a source has no velocity (e.g. StatsBomb freeze-frames) |
| `team` | float32 `[N]` | `2=ball · 1=attack/possession · 0=defence` | role, **not** club — identity-free |
| `isball` | float32 `[N]` | `1` for the ball node, else `0` | exactly one ball node |
| `mask` | float32 `[N]` | `1`=real node, `0`=padding | broadcast/GSR see a partial pitch → masked |

**Constants (frozen):** `PITCH_L=105`, `PITCH_W=68`, `MAX_NODES=28` (padded), `RS=3.0` (residual scale, m).

## Views (derived, never re-ingested)
- **`Window`** = a `CanonicalState` + `acc` + `future` (rollout targets, `STEPS=4`) — the **model-input** view for
  Encode/Predict. Built by `worldmodel._build_windows(frame_ids, ball_of, players_of, fps, stride, cap)`.
- **Oriented state** (`fulcrum.data.state_at`) = rotated so the possession team attacks **+x** (Klein-4 canonicalisation),
  split into `att / dfn / ball (+ velocities)` — the **topology/planner** view for Evaluate/Optimize. `min_players`
  gates usability (16 full-pitch; lowered for partial-pitch GSR).

Both are pure functions of the `CanonicalState` — they never touch raw data.

## Invariants (conformance requires all)
1. **Identity-free.** No player/club identifiers anywhere in the structure. (Enforced analogously to the medical
   label-leak test — a lesion label touching subspace construction is the Med-project sin; a player ID entering the
   state is Fulcrum's.)
2. **Canonical units/frame.** metres, m/s, `105×68`, origin bottom-left. Adapters convert; consumers never re-scale.
3. **Klein-4 symmetry** is a property of the frame (mirror across halfway × touchline) — the equivariance the encoder
   relies on. (Sport-specific: basketball is C₂, not V₄ — a per-implementation constant, see registry.)
4. **Finite + masked.** No NaN/inf in real (unmasked) nodes; missing players are masked, not zero-filled silently.

## Conformance test (every ingestion adapter must pass)
`tests/test_canonical.py` (to build): given a source, assert the produced `CanonicalState`/`Window` — (a) all real
nodes finite; (b) coordinates within `[0,105]×[0,68]`; (c) exactly one ball node; (d) team ∈ {0,1,2}; (e) no
identity fields present; (f) `state_at` round-trips (orientation is involutive); (g) symmetry-augmentation preserves
downstream Evaluate outputs (equivariance check). An adapter that fails does not ship.

## Versioning & deprecation
- **Version = `canonical/1.0`.** Any field/unit/frame/ordering change → **new version**, not a silent edit.
- Consumers pin the canonical version. A new version ships **alongside** the old with a deprecation window; adapters
  and services migrate before removal. This is the ABI stability guarantee products build on.

## What is NOT here (by design)
Identity, club, age, market value, attributes, xG, outcomes — all **downstream labels** attached to service *outputs*,
never to the state. Keeping them out is what makes the state the stable substrate.
