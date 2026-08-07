---
title: "Fulcrum Foundation Services — the six stable APIs"
tags: [fulcrum, platform, services, api, registry, contract]
status: SPEC v1.0 — the permanent service layer. Products compose these; products never touch the encoder.
---

# Foundation Services — the six permanent APIs

Fulcrum owns exactly one thing: **understanding structured spatiotemporal systems.** It exposes that through six
services. Everything else — Scout, Coach, Broadcast, Academy — is a **composition** of these, not a fork of the model.
The right question is never "can Fulcrum scout?" but **"can Scout be implemented on the services?"** (yes: Encode →
Retrieve → Moneyball → Report). All six already exist in code today — this spec *freezes their contracts*, it is a
refactor, not a rewrite.

**Every service takes a `CanonicalState`/`Window` (see `CANONICAL_STATE.md`) and returns outputs tagged with an
epistemic status.** The tag is part of the contract — a product composing an *unproven* output inherits that label
and cannot silently over-claim. Tiers: **`validated`** (scale + CIs) · **`face-valid`** (sensible, not outcome-tested)
· **`descriptor`** (computed, not a predictor) · **`unproven`** (tested and NOT supported).

## The services

| # | Service | Signature | Today's impl | Output → status |
|---|---|---|---|---|
| 1 | **Encode** | `state → embedding` | `fulcrum_v3` (registry) | embedding → *representation validated* (transfer probes) |
| 2 | **Evaluate** | `state → {value, danger, space_creation, containment, structural}` | `score` · `find_holes` · `metrics` | value → **validated** (ρ≈0.54); danger → **validated** (2.2–3.0×, 11 comps); SC → **validated** (1.4–1.6×); containment → *face-valid*; structural → *descriptor* |
| 3 | **Predict** | `state → future states` | rollout / dynamics head + covtoken | twin → **validated** (+25.6% unseen); rollout → *face-valid* |
| 4 | **Explain** | `state → {narrative, features, importance, topology, formation, pressing}` | `narrate` · `opposition` · `find_holes` | topology/formation/pressing → *computed*; narrative → *descriptor* |
| 5 | **Retrieve** | `state → similar states` | latent kNN (v3) + Memory index | temporal retrieval → **validated** (~1.0); **decision-retrieval on real outcome → `unproven`** (shot-soon test reversed it) |
| 6 | **Optimize** | `state → {plans, counterfactuals}` | `plan*` (computed reward) · graph-intervention | plans → *works* (unhackable computed reward, no training); **counterfactual decision-quality → `unproven`** |

**Memory** (foundation service, supporting Retrieve): the persistent store of canonical states + embeddings (a vector
index) + the KB/vault. Backed by the HF bucket (`hf://buckets/Chucks90/fulcrum-data`). This is what makes Retrieve and
cross-match durable and refreshable.

## The registry — one interface, many drivers (this is how forking is avoided)
The invariant is **the service contract + the canonical state, NOT the weights.** `Encode` is frozen; behind it sits a
**versioned registry of implementations** — `fulcrum-football-v3`, `fulcrum-basketball-v1`, `fulcrum-gsr-v2` — same
interface, swappable driver. (PyTorch: one `nn.Module` interface, infinite models. ROS: stable message types, swappable
nodes.) A new sport or modality adds a **driver**, never a fork of the platform. Per-implementation constants live in the
registry entry (e.g. equivariance group V₄ football / C₂ basketball; the danger/value prior; the trained weights).

## Products = compositions (no backbone modification)
```
Scout        : Encode → Retrieve → Moneyball(translation) → Report
Coach        : Predict → Optimize → Explain(narrate)
Broadcast    : Evaluate → Explain → Render
Recruitment  : Encode → Retrieve → Moneyball → Similarity
Academy      : Predict → Compare → Development report
```
None require modifying the services. Each product declares which *tiers* it depends on — a Broadcast product on
Evaluate(danger, **validated**) is defensible; a Coach product leaning on Optimize(counterfactuals, **unproven**) is not,
and the contract makes that visible.

## Applications layer + the Present concern (where viz / simulator / tactical-interpretation live)
Applications are **compositions of the six services** (never touch the encoder). Built (`fulcrum.applications`):
- **Simulator** = Optimize (plan / plan_multi / plan_dynamic) + Predict — play out tactical reorganisations against
  the unhackable computed reward. status: works (no training); aggressive counterfactuals unproven.
- **Tactical interpretation** = Explain (formation / pressing / topology) + Evaluate (danger / space) — a grounded read.
- **Opponent-Reader** = Explain (formation / pressing / anticipation). **Retriever** = Retrieve.

**Vizzes are the PRESENT concern, not a state-service.** The six services are `state → analysis`; rendering is
`analysis → pixels`. The **Renderer** application consumes Evaluate/Explain outputs (+ the state) and feeds the
Broadcast product — a seventh, *presentation* concern sitting beside the analysis services, never a fork of the model.

## Ingestion — the recognition-free video front-end (DATA → CORE)
The agnosticism law applied to INGESTION: identity is never an input, so the adapter that turns video into canonical
state must not depend on *recognising who anyone is* either. `fulcrum.ingest` (heavy stage = the HF job
`lean_ingest.py`; CPU adapter = the module) takes pure **spatial detection + object tracking** — YOLO(person+ball) +
ByteTrack, **no re-ID, no jersey OCR, no role classifier, no tracklab** — plus a dependency-free homography (numpy DLT)
and a shirt-**colour** team split, and emits the same `frames` dict every other source produces. Fewer failure points
than the full GSR stack it replaces; it drops onto any football video, not only annotated broadcast. Conforms to the
Canonical State ABI (`assemble_frames` → `state_at` runs unchanged).
**Validated on SoccerNet-GS SNGS-021 vs GT** (calibration held fixed): player positions **median 0.17 m** ·
recall **0.97** · colour-teams **0.965** · reach property holds — services **run** on the ingested state (666/750
frames). *Honest limits, tagged:* ball recall **0.47** (small/blurred target — *partial*); and **fidelity is
signal-dependent** — coarse structure transfers (att/def counts corr **0.6–0.8**) but sharp role-sensitive extrema do
**not** (danger ~0–0.2 — *unproven on lean input*), because they amplify residual errors (a referee colour-assigned to
a team → ~+1 defender; a metre-slip on the last defender). Recognition-free mitigations before trusting role-sensitive
signals: drop off-pitch detections (done — matches GT count) and 3-way colour-cluster dropping the small officials
cluster (`split_teams(drop_officials=True)`).

## Capability tiers — which services survive which ingestion conditions
Ingestion is not pass/fail; it degrades, and services degrade at **different rates** by how they read the state. This
is the platform's most decision-relevant axis: a product on a video-only source must know which outputs it can still
trust. Three tiers, ordered by robustness to canonical-state degradation:

| Tier | Reads the state as… | Services / signals | Robustness |
|---|---|---|---|
| **1 — Geometry-driven** | the whole spatial field | Encode/**Retrieval**, shape, **compactness**, width, centroid, formation, **tempo** | **very robust** — an embedding/aggregate is stable under local noise |
| **2 — Aggregated tactical** | a region average | **pressing**, **territory**/occupation, **space-creation** | **likely robust** — averages absorb per-player error |
| **3 — Extremal** | one decisive configuration | **danger**, **containment**, **offside**, **last defender**, pressure chains | **sensitive** — a max over a specific arrangement amplifies every residual error |

The recognition-free ingestion study is the first data point: geometry recovered to 0.17 m, coarse **structure**
transferred (att/def counts corr 0.6–0.8, a Tier-1/2 result), while **danger** (Tier 3) did not (~0–0.2). The
`robustness_<seq>.json` characterization (progressive positional noise, dropout, team-flip, ball noise, synthetic
officials) turns this into a **degradation table per service** — the quantified spec a product consults to pick its
tier. Products then compose to their available ingestion quality: a **Broadcast/amateur-video** product leans on
Tier 1–2 (formation, tempo, territory) and treats Tier 3 as advisory; an **optical-tracking** product may trust all
three. *The response to Tier-3 sensitivity is NOT re-identification* (identity must never enter the backbone) but
(a) a **geometry-first official filter** — officials violate football geometry (persistent central position, low ball
involvement, no formation membership) — and (b) **confidence-aware outputs**: every service returns `(value, confidence)`
where confidence is derived from calibration quality, player count, team-cluster certainty, official probability and
ball confidence — so a product decides per-frame whether to trust a Tier-3 read, instead of pretending every frame is equal.

**Measured (`robustness_SNGS-021.json`, 175 frames, retention = corr vs clean):** the tiers hold *on average* (mean T1 > T2 > T3 under most perturbations) but the real structure is **per-signal**, and two mechanisms explain it better than the tier label:
1. **Landmark / average signals are robust; difference signals are fragile.** `centroid_x`, `last_defender`, `offside_line` stay ≈**1.00** through *every* perturbation (single robust geometric reads) — even though last-defender/offside are nominally "Tier 3." Meanwhile `space_creation` and `containment` (differences/interactions) fall to **0.2–0.7**. Fragility tracks *how many differences a signal takes*, not its tier.
2. **Team assignment is the dominant vulnerability.** `team_flip=0.2` is the worst column across the board (width 0.56, space 0.45, containment 0.26, danger 0.59) — which is exactly what the colour-split + official filter defend. Persistent-bias position noise barely moves most signals (T1 ≈1.00); ball noise is mild (0.6–1.0).
Two findings from the **6-sequence** replication (SNGS-021..026, `robustness_multi.json`): (a) landmark signals *generalise* (cross-seq std ≤0.08) while difference signals carry std 0.17–0.19 — their confidence is ±0.15; (b) an added official mainly hurts **`tempo` (→0.24, std 0.03 — rock-solid)**, box-danger unmoved. The single-seq claim that "danger is sturdy" did **not** replicate: across sequences danger is team-error-sensitive (**0.51** at team=0.1) and orientation-fragile under extreme ball noise — a mid-tier signal, not a robust one. The confidence field is a **lookup into this table** keyed by estimated ingestion conditions, not a hand-tuned formula.

**Why this is the real Phase-2 result:** two independent ingestion modalities — optical tracking (SkillCorner/Metrica)
and recognition-free **video** — now feed the *same* canonical state and the *same* services. Nothing above the
ingestion layer changed. That is platform validation: it shows the canonical-state abstraction was the correct seam,
and it expands the addressable market to anyone with only video (lower-league clubs, academies, amateur analysts,
federations without optical tracking).

## Layer enforcement (intent is not architecture — the import graph is)
```
DATA  ←  CORE (geometry/graphs/Window/canonical-state/transforms/equivariance/constraints)
      ←  FOUNDATION SERVICES (the six)  ←  APPLICATIONS  ←  PRODUCTS
```
Enforced structurally: `products/*` may import `services/*` but **not** `core/*` or an encoder directly; `services/*`
may import `core/*`; `core/*` imports only `data/*`. A dependency-rule check **fails the build** on violation.
"Products never touch the encoder" is only true if the build guarantees it.

## Sequencing (build bottom-up from what exists)
- **Freeze now** (substance proven, contracts knowable): the canonical state + these six signatures + status tiers.
- **Extract later, from real usage:** the plugin packages (`fulcrum.scout`, `fulcrum.broadcast`, `fulcrum.coach`,
  `fulcrum.video`) and the product layer. Do NOT design a plugin interface from imagination — build 2–3 products on the
  services, ship them, then distil the plugin boundary from what they actually needed. A backbone distilled from real
  products is an asset; one designed in the abstract is a liability (the same discipline that gates the cross-sport layer).
