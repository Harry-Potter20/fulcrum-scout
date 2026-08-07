---
title: "Fulcrum v3 — the unified football-state encoder (pre-registered spec)"
tags: [fulcrum, v3, representation, jepa, self-supervised, pre-registered]
status: spec (pre-code) — success criteria fixed BEFORE training
---

# Fulcrum v3 — unified encoder

**Hypothesis — REFRAMED (2026-08) by the nonlinear-probe result (validated oracle).** Original question: "can one
encoder do both?" The nonlinear probe (MLP oracle validated: MLP ≥ linear on every property incl. ball_x 0.46→0.66)
showed the contrastive latent **learned representations predictive of topology-DERIVED tactical descriptors, with
zero direct supervision** — danger lin 0.28 → MLP **0.75**, persistence 0.32 → **0.76**, compactness 0.66 → **0.92**,
ball_x 0.81 → **0.96** (careful wording: the probes test danger/persistence/etc., which are *derived from* the
topology engine — not "topology" itself). So the semantics *exist*; they need not be *created*. Sharper question:
**can prediction be added WITHOUT destroying the semantic geometry the contrastive encoder already has?**
Risk = prediction *erases* concepts → v3 starts from the contrastive encoder and **measures representational drift**.

*(Predictive encoder: contains **substantially less information about topology-derived descriptors** than the
contrastive one — danger/persistence ~0.18 even under the validated MLP, vs 0.75/0.76.)*

**The deeper, data-backed finding (a result beyond Fulcrum):** contrastive-from-temporal-continuity recovered
descriptors that prediction never needed. The encoder gap **scales monotonically with topology-dependence /
prediction-irrelevance**: danger/persistence ~0.57–0.58, compactness 0.47, block_height 0.37, ball_x 0.30, tempo
~0.01. Prediction optimises *"where will player A move?"*; contrastive optimises *"what situation is this?"* — and
the two latents diverge exactly on the descriptors only the second question requires. Prediction and contrastive
objectives are **related but not interchangeable**: prediction forecasts dynamics; contrastive organises latent
geometry. (Contra the common world-model assumption that prediction *alone* yields useful representations.)

**Primary risk is gradient interference, not capacity** — so objectives are separated into adapters, NT-Xent runs
on `g(z)` not `z`, and success is judged by a **transfer benchmark**, not by how good the embeddings look.

## Architecture

```
Observation (nodes = ball + players; pos, vel, team, ball, event-ctx)
        │
Relational Encoder  (Klein-4 equivariant transformer, d=96, 2-4 blocks)   ← SHARED
        │
      z  (masked-mean pooled latent)
        │
 ┌──────────────┬────────────────────┬────────────────────┐
 │              │                    │                     │
 Prediction   Representation      Geometry            (frozen at inference:
 Adapter      Adapter             Adapter              z is the product)
 │              │                    │
 dynamics     g(z) → NT-Xent      space descriptors → danger (emergent)
 value        (32-d projection)   free-space area · pitch-control entropy
 event        retrieval           hole persistence · overload score
```

Per-objective **adapters** are the key change vs "5 heads on z": objectives disagree in adapter space, not over z.
NT-Xent operates on `g(z)`, never on `z` (SimCLR/BYOL/DINO/JEPA all separate representation from projection space).

## Loss — three groups (three knobs, not five)

- **L_pred** = dynamics (Gaussian-NLL + rollout) + value (see target below) + event (CE).
- **L_repr** = contrastive (NT-Xent) + invariance (symmetry/jitter consistency).
- **L_sem**  = geometry regression (predict the COMPUTED space descriptors; danger is emergent, NOT supervised directly).

`L = L_pred + α·L_repr + β·L_sem`. Grouped so tuning is 3-D, not 5-D.

**covtoken is preserved (core primitive).** The dual-controlled tail-coverage constraint + residual-economy gate
(worldmodel.py `Loss`: constrain top-10%-deviation "decisive tail" loss ≤ κ·overall via a capped dual variable λ;
model *spends* a gate-budget to emit any residual) stays **inside L_pred's dynamics** unchanged — it protects the
informative tail from the easy majority. **v3 hypothesis (not default):** extend the same coverage principle to
**L_repr** — a covtoken-style constraint that protects rare/hard positive pairs (a rare press-trap, an overload)
from being averaged away in the representation. On-thesis for covtoken ("keep rare, informative configs from being
averaged out"); test it as an ablation, don't assume it.

**Value target (must be explicit — the shipped value_head trains via an untraced path and is unstable across
leagues, 0.37-0.63).** v3 trains value on a defined self-supervised signal: **shot-soon within the possession**
(binary, the same label used to validate danger) — makes value directly the pre-shot worth and stabilizes it.

## Contrastive positives (learn concepts, not just time)

| type | source | note |
|---|---|---|
| temporal | continuous tracking | t ↔ t+1 (gave tempo/shape sep 0.69) |
| symmetry | any | Klein-4 transform of the same state |
| **tactical** | any | different possession, **same structure** — structure similarity from the geometry descriptors |
| retrieval | any | nearest-neighbour → positive — **stage-3 only** (collapse risk if used from scratch) |

## Multi-task scheduler (mixed-source batches, formalized)

| dataset | dynamics | contrastive | value | event | geometry |
|---|---|---|---|---|---|
| SkillCorner (continuous) | ✓ | ✓ (temporal) | ✓ | ✓ | ✓ |
| Metrica (continuous) | ✓ | ✓ (temporal) | ✓ | ✓ | ✓ |
| StatsBomb 360 (discrete) | ✗ | weak (aug only) | ✓ | ✓ | ✓ |

The dataloader is a task scheduler: each source contributes only the terms it can.

## Training regimes — PRIMARY is sequential-from-contrastive (the probe result redirected this)

1. **sequential-from-contrastive (primary).** Init from `fulcrum_contrastive.pt` (which already has the geometry),
   then add prediction heads (+ L_sem to *linearize* the danger it already encodes nonlinearly), then joint-finetune.
   **Don't destroy what's there** — start from it.
2. **joint-from-scratch (control).** All losses together from random init — the baseline the sequential run must beat.

**Semantic-retention panel — logged EVERY epoch of prediction training** (the central measurement, using the
validated MLP oracle): **danger-probe R² · compactness-probe R² · retrieval@10 · world-model (dynamics NLL).** If
danger drops 0.75 → 0.40 at epoch k, you see *exactly* when and how fast prediction overwrites the representation —
which directly guides loss scheduling / staged unfreezing. The research question it opens: **why does prediction
erase concepts?** (task interference · limited latent dim · prediction not needing semantic abstraction · different
invariances). This is the plot that makes the result publishable — retention vs world-model-gain, epoch by epoch.

## Evaluation — TRANSFER, not clustering

Freeze `z`; train tiny **linear probes** for: formation · pressing · phase · danger · retrieval · planner-reward.
The representation genuinely improved only if the probes improve. Report per-probe.

**Baseline measured (2026-08, 4,500 states) — the numbers v3 must beat:**

| probe | predictive (unified) | contrastive | v3 target |
|---|---|---|---|
| danger R² | 0.08 | **0.26** | **> 0.26** (the core gap; geometry supervision) |
| compactness R² | 0.33 | **0.66** | ≥ 0.66 (no regression) |
| tempo R² | 0.98 | 0.99 | ≥ 0.98 (trivial — it's velocity) |
| retrieval@10 | 0.37 | **1.00** | ≥ 0.95 (no regression) |

Confirmed: neither current encoder encodes danger; the contrastive one owns representation (compactness, retrieval);
v3's job is to get **all three in one latent** — keep contrastive's representation, fix danger via geometry, keep prediction.

## NOT doing (prior-result-aware)

- **No explicit latent-space prediction** (the pure-JEPA ingredient). It was tested on Fulcrum and came back within
  noise — its payoff is on video (mise), not sparse pitch geometry. v3 keeps predicting *outputs*. Adopt JEPA
  *philosophy*, not that lever.
- **No parameter scaling** — established as unwarranted; gains are objective + data + adapters, not width.

## Success contract (FIXED before code) — relative, not arbitrary magic numbers

No hardcoded "danger ≥ 0.60" (arbitrary). The falsifiable standard:

**World-model rows — NO regression allowed** (the systems-win requires the model to stay intact):

| Metric | Current | Target |
|---|---|---|
| Value ↔ xG | 0.54 | ≥ 0.54 |
| Danger → chance lift | 2.7–3.0× | ≥ 2.7× |
| Twin generalization | +25.6% | ≥ 25% |

**Representation rows — v3 must ≥ the BEST existing encoder (contrastive), material improvement where claimed:**

| Probe (linear R² unless noted) | best baseline (contrastive) | Target |
|---|---|---|
| danger (linear) | 0.28 | **≥ 25% relative gain** (expose what MLP shows at 0.72) |
| danger (MLP oracle) | 0.72 | ≥ 0.72 (don't lose the entangled signal) |
| compactness / ball_x / retrieval | 0.66 / 0.81 / 1.00 | ≥ baseline (no regression) |

**The systems win (per the critique):** if v3 ≈ contrastive on representation AND the world model is intact, then
**one encoder replaced two** — a success even if no single metric is individually best. **If semantics improve but
any No-regression world-model row drops, the experiment FAILED** — decided against this table, not rationalized after.

**Extra diagnostics (make the result publishable):** representational **drift** curve during prediction training;
**CKA / SVCCA** between predictive · contrastive · unified latents (is prediction *rotating / collapsing / fragmenting*
the semantic space?).

## RESULT (2026-08) — hypothesis confirmed; contract partially closed

Both regimes trained (GPU, 1500 steps, 10k SkillCorner windows). Validated-oracle eval vs the contrastive baseline:

| encoder | danger lin/MLP | compact lin/MLP | retrieval | CKA vs base |
|---|---|---|---|---|
| contrastive (base) | 0.28 / 0.74 | 0.66 / 0.93 | 1.00 | 1.00 |
| **v3 sequential** | **0.47 / 0.76** | **0.82 / 0.95** | **1.00** | 0.86 |
| v3 scratch | 0.42 / 0.73 | 0.79 / 0.92 | 1.00 | 0.84 |

- **Reframed hypothesis CONFIRMED:** prediction did NOT erase the semantic geometry. v3-sequential **preserved** the
  representation (MLP danger 0.76 ≥ 0.74, retrieval 1.0) AND **exposed danger far more linearly** (0.28 → 0.47, +65%)
  via `L_sem`, while the dynamics world model trained (pred loss ~14.9, covtoken λ engaged at 5.0). Beats the
  contrastive baseline on every representation row → **one encoder replaces two (systems win).**
- **Sequential > scratch** — init-from-contrastive won on danger (0.47/0.76 vs 0.42/0.73); scratch dipped *below* the
  contrastive MLP baseline (0.73 < 0.74). "Don't destroy what's there" validated.
- **Contract status:** representation rows PASS (materially better); danger→chance world-model row intact (computed
  topology, encoder-free); **value↔xG NOT closed** — the shot-soon value term was deferred (SkillCorner has no
  shots), so that no-regression row is *untested*, not passed. **Full contract closes only after the StatsBomb value
  increment.** Honest: core science succeeded; one row remains.

## CHARACTERIZATION + DECISION-QUALITY (2026-08, all threads)

- **Property panel:** SSP preserves total info (MLP-R² flat) and LINEARISES the entangled topology descriptors,
  ordered by entanglement (persistence +0.12, danger +0.11 linear; tempo/ball_x ~0).
- **Loss-weight Pareto (W_PRED 0.1–0.6):** NO tradeoff — retention 0.76–0.78 AND world-model NLL ~15 both FLAT;
  weight-insensitive. (short-run scoped)
- **Adapter ablation (FALSIFICATION — my mechanism claim was WRONG).** Hypothesised: adapters decouple objectives →
  flat frontier. Test: remove adapters (NT-Xent on z directly, linear geo readout), re-sweep. **Result: retention
  stays flat (~0.75–0.76) WITHOUT adapters too** → adapters are NOT the cause of the flat frontier. What they *do*
  is reduce representational rotation (CKA 0.86 with vs 0.69–0.82 without) while retention is preserved either way.
  **Corrected mechanism: semantic preservation is driven by L_repr+L_sem + the contrastive-init basin — robust to
  prediction weight AND to adapter presence; adapters reduce drift (CKA), not the flatness.** (Honest: a plausible
  causal story, killed by the controlled ablation.)
- **Seed stability:** danger retention 0.768 / 0.759 / 0.774 across 3 seeds (0.767 ± 0.006) — replicates, not luck.
- **Value↔xG (probe on StatsBomb shots):** v3 latent **0.637 ≥ 0.54** (≈ predictive 0.64, ≥ contrastive 0.61) —
  value information PRESERVED, no regression (head not yet trained; info is present).
- **DECISION QUALITY — RETRACTED.** Slice 1 (danger-target retrieval) showed v3/contrastive ≫ predictive, but the
  OUTCOME-INDEPENDENT test (kNN-**shot-soon** AUC, a real outcome) REVERSED it: predictive 0.696 ≥ contrastive 0.675
  ≥ v3 0.664. The slice-1 win was target-contaminated. **We do NOT claim v3 improves decisions.** v3's earned value
  is representation + systems-win, full stop.

- **Style-conditioning DOES compose with v3 (I earlier wrongly called it a category mismatch — corrected).** The
  twin is a *conditioning of the dynamics head*, not a separate model; v3 is the encoder. Test: freeze the v3 encoder,
  add a team-style vector to the dynamics head, LEAVE-MATCH-OUT. **Result (4-fold CV): correct-style beats
  shuffled-style by MEAN +3.65% ± 1.88% NLL, ALL FOUR folds positive (+1.9% to +6.8%) → style conditioning works on
  v3 and consistently generalises to unseen matches.** (Team-style; small conditioned head on the FROZEN v3 encoder.
  Per-player style — needs a custom window builder carrying per-node style, since `_build_windows` doesn't expose the
  identity→node map — is the remaining refinement, and would be the fair comparison to the original twin's +25.6%.)

**Final scope (held): within Fulcrum's architecture/datasets/run-lengths, SSP preserved + improved + linearised
tactical representations, stably, while enabling prediction, preserving value, and SUPPORTING STYLE-CONDITIONED
PREDICTION THAT GENERALISES — a REPRESENTATION result and a systems win. Decision-quality (retrieval-recommendation)
benefit is unproven (outcome-independent test negative).**

## Output

One checkpoint `fulcrum_v3.pt` (encoder + all adapters) + a recorded **seed / config / data manifest** (the
reproducibility hardening, folded in from the start) — replacing `fulcrum_unified.pt` and `fulcrum_contrastive.pt`
only if the contract is met.
