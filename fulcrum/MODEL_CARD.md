# Fulcrum — Model Card

**Fulcrum** is a player-agnostic, geometric **world model** for football. A shared relational spatiotemporal
encoder (multi-task: dynamics, state value, structured concepts, retrieval) is fused with a *computed*
persistent-homology topology engine, on four constraint-first primitives — **relational attention**,
**Klein-4 pitch equivariance**, **covtoken** (a coverage-constrained tail whose dual variable is the
controller), and **topology**. Its power is in the *constraints, not the backbone*: one modest model does all
of the below from anonymous geometry, no new architecture required.

- **Version:** 0.7.0 · **Checkpoints:** `fulcrum_unified.pt` (heads) · `fulcrum_contrastive.pt` (concept representation)

## Capabilities
| | |
|---|---|
| **Value** | possession value of a state — the pre-shot worth of the configuration |
| **Danger / chance creation** | exploitable space in the defensive block (`find_holes`, computed topology) |
| **Predict** | rollout-aware world model — how the play evolves forward |
| **Simulate** | attribute-conditioned **twin**: play out a style; generalises to *unseen* players (+25.6% leave-match-out); analyse/simulate toggle on one weight set |
| **Per-player attribution** | topological remove-and-recompute, possession-aware |
| **Read the opposition** | **formation** (both teams, from x-banding); **pressing structure** (press type, block height, compactness, ball pressure, engagement, control by third); **anticipation** (Markov map of tactical states → what the block is likely to change to) — all computed (`fulcrum.opposition`) |
| **Plan / optimise** | tactical **planner** (`fulcrum.simulate`) — searches *structural graph edits* (move a player, shift the line), scored by the **computed** danger surface (**top-k exposure** + on-pitch clamp: stress-tested — a top-1 reward was reward-hacked, closing the biggest hole while overall danger rose; top-k reduces overall exposure genuinely, ratio 1.00); returns the best *defensive close* / *attacking exploit*. `plan`/`plan_report` use the immediate topology; `plan_multi` beam-searches *coordinated* multi-player reorganisations (≈2× a single nudge); `plan_dynamic` scores an edit's *rolled-forward* consequence via the frozen world model (the network as an evaluator in the loop). No training, unhackable reward. e.g. "the pulled centre-back should drop into the gap (−0.069 danger)". |
| **Recruit / scout** | recruitment intelligence (`fulcrum.scout`) — attribute **archetypes**, role-adjusted **production**, and surplus-value **boards** (young producers / underused gems / younger comps). Data-source-pluggable (FBref, stoichima); cost proxied by age+minutes (no headless value source). Identity is a label. A client of the planner ("fill the role we're missing"). |
| **Cross-league translation** | Moneyball projection (`fulcrum.scout`) — league **strength** index from the mover network (least-squares on production ratios), **prospective projection** (`translation_candidates`: predict a player's output in a target league), an **outlier board** (over/under-translators), and a style-fit layer (off by default — null within a similar tier). Fit on **1,440 movers across 13 leagues** incl. FEEDERS (Sofascore, top-5 + Championship/Eredivisie/Liga Portugal/2.Bundesliga/Belgian/Ligue 2/LaLiga 2/Scottish, 3 seasons): clean hierarchy (all top-5 above every feeder; Championship the strongest feeder), **predicts 9.7% better than naive out-of-sample**, outliers face-valid (Bellingham/Hazard within top-5; Boniface, Gyökeres feeder→top). Refreshable → current data. Attacking output (G+A/90) only. |
| **Differentiated metrics (off-ball)** | topology-derived stats that xG/xA/xThreat structurally cannot express (`fulcrum.metrics`) — **Space Creation** (per off-ball attacker, danger a run/pin *creates* via remove-and-recompute: the decoy that opens a channel scores zero xA/xG, high SC) and **Containment** (per defender, danger suppressed by *positioning* — not tackles/interceptions). Computed geometry over the whole configuration, no model, no outcome calibration; attributes the *mechanism*. Role-discriminative on SkillCorner (SC → wingers/forwards, Containment → FBs/CBs, ~zero overlap) **and** SC outcome-validated at scale: **1.4–1.6× shot-soon lift, all 3 tournaments, CIs above 1**, distinct from danger (ρ≈0.43). SC is a *secondary* team predictor (danger is stronger); its earned edge is the **per-player off-ball attribution danger can't do**. Also a `fulcrum.scout` **space_creators** board (recruit off-ball value FBref production can't see). |
| **Discover & describe** | anomaly detection, position-residualised gamestate discovery, grounded language narration |
| **Agnostic · refreshable · multi-source** | anonymous geometry in; StatsBomb 360 · SkillCorner tracking · broadcast GSR; swap data → current players; core transfers across sports |

This module (`fulcrum.analysis`) exposes the consolidated **analysis** surface (value, danger/chance-creation,
per-player, **tactical shape** [formation + pressing at the peak phase], phase report, grounded narration).
Positioned **complementary to xG** — xG scores the *shot*;
Fulcrum values the *phase that produces it*, which xG cannot see. It is **not** a match-outcome or betting
predictor — that was never its purpose (testing it as one was a framing error).

## Validation (StatsBomb-360, 2026-07)
Scaled, **per-competition** (never pooled — CLAUDE.md rule), with bootstrap 95% CIs (HF Job, 3 tournaments × 40 matches):

| signal | result (per competition) | notes |
|---|---|---|
| value head vs StatsBomb **xG** | Spearman **ρ ≈ 0.54** — Euro'24 0.54 [0.49,0.59], WC'22 0.56 [0.50,0.61], Euro'20 0.54 [0.48,0.59] | ~2,300 shots; never trained on xG. (Higher than the initial pooled 0.42 — per-competition is cleaner.) |
| topological danger → **chance creation** | **2.7–3.0× top-decile shot-soon lift** — Euro'24 2.89× [2.77,3.00], WC'22 2.99× [2.87,3.11], Euro'20 2.71× [2.59,2.82] | ~110k open-play states *per competition*; base shot-soon ~4.5% |
| **Space Creation** (off-ball) → **chance creation** | **1.4–1.6×** top-decile shot-soon lift — Euro'24 1.43 [1.06,1.89], WC'22 1.57 [1.18,2.01], Euro'20 1.45 [1.05,1.88] | model-free (`state_summary`), ~4.2k open states/comp; distinct from danger (ρ≈0.43), weaker than it — a *secondary* signal; earned value is per-player attribution |
| **danger → chance, MULTI-LEAGUE** (11 comps) | **2.2–3.2× across every competition with data** — incl. Bundesliga 2.71, La Liga 2.62, Ligue 1 3.2, MLS 2.96, **women's** WWC'23 3.01 / W-Euro 2.3–2.8 | the strongest generalization claim: topological danger holds on club + international + men's + women's. value↔xG is *variable* by contrast (0.37–0.63; weaker on some club leagues) — value is more context-sensitive than danger. AFCON'23 fetch-missed. |
| twin (style-conditioned prediction) | **+25.6%** on unseen players | leave-match-out; predictor generalizes |

**Method note:** for the binary shot-soon outcome, use **lift / logistic / AUC** — rank-Spearman is invalid
under mass ties (a bug we caught: point estimate fell outside its own bootstrap CI).

## Honest limits
- **Translation: strength predicts; style-fit does NOT (tested and killed).** Cross-league projection improves
  **+7.9% MAE** over a naive "production carries over" baseline out-of-sample (1,213 movers, 13 leagues incl. feeders;
  current production dominates, league strength is the real adjustment, residuals = the outlier board). Two results
  overturned intuition: (a) **xG+xA is *worse* than G+A for translation** (+3.0% vs +7.9%) — underlying chance quality
  already carries over, so the strength model has less to correct; G+A stays default. (b) The **style-fit layer does
  not earn its place even against feeder→top gaps** — style-distance↔residual correlation −0.03 overall, −0.15 on
  cross-tier moves (wrong sign). League *strength* captures translation; style adds nothing; the up-league-breakout
  outliers (Gyökeres, Boniface) are **residual talent, not style-fit surprises.** Style-fit retained but off.
- **Feeder data path — unblocked via Sofascore.** FBref now CAPTCHA-blocks soccerdata and the Stoichima DB is top-5
  only, but the **Sofascore API reached directly with `curl_cffi`** (browser-TLS/JA3 impersonation) beats the
  Cloudflare 403 that defeats soccerdata *and* cloudscraper; its paginated player-statistics endpoint yields 17
  metrics/player for 13 leagues (persisted to the bucket) — which also gives the scout real unsupervised **archetypes**
  (ball-winners, wide creators, playmakers, ball-playing CBs, poachers, GKs). Still attacking-output for translation;
  academy (below senior feeder leagues) out of coverage.
- **Not a match/outcome predictor.** An aggregate match-level test was underpowered (n=13) and, more
  importantly, off-target — Fulcrum operates per-phase.
- **Danger ≠ shot quality.** `find_holes` measures pre-shot *space*, orthogonal to shot xG (ρ≈0). This is by
  design; it is validated against chance *creation*, not shot quality.
- **Temporal persistence is a descriptor, not (yet) a validated predictor.** `fulcrum.temporal.structural_exposure`
  separates *structural* holes (a channel a defence leaves open ≥1s) from *transient* flickers — the one persistence
  signal that is genuinely distinct (state-level topological persistence just equals the danger score). On SkillCorner
  it reads sensibly (~76% of tracks structural, durable gaps in plausible wide/box channels) but structural holes are
  only ~8% more dangerous *per frame* than transient ones; their real value is exploitability *over time*, an outcome
  question. That test needs continuous tracking **+ shot events**, and the only open source is 3 Metrica matches
  (~75 shots) — too few to power it. So it ships as an honest descriptor; outcome-validation is data-gated.
- **Controllability:** latent-space steering of a frozen predictor trades control against realism (a *frozen-latent*
  limitation, not proven intrinsic). The right abstraction is **structural graph intervention**: editing the state
  graph and rolling the frozen model forward gives directionally-correct topology control (80.7%) at *zero* realism
  cost. The learned dynamics respond coherently (physical, proportional) but *modestly* to edits — fine for modest
  tactical counterfactuals, limited for aggressive ones. Next direction is a simulation/planner engine (control =
  graph edit, reward = computed topology), not a bigger control latent.
- **Evaluation scale:** headline numbers are on StatsBomb-360 tournaments and a handful of SkillCorner A-League
  matches. Broader multi-league evaluation with CIs, per-modality (never pooled), is a hardening follow-up.
- **The frozen encoder latent is not a concept substrate — but a contrastive one partly is (fixed).** Clustering the
  *frozen* pooled latent over 4,800 states yielded no tactical vocabulary (between-cluster separation <0.5 on *every*
  axis: danger 0.10, spread 0.04, tempo ~0.4), mirroring the controllability finding — the frozen *predictive* latent
  is optimised for its heads, not as a general representation. Fix: a **temporal-contrastive encoder** (`fulcrum_contrastive.pt`,
  trained on 24k SkillCorner windows / temporal positives, NT-Xent) now separates **tempo 0.69, spread_y 0.63, spread_x 0.51**
  (all clearing 0.5) with **temporal retrieval 0.995** — i.e. it recovers the transition-vs-settled and stretched-vs-compact
  axes the frozen latent could not. Still flat on **danger (0.36)** and **width (0.26)** — those need a danger-aware
  objective (v2). So the Understand pillar is *partly* unblocked: a real, temporally-coherent representation exists;
  a complete tactical vocabulary needs the danger axis added.

## Design law — agnosticism (enforced)
**Identity never enters the model.** Inputs are anonymous geometry (positions, velocities, team, ball). Player
identity and attributes are **downstream labels** attached to outputs, never features. This is what makes the
model refreshable (swap the data → current players) and transferable. Enforced by
`tests/test_agnosticism.py` (the analogue of the medical project's label-leak test).

## Reproducibility follow-ups (tracked)
- Pin seeds + record training data/config for `fulcrum_unified.pt`.
- Vendor the learned model (`jobs/worldmodel.py`) into the package (`fulcrum.model`).
- Multi-league scaled evaluation with confidence intervals.
