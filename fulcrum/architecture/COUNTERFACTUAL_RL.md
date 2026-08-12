---
title: "Counterfactual Tactical Fit — bridging Scout and the world model (with a gated path to agent RL)"
tags: [fulcrum, optimize, predict, counterfactual, fit, twin, scout, bridge, rl]
status: SPEC v0.2 — PLAN (pre-code). Scientific contract tightened. Gated build. Extends Optimize(#6)+Predict(#3). No new backbone.
---

# Counterfactual Tactical Fit

## One line
Instantiate a player's **measured capability profile** as a conditioning input to the **world-model twin**, play a
fixed scenario forward within the validated horizon, and measure how the model's **predicted tactical reward**
changes. This is the bridge that makes **Scout** and the **Predict/Optimize** services one thing — a **decision
engine**, not a fancier database. It is the point where Fulcrum stops asking *"who is good?"* and starts asking
*"good for **what**, in **our** system?"*

No new model. It composes three things Fulcrum already owns:
- **Environment (dynamics):** the attribute-conditioned twin rollout (`wm_twin.py`, `services.predict`) — simulator validated (+25.6% unseen, stable ~2 s).
- **Reward:** the Evaluate service (`danger` **validated** 2.2–3.0×, `space_creation` **validated** 1.4–1.6×) — computed geometry, **unhackable**, no training.
- **Capability:** Scout's **validated** latent traits (`progressive_intent` η²≈0.70, `press_resistance` reliability 0.57 / predictive −0.32, `off_ball_penetration_rate` η²≈0.38) + directly-estimable kinematics (pace, accel).

---

## The epistemic claim — say exactly what the experiment earns
The measurement is **A → B → Δreward**. That is **not** "signing B raises our danger by X%." What it actually is:

> **Counterfactual Tactical Fit.** *Under this fixed tactical state, and within the twin's validated simulation
> horizon (~2 s), substituting B's measured capability profile for A's changes the model's predicted tactical
> reward by X (CI …), with simulation-validity V.*

Every output carries: **counterfactual delta**, **per-phase decomposition**, **confidence/CI**, and **simulation
validity**. It never carries a causal real-world signing claim.

### Naming & graduation (a claim ladder, not a label)
| Stage | Gate reached | Product name | Claim allowed |
|---|---|---|---|
| 1 | G2 + G2.5 | **Tactical Fit** / **Capability Contribution** | "changes the model's predicted reward, within horizon, holds vs null" |
| 2 | + G3 | **Counterfactual Fit (sim-validated)** | "…and the baseline reproduces the player's real play" |
| 3 | + out-of-sample longitudinal | **Signing Impact** | causal-flavoured, only once it survives contact with reality |

We start at Stage 1. "Signing impact" is **earned later or not at all**, never assumed.

---

## Two modes — and a hard wall between them
- **Mode A — Counterfactual substitution (no policy).** Condition node *i* on a capability profile, roll the twin,
  read the reward; swap the profile, re-roll, report Δ. **This is the flagship research target.** Small, grounded, and
  by itself enough to make Scout a decision engine.
- **Mode B — Agent RL (deliberately deferred).** Learn `π(a | s, c)` that *acts* to maximise reward. A research
  project (MARL credit assignment, sim-to-real drift), **gated strictly behind Mode A** and behind its own signoff.

**The wall:** a Mode-B failure says nothing about the Scout↔world-model bridge. The bridge is validated by A (G0–G3);
RL is a separate, optional escalation (G4). We do not let the sexy part contaminate the valuable part.

---

## The MDP (formal, Mode A first)
- **State** `s_t`: a `CanonicalState` window (anonymous geometry; `CANONICAL_STATE.md`).
- **Controllable node** *i*: one player in Mode A; everyone else is rolled by the twin (it models their reactions).
- **Capability** `c_i ∈ R^k`: player *i*'s measured trait+kinematic vector, **fixed within an episode** (it's who they are).
- **Transition** `s_{t+1} = Twin(s_t, A(c_i))`, `A` = the trained attr residual. Horizon **H ≤ 2 s** (rollout-gate range; longer = untrusted).
- **Reward** `r_t = Evaluate(s_t)`: `danger` (attacking role) / `−danger`,`containment` (defending) / `space_creation` (shaping). Computed geometry ⇒ **an agent cannot game a learned critic** (why RL is even defensible here).
- **Counterfactual quantity:** `Δ = E[R | A(c_B)] − E[R | A(c_A)]` over H, at fixed anchors.
- **Action** `a_t` and return-maximisation belong to **Mode B only**.

## The Scout ↔ twin bridge (the crux)
Injection point exists: `node_attributes(pidss, team_arrays, id2attr, attr_dim)` adds a **per-player** residual after
`enc()`. Two facts respected:
1. **Agnosticism preserved** — `attr=None` is a bitwise no-op; base `fulcrum_unified.pt` unchanged; capability enters only as a downstream label.
2. **The attr path is UNTRAINED today** — a no-op scaffold. Making capability *steer behaviour* is the real work of G1; not glossed.

**Grounding is measured, not invented:** `c` = Scout's validated traits + estimable kinematics. A "profile" is a real object; swapping profiles swaps measured capability.

---

## Gated build (each gate → `gate_reports/cf_<N>.json`, then HALT for human GO)

**G0 — Bridge sanity (CPU).** `attr=0` ≡ base bitwise; a nonzero attr on node *i* is injected only at *i*. **Pass:** identical at 0; single-node locality holds.

**G1 — Capability-grounding battery (the load-bearing gate).** Train `attr_proj` so capability *steers behaviour*, then
prove the steering is **grounded and disentangled** — not a spurious capability↔movement correlation. Trajectory
reconstruction (`conditioned error < base`) is **necessary but not sufficient**; it must also pass:

*Physical grounding*
| Intervention | Expected response |
|---|---|
| ↑ pace | ↑ reachable distance |
| ↑ acceleration | faster displacement onset |
| ↑ max velocity | greater depth coverage |
| ↑ deceleration | stronger stopping capability |

*Tactical grounding*
| Intervention | Expected response |
|---|---|
| ↑ progressive_intent | more forward / off-ball penetration |
| ↑ press_resistance | greater retention under pressure |
| ↑ space_creation | more occupation / creation of exploitable space |

*Negative controls (disentanglement — the crucial part)*
- ↑ `press_resistance` must **not** make the player faster.
- ↑ `pace` must **not** raise `progressive_intent`.
- Each axis moves its own response and **leaves the others ≈ unchanged** (report the full intervention × response matrix; off-diagonal ≈ 0).

**Pass:** signed-correct on-diagonal responses **and** near-zero off-diagonal (negative controls hold). **Honest fail:**
if the attr channel can't steer (cf. `twin_control_v3` `helped:false`) or can't disentangle, report and stop — the
counterfactual is not yet supported.

**G2 — Counterfactual monotonicity (Mode A).** Over many fixed anchors, sweep one axis / swap real profiles.
**Pass:** Δreward **monotonic, sign-correct**, effect > rollout noise, reported **per axis, per role, with CIs**.

**G2.5 — Null tests (trust-the-positive gate).** Before believing any positive effect:
- same player, same anchor, same capability, **different seed** → Δ ≈ 0 (no systematic effect).
- swap A → **synthetic profile with no meaningful capability difference** → Δ ≈ 0.
- **require `|Δ|(real swaps) > |Δ|(synthetic/random swaps)`** in explanatory power.
**Pass:** nulls are ≈ 0 and real swaps dominate synthetic — proves the twin isn't merely twitching at *any* perturbation.

**G3 — Sim-to-real anchor.** Baseline (condition on a player's **own** profile) reproduces their **real** next ~2 s
within the twin's validated rollout error. **Pass:** baseline ≈ twin-gate error; Δ is meaningful only where baseline is
faithful. Guards against confident fantasy.

**G4 — Agent RL (deferred; own signoff).** Train `π(a|s,c)` on the twin env. **Pass:** return > greedy/no-op, rollouts
stable within H, capability changes learned behaviour sensibly. **Honest negative allowed:** if unstable, ship Mode A,
mark Mode B `unproven`.

**G5 — Product composition.** Expose `Optimize.counterfactual(anchor, swap)` + the Scout **Tactical Fit** readout.
Epistemic status from gate results; naming per the claim ladder. Composition/driver only — no backbone change.

---

## The product this unlocks — "good for *what*"
Not *"a LW with 87 pace and 84 progressive carries"* (still a database). Instead: **who would change how our team
plays.** Built around five questions — and the first three ship on the **already-validated** Scout/geometry stack; only
**Simulate** is the research frontier, and it is **not a prerequisite for value**:

| # | Question | Needs | Status now |
|---|---|---|---|
| 1 | **Discover** — who exists? | market-wide capability search (Scout) | **buildable now** |
| 2 | **Diagnose** — what does our team actually need? | structural-deficiency read (Evaluate topology) | **buildable now** |
| 3 | **Match** — who fits that deficiency? | capability + tactical compatibility | **buildable now** |
| 4 | **Simulate** — what happens if we put them here? | counterfactual twin (this spec) | **research frontier** |
| 5 | **Explain** — why does the model think so? | topology + comparable phases + uncertainty | **buildable now** |

### Tactical compatibility function
`F(player, team, role, opponent, phase)` → fit. The **same** player yields different fit for different teams (+14% / +4%
/ −3%) **without** claiming the player got better or worse — their capabilities *interact* differently with each team's
existing geometry. This is precisely what flat player metrics cannot express.

### The Tactical Fit card (Stage-1 language)
```
PLAYER B                              Tactical Fit  █████████░ 91%
Why:  + high press_resistance   + strong off-ball penetration   + compatible accel profile
Counterfactual (fixed phases, ~2 s horizon):
  + 12% exploitable-space creation    + 8% progression opportunity    + 6% phase value
Strongest effect: opponent high press     Weakest effect: low block
Evidence: 184 comparable phases           Simulation validity: HIGH
```
Note the language: **"changes the model's predicted reward within horizon,"** not "will improve results."

### Architecture (no new backbone — exploit what's built)
```
                     FULCRUM CORE  ─ canonical geometric state
        ┌────────────────┼────────────────┐
     Evaluate         World Model         Scout
    danger/space      dynamics/value   capabilities/traits
        └────────────────┼────────────────┘
                 COUNTERFACTUAL ENGINE   (this spec)
                          │  swap capability profiles at fixed anchors
                Tactical Fit Impact  →  Scout product (Discover→Diagnose→Match→Simulate→Explain)
```

---

## Validation philosophy (matches the vault)
- **Per-role, per-axis, per-phase, with CIs — never one pooled number.** A fit that isn't sign-correct, monotonic, and null-separated is noise with a story.
- **Null tests before positives (G2.5).** Prove the twin reacts to *meaningful* capability, not to *any* perturbation.
- **Disentanglement is a gate, not a hope (G1 negative controls).**
- **Sim-to-real is a gate (G3).** Every claim lives inside the ~2 s the twin is trusted.
- **Claim only what's earned.** Fit → sim-validated fit → (maybe) signing impact. Never skip rungs.

## Risks / open questions (stated up front)
- **Controllability strength.** `twin_control_v3` lowered position error but `helped:false` on prediction gain. If the attr residual can't move behaviour enough, G1 stalls — reported, not hidden.
- **Disentanglement.** The attr channel may entangle axes; the negative-control matrix in G1 is designed to catch exactly this.
- **Horizon.** Effects must read within ~2 s; whole-possession what-ifs need the rollout gate extended first (out of scope).
- **Attribution.** G2 isolates the swapped node by holding anchors fixed and swapping only one node; G2.5 bounds twin-reactivity.
- **Capability realism.** Kinematics are directly estimable; higher-order traits are Scout latents with their own uncertainty — carried through, not laundered.

## What it closes
- **Scout, transformed:** from "who fits by percentile" to "who changes how we play, for which opponents, by how much — and how sure are we." A decision engine, on the same canonical state Scout reads, Broadcast narrates, and the twin plays forward. One representation, many products, **no forks.**
