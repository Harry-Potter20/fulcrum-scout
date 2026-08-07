---
title: "Fulcrum → Sport #2: a transfer map (basketball)"
tags: [fulcrum, decision-intelligence, transfer, basketball, abstraction]
status: mapping (not a build)
---

# Fulcrum → Sport #2: a concrete transfer map

**Purpose.** The decision-intelligence thesis is that Fulcrum's value is in its *constraints*, not football
specifics. This document *tests* that by mapping the machinery onto **one concrete second sport — basketball** —
and reporting what transfers cleanly, what needs re-parametrisation, and what genuinely breaks. Per the anti-
premature-abstraction rule ([[fulcrum-twin-controllability]]), the goal is to let the real abstractions be
**extracted from this case**, not designed ahead. Nothing here is built; it is a map to decide *whether* to build.

Basketball is chosen because the transfer is both plausible (5-a-side continuous invasion game, player+ball
tracking, spatial value) and *non-trivial* (a genuinely different geometry) — so it stress-tests the primitives
honestly rather than flattering them (field hockey would transfer too cleanly to be informative).

## The map, primitive by primitive

| Fulcrum primitive | Transfers to basketball? | What changes |
|---|---|---|
| **Relational attention** (nodes = players + ball) | **Clean.** 5v5 + ball = 11 nodes, same as football's graph over a smaller cast. | Node count / court scale only. The attention mechanism is geometry-agnostic. |
| **Agnosticism** (identity is a downstream label) | **Clean, and load-bearing.** Positions in, identity out — the exact refreshability property. | None. This is the abstraction most obviously sport-independent. |
| **covtoken** (coverage-constrained token tail) | **Clean.** A token economy that keeps rare, informative configurations from being averaged away is sport-independent; rare basketball events (a backdoor cut, a defensive rotation) are exactly what it protects. | Threshold recalibration only. |
| **Topology / `find_holes`** (pitch control → court control; holes = open lanes/shots) | **Core transfers; the prior does not.** Persistent-homology holes in a court-control field = driving lanes, open-shooter pockets, gaps in help defence. | The *danger prior* is football-specific (box + deep-cross zones 13/15). Replace with basketball value zones — **rim, corner-3, above-the-break-3, elbow** — the empirical value surface. This is a data swap, not a redesign. |
| **Differentiated metrics** (SC, containment) | **Transfers — and is arguably *more* valuable.** Off-ball space creation = the screen/cut/relocation that basketball is *built* around; containment = help-defence positioning that box-score stats miss even more than in football. | Same remove-and-recompute; only the value surface under it changes. |
| **Klein-4 pitch equivariance** | **PARTIALLY BREAKS — the honest finding.** Football's pitch has a 4-element symmetry (mirror across halfway × mirror across the touchline). A basketball **half-court set** has only **left-right mirror** symmetry (one basket breaks the long-axis symmetry); **full-court transition** restores a two-basket point symmetry but not the full Klein-4. | The equivariance group is **sport-specific** and must be re-derived per sport (here: C₂, not V₄). The *principle* (bake the sport's symmetry into the architecture) transfers; the *specific group* does not. |
| **Learned world model / rollout** (dynamics head) | **Transfers in principle; retraining required.** Player dynamics (accel limits, spacing) differ; the architecture is reusable, the weights are not. | Full retrain on basketball tracking; the shot-clock imposes a horizon the football model never had. |

## What this tells us about the abstractions (the point of the exercise)

**Earned as genuinely sport-independent** (survive both football *and* the basketball map): relational attention,
**agnosticism**, covtoken, the **topological hole/space machinery**, and the **remove-and-recompute attribution**
(SC / containment). These are the real core — an abstraction layer built around *these* would be extracted from
two proven cases, not guessed.

**Explicitly sport-specific** (must be re-parametrised, never abstracted into a false "universal"): the **danger /
value prior** (zone weights), the **equivariance group** (V₄ → C₂), and the **learned dynamics weights** + horizon.
A premature `WorldState/Agent/BasketballAdapter` layer would have wrongly frozen these as shared — the map shows
exactly which fields must stay pluggable.

## Data & feasibility (honest)

- **Tracking:** basketball player+ball tracking is mostly proprietary (Second Spectrum / Hawk-Eye). Open options
  are thin: the legacy 2015-16 SportVU logs (partial, deprecated), or synthetic/broadcast-GSR via the same
  YOLO→track→homography path Fulcrum already uses for football broadcast. **A broadcast-GSR route is the realistic
  entry**, mirroring the football GSR frontend on the roadmap.
- **Effort to a first proof:** re-parametrise the value prior + symmetry group (small), reuse topology/attribution
  as-is (zero), retrain dynamics if simulation is wanted (large — defer; the *analysis* stack needs no training,
  exactly as in football).

## Recommendation

**Do not build the abstraction layer yet.** This single map already shows the split cleanly, but one case is not
two proofs — the abstractions are *hypothesised-earned*, not earned. If a sport #2 is pursued, do it as a **thin
basketball analysis slice** (topology + SC/containment on a handful of tracked possessions, value prior re-zoned,
C₂ symmetry) — the same "prove one capability first" discipline that carried football. The generic platform layer
gets extracted *after* that slice runs, from the two concrete codebases, not from this document.

See also: [[football-sportsmed-vision]] (the eventual multi-domain goal), MODEL_CARD.md (the football capabilities
this map transfers from).
