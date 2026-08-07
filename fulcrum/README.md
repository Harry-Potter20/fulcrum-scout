# Fulcrum

**A player-agnostic, geometric world model for football — and the platform built on it.**

Fulcrum reads the *geometry* of a match — where players and the ball are, how they move — and turns any spatial
source (optical tracking, broadcast video, event freeze-frames) into one **canonical state**, then runs the same
analysis on all of them. Identity is never an input; it is a downstream label. That single constraint is what lets
tracking data, a broadcast clip, and a StatsBomb-360 freeze-frame all feed the *same* services unchanged.

> Traditional scouting tells you **who produced**. Fulcrum is built to tell you **who creates the conditions that
> produce** — and, as the world-model matures, what happens if you change the system.

## The idea

- **Canonical State** — a frozen ABI: positions (metres), velocities, team role, ball, mask. Identity-free by
  construction (`tests/test_canonical.py` fails the build if an identity field leaks in).
- **Six Foundation Services** over that state — `Encode · Evaluate · Predict · Explain · Retrieve · Optimize`.
  Every output is tagged with an **epistemic status** (`validated · face-valid · descriptor · works · unproven`), so a
  product composing an unproven signal inherits that label and can't silently over-claim.
- **Registry, not forks** — one service interface, swappable driver per domain (`football-v3` live; `basketball-v1`
  a scaffold awaiting data + a C₂-equivariant encoder). A new sport is an entry, never a fork.
- **Products = compositions** of the services (Scout, Coach, Broadcast) — never modifications of the model.

## What's real (honest status)

| Capability | Status |
|---|---|
| Off-ball **Space Creation** (topology) | validated (1.4–1.6× off-ball) |
| Danger / hole-finding (pre-shot space) | validated (shot-precursor AUC 0.886) |
| Cross-league **translation** projection | validated (+7.9% out-of-sample) |
| **Twin** generalisation (perturbed play) | validated (+25.6% unseen) |
| Recognition-free **video ingestion** | validated (positions 0.17 m, colour-teams 0.965 vs GT) |
| Latent trait `progressive_intent` | validated (separates roles, η²=0.13) |
| Structural exposure, containment | face-valid (not yet outcome-tested) |
| Counterfactual decision-quality | **unproven** (retracted; do not rely on) |

Service **robustness** under degraded ingestion is characterised (`robustness_multi.json`): landmark signals are
robust, difference signals fragile, team-assignment the dominant vulnerability. `fulcrum.confidence` turns that table
into a per-output confidence keyed by estimated ingestion quality.

## Scout — the differentiated product

- **Layer A (all players, stats only):** `derive_insights` / `player_profiles` — over/under-performance residuals,
  latent style, role-relative percentiles, comparables, and a **generated scouting description**.
- **Layer B (spatial-covered):** off-ball geometry from 360 freeze-frames / tracking (`freeze_frame_to_state`).
- **Estimability boundary (tested):** from stats you can estimate *where* a player operates (advancement, held-out
  corr 0.57) but **not** the topological danger signal (0.04) — that stays geometry-exclusive. `estimate_spatial`
  serves the estimable part, tagged.

## Install & use

```python
import fulcrum
frames = fulcrum.load_match("skillcorner", 4039)          # any tracking source -> canonical
state  = fulcrum.state_at(frames, fps=10, fid=1500)
fulcrum.services.evaluate(state)                          # danger, space, containment — each status-tagged
fulcrum.player_profiles(records)                          # stat-tier scouting intelligence + generated notes
```

Topology-only use needs just `numpy`; the learned encoder adds `torch`. Works on any tracking source with no
retraining (Metrica, SkillCorner) and on broadcast video via the recognition-free ingestion adapter.

## Status

Research-stage. Signals are labelled by what they've actually been validated to do; several are descriptors or
unproven and say so. Contributions and falsification welcome — a trait that doesn't separate is dropped, not renamed.
