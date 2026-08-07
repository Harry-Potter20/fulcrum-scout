"""fulcrum.analysis — the analysis API over Fulcrum, a player-agnostic geometric WORLD MODEL for football.

Fulcrum is not a single-metric tool. It is a shared relational spatiotemporal encoder (multi-task: dynamics,
state value, structured concepts, retrieval) fused with a *computed* persistent-homology topology engine, built
on four constraint-first primitives — relational attention, Klein-4 pitch equivariance, covtoken (a coverage-
constrained tail whose dual variable is the controller), and topology. Working from anonymous geometry, it can:

  * VALUE      states and space         — possession value; exploitable-space danger (find_holes)
  * PREDICT    how the play evolves     — rollout-aware world model (dynamics head)
  * SIMULATE   by playing style         — the attribute-conditioned twin; generalises to UNSEEN players
                                          (+25.6% leave-match-out); analyse/simulate toggle on one weight set
  * ATTRIBUTE  per player               — topological remove-and-recompute, possession-aware
  * DISCOVER   & DESCRIBE               — anomaly, position-residualised gamestate discovery, grounded language

It is SOURCE-agnostic (StatsBomb 360, SkillCorner tracking, broadcast GSR), REFRESHABLE (swap the data -> current
players), and its geometric core transfers across sports. Its power comes from the *constraints*, not the
backbone — no new architecture is needed for the above.

This module exposes the consolidated ANALYSIS surface (value / danger / chance-creation / per-player / phase
report + grounded narration).

CALIBRATION EVIDENCE (StatsBomb-360, 2026-07 — this MEASURES two heads; it is not the limit of what Fulcrum does):
  - value head vs StatsBomb xG :  Spearman rho = 0.42  [0.35, 0.50]   (learns shot-quality structure from
                                  geometry alone; never trained on xG)
  - topological danger -> chance:  3.18x top-decile shot-soon lift    (exploitable space forecasts chances)
Positioned COMPLEMENTARY to xG (Fulcrum values the phase *before* the shot, which xG cannot see); it is not a
match-outcome or betting predictor — that was never its purpose.

DESIGN LAW (enforced by tests/test_agnosticism.py): identity NEVER enters the model. Anonymous geometry in;
identity and attributes are downstream LABELS attached to outputs, never features.

The learned world model currently lives in `worldmodel` (Football_Research/jobs/worldmodel.py) and must be
importable; vendoring it into the package is a tracked hardening follow-up. Topology is `fulcrum.core.find_holes`.
"""
from __future__ import annotations
import numpy as np
from . import core as _fc
from . import model as _m          # the vendored world-model interface (fulcrum.model)
from . import opposition as _opp   # formation / pressing structure (computed, state_at format)

PITCH_L, PITCH_W = _m.PITCH_L, _m.PITCH_W


def load(checkpoint: str, device: str = "cpu"):
    """Load a trained Fulcrum checkpoint (e.g. fulcrum_unified.pt) -> (model, checkpoint_dict)."""
    return _m.load_checkpoint(checkpoint, device=device)


def score(model, windows, device: str = "cpu", bs: int = 256):
    """Per-state tactical signals — the validated heads. -> (value[N], danger[N]).

    value  : possession value (validated vs xG, rho=0.42) — the pre-shot worth of the configuration.
    danger : topological exploitable-space score (find_holes; 3.18x chance-creation lift).
    Identity is never used; a `window` carries only geometry (positions/velocities/team/ball)."""
    import torch
    model.eval()
    val = []
    for i in range(0, len(windows), bs):
        b = _m.collate([_m.make_sample(w, _m.RS) for w in windows[i:i + bs]], _m.MAX_NODES)
        b = {k: v.to(device) for k, v in b.items()}
        with torch.no_grad():
            _, _, vl = model(b["feat"], b["pos"], b["team"], b["mask"], return_value=True)
        val.append(torch.sigmoid(vl).cpu().numpy())
    dng = []
    for w in windows:
        att = w.pos[w.team == 1.0]; dfn = w.pos[w.team == 0.0]
        if len(att) >= 4 and len(dfn) >= 4:
            _, _, _, h = _fc.find_holes(att, np.zeros_like(att), dfn, np.zeros_like(dfn), w.pos[0], top=1)
            dng.append(h[0]["score"] if h else 0.0)
        else:
            dng.append(0.0)
    return np.concatenate(val) if val else np.array([]), np.array(dng)


def _zone(ball):
    x, y = float(ball[0]), float(ball[1])
    third = "attacking third" if x > 70 else "middle third" if x > 35 else "defensive third"
    lane = "left" if y < PITCH_W / 3 else "right" if y > 2 * PITCH_W / 3 else "central"
    return f"{lane} of the {third}"


def narrate(window, value: float, danger: float) -> str:
    """Grounded, non-hallucinating one-line description of a phase — every clause maps to a computed signal."""
    return (f"Ball in the {_zone(window.pos[0])}; danger {danger:.2f}, phase value {value:.2f}"
            + (" — high chance-creation potential." if danger > 0.6 else "."))


def _window_to_state(window):
    """Representation-seam bridge: a `Window` (learned-model format) -> the `state_at` dict (topology/opposition
    format), oriented so the attacked goal is at +x via the ball-half heuristic (matches fulcrum.data.state_at).
    Attack/defend split comes from the possession team labels the window already carries. Most reliable on
    attacking-phase states (high danger), where the ball is in the attacking half."""
    ball = window.pos[0]
    right = float(ball[0]) > PITCH_L / 2                       # attacking +x already? else flip the frame
    opos = (lambda p: (float(p[0]), float(p[1]))) if right else (lambda p: (PITCH_L - float(p[0]), PITCH_W - float(p[1])))
    ovel = (lambda v: (float(v[0]), float(v[1]))) if right else (lambda v: (-float(v[0]), -float(v[1])))
    am, dm = window.team == 1.0, window.team == 0.0
    return {"att": [opos(p) for p in window.pos[am]], "dfn": [opos(p) for p in window.pos[dm]],
            "att_v": [ovel(v) for v in window.vel[am]], "dfn_v": [ovel(v) for v in window.vel[dm]],
            "ball": opos(ball)}


def tactical_shape(window) -> dict:
    """Opposition read at one phase — both teams' FORMATION + the defending team's PRESSING STRUCTURE — bridged
    from a Window into fulcrum.opposition. Computed (no model). Identity never involved."""
    return _opp.opposition_report(_window_to_state(window))


def chance_creation(model, windows, meta=None, device: str = "cpu", top: int = 10):
    """Rank phases by chance-creation potential (danger), with grounded narration. Returns list of dicts."""
    val, dng = score(model, windows, device)
    order = np.argsort(-dng)[:top]
    out = []
    for i in order:
        row = {"danger": round(float(dng[i]), 3), "value": round(float(val[i]), 3),
               "note": narrate(windows[i], float(val[i]), float(dng[i]))}
        if meta and i < len(meta):
            row.update({k: meta[i].get(k) for k in ("player", "team", "minute") if k in meta[i]})
        out.append(row)
    return out


def per_player(model, windows, meta, device: str = "cpu", min_touches: int = 8, top: int = 12):
    """Per-player chance-creation contribution (value accumulated in the decisive tail). Names come from `meta`
    (downstream labels) — the model never saw them. Returns players ranked by decisive contribution."""
    from collections import defaultdict
    val, dng = score(model, windows, device)
    thr = float(np.quantile(val, 0.90)) if len(val) else 0.0
    created, touches, decisive = defaultdict(float), defaultdict(int), defaultdict(float)
    for i, m in enumerate(meta):
        p = m.get("player")
        if p is None:
            continue
        created[p] += val[i]; touches[p] += 1
        if val[i] >= thr:
            decisive[p] += val[i]
    rows = [{"player": p, "touches": touches[p], "value_per_touch": round(created[p] / touches[p], 3),
             "decisive_contribution": round(decisive[p], 2)} for p in touches if touches[p] >= min_touches]
    rows.sort(key=lambda r: -r["decisive_contribution"])
    return rows[:top]


def phase_report(model, windows, meta=None, device: str = "cpu") -> dict:
    """The consolidated Fulcrum phase report — the repeatable deliverable. Per-phase tactical value only
    (validated capability); complementary to xG, not a match/betting predictor."""
    val, dng = score(model, windows, device)
    rep = {
        "positioning": "per-phase tactical value (space, chance creation) — complementary to xG, not a match predictor",
        "n_states": len(windows),
        "summary": {"mean_value": round(float(val.mean()), 3) if len(val) else None,
                    "mean_danger": round(float(dng.mean()), 3) if len(dng) else None,
                    "high_danger_share": round(float((dng > 0.6).mean()), 3) if len(dng) else None},
        "top_chance_creation": chance_creation(model, windows, meta, device, top=10),
    }
    if len(dng):                                               # the opposition shape at the key (peak-danger) moment
        peak = int(np.argmax(dng))
        rep["tactical_at_peak"] = {"danger": round(float(dng[peak]), 3), "shape": tactical_shape(windows[peak])}
    if meta and any("player" in m for m in meta):
        rep["per_player"] = per_player(model, windows, meta, device)
    return rep
