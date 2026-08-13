"""fulcrum.metrics — differentiated topology-derived stats that xG / xA / xThreat structurally CANNOT capture.

xG/xA/xT are outcome-calibrated and BALL-centric: they can only value on-ball event actions (shot, pass-to-shot,
ball-progression). These metrics are computed geometry over the WHOLE configuration, so they attribute danger to
the MECHANISM — who made the space, who denied it — including OFF-BALL, non-event contributions (~most of what
players do, ~none of what event data sees). No outcome calibration, no model; pure `find_holes` remove-and-recompute.

  SPACE CREATION (SC) : per off-ball ATTACKER — remove them, recompute danger; the DROP is the space their
                        movement/presence created. The decoy run that opens a channel: high SC, zero xA/xG.
  CONTAINMENT         : per DEFENDER — remove them, recompute danger; the RISE is the danger they suppress by
                        POSITIONING (invisible defending — not tackles/interceptions).

Uses top-k exposure (not a single hole) so removals reflect OVERALL space, not a relocatable one. Identity is a
downstream label — pass `att_ids` / `dfn_ids` in the state; the computation never uses names.
"""
from __future__ import annotations
import numpy as np
from . import core as _fc


def _topk(att, dfn, ball, k=3):
    att = np.asarray(att, float); dfn = np.asarray(dfn, float); ball = np.asarray(ball, float)
    if len(att) < 4 or len(dfn) < 4:
        return 0.0
    _, _, _, holes = _fc.find_holes(att, np.zeros_like(att), dfn, np.zeros_like(dfn), ball, top=k)
    return float(sum(h["score"] for h in holes))


def space_creation(state, k=3, exclude_ball_carrier=True):
    """Per-ATTACKER OFF-BALL space creation. state = {att, dfn, ball[, att_ids]}. -> {att_id: sc>=0}.
    Excludes the ball-carrier (nearest attacker to the ball) to isolate off-ball movement from on-ball dribbling."""
    att = np.asarray(state["att"], float); dfn = np.asarray(state["dfn"], float); ball = np.asarray(state["ball"], float)
    ids = state.get("att_ids", list(range(len(att))))
    if len(att) < 5 or len(dfn) < 4:
        return {}
    base = _topk(att, dfn, ball, k)
    carrier = int(np.linalg.norm(att - ball, axis=1).argmin()) if exclude_ball_carrier else -1
    return {ids[i]: max(base - _topk(np.delete(att, i, 0), dfn, ball, k), 0.0)
            for i in range(len(att)) if i != carrier}


def containment(state, k=3):
    """Per-DEFENDER danger suppression by positioning. state = {att, dfn, ball[, dfn_ids]}. -> {dfn_id: c>=0}."""
    att = np.asarray(state["att"], float); dfn = np.asarray(state["dfn"], float); ball = np.asarray(state["ball"], float)
    ids = state.get("dfn_ids", list(range(len(dfn))))
    if len(att) < 4 or len(dfn) < 5:
        return {}
    base = _topk(att, dfn, ball, k)
    return {ids[j]: max(_topk(att, np.delete(dfn, j, 0), ball, k) - base, 0.0) for j in range(len(dfn))}


def shape_influence(positions, ids=None):
    """Per-player contribution to THEIR OWN group's shape — remove them, recompute compactness (x-spread) and
    width (y-spread); the size of the change is how much their positioning holds that shape together. Symmetric
    and team-agnostic (unlike space_creation/containment, which are asymmetric by design): pass a defensive
    block's positions to read defensive compactness, or an attacking line's to read attacking width — same
    function either way. No find_holes call, so it's cheap relative to the danger-based metrics.

    Removing a player near the group's positional extremes (widest, deepest) changes the std a lot — high
    shape_influence — while removing someone near the average barely moves it. This is a genuinely different
    signal from containment/space_creation: it's about STRUCTURAL cohesion (does the group hold its shape),
    not immediate danger. positions: Nx2 array, one team's outfield players. ids: optional track ids.
    -> {id: {"compactness_delta": signed, "width_delta": signed, "shape_influence": magnitude}}"""
    pos = np.asarray(positions, float)
    ids = list(ids) if ids is not None else list(range(len(pos)))
    if len(pos) < 5:
        return {}
    base_vert, base_horiz = float(pos[:, 0].std()), float(pos[:, 1].std())
    out = {}
    for i in range(len(pos)):
        rest = np.delete(pos, i, 0)
        cd = float(rest[:, 0].std()) - base_vert
        wd = float(rest[:, 1].std()) - base_horiz
        out[ids[i]] = {"compactness_delta": round(cd, 3), "width_delta": round(wd, 3),
                       "shape_influence": round(abs(cd) + abs(wd), 3)}
    return out


def state_summary(state, k=3):
    """One-pass computed-topology summary of a state (positions only — no ids, no model). For scale validation:
    -> {danger, sc} where
        danger = top-k exposure (sum of hole scores)                       — the validated space signal
        sc     = total OFF-BALL space creation (sum over off-ball attackers) — distinct off-ball attribution
    Returns None if the configuration is too sparse to admit holes.

    Note: `find_holes` reports each hole's topological `persistence`, but for the TOP hole persistence *equals*
    its score (danger is derived from it) — so state-level persistence is not a signal distinct from danger.
    The differentiated persistence signal is TEMPORAL (a gap that survives across frames vs a flicker); that lives
    in `fulcrum.temporal` (track_holes / flicker_stats) and needs continuous tracking, not discrete snapshots."""
    att = np.asarray(state["att"], float); dfn = np.asarray(state["dfn"], float); ball = np.asarray(state["ball"], float)
    if len(att) < 5 or len(dfn) < 4:
        return None
    _, _, _, holes = _fc.find_holes(att, np.zeros_like(att), dfn, np.zeros_like(dfn), ball, top=k)
    if not holes:
        return None
    return {"danger": float(sum(h["score"] for h in holes)), "sc": float(sum(space_creation(state, k).values()))}


def player_metrics(states, k=3):
    """Aggregate SC + containment per player over many oriented states (state_at format with ids).
    -> {"space_creation": {id: total}, "containment": {id: total}, "states": {id: n}}."""
    from collections import defaultdict
    sc, cont, pres = defaultdict(float), defaultdict(float), defaultdict(int)
    for st in states:
        for pid, v in space_creation(st, k).items():
            sc[pid] += v; pres[pid] += 1
        for pid, v in containment(st, k).items():
            cont[pid] += v; pres[pid] += 1
    return {"space_creation": dict(sc), "containment": dict(cont), "states": dict(pres)}
