"""fulcrum.simulate — the tactical PLANNER engine: control as STRUCTURAL graph edits, reward as COMPUTED topology.

This is the engine the network becomes a component of. It is NOT a learned controller. Given a game state and a
tactical GOAL, it searches over *structural interventions* on the state graph (move a player, shift the line) and
returns the edit that best achieves the goal — scored by the **computed** danger surface (`fulcrum.core.find_holes`),
an exact, unlearned, unhackable reward.

Grounded by the graph-intervention experiment (2026-08): a structural do() produces a directionally-correct
topology change (80.7%) at *zero* cost to any predictor — so search + computed topology is a sound tactical
optimiser without training. The frozen world model can later score an edit's *dynamic* consequence (rollout);
this first slice uses the exact immediate topology, which is the clean, validated signal.

Design law holds: the planner works on ANONYMOUS positions. An edit refers to a player by index/location; mapping
"the defender at (78,44)" to a name is the downstream identity sidecar — identity never enters the search.
"""
from __future__ import annotations
import copy
import numpy as np
from . import core as _fc


def state_danger(state):
    """(danger, top_hole) for a state dict {att, dfn, ball[, att_v, dfn_v]}. Danger = top persistent-hole score."""
    att = np.asarray(state["att"], float); dfn = np.asarray(state["dfn"], float); ball = np.asarray(state["ball"], float)
    if len(att) < 4 or len(dfn) < 4:
        return 0.0, None
    av = np.asarray(state.get("att_v") if state.get("att_v") is not None else np.zeros_like(att), float)
    dv = np.asarray(state.get("dfn_v") if state.get("dfn_v") is not None else np.zeros_like(dfn), float)
    _, _, _, holes = _fc.find_holes(att, av, dfn, dv, ball, top=1)
    return (float(holes[0]["score"]), holes[0]) if holes else (0.0, None)


_L, _W = _fc.PITCH_L, _fc.PITCH_W


def _clamp(p):
    """Keep an edited position ON the pitch (legality). Stress-tested fix (2% of raw edits went off-pitch)."""
    return [min(max(float(p[0]), 0.0), _L), min(max(float(p[1]), 0.0), _W)]


def total_danger(state, k=3):
    """The planner's REWARD: sum of the top-k persistent holes = OVERALL defensive exposure.

    Optimising a single hole (top=1) lets a search RELOCATE danger rather than reduce it — the stress test
    (2026-08) showed a top-1 'close' cut the biggest hole by 0.060 while OVERALL (top-3) danger rose +0.007.
    Top-k measures total exposure and is far harder to relocate-game. This is the reward all planners use."""
    att = np.asarray(state["att"], float); dfn = np.asarray(state["dfn"], float); ball = np.asarray(state["ball"], float)
    if len(att) < 4 or len(dfn) < 4:
        return 0.0
    av = np.asarray(state.get("att_v") if state.get("att_v") is not None else np.zeros_like(att), float)
    dv = np.asarray(state.get("dfn_v") if state.get("dfn_v") is not None else np.zeros_like(dfn), float)
    _, _, _, holes = _fc.find_holes(att, av, dfn, dv, ball, top=k)
    return float(sum(h["score"] for h in holes))


def apply_edit(state, side, index, by):
    """Return a new state with player `index` on `side` ('att'|'dfn') moved by vector `by`, clamped on-pitch."""
    s2 = copy.deepcopy(state)
    pts = [list(map(float, p)) for p in s2[side]]
    pts[index] = _clamp([pts[index][0] + float(by[0]), pts[index][1] + float(by[1])])
    s2[side] = pts
    return s2


def _candidate_edits(state, side, step, dirs, line_shift):
    """Structural edits to search: each player on `side` nudged in `dirs` directions of length `step`, plus
    whole-line x-shifts. Deliberately small + interpretable — a coach's vocabulary, not arbitrary vectors."""
    edits = []
    n = len(state[side])
    for i in range(n):
        for a in range(dirs):
            ang = 2 * np.pi * a / dirs
            edits.append((side, i, (step * np.cos(ang), step * np.sin(ang)), f"move {side} #{i}"))
    if line_shift:
        for dx in (-step, step):                                   # push the whole side higher / drop it deeper
            edits.append((side, "LINE", (dx, 0.0), f"shift {side} line {'up' if dx > 0 else 'back'}"))
    return edits


def _apply_any(state, side, idx, by):
    if idx == "LINE":
        s2 = copy.deepcopy(state)
        s2[side] = [_clamp([float(p[0]) + float(by[0]), float(p[1]) + float(by[1])]) for p in s2[side]]
        return s2
    return apply_edit(state, side, idx, by)


def plan(state, goal="reduce_danger", side=None, step=4.0, dirs=8, line_shift=True, top=5, k=3):
    """Search structural edits for a tactical goal and return the best, scored by computed topology (top-k
    OVERALL exposure — see total_danger; single-hole scoring is relocate-gameable).

    goal: 'reduce_danger' (default side='dfn' — how the DEFENCE closes the space) or
          'increase_danger' (default side='att' — the ATTACKING movement that opens the most space).
    Returns {base_danger, top_hole, goal, best:[{op, side, index, by, resulting_danger, delta_danger, note}]}.
    """
    side = side or ("dfn" if goal == "reduce_danger" else "att")
    base = total_danger(state, k); _, hole = state_danger(state)          # reward = top-k exposure; hole = for context
    scored = []
    for (sd, idx, by, note) in _candidate_edits(state, side, step, dirs, line_shift):
        d = total_danger(_apply_any(state, sd, idx, by), k)
        scored.append({"op": "shift_line" if idx == "LINE" else "move_player", "side": sd, "index": idx,
                       "by": [round(float(by[0]), 1), round(float(by[1]), 1)],
                       "resulting_danger": round(d, 3), "delta_danger": round(d - base, 3), "note": note})
    reduce = goal == "reduce_danger"
    scored.sort(key=lambda e: e["delta_danger"] if reduce else -e["delta_danger"])   # most-negative / most-positive first
    return {"base_danger": round(base, 3), "reward": f"top-{k} exposure", "goal": goal, "side": side,
            "top_hole": ({"x": round(hole["x"], 1), "y": round(hole["y"], 1)} if hole else None),
            "best": scored[:top]}


def plan_report(state, step=4.0, top=3):
    """Bidirectional tactical readout for one state: the best DEFENSIVE close and the best ATTACKING exploit,
    both from computed topology. The core deliverable of the planner engine."""
    base, hole = state_danger(state)
    return {
        "base_danger": round(base, 3),
        "top_hole": ({"x": round(hole["x"], 1), "y": round(hole["y"], 1)} if hole else None),
        "defensive_close": plan(state, "reduce_danger", "dfn", step=step, top=top)["best"],
        "attacking_exploit": plan(state, "increase_danger", "att", step=step, top=top)["best"],
    }


def plan_multi(state, goal="reduce_danger", side=None, depth=3, beam=4, step=4.0, dirs=8, k=3):
    """Beam search over COORDINATED sequences of structural edits — a reorganisation, not a single nudge. Each
    step moves a *different* player (coordination), scored by top-k OVERALL exposure (total_danger). Returns the
    best sequence and its total effect. This is where single-edit deltas amplify (move 3 defenders, not one)."""
    side = side or ("dfn" if goal == "reduce_danger" else "att")
    reduce = goal == "reduce_danger"
    base = total_danger(state, k)
    beams = [(state, [], base)]                                     # (state, edit-sequence, danger)
    for _ in range(depth):
        cands = []
        for (st, edits, _dng) in beams:
            used = {e["index"] for e in edits}
            for (sd, i, by, note) in _candidate_edits(st, side, step, dirs, False):
                if i in used:                                      # one move per player -> genuine reorganisation
                    continue
                st2 = _apply_any(st, sd, i, by); d2 = total_danger(st2, k)
                cands.append((st2, edits + [{"index": i, "by": [round(float(by[0]), 1), round(float(by[1]), 1)],
                                             "note": note, "danger_after": round(d2, 3)}], d2))
        if not cands:
            break
        cands.sort(key=lambda c: c[2] if reduce else -c[2])
        beams = cands[:beam]
    best = beams[0]
    return {"base_danger": round(base, 3), "goal": goal, "side": side, "depth": len(best[1]),
            "total_delta_danger": round(best[2] - base, 3), "final_danger": round(best[2], 3), "sequence": best[1]}


# ---------------------------------------------------------------------------
# Dynamic planning: score edits by the FROZEN world model's ROLLED-FORWARD consequence.
# The world model re-enters the loop here — not as a controller, but as an *evaluator* of an edit's future.
# ---------------------------------------------------------------------------

def _roll_final(model, window, K, device):
    """Roll the frozen model K steps from a Window -> (final_positions[N,2], team[N], mask[N]) as numpy."""
    import torch
    from . import model as _m
    b = _m.collate([_m.make_sample(window, _m.RS)], _m.MAX_NODES); b = {k: v.to(device) for k, v in b.items()}
    pos, vel, acc = b["pos"], b["vel"], b["acc"]
    team, isb, mask, ev = b["team"], b["isball"], b["mask"], b["event_ctx"]
    for _ in range(K):
        feat = _m.featurize_torch(pos, vel, acc, team, isb, mask, ev)
        with torch.no_grad():
            mu, _ = model(feat, pos, team, mask)
        step = vel * _m.HORIZON_S + mu * _m.RS
        newpos = pos + step
        newvel = (newpos - pos) / _m.HORIZON_S
        acc = (newvel - vel) / _m.HORIZON_S
        pos, vel = newpos, newvel
    return pos[0].cpu().numpy(), team[0].cpu().numpy(), mask[0].cpu().numpy()


def _future_danger(model, window, K, device, k=3):
    pos, team, mask = _roll_final(model, window, K, device)
    v = mask > 0.5
    att = pos[(team == 1.0) & v]; dfn = pos[(team == 0.0) & v]; ball = pos[0]
    if len(att) < 4 or len(dfn) < 4:
        return 0.0
    _, _, _, holes = _fc.find_holes(att, np.zeros_like(att), dfn, np.zeros_like(dfn), ball, top=k)   # top-k exposure
    return float(sum(h["score"] for h in holes)) if holes else 0.0


def _edit_window(window, side, index, by):
    w2 = copy.deepcopy(window)
    w2.pos = np.asarray(window.pos, float).copy()
    nodes = np.where(np.asarray(window.team) == (1.0 if side == "att" else 0.0))[0]
    by = np.asarray(by, float)
    tgt = nodes if index == "LINE" else nodes[index:index + 1]
    w2.pos[tgt] = np.clip(w2.pos[tgt] + by, [0.0, 0.0], [_L, _W])       # move + clamp on-pitch (legality)
    return w2


def plan_dynamic(model, window, goal="reduce_danger", side=None, step=4.0, dirs=8, K=4,
                 line_shift=True, top=5, device="cpu"):
    """Like `plan`, but scores each structural edit by its DYNAMIC consequence: apply the edit, roll the FROZEN
    world model forward K steps, and evaluate the resulting future's topology. Captures what an edit LEADS TO
    (developing danger), not just its immediate geometric effect. Slower — one rollout per candidate.

    The frozen predictor's interventional response is coherent but MODEST (per the graph-intervention result),
    so dynamic deltas are smaller than `plan`'s immediate-topology deltas; this scores the *developing*
    consequence, complementary to the immediate one. `window` carries velocities (needed for the rollout)."""
    side = side or ("dfn" if goal == "reduce_danger" else "att")
    base = _future_danger(model, window, K, device)
    nodes = np.where(np.asarray(window.team) == (1.0 if side == "att" else 0.0))[0]
    cands = [(i, (step * np.cos(2 * np.pi * a / dirs), step * np.sin(2 * np.pi * a / dirs)), f"move {side} #{i}")
             for i in range(len(nodes)) for a in range(dirs)]
    if line_shift:
        cands += [("LINE", (dx, 0.0), f"shift {side} line {'up' if dx > 0 else 'back'}") for dx in (-step, step)]
    scored = []
    for (idx, by, note) in cands:
        fut = _future_danger(model, _edit_window(window, side, idx, by), K, device)
        scored.append({"op": "shift_line" if idx == "LINE" else "move_player", "side": side, "index": idx,
                       "by": [round(float(by[0]), 1), round(float(by[1]), 1)],
                       "future_danger": round(fut, 3), "delta_future_danger": round(fut - base, 3), "note": note})
    reduce = goal == "reduce_danger"
    scored.sort(key=lambda e: e["delta_future_danger"] if reduce else -e["delta_future_danger"])
    return {"base_future_danger": round(base, 3), "goal": goal, "side": side, "horizon_steps": K, "best": scored[:top]}


# ---------------------------------------------------------------------------
# Live bridge + value-blended reward. No training — the value head is already trained (fulcrum_unified.pt).
# ---------------------------------------------------------------------------

def window_state(window):
    """Bridge a `Window` (learned-model format) -> the state dict this module plans on (att/dfn/ball + velocities),
    oriented so the attacked goal is at +x (ball-half heuristic). Lets the planner run straight off the live/analysis
    pipeline: `plan_report(window_state(w))`."""
    pos = np.asarray(window.pos, float); vel = np.asarray(window.vel, float); team = np.asarray(window.team)
    ball = pos[0]; right = float(ball[0]) > _L / 2
    op = (lambda p: (float(p[0]), float(p[1]))) if right else (lambda p: (_L - float(p[0]), _W - float(p[1])))
    ov = (lambda v: (float(v[0]), float(v[1]))) if right else (lambda v: (-float(v[0]), -float(v[1])))
    am, dm = team == 1.0, team == 0.0
    return {"att": [op(p) for p in pos[am]], "dfn": [op(p) for p in pos[dm]],
            "att_v": [ov(v) for v in vel[am]], "dfn_v": [ov(v) for v in vel[dm]], "ball": op(ball)}


def _window_value(model, window, device):
    """The trained value head on a Window -> possession value (P(shot within ~8s)). No training; just inference."""
    import torch
    from . import model as _m
    b = _m.collate([_m.make_sample(window, _m.RS)], _m.MAX_NODES)
    b = {k: v.to(device) for k, v in b.items()}
    with torch.no_grad():
        _, _, vl = model(b["feat"], b["pos"], b["team"], b["mask"], return_value=True)
    return float(torch.sigmoid(vl)[0])


def _window_topk(window, k):
    pos = np.asarray(window.pos, float); team = np.asarray(window.team)
    att = pos[team == 1.0]; dfn = pos[team == 0.0]; ball = pos[0]
    if len(att) < 4 or len(dfn) < 4:
        return 0.0
    _, _, _, holes = _fc.find_holes(att, np.zeros_like(att), dfn, np.zeros_like(dfn), ball, top=k)
    return float(sum(h["score"] for h in holes))


def plan_value(model, window, goal="reduce_danger", side=None, step=4.0, dirs=8, k=3, lam=0.5, top=5, device="cpu"):
    """Reward = top-k topology exposure + lam * possession value (the validated value head). Blends SPACE
    (topology) with SHOT-THREAT (value, validated → xG ρ=0.42). Static (no rollout); uses the model only for the
    value term (no training).

    HONEST FINDING (sensitivity-tested 2026-08): the value head is a POOLED whole-configuration scalar and barely
    moves under a SINGLE-player edit — across 44 single edits its range was 0.004 vs 0.624 for topology (ratio
    ~0.01), and it did not change a single recommendation. So value-in-reward is effectively a NO-OP for
    local/modest edits; topology-only (`plan`) is the correct default reward. Value may matter only for LARGE
    multi-team reorganisations (the aggressive regime). Kept here for that case + transparency; default to `plan`."""
    side = side or ("dfn" if goal == "reduce_danger" else "att")
    def reward(w):
        return _window_topk(w, k) + lam * _window_value(model, w, device)
    base = reward(window)
    nodes = np.where(np.asarray(window.team) == (1.0 if side == "att" else 0.0))[0]
    cands = [(i, (step * np.cos(2 * np.pi * a / dirs), step * np.sin(2 * np.pi * a / dirs)))
             for i in range(len(nodes)) for a in range(dirs)]
    scored = []
    for (idx, by) in cands:
        w2 = _edit_window(window, side, idx, by)
        scored.append({"side": side, "index": idx, "by": [round(float(by[0]), 1), round(float(by[1]), 1)],
                       "reward": round(reward(w2), 3), "delta": round(reward(w2) - base, 3),
                       "topology": round(_window_topk(w2, k), 3), "value": round(_window_value(model, w2, device), 3)})
    reduce = goal == "reduce_danger"
    scored.sort(key=lambda e: e["delta"] if reduce else -e["delta"])
    return {"base_reward": round(base, 3), "reward_def": f"top-{k} topology + {lam}*value", "goal": goal, "side": side, "best": scored[:top]}
