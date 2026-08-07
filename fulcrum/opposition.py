"""fulcrum.opposition — read the opposition: formation, pressing structure, and what it's likely to change to.

Built on the pieces already in Fulcrum: banding (formation), pitch-control (pressure fields), the 12-dim
tactical signature + state clustering (recurring shapes), and the state sequence (transition tendencies).
Descriptive: it reports what the block IS doing and where it tends to GO, never what it 'should' do.

Orientation follows fulcrum.data.state_at: the attacked goal is at +x, so the defending (pressing) team
protects +x and their block sits toward higher x; a higher press pushes their line to lower x.
"""
from __future__ import annotations
import numpy as np
from .core import team_influence, PITCH_L, PITCH_W
from .recognize import _bands


def team_formation(positions, own_goal="+x"):
    """Formation string for one team from x-banding of its outfield 10 (drops the keeper nearest own goal)."""
    if len(positions) < 8:
        return "?"
    xs = sorted(p[0] for p in positions)
    outfield = xs[:-1] if own_goal == "+x" else xs[1:]        # drop GK (nearest own goal)
    counts = _bands(outfield)
    return "-".join(str(c) for c in counts) if counts else "?"


def pressing_structure(state) -> dict:
    """The defending (pressing) team's structure at this moment. -> dict of descriptive readouts."""
    dfn = np.array(state["dfn"], float)
    dfn_v = np.array(state["dfn_v"], float)
    ball = np.array(state["ball"], float)
    if len(dfn) < 6:
        return {"note": "too few defenders visible"}

    # block height, measured up the pitch from the defending team's own (+x) goal
    from_goal = PITCH_L - dfn[:, 0]
    block_up = float(from_goal.mean()) / PITCH_L                # 0=on own line, 1=on halfway+
    line_up = float(from_goal.max()) / PITCH_L                  # highest pressing line
    press_type = "high press" if block_up > 0.46 else ("low block" if block_up < 0.30 else "mid-block")

    # compactness (lower = tighter)
    vertical = round(float(dfn[:, 0].std()), 2)
    horizontal = round(float(dfn[:, 1].std()), 2)

    # ball pressure + engagement: defenders near the ball and their closing velocity toward it
    to_ball = ball - dfn
    dist = np.linalg.norm(to_ball, axis=1)
    unit = to_ball / (dist[:, None] + 1e-9)
    closing = (dfn_v * unit).sum(1)                             # >0 = stepping toward the ball
    near = dist < 9.0
    ball_pressure = round(float(closing[near].clip(0).sum()), 2) if near.any() else 0.0
    nearest = round(float(dist.min()), 1)
    engaged = int((near & (closing > 0.5)).sum())
    engagement = "engaging" if engaged >= 2 else ("stepping" if engaged == 1 else "holding shape")

    # per-third defensive control (pressure field from pitch control)
    dctrl, gx, _ = team_influence(state["dfn"], state["dfn_v"], state["ball"])
    thirds = {
        "attacking_third": round(float(dctrl[gx <= PITCH_L / 3].mean()), 3),      # where they'd win it high
        "middle_third": round(float(dctrl[(gx > PITCH_L / 3) & (gx <= 2 * PITCH_L / 3)].mean()), 3),
        "defensive_third": round(float(dctrl[gx > 2 * PITCH_L / 3].mean()), 3),
    }
    return {"press_type": press_type, "block_height_up_pitch": round(block_up, 3), "line_height": round(line_up, 3),
            "compactness": {"vertical": vertical, "horizontal": horizontal}, "nearest_defender_to_ball_m": nearest,
            "ball_pressure": ball_pressure, "engagement": engagement, "n_engaging": engaged,
            "control_by_third": thirds}


def opposition_report(state) -> dict:
    """One moment: both teams' formation + the pressing team's structure. `state` from fulcrum.data.state_at."""
    return {"defending_formation": team_formation(state["dfn"], own_goal="+x"),
            "attacking_formation": team_formation(state["att"], own_goal="-x"),
            "pressing": pressing_structure(state)}


def state_transitions(frames, fps, k=6, stride=25, min_players=10):
    """Discover recurring tactical states across a match and their TRANSITION tendencies (a Markov map).
    -> {states, transitions, formation_by_state}. 'what it could change to' = the top transitions from a state."""
    from .recognize import recognize_match, tactical_signature, estimate_formation
    from .data import state_at
    res = recognize_match(frames, fps, stride=stride, k=k, min_players=min_players)
    if "error" in res:
        return res
    labels = res["labels"]
    T = np.zeros((k, k))
    for a, b in zip(labels[:-1], labels[1:]):
        T[a, b] += 1
    row = T.sum(1, keepdims=True)
    P = np.divide(T, row, out=np.zeros_like(T), where=row > 0)   # transition probabilities
    transitions = {}
    for i in range(k):
        nxt = sorted(((int(j), round(float(P[i, j]), 3)) for j in range(k) if j != i and P[i, j] > 0),
                     key=lambda t: -t[1])[:3]
        transitions[i] = nxt                                    # from state i -> likely next states
    # a formation label per state (majority over that state's frames), for human-readable states
    form_by_state = {}
    fids = res["fids"]
    from collections import Counter
    buckets = {}
    for fid, lab in zip(fids, labels):
        st = state_at(frames, fps, fid, min_players=min_players)
        if st:
            buckets.setdefault(lab, []).append(estimate_formation(st["dfn"]))
    for lab, fs in buckets.items():
        form_by_state[lab] = Counter(fs).most_common(1)[0][0]
    return {"states": res["states"], "transitions": transitions, "formation_by_state": form_by_state}


def anticipate(transitions_result, current_label) -> list:
    """Given the transition map and the current state, what it's most likely to change to. -> [(state, prob, formation)]."""
    tr = transitions_result.get("transitions", {})
    fbs = transitions_result.get("formation_by_state", {})
    return [(s, p, fbs.get(s, "?")) for s, p in tr.get(current_label, [])]
