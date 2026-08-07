"""fulcrum.profile — player PROFILE fusion (the statistical multimodal, mechanism layer).

A hole is more dangerous if a *threatening* attacker can reach it, and a defender covers more space if they're
*better/faster*. This module turns per-player attributes into weights the engine consumes:
  - defender `qual` -> scales their pitch-control influence radius (a stronger/faster defender covers more),
  - attacker `threat` -> boosts the danger of nearby holes.

Profiles come from whatever identity source is available. `role_profile` gives a coarse, no-join profile from a
positional role (works on SkillCorner's built-in identity today); an external stats join (FBref/StatsBomb by name)
is the richer upgrade once the identity frontend (GSR / jersey-number OCR) is wired in.
"""
from __future__ import annotations

# role (lowercased substring) -> (attacking threat, defensive/pace quality). ~1.0 = average.
_ROLE_TABLE = [
    ("striker", 1.5, 0.9), ("forward", 1.4, 0.9), ("winger", 1.45, 1.0), ("wide attacker", 1.45, 1.0),
    ("attacking mid", 1.3, 0.95), ("second striker", 1.35, 0.9),
    ("midfield", 1.05, 1.05), ("wing back", 1.1, 1.1), ("full back", 0.85, 1.1), ("wide defender", 0.85, 1.1),
    ("centre back", 0.6, 1.15), ("center back", 0.6, 1.15), ("defender", 0.7, 1.1),
    ("goalkeeper", 0.2, 1.0), ("keeper", 0.2, 1.0),
]


def role_profile(role: str):
    """-> {'threat': float, 'qual': float} from a positional role string. Unknown -> average (1.0, 1.0)."""
    r = (role or "").lower()
    for key, threat, qual in _ROLE_TABLE:
        if key in r:
            return {"threat": threat, "qual": qual}
    return {"threat": 1.0, "qual": 1.0}


def profiled_holes(state, id_to_role, top=3, min_persistence=0.06):
    """Profile-weighted holes: defenders scale pitch control by quality; each hole's danger is boosted by the
    threat of its nearest attacker. Requires a state carrying 'att_ids'/'dfn_ids'. -> ranked holes."""
    import numpy as np
    from .core import pitch_control, zone_danger_grid, superlevel_persistence, PITCH_L, PITCH_W
    dfn_q = [role_profile(id_to_role.get(i, "")).get("qual", 1.0) for i in state["dfn_ids"]]
    att_threat = [role_profile(id_to_role.get(i, "")).get("threat", 1.0) for i in state["att_ids"]]
    pc, gx, gy = pitch_control(state["att"], state["att_v"], state["dfn"], state["dfn_v"], state["ball"], dfn_quals=dfn_q)
    ball = state["ball"]
    value = pc * zone_danger_grid(gx, gy) * np.where(gx > ball[0] - 5, 1.0, 0.25)
    att = np.array(state["att"]) if state["att"] else None
    holes = []
    for h in superlevel_persistence(value):
        if h["persistence"] < min_persistence:
            continue
        i, j = h["peak_ij"]
        x, y = float(gx[i, j]), float(gy[i, j])
        if x < 3 or x > PITCH_L - 3 or y < 3 or y > PITCH_W - 3:
            continue
        boost = att_threat[int(np.argmin(((att - [x, y]) ** 2).sum(1)))] if att is not None else 1.0
        holes.append({"x": round(x, 1), "y": round(y, 1), "score": round(float(value[i, j]) * boost, 3),
                      "nearest_att_threat": round(float(boost), 2)})
    holes.sort(key=lambda z: -z["score"])
    return holes[:top]


def roster_profile(jersey, team: str, roster: dict) -> dict:
    """Look up a (team, jersey) identity in a roster and derive engine weights from real stats.

    roster maps "(team, jersey)" -> a stats dict. We accept either explicit {'threat','qual'} weights or raw
    per-90 stats and convert them: attacking threat scales with goal+assist involvement, defensive/pace qual
    with duels/recoveries. Falls back to the role profile when the identity isn't in the roster. This is the
    mechanism that turns the GSR jersey number into a per-player weight; it is value-gated on a *named* match
    (SoccerNet clips are anonymised), but drops in unchanged once a roster is joined by name.
    """
    _STAT_KEYS = ("goals90", "assists90", "xg90", "duels_won90", "recoveries90")
    rec = roster.get((team, str(jersey))) or roster.get(f"{team}:{jersey}")
    if not rec:
        return role_profile("")
    if "threat" in rec or "qual" in rec:                     # explicit weights (role as base for the missing one)
        base = role_profile(rec.get("role", ""))
        return {"threat": float(rec.get("threat", base["threat"])), "qual": float(rec.get("qual", base["qual"]))}
    if not any(k in rec for k in _STAT_KEYS):                # no stats -> use the positional role
        return role_profile(rec.get("role", ""))
    # derive from raw per-90 stats (all optional; sensible midpoints when absent)
    ga90 = float(rec.get("goals90", 0.0)) + float(rec.get("assists90", 0.0)) + 0.5 * float(rec.get("xg90", 0.0))
    threat = 0.7 + min(ga90, 1.2) * 0.9                       # ~0.7 (low) .. ~1.8 (elite attacker)
    duels = float(rec.get("duels_won90", 0.0)) + float(rec.get("recoveries90", 0.0)) / 5.0
    qual = 0.9 + min(duels, 6.0) / 6.0 * 0.4                  # ~0.9 .. ~1.3
    return {"threat": round(threat, 2), "qual": round(qual, 2)}


def rostered_holes(state, id_to_jerseyteam: dict, roster: dict, top=3, min_persistence=0.06):
    """Profile-weighted holes using a jersey->stats roster (finer than positional role). id_to_jerseyteam maps
    a track/player id -> (team, jersey). Mirrors `profiled_holes` but pulls weights from `roster_profile`."""
    import numpy as np
    from .core import pitch_control, zone_danger_grid, superlevel_persistence, PITCH_L, PITCH_W

    def w_of(pid):
        jt = id_to_jerseyteam.get(pid)
        return roster_profile(jt[1], jt[0], roster) if jt else {"threat": 1.0, "qual": 1.0}

    dfn_q = [w_of(i)["qual"] for i in state["dfn_ids"]]
    att_threat = [w_of(i)["threat"] for i in state["att_ids"]]
    pc, gx, gy = pitch_control(state["att"], state["att_v"], state["dfn"], state["dfn_v"], state["ball"], dfn_quals=dfn_q)
    ball = state["ball"]
    value = pc * zone_danger_grid(gx, gy) * np.where(gx > ball[0] - 5, 1.0, 0.25)
    att = np.array(state["att"]) if state["att"] else None
    holes = []
    for h in superlevel_persistence(value):
        if h["persistence"] < min_persistence:
            continue
        i, j = h["peak_ij"]
        x, y = float(gx[i, j]), float(gy[i, j])
        if x < 3 or x > PITCH_L - 3 or y < 3 or y > PITCH_W - 3:
            continue
        boost = att_threat[int(np.argmin(((att - [x, y]) ** 2).sum(1)))] if att is not None else 1.0
        holes.append({"x": round(x, 1), "y": round(y, 1), "score": round(float(value[i, j]) * boost, 3),
                      "nearest_att_threat": round(float(boost), 2)})
    holes.sort(key=lambda z: -z["score"])
    return holes[:top]


def skillcorner_roles(match_id: int) -> dict:
    """player_id -> role string, from SkillCorner match metadata (the built-in identity, no video needed)."""
    import urllib.request, json
    base = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}"
    meta = json.loads(urllib.request.urlopen(f"{base}/{match_id}_match.json", timeout=90).read().decode())
    out = {}
    for p in meta.get("players", []):
        pid = p.get("player_id", p.get("id"))
        role = (p.get("player_role") or {})
        name = role.get("name") or role.get("position_group") or role.get("acronym") or ""
        if pid is not None:
            out[pid] = name
    return out
