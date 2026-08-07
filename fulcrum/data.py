"""fulcrum.data — load tracking into oriented (attacker/defender) states the engine consumes.

Sources: Metrica (kloppy, 25fps) and SkillCorner open (10fps, LFS media). All physical (metres, m/s), so the
engine is fps-agnostic. `state_at` orients the scene so the attacked goal is at +x and splits possessor's team
(attackers) from the defenders, with velocities from a short seconds window.
"""
from __future__ import annotations
import math
import numpy as np
from .core import PITCH_L, PITCH_W

VEL_WINDOW_S = 0.12


def _is_home(ground) -> float:
    return 1.0 if "home" in str(ground).lower() else 0.0


def load_metrica(match_id: int):
    """-> (frames: {fid: {'ball': (x,y)|None, 'players': {pid:(team01,(x,y))}}}, fps)."""
    from kloppy import metrica
    ds = metrica.load_open_data(match_id=match_id).transform(to_coordinate_system="secondspectrum")
    shift = lambda pt: (pt.x + PITCH_L / 2, pt.y + PITCH_W / 2)
    frames = {}
    for f in ds.frames:
        b = f.ball_coordinates
        ball = None if (b is None or not math.isfinite(b.x)) else shift(b)
        players = {pl.player_id: (_is_home(pl.team.ground), shift(pt))
                   for pl, pt in f.players_coordinates.items()
                   if pt is not None and math.isfinite(pt.x) and math.isfinite(pt.y)}
        frames[f.frame_id] = {"ball": ball, "players": players}
    return frames, 25


def load_skillcorner(match_id: int):
    import urllib.request, json
    raw_base = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}"
    media = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}"
    meta = json.loads(urllib.request.urlopen(f"{raw_base}/{match_id}_match.json", timeout=90).read().decode())
    home = meta.get("home_team")
    home_id = home["id"] if isinstance(home, dict) else home
    team_of = {p.get("player_id", p.get("id")): (1.0 if p.get("team_id") == home_id else 0.0)
               for p in meta.get("players", []) if p.get("team_id") is not None}
    raw = urllib.request.urlopen(f"{media}/{match_id}_tracking_extrapolated.jsonl", timeout=300).read().decode("utf-8", "ignore")
    frames = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("frame") is None or not rec.get("player_data"):
            continue
        bd = rec.get("ball_data") or {}
        ball = None if bd.get("x") is None else (float(bd["x"]) + PITCH_L / 2, float(bd["y"]) + PITCH_W / 2)
        players = {pd["player_id"]: (team_of[pd["player_id"]], (float(pd["x"]) + PITCH_L / 2, float(pd["y"]) + PITCH_W / 2))
                   for pd in rec["player_data"]
                   if pd.get("x") is not None and pd.get("y") is not None and pd.get("player_id") in team_of}
        frames[rec["frame"]] = {"ball": ball, "players": players}
    return frames, 10


def load_match(source: str, match_id: int):
    return {"metrica": load_metrica, "skillcorner": load_skillcorner}[source](match_id)


def state_at(frames, fps, fid, min_players=16):
    """Oriented state for the engine at frame fid, or None if unusable. Attacked goal -> +x.
    `min_players` is the minimum number of tracked players (present in both fid and the velocity
    window frame) required; broadcast sources (GSR) see a partial pitch, so callers lower it."""
    vs = max(1, round(VEL_WINDOW_S * fps))
    dt = 1.0 / fps
    fa, fp = frames.get(fid), frames.get(fid - vs)
    if not fa or not fp or fa["ball"] is None:
        return None
    bx, by = fa["ball"]
    attack_right = bx > PITCH_L / 2
    orient = lambda p: (p[0], p[1]) if attack_right else (PITCH_L - p[0], PITCH_W - p[1])
    cur, prev = fa["players"], fp["players"]
    ids = [p for p in cur if p in prev]
    if len(ids) < min_players:
        return None
    grounds = {}
    for pid in ids:
        grounds.setdefault(cur[pid][0], []).append(orient(cur[pid][1]))
    if len(grounds) != 2:
        return None
    gg = list(grounds)
    ext = lambda pts: max(p[0] for p in pts)                         # after orient, defended goal is at +x
    def_g = max(gg, key=lambda g: ext(grounds[g]))
    att_g = [g for g in gg if g != def_g][0]

    def build(team):
        P, V, ID = [], [], []
        for pid in ids:
            if cur[pid][0] != team:
                continue
            a = np.array(orient(cur[pid][1])); b = np.array(orient(prev[pid][1]))
            P.append((float(a[0]), float(a[1])))
            V.append(tuple(map(float, (a - b) / (vs * dt))))
            ID.append(pid)
        return P, V, ID

    att, att_v, att_ids = build(att_g)
    dfn, dfn_v, dfn_ids = build(def_g)
    ball = orient((bx, by))
    return {"att": att, "att_v": att_v, "dfn": dfn, "dfn_v": dfn_v, "ball": ball, "attack_right": attack_right,
            "att_ids": att_ids, "dfn_ids": dfn_ids,
            # which real side each group is (1.0 home / 0.0 away) — needed to attribute events to a team
            "att_ground": float(att_g), "def_ground": float(def_g)}


def find_chances(frames, fps, max_chances=8, gap_s=5.0):
    """Auto-detect attacking chances: ball deep near a box with both full teams present. Returns anchor fids."""
    gap = int(gap_s * fps)
    fids = sorted(frames)
    out, last = [], -10 ** 9
    for fid in fids:
        fr = frames[fid]
        if fr["ball"] is None or fid - last < gap:
            continue
        bx, _ = fr["ball"]
        if not (bx > 92 or bx < 13):
            continue
        counts = {}
        for _, (t, _) in fr["players"].items():
            counts[t] = counts.get(t, 0) + 1
        if len(counts) == 2 and all(v >= 10 for v in counts.values()):
            out.append(fid); last = fid
        if len(out) >= max_chances:
            break
    return out
