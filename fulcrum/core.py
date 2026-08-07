"""fulcrum.core — the validated topological hole-finder engine.

Openness = velocity-aware PITCH CONTROL (Fernandez-Bornn-style motion model, not a physics engine). Holes =
persistent peaks of the DAS value surface (attacking control x danger x ahead-of-ball) via 0-dim superlevel-set
persistent homology. Ranked by score. This is the core validated by the shot-precursor test (AUC 0.886);
it needs no training, so it generalises to any tracking source.
"""
from __future__ import annotations
import math
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0
RES = 1.6


def team_influence(players, vels, ball, res=RES, quals=None):
    """Summed velocity-aware influence of a set of players. Each player's control is a Gaussian centred 0.5s
    ahead by velocity, elongated along motion, radius growing with distance to the ball. `quals` (per-player,
    ~1.0 default) scales the influence radius -- a better/faster player controls more space (profile fusion)."""
    quals = quals if quals is not None else [1.0] * len(players)
    nx, ny = int(PITCH_L / res), int(PITCH_W / res)
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    total = np.zeros((nx, ny))
    for (px, py), (vx, vy), q in zip(players, vels, quals):
        speed = math.hypot(vx, vy)
        srat = min(speed / 13.0, 1.0)
        dball = math.hypot(px - ball[0], py - ball[1])
        R = float(np.clip(4 + (min(dball, 18) ** 3) / (18 ** 3) * 6, 4, 10)) * float(q)
        mux, muy = px + vx * 0.5, py + vy * 0.5
        ang = math.atan2(vy, vx) if speed > 0.3 else 0.0
        ca, sa = math.cos(-ang), math.sin(-ang)
        dx, dy = gx - mux, gy - muy
        rx, ry = dx * ca - dy * sa, dx * sa + dy * ca
        total += np.exp(-0.5 * ((rx / (R * (1 + srat))) ** 2 + (ry / (R * (1 - 0.5 * srat))) ** 2))
    return total, gx, gy


def pitch_control(att, att_v, dfn, dfn_v, ball, res=RES, att_quals=None, dfn_quals=None):
    """Attacking-team control probability [0,1] over the pitch. Optional per-player quals scale influence (profiles)."""
    ca, gx, gy = team_influence(att, att_v, ball, res, quals=att_quals)
    cd, _, _ = team_influence(dfn, dfn_v, ball, res, quals=dfn_quals)
    return 1.0 / (1.0 + np.exp(-(ca - cd))), gx, gy


def zone_danger_grid(gx, gy):
    """Tactical danger prior. Deep half-space/wide zones (13/15) boosted -- deep crosses are hardest to defend."""
    d = np.full(gx.shape, 0.20)
    lane_c = (gx >= 52.5) & (gy >= 22.7) & (gy <= 45.3)
    lane_w = (gx >= 52.5) & ~((gy >= 22.7) & (gy <= 45.3))
    d[gx >= 52.5] = 0.20
    d[lane_c & (gx >= 70)] = 0.50; d[lane_w & (gx >= 70)] = 0.42
    d[lane_c & (gx >= 83)] = 0.92; d[lane_w & (gx >= 83)] = 0.80
    d[lane_c & (gx >= 88.5)] = 0.97; d[lane_w & (gx >= 88.5)] = 0.88
    wide_deep = (gx >= 92) & ((gy < 22.7) | (gy > 45.3))
    d[wide_deep] = np.maximum(d[wide_deep], 0.90)
    d[gx < 52.5] = 0.05
    return d


def _find(p, x):
    r = x
    while p[r] != r:
        r = p[r]
    while p[x] != r:
        p[x], x = r, p[x]
    return r


def superlevel_persistence(field):
    """0-dim persistent homology of a scalar field (union-find over descending superlevel sets). numpy-only."""
    nx, ny = field.shape
    flat = field.ravel()
    order = np.argsort(-flat)
    p = np.full(nx * ny, -1)
    birth, peak = {}, {}
    active = np.zeros(nx * ny, bool)
    out = []
    for idx in order:
        active[idx] = True; p[idx] = idx
        i, j = divmod(idx, ny)
        roots = set()
        for d in (-ny, ny, -1, 1):
            n = idx + d
            if 0 <= n < nx * ny and abs(divmod(n, ny)[0] - i) + abs(divmod(n, ny)[1] - j) == 1 and active[n]:
                roots.add(_find(p, n))
        if not roots:
            birth[idx] = flat[idx]; peak[idx] = idx
        else:
            roots.add(idx)
            surv = max(roots, key=lambda r: birth.get(r, flat[idx]))
            for r in roots:
                if r == surv or r not in birth:
                    p[r] = surv; continue
                out.append({"peak_ij": divmod(peak[r], ny), "persistence": float(birth[r] - flat[idx])})
                p[r] = surv
            birth.setdefault(surv, flat[idx]); peak.setdefault(surv, idx)
    for r, b in birth.items():
        if _find(p, r) == r:
            out.append({"peak_ij": divmod(peak[r], ny), "persistence": float(b)})
    return out


def describe_hole(h):
    """Short, shareable caption for a hole based on its zone (for socials/clip-generator)."""
    x, y = h["x"], h["y"]
    if x < 70:
        where = "midfield space between the lines"
    elif x >= 92 and (y < 22.7 or y > 45.3):
        where = "the deep " + ("right" if y < 34 else "left") + " channel — cut-back / low-cross territory"
    elif y < 22.7 or y > 45.3:
        where = "the " + ("right" if y < 34 else "left") + " half-space at the box edge"
    else:
        where = "central space at the edge of the box — zone 14"
    return f"Exploitable pocket in {where} (danger {h['score']})."


def find_holes(att, att_v, dfn, dfn_v, ball, min_persistence=0.06, top=3):
    """Ranked exploitable holes. -> (value_surface, gx, gy, holes[{x,y,score,persistence}])."""
    pc, gx, gy = pitch_control(att, att_v, dfn, dfn_v, ball)
    danger = zone_danger_grid(gx, gy)
    ahead = np.where(gx > ball[0] - 5, 1.0, 0.25)
    value = pc * danger * ahead
    holes = []
    for h in superlevel_persistence(value):
        if h["persistence"] < min_persistence:
            continue
        i, j = h["peak_ij"]
        x, y = float(gx[i, j]), float(gy[i, j])
        if x < 3 or x > PITCH_L - 3 or y < 3 or y > PITCH_W - 3:
            continue
        holes.append({"x": round(x, 1), "y": round(y, 1), "score": round(float(value[i, j]), 3),
                      "persistence": round(h["persistence"], 3)})
    holes.sort(key=lambda z: -z["score"])
    return value, gx, gy, holes[:top]


def defender_attribution(att, att_v, dfn, dfn_v, ball, hole, step=2.0, min_persistence=0.05):
    """Which defender's positioning owns this hole? -> list per defender, sorted by responsibility.

    Two distinct questions, answered separately because they are not the same thing:
      `containment`  — remove this defender and the surface re-solves; how much MORE dangerous does the hole
                       get? This is what the player is currently holding shut.
      `responsibility` — move this defender `step` metres toward the hole; how much danger does that remove?
                       This is who is best placed to fix it, which is the coachable question.
    A defender can score high on containment and low on responsibility (already doing their job, too far to
    help elsewhere), so both are reported rather than collapsed into one number.

    Only possible because the surface is COMPUTED, not fitted: a learned model asked about a defender who
    isn't there is extrapolating off-distribution, whereas this re-solves the geometry exactly.
    """
    import numpy as _np
    D = _np.array(dfn, float); DV = _np.array(dfn_v, float)
    target = _np.array([hole["x"], hole["y"]], float)

    def _top(dd, vv):
        _, _, _, hs = find_holes(att, att_v, dd, vv, ball, min_persistence=min_persistence, top=6)
        if not hs:
            return 0.0
        near = min(hs, key=lambda h: (h["x"] - target[0]) ** 2 + (h["y"] - target[1]) ** 2)
        return float(near["score"])

    base = _top(D, DV)
    out = []
    for i in range(len(D)):
        keep = _np.delete(_np.arange(len(D)), i)
        removed = _top(D[keep], DV[keep]) if len(keep) >= 3 else base
        moved = D.copy()
        v = target - moved[i]
        n = float(_np.linalg.norm(v))
        if n > 1e-6:
            moved[i] = moved[i] + v / n * min(step, n)
        shifted = _top(moved, DV)
        out.append({"defender": i,
                    "x": round(float(D[i][0]), 1), "y": round(float(D[i][1]), 1),
                    "distance_to_hole_m": round(n, 1),
                    "containment": round(removed - base, 4),
                    "responsibility": round(base - shifted, 4)})
    out.sort(key=lambda z: -z["responsibility"])
    return {"base_danger": round(base, 4), "step_m": step, "defenders": out}
