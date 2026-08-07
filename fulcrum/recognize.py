"""fulcrum.recognize — topological tactical RECOGNITION & understanding (not just exploitable space).

Turns each frame's tactical structure into a compact SIGNATURE: a topological part (persistence statistics of the
pitch-control surface -- a translation/robust fingerprint of the shape of space) plus interpretable shape metrics
(defensive block height/width/compactness, line count / formation, control balance). Signatures let us:
  - estimate the defensive FORMATION,
  - DISCOVER recurring tactical states by clustering (unsupervised),
  - RETRIEVE tactically-similar moments ("find frames like this").
Pure numpy; no player identity or stats required (that arrives with the YOLO/GSR + stats frontend).
"""
from __future__ import annotations
import numpy as np
from .core import pitch_control, zone_danger_grid, superlevel_persistence, PITCH_L, PITCH_W


def _persistence_stats(value):
    """Topological fingerprint of a scalar surface: [n_persistent, total, max, mean, entropy] of the 0-dim diagram."""
    ps = np.array([h["persistence"] for h in superlevel_persistence(value)], float)
    ps = ps[ps > 1e-3]
    if len(ps) == 0:
        return np.zeros(5, float)
    p = ps / ps.sum()
    entropy = float(-(p * np.log(p + 1e-9)).sum())
    return np.array([float((ps > 0.05).sum()), float(ps.sum()), float(ps.max()), float(ps.mean()), entropy])


def _bands(xs, gap=8.0):
    """1-D banding of coordinates -> counts per band (deepest first). A formation proxy for a set of players."""
    xs = np.sort(np.asarray(xs))[::-1]                          # deepest defensive line (high x) first
    if len(xs) == 0:
        return []
    counts, cur = [], 1
    for a, b in zip(xs[:-1], xs[1:]):
        if a - b > gap:
            counts.append(cur); cur = 1
        else:
            cur += 1
    counts.append(cur)
    return counts


# canonical shapes, defence-first. Free banding produced face-invalid labels (2-8, 10, 9-1) whenever a team
# was stretched mid-transition; matching against real formations with an explicit fit score fixes both the
# labels and the honesty (a shape that fits nothing is REPORTED as transitional, not force-named).
FORMATION_TEMPLATES = {
    "4-4-2": (4, 4, 2), "4-3-3": (4, 3, 3), "4-5-1": (4, 5, 1), "3-5-2": (3, 5, 2),
    "3-4-3": (3, 4, 3), "5-3-2": (5, 3, 2), "5-4-1": (5, 4, 1), "5-2-3": (5, 2, 3),
    "4-2-3-1": (4, 2, 3, 1), "4-1-4-1": (4, 1, 4, 1), "4-4-1-1": (4, 4, 1, 1), "3-4-2-1": (3, 4, 2, 1),
}


def formation_fit(dfn):
    """Best canonical formation for the defending team. -> (label, fit in [0,1], per-template scores).

    Players are depth-ordered (they defend +x, so the deepest band is nearest +x, GK dropped) and partitioned
    by each template's band sizes; fit = 1 - within-band SS / total SS of depth, i.e. how much of the depth
    structure the banding explains. 4-band templates get a small complexity penalty so they cannot win by
    slicing thinner. Templates need all 10 outfield players; with fewer we fall back to free banding."""
    if len(dfn) < 8:
        return "?", 0.0, {}
    xs = sorted((p[0] for p in dfn), reverse=True)              # deepest (own-goal side, +x) first
    outfield = xs[1:]                                           # drop GK (deepest)
    if len(outfield) != 10:
        counts = _bands(sorted(outfield))
        return ("-".join(str(c) for c in counts) if counts else "?"), 0.0, {}
    arr = np.array(outfield, float)
    total = float(((arr - arr.mean()) ** 2).sum()) + 1e-9
    scores = {}
    for name, bands in FORMATION_TEMPLATES.items():
        within, i = 0.0, 0
        for b in bands:
            seg = arr[i:i + b]; i += b
            within += float(((seg - seg.mean()) ** 2).sum())
        fit = 1.0 - within / total
        fit -= 0.02 * (len(bands) - 3)                          # complexity penalty for 4-band shapes
        scores[name] = round(fit, 4)
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def estimate_formation(dfn, min_fit=0.90):
    """Formation string for the defending team; canonical when it fits, honest when it does not.
    `min_fit` [CALIBRATED]: on the 28 Metrica sample states fits ran 0.867-0.979, and the two lowest
    (0.867, 0.884) were exactly the states whose free-banded labels were most deranged (2-3-5, 2-8) — i.e.
    genuinely stretched, mid-transition shapes. 0.90 flags those as ~approximate and passes the rest."""
    label, fit, _ = formation_fit(dfn)
    if fit == 0.0 and label not in ("?",):
        return label                                            # free-banding fallback (partial tracking)
    if label == "?":
        return "?"
    return label if fit >= min_fit else f"~{label}"             # ~ = in flux / transitional, nearest shape


def tactical_signature(state):
    """Fixed-length signature: [5 topological] + [7 interpretable shape metrics]. -> np.ndarray(12)."""
    att, att_v, dfn, dfn_v, ball = state["att"], state["att_v"], state["dfn"], state["dfn_v"], state["ball"]
    pc, gx, gy = pitch_control(att, att_v, dfn, dfn_v, ball)
    value = pc * zone_danger_grid(gx, gy)
    topo = _persistence_stats(value)
    D = np.array(dfn)
    if len(D) == 0:
        shape = np.zeros(7)
    else:
        block_height = D[:, 0].mean() / PITCH_L                 # how high the block sits (0..1)
        line_height = D[:, 0].min() / PITCH_L                   # highest line (most advanced press)
        width = (D[:, 1].max() - D[:, 1].min()) / PITCH_W
        compact = float(np.linalg.norm(D - D.mean(0), axis=1).mean()) / 20.0
        n_bands = len(_bands(sorted(p[0] for p in dfn)[:-1]))
        att_control = float(pc.mean())                          # territorial dominance of the attack
        ball_x = ball[0] / PITCH_L
        shape = np.array([block_height, line_height, width, compact, n_bands / 5.0, att_control, ball_x])
    return np.concatenate([topo, shape]).astype(float)


def _kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def recognize_match(frames, fps, stride=25, k=5, sample_cap=800, min_players=16):
    """Compute signatures across a match, discover k recurring tactical states (unsupervised), estimate formation
    distribution, and return a signature index for retrieval. `min_players` is lowered for partial-pitch
    broadcast (GSR) sources. -> dict."""
    from .data import state_at
    sigs, fids, forms = [], [], []
    for fid in sorted(frames)[::stride]:
        st = state_at(frames, fps, fid, min_players=min_players)
        if st is None:
            continue
        sigs.append(tactical_signature(st)); fids.append(fid); forms.append(estimate_formation(st["dfn"]))
        if len(sigs) >= sample_cap:
            break
    if len(sigs) < k:
        return {"error": "too few analysable frames"}
    X = np.array(sigs)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)                    # standardise before clustering
    lab, _ = _kmeans(Xn, k)
    states = [{"state": int(j), "n": int((lab == j).sum()),
               "share": round(float((lab == j).mean()), 3)} for j in range(k)]
    from collections import Counter
    formation_dist = Counter(forms).most_common(6)
    return {"n_frames": len(sigs), "states": sorted(states, key=lambda s: -s["n"]),
            "formation_dist": formation_dist, "sig_dim": X.shape[1],
            "labels": lab.tolist(), "fids": fids,            # per-sample state label in time order (for transitions)
            "index": {"fids": fids, "sig": Xn.tolist()}}


def similar_moments(index, query_i, top=5):
    """Retrieve the most tactically-similar frames to index frame query_i (nearest in signature space)."""
    S = np.array(index["sig"]); fids = index["fids"]
    d = ((S - S[query_i]) ** 2).sum(1)
    order = np.argsort(d)
    return [{"fid": fids[j], "dist": round(float(d[j]), 3)} for j in order[1:top + 1]]
