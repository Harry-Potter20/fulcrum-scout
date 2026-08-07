"""fulcrum.ingest — the recognition-free video ingestion adapter (the drop-in-video front-end).

The agnosticism law applied to INGESTION: identity is never an input. This turns pure SPATIAL detection + object
tracking (YOLO + ByteTrack: person/ball boxes + track ids — NO re-identification, NO jersey OCR, NO role classifier,
NO tracklab) into the engine's canonical `frames` dict, so the SAME services run on any football video, not only
annotated broadcast. Teams are split by shirt COLOUR (appearance, not identity); image->pitch is a dependency-free
homography (numpy DLT). The heavy stage (YOLO+ByteTrack) lives in the HF job `lean_ingest.py`; THIS module is the
pure-CPU adapter — detections + per-frame homography -> canonical — plus the geometry/colour helpers it uses.

VALIDATED on SoccerNet-GS SNGS-021 vs ground truth (calibration held fixed via a homography solved from
foot-correspondences; probe median 0.17 m):
  · player positions   median 0.17 m, p90 0.46 m      [validated]
  · player recall      0.97 within 4 m                 [validated]
  · teams by colour    0.965 (grass-masked kmeans)     [validated]
  · ball recall        0.47 (imgsz=1280) — WEAK spot   [partial]  (small/blurred target; wants a dedicated detector)
  · reach property     services RUN on the state        [validated]  (666/750 frames)
FIDELITY IS SIGNAL-DEPENDENT — the honest finding: coarse structure transfers (att/def counts corr 0.6–0.8) but
sharp role-sensitive extrema do NOT (danger ~0–0.2, containment ~0), because they amplify residual errors (a referee
colour-assigned to a team → ~+1 defender; a 0.5 m slip on the last defender; a team flip near the box). So drop
off-pitch detections AND the small officials colour-cluster before trusting role-sensitive signals like danger.
"""
from __future__ import annotations
import numpy as np
from .core import PITCH_L, PITCH_W


# ---- image -> pitch homography (Hartley-normalized DLT; dependency-free, no OpenCV) ----------------------------
def _normalize(pts):
    c = pts.mean(0); d = np.sqrt(((pts - c) ** 2).sum(1)).mean() + 1e-12; s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return (np.c_[pts, np.ones(len(pts))] @ T.T)[:, :2], T


def solve_homography(src, dst):
    """3x3 H mapping src(image px) -> dst(pitch metres) from >=4 point correspondences. In a true drop-in these
    correspondences come from a pitch-line calibration model; here they come from foot-points."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    sn, Ts = _normalize(src); dn, Td = _normalize(dst); A = []
    for (x, y), (X, Y) in zip(sn, dn):
        A += [[-x, -y, -1, 0, 0, 0, x * X, y * X, X], [0, 0, 0, -x, -y, -1, x * Y, y * Y, Y]]
    _, _, Vt = np.linalg.svd(np.asarray(A)); H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def apply_H(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


# ---- teams by shirt COLOUR (appearance, identity-free) --------------------------------------------------------
def kit_feat(img, ltwh):
    """Kit-colour cue: mean RGB of the torso with GRASS masked out (green bleed washes small broadcast crops to
    chance). Brightness is preserved so white-vs-dark kits separate. `img` is an HxWx3 RGB array; `ltwh` a box."""
    l, t, w, h = [int(v) for v in ltwh]
    y0, y1 = max(0, t + int(0.15 * h)), max(1, t + int(0.55 * h))
    x0, x1 = max(0, l + int(0.25 * w)), max(1, l + int(0.75 * w))
    crop = np.asarray(img)[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    if len(crop) == 0:
        return None
    R, G, B = crop[:, 0], crop[:, 1], crop[:, 2]
    grass = (G > R * 1.05) & (G > B * 1.05)
    kit = crop[~grass] if int((~grass).sum()) >= 8 else crop
    return kit.mean(0)


def kmeans2_std(X, iters=40, seed=0):
    """2-means on z-scored features (keeps hue AND brightness contrast, unlike unit-normalized kmeans)."""
    X = (np.asarray(X, float) - np.mean(X, 0)) / (np.std(X, 0) + 1e-9)
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), 2, replace=False)].copy(); a = np.zeros(len(X), int)
    for _ in range(iters):
        a = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in (0, 1):
            if (a == k).any():
                c[k] = X[a == k].mean(0)
    return a


def split_teams(track_colours, drop_officials=False):
    """{track_id: [kit_feat,...]} -> {track_id: team01}. drop_officials (recognition-free): 3-way cluster and drop
    the SMALLEST cluster (referees/keepers wear distinct colours and are few) before the 2-team split — this removes
    the ~+1 defender bias that pollutes role-sensitive signals. NOTE: dropped ids simply get no team (excluded)."""
    tids = [t for t, c in track_colours.items() if len(c) >= 1]
    feat = np.stack([np.mean(track_colours[t], 0) for t in tids])
    if drop_officials and len(tids) >= 6:
        lab3 = _kmeans3_std(feat)
        sizes = {k: int((lab3 == k).sum()) for k in (0, 1, 2)}
        small = min(sizes, key=sizes.get)
        keep = [i for i in range(len(tids)) if lab3[i] != small]
        tids = [tids[i] for i in keep]; feat = feat[keep]
    lab = kmeans2_std(feat)
    return {t: float(lab[i]) for i, t in enumerate(tids)}


def _kmeans3_std(X, iters=40, seed=0):
    X = (np.asarray(X, float) - np.mean(X, 0)) / (np.std(X, 0) + 1e-9)
    rng = np.random.default_rng(seed); c = X[rng.choice(len(X), 3, replace=False)].copy(); a = np.zeros(len(X), int)
    for _ in range(iters):
        a = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in (0, 1, 2):
            if (a == k).any():
                c[k] = X[a == k].mean(0)
    return a


# ---- assemble the canonical `frames` dict (identity-free) ------------------------------------------------------
def _on_pitch(xy, margin=1.0):
    return (-margin <= xy[0] <= PITCH_L + margin) and (-margin <= xy[1] <= PITCH_W + margin)


def flag_officials(frames, ball_R=8.0, isol_m=12.0, min_frames=5):
    """GEOMETRY-FIRST official detection — NO identity, NO re-ID. Referees violate football geometry: they keep
    low ball involvement and never join a team's spatial block. Per track, over its frames, score two geometric
    cues: (a) ball-involvement = fraction of frames within `ball_R` m of the ball; (b) isolation = median distance
    to the nearest SAME-team player. An official is almost never near the ball AND spatially isolated from its
    (mis-)assigned team. Returns the set of track_ids to drop before trusting role-sensitive Tier-3 signals.
    This is the principled response to Tier-3 sensitivity — solve the specific geometric error, don't add recognition.

    FIRST CUT — HONEST LIMIT: these two cues OVER-FLAG. On SNGS-021 it removes ~2 defenders/frame (9.3->7.0) when it
    should remove ~1 referee, because a deep isolated centre-back shares both cues (low ball-involvement + far from
    teammates). Separating referee from deep defender needs the fuller geometric feature set the tier note lists
    (formation-membership test, per-track movement statistics, interaction patterns) — not shipped yet. Use with care;
    the robustness 'officials' column quantifies how much an unfiltered official actually costs each service."""
    stats = {}
    for fr in frames.values():
        ball = fr.get("ball"); pls = fr["players"]
        pos = {t: np.asarray(xy, float) for t, (_, xy) in pls.items()}
        team = {t: tm for t, (tm, _) in pls.items()}
        for t, p in pos.items():
            s = stats.setdefault(t, {"nball": 0, "n": 0, "isol": []})
            s["n"] += 1
            if ball is not None and np.hypot(p[0] - ball[0], p[1] - ball[1]) <= ball_R:
                s["nball"] += 1
            same = [q for u, q in pos.items() if u != t and team[u] == team[t]]
            if same:
                s["isol"].append(min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in same))
    officials = set()
    for t, s in stats.items():
        if s["n"] < min_frames:
            continue
        ball_inv = s["nball"] / s["n"]
        isol = float(np.median(s["isol"])) if s["isol"] else 0.0
        if ball_inv < 0.02 and isol > isol_m:
            officials.add(t)
    return officials


def freeze_frame_to_state(freeze_frame, ball_location, sb_dims=(120.0, 80.0)):
    """StatsBomb-360 FREEZE-FRAME -> canonical oriented `state` — the event-data ingestion adapter (freeze-frame tier).
    A 360 freeze-frame is a sparse point cloud of camera-visible teammates/opponents at an event; this maps it onto
    the SAME canonical state every other source produces, so the SAME off-ball metrics (`space_creation`, `find_holes`,
    `state_summary`) and services run unchanged on event data — no re-implementation per source. StatsBomb normalises
    the possessing team to attack +x, so the state is already oriented (attack_right=True); velocities are 0 (a single
    instant). Returns the oriented state dict, or None if too few visible players. Identity never enters — the actor's
    name is carried by the EVENT, not the state.

    freeze_frame: list of {"location":[x,y], "teammate":bool, ...} (StatsBomb 360). ball_location: the event [x,y].
    NOTE: 360 captures visible players only, so metrics from it are a proxy vs visible opponents — tag the coverage."""
    sx, sy = sb_dims
    def cv(p): return (float(p[0]) * PITCH_L / sx, float(p[1]) * PITCH_W / sy)
    att = [cv(p["location"]) for p in freeze_frame if p.get("teammate")]
    dfn = [cv(p["location"]) for p in freeze_frame if not p.get("teammate")]
    if len(att) < 4 or len(dfn) < 4:
        return None
    return {"att": att, "att_v": [(0.0, 0.0)] * len(att), "dfn": dfn, "dfn_v": [(0.0, 0.0)] * len(dfn),
            "ball": cv(ball_location), "attack_right": True}


def official_by_formation(team_positions, min_fit=0.95, margin=0.008):
    """RULE-BASED official detection via FORMATION MEMBERSHIP — the discriminator the two geometric cues lack.
    Reuses the rule-based tactical structure (`recognize.formation_fit`): the referee is the node whose REMOVAL
    restores a canonical formation, while a deep defender's removal breaks a line. `team_positions` are one team's
    (x,y) oriented so they defend +x. Returns the index of the flagged official, or None.

    Validated synthetically at **97%** (inject a referee into a real 4-4-2; leave-one-out picks it, not a defender).
    Needs a NEAR-FULL team (>=11 nodes -> 10 outfield after the GK drop) — so it works on full-pitch optical tracking
    but DEGRADES on partial broadcast crops (only ~7-8 players/side visible); there, fall back to shirt-colour
    (`split_teams(drop_officials=True)`) + motion cues. This is why the official-filter method is itself
    ingestion-quality-tiered, mirroring the service tiers."""
    from .recognize import formation_fit
    T = list(team_positions)
    if len(T) < 11:
        return None
    fits = [formation_fit([T[j] for j in range(len(T)) if j != i])[1] for i in range(len(T))]
    i = int(np.argmax(fits)); second = sorted(fits, reverse=True)[1]
    return i if (fits[i] >= min_fit and fits[i] - second >= margin) else None


def assemble_frames(det, H_of, tid_team, on_pitch=True):
    """Detections + per-frame homography + team labels -> the engine's `frames` dict (== load_gamestate's schema),
    so the SAME `state_at` -> services run unchanged. Coordinates are the engine frame [0,PITCH_L]x[0,PITCH_W].

    det:      {fid: {'players': [(track_id, foot_uv, ltwh), ...], 'ball': foot_uv|None}}   (image pixels)
    H_of:     {fid: 3x3 homography}   (image px -> pitch metres, midfield-centred)
    tid_team: {track_id: team01}      (from split_teams; ids not present are dropped)
    on_pitch: drop detections that fall outside the touchlines (removes sideline/coach false positives)

    -> {fid: {'ball': (x,y)|None, 'players': {track_id: (team01, (x,y))}}}
    """
    frames = {}
    for fid, dd in det.items():
        H = H_of.get(fid)
        if H is None:
            continue
        st = {"ball": None, "players": {}}
        if dd.get("ball") is not None:
            xy = apply_H(H, [dd["ball"]])[0]; xy = (xy[0] + PITCH_L / 2, xy[1] + PITCH_W / 2)
            if not on_pitch or _on_pitch(xy):
                st["ball"] = (float(xy[0]), float(xy[1]))
        for tid, foot, _ in dd.get("players", []):
            if tid not in tid_team:
                continue
            xy = apply_H(H, [foot])[0]; xy = (xy[0] + PITCH_L / 2, xy[1] + PITCH_W / 2)
            if on_pitch and not _on_pitch(xy):
                continue
            st["players"][tid] = (tid_team[tid], (float(xy[0]), float(xy[1])))
        frames[fid] = st
    return frames
