"""fulcrum.gsr — adapter from SoccerNet Game-State Reconstruction output to the engine's state.

The GSR pipeline (or its ground-truth `Labels-GameState.json`) emits, per frame, every athlete's
pitch position (metres, midfield-centred), team, role and jersey number — the same information the
engine consumes. This module maps that schema onto fulcrum's frame dict so the *identical* engine
(`state_at` -> `find_holes`/`profiled_holes`) runs on real broadcast tracking, no retraining.

GSR pitch coords are centred at midfield (x in [-52.5, 52.5], y in [-34, 34]); the engine works in
[0, PITCH_L] x [0, PITCH_W], so we shift by (+PITCH_L/2, +PITCH_W/2). A player's ground position is
the midpoint of their box's two bottom-pitch corners (their feet). Ships GT-first so the adapter is
validated on clean labels before it is ever fed noisy detector output.
"""
from __future__ import annotations
import os
import json
from .core import PITCH_L, PITCH_W

GSR_FPS = 25  # SoccerNet-GS sequences are 25 fps

# GSR pitch category ids
_CAT_PLAYER, _CAT_GK, _CAT_REFEREE, _CAT_BALL = 1, 2, 3, 4
_OUTFIELD = {_CAT_PLAYER, _CAT_GK}
_TEAM01 = {"left": 0.0, "right": 1.0}


def _foot_xy(bbox_pitch):
    """Player ground position (metres, engine frame) = midpoint of the box's bottom-pitch corners."""
    x = (bbox_pitch["x_bottom_left"] + bbox_pitch["x_bottom_right"]) / 2.0 + PITCH_L / 2.0
    y = (bbox_pitch["y_bottom_left"] + bbox_pitch["y_bottom_right"]) / 2.0 + PITCH_W / 2.0
    return (float(x), float(y))


def load_gamestate(labels_path: str):
    """Parse a SoccerNet-GS Labels-GameState.json.

    -> (frames, roles) where
       frames: {fid: {'ball': (x,y)|None, 'players': {track_id: (team01, (x,y))}}}  (engine schema)
       roles:  {track_id: {'role': str, 'jersey': str|None, 'team': str}}           (for profiles/identity)
    fid is the integer frame number parsed from the image file name (000123.jpg -> 123).
    Only outfield players + goalkeepers with a team and pitch box become 'players'; referees are dropped.
    """
    d = json.load(open(labels_path))
    fid_of = {img["image_id"]: int(img["file_name"].split(".")[0]) for img in d["images"]}
    frames = {fid: {"ball": None, "players": {}} for fid in fid_of.values()}
    roles: dict = {}
    for a in d["annotations"]:
        fid = fid_of.get(a["image_id"])
        if fid is None:
            continue
        cat = a["category_id"]
        bp = a.get("bbox_pitch")
        if cat == _CAT_BALL:
            if bp is not None:
                frames[fid]["ball"] = _foot_xy(bp)
            continue
        if cat not in _OUTFIELD or bp is None:
            continue
        attr = a.get("attributes", {}) or {}
        team = attr.get("team")
        if team not in _TEAM01:
            continue
        tid = a["track_id"]
        frames[fid]["players"][tid] = (_TEAM01[team], _foot_xy(bp))
        roles.setdefault(tid, {"role": attr.get("role", "player"),
                               "jersey": attr.get("jersey"), "team": team})
    return frames, roles


def id_to_role(roles: dict) -> dict:
    """track_id -> role string for `profiled_holes` (goalkeepers get the keeper profile; players average
    until a jersey->roster join supplies a finer position)."""
    return {tid: r.get("role", "") for tid, r in roles.items()}


def _kmeans2(X, iters=30, seed=0):
    """Tiny deterministic 2-means (numpy). -> cluster label (0/1) per row. Used to split teams from reid
    embeddings when no team labels exist (YOLO26 -> tracklab path)."""
    import numpy as np
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    c = Xn[rng.choice(len(Xn), 2, replace=False)].copy()
    a = np.zeros(len(Xn), int)
    for _ in range(iters):
        a = ((Xn[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in (0, 1):
            if (a == k).any():
                c[k] = Xn[a == k].mean(0)
    return a


def _torso_color(img, ltwh):
    """Mean RGB of a player's torso (upper-middle of the box) — the kit-colour cue for team clustering."""
    import numpy as np
    l, t, w, h = [int(v) for v in ltwh]
    y0, y1 = max(0, t + int(0.15 * h)), max(1, t + int(0.5 * h))
    x0, x1 = max(0, l + int(0.2 * w)), max(1, l + int(0.8 * w))
    crop = np.asarray(img)[y0:y1, x0:x1].reshape(-1, 3)
    return crop.mean(0) if len(crop) else np.zeros(3)


def load_gsr_trackerstate(pklz_path: str, ball_frames=None, frames_dir=None):
    """Adapter from a tracklab TrackerState (.pklz) — YOLO26 detections tracked + calibrated by the GS
    pipeline — to the engine's frame dict. Every player keeps its **reid embedding** (a persistent identity
    signature, so individual ability to exploit a hole can be attributed and tracked across frames) and its
    **track_id**. Teams are split by jersey COLOUR when `frames_dir` is given (reid embeddings identify
    individuals, not kits, so they mis-cluster teams); otherwise reid-embedding 2-means is the fallback.
    `ball_frames` supplies the ball per fid (the external person-detector carries none).

    -> (frames, ident) where ident[track_id] = {role, jersey, team, embedding, track_id}. frames matches
       load_gamestate's schema, so the SAME engine runs unchanged.
    """
    import zipfile, pickle
    import numpy as np
    z = zipfile.ZipFile(pklz_path)
    dfname = [n for n in z.namelist() if n.endswith(".pkl") and "image" not in n][0]
    df = pickle.loads(z.read(dfname))
    df = df[df["bbox_pitch"].apply(lambda b: isinstance(b, dict) and b.get("x_bottom_middle") is not None)]

    # per-track: mean reid embedding (identity), majority role, and mean torso colour (kit) if frames given
    _img_cache = {}

    def _frame_img(fid):
        if frames_dir is None:
            return None
        if fid not in _img_cache:
            from PIL import Image
            p = f"{frames_dir}/{fid:06d}.jpg"
            _img_cache[fid] = Image.open(p).convert("RGB") if os.path.exists(p) else None
        return _img_cache[fid]

    tracks = {}
    for tid, g in df.groupby("track_id"):
        embs = [np.asarray(e).ravel() for e in g["embeddings"] if e is not None]
        roles_g = [r for r in g["role_detection"].tolist() if r]
        cols = []
        if frames_dir is not None:
            for _, r in g.iterrows():
                im = _frame_img(int(str(r["image_id"])[-6:]))
                if im is not None:
                    cols.append(_torso_color(im, r["bbox_ltwh"]))
        tracks[tid] = {"emb": np.mean(embs, 0) if embs else None,
                       "role": max(set(roles_g), key=roles_g.count) if roles_g else "player",
                       "color": np.mean(cols, 0) if cols else None}

    outfield = [t for t, v in tracks.items() if v["role"] != "referee" and v["emb"] is not None]
    use_color = frames_dir is not None and all(tracks[t]["color"] is not None for t in outfield)
    feat = np.stack([tracks[t]["color"] if use_color else tracks[t]["emb"] for t in outfield])
    lab = _kmeans2(feat)
    tid_team = {t: float(lab[i]) for i, t in enumerate(outfield)}

    frames, ident = {}, {}
    for _, r in df.iterrows():
        tid = r["track_id"]
        if tid not in tid_team:
            continue
        fid = int(str(r["image_id"])[-6:])
        bp = r["bbox_pitch"]
        xy = (float(bp["x_bottom_middle"]) + PITCH_L / 2.0, float(bp["y_bottom_middle"]) + PITCH_W / 2.0)
        frames.setdefault(fid, {"ball": None, "players": {}})
        frames[fid]["players"][tid] = (tid_team[tid], xy)
        ident.setdefault(tid, {"role": tracks[tid]["role"], "jersey": None,
                               "team": "A" if tid_team[tid] == 0 else "B",
                               "embedding": tracks[tid]["emb"], "track_id": tid})
    if ball_frames:
        for fid in frames:
            b = ball_frames.get(fid)
            if b is not None:
                frames[fid]["ball"] = b["ball"] if isinstance(b, dict) else b
    return frames, ident
