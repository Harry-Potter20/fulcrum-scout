# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "kloppy", "numpy", "huggingface_hub>=0.34"]
# ///
"""Canonical, config-driven football World Model (covtoken coverage-constrained). HF Job (GPU).

Replaces the v1..v5 copy-paste jobs with ONE DRY module. Experiments are CONFIGS, not new files.
Modes (env MODE): "run" (default, full experiment), "smoke" (cheap CPU end-to-end test of every code path),
"selfcheck" (cheap CPU data-pipeline shape check). Config via env WM_CONFIG (JSON) overrides DEFAULT_CONFIG.

v6 corrections over v5: (a) proper big-model training (lr warmup + cosine, tuned lr/epochs per regime);
(b) easy-bin protection via direct mu-shrinkage on low-deviation players (targets the real harm: nonzero residuals
where physics is optimal), not an NLL constraint; (c) clean SINGLE-STEP deterministic stratification (no rollout
compounding noise). Coverage of the decisive tail stays a dual-controlled constraint (the covtoken mechanism).
"""
from __future__ import annotations
from dataclasses import dataclass
import json, math, os, time
import numpy as np

# ============================================================ constants
PITCH_L, PITCH_W = 105.0, 68.0
VEL_WINDOW_S = 0.12                   # velocity-smoothing window (seconds) -- fps-agnostic
HORIZON_S = 0.52                      # one prediction step (seconds); STEPS*HORIZON_S ~= 2.08s rollout
STEPS, MAX_NODES, RS = 4, 28, 3.0     # rollout steps; padded node count (+1 ball); residual scale (metres)
# events modality (Metrica vocabulary; sources without events -> UNKNOWN=0). One-hot conditions the dynamics.
EVENT_VOCAB = ["UNKNOWN", "PASS", "SHOT", "CHALLENGE", "RECOVERY", "BALL LOST", "BALL OUT", "SET PIECE",
               "FAULT RECEIVED", "CARD"]
EVENT_ID = {name: i for i, name in enumerate(EVENT_VOCAB)}
N_EVENTS = len(EVENT_VOCAB)


# ============================================================ data
@dataclass
class Window:
    """One training/eval example. Node 0 is the ball; nodes 1.. are players. All positions in metres,
    velocities in m/s (physical, so 10fps and 25fps sources are directly comparable)."""
    pos: np.ndarray                   # [N,2]
    vel: np.ndarray                   # [N,2] m/s
    acc: np.ndarray                   # [N,2] m/s^2
    team: np.ndarray                  # [N]  2.0=ball, 1.0=home, 0.0=away
    isball: np.ndarray                # [N]  1.0=ball else 0.0
    futs: list                        # STEPS x [N,2] future positions at t + k*HORIZON_S
    event_ctx: int = 0                # active event-type id at the anchor frame (EVENT_ID); 0=UNKNOWN
    next_event: int = 0               # id of the NEXT event to occur after the anchor (M2 target); 0=UNKNOWN


def _build_windows(frame_ids, ball_of, players_of, fps, stride, cap, event_of=None, next_event_of=None,
                   min_players=16) -> list[Window]:
    """Source-agnostic window builder. ball_of(fid)->(x,y)|None; players_of(fid)->{pid:(team01,(x,y))};
    event_of(fid)->current event id, next_event_of(fid)->next event id (optional; default UNKNOWN). Frame
    offsets derive from fps so a fixed SECONDS horizon is sampled correctly for any source. `min_players` is
    the minimum tracked players present across the whole window; broadcast (GSR) sources see a partial pitch
    so callers lower it."""
    event_of = event_of or (lambda fid: 0)
    next_event_of = next_event_of or (lambda fid: 0)
    vs = max(1, round(VEL_WINDOW_S * fps))
    hs = max(1, round(HORIZON_S * fps))
    dt = 1.0 / fps
    idset = set(frame_ids)
    out: list[Window] = []
    for fid in sorted(idset)[::stride]:
        need = [fid - 2 * vs, fid - vs, fid] + [fid + k * hs for k in range(1, STEPS + 1)]
        if any(f not in idset for f in need):
            continue
        balls = [ball_of(f) for f in (need[2], need[1], need[0], *need[3:])]     # a, v, v2, futs
        if any(b is None for b in balls):
            continue
        b_a, b_v, b_v2, b_futs = balls[0], balls[1], balls[2], balls[3:]
        cur, prev, prev2 = players_of(need[2]), players_of(need[1]), players_of(need[0])
        futs = [players_of(f) for f in need[3:]]
        pids = [p for p in cur if p in prev and p in prev2 and all(p in ff for ff in futs)]
        if len(pids) < min_players:
            continue
        pids = pids[:MAX_NODES - 1]
        pos = np.array([b_a] + [cur[p][1] for p in pids])
        pv = np.array([b_v] + [prev[p][1] for p in pids])
        pv2 = np.array([b_v2] + [prev2[p][1] for p in pids])
        vel = (pos - pv) / (vs * dt)
        acc = (vel - (pv - pv2) / (vs * dt)) / (vs * dt)
        team = np.array([2.0] + [cur[p][0] for p in pids])
        isball = np.array([1.0] + [0.0] * len(pids))
        futarr = [np.array([b_futs[k]] + [futs[k][p][1] for p in pids]) for k in range(STEPS)]
        out.append(Window(pos, vel, acc, team, isball, futarr,
                          event_ctx=int(event_of(need[2])), next_event=int(next_event_of(need[2]))))
        if len(out) >= cap:
            break
    return out


def _metrica_events(match_id: int):
    """Frame -> active event-type id (most recent event started at/before the frame). O(log E) per lookup.
    Returns a constant-UNKNOWN function if the events CSV is absent (e.g. Metrica game 3)."""
    import urllib.request, csv, io, bisect
    url = (f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
           f"Sample_Game_{match_id}/Sample_Game_{match_id}_RawEventsData.csv")
    try:
        raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    except Exception:
        return (lambda fid: 0), (lambda fid: 0)
    rows = list(csv.DictReader(io.StringIO(raw)))
    ev = sorted((int(r["Start Frame"]), EVENT_ID.get((r.get("Type") or "").strip().upper(), 0))
                for r in rows if (r.get("Start Frame") or "").strip().isdigit())
    starts = [e[0] for e in ev]
    ids = [e[1] for e in ev]

    def event_of(fid):                                # current: most recent event started at/before fid
        i = bisect.bisect_right(starts, fid) - 1
        return ids[i] if i >= 0 else 0

    def next_event_of(fid):                           # next: first event that starts strictly after fid
        i = bisect.bisect_right(starts, fid)
        return ids[i] if i < len(ids) else 0

    return event_of, next_event_of


def load_metrica(match_id: int, stride: int, cap: int) -> list[Window]:
    from kloppy import metrica
    ds = metrica.load_open_data(match_id=match_id).transform(to_coordinate_system="secondspectrum")
    fmap = {f.frame_id: f for f in ds.frames}
    shift = lambda pt: (pt.x + PITCH_L / 2, pt.y + PITCH_W / 2)

    def ball_of(fid):
        b = fmap[fid].ball_coordinates
        return None if (b is None or not math.isfinite(b.x)) else shift(b)

    def players_of(fid):
        return {pl.player_id: (1.0 if "home" in str(pl.team.ground).lower() else 0.0, shift(pt))
                for pl, pt in fmap[fid].players_coordinates.items()
                if pt is not None and math.isfinite(pt.x) and math.isfinite(pt.y)}

    event_of, next_event_of = _metrica_events(match_id)
    return _build_windows(fmap.keys(), ball_of, players_of, fps=25, stride=stride, cap=cap,
                          event_of=event_of, next_event_of=next_event_of)


def load_skillcorner(match_id: int, stride: int, cap: int) -> list[Window]:
    import urllib.request
    raw_base = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}"
    media = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}"
    meta = json.loads(urllib.request.urlopen(f"{raw_base}/{match_id}_match.json", timeout=90).read().decode())
    home = meta.get("home_team")
    home_id = home["id"] if isinstance(home, dict) else home
    team_of = {}                                       # player_id -> 1.0 home / 0.0 away (team is only in meta)
    for p in meta.get("players", []):
        pid = p.get("player_id", p.get("id"))
        tid = p.get("team_id")
        if pid is not None and tid is not None:
            team_of[pid] = 1.0 if tid == home_id else 0.0
    raw = urllib.request.urlopen(f"{media}/{match_id}_tracking_extrapolated.jsonl", timeout=300).read().decode("utf-8", "ignore")
    frames = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("frame") is not None and rec.get("player_data"):
            frames[rec["frame"]] = rec

    def ball_of(fid):
        b = frames[fid].get("ball_data") or {}
        x, y = b.get("x"), b.get("y")
        return None if x is None or y is None else (float(x) + PITCH_L / 2, float(y) + PITCH_W / 2)

    def players_of(fid):
        out = {}
        for pd in frames[fid]["player_data"]:
            x, y, pid = pd.get("x"), pd.get("y"), pd.get("player_id")
            if x is None or y is None or pid not in team_of:
                continue
            out[pid] = (team_of[pid], (float(x) + PITCH_L / 2, float(y) + PITCH_W / 2))
        return out

    return _build_windows(frames.keys(), ball_of, players_of, fps=10, stride=stride, cap=cap)


def load_gsr(labels_ref, stride: int, cap: int) -> list[Window]:
    """Real broadcast tracking from a SoccerNet Game-State `Labels-GameState.json` (URL or local path).
    Same physical schema as the other sources, so it feeds the identical featurizer. Partial-pitch view ->
    min_players lowered. No event stream (event_ctx=UNKNOWN). This is the video->world-model data path: once
    the GSR detector runs on many clips, each becomes training windows here."""
    import urllib.request
    if str(labels_ref).startswith("http"):
        d = json.loads(urllib.request.urlopen(labels_ref, timeout=180).read().decode("utf-8", "ignore"))
    else:
        d = json.load(open(labels_ref))
    fid_of = {img["image_id"]: int(img["file_name"].split(".")[0]) for img in d["images"]}
    frames = {fid: {"ball": None, "players": {}} for fid in fid_of.values()}
    for a in d["annotations"]:
        fid = fid_of.get(a["image_id"])
        bp = a.get("bbox_pitch")
        if fid is None or bp is None:
            continue
        x = (bp["x_bottom_left"] + bp["x_bottom_right"]) / 2 + PITCH_L / 2
        y = (bp["y_bottom_left"] + bp["y_bottom_right"]) / 2 + PITCH_W / 2
        cat = a["category_id"]
        if cat == 4:                                      # ball
            frames[fid]["ball"] = (x, y)
        elif cat in (1, 2):                               # player / goalkeeper
            team = (a.get("attributes", {}) or {}).get("team")
            if team in ("left", "right"):
                frames[fid]["players"][a["track_id"]] = (1.0 if team == "right" else 0.0, (x, y))
    return _build_windows(frames.keys(), lambda f: frames[f]["ball"], lambda f: frames[f]["players"],
                          fps=25, stride=stride, cap=cap, min_players=10)


SPORTEC_IDS = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]   # IDSSE open matches


def load_pff(match_id, stride: int, cap: int) -> list[Window]:
    """PFF FC World Cup 2022 (private repo, research licence — never redistribute). Probed 2026-07: kloppy
    pff.load_tracking, 29.97fps, 22 players, ball present. fps read from metadata; velocities are physical
    (m/s) so mixing 25fps (Metrica/Sportec) and ~30fps (PFF) sources is sound."""
    import os
    from kloppy import pff
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    dl = lambda rp: hf_hub_download("Chucks90/pff-wc22", rp, repo_type="dataset", token=tok)
    ds = pff.load_tracking(meta_data=dl(f"metadata/{match_id}.json"),
                           roster_meta_data=dl(f"rosters/{match_id}.json"),
                           raw_data=dl(f"tracking_data/{match_id}.jsonl.bz2"),
                           coordinates="secondspectrum", only_alive=True)
    fmap = {f.frame_id: f for f in ds.frames}
    fps = getattr(ds.metadata, "frame_rate", 29.97) or 29.97
    shift = lambda pt: (pt.x + PITCH_L / 2, pt.y + PITCH_W / 2)

    def ball_of(fid):
        b = fmap[fid].ball_coordinates
        return None if (b is None or not math.isfinite(b.x)) else shift(b)

    def players_of(fid):
        return {pl.player_id: (1.0 if "home" in str(pl.team.ground).lower() else 0.0, shift(pt))
                for pl, pt in fmap[fid].players_coordinates.items()
                if pt is not None and math.isfinite(pt.x) and math.isfinite(pt.y)}

    return _build_windows(fmap.keys(), ball_of, players_of, fps=fps, stride=stride, cap=cap)


def load_sportec(match_id: str, stride: int, cap: int) -> list[Window]:
    """DFL/Sportec IDSSE open matches (25Hz, both teams + ball). Probed 2026-07: loads via kloppy, 21-22
    players mid-match, ball present. Events exist but their frame alignment (period-relative timestamps) is a
    separate task — windows carry event_ctx=UNKNOWN like GSR; the dynamics signal is what training needs."""
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id, coordinates="secondspectrum")
    fmap = {f.frame_id: f for f in ds.frames}
    fps = getattr(ds.metadata, "frame_rate", 25) or 25
    shift = lambda pt: (pt.x + PITCH_L / 2, pt.y + PITCH_W / 2)

    def ball_of(fid):
        b = fmap[fid].ball_coordinates
        return None if (b is None or not math.isfinite(b.x)) else shift(b)

    def players_of(fid):
        return {pl.player_id: (1.0 if "home" in str(pl.team.ground).lower() else 0.0, shift(pt))
                for pl, pt in fmap[fid].players_coordinates.items()
                if pt is not None and math.isfinite(pt.x) and math.isfinite(pt.y)}

    return _build_windows(fmap.keys(), ball_of, players_of, fps=fps, stride=stride, cap=cap)


def load_source(spec: dict) -> list[Window]:
    """spec = {'source': 'metrica'|'skillcorner'|'gsr', 'match': id-or-labels-ref, 'stride': n, 'cap': m}."""
    loaders = {"metrica": load_metrica, "skillcorner": load_skillcorner, "gsr": load_gsr,
               "sportec": load_sportec, "pff": load_pff}
    return loaders[spec["source"]](spec["match"], spec["stride"], spec.get("cap", 20000))


def featurize(w: Window) -> np.ndarray:
    """Per-node input features (relative coords + kinematics + roles). Single source of truth. -> [N, F]."""
    pos, vel, acc, team, isball = w.pos, w.vel, w.acc, w.team, w.isball
    n = len(pos)
    ball = pos[0]
    centroid = pos[1:].mean(0) if n > 1 else pos.mean(0)
    d_ball = np.linalg.norm(pos - ball, axis=1)
    poss = int(1 + np.argmin(d_ball[1:])) if n > 1 else 0        # ball-carrier = player nearest the ball
    same_team_as_poss = (team == team[poss]).astype(float)
    is_outfield = (team < 2)
    d_nearest_opp = np.array([
        np.min([np.linalg.norm(pos[i] - pos[j]) for j in range(n)
                if is_outfield[i] and is_outfield[j] and team[j] != team[i]] or [30.0])
        for i in range(n)])
    speed = np.linalg.norm(vel, axis=1)
    ev = np.zeros((n, N_EVENTS), np.float32)
    ev[:, w.event_ctx] = 1.0                                     # broadcast event one-hot (dynamics conditioning)
    return np.concatenate([
        (pos - ball) / 50, (pos - centroid) / 50, vel / 8, acc / 8, speed[:, None] / 8,
        d_ball[:, None] / 50, d_nearest_opp[:, None] / 50,
        same_team_as_poss[:, None], (np.arange(n) == poss).astype(float)[:, None], isball[:, None], ev,
    ], axis=1).astype(np.float32)


def featurize_torch(pos, vel, acc, team, isball, mask, event_ctx):
    """Batched, differentiable featurizer matching featurize() exactly (verified in smoke). Enables on-GPU
    multi-step rollout training. pos/vel/acc [B,N,2]; team/isball/mask [B,N]; event_ctx [B]. -> feat [B,N,F]."""
    import torch
    B, N, _ = pos.shape
    player = (mask > 0.5) & (isball < 0.5)                                   # valid outfield players (not ball/pad)
    ball = pos[:, 0, :]                                                      # [B,2]
    pm = player.float()
    centroid = (pos * pm[..., None]).sum(1) / pm.sum(1, keepdim=True).clamp(min=1)          # [B,2]
    d_ball = torch.linalg.norm(pos - ball[:, None, :], dim=-1)                              # [B,N]
    poss = torch.argmin(d_ball + (~player).float() * 1e9, dim=1)                            # [B]
    team_poss = team.gather(1, poss[:, None]).squeeze(1)                                    # [B]
    same = (team == team_poss[:, None]).float()
    is_poss = torch.zeros(B, N, device=pos.device).scatter_(1, poss[:, None], 1.0)
    D = torch.cdist(pos, pos)                                                               # [B,N,N]
    opp = pm[:, :, None] * pm[:, None, :] * (team[:, :, None] != team[:, None, :]).float()  # [B,N,N]
    has_opp = opp.sum(-1) > 0
    d_opp = torch.where(has_opp, (D + (1 - opp) * 1e9).min(-1).values, torch.full((B, N), 30.0, device=pos.device))
    speed = torch.linalg.norm(vel, dim=-1)
    ev = torch.zeros(B, N, N_EVENTS, device=pos.device)
    ev.scatter_(2, event_ctx.long()[:, None, None].expand(B, N, 1), 1.0)                    # broadcast one-hot
    return torch.cat([
        (pos - ball[:, None, :]) / 50, (pos - centroid[:, None, :]) / 50, vel / 8, acc / 8,
        (speed / 8)[..., None], (d_ball / 50)[..., None], (d_opp / 50)[..., None],
        same[..., None], is_poss[..., None], isball[..., None], ev,
    ], dim=-1)


def symmetry_augment(w: Window, rng) -> Window:
    """Random pitch symmetry (the rectangle's Klein 4-group: L<->R reflection, attack-flip, 180deg). Positions
    reflect about pitch centre; velocities/acc negate the reflected component. Exact football symmetry -> cheap
    E(2)-style equivariance / sample efficiency. Kinematics stay physical; team/event unchanged."""
    import dataclasses
    fx, fy = rng.random() < 0.5, rng.random() < 0.5
    if not (fx or fy):
        return w
    def rpos(a):
        a = a.copy()
        if fx: a[..., 0] = PITCH_L - a[..., 0]
        if fy: a[..., 1] = PITCH_W - a[..., 1]
        return a
    def rvec(a):
        a = a.copy()
        if fx: a[..., 0] = -a[..., 0]
        if fy: a[..., 1] = -a[..., 1]
        return a
    return dataclasses.replace(w, pos=rpos(w.pos), vel=rvec(w.vel), acc=rvec(w.acc), futs=[rpos(f) for f in w.futs])


def make_sample(w: Window, rs: float) -> dict:
    """Training sample: input features, kinematic state, 1-step residual target, and all future positions
    (futs) for multi-step rollout-aware training."""
    f32 = np.float32
    phys = w.pos + w.vel * HORIZON_S
    return {"feat": featurize(w), "pos": w.pos.astype(f32), "vel": w.vel.astype(f32), "acc": w.acc.astype(f32),
            "team": w.team.astype(f32), "isball": w.isball.astype(f32),
            "resid": ((w.futs[0] - phys) / rs).astype(f32), "speed": np.linalg.norm(w.vel, axis=1).astype(f32),
            "futs": np.stack(w.futs).astype(f32),
            "event_ctx": int(w.event_ctx), "next_event": int(w.next_event)}   # [STEPS,n,2]; event ids


def collate(samples: list[dict], max_nodes: int) -> dict:
    import torch
    B, F = len(samples), samples[0]["feat"].shape[1]
    S = samples[0]["futs"].shape[0]
    z = lambda *s: np.zeros(s, np.float32)
    buf = {"feat": z(B, max_nodes, F), "pos": z(B, max_nodes, 2), "vel": z(B, max_nodes, 2),
           "acc": z(B, max_nodes, 2), "team": z(B, max_nodes), "isball": z(B, max_nodes),
           "resid": z(B, max_nodes, 2), "speed": z(B, max_nodes), "mask": z(B, max_nodes),
           "futs": z(B, S, max_nodes, 2)}
    ev = np.zeros(B, np.int64)
    nxt = np.zeros(B, np.int64)
    for b, s in enumerate(samples):
        n = min(len(s["feat"]), max_nodes)
        for key in ("feat", "pos", "vel", "acc", "team", "isball", "resid", "speed"):
            buf[key][b, :n] = s[key][:n]
        buf["futs"][b, :, :n] = s["futs"][:, :n]
        buf["mask"][b, :n] = 1.0
        ev[b] = s["event_ctx"]
        nxt[b] = s["next_event"]
    out = {k: torch.from_numpy(v) for k, v in buf.items()}
    out["event_ctx"] = torch.from_numpy(ev)
    out["next_event"] = torch.from_numpy(nxt)
    return out


# ============================================================ model
def build_model(feat_dim: int, d: int = 96, heads: int = 4, layers: int = 2, gated: bool = False):
    import torch
    import torch.nn as nn

    class RelationAttention(nn.Module):
        """Multi-head attention biased by inter-player distance (soft locality) and teammate/opponent type."""
        def __init__(s):
            super().__init__()
            s.q, s.k, s.v, s.o = (nn.Linear(d, d) for _ in range(4))
            s.type_bias = nn.Parameter(torch.zeros(2))       # [same_team, opp_team]
            s.h, s.dh = heads, d // heads

        def forward(s, x, pos, team, mask):
            B, N, _ = x.shape
            q, k, v = (proj(x).view(B, N, s.h, s.dh).transpose(1, 2) for proj in (s.q, s.k, s.v))
            score = (q @ k.transpose(-1, -2)) / math.sqrt(s.dh)
            same = (team[:, :, None] == team[:, None, :]).float()
            bias = (-torch.cdist(pos, pos) / 10.0) + same * s.type_bias[0] + (1 - same) * s.type_bias[1]
            score = (score + bias[:, None]).masked_fill(mask[:, None, None, :] < 0.5, -1e9)
            out = (torch.softmax(score, dim=-1) @ v).transpose(1, 2).reshape(B, N, d)
            return s.o(out)

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.attn, s.n1, s.n2 = RelationAttention(), nn.LayerNorm(d), nn.LayerNorm(d)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(s, x, pos, team, mask):
            x = x + s.attn(s.n1(x), pos, team, mask)
            return x + s.ff(s.n2(x))

    class WorldModel(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d))
            s.blocks = nn.ModuleList([Block() for _ in range(layers)])
            # gated: one extra channel is a per-player GATE on whether to emit a residual at all.
            # The shrink penalty in TwoSidedLoss needs the TRUE deviation to know who is "easy", so it exists
            # only at training time and is a regulariser, not a mechanism. A gate is available at inference:
            # the model itself decides to defer to physics, and pays a budget for every residual it emits.
            s.gated = gated
            s.head = nn.Linear(d, 5 if gated else 4)          # (mu_x, mu_y, logsig_x, logsig_y[, gate])
            s.event_head = nn.Linear(d, N_EVENTS)             # M2: next-event prediction from the pooled latent
            s.value_head = nn.Linear(d, 1)                    # state valuation (possession value) from the SAME latent

        def forward(s, feat, pos, team, mask, return_event=False, return_value=False, return_embedding=False):
            x = s.enc(feat)
            for blk in s.blocks:
                x = blk(x, pos, team, mask)
            o = s.head(x)
            if s.gated:
                gate = torch.sigmoid(o[..., 4])
                mu, logsig = o[..., :2] * gate[..., None], o[..., 2:4].clamp(-4, 3)
                s.last_gate = gate
            else:
                mu, logsig = o[..., :2], o[..., 2:].clamp(-4, 3)
            if return_event or return_value or return_embedding:
                pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)   # masked mean latent
                if return_embedding:                          # schematic's retrieval head: the shared latent itself
                    return mu, logsig, pooled
                if return_value:                              # schematic's state-valuation head on the shared encoder
                    return mu, logsig, s.value_head(pooled).squeeze(-1)
                return mu, logsig, s.event_head(pooled)
            return mu, logsig

    return WorldModel()


def load_checkpoint(path: str, device="cpu"):
    """Rebuild a trained model from a checkpoint saved by run(). -> (model, checkpoint_dict)."""
    import torch
    ck = torch.load(path, map_location=device)
    model = build_model(ck["feat_dim"], d=ck["config"]["d"], layers=ck["config"]["layers"],
                        gated=bool(ck["config"].get("gated"))).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)   # strict=False: pre-value_head checkpoints still load
    model.eval()
    return model, ck


# ============================================================ losses (strategy pattern; DRY)
def gaussian_nll(mu, logsig, resid):
    """Per-player Gaussian NLL. -> [B, N]."""
    import torch
    return (0.5 * ((resid - mu) / torch.exp(logsig)) ** 2 + logsig).sum(-1)


def _masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1)


class Loss:
    """Base loss policy. __call__(l, mu, batch) -> (scalar_loss, info_dict)."""
    def __call__(self, l, mu, batch):
        raise NotImplementedError


class UniformLoss(Loss):
    def __call__(self, l, mu, b):
        return _masked_mean(l, b["mask"]), {}


class SoftWeightLoss(Loss):
    """Fixed movement-weighting (the v4/v5 baseline)."""
    def __call__(self, l, mu, b):
        w = (0.2 + b["speed"]) * b["mask"]
        return (l * w).sum() / w.sum().clamp(min=1), {}


class ConstrainedLoss(Loss):
    """covtoken: constrain tail loss <= kappa * overall, enforced by a capped dual variable (dual ascent)."""
    def __init__(self, kappa=2.0, eta=0.03, lam_max=5.0, tail_q=0.90):
        self.kappa, self.eta, self.lam_max, self.tail_q, self.lam = kappa, eta, lam_max, tail_q, 0.0

    def _tail_mask(self, dev, valid):
        import torch
        thr = torch.quantile(dev[valid], self.tail_q) if valid.any() else dev.new_zeros(())
        return ((dev >= thr) & valid).float()

    def __call__(self, l, mu, b):
        import torch
        m = b["mask"]
        L_all = _masked_mean(l, m)
        dev = torch.linalg.norm(b["resid"], dim=-1)                # true deviation magnitude
        valid = m > 0.5
        tail = self._tail_mask(dev, valid)
        L_tail = (l * tail).sum() / tail.sum().clamp(min=1)
        loss = L_all + self.lam * torch.relu(L_tail - self.kappa * L_all.detach())
        with torch.no_grad():
            self.lam = float(min(self.lam_max, max(0.0, self.lam + self.eta * (L_tail.item() - self.kappa * L_all.item()))))
        return loss, {"lam": round(self.lam, 3), "L_all": round(L_all.item(), 3), "L_tail": round(L_tail.item(), 3)}


class TwoSidedLoss(ConstrainedLoss):
    """Constrained tail coverage + direct mu-shrinkage on easy (low-deviation) players -> keep residual ~0 where
    physics is optimal (the real harm, in position space, not NLL)."""
    def __init__(self, beta=2.0, easy_q=0.50, **kw):
        super().__init__(**kw)
        self.beta, self.easy_q = beta, easy_q

    def __call__(self, l, mu, b):
        import torch
        loss, info = super().__call__(l, mu, b)
        dev = torch.linalg.norm(b["resid"], dim=-1)
        valid = b["mask"] > 0.5
        ethr = torch.quantile(dev[valid], self.easy_q) if valid.any() else dev.new_zeros(())
        easy = ((dev <= ethr) & valid).float()
        mu_sq = (mu ** 2).sum(-1)                                   # ||predicted residual||^2 per player
        shrink = (mu_sq * easy).sum() / easy.sum().clamp(min=1)
        info["shrink"] = round(shrink.item(), 4)
        return loss + self.beta * shrink, info


class GatedConstrainedLoss(ConstrainedLoss):
    """v7 residual economy: the tail-coverage constraint, plus a budget on how much residual the model emits.

    Rationale from v6: the constraint buys the decisive tail (+30.6% vs +5.8) but costs the easy majority
    (+8.4 vs +25.2), because the model keeps emitting residuals for players physics already predicts. Here the
    model must *spend* to emit one — mean gate is pushed toward `budget` — so predictable players fall back to
    physics exactly (gate -> 0) and the constraint pushes the remaining spend onto the decisive tail. Unlike
    TwoSidedLoss's mu-shrink (which needs the TRUE deviation, so it exists only at training time), the gate is
    part of the model and acts at inference."""
    def __init__(self, budget=0.35, gamma=1.0, **kw):
        super().__init__(**kw)
        self.budget, self.gamma = budget, gamma

    def __call__(self, l, mu, b):
        import torch
        loss, info = super().__call__(l, mu, b)
        gate = b.get("gate")
        if gate is not None:
            m = b["mask"]
            g_mean = (gate * m).sum() / m.sum().clamp(min=1)
            loss = loss + self.gamma * torch.relu(g_mean - self.budget)     # pay only when over budget
            info["gate_mean"] = round(float(g_mean.item()), 4)
        return loss, info


def build_loss(cfg: dict) -> Loss:
    mode = cfg["loss"]
    if mode == "uniform":
        return UniformLoss()
    if mode == "soft":
        return SoftWeightLoss()
    if mode == "constrained":
        return ConstrainedLoss(**cfg.get("loss_kw", {}))
    if mode == "twosided":
        return TwoSidedLoss(**cfg.get("loss_kw", {}))
    if mode == "gated":
        return GatedConstrainedLoss(**cfg.get("loss_kw", {}))
    raise ValueError(f"unknown loss {mode}")


# ============================================================ training
def lr_scale(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))          # cosine decay to 0


def iter_batches(windows, bs, rs, max_nodes, rng, augment=False):
    order = rng.permutation(len(windows))
    for i in range(0, len(order), bs):
        ws = (symmetry_augment(windows[j], rng) for j in order[i:i + bs]) if augment else (windows[j] for j in order[i:i + bs])
        yield collate([make_sample(w, rs) for w in ws], max_nodes)


def rollout_train_loss(model, batch, loss_fn, K, rs):
    """Rollout-aware training (fixes the constraint's easy-bin drift): supervise K steps on the model's OWN
    (detached) rolled-forward state -- scheduled sampling, no backprop-through-time. Step 0 uses the full loss
    policy (tail-coverage constraint etc.); later steps add plain masked-NLL so residuals that compound/drift on
    the rollout (the exact easy-bin failure) get penalised at training time, the way they're measured at eval."""
    import torch
    pos, vel, acc = batch["pos"], batch["vel"], batch["acc"]
    team, isball, mask, futs = batch["team"], batch["isball"], batch["mask"], batch["futs"]
    step_losses, info = [], {}
    event_ctx = batch["event_ctx"]
    for k in range(K):
        feat = featurize_torch(pos, vel, acc, team, isball, mask, event_ctx)
        mu, logsig = model(feat, pos, team, mask)
        resid_true = (futs[:, k] - (pos + vel * HORIZON_S)) / rs
        l = gaussian_nll(mu, logsig, resid_true)
        if k == 0:
            lk, info = loss_fn(l, mu, {**batch, "resid": resid_true,
                                       "gate": getattr(model, "last_gate", None)})   # constraint/coverage at step 0
        else:
            lk = _masked_mean(l, mask)                                      # rollout-consistency at later steps
        step_losses.append(lk)
        newpos = pos + vel * HORIZON_S + mu * rs                            # advance, then DETACH (scheduled sampling)
        newvel = (newpos - pos) / HORIZON_S
        acc = ((newvel - vel) / HORIZON_S).detach()
        pos, vel = newpos.detach(), newvel.detach()
    info["rollout_K"] = K
    return sum(step_losses) / K, info


def train(cfg: dict, windows, feat_dim: int, rs: float, max_nodes: int, device, log, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(feat_dim, d=cfg["d"], layers=cfg["layers"], gated=bool(cfg.get("gated"))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    loss_fn = build_loss(cfg)
    steps_per_epoch = math.ceil(len(windows) / cfg["bs"])
    total = cfg["epochs"] * steps_per_epoch
    warmup = cfg.get("warmup", max(1, total // 20))
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        run_loss = 0.0
        nb = 0
        last = {}
        # rollout warmup: single-step for the first `rollout_warmup_epochs` so the early untrained model
        # doesn't explode the rollout loss (the Phase-2 epoch-1 spike), then ramp to full rollout depth.
        K_ep = 1 if ep < int(cfg.get("rollout_warmup_epochs", 0)) else int(cfg.get("rollout_steps", 1))
        for batch in iter_batches(windows, cfg["bs"], rs, max_nodes, rng, augment=cfg.get("augment_symmetry", False)):
            batch = {k: v.to(device) for k, v in batch.items()}
            K = K_ep
            ev_w = cfg.get("event_loss_w", 0.0)
            if K <= 1:
                if ev_w > 0:                                    # M2: joint dynamics + next-event prediction
                    mu, logsig, ev_logits = model(batch["feat"], batch["pos"], batch["team"], batch["mask"], return_event=True)
                    l = gaussian_nll(mu, logsig, batch["resid"])
                    loss, last = loss_fn(l, mu, {**batch, "gate": getattr(model, "last_gate", None)})
                    ce = torch.nn.functional.cross_entropy(ev_logits, batch["next_event"])
                    loss = loss + ev_w * ce
                    last = {**last, "event_ce": round(ce.item(), 3)}
                else:
                    mu, logsig = model(batch["feat"], batch["pos"], batch["team"], batch["mask"])
                    l = gaussian_nll(mu, logsig, batch["resid"])
                    # the single-step path must charge the gate too — v7's first run only wired the rollout
                    # path, so the gate trained unbudgeted (free capacity knob: tail +33.1 but easy -5.9)
                    loss, last = loss_fn(l, mu, {**batch, "gate": getattr(model, "last_gate", None)})
            else:
                loss, last = rollout_train_loss(model, batch, loss_fn, K, rs)
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * lr_scale(step, warmup, total)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += loss.item()
            nb += 1
            step += 1
        log(f"  [{cfg['name']}] epoch {ep + 1}/{cfg['epochs']} loss={run_loss / max(nb, 1):.3f} {last}")
    return model, last


# ============================================================ eval (clean, single-step, deterministic)
def predict_step(model, pos, vel, acc, team, isball, event_ctx, rs, max_nodes, device):
    """One-step (HORIZON_S) position prediction for a single window. -> [N,2]."""
    import torch
    n = len(pos)
    feat = featurize(Window(pos, vel, acc, team, isball, [], event_ctx=event_ctx))
    f = torch.zeros(1, max_nodes, feat.shape[1]); f[0, :n] = torch.from_numpy(feat)
    p = torch.zeros(1, max_nodes, 2); p[0, :n] = torch.from_numpy(pos.astype(np.float32))
    tm = torch.zeros(1, max_nodes); tm[0, :n] = torch.from_numpy(team.astype(np.float32))
    mk = torch.zeros(1, max_nodes); mk[0, :n] = 1.0
    with torch.no_grad():
        mu, _ = model(f.to(device), p.to(device), tm.to(device), mk.to(device))
    return pos + vel * HORIZON_S + mu[0, :n].cpu().numpy() * rs


def _rollout_final(model, w, steps, rs, max_nodes, device):
    """Autoregressive rollout `steps` (steps=1 => single-step). -> predicted positions at t + steps*HORIZON_S."""
    pos, vel, acc = w.pos.copy(), w.vel.copy(), w.acc.copy()
    for _ in range(steps):
        newpos = predict_step(model, pos, vel, acc, w.team, w.isball, w.event_ctx, rs, max_nodes, device)
        newvel = (newpos - pos) / HORIZON_S
        acc = (newvel - vel) / HORIZON_S
        pos, vel = newpos, newvel
    return pos


def stratify(model, windows, mode, rs, max_nodes, device):
    """Error vs constant-velocity physics, binned by that physics deviation. Unified for both eval modes:
    mode='single_step' (1 step, low noise) or mode='rollout' (STEPS to 2s, the downstream question)."""
    model.eval()
    if mode == "single_step":
        edges, steps, gt_idx = [0, 0.5, 1.5, 3.0, 1e9], 1, 0
    else:
        edges, steps, gt_idx = [0, 2, 5, 10, 1e9], STEPS, STEPS - 1
    nb = len(edges) - 1
    cnt, sum_kin, sum_model = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for w in windows:
        pred = _rollout_final(model, w, steps, rs, max_nodes, device)
        gt = w.futs[gt_idx]
        phys = w.pos + w.vel * (HORIZON_S * steps)
        e_model = np.linalg.norm(pred - gt, axis=1)
        e_kin = np.linalg.norm(phys - gt, axis=1)
        for i in range(1, len(gt)):                                # skip ball
            b = min(max(np.searchsorted(edges, e_kin[i], side="right") - 1, 0), nb - 1)
            cnt[b] += 1; sum_kin[b] += e_kin[i]; sum_model[b] += e_model[i]
    rows = []
    for b in range(nb):
        lab = f"{edges[b]}-{edges[b + 1] if edges[b + 1] < 1e9 else 'inf'}"
        if cnt[b] < 1:
            rows.append({"bin_m": lab, "n": 0}); continue
        mk, mm = sum_kin[b] / cnt[b], sum_model[b] / cnt[b]
        rows.append({"bin_m": lab, "n": int(cnt[b]), "kin": round(mk, 3), "model": round(mm, 3),
                     "impr_pct": round(100 * (mk - mm) / mk, 1)})
    return rows


def event_accuracy(model, windows, max_nodes, device):
    """M2 eval: does the tracking latent predict the NEXT event? Returns (model_acc, majority_baseline_acc)."""
    import torch
    model.eval()
    correct = tot = 0
    labels = []
    for batch in iter_batches(windows, 128, RS, max_nodes, np.random.default_rng(0)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            _, _, ev = model(batch["feat"], batch["pos"], batch["team"], batch["mask"], return_event=True)
        pred = ev.argmax(-1)
        correct += (pred == batch["next_event"]).sum().item()
        tot += len(pred)
        labels.extend(batch["next_event"].cpu().tolist())
    maj = max(np.bincount(labels, minlength=N_EVENTS)) / max(len(labels), 1) if labels else 0.0
    return round(correct / max(tot, 1), 3), round(float(maj), 3)


def aggregate_seeds(seed_rows: list) -> list:
    """Combine per-seed stratifications -> per-bin mean/std of improvement% (error bars across seeds)."""
    agg = []
    for b in range(len(seed_rows[0])):
        lab = seed_rows[0][b]["bin_m"]
        imprs = [sr[b]["impr_pct"] for sr in seed_rows if sr[b].get("n", 0) > 0]
        ns = [sr[b].get("n", 0) for sr in seed_rows]
        if imprs:
            agg.append({"bin_m": lab, "n": int(np.mean(ns)), "impr_mean": round(float(np.mean(imprs)), 1),
                        "impr_std": round(float(np.std(imprs)), 1), "impr_seeds": imprs})
        else:
            agg.append({"bin_m": lab, "n": 0})
    return agg


# ============================================================ experiment config + driver
DEFAULT_CONFIG = {
    "train_sources": [{"source": "metrica", "match": 1, "stride": 6, "cap": 20000},
                      {"source": "metrica", "match": 2, "stride": 6, "cap": 20000}],
    "test_source": {"source": "metrica", "match": 3, "stride": 25, "cap": 1500},
    "regimes": [
        {"name": "soft_small",  "d": 96,  "layers": 2, "lr": 2e-3, "epochs": 10, "bs": 64, "loss": "soft"},
        {"name": "twosided_small", "d": 96, "layers": 2, "lr": 2e-3, "epochs": 10, "bs": 64, "loss": "twosided",
         "loss_kw": {"kappa": 2.0, "lam_max": 5.0, "beta": 2.0}},
        {"name": "soft_big",    "d": 192, "layers": 4, "lr": 5e-4, "epochs": 18, "bs": 64, "loss": "soft"},
        {"name": "constr_big",  "d": 192, "layers": 4, "lr": 5e-4, "epochs": 18, "bs": 64, "loss": "constrained",
         "loss_kw": {"kappa": 2.0, "lam_max": 5.0}},
    ],
}


def _set_determinism():
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def run(cfg, log):
    import torch
    t0 = time.time()
    _set_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_windows = []
    for spec in cfg["train_sources"]:
        w = load_source(spec)
        train_windows += w
        log(f"loaded {spec['source']} match={spec['match']}: {len(w)} windows")
    test_windows = load_source(cfg["test_source"])
    test2 = load_source(cfg["test_source2"]) if cfg.get("test_source2") else None
    feat_dim = featurize(train_windows[0]).shape[1]
    log(f"device={device} train={len(train_windows)} test={len(test_windows)} feat_dim={feat_dim}")
    ckpt_dir = "/mnt/checkpoints" if os.path.isdir("/mnt") else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    seeds = cfg.get("seeds", [0])
    eval_mode = cfg.get("eval_mode", "single_step")
    import dataclasses
    def ablate(ws, on):                                    # force event_ctx=UNKNOWN (controlled events on/off)
        return [dataclasses.replace(w, event_ctx=0) for w in ws] if on else ws
    results = {}
    for regime in cfg["regimes"]:
        tr_w = ablate(train_windows, regime.get("ablate_events"))
        te_w = ablate(test_windows, regime.get("ablate_events"))
        seed_rows, model0, last0 = [], None, {}
        for si, seed in enumerate(seeds):
            model, last = train(regime, tr_w, feat_dim, RS, MAX_NODES, device, log, seed=seed)
            seed_rows.append(stratify(model, te_w, eval_mode, RS, MAX_NODES, device))
            if test2 is not None:
                s2 = stratify(model, test2, eval_mode, RS, MAX_NODES, device)
                log(f"  [{regime['name']}] seed={seed} HELD-OUT-2 ({cfg['test_source2']['source']} "
                    f"{cfg['test_source2']['match']}) {json.dumps(s2)}")
            log(f"[{regime['name']}] seed={seed} {json.dumps(seed_rows[-1])}")
            if si == 0:
                model0, last0 = model, last
        results[regime["name"]] = {"stratification": aggregate_seeds(seed_rows),
                                   "params": int(sum(p.numel() for p in model0.parameters())), "loss_info": last0}
        if regime.get("event_loss_w", 0) > 0:              # M2: does the latent predict the next event?
            acc, maj = event_accuracy(model0, te_w, MAX_NODES, device)
            results[regime["name"]]["next_event_acc"] = acc
            results[regime["name"]]["next_event_majority"] = maj
            log(f"[{regime['name']}] next-event acc={acc} vs majority={maj}")
        if ckpt_dir:                                       # persist seed-0 weights + everything to rebuild/reload
            path = f"{ckpt_dir}/{regime['name']}.pt"
            torch.save({"state_dict": model0.state_dict(), "config": regime, "feat_dim": feat_dim,
                        "horizon_s": HORIZON_S, "rs": RS, "max_nodes": MAX_NODES}, path)
            results[regime["name"]]["checkpoint"] = path
        log(f"[{regime['name']}] AGG {json.dumps(results[regime['name']]['stratification'])}")
    out = {"eval": eval_mode, "seeds": seeds, "regimes": results, "elapsed_s": round(time.time() - t0, 1)}
    if os.path.isdir("/mnt"):
        json.dump(out, open("/mnt/worldmodel_v6_result.json", "w"))
    print("FB_WM_RESULT " + json.dumps(out))


def smoke(log):
    """Cheap CPU end-to-end test of every code path (data -> featurize -> model -> each loss -> stratify)."""
    import torch
    device = "cpu"
    ws = load_metrica(1, 60, 30)
    assert ws, "no windows"
    feat_dim = featurize(ws[0]).shape[1]
    # verify featurize_torch EXACTLY matches the numpy featurize (train/eval consistency is critical)
    b = collate([make_sample(w, RS) for w in ws[:6]], MAX_NODES)
    ft = featurize_torch(b["pos"], b["vel"], b["acc"], b["team"], b["isball"], b["mask"], b["event_ctx"]).numpy()
    for wi in range(6):
        n = len(ws[wi].pos)
        assert np.allclose(ft[wi, :n], featurize(ws[wi]), atol=1e-4), "featurize_torch != featurize"
    log(f"  featurize_torch matches numpy featurize (atol 1e-4); F={featurize(ws[0]).shape[1]} (incl {N_EVENTS} event dims)")
    for loss_name, kw in [("uniform", {}), ("soft", {}), ("constrained", {}), ("twosided", {})]:
        cfg = {"name": f"smoke_{loss_name}", "d": 32, "layers": 1, "lr": 1e-3, "epochs": 1, "bs": 8,
               "loss": loss_name, "loss_kw": kw}
        model = build_model(feat_dim, d=32, layers=1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = build_loss(cfg)
        rng = np.random.default_rng(0)
        for i, batch in enumerate(iter_batches(ws, 8, RS, MAX_NODES, rng)):
            mu, logsig = model(batch["feat"], batch["pos"], batch["team"], batch["mask"])
            l = gaussian_nll(mu, logsig, batch["resid"])
            loss, info = loss_fn(l, mu, batch)
            assert torch.isfinite(loss), f"{loss_name} loss not finite"
            opt.zero_grad(); loss.backward(); opt.step()
            if i >= 2:
                break
        for eval_mode in ("single_step", "rollout"):
            rows = stratify(model, ws[:10], eval_mode, RS, MAX_NODES, device)
            assert rows and any(r.get("n", 0) > 0 for r in rows), f"{loss_name}/{eval_mode} stratify empty"
        agg = aggregate_seeds([stratify(model, ws[:10], "rollout", RS, MAX_NODES, device) for _ in range(2)])
        assert agg, "aggregate empty"
        log(f"  smoke {loss_name}: loss={float(loss):.3f} info={info} agg_bins={len(agg)}")
    # exercise the rollout-aware training path (K>1)
    for lname in ("soft", "constrained"):
        model = build_model(feat_dim, d=32, layers=1).to(device)
        loss_fn = build_loss({"loss": lname, "loss_kw": {}})
        batch = next(iter_batches(ws, 8, RS, MAX_NODES, np.random.default_rng(0)))
        rloss, rinfo = rollout_train_loss(model, batch, loss_fn, K=3, rs=RS)
        assert torch.isfinite(rloss), f"rollout_train {lname} not finite"
        rloss.backward()
        log(f"  rollout_train {lname}: loss={float(rloss):.3f} info={rinfo}")
    # E(2) symmetry augmentation + M2 next-event head
    aug = symmetry_augment(ws[0], np.random.default_rng(0))
    assert aug.pos.shape == ws[0].pos.shape and np.isfinite(aug.pos).all(), "augment invalid"
    ab = next(iter_batches(ws, 8, RS, MAX_NODES, np.random.default_rng(1), augment=True))
    assert torch.isfinite(ab["feat"]).all(), "augmented batch not finite"
    model = build_model(feat_dim, d=32, layers=1).to(device)
    mu, logsig, ev = model(ab["feat"], ab["pos"], ab["team"], ab["mask"], return_event=True)
    assert ev.shape[-1] == N_EVENTS, f"event logits dim {ev.shape}"
    ce = torch.nn.functional.cross_entropy(ev, ab["next_event"])
    assert torch.isfinite(ce), "event ce not finite"
    ce.backward()
    acc, maj = event_accuracy(model, ws[:16], MAX_NODES, device)
    log(f"  E(2) augment OK; M2 event head OK (ce={float(ce):.3f}, acc={acc}, majority={maj})")
    print("SMOKE_OK")


def selfcheck(log):
    for spec in (("metrica", 1), ("metrica", 3), ("skillcorner", 2017461)):
        ws = load_source({"source": spec[0], "match": spec[1], "stride": 30, "cap": 20})
        assert ws, f"no windows for {spec}"
        for w in ws:
            n = len(w.pos)
            assert w.pos.shape == (n, 2) and w.vel.shape == (n, 2) and w.acc.shape == (n, 2)
            assert len(w.futs) == STEPS and all(f.shape == (n, 2) for f in w.futs)
            assert np.isfinite(w.pos).all() and np.isfinite(np.concatenate(w.futs)).all()
            assert featurize(w).shape[0] == n
        teams = {int(t) for w in ws for t in w.team}
        n_home = np.mean([int((w.team == 1.0).sum()) for w in ws])
        n_away = np.mean([int((w.team == 0.0).sum()) for w in ws])
        assert {0, 1} <= teams, f"{spec}: both teams must be present, got {teams}"    # catches team-mapping bugs
        evs = {w.event_ctx for w in ws}
        assert all(0 <= e < N_EVENTS for e in evs), f"{spec}: bad event id {evs}"
        if spec[0] == "metrica":                                                     # games 1&2 carry events
            assert evs != {0} or spec[1] == 3, f"{spec}: expected some events attached, got only UNKNOWN"
        log(f"{spec}: {len(ws)} windows OK, nodes [{min(len(w.pos) for w in ws)},{max(len(w.pos) for w in ws)}], "
            f"~home {n_home:.0f}/away {n_away:.0f}, teams={teams}, events={sorted(evs)}")
    print("SELFCHECK_OK")


def main():
    def log(m):
        print(f"[wm] {m}", flush=True)
    mode = os.environ.get("MODE", "run")
    cfg = json.loads(os.environ["WM_CONFIG"]) if os.environ.get("WM_CONFIG") else DEFAULT_CONFIG
    if mode == "selfcheck":
        selfcheck(log)
    elif mode == "smoke":
        smoke(log)
    else:
        run(cfg, log)


if __name__ == "__main__":
    main()
