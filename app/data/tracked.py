"""app.data.tracked — the MEASURED-from-geometry capability path (spec §9). Named players carry Estimated-from-
production capabilities; here we compute the REAL Fulcrum measurement — per-player space_creation + containment from
the topology stack — over actual tracked states (SoccerNet lean_state). Players are anonymous track ids (identity
never enters the geometry, §37), which is exactly right: this is capability measured from geometry alone.

Topology per frame is expensive, so we precompute per sequence and cache a small JSON to the bucket (§50); the app
just reads it. `compute_measured(seq)` builds + uploads; `load_measured(seq)` reads.
"""
from __future__ import annotations
import os, json, pickle, functools
import numpy as np

REPO = "Chucks90/football-gsr-data"


def _tok():
    return os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()


def _load_states(seq: str) -> dict:
    from huggingface_hub import hf_hub_download
    d = pickle.load(open(hf_hub_download(REPO, f"gsr/lean_state_{seq}.pkl", repo_type="dataset", token=_tok()), "rb"))
    return d["lean"] if "lean" in d else d


def compute_measured(seq: str, stride: int = 3, upload: bool = True) -> dict:
    """Per tracked player: mean space_creation (when their team attacks) + containment (when defending) +
    shape_influence (their own team's structural cohesion, possession-independent) over the match, then
    percentile-ranked within the match. Returns {players:[...], seq, n_frames}. Cached to bucket."""
    from fulcrum import metrics as M
    states = _load_states(seq)
    fids = sorted(f for f in states if isinstance(states[f], dict) and states[f].get("players"))[::stride]
    sc_sum, cont_sum, shp_sum, sc_n, cont_n, shp_n = {}, {}, {}, {}, {}, {}
    team_of, x_sum, x_n = {}, {}, {}
    for f in fids:
        st = states[f]
        ball = list(st["ball"]) if st.get("ball") else [52.5, 34.0]
        by_team = {0.0: ([], []), 1.0: ([], [])}                      # team -> (xy list, id list)
        for tid, (tm, xy) in st["players"].items():
            if tm in by_team:
                by_team[tm][0].append(xy); by_team[tm][1].append(tid)
                team_of[tid] = tm; x_sum[tid] = x_sum.get(tid, 0) + xy[0]; x_n[tid] = x_n.get(tid, 0) + 1
        for att_t in (0.0, 1.0):
            dfn_t = 1.0 - att_t
            att, aid = by_team[att_t]; dfn, did = by_team[dfn_t]
            if len(att) < 5 or len(dfn) < 5:
                continue
            sc = M.space_creation({"att": att, "dfn": dfn, "ball": ball, "att_ids": aid})
            for tid, v in (sc or {}).items():
                sc_sum[tid] = sc_sum.get(tid, 0) + float(v); sc_n[tid] = sc_n.get(tid, 0) + 1
            cont = M.containment({"att": att, "dfn": dfn, "ball": ball, "dfn_ids": did})
            for tid, v in (cont or {}).items():
                cont_sum[tid] = cont_sum.get(tid, 0) + float(v); cont_n[tid] = cont_n.get(tid, 0) + 1
        # shape_influence: own team's structural cohesion — possession-independent, so computed per FIXED team,
        # not per att/dfn role (a team's own shape is a property of that team, not of who has the ball)
        for tm, (xy, tid_list) in by_team.items():
            if len(xy) < 5:
                continue
            shp = M.shape_influence(xy, tid_list)
            for tid, v in (shp or {}).items():
                shp_sum[tid] = shp_sum.get(tid, 0) + v["shape_influence"]; shp_n[tid] = shp_n.get(tid, 0) + 1

    tids = [t for t in team_of if sc_n.get(t, 0) >= 8 or cont_n.get(t, 0) >= 8 or shp_n.get(t, 0) >= 8]
    sc_mean = {t: sc_sum.get(t, 0) / sc_n[t] for t in tids if sc_n.get(t)}
    cont_mean = {t: cont_sum.get(t, 0) / cont_n[t] for t in tids if cont_n.get(t)}
    shp_mean = {t: shp_sum.get(t, 0) / shp_n[t] for t in tids if shp_n.get(t)}

    def pctile(d):
        if not d: return {}
        ks = list(d); v = np.array([d[k] for k in ks])
        order = v.argsort(); pr = np.empty(len(v)); pr[order] = np.linspace(0, 100, len(v))
        return {k: round(float(pr[i]), 1) for i, k in enumerate(ks)}
    sc_pct, cont_pct, shp_pct = pctile(sc_mean), pctile(cont_mean), pctile(shp_mean)

    players = []
    for t in sorted(tids):
        players.append({
            "tid": int(t), "team": int(team_of[t]),
            "space_creation": sc_pct.get(t), "containment": cont_pct.get(t), "shape_influence": shp_pct.get(t),
            "mean_x": round(x_sum[t] / max(x_n[t], 1), 1),           # role hint (0=own goal side..105)
            "frames": int(max(sc_n.get(t, 0), cont_n.get(t, 0), shp_n.get(t, 0))),
        })
    out = {"seq": seq, "n_frames": len(fids), "stride": stride, "players": players,
           "note": "Measured from geometry (topology) on recognition-free tracked states — anonymous track ids."}
    if upload:
        from huggingface_hub import HfApi
        json.dump(out, open(f"/tmp/measured_{seq}.json", "w"), indent=2)
        HfApi(token=_tok()).upload_file(path_or_fileobj=f"/tmp/measured_{seq}.json",
                                        path_in_repo=f"gsr/measured_caps_{seq}.json", repo_id=REPO, repo_type="dataset")
    return out


def load_measured_from_bucket(seq: str) -> dict:
    """The bucket is the source of truth (written by compute_measured); ALWAYS reads from there, never Postgres.
    Used by jobs/sync_db.py — syncing FROM the db it's supposed to be refreshing would just write stale data back
    into itself. App code should call load_measured() instead (DB-first, faster for the live product)."""
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(REPO, f"gsr/measured_caps_{seq}.json", repo_type="dataset", token=_tok())))


def available_sequences_from_bucket() -> list:
    """Bucket-direct sequence list — see load_measured_from_bucket for why sync needs this instead of the
    DB-preferring available_sequences() (a sequence with 0 rows so far in Postgres would never appear as
    'available' via the DB path, so a first sync would silently skip it)."""
    from huggingface_hub import HfApi
    fs = HfApi(token=_tok()).list_repo_files(REPO, repo_type="dataset")
    return sorted(f.split("/")[-1].replace("measured_caps_", "").replace(".json", "")
                  for f in fs if "measured_caps_SNGS" in f)


@functools.lru_cache(maxsize=8)
def load_measured(seq: str) -> dict:
    """Fulcrum Scout's own db first (kept fresh by jobs/sync_db.py); HF bucket as fallback when the db isn't
    provisioned or doesn't have this sequence yet."""
    from app import db as APPDB
    if APPDB.available():
        try:
            players = APPDB.read_measured(seq)
            if players:
                return {"seq": seq, "n_frames": None, "players": players}
        except Exception:
            pass
    return load_measured_from_bucket(seq)


def available_sequences() -> list:
    from app import db as APPDB
    if APPDB.available():
        try:
            seqs = APPDB.measured_sequences()
            if seqs:
                return seqs
        except Exception:
            pass
    from huggingface_hub import HfApi
    fs = HfApi(token=_tok()).list_repo_files(REPO, repo_type="dataset")
    return sorted(f.split("/")[-1].replace("measured_caps_", "").replace(".json", "")
                  for f in fs if "measured_caps_SNGS" in f)


def role_hint(mean_x: float, team: int) -> str:
    """Rough role from mean pitch-x + attacking direction (team 0 attacks +x here by convention)."""
    x = mean_x if team == 0 else (105.0 - mean_x)
    return "Forward line" if x >= 68 else ("Midfield" if x >= 42 else ("Defensive line" if x >= 20 else "Deep / keeper"))
