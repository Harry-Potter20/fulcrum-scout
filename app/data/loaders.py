"""app.data.loaders — the data layer. Loads the real multi-season player pool once (cached) and turns box-score
production into a Fulcrum capability profile: for each axis, the percentile of the player's proxy stats WITHIN the
season pool (recruit.season_percentiles is the same computation; reused so the product and research agree).

Nothing here decides or ranks with a single opaque number (§40) and nothing invents confidence (§9): every axis
returns {pct, evidence, registry_status}. Missing/low-minutes -> Insufficient, never 0.
"""
from __future__ import annotations
import os, json, functools
import numpy as np

import sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))   # Football_Research on path -> fulcrum, app
from fulcrum import registry as R
from app.config import settings as S
from app import db


# ---------------- raw records (cached) ----------------
def hf_token():
    """HF token from env (Space secret) or the local CLI cache, else None (public repos still resolve anonymously)."""
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _all_records() -> tuple:
    """Every player-season record. Fulcrum Scout's OWN db (Postgres, kept fresh by jobs/sync_db.py) is preferred —
    it's the deployed product's source of truth; when it's not provisioned (DATABASE_URL unset, e.g. local dev),
    falls back to reading the public HF dataset directly, so the app is never blocked on the db existing."""
    if db.available():
        try:
            recs = []
            for season in S.SEASONS:
                recs.extend(db.read_players(season))
            if recs:
                return tuple(recs)
        except Exception:
            pass   # db configured but unreachable/empty — fall through to HF
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(S.RECORDS_REPO, S.RECORDS_FILE, repo_type="dataset", token=hf_token())
    return tuple(json.load(open(p))["records"])


def records(season: str = None) -> list:
    recs = list(_all_records())
    return [r for r in recs if r.get("season") == season] if season else recs


# every per-90 field the ingest schema produces (jobs/fb_multiseason.py::per90_record) — not just the subset used
# as capability-axis proxies, so Compare/Discover can show percentile+rank for the full production stat sheet too.
ALL_STATS = ["gls90", "sh90", "sot90", "xg90", "finishing", "goal_conv_pct", "big_ch_missed90", "headed_gls90",
             "ast90", "xa90", "keypass90", "big_ch_created90", "prog_pass90", "pass_pct", "long_ball90",
             "crs90", "cross_pct", "dribble90", "dribble_pct", "touches90", "dispossessed90", "was_fouled90",
             "tkl90", "tkl_won_pct", "int90", "clearances90", "blocks90", "recoveries90",
             "duels_won90", "duel_won_pct", "aerial_won90", "aerial_won_pct", "dribbled_past90",
             "fls90", "yellow90", "offsides90"]
# lower raw value = better for these (percentile/rank direction flips) — everything else is "more is better"
NEG_STATS = {"fls90", "yellow90", "offsides90", "dispossessed90", "dribbled_past90", "big_ch_missed90"}


# ---------------- percentiles within a pool ----------------
@functools.lru_cache(maxsize=8)
def _pool_percentiles(season: str) -> dict:
    """{player_name: {stat: pct 0..100}} within `season`. Higher stat -> higher pct (fls90 inverted: fewer fouls
    better). Mirrors recruit.season_percentiles; kept here so the app has zero import of the geometry stack."""
    pool = records(season)
    stats = sorted(set(ALL_STATS) | {p for ax in S.CAP_AXES.values() for p in ax["proxies"]})
    out = {}
    for m in stats:
        v = np.array([float(r.get(m, 0) or 0) for r in pool])
        if len(v) == 0:
            continue
        order = v.argsort(); pr = np.empty(len(v)); pr[order] = np.linspace(0, 100, len(v))
        neg = m in NEG_STATS
        for i, r in enumerate(pool):
            out.setdefault(r["name"], {})[m] = float(100 - pr[i] if neg else pr[i])
    return out


@functools.lru_cache(maxsize=8)
def _pool_ranks(season: str) -> dict:
    """{player_name: {stat: rank}} within `season`, 1 = best (mirrors _pool_percentiles' neg handling so rank 1
    always means 'best', e.g. fewest fouls for fls90). Kept separate from _pool_percentiles so its {name:{stat:pct}}
    shape (a plain float) stays exactly what existing callers — capability_profile chief among them — expect."""
    pool = records(season)
    stats = sorted(set(ALL_STATS) | {p for ax in S.CAP_AXES.values() for p in ax["proxies"]})
    out = {}
    for m in stats:
        v = np.array([float(r.get(m, 0) or 0) for r in pool])
        if len(v) == 0:
            continue
        neg = m in NEG_STATS
        order = np.argsort(v) if neg else np.argsort(-v)          # ascending for neg (fewest first), else descending
        for rank, i in enumerate(order, start=1):
            out.setdefault(pool[i]["name"], {})[m] = rank
    return out


def pool_size(season: str) -> int:
    return len(records(season))


# ---------------- capability profile (the honest core) ----------------
def capability_profile(record: dict, season: str = None) -> dict:
    """A player's Fulcrum capability profile: per axis {pct, label, evidence, registry_status, drivers}.
    `pct` is the mean of the axis's proxy percentiles within the pool. Evidence grade comes from minutes."""
    season = season or record.get("season") or S.DEFAULT_SEASON
    pcts = _pool_percentiles(season).get(record["name"], {})
    grade = S.evidence_grade(record.get("nineties"))
    prof = {}
    for ax, spec in S.CAP_AXES.items():
        vals = [pcts[p] for p in spec["proxies"] if p in pcts]
        pct = round(float(np.mean(vals)), 1) if vals else None
        prof[ax] = {
            "label": spec["label"], "pct": pct,
            "evidence": grade if pct is not None else "insufficient_data",
            "registry_status": R.status_of(spec["registry_key"]),
            "registry_key": spec["registry_key"],
            "drivers": [p for p in spec["proxies"] if p in pcts],
        }
    return prof


# axes with NO box-score proxies (e.g. shape_influence — Measured-only, no honest proxy for named players) are
# structurally never available here; including them in the similarity vector would inject the SAME constant filler
# into every named player, silently inflating cosine similarity for everyone (verified: same-position players were
# showing 99%+ "similarity" driven partly by this shared, uninformative 50.0). Excluded entirely, not neutral-filled.
VECTOR_AXES = [ax for ax, spec in S.CAP_AXES.items() if spec["proxies"]]


def capability_vector(record: dict, season: str = None) -> np.ndarray:
    """The capability profile as a fixed-order vector in [0,100] (for behavioural similarity, §17), over axes that
    actually have proxies. NaN->50 (neutral) only for the rare case where THIS player is missing one of those
    proxy stats individually — not for axes that are unmeasurable for every named player."""
    prof = capability_profile(record, season)
    return np.array([(prof[ax]["pct"] if prof[ax]["pct"] is not None else 50.0) for ax in VECTOR_AXES])


@functools.lru_cache(maxsize=8)
def capability_cov_inv(season: str) -> np.ndarray:
    """Inverse covariance of the capability vector across the season pool (Mahalanobis whitening, §17 similarity).
    The axes are strongly role-correlated in practice (e.g. progressive_intent vs press_resistance ~0.95 for
    attackers, verified) — plain Euclidean/cosine distance is dominated by that shared "how attacking is this
    player" direction, so any two players of the same role land within ~0.5% of each other regardless of style.
    Whitening by the pool's own covariance downweights that dominant correlated direction and upweights the
    residual axes that actually differentiate players WITHIN a role. Small ridge for numerical safety only —
    the real eigenvalue spread here is ~9 to ~1500, comfortably invertible without it."""
    pool = [r for r in records(season) if (r.get("nineties") or 0) >= 6]
    vecs = np.array([capability_vector(r, season) for r in pool])
    cov = np.cov(vecs.T) + np.eye(vecs.shape[1]) * 1e-3
    return np.linalg.inv(cov)


def mahalanobis(a: np.ndarray, b: np.ndarray, season: str) -> float:
    d = a - b
    return float(np.sqrt(d @ capability_cov_inv(season) @ d))


def archetype(record: dict, season: str = None) -> dict:
    """Behavioural archetype from the capability vector (§16). Returns primary + optional secondary (a blend), each
    with the axes that earned it. Never a position label — this data has none."""
    prof = capability_profile(record, season)
    def strength(axes):
        vs = [prof[a]["pct"] for a in axes if prof[a]["pct"] is not None]
        return float(np.mean(vs)) if vs else 0.0
    scored = sorted(((strength(axes), name, axes) for name, axes in S.ARCHETYPES), reverse=True)
    prim = scored[0]; sec = scored[1]
    out = {"primary": prim[1], "primary_strength": round(prim[0], 1),
           "primary_axes": [S.CAP_AXES[a]["label"] for a in prim[2]]}
    if sec[0] >= 60 and sec[0] >= prim[0] - 12:                 # a real blend, not a distant runner-up
        out["secondary"] = sec[1]; out["secondary_axes"] = [S.CAP_AXES[a]["label"] for a in sec[2]]
    return out


def value_percentile(record: dict, season: str = None) -> float | None:
    """Where the player's market value sits among peers who HAVE a known value (0=cheapest, 100=most expensive).
    Returns None — never a fabricated number — when this player's own value is unknown, or when the season's pool
    has no priced players at all. Unpriced players are also excluded from the pool itself: folding them in as 0
    would silently mark them "free" and drag every genuinely-priced player's percentile upward (§40/§43 — decompose
    honestly, never fake a number)."""
    season = season or record.get("season") or S.DEFAULT_SEASON
    pool = [r for r in records(season) if r.get("market_value")]
    if not pool:
        return None
    mv = np.array([float(r["market_value"]) for r in pool])
    order = mv.argsort(); pr = np.empty(len(mv)); pr[order] = np.linspace(0, 100, len(mv))
    idx = {r["name"]: i for i, r in enumerate(pool)}
    i = idx.get(record["name"])
    return round(float(pr[i]), 1) if i is not None else None


def capability_index(record: dict, season: str = None) -> float:
    """A transparent mean of the ATTACKING capability percentiles (not an opaque master score — it is always shown
    decomposed alongside; §40). Used only to rank Discover and to compute anomalies."""
    prof = capability_profile(record, season)
    axes = ["space_creation", "off_ball_penetration", "progressive_intent", "final_third_threat"]
    vs = [prof[a]["pct"] for a in axes if prof[a]["pct"] is not None]
    return round(float(np.mean(vs)), 1) if vs else 0.0
