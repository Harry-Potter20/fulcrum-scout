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


# ---------------- percentiles within a pool ----------------
@functools.lru_cache(maxsize=8)
def _pool_percentiles(season: str) -> dict:
    """{player_name: {stat: pct 0..100}} within `season`. Higher stat -> higher pct (fls90 inverted: fewer fouls
    better). Mirrors recruit.season_percentiles; kept here so the app has zero import of the geometry stack."""
    pool = records(season)
    stats = sorted({p for ax in S.CAP_AXES.values() for p in ax["proxies"]})
    out = {}
    for m in stats:
        v = np.array([float(r.get(m, 0) or 0) for r in pool])
        if len(v) == 0:
            continue
        order = v.argsort(); pr = np.empty(len(v)); pr[order] = np.linspace(0, 100, len(v))
        neg = m in ("fls90",)
        for i, r in enumerate(pool):
            out.setdefault(r["name"], {})[m] = float(100 - pr[i] if neg else pr[i])
    return out


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


def capability_vector(record: dict, season: str = None) -> np.ndarray:
    """The capability profile as a fixed-order vector in [0,100] (for behavioural similarity, §17). NaN->50 (neutral)."""
    prof = capability_profile(record, season)
    return np.array([(prof[ax]["pct"] if prof[ax]["pct"] is not None else 50.0) for ax in S.CAP_AXES])


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


def value_percentile(record: dict, season: str = None) -> float:
    """Where the player's market value sits in the pool (0=cheapest, 100=most expensive). Used for VALUE and for the
    market-anomaly signal (§43) — high capability + low value percentile = potentially undervalued."""
    season = season or record.get("season") or S.DEFAULT_SEASON
    pool = records(season)
    mv = np.array([float(r.get("market_value", 0) or 0) for r in pool])
    order = mv.argsort(); pr = np.empty(len(mv)); pr[order] = np.linspace(0, 100, len(mv))
    idx = {r["name"]: i for i, r in enumerate(pool)}
    i = idx.get(record["name"])
    return round(float(pr[i]), 1) if i is not None else 50.0


def capability_index(record: dict, season: str = None) -> float:
    """A transparent mean of the ATTACKING capability percentiles (not an opaque master score — it is always shown
    decomposed alongside; §40). Used only to rank Discover and to compute anomalies."""
    prof = capability_profile(record, season)
    axes = ["space_creation", "off_ball_penetration", "progressive_intent", "final_third_threat"]
    vs = [prof[a]["pct"] for a in axes if prof[a]["pct"] is not None]
    return round(float(np.mean(vs)), 1) if vs else 0.0
