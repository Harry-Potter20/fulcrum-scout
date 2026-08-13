"""app.services.scout_service — the Scout product surface. Pages call these; they never touch the data layer's
internals or a model directly (§49). Everything returns decomposed, explainable structures (§40): FIT/VALUE/UPSIDE/
EVIDENCE, capability profiles, archetypes, behavioural similarity — never one opaque score.
"""
from __future__ import annotations
import numpy as np
from app.data import loaders as D
from app.config import settings as S
from fulcrum import registry as R


def player_index(season: str) -> list:
    """Lightweight index for lists/search: name, league, age, value, capability_index (transparent, shown decomposed)."""
    out = []
    for r in D.records(season):
        out.append({"name": r["name"], "league": r.get("league"), "age": r.get("age"),
                    "value_m": round(float(r.get("market_value", 0) or 0) / 1e6, 1),
                    "nineties": r.get("nineties"), "cap_index": D.capability_index(r, season),
                    "player_id": r.get("player_id")})
    return out


def get_record(name: str, season: str) -> dict | None:
    for r in D.records(season):
        if r["name"] == name:
            return r
    return None


def scorecard(record: dict, season: str) -> dict:
    """The decomposed headline (§40): FIT (attacking capability), VALUE (cheaper-than-peers), UPSIDE (age-adjusted),
    EVIDENCE (minutes grade). No master score — each is independently explainable."""
    cap = D.capability_index(record, season)
    val_pct = D.value_percentile(record, season)
    age = record.get("age") or 27
    upside = float(np.clip(cap + max(0, 24 - age) * 4, 0, 100))     # young + capable -> upside
    grade = S.evidence_grade(record.get("nineties"))
    ev = {"measured_proxy_high": 82, "estimated": 55, "insufficient_data": 25}.get(grade, 40)
    return {"FIT": round(cap), "VALUE": round(100 - val_pct), "UPSIDE": round(upside), "EVIDENCE": ev,
            "value_percentile": val_pct}


def get_player(name: str, season: str) -> dict | None:
    """Full player dossier: identity, scorecard, capability profile, archetype, and the 'beyond stats' read (§42/§53)."""
    r = get_record(name, season)
    if r is None:
        return None
    prof = D.capability_profile(r, season)
    arche = D.archetype(r, season)
    return {"record": r, "name": r["name"], "league": r.get("league"), "age": r.get("age"),
            "value_m": round(float(r.get("market_value", 0) or 0) / 1e6, 1), "nineties": r.get("nineties"),
            "foot": r.get("foot"), "height": r.get("height"),
            "scorecard": scorecard(r, season), "capabilities": prof, "archetype": arche,
            "beyond_stats": _beyond_stats(prof)}


def _beyond_stats(prof: dict) -> list:
    """The capabilities a box score does NOT record, highest first — the product's 'aha' (§42/§53). Each carries the
    METHOD's registry tier so we never over-claim."""
    diff_axes = ["space_creation", "off_ball_penetration", "progressive_intent", "press_resistance"]
    rows = []
    for ax in diff_axes:
        p = prof.get(ax)
        if p and p["pct"] is not None:
            reg = R.get(p["registry_key"])
            rows.append({"axis": ax, "label": p["label"], "pct": p["pct"],
                         "tier": reg["tier"]["label"], "tier_badge": reg["tier"]["badge"],
                         "metric": reg["metric"], "say": reg["say"], "evidence": p["evidence"]})
    return sorted(rows, key=lambda x: -x["pct"])


# ---------------- Discover (§27/§19/§43) ----------------
def discover(season: str, *, min_minutes=8.0, max_age=40, max_value_m=1e9, leagues=None,
             priority_axis="space_creation", top=40) -> list:
    """Find players by CAPABILITY, not production totals. Ranked by the priority capability; each row decomposed."""
    rows = []
    for r in D.records(season):
        if (r.get("nineties") or 0) < min_minutes:
            continue
        if (r.get("age") or 0) > max_age:
            continue
        if float(r.get("market_value", 0) or 0) / 1e6 > max_value_m:
            continue
        if leagues and r.get("league") not in leagues:
            continue
        prof = D.capability_profile(r, season)
        pa = prof.get(priority_axis, {})
        if pa.get("pct") is None:
            continue
        sc = scorecard(r, season)
        vp = D.value_percentile(r, season)
        rows.append({"name": r["name"], "league": r.get("league"), "age": r.get("age"),
                     "value_m": round(float(r.get("market_value", 0) or 0) / 1e6, 1),
                     "priority_pct": pa["pct"], "scorecard": sc,
                     "archetype": D.archetype(r, season)["primary"],
                     "cap_index": D.capability_index(r, season), "value_percentile": vp,
                     "anomaly": round(pa["pct"] - vp, 1)})
    rows.sort(key=lambda x: -x["priority_pct"])
    return rows[:top]


def market_anomalies(season: str, *, min_minutes=10.0, top=20) -> list:
    """High capability + low market value = potentially undervalued (§43). anomaly = cap_index - value_percentile."""
    rows = []
    for r in D.records(season):
        if (r.get("nineties") or 0) < min_minutes:
            continue
        cap = D.capability_index(r, season)
        vp = D.value_percentile(r, season)
        rows.append({"name": r["name"], "league": r.get("league"), "age": r.get("age"),
                     "value_m": round(float(r.get("market_value", 0) or 0) / 1e6, 1),
                     "cap_index": cap, "value_percentile": vp, "anomaly": round(cap - vp, 1),
                     "archetype": D.archetype(r, season)["primary"]})
    rows.sort(key=lambda x: -x["anomaly"])
    return rows[:top]


# ---------------- behavioural similarity (§17) ----------------
def similar(name: str, season: str, top=6) -> list:
    """Players most similar in BEHAVIOUR (capability vector), not position/production. Explains the similarity via the
    axes they most share — the product's differentiated similarity (§17)."""
    base_r = get_record(name, season)
    if base_r is None:
        return []
    bv = D.capability_vector(base_r, season)
    axes = list(S.CAP_AXES)
    out = []
    for r in D.records(season):
        if r["name"] == name or (r.get("nineties") or 0) < 6:
            continue
        v = D.capability_vector(r, season)
        cos = float(np.dot(bv, v) / (np.linalg.norm(bv) * np.linalg.norm(v) + 1e-9))
        shared = sorted(range(len(axes)), key=lambda i: -min(bv[i], v[i]))[:2]
        own_top = sorted(range(len(axes)), key=lambda i: -v[i])[:2]         # the neighbour's OWN signature
        out.append({"name": r["name"], "league": r.get("league"),
                    "value_m": round(float(r.get("market_value", 0) or 0) / 1e6, 1),
                    "similarity": round(cos * 100, 1),
                    "shared": [S.CAP_AXES[axes[i]]["label"] for i in shared],
                    "mechanism": [(S.CAP_AXES[axes[i]]["label"], int(v[i])) for i in own_top]})
    out.sort(key=lambda x: -x["similarity"])
    return out[:top]
