"""app.services.scout_service — the Scout product surface. Pages call these; they never touch the data layer's
internals or a model directly (§49). Everything returns decomposed, explainable structures (§40): FIT/VALUE/UPSIDE/
EVIDENCE, capability profiles, archetypes, behavioural similarity — never one opaque score.
"""
from __future__ import annotations
import numpy as np
from app.data import loaders as D
from app.config import settings as S
from fulcrum import registry as R


def _value_m(r: dict) -> float | None:
    """€M market value, or None (never a fabricated 0.0 = "free") when unknown — §40/§43."""
    mv = r.get("market_value")
    return round(float(mv) / 1e6, 1) if mv else None


def player_index(season: str) -> list:
    """Lightweight index for lists/search: name, league, age, value, capability_index (transparent, shown decomposed)."""
    out = []
    for r in D.records(season):
        out.append({"name": r["name"], "league": r.get("league"), "age": r.get("age"),
                    "value_m": _value_m(r),
                    "nineties": r.get("nineties"), "cap_index": D.capability_index(r, season),
                    "player_id": r.get("player_id"), "club": r.get("club"), "nationality": r.get("nationality"),
                    "nationality_code": r.get("nationality_code"), "nationality_alpha2": r.get("nationality_alpha2"),
                    "photo_url": r.get("photo_url")})
    return out


def get_record(name: str, season: str) -> dict | None:
    for r in D.records(season):
        if r["name"] == name:
            return r
    return None


def scorecard(record: dict, season: str) -> dict:
    """The decomposed headline (§40): FIT (attacking capability), VALUE (cheaper-than-peers), UPSIDE (age-adjusted),
    EVIDENCE (minutes grade). No master score — each is independently explainable. VALUE/UPSIDE are None (never a
    fabricated number, e.g. an assumed age-27) when the player's market_value/age is unknown — §40/§43."""
    cap = D.capability_index(record, season)
    val_pct = D.value_percentile(record, season)
    age = record.get("age")
    upside = float(np.clip(cap + max(0, 24 - age) * 4, 0, 100)) if age else None
    grade = S.evidence_grade(record.get("nineties"))
    ev = {"measured_proxy_high": 82, "estimated": 55, "insufficient_data": 25}.get(grade, 40)
    return {"FIT": round(cap), "VALUE": round(100 - val_pct) if val_pct is not None else None,
            "UPSIDE": round(upside) if upside is not None else None, "EVIDENCE": ev,
            "value_percentile": val_pct}


def get_player(name: str, season: str) -> dict | None:
    """Full player dossier: identity, scorecard, capability profile, archetype, and the 'beyond stats' read (§42/§53)."""
    r = get_record(name, season)
    if r is None:
        return None
    prof = D.capability_profile(r, season)
    arche = D.archetype(r, season)
    return {"record": r, "name": r["name"], "league": r.get("league"), "age": r.get("age"),
            "value_m": _value_m(r), "nineties": r.get("nineties"),
            "foot": r.get("foot"), "height": r.get("height"),
            "club": r.get("club"), "nationality": r.get("nationality"), "nationality_code": r.get("nationality_code"),
            "nationality_alpha2": r.get("nationality_alpha2"),
            "photo_url": r.get("photo_url"), "club_logo_url": r.get("club_logo_url"),
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
                     "value_m": _value_m(r),
                     "priority_pct": pa["pct"], "scorecard": sc,
                     "archetype": D.archetype(r, season)["primary"],
                     "cap_index": D.capability_index(r, season), "value_percentile": vp,
                     "anomaly": round(pa["pct"] - vp, 1) if vp is not None else None,
                     "club": r.get("club"), "nationality": r.get("nationality"),
                     "nationality_code": r.get("nationality_code"), "nationality_alpha2": r.get("nationality_alpha2"),
                    "photo_url": r.get("photo_url")})
    rows.sort(key=lambda x: -x["priority_pct"])
    return rows[:top]


def market_anomalies(season: str, *, min_minutes=10.0, top=20) -> list:
    """High capability + low market value = potentially undervalued (§43). anomaly = cap_index - value_percentile."""
    rows = []
    for r in D.records(season):
        if (r.get("nineties") or 0) < min_minutes:
            continue
        vp = D.value_percentile(r, season)
        if vp is None:                                              # anomaly needs a real value to compare against
            continue
        cap = D.capability_index(r, season)
        rows.append({"name": r["name"], "league": r.get("league"), "age": r.get("age"),
                     "value_m": _value_m(r),
                     "cap_index": cap, "value_percentile": vp, "anomaly": round(cap - vp, 1),
                     "archetype": D.archetype(r, season)["primary"]})
    rows.sort(key=lambda x: -x["anomaly"])
    return rows[:top]


# ---------------- behavioural similarity (§17) ----------------
def similar(name: str, season: str, top=6) -> list:
    """Players most similar in BEHAVIOUR (capability vector), not position/production. Similarity is a
    percentile-normalized MAHALANOBIS distance — not cosine (measures direction only; same-position players share
    a near-identical relative shape almost by construction, so it showed 99%+ "similarity" for any two strikers
    regardless of style) and not plain Euclidean either (treats every axis as equally informative, but the axes
    are strongly role-correlated — e.g. progressive_intent/press_resistance ~0.95 for attackers — so Euclidean
    distance is still dominated by that one shared "how attacking is this player" direction). Mahalanobis whitens
    by the pool's own covariance, downweighting that dominant correlated direction so the residual, more nuanced
    differences actually drive the ranking (verified: Haaland's Euclidean neighbours were just "other elite
    strikers"; Mahalanobis instead surfaces strikers matching his specific relative shape).

    Mahalanobis whitens away the dominant correlated direction on purpose — which is also, largely, the "overall
    quality" direction. So it can rank a much weaker player as a top style-match (verified: a €2.4M MLS forward
    outranked several senior internationals as Haaland's #1 neighbour) — checked this isn't a covariance-estimation
    artifact (Ledoit-Wolf shrinkage on this pool is ~0.003, negligible; the sample covariance is already
    well-estimated at N≈2000 for 6 axes). It's a real result, not noise — style and tier are just different things
    — so `cap_gap` reports the tier difference explicitly (§40) rather than letting style-similarity alone imply
    "viable replacement"."""
    base_r = get_record(name, season)
    if base_r is None:
        return []
    bv = D.capability_vector(base_r, season)
    base_cap = D.capability_index(base_r, season)
    axes = D.VECTOR_AXES
    pool = [r for r in D.records(season) if r["name"] != name and (r.get("nineties") or 0) >= 6]
    if not pool:
        return []
    vecs = [D.capability_vector(r, season) for r in pool]
    dists = np.array([D.mahalanobis(bv, v, season) for v in vecs])
    order = dists.argsort()                                          # nearest first
    sim_pct = np.empty(len(dists)); sim_pct[order] = np.linspace(100, 0, len(dists))
    out = []
    for r, v, sim in zip(pool, vecs, sim_pct):
        shared = sorted(range(len(axes)), key=lambda i: -min(bv[i], v[i]))[:2]
        own_top = sorted(range(len(axes)), key=lambda i: -v[i])[:2]         # the neighbour's OWN signature
        out.append({"name": r["name"], "league": r.get("league"),
                    "value_m": _value_m(r),
                    # 2dp: with a 2000+ player pool the top handful sit within the top ~0.5%, so 1dp visually
                    # collapses genuinely-different ranks (e.g. 100.0/99.9/99.9/99.9) into apparent ties.
                    "similarity": round(float(sim), 2),
                    "cap_gap": round(D.capability_index(r, season) - base_cap, 1),  # tier delta — style ≠ level
                    "shared": [S.CAP_AXES[axes[i]]["label"] for i in shared],
                    "mechanism": [(S.CAP_AXES[axes[i]]["label"], int(v[i])) for i in own_top]})
    out.sort(key=lambda x: -x["similarity"])
    return out[:top]
