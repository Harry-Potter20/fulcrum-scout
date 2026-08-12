"""app.services.fit_service — Tactical Fit (§20/§21/§31): the user states a tactical NEED (prioritised capabilities),
Scout returns players whose measured capability addresses it, every recommendation decomposable into WHY (§41).

This is the ships-now Diagnose->Match surface (recruit.py's philosophy) over the named-player universe. It does NOT
claim signing impact — that's the gated counterfactual (registry: counterfactual_signing_impact = unproven).
"""
from __future__ import annotations
import numpy as np
from app.data import loaders as D
from app.config import settings as S


# named tactical problems -> the capability priority stack that addresses them (§31)
NEEDS = {
    "Break a low block":          ["space_creation", "off_ball_penetration", "progressive_intent"],
    "Progress vs a high press":   ["press_resistance", "progressive_intent", "space_creation"],
    "Add depth / run in behind":  ["off_ball_penetration", "final_third_threat"],
    "Create chances from wide":   ["space_creation", "off_ball_penetration"],
    "Add defensive containment":  ["containment", "press_resistance"],
    "More final-third threat":    ["final_third_threat", "off_ball_penetration"],
}


def evaluate(name: str, season: str, priorities: list[str]) -> dict:
    """Fit of ONE player against a prioritised capability need. Weighted (rank-decayed) mean of the relevant axis
    percentiles, returned WITH the per-axis breakdown so the number is auditable (§40/§41)."""
    r = D.records(season)
    rec = next((x for x in r if x["name"] == name), None)
    if rec is None:
        return {"error": "not found"}
    prof = D.capability_profile(rec, season)
    weights = [1.0 / (i + 1) for i in range(len(priorities))]      # 1, 1/2, 1/3 ...
    num = den = 0.0
    breakdown = []
    for w, ax in zip(weights, priorities):
        p = prof.get(ax, {}).get("pct")
        if p is None:
            breakdown.append({"axis": S.CAP_AXES[ax]["label"], "pct": None, "weight": round(w, 2)})
            continue
        num += w * p; den += w
        breakdown.append({"axis": S.CAP_AXES[ax]["label"], "pct": p, "weight": round(w, 2)})
    fit = round(num / den, 1) if den else None
    return {"name": name, "fit": fit, "breakdown": breakdown,
            "primary_reason": breakdown[0]["axis"] if breakdown else None}


def best_fits(season: str, priorities: list[str], *, min_minutes=8.0, max_age=40, max_value_m=1e9,
              leagues=None, top=15) -> list:
    """Rank the universe against a tactical need. Each recommendation is decomposable (breakdown + primary reason)."""
    out = []
    for rec in D.records(season):
        if (rec.get("nineties") or 0) < min_minutes or (rec.get("age") or 0) > max_age:
            continue
        if float(rec.get("market_value", 0) or 0) / 1e6 > max_value_m:
            continue
        if leagues and rec.get("league") not in leagues:
            continue
        ev = evaluate(rec["name"], season, priorities)
        if ev.get("fit") is None:
            continue
        ev.update({"league": rec.get("league"), "age": rec.get("age"),
                   "value_m": round(float(rec.get("market_value", 0) or 0) / 1e6, 1),
                   "archetype": D.archetype(rec, season)["primary"]})
        out.append(ev)
    out.sort(key=lambda x: -x["fit"])
    return out[:top]
