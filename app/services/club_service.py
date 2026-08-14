"""app.services.club_service — team-level aggregation (§40 still applies: decomposed, never one score). Groups the
existing PER-PLAYER capability profiles + production stats by `club` (Transfermarkt-sourced, ~98% populated as of
this session's enrichment run — this service was not viable before that data existed).

Honesty note: the underlying player pool is TOPN goal+assist contributors per league-season (jobs/fb_multiseason.py),
not full squads — median club representation in this data is ~4 players, not 25. A club's aggregate is therefore an
attack-biased SAMPLE of its productive players, not a roster. `n_players` is always surfaced so nobody mistakes a
4-player sample for a full-squad summary; call sites should gate on it (see MIN_SQUAD) rather than hide it.
"""
from __future__ import annotations
import numpy as np
from app.data import loaders as D
from app.config import settings as S
from app.services import scout_service as scout

_PLACEHOLDER_CLUBS = {"Without Club", "Retired", None, ""}
MIN_SQUAD = 3   # below this, a "team profile" is just 1-2 players wearing a club badge — show n_players, not a radar


def _real_club(name: str) -> bool:
    return name not in _PLACEHOLDER_CLUBS


def club_list(season: str) -> list:
    """{name, n_players} for every real club with at least one player in `season`, sorted by squad size desc —
    the clubs with enough representation to say anything meaningful surface first, but nothing is hidden."""
    from collections import Counter
    counts = Counter(r["club"] for r in D.records(season) if _real_club(r.get("club")))
    return sorted(({"name": c, "n_players": n} for c, n in counts.items()), key=lambda x: -x["n_players"])


def club_squad(club: str, season: str) -> list:
    return [r for r in D.records(season) if r.get("club") == club]


def club_profile(club: str, season: str) -> dict:
    """Team aggregate: mean capability profile across the squad sample (the 'team style' radar), archetype mix,
    and per-axis min/max to show spread (a team is not one number even in aggregate — §40). `n_players` and
    `enough_for_profile` (>=MIN_SQUAD) are always present so a thin sample is visible, not silently averaged away."""
    squad = club_squad(club, season)
    n = len(squad)
    if n == 0:
        return {"club": club, "season": season, "n_players": 0, "enough_for_profile": False}

    axes = list(S.CAP_AXES)
    per_axis = {ax: [] for ax in axes}
    archetypes = {}
    rows = []
    for r in squad:
        prof = D.capability_profile(r, season)
        arche = D.archetype(r, season)
        archetypes[arche["primary"]] = archetypes.get(arche["primary"], 0) + 1
        row = {"name": r["name"], "cap_index": D.capability_index(r, season), "archetype": arche["primary"],
              "nineties": r.get("nineties"), "age": r.get("age"), "photo_url": r.get("photo_url"),
              "club": r.get("club"), "league": r.get("league"), "nationality": r.get("nationality"),
              "nationality_alpha2": r.get("nationality_alpha2"), "value_m": scout._value_m(r),
              "scorecard": scout.scorecard(r, season)}
        for ax in axes:
            pct = prof[ax]["pct"]
            row[ax] = pct
            if pct is not None:
                per_axis[ax].append(pct)
        rows.append(row)
    rows.sort(key=lambda x: -(x["cap_index"] or 0))

    team_profile = {}
    for ax in axes:
        vals = per_axis[ax]
        team_profile[ax] = {
            "label": S.CAP_AXES[ax]["label"],
            "mean": round(float(np.mean(vals)), 1) if vals else None,
            "min": round(float(np.min(vals)), 1) if vals else None,
            "max": round(float(np.max(vals)), 1) if vals else None,
            "n": len(vals),
        }
    weak_axes = sorted((ax for ax in axes if team_profile[ax]["mean"] is not None),
                       key=lambda ax: team_profile[ax]["mean"])
    strong_axes = list(reversed(weak_axes))

    return {
        "club": club, "season": season, "n_players": n, "enough_for_profile": n >= MIN_SQUAD,
        "team_profile": team_profile,
        "weakest_axis": weak_axes[0] if weak_axes else None,
        "strongest_axis": strong_axes[0] if strong_axes else None,
        "archetype_mix": sorted(archetypes.items(), key=lambda kv: -kv[1]),
        "players": rows,
    }
