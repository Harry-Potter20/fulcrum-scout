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
STRUCT_REPO = "Chucks90/football-gsr-data"


def _real_club(name: str) -> bool:
    return name not in _PLACEHOLDER_CLUBS


def _hf_token():
    import os
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        return None


def _structural_index() -> dict:
    """{team_name: {match_id, opponent, ...structural_exposure fields}} scanned from every
    gsr/structural_exposure_<match_id>.json in the bucket — real per-team tracked-match findings, covering only
    the specific clubs we've actually tracked (currently: whichever teams appear in SkillCorner's open-data
    sample + any match we've run youtube_flip/footballia on with a long enough window). Never extrapolated to
    clubs we haven't tracked."""
    from huggingface_hub import HfApi, hf_hub_download
    import json
    tok = _hf_token()
    try:
        files = HfApi(token=tok).list_repo_files(STRUCT_REPO, repo_type="dataset")
    except Exception:
        return {}
    out = {}
    for f in files:
        if not (f.startswith("gsr/structural_exposure_") and f.endswith(".json")):
            continue
        try:
            p = hf_hub_download(STRUCT_REPO, f, repo_type="dataset", token=tok)
            d = json.load(open(p))
        except Exception:
            continue
        home, away = d.get("home_team"), d.get("away_team")
        by_team = d.get("by_team", {})
        if home and "0.0" in by_team:
            out[home] = {**by_team["0.0"], "match_id": d["match_id"], "opponent": away}
        if away and "1.0" in by_team:
            out[away] = {**by_team["1.0"], "match_id": d["match_id"], "opponent": home}
    return out


def club_list(season: str) -> list:
    """{name, n_players, measured} for every real club with a player in `season` OR real tracked-match data —
    the latter (measured=True, n_players may be 0) surfaces clubs we have genuine Measured insight for even when
    they have no box-score representation at all (e.g. leagues outside the 18 this database covers)."""
    from collections import Counter
    counts = Counter(r["club"] for r in D.records(season) if _real_club(r.get("club")))
    out = {c: {"name": c, "n_players": n, "measured": False} for c, n in counts.items()}
    for team in _structural_index():
        if team in out:
            out[team]["measured"] = True
        else:
            out[team] = {"name": team, "n_players": 0, "measured": True}
    return sorted(out.values(), key=lambda x: (-x["n_players"], not x["measured"]))


def club_squad(club: str, season: str) -> list:
    return [r for r in D.records(season) if r.get("club") == club]


def club_profile(club: str, season: str) -> dict:
    """Team aggregate: mean capability profile across the squad sample (the 'team style' radar), archetype mix,
    and per-axis min/max to show spread (a team is not one number even in aggregate — §40). `n_players` and
    `enough_for_profile` (>=MIN_SQUAD) are always present so a thin sample is visible, not silently averaged away.
    `measured` carries real tracked-match structural_exposure when we have it for this specific club — independent
    of whether the club has any box-score representation at all."""
    measured = _structural_index().get(club)
    squad = club_squad(club, season)
    n = len(squad)
    if n == 0:
        return {"club": club, "season": season, "n_players": 0, "enough_for_profile": False, "measured": measured}

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
        "measured": measured,
    }
