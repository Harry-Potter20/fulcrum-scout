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


def club_fit(club: str, season: str, top: int = 10, max_value_m: float = 1e9) -> dict:
    """Club-conditioned Tactical Fit (§20/§21/§31 applied to a squad's actual gap, not a generic stated need):
    auto-derives the priority stack from the club's OWN weakest capability axes — the gap a signing would target —
    excludes players already at the club, and reports each candidate's fit as a DELTA against the club's current
    squad average on those same axes. Not just 'FIT 91' but '+18 vs YOUR current space_creation average', so the
    recommendation is checkable against this specific squad, not some generic baseline."""
    from app.services import fit_service as fit
    prof = club_profile(club, season)
    if not prof.get("enough_for_profile"):
        return {"club": club, "season": season, "viable": False,
                "reason": f"squad sample too small (need ≥{MIN_SQUAD} players with capability data)"}

    ranked_weak = sorted((ax for ax in S.CAP_AXES if prof["team_profile"][ax]["mean"] is not None),
                         key=lambda ax: prof["team_profile"][ax]["mean"])
    axes = ranked_weak[:2] if len(ranked_weak) > 1 else ranked_weak
    if not axes:
        return {"club": club, "season": season, "viable": False, "reason": "no capability data to target"}

    squad_names = {r["name"] for r in prof["players"]}
    raw = fit.best_fits(season, axes, min_minutes=8.0, max_age=40, max_value_m=max_value_m, top=top + len(squad_names))
    candidates = [c for c in raw if c["name"] not in squad_names][:top]

    label_to_axis = {S.CAP_AXES[a]["label"]: a for a in axes}
    for c in candidates:
        deltas = []
        for b in c["breakdown"]:
            ax_key = label_to_axis.get(b["axis"])
            team_mean = prof["team_profile"].get(ax_key, {}).get("mean") if ax_key else None
            delta = round(b["pct"] - team_mean, 1) if (b["pct"] is not None and team_mean is not None) else None
            deltas.append({"axis": b["axis"], "pct": b["pct"], "team_mean": team_mean, "delta": delta})
        c["deltas"] = deltas

    return {"club": club, "season": season, "viable": True,
           "target_axes": [S.CAP_AXES[a]["label"] for a in axes], "candidates": candidates}


# ---------------- formation optimization (§40: decomposed, every role traces to a specific player + fit) ----------
# Deliberately ATTACKING shapes, not full 11-man formations: this database has no goalkeeper signal at all and is
# itself attack-biased (top goal+assist contributors per league, jobs/fb_multiseason.py) — optimizing a defensive
# shape would be pretending to data we don't have. This optimizes the part that's honestly measurable: given the
# attacking players a club actually has, which front-line shape fits them best. Roles are capability-priority
# stacks (same pattern as fit_service.NEEDS), not position labels — this data has no real position field either.
SHAPES = {
    "4-3-3 (front five)": [
        ("LW", ["space_creation", "off_ball_penetration"]),
        ("ST", ["final_third_threat", "off_ball_penetration"]),
        ("RW", ["space_creation", "off_ball_penetration"]),
        ("CM (creator)", ["progressive_intent", "space_creation"]),
        ("CM (carrier)", ["press_resistance", "progressive_intent"]),
    ],
    "4-2-3-1 (front four)": [
        ("CAM", ["space_creation", "progressive_intent"]),
        ("LW", ["space_creation", "off_ball_penetration"]),
        ("RW", ["space_creation", "off_ball_penetration"]),
        ("ST", ["final_third_threat", "off_ball_penetration"]),
    ],
    "4-4-2 (front four)": [
        ("ST", ["final_third_threat", "off_ball_penetration"]),
        ("ST (partner)", ["off_ball_penetration", "final_third_threat"]),
        ("LW", ["space_creation", "off_ball_penetration"]),
        ("RW", ["space_creation", "off_ball_penetration"]),
    ],
    "3-5-2 (front four)": [
        ("ST", ["final_third_threat", "off_ball_penetration"]),
        ("ST (partner)", ["off_ball_penetration", "final_third_threat"]),
        ("CM (creator)", ["progressive_intent", "space_creation"]),
        ("CM (carrier)", ["press_resistance", "progressive_intent"]),
    ],
}


def _role_fit(prof: dict, priorities: list) -> float:
    """Rank-decayed weighted mean of the relevant axis percentiles — the same formula as fit_service.evaluate(),
    inlined here to keep club_service free of a service->service import cycle."""
    weights = [1.0 / (i + 1) for i in range(len(priorities))]
    num = den = 0.0
    for w, ax in zip(weights, priorities):
        p = prof.get(ax, {}).get("pct")
        if p is not None:
            num += w * p; den += w
    return (num / den) if den else 0.0


def formation_fit(club: str, season: str, min_nineties: float = 8.0) -> dict:
    """Assignment-problem formation optimizer over a club's attacking players (see SHAPES docstring for scope).
    For each candidate shape, solves the optimal one-to-one player-role assignment (Hungarian algorithm,
    scipy.optimize.linear_sum_assignment) maximising total fit, then ranks shapes by mean role fit. Every number
    traces to a specific player/role/fit triple — never a single 'best formation' score with nothing to check."""
    from scipy.optimize import linear_sum_assignment
    squad = club_squad(club, season)
    eligible = [r for r in squad if (r.get("nineties") or 0) >= min_nineties]
    if len(eligible) < 4:
        return {"club": club, "season": season, "viable": False,
                "reason": f"only {len(eligible)} player(s) with ≥{min_nineties} 90s — need at least 4 for the "
                          f"smallest shape"}

    profiles = {r["name"]: D.capability_profile(r, season) for r in eligible}
    names = list(profiles)

    results = []
    for shape_name, roles in SHAPES.items():
        if len(names) < len(roles):
            continue
        cost = np.zeros((len(names), len(roles)))
        for i, nm in enumerate(names):
            for j, (_, priorities) in enumerate(roles):
                cost[i, j] = -_role_fit(profiles[nm], priorities)     # negate: solver minimizes cost
        row_idx, col_idx = linear_sum_assignment(cost)
        assignment, total = [], 0.0
        for i, j in zip(row_idx, col_idx):
            fit = -cost[i, j]
            total += fit
            assignment.append({"role": roles[j][0], "player": names[i], "fit": round(fit, 1)})
        assignment.sort(key=lambda a: -a["fit"])
        results.append({"shape": shape_name, "total_fit": round(total, 1),
                        "mean_fit": round(total / len(roles), 1), "assignment": assignment})
    results.sort(key=lambda r: -r["mean_fit"])
    return {"club": club, "season": season, "viable": bool(results), "shapes": results, "n_eligible": len(eligible)}
