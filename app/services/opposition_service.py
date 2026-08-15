"""app.services.opposition_service — "where can we hurt them?" reads an opponent's existing team capability
profile (club_service.club_profile — same data as the Club tab, nothing new computed) the other way round, and
where the data honestly supports it, bridges straight into the same fit_service.best_fits() mechanism the Club
tab's "Who solves this?" panel already uses.

Honesty constraint: of the 7 capability axes, only `containment` (tackles+interceptions) describes a squad's own
defensive output — the rest describe THEIR attacking players. A low score in an attacking axis is useful context
(they don't create/convert much) but is not something a signing or a game plan "exploits" — so it is surfaced as
description only, never forced into an invented recruiting angle. `press_resistance` sits in between: it's a
player trait, but a squad-wide weakness in it is a real, actionable pressing-trigger insight, not a recruit gap.
"""
from __future__ import annotations
from app.services import club_service as club
from app.config import settings as S

# The only axes with a defensible "how we exploit this" story, and what that story is.
# containment: weak tackling/interceptions -> ball-carriers and dribblers find space against this squad.
# press_resistance: this squad's players lose the ball more under pressure -> a pressing trigger, not a recruit gap.
EXPLOIT_AXES = {
    "containment": {
        "priorities": ["off_ball_penetration", "progressive_intent"],
        "insight": "Weak tackling and interception numbers as a group — ball-carriers and dribblers should find "
                   "space against this defence.",
        "mode": "recruit",
    },
    "press_resistance": {
        "priorities": None,
        "insight": "This squad loses the ball more often than most under pressure — a higher press is likely to "
                   "force turnovers here, independent of who you field.",
        "mode": "press_trigger",
    },
}


def opponent_list(season: str) -> list:
    return club.club_list(season)


def opponent_report(opponent: str, season: str, top: int = 10) -> dict:
    """Full opponent read: team profile (radar + weakest/strongest axis, same shape as club_profile), real
    Measured structural exposure when we have a tracked match for them, and the exploit angle — a defensible
    priority-axis bridge into Tactical Fit's candidate search, only when the data supports one."""
    prof = club.club_profile(opponent, season)
    base = {"opponent": opponent, "season": season, "n_players": prof["n_players"], "measured": prof.get("measured")}
    if not prof.get("enough_for_profile"):
        return {**base, "viable": False,
                "reason": f"squad sample too small (need ≥{club.MIN_SQUAD} players with capability data)"}

    weak_axes = sorted((ax for ax in S.CAP_AXES if prof["team_profile"][ax]["mean"] is not None),
                       key=lambda ax: prof["team_profile"][ax]["mean"])
    bottom3 = weak_axes[:3]

    exploit_axis = next((ax for ax in bottom3 if ax in EXPLOIT_AXES), None)
    if exploit_axis:
        spec = EXPLOIT_AXES[exploit_axis]
        angle = {"axis": exploit_axis, "label": S.CAP_AXES[exploit_axis]["label"],
                 "mean": prof["team_profile"][exploit_axis]["mean"], "mode": spec["mode"], "insight": spec["insight"]}
    else:
        wk = weak_axes[0] if weak_axes else None
        angle = {"axis": wk, "label": S.CAP_AXES[wk]["label"] if wk else None,
                 "mean": prof["team_profile"][wk]["mean"] if wk else None, "mode": "context_only",
                 "insight": (f"This squad's lowest measured axis is {S.CAP_AXES[wk]['label']} — that's their own "
                            f"attacking output, not a defensive trait this data can game-plan against." if wk else
                            "No standout weak axis in this data.")}

    candidates, target_labels = [], []
    if angle["mode"] == "recruit":
        from app.services import fit_service as fit
        priorities = EXPLOIT_AXES[angle["axis"]]["priorities"]
        target_labels = [S.CAP_AXES[a]["label"] for a in priorities]
        exclude = {r["name"] for r in prof["players"]}
        raw = fit.best_fits(season, priorities, min_minutes=8.0, max_age=40, max_value_m=1e9, top=top + len(exclude))
        candidates = [c for c in raw if c["name"] not in exclude][:top]

    return {**base, "viable": True, "team_profile": prof["team_profile"], "weakest_axis": prof["weakest_axis"],
           "strongest_axis": prof["strongest_axis"], "archetype_mix": prof["archetype_mix"],
           "angle": angle, "target_labels": target_labels, "candidates": candidates}
