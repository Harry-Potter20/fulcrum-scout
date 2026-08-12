"""app.services.compare_service — mechanism comparison (spec §8): "these players may produce similar output — do
they produce it through the same mechanisms?" Never a similarity score alone; always the capability matrix +
per-player mechanism language + production context + evidence tiers, so the comparison is auditable (§40/§41).
"""
from __future__ import annotations
from app.data import loaders as D
from app.config import settings as S
from fulcrum import registry as R

PRODUCTION_STATS = [("gls90", "Goals/90"), ("ast90", "Assists/90"), ("sh90", "Shots/90"),
                     ("keypass90", "Key passes/90"), ("prog_pass90", "Prog. passes/90"),
                     ("dribble90", "Dribbles/90"), ("tkl90", "Tackles/90"), ("int90", "Interceptions/90")]


def _mechanism_sentence(name: str, prof: dict, top_n: int = 2) -> str:
    """Mechanism-language description (spec §20): 'creates separation...' not 'high creativity'."""
    ranked = sorted(((prof[ax]["pct"], ax) for ax in S.CAP_AXES if prof[ax]["pct"] is not None), reverse=True)
    if not ranked:
        return f"{name.split()[-1]} has insufficient evidence to describe a mechanism."
    top = ranked[:top_n]
    reg_says = [R.get(S.CAP_AXES[ax]["registry_key"])["say"] for _, ax in top]
    surname = name.split()[-1]
    if len(reg_says) == 1:
        return f"{surname} creates value primarily by: {reg_says[0]}."
    return f"{surname} creates value primarily by: {reg_says[0]}; secondarily: {reg_says[1]}."


def compare(names: list, season: str) -> dict:
    """Full comparison bundle for 2-3 players: capability matrix, mechanism sentence, production, evidence tiers.
    Never collapses to a single similarity number — every row is independently inspectable."""
    records, profiles = {}, {}
    for name in names:
        r = D.records(season)
        rec = next((x for x in r if x["name"] == name), None)
        if rec is None:
            continue
        records[name] = rec
        profiles[name] = D.capability_profile(rec, season)

    matrix = []
    for ax, spec in S.CAP_AXES.items():
        row = {"axis": ax, "label": spec["label"], "registry_key": spec["registry_key"]}
        for name in names:
            p = profiles.get(name, {}).get(ax, {})
            row[name] = p.get("pct")
        matrix.append(row)

    production = []
    for stat, label in PRODUCTION_STATS:
        row = {"stat": stat, "label": label}
        for name in names:
            rec = records.get(name)
            row[name] = round(float(rec.get(stat, 0) or 0), 2) if rec else None
        production.append(row)

    mechanism = {name: _mechanism_sentence(name, profiles[name]) for name in names if name in profiles}

    # pairwise "same output, different mechanism" flag: similar production, divergent top capability
    divergence = None
    if len(names) == 2 and all(n in profiles for n in names):
        a, b = names
        top_a = max(((profiles[a][ax]["pct"] or 0), ax) for ax in S.CAP_AXES)
        top_b = max(((profiles[b][ax]["pct"] or 0), ax) for ax in S.CAP_AXES)
        if top_a[1] != top_b[1]:
            divergence = {"a_leads": S.CAP_AXES[top_a[1]]["label"], "b_leads": S.CAP_AXES[top_b[1]]["label"]}

    return {"names": [n for n in names if n in profiles], "records": records, "profiles": profiles,
            "matrix": matrix, "production": production, "mechanism": mechanism, "divergence": divergence}
