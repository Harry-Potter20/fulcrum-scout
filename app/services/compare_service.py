"""app.services.compare_service — mechanism comparison (spec §8): "these players may produce similar output — do
they produce it through the same mechanisms?" Never a similarity score alone; always the capability matrix +
per-player mechanism language + production context + evidence tiers, so the comparison is auditable (§40/§41).

Every production stat carries percentile AND rank within the season pool (not just a raw number) — the same
decision that makes Footverse's stat sheet legible applies here: "14 tackles" means nothing alone, "87th percentile,
rank 12/950" does.
"""
from __future__ import annotations
from app.data import loaders as D
from app.config import settings as S
from fulcrum import registry as R

# categorized production stats (Footverse-style stat-sheet grouping) — every field the ingest schema produces,
# sourced from real Sofascore statistics/overall keys (verified against a live response, jobs/fb_multiseason.py)
STAT_CATEGORIES = [
    ("Attacking", [("gls90", "Goals/90"), ("xg90", "xG/90"), ("sh90", "Shots/90"), ("sot90", "Shots on target/90"),
                   ("finishing", "Finishing (G/xG)"), ("goal_conv_pct", "Goal conversion %"),
                   ("big_ch_missed90", "Big chances missed/90"), ("headed_gls90", "Headed goals/90")]),
    ("Creativity", [("ast90", "Assists/90"), ("xa90", "xA/90"), ("keypass90", "Key passes/90"),
                    ("big_ch_created90", "Big chances created/90"), ("prog_pass90", "Final-third passes/90"),
                    ("pass_pct", "Pass accuracy %"), ("long_ball90", "Long balls/90"),
                    ("crs90", "Crosses/90"), ("cross_pct", "Cross accuracy %")]),
    ("Carrying", [("dribble90", "Dribbles/90"), ("dribble_pct", "Dribble success %"),
                  ("touches90", "Touches/90"), ("dispossessed90", "Dispossessed/90"), ("was_fouled90", "Was fouled/90")]),
    ("Defending", [("tkl90", "Tackles/90"), ("tkl_won_pct", "Tackles won %"), ("int90", "Interceptions/90"),
                   ("clearances90", "Clearances/90"), ("blocks90", "Blocks/90"), ("recoveries90", "Recoveries/90")]),
    ("Duels", [("duels_won90", "Duels won/90"), ("duel_won_pct", "Duels won %"),
              ("aerial_won90", "Aerial duels won/90"), ("aerial_won_pct", "Aerial duels won %"),
              ("dribbled_past90", "Dribbled past/90")]),
    ("Discipline", [("fls90", "Fouls/90"), ("yellow90", "Yellow cards/90"), ("offsides90", "Offsides/90")]),
]
STAT_PRESETS = {"Essentials": ["Attacking", "Creativity", "Defending"],
                "Advanced": ["Attacking", "Creativity", "Carrying", "Defending", "Duels", "Discipline"]}
PLAYER_COLORS = [S.PALETTE["cy"], S.PALETTE["mag"], S.PALETTE["amber"]]


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
    """Full comparison bundle for 2-3 players: capability matrix, mechanism sentence, categorized+ranked production
    stats, evidence tiers, and a colour assignment for the overlay charts. Never collapses to a single similarity
    number — every row is independently inspectable."""
    records, profiles, colors = {}, {}, {}
    pcts, ranks = D._pool_percentiles(season), D._pool_ranks(season)
    n_pool = D.pool_size(season)
    for i, name in enumerate(names):
        r = D.records(season)
        rec = next((x for x in r if x["name"] == name), None)
        if rec is None:
            continue
        records[name] = rec
        profiles[name] = D.capability_profile(rec, season)
        colors[name] = PLAYER_COLORS[i % len(PLAYER_COLORS)]

    matrix = []
    for ax, spec in S.CAP_AXES.items():
        row = {"axis": ax, "label": spec["label"], "registry_key": spec["registry_key"]}
        for name in names:
            p = profiles.get(name, {}).get(ax, {})
            row[name] = p.get("pct")
        matrix.append(row)

    production = []
    for cat, stats in STAT_CATEGORIES:
        cat_rows = []
        for stat, label in stats:
            row = {"stat": stat, "label": label}
            for name in names:
                rec = records.get(name)
                if not rec:
                    row[name] = None
                    continue
                val = round(float(rec.get(stat, 0) or 0), 2)
                p = pcts.get(name, {}).get(stat)
                rk = ranks.get(name, {}).get(stat)
                row[name] = {"value": val, "pct": p, "rank": rk, "n": n_pool}
            cat_rows.append(row)
        production.append({"category": cat, "rows": cat_rows})

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
            "colors": colors, "matrix": matrix, "production": production, "mechanism": mechanism,
            "divergence": divergence, "pool_size": n_pool}


def report_json(r: dict, season: str) -> str:
    """The comparison as a downloadable JSON report — every number traceable to registry tier + pool percentile/rank,
    never just the bare figures (spec §45: 'every claim must map back to a metric or evidence source')."""
    import json
    names = r["names"]
    out = {"season": season, "players": names, "pool_size": r["pool_size"], "mechanism": r["mechanism"],
           "divergence": r["divergence"],
           "capabilities": [{"axis": row["label"], "registry_status": R.get(row["registry_key"])["status"],
                             **{n: row.get(n) for n in names}} for row in r["matrix"]],
           "production": [{"category": cat["category"],
                           "stats": [{"stat": row["label"], **{n: row.get(n) for n in names}} for row in cat["rows"]]}
                          for cat in r["production"]]}
    return json.dumps(out, indent=2, default=str)


def report_csv(r: dict) -> str:
    """A flat CSV of the production stat sheet (one row per stat per player) — the format a scout actually pastes
    into a spreadsheet, vs. the JSON's full nested structure with mechanism/registry context."""
    import csv, io
    names = r["names"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["category", "stat", "player", "value", "percentile", "rank", "pool_size"])
    for cat in r["production"]:
        for row in cat["rows"]:
            for n in names:
                cell = row.get(n)
                if cell:
                    w.writerow([cat["category"], row["label"], n, cell["value"], cell["pct"], cell["rank"], cell["n"]])
    for row in r["matrix"]:
        for n in names:
            if row.get(n) is not None:
                w.writerow(["Capability", row["label"], n, row[n], "", "", ""])
    return buf.getvalue()
