"""fulcrum.products.scout — the ONE scouting surface.

Composes the model-derived layers into a single call, instead of making users wire the functions themselves:
  · stat-model      : archetypes + over/under-performance residuals + latent style + a GENERATED profile per player
  · semantic layer  : style comparables ("plays like…", cheapest equivalents)
  · translation     : cross-league production projection (football; validated +7.9% OOS)
  · estimated tier  : positional advancement for stat-only players, IF a spatial label is present (tagged, conf ~0.57)
Sport-agnostic — the engine is one; only the feature set + vocabulary swap (mirrors the platform's driver registry).
Geometry (off-ball space/danger from 360/tracking) composes separately via `fulcrum.freeze_frame_to_state` -> services,
because it needs spatial data this stat-tier call doesn't require. Products compose; the model is never modified.
"""
import fulcrum

FEAT = {"football": ["gls90", "ast90", "sh90", "sot90", "finishing", "tkl90", "int90", "crs90", "fls90"],
        "basketball": ["pts", "ast", "reb", "stl", "blk", "fg_pct", "fg3_pct", "tov", "usg"]}
OUTPUTS = {"football": ("gls90", "ast90"), "basketball": ("pts", "ast")}


def scout(records, sport="football", target_league=None, spatial_key=None, traits=None, geometry=None, top=20):
    """ONE scouting report from player-season records — the coherent surface that maps every Scout capability.
    `sport` selects features + vocabulary. `target_league` (football) adds the moneyball translation board.
    `spatial_key` switches on the estimated-positional tier. `traits`/`geometry` are optional {name: {...}} of
    precomputed latent traits / off-ball geometry (from the 360/tracking pipelines) merged per player. The report's
    `capabilities` map states which tiers are present and their epistemic status — so a UI can render exactly what's
    available and nothing it can't back up."""
    feat = FEAT.get(sport); outputs = OUTPUTS.get(sport)
    prof = fulcrum.player_profiles(records, feat=feat, outputs=outputs, sport=sport, top_compare=3)
    caps = {"profiles": "validated: model-derived (residuals + latent style + generated read), all players"}
    report = {"sport": sport, "n_players": prof.get("n", 0), "archetypes": prof.get("archetypes", {}),
              "profiles": prof.get("players", [])[:top], "capabilities": caps,
              "boundary": "stat-tier recombines existing metrics; spatial signal needs the geometry tier"}
    if sport == "football" and target_league:
        try:
            report["translation"] = fulcrum.recruit(records, target_league, top=top).get("candidates")
            caps["moneyball"] = "validated: cross-league strength network + translation (+7.9% OOS)"
        except Exception as e:
            report["translation_error"] = str(e)[:80]
    if spatial_key and any(r.get(spatial_key) is not None for r in records):
        est = fulcrum.fit_spatial_estimator(records, feat, target_key=spatial_key)
        if est:
            byname = {e["player"]: e for e in fulcrum.estimate_spatial(records, est)}
            for p in report["profiles"]:
                if p["player"] in byname:
                    p["estimated_advancement"] = byname[p["player"]]["estimated_advancement"]
            caps["estimated_positional"] = f"estimated only (positional), held-out conf {est['confidence']}"
    if traits:                                     # validated latent traits (progressive_intent, press_resistance, ...)
        for p in report["profiles"]:
            if p["player"] in traits:
                p["traits"] = traits[p["player"]]
        caps["latent_traits"] = "validated: progressive_intent / press_resistance / off_ball_penetration_rate"
    if geometry:                                   # off-ball geometry from 360/tracking
        for p in report["profiles"]:
            if p["player"] in geometry:
                p["geometry"] = geometry[p["player"]]
        caps["geometry"] = "off-ball space/danger from freeze-frame/tracking (coverage-tagged)"
    return report
