"""app.config.settings — one place for the product's visual identity, the season pool, and the capability model.

The capability model is the honest core (app_build.md §9): each Scout capability axis is estimated from a set of
per-90 PRODUCTION proxies, because the named-player universe here has box-score stats, not per-player tracking.
So the axis carries TWO independent facts, never conflated (§36):
  - the METHOD's scientific tier — read from fulcrum.registry (`registry_key`)   [is the capability itself validated?]
  - THIS player's evidence grade — Measured / Estimated / Insufficient, from minutes + data source   [§39]
When a player has geometry (tracking), the same axis switches from Estimated(proxy) to Measured(geometry). The UI
shows both. Nothing is filled with artificial confidence: Unknown ≠ zero.
"""
from __future__ import annotations

# ---- data ----
RECORDS_REPO = "Chucks90/football-sofascore-data"
RECORDS_FILE = "football_player_seasons.json"
SEASONS = ["25/26", "24/25", "23/24"]
DEFAULT_SEASON = "25/26"

# ---- visual identity: restrained scientific cyberpunk (§24/§47) — dark, technical, legible, one accent + one signal ----
PALETTE = {
    "bg": "#070a10", "panel": "#0b1420", "panel2": "#0e1a28", "line": "#16324a",
    "cy": "#22d3ee",   # primary accent (Fulcrum / measured)
    "mag": "#ff2d78",  # counterfactual / secondary signal
    "amber": "#ffb445", "danger": "#ff5c3a", "good": "#3ddc97",
    "tx": "#d6e6f2", "mut": "#5f7183", "hi": "#eaf4fb", "dim": "#54677d",
}

# ---- capability axes: label, per-90 proxies, and the registry key whose tier governs the METHOD ----
# `hi=True` means higher stat -> higher capability. Percentile is the MEAN of the proxies' within-pool percentiles.
CAP_AXES = {
    "space_creation":      {"label": "Space creation",      "registry_key": "space_creation",
                            "proxies": ["keypass90", "crs90", "ast90"], "hi": True},
    "off_ball_penetration":{"label": "Off-ball penetration", "registry_key": "off_ball_penetration",
                            "proxies": ["sh90", "sot90", "dribble90"], "hi": True},
    "progressive_intent":  {"label": "Progressive intent",  "registry_key": "progressive_intent",
                            "proxies": ["prog_pass90", "dribble90", "keypass90"], "hi": True},
    "final_third_threat":  {"label": "Danger creation",     "registry_key": "danger",
                            "proxies": ["gls90", "sot90", "finishing"], "hi": True},
    "press_resistance":    {"label": "Press resistance",    "registry_key": "press_resistance",
                            "proxies": ["dribble90", "prog_pass90"], "hi": True},
    "containment":         {"label": "Defensive containment", "registry_key": "containment",
                            "proxies": ["tkl90", "int90"], "hi": True},
}

# ---- archetypes emerge from the capability vector (§16), never from a position label (there is none in this data) ----
# each: the 1-2 axes that must be high, and a human name. First matching (by combined strength) wins; blends reported.
ARCHETYPES = [
    ("Space Creator",        ["space_creation", "off_ball_penetration"]),
    ("Depth Attacker",       ["off_ball_penetration", "final_third_threat"]),
    ("Progressive Carrier",  ["progressive_intent", "press_resistance"]),
    ("Half-Space Connector", ["space_creation", "progressive_intent"]),
    ("Finisher",             ["final_third_threat"]),
    ("Containment Defender", ["containment"]),
    ("Pressure Escapist",    ["press_resistance", "progressive_intent"]),
]

# evidence grade from minutes played (nineties). Unknown ≠ zero — low minutes = low certainty, shown, not hidden (§39).
def evidence_grade(nineties: float) -> str:
    if nineties is None:
        return "insufficient_data"
    if nineties >= 18:
        return "measured_proxy_high"     # plenty of production to estimate from
    if nineties >= 7:
        return "estimated"
    return "insufficient_data"

# Semantic colour law (§9): green=validated/measured · amber=estimated/uncertain · magenta=research/counterfactual
# only · grey=insufficient. Estimated is amber (grounded but uncertain), never magenta (which means "research preview").
EVIDENCE_LABEL = {
    "measured_proxy_high": ("Estimated · high sample",   "b-face"),   # amber
    "estimated":           ("Estimated · low sample",    "b-face"),   # amber (was magenta — corrected)
    "insufficient_data":   ("Insufficient minutes",      "b-mut"),    # grey
    "measured_geometry":   ("Measured · geometry",       "b-val"),    # green — the real measurement
}
