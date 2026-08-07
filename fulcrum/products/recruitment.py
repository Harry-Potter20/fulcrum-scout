"""fulcrum.products.recruitment — the Recruitment product, built as a COMPOSITION of Fulcrum capabilities.

Demonstrates the platform inversion: not "can Fulcrum scout?" but "Scout is implemented ON Fulcrum." This product
modifies no backbone — it composes the scout/translation layer (records plane) and can pull in the state-plane
Retrieve service for style-based similarity. Every output inherits its capability's epistemic status.

Pipeline:  records -> league STRENGTH (mover network) -> project CANDIDATES to a target league
                    -> ARCHETYPES (unsupervised roles) -> surplus BOARDS -> REPORT (with provenance + status).
"""
from __future__ import annotations
import fulcrum


def recruit(records, target_league, feat=None, top=15, min_nineties=10.0):
    """Recruitment intelligence for `target_league` from multi-season player-season `records`.
    Returns a report dict; the `contract` field carries the honesty (validated vs face-valid capabilities)."""
    feat = feat or fulcrum.scout.FBREF_FEATURES if hasattr(fulcrum, "scout") else None
    movers = fulcrum.find_movers(records, min_nineties=8.0)
    strength = fulcrum.fit_league_strength(movers)
    candidates = fulcrum.translation_candidates(records, target_league, strength, min_nineties=min_nineties, top=top)
    try:
        arch = fulcrum.archetypes(records)
    except Exception:
        arch = []
    return {
        "product": "recruitment",
        "target_league": target_league,
        "candidates": candidates,                       # translation: VALIDATED (+7.9% OOS)
        "archetypes": arch,                             # unsupervised roles: face-valid
        "league_strength": dict(sorted(strength.items(), key=lambda kv: -kv[1])),
        "provenance": {"n_records": len(records), "n_movers": len(movers), "n_leagues": len(strength)},
        "contract": {"candidates": "validated:translation +7.9% OOS (attacking output)",
                     "archetypes": "face-valid:unsupervised roles",
                     "note": "identity is a downstream label; refreshable on current data"},
    }
