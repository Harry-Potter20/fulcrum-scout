"""fulcrum — topological, danger-ranked hole detection + forecasting for football.

Reads a defence's shape from tracking data, finds exploitable open pockets via velocity-aware PITCH CONTROL +
0-dim superlevel-set persistent homology, ranks them by a tactical danger prior (box + the deep-cross zones
13/15), and renders shareable annotated clips. Validated by a shot-precursor test (AUC 0.886); needs no training,
so it works on any tracking source (Metrica, SkillCorner). Live-video input via a GSR frontend is the roadmap.
"""
from .core import (find_holes, pitch_control, superlevel_persistence, zone_danger_grid, describe_hole,
                   defender_attribution, PITCH_L, PITCH_W)
from .data import load_match, state_at, find_chances
from .render import render_frame, render_clip, analyze_frame, render_hero
from .pipeline import analyze
from .recognize import tactical_signature, estimate_formation, recognize_match, similar_moments
from .reach import pass_reachability, exploitable_holes
from .profile import role_profile, profiled_holes, skillcorner_roles, roster_profile, rostered_holes
from .temporal import track_holes, flicker_stats, structural_exposure
from .opposition import (team_formation, pressing_structure, opposition_report,
                         state_transitions, anticipate)
from .gsr import load_gamestate, load_gsr_trackerstate, id_to_role, GSR_FPS
from .confidence import service_confidence, estimate_conditions
from .ingest import (solve_homography, apply_H, kit_feat, split_teams, assemble_frames, flag_officials, official_by_formation, freeze_frame_to_state)  # recognition-free video adapter
from .simulate import (state_danger, total_danger, apply_edit, plan, plan_report, plan_multi, plan_dynamic,
                       window_state, plan_value)   # tactical planner: structural edits + computed / value / rolled-forward reward
from .scout import (archetypes, role_production, similar, boards, scout_report, space_creators,   # recruitment
                    find_movers, fit_league_strength, project, style_distance, translation_board, translation_report,
                    translation_candidates, derive_insights, player_profiles, fit_spatial_estimator, estimate_spatial)  # cross-league translation + prospective projection
from .metrics import space_creation, containment, player_metrics, state_summary   # differentiated topology stats (off-ball SC, containment, persistence)
# analysis: the consolidated, VALIDATED per-phase tactical analysis layer (the learned world model + topology).
# Imported lazily-safe — needs `worldmodel` importable; guarded so the topology-only package still loads without it.
try:
    from .analysis import load, score, chance_creation, per_player, phase_report, narrate, tactical_shape
    _ANALYSIS = ["load", "score", "chance_creation", "per_player", "phase_report", "narrate", "tactical_shape"]
except Exception:  # pragma: no cover  (topology-only use without the learned model on path)
    _ANALYSIS = []

__all__ = ["defender_attribution", "analyze", "find_holes", "pitch_control", "superlevel_persistence", "zone_danger_grid", "describe_hole",
           "load_match", "state_at", "find_chances", "render_frame", "render_clip", "analyze_frame",
           "tactical_signature", "estimate_formation", "recognize_match", "similar_moments",
           "pass_reachability", "exploitable_holes", "role_profile", "profiled_holes", "skillcorner_roles",
           "roster_profile", "rostered_holes", "load_gamestate", "id_to_role", "GSR_FPS", "PITCH_L", "PITCH_W",
           "solve_homography", "apply_H", "kit_feat", "split_teams", "assemble_frames", "flag_officials", "official_by_formation", "freeze_frame_to_state", "service_confidence", "estimate_conditions",
           "state_danger", "total_danger", "apply_edit", "plan", "plan_report", "plan_multi", "plan_dynamic",
           "window_state", "plan_value",
           "archetypes", "role_production", "similar", "boards", "scout_report",
           "space_creation", "containment", "player_metrics", "state_summary", "space_creators",
           "track_holes", "flicker_stats", "structural_exposure",
           "find_movers", "fit_league_strength", "project", "style_distance", "translation_board", "translation_report",
           "translation_candidates", "derive_insights", "player_profiles", "fit_spatial_estimator", "estimate_spatial", "scouting", "broadcast", "services"] + _ANALYSIS
from .products.scout import scout as scouting
from .products.broadcast import broadcast   # unified scouting surface
from . import services   # the six Foundation Services — the stable platform API (architecture/SERVICES.md)
__version__ = "0.8.0"
