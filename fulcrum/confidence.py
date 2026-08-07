"""fulcrum.confidence — trust annotation for service outputs, CALIBRATED from the robustness table.

A service returns a value; this says how much to trust it GIVEN the ingestion conditions. `Danger = 0.82` arrives
with `Confidence = 0.38`, and the product decides whether to act — much stronger than pretending every frame is equal.
Confidence is a **lookup into the measured degradation table** (`robustness_<seq>.json`): per service, the retention
under each degradation axis at the estimated severity, multiplied across axes (independent degradations). This is
data-calibrated, not a hand-tuned formula — the confidence-aware-services idea grounded in the robustness study.

CALIBRATION v2 = MEAN over 6 sequences (SNGS-021..026), `robustness_multi.json`. Landmark signals generalise (cross-seq
std ≤0.08); difference signals (space/containment/territory/danger) carry std 0.17–0.19 — treat their confidence as
±0.15. Refine further by adding sequences/seeds and replacing RETENTION. The mechanism (below) is stable regardless.
"""
from __future__ import annotations
import numpy as np

# severity at which each RETENTION value was measured (the reference level per axis)
REF_LEVEL = {"pos_noise": 1.0, "dropout": 0.2, "team_err": 0.1, "ball_noise": 5.0, "officials": 1.0}

# mean retention (corr vs clean) at REF_LEVEL, per service per axis  [robustness_multi.json, 6 sequences]
RETENTION = {
    # Tier 1 — geometry (landmarks/averages robust & generalise; note tempo's phantom-official hole)
    "compactness": {"pos_noise": 0.94, "dropout": 0.76, "team_err": 0.91, "ball_noise": 0.98, "officials": 0.82},
    "width":       {"pos_noise": 0.96, "dropout": 0.84, "team_err": 0.96, "ball_noise": 0.92, "officials": 0.82},
    "centroid":    {"pos_noise": 0.99, "dropout": 0.84, "team_err": 0.98, "ball_noise": 0.87, "officials": 0.99},
    "tempo":       {"pos_noise": 0.96, "dropout": 0.95, "team_err": 1.00, "ball_noise": 1.00, "officials": 0.24},
    # Tier 2 — aggregated (team assignment is the dominant vulnerability)
    "pressing":    {"pos_noise": 0.85, "dropout": 0.59, "team_err": 0.75, "ball_noise": 0.94, "officials": 0.99},
    "territory":   {"pos_noise": 0.75, "dropout": 0.69, "team_err": 0.36, "ball_noise": 0.89, "officials": 0.97},
    "space":       {"pos_noise": 0.66, "dropout": 0.31, "team_err": 0.46, "ball_noise": 0.70, "officials": 0.94},
    # Tier 3 — extremal (landmark reads bulletproof; difference reads fragile; danger IS team-error-sensitive)
    "danger":      {"pos_noise": 0.82, "dropout": 0.67, "team_err": 0.51, "ball_noise": 0.79, "officials": 0.99},
    "containment": {"pos_noise": 0.60, "dropout": 0.49, "team_err": 0.45, "ball_noise": 0.64, "officials": 0.94},
    "last_def":    {"pos_noise": 1.00, "dropout": 0.98, "team_err": 1.00, "ball_noise": 0.88, "officials": 1.00},
    "offside":     {"pos_noise": 0.99, "dropout": 0.82, "team_err": 0.97, "ball_noise": 0.86, "officials": 1.00},
}


def _axis_retention(service, axis, severity):
    """Linear from 1.0 (no degradation) to the measured retention at REF_LEVEL, extrapolated/clipped to [0,1]."""
    r_ref = RETENTION[service][axis]; ref = REF_LEVEL[axis]
    if severity <= 0 or ref <= 0:
        return 1.0
    return float(np.clip(1.0 - (1.0 - r_ref) * (severity / ref), 0.0, 1.0))


def service_confidence(service, *, pos_noise=0.0, dropout=0.0, team_err=0.0, ball_noise=0.0, officials=0.0):
    """Confidence in [0,1] for a service's output under estimated ingestion conditions (product of per-axis
    retentions). `service` is one of RETENTION's keys. Severities are physical: pos_noise/ball_noise in metres,
    dropout/team_err as fractions, officials as a count (or expected #phantom nodes)."""
    if service not in RETENTION:
        raise KeyError(f"unknown service {service!r}; known: {list(RETENTION)}")
    conds = {"pos_noise": pos_noise, "dropout": dropout, "team_err": team_err,
             "ball_noise": ball_noise, "officials": officials}
    r = 1.0
    for axis, sev in conds.items():
        r *= _axis_retention(service, axis, sev)
    return round(float(r), 3)


def estimate_conditions(calib_resid_m=None, team_margin=None, player_count=None, expected=22,
                        ball_conf=None, n_officials=0):
    """Map INGESTION-LAYER signals -> degradation severities for `service_confidence`. Each input is optional; the
    ingestion adapter supplies what it has:
      calib_resid_m : homography reprojection residual (m)        -> pos_noise
      team_margin   : colour-cluster separation in [0,1] (1=clean)-> team_err = (1-margin)*0.2
      player_count  : players seen vs `expected`                  -> dropout
      ball_conf     : ball detector confidence in [0,1]           -> ball_noise (low conf -> larger positional noise)
      n_officials   : suspected officials (e.g. len(flag_officials))-> officials
    """
    c = {"pos_noise": 0.0, "dropout": 0.0, "team_err": 0.0, "ball_noise": 0.0, "officials": float(n_officials)}
    if calib_resid_m is not None:
        c["pos_noise"] = float(max(0.0, calib_resid_m))
    if player_count is not None and expected:
        c["dropout"] = float(np.clip(1.0 - player_count / expected, 0.0, 1.0))
    if team_margin is not None:
        c["team_err"] = float(np.clip((1.0 - team_margin) * 0.2, 0.0, 0.2))
    if ball_conf is not None:
        c["ball_noise"] = float(np.clip((1.0 - ball_conf) * 10.0, 0.0, 10.0))
    return c
