"""app.services.counterfactual_service — the LIVE counterfactual. Loads the twin (fulcrum.products.tactical_fit,
validated mechanism: attack corr 0.994, defend corr -0.865, §COUNTERFACTUAL_RL.md) once as a process-wide singleton,
builds REAL anchor windows from actual tracked broadcast phases (SNGS, the same states the Measured page reads —
no synthetic data), and runs the simulation on demand.

This makes the "Signing impact" tab an actual computation, not static text — but the epistemic status does NOT
change because the compute is live: it is still §63 territory (representation -> a simulated trajectory, never a
causal decision claim) until sim-to-real (G3) passes. `registry.counterfactual_signing_impact` stays UNPROVEN; this
service's output is always returned with that framing attached, and the app is not allowed to strip it.

Torch + the twin checkpoint are HEAVY relative to the rest of this lightweight app, so everything here is lazy
(nothing imports torch at module load) and fails soft: if torch/the checkpoint/HF_TOKEN aren't available in this
deployment, `available()` returns False and the UI shows "simulation engine unavailable here" instead of crashing.
"""
from __future__ import annotations
import os, sys, functools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # -> fulcrum, jobs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "jobs")))  # -> worldmodel

STEPF, FPS = 13, 25.0     # the twin's ~0.5s step, matching jobs/cf_g1_real.py's grounding
DT = STEPF / FPS
ANCHOR_SEQ_DEFAULT = "SNGS-021"


def available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.cache/huggingface/token")))
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def _engine():
    """The TacticalFit engine, loaded once per process (checkpoint download + torch init is the expensive part)."""
    from fulcrum.products.tactical_fit import TacticalFit
    return TacticalFit()


def _build_anchors(seq: str, n: int = 10):
    """REAL anchor windows from a tracked sequence's states (same source as the Measured page) — sampled evenly
    across the match so the simulation runs over varied phases, not one repeated moment."""
    import numpy as np
    import worldmodel as W
    from app.data.tracked import _load_states
    states = _load_states(seq)
    fids = sorted(f for f in states if isinstance(states[f], dict) and states[f].get("players"))
    wins = []
    step = max(1, (len(fids) - STEPF) // max(n, 1))
    for k in range(STEPF, len(fids), step):
        f0, fp = fids[k], fids[k - STEPF]
        cur, prev = states[f0], states[fp]
        if cur.get("ball") is None:
            continue
        nodes = [("ball", 2.0, cur["ball"])]
        for tid, v in cur["players"].items():
            nodes.append((tid, v[0], v[1]))
        if len(nodes) < 12:
            continue
        pos = np.array([xy for _, _, xy in nodes], np.float32)
        vel = np.zeros_like(pos)
        for i, (tid, tm, xy) in enumerate(nodes):
            if tid != "ball" and tid in prev.get("players", {}):
                vel[i] = (np.array(xy) - np.array(prev["players"][tid][1])) / DT
        team = np.array([t for _, t, _ in nodes], np.float32)
        isball = np.array([1.0] + [0.0] * (len(nodes) - 1), np.float32)
        wins.append(W.Window(pos, vel, np.zeros_like(pos), team, isball, [pos.copy()]))
        if len(wins) >= n:
            break
    return wins


@functools.lru_cache(maxsize=8)
def _anchors_for(seq: str, n: int = 10):
    return tuple(_build_anchors(seq, n))


def run_signing_simulation(capability: dict, seq: str = ANCHOR_SEQ_DEFAULT, n_anchors: int = 10) -> dict:
    """Run the LIVE simulation: `capability` e.g. {"forward_intent": 1.5, "press_resistance": 0.6} rolled forward
    through the twin over `n_anchors` real tracked phases. Returns TacticalFit.fit_report's honest dict verbatim
    (mean_delta_danger, CI95, epistemic string, status) plus a `live` flag so the UI can show it ran, not cached."""
    if not available():
        return {"error": "simulation engine unavailable in this environment (torch/HF token not present)"}
    try:
        eng = _engine()
        anchors = _anchors_for(seq, n_anchors)
        if not anchors:
            return {"error": f"no usable anchor phases in {seq}"}
        report = eng.fit_report(capability, list(anchors), boot=200)
        report["live"] = True
        report["anchor_seq"] = seq
        report["n_anchors_requested"] = n_anchors
        return report
    except Exception as e:
        return {"error": f"simulation failed: {e}"}


def _pts(arr) -> list:
    """numpy Nx2 -> plain [[x,y], ...] floats, pitch metres. JSON/session-state safe."""
    return [[round(float(x), 2), round(float(y), 2)] for x, y in arr]


def rollout_for_viz(capability: dict, seq: str = ANCHOR_SEQ_DEFAULT, anchor_index: int = 0) -> dict:
    """ONE real anchor's actual before/after positions — baseline (no capability) vs conditioned (capability
    injected) — for the pitch visualization. Every point returned is a REAL computed rollout position (the twin's
    own output), never an illustrative/synthetic one: this is what makes the pitch honest rather than a mockup.
    Baseline and conditioned share the same starting frame (the capability only perturbs the initial velocity, so
    position at t0 is identical either way) — returned once as `start`."""
    if not available():
        return {"error": "simulation engine unavailable in this environment (torch/HF token not present)"}
    try:
        eng = _engine()
        anchors = _anchors_for(seq, max(anchor_index + 1, 10))
        if not anchors or anchor_index >= len(anchors):
            return {"error": f"no anchor #{anchor_index} available in {seq}"}
        window = anchors[anchor_index]
        att0, dfn0, ball0, start0 = eng._rollout(window, {}, return_start=True)
        att1, dfn1, ball1, _ = eng._rollout(window, capability, return_start=True)
        return {
            "seq": seq, "anchor_index": anchor_index, "capability": capability,
            "start": {"att": _pts(start0[0]), "dfn": _pts(start0[1]), "ball": _pts(start0[2])},
            "baseline_end": {"att": _pts(att0), "dfn": _pts(dfn0), "ball": _pts(ball0)},
            "conditioned_end": {"att": _pts(att1), "dfn": _pts(dfn1), "ball": _pts(ball1)},
        }
    except Exception as e:
        return {"error": f"simulation failed: {e}"}
