"""fulcrum.reach — pass reachability: is a detected hole actually EXPLOITABLE from the ball?

A hole matters only if the ball can get there. We model a straight pass ball->hole and ask whether any defender
can intercept the passing lane in time (time-to-intercept vs ball travel time). Reachability in [0,1] then
re-weights the danger score into an EXPLOIT score, so the engine ranks holes you can actually attack -- the
grounding for 'best approach'. (A geodesic-through-control-cost variant is the natural richer upgrade.)
"""
from __future__ import annotations
import math

PASS_SPEED = 15.0            # m/s, a firm ground pass
DEF_MAX_SPEED = 7.0          # m/s, defender recovery speed
INTERCEPT_R = 1.2           # m, how close a defender must get to the lane to intercept


def pass_reachability(ball, target, dfn, dfn_v):
    """Probability a straight pass ball->target survives all defenders' interception attempts. -> [0,1]."""
    bx, by = ball
    dx, dy = target[0] - bx, target[1] - by
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return 1.0
    ux, uy = dx / L, dy / L
    survive = 1.0
    for (px, py), (vx, vy) in zip(dfn, dfn_v):
        t_along = max(0.0, min(L, (px - bx) * ux + (py - by) * uy))     # closest approach along the lane
        lx, ly = bx + t_along * ux, by + t_along * uy                   # the lane point the ball passes
        t_ball = t_along / PASS_SPEED                                   # when the ball reaches it
        fx, fy = px + vx * t_ball, py + vy * t_ball                     # defender drifts with current velocity
        need = math.hypot(fx - lx, fy - ly)                            # gap still to close to the lane
        slack = DEF_MAX_SPEED * t_ball                                  # extra distance they can cover in time
        margin = need - INTERCEPT_R - slack                            # >0 => cannot intercept this pass
        survive *= 1.0 - 1.0 / (1.0 + math.exp(margin))                # sigmoid interception probability
    return survive


def exploitable_holes(state, holes):
    """Attach reachability + exploit_score (= danger x reachability) and re-rank by what's actually attackable."""
    out = []
    for h in holes:
        r = pass_reachability(state["ball"], (h["x"], h["y"]), state["dfn"], state["dfn_v"])
        out.append({**h, "reach": round(r, 3), "exploit_score": round(h["score"] * r, 3)})
    out.sort(key=lambda z: -z["exploit_score"])
    return out
