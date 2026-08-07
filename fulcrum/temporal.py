"""fulcrum.temporal — holes as objects in TIME, not per-frame pixels.

The engine's core argument is that a hole is structure that survives — but until now "survives" only meant
across intensity thresholds *within* a frame, so holes flickered frame-to-frame and every downstream consumer
(match report, anticipation, the twin) had to smooth after the fact. This applies the same argument along the
time axis: link holes across frames into tracks, give each a birth/death/lifetime, and rank by how long the
space actually stayed open. A hole that exists for 200ms is noise; one that persists for 3 seconds is a
structural failure someone could have played into.
"""
from __future__ import annotations
import numpy as np
from .core import find_holes
from .data import state_at


def track_holes(frames, fps, t0_s: float, t1_s: float, stride: int = 5, link_radius_m: float = 8.0,
                max_gap_s: float = 0.6, min_persistence: float = 0.05, min_players: int = 16):
    """Track holes through [t0_s, t1_s]. -> list of hole-tracks, longest-lived first.

    Greedy nearest linking (the same light pattern as mise's subject tracker): a hole in frame t links to the
    nearest live track's last position within `link_radius_m`; tracks survive gaps up to `max_gap_s` (holes
    blink as players cross them — a blink is not a death). Each track carries its samples, lifetime, mean
    position and peak danger."""
    f0, f1 = int(t0_s * fps), int(t1_s * fps)
    fids = [f for f in sorted(frames) if f0 <= f <= f1][::stride]
    gap_frames = max_gap_s * fps
    tracks = []
    for fid in fids:
        st = state_at(frames, fps, fid, min_players=min_players)
        if not st:
            continue
        _, _, _, holes = find_holes(st["att"], st["att_v"], st["dfn"], st["dfn_v"], st["ball"],
                                    min_persistence=min_persistence, top=3)
        live = [t for t in tracks if fid - t["samples"][-1][0] <= gap_frames]
        used = set()
        for h in holes:
            hp = np.array([h["x"], h["y"]])
            best, bt = link_radius_m, None
            for t in live:
                if id(t) in used:
                    continue
                d = float(np.linalg.norm(hp - np.array(t["samples"][-1][1])))
                if d < best:
                    best, bt = d, t
            if bt is not None:
                bt["samples"].append((fid, (h["x"], h["y"]), h["score"]))
                used.add(id(bt))
            else:
                tracks.append({"samples": [(fid, (h["x"], h["y"]), h["score"])]})
    out = []
    for t in tracks:
        s = t["samples"]
        if len(s) < 2:
            continue                                        # a single sighting is exactly the flicker we kill
        xs = np.array([p for _, p, _ in s])
        scores = [sc for _, _, sc in s]
        out.append({
            "birth_s": round(s[0][0] / fps, 2), "death_s": round(s[-1][0] / fps, 2),
            "lifetime_s": round((s[-1][0] - s[0][0]) / fps, 2),
            "x": round(float(xs[:, 0].mean()), 1), "y": round(float(xs[:, 1].mean()), 1),
            "drift_m": round(float(np.linalg.norm(xs[-1] - xs[0])), 1),
            "peak_danger": round(max(scores), 3), "mean_danger": round(float(np.mean(scores)), 3),
            "n_samples": len(s),
        })
    out.sort(key=lambda z: -z["lifetime_s"])
    return out


def structural_exposure(frames, fps, t0_s: float = 0.0, t1_s=None, stride: int = 5, min_players: int = 16,
                        structural_s: float = 1.0):
    """A defence's TEMPORAL exposure: how much of the space it concedes is STRUCTURAL (a durable gap that survives
    >= `structural_s`) vs a TRANSIENT flicker. This is the 'structural weakness vs momentary lapse' signal that
    xG/xThreat cannot express AND that state-level topology cannot either — a single freeze-frame hole has no
    lifetime (indeed its persistence == its danger score; see fulcrum.metrics.state_summary). Only the time axis
    separates a channel a team *always* leaves open from one that blinked once. Wraps `track_holes`. -> dict with
    the rate of durable gaps and whether structural holes carry more danger than transient ones."""
    if t1_s is None:
        t1_s = max(frames) / fps
    trks = track_holes(frames, fps, t0_s, t1_s, stride=stride, min_players=min_players)
    struc = [t for t in trks if t["lifetime_s"] >= structural_s]
    trans = [t for t in trks if t["lifetime_s"] < structural_s]
    dur_min = max((t1_s - t0_s) / 60.0, 1e-6)
    md = lambda ts: round(float(np.mean([t["mean_danger"] for t in ts])), 3) if ts else 0.0
    return {"minutes": round(dur_min, 1), "n_tracks": len(trks),
            "n_structural": len(struc), "n_transient": len(trans),
            "structural_per_min": round(len(struc) / dur_min, 2),
            "structural_fraction": round(len(struc) / max(len(trks), 1), 3),
            "mean_lifetime_structural_s": round(float(np.mean([t["lifetime_s"] for t in struc])), 2) if struc else 0.0,
            "danger_structural": md(struc), "danger_transient": md(trans),
            "structural_holes": [{"x": t["x"], "y": t["y"], "lifetime_s": t["lifetime_s"], "mean_danger": t["mean_danger"]}
                                 for t in struc[:10]]}


def flicker_stats(frames, fps, t0_s: float, t1_s: float, stride: int = 5, min_players: int = 16):
    """How unstable is the raw per-frame top hole vs the temporal view? -> dict.
    Measures the median frame-to-frame jump of the raw top hole, and what fraction of raw holes belong to a
    track that survives >= 1s — i.e. how much of what the per-frame engine reports is actually structure."""
    f0, f1 = int(t0_s * fps), int(t1_s * fps)
    fids = [f for f in sorted(frames) if f0 <= f <= f1][::stride]
    tops = []
    for fid in fids:
        st = state_at(frames, fps, fid, min_players=min_players)
        if not st:
            continue
        _, _, _, holes = find_holes(st["att"], st["att_v"], st["dfn"], st["dfn_v"], st["ball"],
                                    min_persistence=0.05, top=1)
        if holes:
            tops.append((fid, np.array([holes[0]["x"], holes[0]["y"]])))
    jumps = [float(np.linalg.norm(b - a)) for (_, a), (_, b) in zip(tops[:-1], tops[1:])]
    trks = track_holes(frames, fps, t0_s, t1_s, stride=stride, min_players=min_players)
    total_samples = sum(t["n_samples"] for t in trks) or 1
    stable = sum(t["n_samples"] for t in trks if t["lifetime_s"] >= 1.0)
    return {"n_frames": len(tops),
            "median_top_jump_m": round(float(np.median(jumps)), 2) if jumps else None,
            "p90_top_jump_m": round(float(np.percentile(jumps, 90)), 2) if jumps else None,
            "frac_samples_in_stable_tracks": round(stable / total_samples, 3),
            "n_tracks": len(trks),
            "n_stable_tracks_1s": sum(1 for t in trks if t["lifetime_s"] >= 1.0)}
