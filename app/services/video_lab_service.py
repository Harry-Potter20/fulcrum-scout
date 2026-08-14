"""app.services.video_lab_service — turns a completed youtube_flip.py run (states.pkl + measured.json, already in
the bucket) into the "phase card" product surface: a detected high-danger window, plain-language interpretation,
a decomposed evidence table, and the real annotated clip to watch. No new modeling — this packages outputs that
already exist (danger time series, space_creation percentiles) into the shape a scout can actually read.
"""
from __future__ import annotations
import os, pickle
import numpy as np

REPO = "Chucks90/football-gsr-data"


def _hf_token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        return None


def available_clips() -> list:
    """{slug, label} for every youtube/<slug>/ with a states.pkl — a real completed flip, not a placeholder."""
    from huggingface_hub import HfApi
    tok = _hf_token()
    try:
        files = HfApi(token=tok).list_repo_files(REPO, repo_type="dataset")
    except Exception:
        return []
    slugs = sorted({f.split("/")[1] for f in files if f.startswith("youtube/") and f.endswith("/states.pkl")})
    return [{"slug": s, "label": s.replace("-", " ").title()} for s in slugs]


def _danger_series(states: dict, fps: float, min_players: int = 10):
    import fulcrum as F
    from fulcrum import services as SV
    fids = sorted(states)
    out = []
    for fid in fids:
        try:
            s = F.state_at(states, fps, fid, min_players=min_players)
            if s is not None:
                out.append((fid, float(SV.evaluate(s)["danger"].value)))
        except Exception:
            continue
    return out


def phase_card(slug: str, fps: float = 5.0, window_frames: int = 5) -> dict | None:
    """The Video Lab 'what Fulcrum saw' card for one flip clip: the highest-danger contiguous window, the delta
    from the clip's baseline danger to that peak, the top measured space-creator, and a plain-language read —
    every number here traces to a real computed value, nothing is asserted without a source field."""
    from huggingface_hub import hf_hub_download
    import json
    tok = _hf_token()
    try:
        sp = hf_hub_download(REPO, f"youtube/{slug}/states.pkl", repo_type="dataset", token=tok)
        mp = hf_hub_download(REPO, f"youtube/{slug}/measured.json", repo_type="dataset", token=tok)
    except Exception:
        return None
    d = pickle.load(open(sp, "rb"))
    states, tid_team = d["states"], d["tid_team"]
    meta = json.load(open(mp))

    series = _danger_series(states, fps)
    if len(series) < 2:
        return {"slug": slug, "meta": meta, "insufficient": True}

    vals = np.array([v for _, v in series])
    fids = [f for f, _ in series]
    baseline = float(np.median(vals[: max(1, len(vals) // 4)]))   # first quarter of live-danger frames = baseline
    peak_i = int(np.argmax(vals))
    peak_fid, peak_val = fids[peak_i], float(vals[peak_i])

    lo = max(0, peak_i - window_frames // 2)
    hi = min(len(fids), peak_i + window_frames // 2 + 1)
    win_start_s, win_end_s = fids[lo] / fps, fids[hi - 1] / fps

    players = sorted([p for p in meta["players"] if p.get("space_creation") is not None],
                     key=lambda p: -p["space_creation"])
    top = players[0] if players else None

    delta_danger_pct = round((peak_val - baseline) / max(abs(baseline), 1e-6) * 100, 0)
    interp = None
    if top:
        interp = (f'Track #{top["tid"]} was the clip\'s top measured space-creator ({top["space_creation"]:.0f}th '
                 f'percentile within this clip) — danger rose from {baseline:.2f} to a peak of {peak_val:.2f} '
                 f'around {win_start_s:.0f}s–{win_end_s:.0f}s into the tracked window.')

    return {
        "slug": slug, "meta": meta, "insufficient": False,
        "baseline_danger": round(baseline, 2), "peak_danger": round(peak_val, 2),
        "delta_danger_pct": delta_danger_pct,
        "window_start_s": round(win_start_s, 1), "window_end_s": round(win_end_s, 1),
        "top_space_creator": top, "interpretation": interp,
        "n_danger_samples": len(series),
        "danger_series": [{"t": round(f / fps, 1), "danger": round(v, 3)} for f, v in series],  # full per-frame trace
        "video_url": f"https://huggingface.co/datasets/{REPO}/resolve/main/youtube/{slug}/annotated.mp4",
        "gif_url": f"https://huggingface.co/datasets/{REPO}/resolve/main/youtube/{slug}/annotated.gif",
    }
