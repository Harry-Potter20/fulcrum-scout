"""fulcrum.pipeline — the one-call product entrypoint: match in, annotated tactical clips out."""
from __future__ import annotations
import os
from .core import describe_hole
from .data import load_match, find_chances
from .render import render_clip, render_frame


def _nearest_fid(frames, target):
    return min(frames, key=lambda f: abs(f - target))


def analyze(source: str, match_id: int, moment_s: float | None = None, out_dir: str = ".",
            top: int = 3, animate: bool = True, max_chances: int = 6, fmt: str = "gif"):
    """Analyse a match. If moment_s is given, render that moment; else auto-detect chances. -> list of results."""
    frames, fps = load_match(source, match_id)
    os.makedirs(out_dir, exist_ok=True)
    if moment_s is not None:
        fids = [_nearest_fid(frames, int(moment_s * fps) + min(frames))]
    else:
        fids = find_chances(frames, fps, max_chances=max_chances)
    results = []
    for fid in fids:
        stem = f"{out_dir}/fulcrum_{source}{match_id}_f{fid}"
        try:
            if animate:
                out = f"{stem}.{fmt}"
                holes = render_clip(frames, fps, fid, out, top=top)
            else:
                out = f"{stem}.png"
                holes = render_frame(frames, fps, fid, out, top=top)
            caption = describe_hole(holes[0]) if holes else "No clear exploitable pocket."
            results.append({"fid": fid, "holes": holes, "out": out, "caption": caption})
        except Exception as e:                                  # skip unusable moments, keep going
            results.append({"fid": fid, "error": str(e)[:120]})
    return results
