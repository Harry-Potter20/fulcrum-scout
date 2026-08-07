"""fulcrum.render — turn a state + detected holes into the shareable visual (static frame or animated clip).

Attacking team red, defenders blue, velocity arrows, ball white; the danger-weighted value surface glows; the
top-k holes are ranked green stars with their score. This is the product's visual money-shot.
"""
from __future__ import annotations
import numpy as np
from .core import PITCH_L, PITCH_W, find_holes
from .data import state_at
from .reach import exploitable_holes

_STAR_SIZES = [620, 360, 240, 180, 140]


def analyze_frame(frames, fps, fid, top=3, min_players=16):
    """-> (state, value_surface, gx, gy, holes) or None. `min_players` lowered for partial-pitch GSR frames."""
    st = state_at(frames, fps, fid, min_players=min_players)
    if st is None:
        return None
    value, gx, gy, holes = find_holes(st["att"], st["att_v"], st["dfn"], st["dfn_v"], st["ball"], top=max(top, 5))
    holes = exploitable_holes(st, holes)[:top]        # re-rank by danger x pass-reachability
    return st, value, gx, gy, holes


def _draw(ax, pitch, st, value, holes, title, forecast=None):
    ax.imshow(value.T, extent=[0, PITCH_L, 0, PITCH_W], origin="lower", cmap="hot", alpha=0.55,
              aspect="auto", vmin=0, vmax=float(max(value.max(), 1e-6)))
    for (P, V, c) in [(st["att"], st["att_v"], "#d62728"), (st["dfn"], st["dfn_v"], "#1f77b4")]:
        for (px, py), (vx, vy) in zip(P, V):
            ax.scatter([px], [py], s=170, color=c, edgecolors="white", zorder=3)
            ax.arrow(px, py, vx * 0.4, vy * 0.4, head_width=1.0, color="black", zorder=4, length_includes_head=True)
    ax.scatter([st["ball"][0]], [st["ball"][1]], s=80, color="white", edgecolors="black", zorder=5)
    for k, h in enumerate(holes):
        ax.scatter([h["x"]], [h["y"]], marker="*", s=_STAR_SIZES[min(k, 4)], color="#39ff14",
                   edgecolors="black", zorder=6)
        lbl = f"{k + 1} ({h.get('exploit_score', h['score'])})"
        ax.text(h["x"] + 1, h["y"] + 1, lbl, color="#39ff14", fontsize=11, zorder=7)
    if forecast:
        for f in forecast:
            ax.scatter([f["x"]], [f["y"]], marker="X", s=200, color="#00e5ff", edgecolors="black", zorder=6)
    ax.set_title(title, fontsize=10)


def _hero_cmap():
    """Danger colormap for the dark hero render: transparent (safe) -> indigo -> magenta -> hot orange -> gold,
    with alpha rising with danger so safe space stays dark and dangerous space GLOWS."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("fulcrum_danger", [
        (0.00, (0.04, 0.02, 0.12, 0.00)), (0.22, (0.25, 0.06, 0.45, 0.30)),
        (0.48, (0.72, 0.10, 0.52, 0.58)), (0.74, (1.00, 0.42, 0.20, 0.80)),
        (1.00, (1.00, 0.90, 0.52, 0.95))])


def render_hero(frames, fps, fid, out_path, top=3, min_players=16, title="FULCRUM",
                subtitle="EXPLOITABLE SPACE · RANKED BY DANGER", match_label=""):
    """The 'stunning' render: dark premium pitch, smooth glowing danger field, refined players, target-ring holes,
    clean typography + branding. Same data as render_frame (find_holes topology), designed presentation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    res = analyze_frame(frames, fps, fid, top=top, min_players=min_players)
    if res is None:
        raise ValueError(f"frame {fid} not analysable")
    st, value, gx, gy, holes = res
    BG, PITCH_C, LINE, GOLD = "#0a0e14", "#0d2018", "#22402f", "#ffd166"
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W,
                  pitch_color=PITCH_C, line_color=LINE, linewidth=1.1)
    fig = plt.figure(figsize=(12.8, 8.7)); fig.set_facecolor(BG)
    ax = fig.add_axes([0.035, 0.055, 0.93, 0.80]); pitch.draw(ax=ax)
    vmax = float(max(value.max(), 1e-6))
    ax.imshow(value.T, extent=[0, PITCH_L, 0, PITCH_W], origin="lower", cmap=_hero_cmap(),
              vmin=0, vmax=vmax, interpolation="bicubic", aspect="auto", zorder=1)
    for (P, V, glow, core) in [(st["att"], st["att_v"], "#ff4d6d", "#ff9fb0"), (st["dfn"], st["dfn_v"], "#3fd0e6", "#b0edf6")]:
        for (px, py), (vx, vy) in zip(P, V):
            ax.scatter([px], [py], s=460, color=glow, alpha=0.20, zorder=2, edgecolors="none")
            ax.scatter([px], [py], s=145, color=core, edgecolors="white", linewidths=1.1, zorder=4)
            if (vx * vx + vy * vy) ** 0.5 > 0.4:
                ax.arrow(px, py, vx * 0.35, vy * 0.35, head_width=0.9, head_length=0.9, color="white",
                         alpha=0.65, zorder=5, length_includes_head=True, lw=0.9)
    bx, by = st["ball"]
    ax.scatter([bx], [by], s=250, color=GOLD, alpha=0.28, zorder=5, edgecolors="none")
    ax.scatter([bx], [by], s=70, color="white", edgecolors=GOLD, linewidths=1.5, zorder=6)
    for k, h in enumerate(holes):
        for r, a in [(5.0, 0.10), (3.3, 0.20), (1.9, 0.45)]:
            ax.scatter([h["x"]], [h["y"]], s=r * 95, facecolors="none", edgecolors=GOLD, alpha=a, linewidths=2.0, zorder=7)
        sc = h["score"]   # raw topological danger of the space (0-1-ish), not the reachability-weighted score
        ax.text(h["x"], h["y"], f"{k + 1}", color=BG, fontsize=10, fontweight="bold", zorder=9, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.25", fc=GOLD, ec="none"))
        ax.text(h["x"] + 4.2, h["y"], f"{sc:.2f}", color=GOLD, fontsize=10, zorder=9, va="center", family="monospace")
    fig.text(0.037, 0.945, title, color="white", fontsize=30, fontweight="bold", ha="left", family="sans-serif")
    fig.text(0.040, 0.905, subtitle, color=GOLD, fontsize=11.5, ha="left", family="sans-serif")
    peak = max((h["score"] for h in holes), default=0.0)
    fig.text(0.963, 0.945, f"PEAK DANGER  {peak:.2f}", color="white", fontsize=13, ha="right", fontweight="bold")
    if match_label:
        fig.text(0.963, 0.907, match_label, color="#6b7684", fontsize=10.5, ha="right")
    fig.text(0.037, 0.022, "topological pitch-control · the space before the shot, not xG", color="#5a6572", fontsize=9.5, ha="left")
    fig.text(0.963, 0.022, "◍ ball   ● attack   ● defence   ◎ exploitable space", color="#5a6572", fontsize=9.5, ha="right")
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    return holes


def render_frame(frames, fps, fid, out_path, top=3, forecast=None, min_players=16):
    import matplotlib
    matplotlib.use("Agg")
    from mplsoccer import Pitch
    res = analyze_frame(frames, fps, fid, top=top, min_players=min_players)
    if res is None:
        raise ValueError(f"frame {fid} not analysable")
    st, value, gx, gy, holes = res
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W, line_color="black")
    fig, ax = pitch.draw(figsize=(11, 7))
    _draw(ax, pitch, st, value, holes, "fulcrum — exploitable space (green *), ranked by danger", forecast)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return holes


def render_clip(frames, fps, center_fid, out_path, window_s=3.0, top=3, render_fps=8, min_players=16):
    """Animate the ~window_s before center_fid with holes overlaid. Saves GIF (.gif) or MP4 (.mp4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mplsoccer import Pitch
    step = max(1, round(fps / render_fps))
    start = center_fid - int(window_s * fps)
    seq = []
    for fid in range(start, center_fid + 1, step):
        res = analyze_frame(frames, fps, fid, top=top, min_players=min_players)
        if res is not None:
            seq.append(res)
    if not seq:
        raise ValueError("no analysable frames in window")
    vmax = max(float(r[1].max()) for r in seq)
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W, line_color="black")
    fig, ax = pitch.draw(figsize=(11, 7))

    def upd(i):
        ax.clear()
        pitch.draw(ax=ax)
        st, value, gx, gy, holes = seq[i]
        value = np.minimum(value, vmax)
        _draw(ax, pitch, st, value, holes, f"fulcrum — attack developing  ({i + 1}/{len(seq)})")

    anim = FuncAnimation(fig, upd, frames=len(seq), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=render_fps))
    plt.close(fig)
    return seq[-1][4]
