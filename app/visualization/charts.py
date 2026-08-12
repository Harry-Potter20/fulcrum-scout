"""app.visualization.charts — reusable SVG visual primitives (§46). Pure strings, theme-aware via PALETTE, no
external chart lib (keeps the app self-contained and the cyberpunk look precise). These are the shared primitives
Scout/Player/Fit all draw from — one instrument, many surfaces.
"""
from __future__ import annotations
import math
from app.config import settings as S

P = S.PALETTE


def capability_radar(profile: dict, size=320, accent=None) -> str:
    """Hexagonal capability web over the 6 axes. Missing axes render hollow (Unknown ≠ zero, §9)."""
    accent = accent or P["cy"]
    axes = list(S.CAP_AXES)
    n = len(axes); cx = cy = size / 2; rad = size * 0.34
    def pt(i, frac):
        a = -math.pi / 2 + i * 2 * math.pi / n
        return cx + rad * frac * math.cos(a), cy + rad * frac * math.sin(a)
    svg = [f'<svg viewBox="0 0 {size} {size}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">']
    # rings
    for g in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for i in range(n) for x, y in [pt(i, g)])
        svg.append(f'<polygon points="{pts}" fill="none" stroke="{P["line"]}" stroke-width="1" opacity="0.55"/>')
    for i in range(n):
        x, y = pt(i, 1.0)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{P["line"]}" stroke-width="1" opacity="0.5"/>')
    # data polygon
    fr = []
    for i, ax in enumerate(axes):
        p = profile[ax]["pct"]
        fr.append((p / 100.0) if p is not None else 0.0)
    dpts = " ".join(f"{x:.1f},{y:.1f}" for i in range(n) for x, y in [pt(i, fr[i])])
    svg.append(f'<polygon points="{dpts}" fill="{accent}" fill-opacity="0.15" stroke="{accent}" stroke-width="2"/>')
    for i, ax in enumerate(axes):
        if profile[ax]["pct"] is not None:
            x, y = pt(i, fr[i]); svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{accent}"/>')
    # labels
    for i, ax in enumerate(axes):
        x, y = pt(i, 1.22)
        anchor = "middle" if abs(x - cx) < 8 else ("start" if x > cx else "end")
        lab = S.CAP_AXES[ax]["label"]
        pct = profile[ax]["pct"]
        val = f"{int(pct)}" if pct is not None else "n/a"
        col = accent if (pct is not None and pct >= 80) else P["mut"]
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="9.5" fill="{P["dim"]}" text-anchor="{anchor}">{lab}</text>')
        svg.append(f'<text x="{x:.1f}" y="{y+11:.1f}" font-size="10" font-weight="bold" fill="{col}" text-anchor="{anchor}">{val}</text>')
    svg.append("</svg>")
    return "".join(svg)


def bar(pct, w=180, accent=None, insufficient=False) -> str:
    """A single capability bar (percentile). Insufficient evidence renders as a hollow track (never a filled 0)."""
    accent = accent or P["cy"]
    if insufficient or pct is None:
        return (f'<span style="display:inline-block;width:{w}px;height:7px;border:1px dashed {P["line"]};'
                f'border-radius:4px;vertical-align:middle"></span>')
    col = accent if pct >= 80 else (P["tx"] if pct >= 55 else P["mut"])
    return (f'<span style="display:inline-block;width:{w}px;height:7px;background:{P["panel2"]};border-radius:4px;'
            f'vertical-align:middle;overflow:hidden"><span style="display:block;height:100%;width:{pct}%;'
            f'background:{col}"></span></span>')


def anomaly_map(rows: list, width=680, height=320, max_labels=4) -> str:
    """Capability (y) vs market value percentile (x). Top-left = high capability + low cost = market anomaly (§43).
    Labels are SELECTIVE (dataviz skill: never a number on every point) — only the top `max_labels` by anomaly get
    a permanent label; every point still carries a native hover tooltip (<title>) so nothing is unreachable, and
    the ranked table beneath the chart (Discover page) carries the rest. Recessive gridlines, surface-ringed dots."""
    m = 36
    def X(vp): return m + (vp / 100) * (width - 2 * m)
    def Y(cap): return height - m - (cap / 100) * (height - 2 * m)
    top_ids = {id(r) for r in sorted(rows, key=lambda r: -r["anomaly"])[:max_labels]}

    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{P["panel"]}" rx="6"/>')
    # recessive gridlines — 0/50/100 only, hairline
    for g in (0, 50, 100):
        svg.append(f'<line x1="{X(g):.0f}" y1="{m}" x2="{X(g):.0f}" y2="{height-m}" stroke="{P["line"]}" stroke-width="1" opacity="0.35"/>')
        svg.append(f'<line x1="{m}" y1="{Y(g):.0f}" x2="{width-m}" y2="{Y(g):.0f}" stroke="{P["line"]}" stroke-width="1" opacity="0.35"/>')
    # undervalued quadrant — a quiet wash + one small corner label, not a text banner across the data
    svg.append(f'<rect x="{X(0):.0f}" y="{Y(100):.0f}" width="{X(45)-X(0):.0f}" height="{Y(65)-Y(100):.0f}" fill="{P["cy"]}" opacity="0.045"/>')
    svg.append(f'<text x="{X(2):.0f}" y="{Y(100)+13:.0f}" font-size="8.5" letter-spacing="0.06em" fill="{P["cy"]}" opacity="0.8">UNDERVALUED</text>')

    for r in rows:
        x, y = X(r["value_percentile"]), Y(r["cap_index"])
        labelled = id(r) in top_ids
        c = P["cy"] if labelled else P["mut"]
        safe_name = r["name"].replace("&", "&amp;").replace("<", "")
        # transparent oversized hit target (≥24px) carries the hover tooltip; the visible mark stays small
        svg.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="13" fill="transparent"><title>{safe_name} — '
                    f'capability {r["cap_index"]:.0f}, value pct {r["value_percentile"]:.0f}, anomaly {r["anomaly"]:+.0f}</title></circle>')
        svg.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{4.5 if labelled else 3}" fill="{c}" opacity="{0.95 if labelled else 0.55}" '
                    f'stroke="{P["panel"]}" stroke-width="2"/>')
        if labelled:
            svg.append(f'<text x="{x+7:.0f}" y="{y+3:.0f}" font-size="9" fill="{P["tx"]}">{safe_name.split()[-1]}</text>')
    svg.append(f'<text x="{width/2:.0f}" y="{height-8}" font-size="9" fill="{P["dim"]}" text-anchor="middle">market value percentile →</text>')
    svg.append(f'<text x="12" y="{height/2:.0f}" font-size="9" fill="{P["dim"]}" transform="rotate(-90 12 {height/2:.0f})" text-anchor="middle">capability index →</text>')
    svg.append("</svg>")
    return "".join(svg)
