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


PL, PW = 105.0, 68.0   # pitch metres — same convention as fulcrum.core / gen_gif.py, center-origin-compatible


def counterfactual_pitch(data: dict, width=680, height=452) -> str:
    """The real 'play sim' display for Simulate (spec §9): ONE anchor's actual twin rollout, baseline vs capability-
    injected, both computed — never illustrative. The attacking team (the capability's subject) gets the full
    start->baseline / start->conditioned treatment; defenders show only their conditioned reaction, small and muted,
    as context rather than the focus (dataviz: label/emphasize selectively, not every mark equally loud)."""
    m = 22
    sx, sy = (width - 2 * m) / PL, (height - 2 * m) / PW
    def X(x): return m + x * sx
    def Y(y): return m + y * sy

    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{P["panel"]}" rx="6"/>')
    # pitch markings — recessive, hairline
    LN = P["line"]
    for a, b in [((0, 0), (PL, 0)), ((0, PW), (PL, PW)), ((0, 0), (0, PW)), ((PL, 0), (PL, PW)), ((PL / 2, 0), (PL / 2, PW))]:
        svg.append(f'<line x1="{X(a[0]):.0f}" y1="{Y(a[1]):.0f}" x2="{X(b[0]):.0f}" y2="{Y(b[1]):.0f}" stroke="{LN}" stroke-width="1" opacity="0.7"/>')
    svg.append(f'<circle cx="{X(PL/2):.0f}" cy="{Y(PW/2):.0f}" r="{9.15*sx:.0f}" fill="none" stroke="{LN}" stroke-width="1" opacity="0.7"/>')
    for x0 in (0, PL - 16.5):
        svg.append(f'<rect x="{X(x0):.0f}" y="{Y(PW/2-20.16):.0f}" width="{16.5*sx:.0f}" height="{40.32*sy:.0f}" fill="none" stroke="{LN}" stroke-width="1" opacity="0.7"/>')

    if data.get("error"):
        svg.append(f'<text x="{width/2}" y="{height/2}" font-size="11" fill="{P["mut"]}" text-anchor="middle">{data["error"]}</text></svg>')
        return "".join(svg)

    start, base, cond = data["start"], data["baseline_end"], data["conditioned_end"]
    # defenders — conditioned reaction only, small and muted (context, not the subject)
    for x, y in cond["dfn"]:
        svg.append(f'<circle cx="{X(x):.0f}" cy="{Y(y):.0f}" r="4" fill="{P["mut"]}" opacity="0.55" stroke="{P["panel"]}" stroke-width="1.5"/>')
    # attackers — start (hollow) -> baseline (muted line+dot) -> conditioned (accent line+dot)
    for i, (sxp, syp) in enumerate(start["att"]):
        if i < len(base["att"]):
            bx, by = base["att"][i]
            svg.append(f'<line x1="{X(sxp):.1f}" y1="{Y(syp):.1f}" x2="{X(bx):.1f}" y2="{Y(by):.1f}" stroke="{P["mut"]}" stroke-width="2" stroke-linecap="round" opacity="0.55"/>')
            svg.append(f'<circle cx="{X(bx):.1f}" cy="{Y(by):.1f}" r="4" fill="{P["mut"]}" opacity="0.8" stroke="{P["panel"]}" stroke-width="1.5"/>')
        if i < len(cond["att"]):
            cx, cy = cond["att"][i]
            svg.append(f'<line x1="{X(sxp):.1f}" y1="{Y(syp):.1f}" x2="{X(cx):.1f}" y2="{Y(cy):.1f}" stroke="{P["cy"]}" stroke-width="2" stroke-linecap="round"/>')
            svg.append(f'<circle cx="{X(cx):.1f}" cy="{Y(cy):.1f}" r="4.5" fill="{P["cy"]}" stroke="{P["panel"]}" stroke-width="1.5"/>')
        svg.append(f'<circle cx="{X(sxp):.1f}" cy="{Y(syp):.1f}" r="3.5" fill="{P["panel"]}" stroke="{P["dim"]}" stroke-width="1.5"/>')
    for label, pts, col in (("ball baseline", base["ball"], P["mut"]), ("ball", cond["ball"], P["hi"])):
        if pts:
            bx, by = pts[0]
            svg.append(f'<circle cx="{X(bx):.1f}" cy="{Y(by):.1f}" r="3" fill="{col}"/>')

    svg.append(f'<g font-size="9.5">'
              f'<circle cx="{width-190}" cy="{height-14}" r="3.5" fill="{P["panel"]}" stroke="{P["dim"]}" stroke-width="1.5"/>'
              f'<text x="{width-182}" y="{height-11}" fill="{P["dim"]}">start</text>'
              f'<circle cx="{width-138}" cy="{height-14}" r="4" fill="{P["mut"]}"/>'
              f'<text x="{width-130}" y="{height-11}" fill="{P["mut"]}">baseline</text>'
              f'<circle cx="{width-64}" cy="{height-14}" r="4.5" fill="{P["cy"]}"/>'
              f'<text x="{width-56}" y="{height-11}" fill="{P["cy"]}">with capability</text></g>')
    svg.append("</svg>")
    return "".join(svg)
