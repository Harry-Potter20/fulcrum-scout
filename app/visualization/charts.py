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
    PAD = size * 0.34  # axis labels sit past the ring at 1.22x rad; long labels (e.g. "Defensive containment")
                        # on the left/right axes need canvas room beyond `size` or the svg clips them
    def pt(i, frac):
        a = -math.pi / 2 + i * 2 * math.pi / n
        return cx + rad * frac * math.cos(a), cy + rad * frac * math.sin(a)
    svg = [f'<svg viewBox="{-PAD:.0f} 0 {size + 2*PAD:.0f} {size}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">']
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


def capability_radar_multi(profiles: dict, colors: dict, size=420) -> str:
    """Overlay radar for 2-3 players on ONE web — the compare 'stunning viz' moment: shape difference reads
    instantly where a side-by-side table needs reading. `profiles` = {name: capability_profile}, `colors` =
    {name: hex}. Each player's polygon is a translucent fill + a distinct stroke colour + a legend row, so identity
    is never colour-alone (a name label always sits beside its swatch)."""
    axes = list(S.CAP_AXES)
    n = len(axes); cx = cy = size / 2; rad = size * 0.32
    PAD = size * 0.30  # same edge-clipping fix as capability_radar — long axis labels need room past `size`
    def pt(i, frac):
        a = -math.pi / 2 + i * 2 * math.pi / n
        return cx + rad * frac * math.cos(a), cy + rad * frac * math.sin(a)
    svg = [f'<svg viewBox="{-PAD:.0f} 0 {size + 2*PAD:.0f} {size+40}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">']
    for g in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for i in range(n) for x, y in [pt(i, g)])
        svg.append(f'<polygon points="{pts}" fill="none" stroke="{P["line"]}" stroke-width="1" opacity="0.5"/>')
    for i in range(n):
        x, y = pt(i, 1.0)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{P["line"]}" stroke-width="1" opacity="0.45"/>')
    for name, profile in profiles.items():
        accent = colors.get(name, P["cy"])
        fr = [(profile[ax]["pct"] / 100.0) if profile[ax]["pct"] is not None else 0.0 for ax in axes]
        dpts = " ".join(f"{x:.1f},{y:.1f}" for i in range(n) for x, y in [pt(i, fr[i])])
        svg.append(f'<polygon points="{dpts}" fill="{accent}" fill-opacity="0.10" stroke="{accent}" stroke-width="2.2"/>')
        for i, ax in enumerate(axes):
            if profile[ax]["pct"] is not None:
                x, y = pt(i, fr[i]); svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{accent}"/>')
    for i, ax in enumerate(axes):
        x, y = pt(i, 1.2)
        anchor = "middle" if abs(x - cx) < 8 else ("start" if x > cx else "end")
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="{P["dim"]}" text-anchor="{anchor}">{S.CAP_AXES[ax]["label"]}</text>')
    # legend — colour + name, so identity never relies on colour alone
    lx, ly = 14 - PAD, size + 22  # shift with the viewBox's new min-x so it stays flush-left, not centered
    for name, profile in profiles.items():
        accent = colors.get(name, P["cy"])
        safe = name.replace("&", "&amp;").replace("<", "")
        svg.append(f'<circle cx="{lx}" cy="{ly}" r="4.5" fill="{accent}"/>')
        svg.append(f'<text x="{lx+10}" y="{ly+4}" font-size="11" fill="{P["tx"]}">{safe}</text>')
        lx += 26 + len(name) * 7
    svg.append("</svg>")
    return "".join(svg)


def capability_delta_bars(name_a: str, prof_a: dict, name_b: str, prof_b: dict, color_a: str, color_b: str,
                           width=640, row_h=34) -> str:
    """Diverging bars, one per capability axis, sorted by |gap| descending — 'who leads, by how much' at a glance,
    pairwise only (a third player has no single delta to diverge against). Bars point toward whichever player
    leads that axis; axes with insufficient evidence for either player are skipped, not shown as a false 0 gap."""
    axes = []
    for ax in S.CAP_AXES:
        pa, pb = prof_a[ax]["pct"], prof_b[ax]["pct"]
        if pa is None or pb is None:
            continue
        axes.append((ax, pa, pb, pa - pb))
    axes.sort(key=lambda t: -abs(t[3]))
    height = 24 + len(axes) * row_h + 10
    m, half = 150, (width - 150 - 20) / 2
    cx = m + half
    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">']
    svg.append(f'<line x1="{cx}" y1="4" x2="{cx}" y2="{height-4}" stroke="{P["line"]}" stroke-width="1"/>')
    for i, (ax, pa, pb, gap) in enumerate(axes):
        y = 24 + i * row_h
        label = S.CAP_AXES[ax]["label"]
        svg.append(f'<text x="{m-10}" y="{y+4}" font-size="11" fill="{P["tx"]}" text-anchor="end">{label}</text>')
        frac = min(abs(gap) / 100.0, 1.0)
        bw = frac * half
        if gap >= 0:   # a leads
            svg.append(f'<rect x="{cx:.1f}" y="{y-9}" width="{bw:.1f}" height="18" rx="3" fill="{color_a}" opacity="0.85"/>')
            svg.append(f'<text x="{cx+bw+6:.1f}" y="{y+4}" font-size="10" fill="{color_a}">+{gap:.0f}</text>')
        else:
            svg.append(f'<rect x="{cx-bw:.1f}" y="{y-9}" width="{bw:.1f}" height="18" rx="3" fill="{color_b}" opacity="0.85"/>')
            svg.append(f'<text x="{cx-bw-6:.1f}" y="{y+4}" font-size="10" fill="{color_b}" text-anchor="end">+{-gap:.0f}</text>')
    safe_a = name_a.replace("&", "&amp;").replace("<", ""); safe_b = name_b.replace("&", "&amp;").replace("<", "")
    svg.append(f'<text x="{m}" y="14" font-size="10" fill="{color_a}">← {safe_a}</text>')
    svg.append(f'<text x="{width-20}" y="14" font-size="10" fill="{color_b}" text-anchor="end">{safe_b} →</text>')
    svg.append("</svg>")
    return "".join(svg)


def danger_sparkline(series: list, peak_t: float = None, width=640, height=120, margin=16) -> str:
    """Danger-over-time line for the Video Lab phase card. `series` = [{"t":sec,"danger":val}, ...]. The peak point
    (if within `series`) gets a marker + label so the "where did it spike" claim is visible, not just implied by
    a single before/after number."""
    if not series:
        return '<div class="mut mono" style="font-size:11px">no danger samples in this window</div>'
    ts = [p["t"] for p in series]; ds = [p["danger"] for p in series]
    t0, t1 = min(ts), max(ts); dmax = max(ds) or 1.0
    W, H = width, height
    def px(t, d):
        x = margin + (t - t0) / max(t1 - t0, 1e-6) * (W - 2 * margin)
        y = H - margin - (d / dmax) * (H - 2 * margin)
        return x, y
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (px(t, d) for t, d in zip(ts, ds)))
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    for frac in (0.0, 0.5, 1.0):
        y = H - margin - frac * (H - 2 * margin)
        svg.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{W-margin}" y2="{y:.1f}" stroke="{P["line"]}" stroke-width="1" opacity="0.4"/>')
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{P["cy"]}" stroke-width="2"/>')
    area = f"{margin},{H-margin} " + pts + f" {W-margin},{H-margin}"
    svg.append(f'<polygon points="{area}" fill="{P["cy"]}" fill-opacity="0.08"/>')
    if peak_t is not None:
        pd = min(series, key=lambda p: abs(p["t"] - peak_t))
        x, y = px(pd["t"], pd["danger"])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{P["amber"]}"/>')
        svg.append(f'<text x="{x:.1f}" y="{max(10,y-9):.1f}" font-size="10" fill="{P["amber"]}" text-anchor="middle" font-family="ui-monospace,monospace">{pd["danger"]:.2f}</text>')
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


def anomaly_map(rows: list, width=680, height=320, max_labels=5) -> str:
    """Capability (y) vs market value percentile (x) — DOMAIN AUTO-FITS the actual plotted rows with padding,
    rather than a fixed 0-100 square. A pre-filtered set (e.g. a narrow age/value band, or already-anomalous
    players) naturally clusters in real percentile space; fixing the axes to 0-100 anyway wastes most of the
    canvas and crushes every point into one corner. Auto-fitting means the SAME points always spread across the
    visible plot, whatever the underlying range. The quadrant split is the MEDIAN of the plotted rows, not a fixed
    45/65 threshold, so "undervalued" always means "above-median capability, below-median cost IN THIS VIEW" —
    still true after auto-fit.

    Labels are SELECTIVE (dataviz skill: never a number on every point) — only the top `max_labels` by anomaly get
    a permanent label, and only if not already crowded against an earlier label (a simple pixel-distance check);
    every point still carries a native hover tooltip (<title>) so nothing is unreachable, and the ranked list
    beneath the chart carries the rest."""
    if not rows:
        return f'<svg viewBox="0 0 {width} {height}" width="100%"><rect width="{width}" height="{height}" fill="{P["panel"]}" rx="6"/></svg>'
    m = 38
    vps = [r["value_percentile"] for r in rows]; caps = [r["cap_index"] for r in rows]
    pad = 8
    vlo, vhi = max(0, min(vps) - pad), min(100, max(vps) + pad)
    clo, chi = max(0, min(caps) - pad), min(100, max(caps) + pad)
    if vhi - vlo < 1: vlo, vhi = max(0, vlo - 10), min(100, vhi + 10)
    if chi - clo < 1: clo, chi = max(0, clo - 10), min(100, chi + 10)
    vmed = sorted(vps)[len(vps) // 2]; cmed = sorted(caps)[len(caps) // 2]

    def X(vp): return m + (vp - vlo) / (vhi - vlo) * (width - 2 * m)
    def Y(cap): return height - m - (cap - clo) / (chi - clo) * (height - 2 * m)

    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{P["panel"]}" rx="6"/>')
    # recessive gridlines at axis extremes + the median split (which IS the quadrant boundary now)
    for g in (vlo, vmed, vhi):
        svg.append(f'<line x1="{X(g):.0f}" y1="{m}" x2="{X(g):.0f}" y2="{height-m}" stroke="{P["line"]}" stroke-width="1" opacity="0.3"/>')
    for g in (clo, cmed, chi):
        svg.append(f'<line x1="{m}" y1="{Y(g):.0f}" x2="{width-m}" y2="{Y(g):.0f}" stroke="{P["line"]}" stroke-width="1" opacity="0.3"/>')
    # undervalued quadrant — median split, a quiet wash + one small corner label
    svg.append(f'<rect x="{X(vlo):.0f}" y="{Y(chi):.0f}" width="{X(vmed)-X(vlo):.0f}" height="{Y(cmed)-Y(chi):.0f}" fill="{P["cy"]}" opacity="0.045"/>')
    svg.append(f'<text x="{X(vlo)+6:.0f}" y="{Y(chi)+13:.0f}" font-size="8.5" letter-spacing="0.06em" fill="{P["cy"]}" opacity="0.8">UNDERVALUED</text>')

    # selective labels: top-anomaly first, skip any candidate whose mark would sit within 26px of an already-placed label
    placed = []
    label_ids = set()
    for r in sorted(rows, key=lambda r: -r["anomaly"]):
        if len(label_ids) >= max_labels:
            break
        x, y = X(r["value_percentile"]), Y(r["cap_index"])
        if all((x - px) ** 2 + (y - py) ** 2 > 26 ** 2 for px, py in placed):
            placed.append((x, y)); label_ids.add(id(r))

    for r in rows:
        x, y = X(r["value_percentile"]), Y(r["cap_index"])
        labelled = id(r) in label_ids
        c = P["cy"] if labelled else P["mut"]
        safe_name = r["name"].replace("&", "&amp;").replace("<", "")
        # transparent oversized hit target (≥24px) carries the hover tooltip; the visible mark stays small
        svg.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="13" fill="transparent"><title>{safe_name} — '
                    f'capability {r["cap_index"]:.0f}, value pct {r["value_percentile"]:.0f}, anomaly {r["anomaly"]:+.0f}</title></circle>')
        svg.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{4.5 if labelled else 3}" fill="{c}" opacity="{0.95 if labelled else 0.5}" '
                    f'stroke="{P["panel"]}" stroke-width="2"/>')
        if labelled:
            anchor = "end" if x > width - 90 else "start"
            dx = -7 if anchor == "end" else 7
            svg.append(f'<text x="{x+dx:.0f}" y="{y+3:.0f}" font-size="9" fill="{P["tx"]}" text-anchor="{anchor}">{safe_name.split()[-1]}</text>')
    svg.append(f'<text x="{width/2:.0f}" y="{height-8}" font-size="9" fill="{P["dim"]}" text-anchor="middle">market value percentile → (range {vlo:.0f}-{vhi:.0f})</text>')
    svg.append(f'<text x="12" y="{height/2:.0f}" font-size="9" fill="{P["dim"]}" transform="rotate(-90 12 {height/2:.0f})" text-anchor="middle">capability index →</text>')
    svg.append("</svg>")
    return "".join(svg)


PL, PW = 105.0, 68.0   # pitch metres — same convention as fulcrum.core / gen_gif.py, center-origin-compatible


def _motion_dot(X, Y, traj, *, r, fill, stroke, stroke_w, dur, opacity=1.0):
    """A circle placed at the trajectory's start, animated through every REMAINING real step via <animateMotion> on
    a RELATIVE path (each waypoint expressed as a delta from the previous one, per SVG path semantics) — a genuine
    frame-by-frame replay of the twin's own intermediate output, not a tween fabricated between two endpoints."""
    if not traj:
        return ""
    x0, y0 = X(traj[0][0]), Y(traj[0][1])
    pts = [(X(x), Y(y)) for x, y in traj]
    d = f"M0,0"
    for i in range(1, len(pts)):
        dx, dy = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        d += f" l{dx:.2f},{dy:.2f}"
    n = len(pts)
    key_times = ";".join(f"{i/(n-1):.3f}" for i in range(n)) if n > 1 else "0"
    return (f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{r}" fill="{fill}" opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{stroke_w}">'
            f'<animateMotion path="{d}" dur="{dur}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" keyPoints="{key_times}"/></circle>')


def counterfactual_pitch(data: dict, width=680, height=452, dur=2.6) -> str:
    """The real 'play sim' display for Simulate (spec §9): ONE anchor's actual twin rollout, ANIMATED frame-by-frame
    through every real intermediate step (not just start/end) — baseline vs capability-injected, both computed,
    playing simultaneously so the divergence is visible as motion, not just a static delta. A faint static trail
    under each dot keeps the full path legible even off-animation (a screenshot, or reduced-motion). The attacking
    team (the capability's subject) gets the full treatment; defenders show only their conditioned reaction, small
    and muted, as context rather than the focus (dataviz: emphasize selectively, not every mark equally loud)."""
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

    base, cond = data["baseline"], data["conditioned"]

    def trail(traj, col, op):
        if len(traj) < 2:
            return ""
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in traj)
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.5" opacity="{op}" stroke-linecap="round"/>'

    # defenders — conditioned reaction only, small and muted, animated but understated (context, not the subject)
    for traj in cond["dfn"]:
        svg.append(trail(traj, P["mut"], 0.18))
        svg.append(_motion_dot(X, Y, traj, r=3.5, fill=P["mut"], stroke=P["panel"], stroke_w=1.2, dur=dur, opacity=0.55))
    # attackers — hollow start marker (static) + baseline (muted, animated) + conditioned (accent, animated) playing together
    for i in range(len(base["att"])):
        b_traj = base["att"][i]
        if b_traj:
            svg.append(trail(b_traj, P["mut"], 0.28))
            svg.append(_motion_dot(X, Y, b_traj, r=4, fill=P["mut"], stroke=P["panel"], stroke_w=1.5, dur=dur, opacity=0.85))
        if i < len(cond["att"]) and cond["att"][i]:
            c_traj = cond["att"][i]
            svg.append(trail(c_traj, P["cy"], 0.45))
            svg.append(_motion_dot(X, Y, c_traj, r=4.5, fill=P["cy"], stroke=P["panel"], stroke_w=1.5, dur=dur))
        if b_traj:
            sxp, syp = b_traj[0]
            svg.append(f'<circle cx="{X(sxp):.1f}" cy="{Y(syp):.1f}" r="3.5" fill="{P["panel"]}" stroke="{P["dim"]}" stroke-width="1.5"/>')
    for balls, col in ((base["ball"], P["mut"]), (cond["ball"], P["hi"])):
        if balls and balls[0]:
            svg.append(_motion_dot(X, Y, balls[0], r=3, fill=col, stroke="none", stroke_w=0, dur=dur))

    svg.append(f'<g font-size="9.5">'
              f'<circle cx="{width-190}" cy="{height-14}" r="3.5" fill="{P["panel"]}" stroke="{P["dim"]}" stroke-width="1.5"/>'
              f'<text x="{width-182}" y="{height-11}" fill="{P["dim"]}">start</text>'
              f'<circle cx="{width-138}" cy="{height-14}" r="4" fill="{P["mut"]}"/>'
              f'<text x="{width-130}" y="{height-11}" fill="{P["mut"]}">baseline</text>'
              f'<circle cx="{width-64}" cy="{height-14}" r="4.5" fill="{P["cy"]}"/>'
              f'<text x="{width-56}" y="{height-11}" fill="{P["cy"]}">with capability</text></g>')
    svg.append("</svg>")
    return "".join(svg)
