import os, json
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import fulcrum

st.set_page_config(page_title="Fulcrum Scout", page_icon="⚡", layout="wide")
CY, MG = "#22d3ee", "#ff2d78"
st.markdown("""<style>
.stApp{background:#070a10}
h1,h2,h3,h4{color:#eaf4fb !important;font-family:ui-monospace,monospace !important}
.stApp,p,span,label,div{color:#d6e6f2}
[data-testid=stSidebar]{background:#0b131f;border-right:1px solid #16324a}
.step{color:#22d3ee;font-family:ui-monospace,monospace;font-size:12px;letter-spacing:.18em;text-transform:uppercase;border-bottom:1px solid #16324a;padding-bottom:6px;margin:6px 0 12px}
.brk{border:1px solid #16324a;border-left:3px solid #22d3ee;border-radius:4px;background:#0b1420;padding:16px 20px}
.nm{font-size:24px;font-weight:700;color:#eaf4fb}.meta{color:#54677d;font-size:13px}
.arch{color:#22d3ee;font-family:ui-monospace,monospace;font-size:12px;letter-spacing:.12em;text-transform:uppercase}
.desc{color:#c9dae9;line-height:1.6;margin:10px 0}
.rx{font-family:ui-monospace,monospace;font-size:12px;color:#7b8ea3;margin-top:8px}.rx b{color:#22d3ee}.rx .mg{color:#ff2d78}
.stDataFrame{border:1px solid #16324a}
</style>""", unsafe_allow_html=True)

MLAB = {"gls90": "goals", "ast90": "assists", "sh90": "shots", "sot90": "shots OT", "finishing": "finishing",
        "tkl90": "tackles", "int90": "interceptions", "crs90": "crosses", "fls90": "discipline",
        "pts": "scoring", "reb": "rebounding", "ast": "playmaking", "stl": "steals", "blk": "rim protect",
        "tov": "ball security", "fg_pct": "FG%", "fg3_pct": "3P%", "usg": "usage", "ts_pct": "true shoot", "per": "efficiency"}
NEG = {"fls90", "tov"}
FBM = ["gls90", "ast90", "sh90", "sot90", "finishing", "tkl90", "int90", "crs90", "fls90"]
NBAM = ["pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "usg", "ts_pct", "per"]


@st.cache_data(show_spinner="Loading Fulcrum engine + data…")
def load():
    tok = os.environ.get("HF_TOKEN")
    def g(r, p): return json.load(open(hf_hub_download(r, p, repo_type="dataset", token=tok)))
    nba = g("Chucks90/nba-sofascore-data", "nba_scout_records_80229.json")["players"]
    fbp = g("Chucks90/football-sofascore-data", "moneyball_per90_25_26.json")
    fb = [r for r in fbp["records"] if r.get("season") == "25/26"]
    nrep = fulcrum.scouting(nba, sport="basketball", top=len(nba))
    frep = fulcrum.scouting(fb, sport="football", top=len(fb))
    def gpct(recs, metrics):
        out = {}
        for m in metrics:
            v = np.array([float(r.get(m, 0) or 0) for r in recs]); o = v.argsort(); pr = np.empty(len(v)); pr[o] = np.linspace(0, 100, len(v))
            if m in NEG: pr = 100 - pr
            for i, r in enumerate(recs): out.setdefault(r["name"], {})[m] = int(round(pr[i]))
        return out
    return {"nba": nba, "nba_prof": {p["player"]: p for p in nrep["profiles"]},
            "fb": fb, "fb_prof": {p["player"]: p for p in frep["profiles"]},
            "fb_records": fbp["records"], "leagues": sorted({r["league"] for r in fbp["records"]}),
            "fit_fb": gpct(fb, FBM), "fit_nba": gpct(nba, NBAM),
            "meta_fb": {r["name"]: r for r in fb}}

D = load()


def pizza(pct):
    it = list(pct.items()); vals = [v for _, v in it]; N = len(vals); ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    fig = plt.figure(figsize=(4.2, 4.2), facecolor="#0b1420"); ax = fig.add_subplot(111, polar=True); ax.set_facecolor("#0b1420")
    cols = ["#ff2d78" if v >= 80 else ("#22d3ee" if v >= 50 else "#2a6b7a") for v in vals]
    ax.bar(ang, vals, width=2 * np.pi / N * 0.9, color=cols, alpha=0.6, edgecolor=cols, linewidth=1.2)
    ax.set_ylim(0, 100); ax.set_yticks([]); ax.set_xticks(ang)
    ax.set_xticklabels([MLAB.get(k, k) for k, _ in it], color="#8aa0b6", fontsize=8.5, fontfamily="monospace")
    ax.spines["polar"].set_color("#16324a"); ax.grid(color="#16324a", alpha=0.4); ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    return fig


def profile_card(p, meta=None):
    h = f"<div class='brk'><div class='nm'>{p['player']} <span class='meta'>· {p.get('team','')}"
    if meta:
        bits = []
        if meta.get("age"): bits.append(f"{meta['age']:.0f}y")
        if meta.get("market_value"): bits.append(f"€{meta['market_value']/1e6:.0f}M")
        if meta.get("foot"): bits.append(meta["foot"].lower())
        if bits: h += " · " + " · ".join(bits)
    h += f"</span></div><div class='arch'>◇ {p['archetype']}</div><div class='desc'>{p['description']}</div>"
    if p.get("residual"):
        h += "<div class='rx'>over / under expected &nbsp; " + " &nbsp;·&nbsp; ".join(f"<b>{k}</b> {'+' if v>=0 else ''}{v:.1f}σ" for k, v in p["residual"].items()) + "</div>"
    if p.get("comparables"):
        h += "<div class='rx'>plays like &nbsp; <span class='mg'>" + "</span> · <span class='mg'>".join(p["comparables"][:3]) + "</span></div>"
    return h + "</div>"


st.title("⚡ Fulcrum Scout")
sport = st.sidebar.radio("Sport", ["Football", "Basketball"])
FB = sport == "Football"
prof = D["fb_prof"] if FB else D["nba_prof"]
fitpct = D["fit_fb"] if FB else D["fit_nba"]
metrics = FBM if FB else NBAM

st.sidebar.markdown("### Brief")
archs = sorted(set(p["archetype"] for p in prof.values()))
arch_f = st.sidebar.multiselect("Role (archetype)", archs, default=[])
wcols = st.sidebar
st.sidebar.caption("Weight what matters (0–5)")
W = {m: st.sidebar.slider(MLAB.get(m, m), 0, 5, (3 if i < 2 else 0), key="w" + m) for i, m in enumerate(metrics)}
maxval = None
if FB:
    maxval = st.sidebar.slider("Max market value (€M)", 0, 250, 250)
maxage = st.sidebar.slider("Max age", 16, 40, 40)

# ---------- DISCOVER ----------
st.markdown("<div class='step'>01 · Discover — candidates for the brief</div>", unsafe_allow_html=True)
active = {m: w for m, w in W.items() if w > 0}
rows = []
for pn, pc in fitpct.items():
    p = prof.get(pn)
    if not p: continue
    if arch_f and p["archetype"] not in arch_f: continue
    m = D["meta_fb"].get(pn, {}) if FB else {}
    if FB and maxval is not None and m.get("market_value") and m["market_value"] / 1e6 > maxval: continue
    if FB and m.get("age") and m["age"] > maxage: continue
    fit = round(sum(w * pc.get(k, 0) for k, w in active.items()) / sum(active.values())) if active else 0
    row = {"player": pn, "team": p.get("team", ""), "archetype": p["archetype"], "fit": fit}
    if FB:
        row["age"] = round(m.get("age")) if m.get("age") else None
        row["€M"] = round(m["market_value"] / 1e6, 1) if m.get("market_value") else None
        row["value (fit/€10M)"] = round(fit / max(m["market_value"] / 1e7, 0.3), 1) if m.get("market_value") else None
    for k in active: row[MLAB.get(k, k)] = pc.get(k, 0)
    rows.append(row)
rows.sort(key=lambda r: -r["fit"])
if active:
    st.caption(f"{len(rows)} candidates · ranked on " + " · ".join(f"{MLAB.get(k,k)}×{w}" for k, w in active.items())
               + (" · value = fit per €10M" if FB else ""))
    st.dataframe(rows[:40], use_container_width=True, height=430, hide_index=True,
                 column_config={"fit": st.column_config.ProgressColumn("fit", min_value=0, max_value=100, format="%d")})
else:
    st.info("Set at least one weight in the Brief to rank candidates.")

# ---------- UNDERSTAND ----------
st.markdown("<div class='step'>02 · Understand — why this player</div>", unsafe_allow_html=True)
names = [r["player"] for r in rows] or sorted(prof)
sel = st.selectbox("Player", names)
if sel in prof:
    a, b = st.columns([1.15, 1])
    with a:
        st.markdown(profile_card(prof[sel], D["meta_fb"].get(sel) if FB else None), unsafe_allow_html=True)
        pc = fitpct.get(sel, {})
        drivers = sorted(pc.items(), key=lambda kv: -kv[1])[:3]
        if drivers:
            st.caption("value driven by: " + " · ".join(f"{MLAB.get(k,k)} ({v} pct)" for k, v in drivers))
    with b:
        if prof[sel].get("percentiles"):
            st.pyplot(pizza(prof[sel]["percentiles"]), use_container_width=True)

# ---------- RECOMMEND ----------
st.markdown("<div class='step'>03 · Recommend — compare & shortlist</div>", unsafe_allow_html=True)
short = st.multiselect("Shortlist (up to 4)", names, default=names[:3] if names else [], max_selections=4)
if short:
    comp = []
    for pn in short:
        p = prof.get(pn, {}); m = D["meta_fb"].get(pn, {}) if FB else {}
        pc = fitpct.get(pn, {})
        fit = round(sum(w * pc.get(k, 0) for k, w in active.items()) / sum(active.values())) if active else 0
        c = {"player": pn, "archetype": p.get("archetype", ""), "fit": fit}
        if FB:
            c["€M"] = round(m["market_value"] / 1e6, 1) if m.get("market_value") else None
            c["age"] = round(m.get("age")) if m.get("age") else None
            c["value"] = round(fit / max(m["market_value"] / 1e7, 0.3), 1) if m.get("market_value") else None
        comp.append(c)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    best = max(comp, key=lambda c: c.get("value", c["fit"]) if FB else c["fit"])
    p = prof.get(best["player"], {})
    brief = f"**{best['player']}** — {p.get('archetype','')}. {p.get('description','')}"
    if FB and best.get("€M"):
        cheaper = [c for c in comp if c.get("€M") and best.get("€M") and c["€M"] < best["€M"] and c["fit"] >= best["fit"] - 8]
        brief += f"\n\nBest value on the shortlist (fit {best['fit']} at €{best['€M']}M)."
        if cheaper:
            brief += " Cheaper comparable: " + ", ".join(f"{c['player']} (€{c['€M']}M, fit {c['fit']})" for c in cheaper[:2]) + "."
    st.markdown("<div class='step' style='margin-top:14px'>Recruitment brief</div>", unsafe_allow_html=True)
    st.markdown(brief)
