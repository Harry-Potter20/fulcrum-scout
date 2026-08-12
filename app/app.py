"""Fulcrum Scout — MVP shell (app_build.md §48). Pages call services; services read the data layer + the validation
registry. No model() calls from UI, no opaque master score, every claim carries its tier. Run:

    cd Football_Research && streamlit run app/app.py

P0 surfaces: Home (command center) · Discover (capability, not production) · Player (dossier) · Tactical Fit ·
Evidence (the registry, made visible). Counterfactual is present but marked EXPERIMENTAL until G3 (§28/§32).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # Football_Research on path

import streamlit as st
from app.config import settings as S
from app.components import ui
from app.visualization import charts
from app.services import scout_service as scout
from app.services import fit_service as fit
from app.services import measured_service as measured
from app.services import counterfactual_service as cfsvc
from app import db
from fulcrum import registry as R

P = S.PALETTE
st.set_page_config(page_title="Fulcrum Scout", page_icon="🎯", layout="wide")

# Note: auto-ingest is NOT triggered from inside this app process. Streamlit only executes app.py per browser
# WEBSOCKET session — a plain HTTP GET (which is what a cron job sends) never reaches this script at all, so a
# query-param trigger here would be dead code. The real ingest webhook is app/ingest_server.py, a separate minimal
# HTTP service deployed alongside this one (see app/DEPLOY.md) — genuine stdlib HTTP, no session model involved.
ui.inject_css()


# ---- cached service wrappers (recompute only when inputs change) ----
@st.cache_data(show_spinner=False)
def c_discover(season, mm, ma, mv, leagues, axis): return scout.discover(season, min_minutes=mm, max_age=ma, max_value_m=mv, leagues=leagues, priority_axis=axis)
@st.cache_data(show_spinner=False)
def c_anomalies(season): return scout.market_anomalies(season)
@st.cache_data(show_spinner=False)
def c_index(season): return scout.player_index(season)
@st.cache_data(show_spinner=False)
def c_fits(season, pri, mm, ma, mv, leagues): return fit.best_fits(season, list(pri), min_minutes=mm, max_age=ma, max_value_m=mv, leagues=leagues)
@st.cache_data(show_spinner=False)
def c_player(name, season): return scout.get_player(name, season)
@st.cache_data(show_spinner=False)
def c_similar(name, season): return scout.similar(name, season)
@st.cache_data(show_spinner=False)
def c_measured(seq): return measured.measured_players(seq)
@st.cache_data(show_spinner=False)
def c_measured_seqs(): return measured.sequences()


ss = st.session_state
ss.setdefault("page", "Home"); ss.setdefault("player", None); ss.setdefault("season", S.DEFAULT_SEASON)


def goto(page, player=None):
    ss.page = page
    if player: ss.player = player


# ---- access gate: if FULCRUM_APP_PASSWORD is set (public deploy), require it; unset (local dev) = open ----
def _gate():
    pw = os.environ.get("FULCRUM_APP_PASSWORD", "")
    if not pw or ss.get("auth_ok"):
        return
    st.markdown(f'<div style="max-width:360px;margin:12vh auto 0"><div style="font-family:ui-monospace,monospace;'
                f'font-size:22px;font-weight:700;color:{P["cy"]}">FULCRUM</div>'
                f'<div class="eyebrow">Scout · access required</div></div>', unsafe_allow_html=True)
    c = st.columns([1, 2, 1])[1]
    with c:
        entered = st.text_input("Access key", type="password", label_visibility="collapsed", placeholder="access key")
        if entered:
            if entered == pw:
                ss.auth_ok = True; st.rerun()
            else:
                st.error("Incorrect access key.")
    st.stop()


_gate()


# ================= sidebar =================
with st.sidebar:
    st.markdown(f'<div style="font-family:ui-monospace,monospace;font-size:20px;font-weight:700;color:{P["cy"]};'
                f'letter-spacing:.04em">FULCRUM</div><div class="eyebrow" style="margin-bottom:14px">Scout · tactical intelligence</div>',
                unsafe_allow_html=True)
    ss.season = st.selectbox("Season", S.SEASONS, index=S.SEASONS.index(ss.season))
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    for pg in ["Home", "Discover", "Player", "Tactical Fit", "Measured", "Evidence"]:
        if st.button(pg, key=f"nav_{pg}", use_container_width=True):
            goto(pg)
    st.markdown(f'<div class="mut mono" style="font-size:9.5px;margin-top:20px;line-height:1.6">'
                f'Identity never enters the backbone.<br>No capability claim without evidence.<br>'
                f'No opaque score where an explanation should exist.</div>', unsafe_allow_html=True)


# ================= HOME =================
def home():
    st.markdown('<div class="eyebrow">Command center</div>', unsafe_allow_html=True)
    st.markdown("# Fulcrum Scout")
    st.markdown(f'<p style="color:{P["mut"]};max-width:620px">Scouting by <b style="color:{P["tx"]}">capability</b>, '
                f'not production. Fulcrum measures the geometry and behaviour producing the numbers — space creation, '
                f'off-ball penetration, press resistance — then connects them to the tactical problem your team needs to solve.</p>',
                unsafe_allow_html=True)
    # the intelligence workflow as the mental model (§1): discover → fit → simulate → evidence
    actions = [("DISCOVER", "Find players with capabilities we need", "Discover"),
               ("FIT", "Test them against our tactical problem", "Tactical Fit"),
               ("SIMULATE", "See what changes if we sign them", "Player"),
               ("EVIDENCE", "Inspect how Fulcrum knows", "Evidence")]
    c = st.columns(4)
    for col, (lab, sub, pg) in zip(c, actions):
        with col:
            st.markdown(f'<div class="fcard" style="min-height:74px"><b class="cy" style="font-size:14px;letter-spacing:.06em">{lab}</b>'
                        f'<div class="mut" style="font-size:10.5px;margin-top:5px;line-height:1.35">{sub}</div></div>',
                        unsafe_allow_html=True)
            if st.button(f"→ {lab.title()}", key=f"home_{lab}", use_container_width=True): goto(pg)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    anoms = c_anomalies(ss.season)
    idx = c_index(ss.season)
    n_high = sum(1 for r in idx if r["cap_index"] >= 80)
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="eyebrow">Market intelligence · ' + ss.season + '</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fpanel"><div class="kv"><span class="mut">Players in pool</span><b class="mono">{len(idx)}</b></div>'
                    f'<div class="kv"><span class="mut">High-capability profiles (≥80)</span><b class="mono cy">{n_high}</b></div>'
                    f'<div class="kv"><span class="mut">Market anomalies (cap ≫ cost)</span><b class="mono cy">{sum(1 for a in anoms if a["anomaly"]>=25)}</b></div>'
                    f'<div class="kv" style="border:0"><span class="mut">Leagues covered</span><b class="mono">8</b></div></div>',
                    unsafe_allow_html=True)
    with right:
        st.markdown('<div class="eyebrow">Top market anomaly</div>', unsafe_allow_html=True)
        if anoms:
            a = anoms[0]
            st.markdown(f'<div class="fpanel"><b style="font-size:15px">{a["name"]}</b> '
                        f'<span class="mut mono" style="font-size:11px">{a["league"]} · €{a["value_m"]}M · {a["archetype"]}</span>'
                        f'<div style="margin-top:8px;color:{P["mut"]};font-size:12.5px">Capability index '
                        f'<b class="cy">{a["cap_index"]:.0f}</b> vs value percentile <b>{a["value_percentile"]:.0f}</b> — '
                        f'the market may under-price capability generated off the ball.</div></div>', unsafe_allow_html=True)
            if st.button("Open profile →", key="home_open_anom"): goto("Player", a["name"])

    # ---- guided flows: the three things to try first (§53/§54) ----
    st.markdown('<div class="eyebrow" style="margin-top:22px">Start here · three flows</div>', unsafe_allow_html=True)
    g = st.columns(3)
    top_creator = c_discover(ss.season, 8.0, 25, 60.0, None, "space_creation")
    seed_player = (top_creator[0]["name"] if top_creator else (anoms[0]["name"] if anoms else None))
    flows = [
        ("① Capability, not production",
         f"See a young space-creator described the way a box score never could — {seed_player}.",
         "Open a capability profile", lambda: goto("Player", seed_player)),
        ("② Solve a tactical problem",
         "State a need — “break a low block” — and rank the market by the capability that addresses it.",
         "Open Tactical Fit", lambda: goto("Tactical Fit")),
        ("③ Measured, not estimated",
         "The real Fulcrum measurement — space creation computed from geometry on live tracked players.",
         "Open Measured geometry", lambda: goto("Measured")),
    ]
    for col, (title, desc, btn, act) in zip(g, flows):
        with col:
            st.markdown(f'<div class="fcard" style="min-height:120px"><b class="cy" style="font-size:13px">{title}</b>'
                        f'<div class="mut" style="font-size:12px;margin:6px 0 10px">{desc}</div></div>', unsafe_allow_html=True)
            if st.button(btn, key=f"flow_{btn}", use_container_width=True): act()


# ================= DISCOVER =================
def discover():
    st.markdown('<div class="eyebrow">Discover · capability, not production</div>', unsafe_allow_html=True)
    st.markdown("# Discover")
    f = st.columns([2, 1, 1, 1])
    axis = f[0].selectbox("Priority capability", list(S.CAP_AXES), format_func=lambda a: S.CAP_AXES[a]["label"])
    max_age = f[1].slider("Max age", 17, 38, 30)
    max_val = f[2].slider("Max value €M", 1, 200, 200)
    min_min = f[3].slider("Min 90s", 1, 30, 8)
    rows = c_discover(ss.season, float(min_min), max_age, float(max_val), None, axis)

    st.markdown(f'<div class="mut mono" style="font-size:11px;margin:6px 0 2px">{len(rows)} players · ranked by '
                f'{S.CAP_AXES[axis]["label"]} · {ui.tier_badge(S.CAP_AXES[axis]["registry_key"])}</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="margin-top:14px">Market map · capability vs cost</div>', unsafe_allow_html=True)
    st.markdown(charts.anomaly_map(c_anomalies(ss.season)), unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    for r in rows[:25]:
        if ui.player_row(r, prefix="disc"): goto("Player", r["name"])


# ================= PLAYER =================
def player_page():
    idx = c_index(ss.season)
    names = [r["name"] for r in idx]
    default = ss.player if ss.player in names else names[0]
    pick = st.selectbox("Player", names, index=names.index(default))
    ss.player = pick
    pl = c_player(pick, ss.season)
    if not pl:
        st.warning("No record."); return

    st.markdown(f'<div class="eyebrow">Player intelligence</div><h1 style="margin-bottom:0">{pl["name"]}</h1>'
                f'<div class="mut mono" style="font-size:12px">{pl["league"]} · {pl["age"]}y · €{pl["value_m"]}M · '
                f'{pl["foot"]}-footed · {pl["nineties"]:.0f}×90 played</div>', unsafe_allow_html=True)
    arc = pl["archetype"]
    blend = f' / {arc["secondary"]}' if "secondary" in arc else ""
    st.markdown(f'<div style="margin:10px 0"><span class="eyebrow">Archetype</span> '
                f'<b style="color:{P["cy"]};font-size:15px">{arc["primary"]}{blend}</b> '
                f'<span class="mut mono" style="font-size:10.5px">· from {", ".join(arc["primary_axes"])} (behaviour, not position)</span></div>',
                unsafe_allow_html=True)
    ui.scorecard(pl["scorecard"])

    # WHAT HE ACTUALLY DOES — the capability translated into behaviour, not a number (§4)
    bs = pl["beyond_stats"]
    if bs:
        top = bs[0]; more = bs[1] if len(bs) > 1 else None
        surname = pl["name"].split()[-1]
        narr = (f'{surname}\'s strongest contribution is <b style="color:{P["cy"]}">{top["label"].lower()}</b> '
                f'({top["pct"]:.0f}) — {top["say"]}.')
        if more:
            narr += f' He also brings <b style="color:{P["tx"]}">{more["label"].lower()}</b> ({more["pct"]:.0f}).'
        st.markdown(f'<div class="eyebrow" style="margin-top:16px">What he actually does</div>'
                    f'<div class="fpanel" style="margin-top:6px"><div style="font-size:13.5px;line-height:1.65;color:{P["tx"]}">{narr}</div></div>',
                    unsafe_allow_html=True)

    tabs = st.tabs(["Capabilities", "Behaviour", "Neighbours", "Signing impact"])
    with tabs[0]:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown(charts.capability_radar(pl["capabilities"]), unsafe_allow_html=True)
        with c2:
            ui.capability_panel(pl["capabilities"])
        st.markdown(f'<div class="mut mono" style="font-size:10px;margin-top:8px">Left badge = the METHOD\'s tier '
                    f'(registry). Right badge = THIS player\'s evidence grade. Amber = estimated from production; '
                    f'green = measured from geometry (see Measured page).</div>', unsafe_allow_html=True)
        with st.expander("ⓘ  Why these scores — and what Fulcrum did NOT measure"):
            st.markdown("**How each capability is derived here.** For named players these are *estimated from "
                        "production proxies*; the validated Fulcrum method measures the same thing from geometry "
                        "(attaching this player's tracking replaces the proxy — see the **Measured** page).")
            for ax, spec in S.CAP_AXES.items():
                cap = pl["capabilities"][ax]; reg = R.get(spec["registry_key"])
                drv = ", ".join(cap["drivers"]) or "—"
                val = f'{cap["pct"]:.0f}' if cap["pct"] is not None else "n/a"
                st.markdown(f'- **{cap["label"]}** `{val}` — proxies `{drv}` · method **{reg["tier"]["label"]}** ({reg["metric"]})')
            st.markdown(f"**Fulcrum does not measure:** finishing quality as a trait · leadership · personality · "
                        f"injury risk · off-camera work. *Unknown ≠ zero* — missing capability is shown, not filled with 0.")
    with tabs[1]:
        st.markdown(f'<div class="eyebrow">Behaviour · how value is generated, not what it produced</div>', unsafe_allow_html=True)
        for b in pl["beyond_stats"]:
            st.markdown(f'<div class="fcard" style="margin:8px 0"><div style="display:flex;justify-content:space-between">'
                        f'<b>{b["label"]}</b><span class="mono cy">{b["pct"]:.0f}<span class="mut" style="font-size:10px">/100</span></span></div>'
                        f'<div style="font-size:12.5px;margin-top:3px;color:{P["tx"]}">{b["say"]}</div>'
                        f'<div style="margin-top:6px">{ui.badge(b["tier"], b["tier_badge"])}'
                        f'{ui.evidence_badge(b["evidence"])}'
                        f'<span class="mut mono" style="font-size:10px">method validated at {b["metric"]}</span></div></div>',
                        unsafe_allow_html=True)
    with tabs[2]:
        st.markdown('<div class="eyebrow">Capability neighbours · same problem, sometimes a different mechanism</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mut mono" style="font-size:10.5px;margin-bottom:6px">Ranked by behavioural similarity '
                    f'(capability vector) — not position, league, or production totals.</div>', unsafe_allow_html=True)
        for s in c_similar(pick, ss.season):
            mech = " · ".join(f'{lab} {v}' for lab, v in s["mechanism"])
            st.markdown(f'<div class="kv"><span><b>{s["name"]}</b> <span class="mut mono" style="font-size:11px">'
                        f'{s["league"]} · €{s["value_m"]}M</span></span><span class="mono cy">{s["similarity"]:.0f}%</span></div>'
                        f'<div class="mut mono" style="font-size:10.5px;margin:-2px 0 7px">solves it via <span style="color:{P["tx"]}">{mech}</span> '
                        f'· shares {", ".join(s["shared"])}</div>', unsafe_allow_html=True)
    with tabs[3]:
        st.markdown(f'{ui.badge("Simulation · research preview", "b-exp")}', unsafe_allow_html=True)
        reg = R.get("counterfactual_mechanism_attack")
        st.markdown(f'<div class="fpanel" style="margin-top:10px"><div class="eyebrow">Signing impact</div>'
                    f'<div class="mut" style="font-size:13px;line-height:1.7;margin-top:6px">'
                    f'Drops this capability into the world model and rolls real tracked phases forward — everyone '
                    f'reacts — to show a <b style="color:{P["tx"]}">different simulated trajectory</b>. The <i>mechanism</i> '
                    f'is validated ({reg["metric"]}); the signing-impact <i>number</i> below is not — it washes out on '
                    f'noisy tracking and needs sim-to-real (G3). This runs live on request; the result is always a '
                    f'research preview, never a prediction.</div>'
                    f'<div style="margin-top:10px">{ui.tier_badge("counterfactual_signing_impact")}'
                    f'<span class="mut mono" style="font-size:10px">{R.get("counterfactual_signing_impact")["evidence"]}</span></div></div>',
                    unsafe_allow_html=True)

        if not cfsvc.available():
            st.markdown(f'<div class="mut mono" style="font-size:11px;margin-top:10px">Simulation engine unavailable '
                        f'in this environment (needs torch + HF access) — showing the mechanism description only.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            axis = st.selectbox("Capability to inject", ["forward_intent", "pace", "width", "press_resistance"],
                                format_func=lambda a: {"forward_intent": "Forward intent", "pace": "Pace",
                                                       "width": "Width", "press_resistance": "Press resistance"}[a],
                                key="cf_axis")
            level = st.slider("Level (capability units, roughly σ of real player variation)", -2.0, 3.0, 1.5, 0.5, key="cf_level")
            n_anc = st.slider("Real anchor phases to simulate over", 5, 30, 15, key="cf_n")
            if st.button("▶ Run simulation", key="cf_run"):
                with st.spinner(f"Rolling {n_anc} real tracked phases through the twin ..."):
                    r = cfsvc.run_signing_simulation({axis: level}, n_anchors=n_anc)
                ss.cf_result = r
            r = ss.get("cf_result")
            if r:
                if r.get("error"):
                    st.markdown(f'<div class="fcard" style="margin-top:10px;border-color:{P["danger"]}44">'
                                f'<span class="mut" style="font-size:12.5px">{r["error"]}</span></div>', unsafe_allow_html=True)
                else:
                    lo, hi = r["delta_danger_CI95"]
                    sign = "+" if r["mean_delta_danger"] >= 0 else ""
                    st.markdown(f'<div class="fcard" style="margin-top:10px">'
                                f'<div class="sclab">MEAN Δ DANGER · {r["phases"]} real phases · {r["anchor_seq"]}</div>'
                                f'<div class="scnum mag">{sign}{r["mean_delta_danger"]:.3f}</div>'
                                f'<div class="mut mono" style="font-size:10.5px;margin-top:2px">95% CI [{lo:+.3f}, {hi:+.3f}] · '
                                f'validity <b style="color:{P["tx"]}">{r["simulation_validity"]}</b> · '
                                f'{ui.badge("LIVE", "b-val") if r.get("live") else ""}</div>'
                                f'<div style="margin-top:8px;font-size:11.5px;color:{P["mut"]}">{r["epistemic"]}</div></div>',
                                unsafe_allow_html=True)


# ================= TACTICAL FIT =================
def tactical_fit():
    st.markdown('<div class="eyebrow">Tactical fit · solve a problem, not rank a market</div>', unsafe_allow_html=True)
    st.markdown("# Tactical Fit")
    need = st.selectbox("Our tactical need", list(fit.NEEDS))
    priorities = fit.NEEDS[need]
    st.markdown(f'<div class="mut mono" style="font-size:11px">priority stack: ' +
                " › ".join(f'<b class="cy">{S.CAP_AXES[a]["label"]}</b>' for a in priorities) + '</div>', unsafe_allow_html=True)
    f = st.columns(3)
    max_age = f[0].slider("Max age", 17, 38, 27, key="fit_age")
    max_val = f[1].slider("Max value €M", 1, 200, 80, key="fit_val")
    min_min = f[2].slider("Min 90s", 1, 30, 8, key="fit_min")
    fits = c_fits(ss.season, tuple(priorities), float(min_min), max_age, float(max_val), None)
    st.markdown('<div class="eyebrow" style="margin-top:12px">Best fits</div>', unsafe_allow_html=True)
    for r in fits[:12]:
        bd = "  ".join(f'<span class="mut">{b["axis"]}</span> <b class="mono" style="color:{P["cy"] if (b["pct"] or 0)>=75 else P["tx"]}">{int(b["pct"]) if b["pct"] is not None else "n/a"}</b>'
                       for b in r["breakdown"])
        cc = st.columns([3, 4, 1])
        cc[0].markdown(f'**{r["name"]}**  \n<span class="mut mono" style="font-size:11px">{r["league"]} · {r["age"]}y · '
                       f'€{r["value_m"]}M · {r["archetype"]}</span>', unsafe_allow_html=True)
        cc[1].markdown(f'<div style="padding-top:2px"><span class="sclab">FIT</span> '
                       f'<b class="mono cy" style="font-size:16px">{r["fit"]:.0f}</b><br>'
                       f'<span style="font-size:11px">{bd}</span></div>', unsafe_allow_html=True)
        if cc[2].button("Open", key=f"fit_{r['name']}"): goto("Player", r["name"])
    st.markdown(f'<div class="mut mono" style="font-size:10px;margin-top:12px">Fit = rank-weighted mean of the '
                f'relevant capability percentiles — decomposed above, never an opaque score. This reports capability '
                f'that <i>addresses</i> the need; whether a signing changes the rollout is the gated counterfactual.</div>',
                unsafe_allow_html=True)


# ================= EVIDENCE =================
def evidence():
    st.markdown('<div class="eyebrow">System · why you should trust this</div>', unsafe_allow_html=True)
    st.markdown("# Evidence")
    st.markdown(f'<p class="mut" style="max-width:640px">Every capability the product shows resolves its tier and '
                f'evidence from one machine-readable registry — so no screen can silently over-claim.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="eyebrow">Capability tiers</div>', unsafe_allow_html=True)
        for k, cap in R.CAPABILITIES.items():
            reg = R.get(k)
            st.markdown(f'<div class="kv"><span>{ui.badge(reg["tier"]["label"], reg["tier"]["badge"])} '
                        f'<span style="font-size:12.5px">{cap["headline"]}</span></span>'
                        f'<span class="mono mut" style="font-size:10.5px">{cap["metric"]}</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="eyebrow">System-level results</div>', unsafe_allow_html=True)
        for k, v in R.SYSTEM.items():
            st.markdown(f'<div class="kv"><span class="mut" style="font-size:12px">{v["of"]}</span>'
                        f'<b class="mono cy">{v["metric"]}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mut mono" style="font-size:10px;margin-top:16px">Design law · no score() · identity '
                f'never enters the backbone · counterfactual signing-impact stays UNPROVEN until sim-to-real (G3).</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:20px">Data source · auto-ingest</div>', unsafe_allow_html=True)
    if db.available():
        try:
            li = db.last_ingest()
        except Exception:
            li = None
        if li:
            ok = li["status"] == "ok"
            st.markdown(f'<div class="fpanel"><div class="kv"><span class="mut">Own database</span>'
                        f'{ui.badge("Connected", "b-val")}</div>'
                        f'<div class="kv"><span class="mut">Last ingest</span><b class="mono">{li["finished_at"] or li["started_at"]}</b></div>'
                        f'<div class="kv"><span class="mut">Status</span>{ui.badge(li["status"], "b-val" if ok else "b-exp")}</div>'
                        f'<div class="kv" style="border:0"><span class="mut">Records upserted</span><b class="mono cy">{li["n_upserted"]}</b></div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="mut mono" style="font-size:11px">Database connected, no ingest recorded yet.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="mut mono" style="font-size:11px">Reading directly from the HF dataset (no '
                    f'DATABASE_URL configured in this environment) — auto-ingest runs against the deployed db.</div>', unsafe_allow_html=True)


# ================= MEASURED · GEOMETRY =================
def measured_page():
    st.markdown('<div class="eyebrow">Measured · from geometry, not production</div>', unsafe_allow_html=True)
    st.markdown("# Measured geometry")
    st.markdown(f'<p class="mut" style="max-width:660px">Everywhere else, capabilities are <b style="color:{P["tx"]}">'
                f'estimated</b> from a box score. Here they are <b style="color:{P["cy"]}">measured</b> — space creation '
                f'and containment computed by the topology engine on real tracked states. Players are anonymous track '
                f'ids: identity never touches the geometry. This is the actual Fulcrum measurement the estimates approximate.</p>',
                unsafe_allow_html=True)
    seqs = c_measured_seqs()
    if not seqs:
        st.info("No precomputed sequences yet."); return
    seq = st.selectbox("Tracked sequence (SoccerNet 1080p broadcast)", seqs)
    m = c_measured(seq)
    st.markdown(f'<div class="mut mono" style="font-size:11px">{len(m["players"])} tracked players · {m["n_frames"]} '
                f'sampled frames · space creation {ui.tier_badge("space_creation")} containment {ui.tier_badge("containment")} '
                f'· evidence {ui.evidence_badge("measured_geometry")}</div>', unsafe_allow_html=True)

    ranked = sorted([p for p in m["players"] if p["space_creation"] is not None],
                    key=lambda p: -(p["space_creation"] or 0))
    names = [f'#{p["tid"]} · {p["role"]} · team {p["team"]}  (SC {p["space_creation"]:.0f}, {p["frames"]}f)' for p in ranked]
    pick = st.selectbox("Player (ranked by measured space creation)", range(len(ranked)),
                        format_func=lambda i: names[i])
    row = ranked[pick]
    prof = measured.measured_profile(row)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(charts.capability_radar(prof, accent=P["cy"]), unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="fcard"><div class="sclab">MEASURED · SPACE CREATION</div>'
                    f'<div class="scnum cy">{row["space_creation"]:.0f}<span class="mut" style="font-size:12px">/100</span></div>'
                    f'<div class="mut mono" style="font-size:10px;margin-top:2px">{row["frames"]} frames of coverage · '
                    f'{row["role"]}</div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        ui.capability_panel(prof)
    st.markdown(f'<div class="fcard" style="margin-top:12px"><span class="eyebrow">The contrast</span>'
                f'<div class="mut" style="font-size:12.5px;margin-top:6px">On the <b style="color:{P["tx"]}">Player</b> '
                f'page, a named player\'s space creation is <i>estimated</i> from crosses/key-passes/assists (a proxy). '
                f'Here it is <i>measured</i> from where the player actually moved defenders and opened geometry. Attaching '
                f'a named player\'s own tracking is what flips their profile from Estimated → Measured.</div></div>',
                unsafe_allow_html=True)


PAGES = {"Home": home, "Discover": discover, "Player": player_page, "Tactical Fit": tactical_fit,
         "Measured": measured_page, "Evidence": evidence}
PAGES.get(ss.page, home)()
