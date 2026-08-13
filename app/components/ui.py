"""app.components.ui — Streamlit UI primitives. CSS for the restrained scientific-cyberpunk shell (§24/§47), plus
badge/scorecard/capability renderers that read tiers from fulcrum.registry so no component hard-codes a claim (§60).
"""
from __future__ import annotations
import streamlit as st
from app.config import settings as S
from fulcrum import registry as R
from app.visualization import charts

P = S.PALETTE

# badge classes -> (fg, border). Kept in sync with the published artifact's epistemic badges.
_BADGE = {
    "b-val":  (P["good"],  "#1f5c43"), "b-face": (P["amber"], "#5c471f"),
    "b-exp":  (P["mag"],   "#5c1f3a"), "b-cf":   (P["mag"],   "#5c1f3a"),
    "b-mut":  (P["mut"],   P["line"]),
}


MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace"
SANS = "'IBM Plex Sans', -apple-system, 'Helvetica Neue', Arial, sans-serif"


def inject_css():
    st.markdown(f"""<style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700;800&display=swap');

      .stApp {{
        background:
          radial-gradient(ellipse 800px 500px at 4% -6%, {P['cy']}0e 0%, transparent 55%),
          radial-gradient(ellipse 700px 500px at 100% 30%, {P['mag']}09 0%, transparent 55%),
          linear-gradient(180deg, {P['bg']} 0%, #060910 100%);
        color:{P['tx']}; font-family:{SANS};
      }}
      /* kill Streamlit's own chrome — header bar, Deploy button, hamburger menu, footer, the default top gap */
      header[data-testid="stHeader"] {{ display:none; }}
      #MainMenu {{ visibility:hidden; }}
      footer {{ display:none; }}
      .stAppDeployButton {{ display:none; }}
      div[data-testid="stDecoration"] {{ display:none; }}
      div[data-testid="stToolbar"] {{ display:none; }}
      .block-container {{ padding-top:2.6rem; padding-bottom:3rem; max-width:1160px; }}
      * {{ font-variant-numeric: tabular-nums; }}
      /* native inputs/selects — match the card system instead of Streamlit's generic form chrome */
      div[data-baseweb="select"] > div, div[data-baseweb="base-input"], .stTextInput input {{
        background:{P['panel2']} !important; border:1px solid {P['line']} !important; border-radius:12px !important;
        font-family:{MONO} !important; color:{P['tx']} !important;
      }}
      div[data-baseweb="select"] > div:hover, .stTextInput input:hover {{ border-color:{P['cy']}66 !important; }}
      div[data-baseweb="popover"] {{ background:{P['panel2']}; border:1px solid {P['line']}; }}
      li[role="option"] {{ font-family:{MONO}; }}
      li[role="option"]:hover, li[aria-selected="true"] {{ background:{P['cy']}18 !important; }}
      label p {{ font-family:{MONO} !important; font-size:11.5px !important; color:{P['mut']} !important;
                 letter-spacing:.03em; text-transform:uppercase; }}
      h1 {{ color:{P['hi']}; font-family:{SANS}; font-weight:700; letter-spacing:-.01em; font-size:1.65rem !important; }}
      h2,h3,h4 {{ color:{P['hi']}; font-family:{SANS}; font-weight:700; letter-spacing:-.01em; }}
      p, div, span, label {{ font-family:{SANS}; }}
      .eyebrow {{ font-family:{MONO}; font-size:10px; font-weight:600; letter-spacing:.24em; text-transform:uppercase;
                  color:{P['cy']}; border-left:2px solid {P['cy']}55; padding-left:7px; }}
      .mut {{ color:{P['mut']}; }} .cy {{ color:{P['cy']}; }} .mag {{ color:{P['mag']}; }}
      .fpanel {{ background:{P['panel']}; border:1px solid {P['line']}; border-radius:16px; padding:18px 20px; }}
      .fcard {{ background:{P['panel']}; border:1px solid {P['line']}; border-radius:16px; padding:14px 16px;
                transition:border-color .15s; }}
      .fcard:hover {{ border-color:{P['cy']}55; }}
      .badge {{ display:inline-block; font-family:{MONO}; font-size:9.5px; font-weight:500; letter-spacing:.1em;
                text-transform:uppercase; padding:2px 8px; border-radius:20px; border:1px solid; margin-right:5px; }}
      .kv {{ display:flex; justify-content:space-between; align-items:center; padding:6.5px 0;
             border-bottom:1px solid {P['line']}55; font-size:13px; }}
      .mono {{ font-family:{MONO}; }}
      .scnum {{ font-family:{MONO}; font-size:30px; font-weight:700; line-height:1; letter-spacing:-.01em; }}
      .sclab {{ font-family:{MONO}; font-size:8.5px; font-weight:600; letter-spacing:.17em; color:{P['mut']};
                text-transform:uppercase; }}
      .stButton>button {{ background:{P['panel2']}; color:{P['tx']}; border:1px solid {P['line']};
                          border-radius:12px; font-family:{MONO}; font-size:13px; transition:border-color .12s, color .12s; }}
      .stButton>button:hover {{ border-color:{P['cy']}; color:{P['cy']}; }}
      .stButton>button[kind="primary"] {{ background:{P['cy']}14; border-color:{P['cy']}; color:{P['cy']}; }}
      .stButton>button[kind="primary"]:hover {{ background:{P['cy']}22; }}
      section[data-testid="stSidebar"] {{ background:{P['panel']}; border-right:1px solid {P['line']}; }}
      .stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {P['line']}; }}
      .stTabs [data-baseweb="tab"] {{ font-family:{MONO}; color:{P['mut']}; }}
      .stTabs [aria-selected="true"] {{ color:{P['cy']} !important; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background-color:{P['cy']} !important; }}
      .stTabs [data-baseweb="tab-border"] {{ background-color:{P['line']} !important; }}
      .stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color:{P['cy']} !important; }}
      .stSlider div[data-testid="stTickBarMin"], .stSlider div[data-testid="stTickBarMax"] {{ color:{P['mut']}; }}
      table {{ border-collapse:collapse; width:100%; }}
      th {{ text-align:right; font-family:{MONO}; font-size:10px; font-weight:600; letter-spacing:.08em;
            color:{P['mut']}; text-transform:uppercase; padding:6px 10px; border-bottom:1px solid {P['line']}; }}
      td {{ font-family:{MONO}; padding:6px 10px; }}
      ::-webkit-scrollbar {{ width:9px; height:9px; }}
      ::-webkit-scrollbar-thumb {{ background:{P['line']}; border-radius:5px; }}
      ::-webkit-scrollbar-track {{ background:transparent; }}
    </style>""", unsafe_allow_html=True)


def badge(label: str, cls: str = "b-mut") -> str:
    fg, br = _BADGE.get(cls, _BADGE["b-mut"])
    return f'<span class="badge" style="color:{fg};border-color:{br}">{label}</span>'


def tier_badge(registry_key: str) -> str:
    """Badge for a capability METHOD's scientific tier, straight from the registry."""
    reg = R.get(registry_key)
    return badge(reg["tier"]["label"], reg["tier"]["badge"])


def evidence_badge(evidence_key: str) -> str:
    lab, cls = S.EVIDENCE_LABEL.get(evidence_key, ("—", "b-mut"))
    return badge(lab, cls)


def scorecard(sc: dict):
    """FIT / VALUE / UPSIDE / EVIDENCE as four separate, explainable numbers (§40 — never one master score)."""
    cols = st.columns(4)
    meta = [("FIT", "FIT", P["cy"], "attacking capability index"),
            ("VALUE", "VALUE", P["hi"], "cheaper than same-capability peers"),
            ("UPSIDE", "UPSIDE", P["hi"], "capability + age headroom"),
            ("EVIDENCE", "EVIDENCE", P["amber"], "minutes / data grade")]
    for c, (key, lab, col, tip) in zip(cols, meta):
        with c:
            st.markdown(f'<div class="fcard" title="{tip}"><div class="sclab">{lab}</div>'
                        f'<div class="scnum" style="color:{col}">{sc.get(key,"—")}</div></div>', unsafe_allow_html=True)


def capability_panel(profile: dict):
    """Every axis: label, bar, percentile, the method's registry tier, and this player's evidence grade (§9/§36)."""
    rows = []
    for ax, spec in S.CAP_AXES.items():
        p = profile[ax]
        insuff = p["pct"] is None or p["evidence"] == "insufficient_data"
        rows.append(
            f'<div style="display:grid;grid-template-columns:150px 190px 42px 1fr;align-items:center;gap:10px;'
            f'padding:7px 0;border-bottom:1px solid {P["line"]}44">'
            f'<span style="font-size:12.5px;color:{P["tx"]}">{p["label"]}</span>'
            f'{charts.bar(p["pct"], insufficient=insuff)}'
            f'<span class="mono" style="font-size:12px;color:{P["cy"] if (p["pct"] or 0)>=80 else P["mut"]}">'
            f'{int(p["pct"]) if p["pct"] is not None else "n/a"}</span>'
            f'<span style="font-size:10px">{tier_badge(p["registry_key"])}{evidence_badge(p["evidence"])}</span>'
            f'</div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def player_row(r: dict, prefix: str = ""):
    """Compact discover/list row: name, archetype, decomposed scorecard, open button."""
    c1, c2, c3 = st.columns([3, 2.6, 0.9])
    with c1:
        st.markdown(f'**{r["name"]}**  \n<span class="mut mono" style="font-size:11px">{r.get("league","")} · '
                    f'{r.get("age","?")}y · €{r.get("value_m","?")}M · {r.get("archetype","")}</span>',
                    unsafe_allow_html=True)
    with c2:
        sc = r.get("scorecard", {})
        chips = "  ".join(f'<span class="sclab">{k}</span> <b class="mono" style="color:{P["cy"] if k=="FIT" else P["tx"]}">{sc.get(k,"—")}</b>'
                          for k in ("FIT", "VALUE", "UPSIDE"))
        anom = r.get("anomaly")
        extra = f'  ·  <span class="mono" style="color:{P["cy"] if (anom or 0)>=20 else P["mut"]}">anomaly {anom:+.0f}</span>' if anom is not None else ""
        st.markdown(f'<div style="padding-top:4px">{chips}{extra}</div>', unsafe_allow_html=True)
    with c3:
        return st.button("Open", key=f"{prefix}_{r['name']}")
