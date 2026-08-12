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


def inject_css():
    st.markdown(f"""<style>
      .stApp {{ background:{P['bg']}; color:{P['tx']}; }}
      .block-container {{ padding-top:2.4rem; max-width:1180px; }}
      h1,h2,h3,h4 {{ color:{P['hi']}; font-family:ui-monospace,'SF Mono',monospace; letter-spacing:.01em; }}
      .eyebrow {{ font-family:ui-monospace,monospace; font-size:10px; letter-spacing:.22em; text-transform:uppercase;
                  color:{P['cy']}; }}
      .mut {{ color:{P['mut']}; }} .cy {{ color:{P['cy']}; }} .mag {{ color:{P['mag']}; }}
      .fpanel {{ background:{P['panel']}; border:1px solid {P['line']}; border-radius:10px; padding:18px 20px; }}
      .fcard {{ background:{P['panel']}; border:1px solid {P['line']}; border-radius:10px; padding:14px 16px;
                transition:border-color .15s; }}
      .fcard:hover {{ border-color:{P['cy']}44; }}
      .badge {{ display:inline-block; font-family:ui-monospace,monospace; font-size:9.5px; letter-spacing:.1em;
                text-transform:uppercase; padding:2px 7px; border-radius:5px; border:1px solid; margin-right:5px; }}
      .kv {{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid {P['line']}55;
             font-size:13px; }}
      .mono {{ font-family:ui-monospace,monospace; }}
      .scnum {{ font-family:ui-monospace,monospace; font-size:26px; font-weight:700; line-height:1; }}
      .sclab {{ font-family:ui-monospace,monospace; font-size:8.5px; letter-spacing:.16em; color:{P['mut']};
                text-transform:uppercase; }}
      .stButton>button {{ background:{P['panel2']}; color:{P['tx']}; border:1px solid {P['line']};
                          border-radius:8px; font-family:ui-monospace,monospace; }}
      .stButton>button:hover {{ border-color:{P['cy']}; color:{P['cy']}; }}
      section[data-testid="stSidebar"] {{ background:{P['panel']}; border-right:1px solid {P['line']}; }}
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
    c1, c2, c3 = st.columns([3, 3, 1])
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
