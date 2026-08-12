"""fulcrum.products.recruit — the Scout DECISION surface: Diagnose -> Match (the non-counterfactual, ships-now core).

Not "who is good?" but "good for WHAT, in OUR system?". Two functions over the VALIDATED geometry stack — no twin,
no training, so it is available today (per architecture/COUNTERFACTUAL_RL.md, the Simulate step is the gated frontier
that later deepens Match into a counterfactual; Diagnose/Match/Explain do not depend on it):

  team_fingerprint(states, our_team)  -> a team's structural profile from its canonical states (Evaluate axes)
  diagnose(fingerprint[, ref])        -> ranked structural DEFICIENCIES (weak axes + where on the pitch)
  match(deficiencies, players, cap)   -> players whose measured CAPABILITY addresses each deficiency, ranked

Epistemic honesty: this reports "capability that addresses a diagnosed structural weakness", NOT "signing impact".
Whether inserting the player actually changes the rollout is the gated counterfactual (Simulate). Axes carry the
Evaluate tiers: space_creation/danger **validated**, containment *face-valid*, pressing/exposure *computed geometry*.
"""
import numpy as np
import fulcrum as F
from fulcrum import metrics as M


# deficiency axis -> the capability that would address it (both the Scout latent trait AND a per-90 proxy metric)
AXIS_CAPABILITY = {
    "off_ball_space_creation": {"trait": "off_ball_penetration_rate", "per90": ["crs90", "ast90", "sh90"], "hi": True},
    "buildup_press_resistance": {"trait": "press_resistance", "per90": ["ast90"], "hi": True},
    "final_third_threat":       {"trait": "progressive_intent", "per90": ["gls90", "sh90", "sot90"], "hi": True},
    "containment":              {"trait": None, "per90": ["tkl90", "int90"], "hi": True},
}


def _split(state_engine, our_team):
    """engine state {players:{tid:(team,(x,y))}, ball} -> Evaluate format {att, dfn, ball, att_ids, dfn_ids} with
    OUR team as att. Returns None if too sparse for the topology functions."""
    att, dfn, aid, did = [], [], [], []
    for tid, (team, xy) in state_engine.get("players", {}).items():
        (att if team == our_team else dfn).append(xy)
        (aid if team == our_team else did).append(tid)
    ball = state_engine.get("ball")
    if ball is None or len(att) < 5 or len(dfn) < 5:
        return None
    return {"att": att, "dfn": dfn, "ball": list(ball), "att_ids": aid, "dfn_ids": did}


def team_fingerprint(states_engine, our_team=0):
    """Aggregate a team's structural profile over its canonical states. Attack axes computed with our team as att;
    defence axes with our team as dfn (the opponent attacking). -> {axis: value, n, zones}."""
    sc, dg, cont, dg_conc, prog = [], [], [], [], []
    holes_att = []                                            # where OUR attack finds space (attacking exposure)
    for se in states_engine:
        a = _split(se, our_team)
        if a is not None:
            ss = M.state_summary(a)
            if ss: sc.append(ss["sc"]); dg.append(ss["danger"])
            att = np.asarray(a["att"], float)
            prog.append(float(att[:, 0].mean()))             # mean attacking x (progression proxy, 0..105)
            _, _, _, hs = F.find_holes(att, np.zeros_like(att), np.asarray(a["dfn"], float),
                                       np.zeros_like(np.asarray(a["dfn"], float)), np.asarray(a["ball"], float), top=1)
            if hs: holes_att.append((hs[0]["x"] if "x" in hs[0] else hs[0].get("cx", 52), hs[0]["score"]))
        d = _split(se, 1 - our_team)                          # opponent as att -> our containment / danger conceded
        if d is not None:
            ss = M.state_summary(d)
            if ss: dg_conc.append(ss["danger"])
            cv = M.containment(d)
            if cv: cont.append(float(np.mean(list(cv.values()))))
    def mean(x): return round(float(np.mean(x)), 3) if x else None
    fp = {"off_ball_space_creation": mean(sc), "final_third_threat": mean(dg), "progression": mean(prog),
          "containment": mean(cont), "danger_conceded": mean(dg_conc),
          "n_attack_states": len([1 for se in states_engine if _split(se, our_team)]),
          "n_defend_states": len([1 for se in states_engine if _split(se, 1 - our_team)])}
    # where conceded: mean x of top attacking hole against us
    if holes_att:
        xs = [h[0] for h in holes_att]
        fp["threat_zone_x"] = round(float(np.mean(xs)), 1)
    return fp


def diagnose(fp, ref=None):
    """Rank a fingerprint's structural DEFICIENCIES. `ref` = {axis: (mean,std)} league benchmark; without it, flags
    the axes lowest vs a neutral prior. -> [{axis, value, z, severity}] worst first. Buildup-press-resistance is
    inferred from low progression + low space creation (a proxy until tracking-derived press-resistance is wired)."""
    axes = {"off_ball_space_creation": fp.get("off_ball_space_creation"),
            "final_third_threat": fp.get("final_third_threat"),
            "containment": fp.get("containment"),
            "progression": fp.get("progression")}
    out = []
    for ax, v in axes.items():
        if v is None: continue
        if ref and ax in ref:
            mu, sd = ref[ax]; z = (v - mu) / (sd + 1e-9)
        else:
            z = None
        out.append({"axis": ax, "value": v, "z": (round(z, 2) if z is not None else None)})
    # rank: with ref, most-negative z first; without, lowest raw value first (per-axis, so scale-normalise by max)
    if ref:
        out.sort(key=lambda r: (r["z"] if r["z"] is not None else 0))
    else:
        mx = {r["axis"]: r["value"] for r in out}
        out.sort(key=lambda r: r["value"])
    for i, r in enumerate(out):
        r["severity"] = "primary" if i == 0 else ("secondary" if i == 1 else "minor")
    # translate the worst axis into a buildup label if it's a possession axis
    if out and out[0]["axis"] in ("off_ball_space_creation", "progression"):
        out[0]["reads_as"] = "build-up struggles to create/advance — vulnerable to compression/press"
    return out


def season_percentiles(records, season, metrics):
    """Per-player capability percentiles WITHIN a given season's pool (metrics computed on that season's minutes).
    records: multi-season list (each has 'name','season',<metrics>). -> {name: {metric: pct 0..100}}."""
    pool = [r for r in records if r.get("season") == season]
    out = {}
    for m in metrics:
        v = np.array([float(r.get(m, 0) or 0) for r in pool])
        if len(v) == 0: continue
        o = v.argsort(); pr = np.empty(len(v)); pr[o] = np.linspace(0, 100, len(v))
        neg = m in ("fls90",)
        for i, r in enumerate(pool):
            out.setdefault(r["name"], {})[m] = int(round(100 - pr[i] if neg else pr[i]))
    return out


def player_trajectory(records, name, metrics):
    """A player's metric values across the seasons present -> {season: {metric: value}}. The multi-season view:
    trend/aging/projection all read off this (e.g. gls90 23/24->24/25->25/26)."""
    traj = {}
    for r in records:
        if r.get("name") == name:
            traj[r["season"]] = {m: round(float(r.get(m, 0) or 0), 3) for m in metrics}
    return dict(sorted(traj.items()))


def match(deficiencies, players, capability_of, top=8):
    """Rank players whose measured capability addresses each diagnosed deficiency.
    players: iterable of records with a 'name'. capability_of(player, axis) -> percentile 0..100 (caller supplies,
    from Scout traits or per-90 pool percentiles). -> {axis: [ {name, fit, drivers} ]}."""
    result = {}
    for d in deficiencies:
        ax = d["axis"]
        cap = AXIS_CAPABILITY.get(ax)
        if not cap: continue
        ranked = []
        for p in players:
            pct = capability_of(p, ax)
            if pct is None: continue
            ranked.append({"name": p.get("name", "?"), "team": p.get("team", ""), "fit": int(round(pct)),
                           "addresses": ax, "via": cap["trait"] or "+".join(cap["per90"])})
        ranked.sort(key=lambda r: -r["fit"])
        result[ax] = ranked[:top]
    return result
