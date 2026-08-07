"""fulcrum.scout — recruitment intelligence: player ARCHETYPES + role-adjusted PRODUCTION + surplus-value BOARDS.

The Recruitment pillar, rebuilt as a durable package module (was scratchpad-only, lost to a reset). Works on
ATTRIBUTE vectors (per-90, role-aware stats); **identity (name/team/age) is a downstream LABEL, never a model
input** — the same agnostic discipline as the rest of Fulcrum. The data source is pluggable (FBref, stoichima):
pass `players` as a list of dicts with numeric attributes + optional {name, team, pos, age, nineties}.

Honest on cost: with no headless market-value source, surplus value is proxied by **age + minutes** (young =
resale upside; benched-but-productive = underrated by their own club) — this captures upside, not a fee. A
client of the decision loop: "fill the role the planner says we're missing".
"""
from __future__ import annotations
import numpy as np

# Default FBref attribute vocabulary (attacking + creation + DEFENCE, per-90). Override via `feat`.
FBREF_FEATURES = ["gls90", "ast90", "sh90", "sot90", "finishing", "tkl90", "int90", "crs90", "fls90"]
ROLE_WEIGHTS = {   # role-adjusted production: a defender competes on tackles/interceptions, not goals
    "FW": {"gls90": 1.0, "sot90": 0.6, "finishing": 0.5, "ast90": 0.4, "sh90": 0.3},
    "MF": {"ast90": 0.8, "gls90": 0.4, "sh90": 0.4, "tkl90": 0.6, "int90": 0.6, "crs90": 0.4},
    "DF": {"tkl90": 1.0, "int90": 1.0, "crs90": 0.3, "ast90": 0.2, "fls90": -0.3},
}


def _matrix(players, feat):
    return np.array([[float(p.get(f, 0) or 0) for f in feat] for p in players], float)


def _z(A):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    return (A - mu) / sd, mu, sd


def _kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), min(k, len(X)), replace=False)].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(len(C)):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


def archetypes(players, feat=FBREF_FEATURES, k=6):
    """Cluster players into role archetypes on their attribute vectors (identity untouched)."""
    A = _matrix(players, feat); Z, mu, sd = _z(A); lab, C = _kmeans(Z, k)
    out = []
    for j in range(len(C)):
        idx = np.where(lab == j)[0]
        if not len(idx):
            continue
        center = C[j] * sd + mu
        dom = ", ".join(f"{feat[i]} {center[i]:.2f}" for i in np.argsort(-C[j])[:3])
        ex = [players[i].get("name", "?") for i in idx[np.argsort(np.linalg.norm(Z[idx] - C[j], axis=1))[:3]]]
        out.append({"archetype": j, "n": int(len(idx)), "signature": dom, "exemplars": ex})
    return out


def role_production(players, feat=FBREF_FEATURES, weights=ROLE_WEIGHTS):
    """Role-adjusted production (z-scored within position) + within-role percentile. -> (prod, pct, roles)."""
    A = _matrix(players, feat); Z, _, _ = _z(A)
    roles = [str(p.get("pos", "")).split(",")[0] or "MF" for p in players]
    prod = np.array([sum(wt * Z[i, feat.index(d)] for d, wt in weights.get(roles[i], weights["MF"]).items())
                     for i in range(len(players))])
    pct = np.zeros(len(players))
    for r in set(roles):
        m = np.array([roles[i] == r for i in range(len(players))])
        if m.sum() > 5:
            pct[m] = 100.0 * prod[m].argsort().argsort() / (m.sum() - 1)
    return prod, pct, roles


def similar(players, name, feat=FBREF_FEATURES, k=5, younger=False):
    """Attribute-nearest players to `name` (optionally only younger — the 'cheaper alternative' scan)."""
    A = _matrix(players, feat); Z, _, _ = _z(A); names = [p.get("name") for p in players]
    if name not in names:
        return []
    qi = names.index(name); d = np.linalg.norm(Z - Z[qi], axis=1); d[qi] = 1e9
    ages = np.array([p.get("age", 99) or 99 for p in players])
    cand = [i for i in range(len(players)) if (not younger or ages[i] < ages[qi])]
    cand.sort(key=lambda i: d[i])
    return [{"name": names[i], "age": int(ages[i]), "team": players[i].get("team", ""), "sim": round(float(d[i]), 2)}
            for i in cand[:k]]


def boards(players, feat=FBREF_FEATURES, weights=ROLE_WEIGHTS):
    """The surplus-value boards: young producers (per role) + underused gems (high production, benched)."""
    prod, pct, roles = role_production(players, feat, weights)
    ages = np.array([p.get("age", 99) or 99 for p in players])
    mins = np.array([p.get("nineties", 0) or 0 for p in players])
    def row(i):
        return {"name": players[i].get("name"), "pos": roles[i], "age": int(ages[i]),
                "team": players[i].get("team", ""), "nineties": round(float(mins[i]), 1), "production_pct": round(float(pct[i]))}
    young = {}
    for r in ["FW", "MF", "DF"]:
        c = [i for i in range(len(players)) if roles[i] == r and 16 <= ages[i] <= 21 and pct[i] >= 75 and mins[i] >= 8]
        c.sort(key=lambda i: -pct[i]); young[r] = [row(i) for i in c[:5]]
    under = [i for i in range(len(players)) if pct[i] >= 90 and 5 <= mins[i] < 10]; under.sort(key=lambda i: -pct[i])
    return {"young_producers": young, "underused_gems": [row(i) for i in under[:8]]}


# Tracking-derived features (from fulcrum.metrics over a tracking corpus). Identity stays a downstream label.
TRACKING_FEATURES = ["sc", "containment"]   # per-state off-ball space creation / danger suppressed by positioning


def space_creators(players, min_states=120):
    """Differentiated recruitment board on TOPOLOGY metrics FBref production can't see (`fulcrum.metrics`):
    off-ball SPACE CREATORS (high `sc`) and CONTAINERS (high `containment`), attribute-only, with a surplus tilt
    (young / low-minutes = under-priced). `players` = dicts carrying per-state `sc` / `containment` + `states`
    (from tracking) and optional {name, pos, team, age, nineties}. Players below `min_states` are too noisy to rank.
    -> {space_creators, containers, undervalued_creators}."""
    elig = [p for p in players if (p.get("states", 0) or 0) >= min_states]
    def row(p, key):
        return {"name": p.get("name", "?"), "pos": str(p.get("pos", "") or ""), "team": p.get("team", ""),
                "age": int(p.get("age", 0) or 0), "nineties": round(float(p.get("nineties", 0) or 0), 1),
                key: round(float(p.get(key, 0) or 0), 3), "states": int(p.get("states", 0) or 0)}
    creators = sorted(elig, key=lambda p: -(p.get("sc", 0) or 0))
    containers = sorted(elig, key=lambda p: -(p.get("containment", 0) or 0))
    # surplus lens: strong creators who are KNOWN-young or KNOWN-under-used (needs a real age/minutes roster join —
    # absent those cost proxies this is correctly empty, never guessed). Note `x or 0` keeps a true 0, not a default.
    cut = np.quantile([p.get("sc", 0) or 0 for p in elig], 0.75) if len(elig) > 4 else 0
    under = [p for p in creators if (p.get("sc", 0) or 0) >= cut
             and (0 < (p.get("age") or 0) <= 23 or 0 < (p.get("nineties") or 0) < 10)]
    return {"space_creators": [row(p, "sc") for p in creators[:8]],
            "containers": [row(p, "containment") for p in containers[:8]],
            "undervalued_creators": [row(p, "sc") for p in under[:6]]}


# ---------------------------------------------------------------------------------------------------------------
# CROSS-LEAGUE TRANSLATION (Moneyball) — project production across leagues, style-aware, with an outlier board.
# Strength is style-BLIND (a single index per league from movers); Fulcrum's edge is the STYLE-fit layer on top,
# which most translation models lack and which explains the outliers you don't expect to translate. Identity stays
# a label. Works on ANY multi-season records (FBref etc.) -> functional on recent data by construction (refreshable).
# ---------------------------------------------------------------------------------------------------------------
def _prod(r, keys=("gls90", "ast90")):
    return sum(float(r.get(k, 0) or 0) for k in keys)   # attacking output per 90 (default goals+assists; xg90+xa90 = less noisy)


def find_movers(records, min_nineties=8.0, prod=("gls90", "ast90")):
    """Players who changed league between consecutive seasons. `records` = dicts with {name, league, season,
    <prod fields>, nineties[, age, pos]}. `prod` picks the production metric (e.g. ("xg90","xa90") for xG+xA).
    -> list of movers {name, from, to, before, after, age, pos}."""
    from collections import defaultdict
    by_player = defaultdict(list)
    for r in records:
        if (r.get("nineties", 0) or 0) >= min_nineties and r.get("name") and r.get("league") and r.get("season"):
            by_player[r["name"]].append(r)
    movers = []
    for name, rs in by_player.items():
        rs = sorted(rs, key=lambda r: str(r["season"]))
        for a, b in zip(rs[:-1], rs[1:]):
            if a["league"] != b["league"]:
                movers.append({"name": name, "from": a["league"], "to": b["league"],
                               "before": _prod(a, prod), "after": _prod(b, prod), "age": b.get("age", 0), "pos": b.get("pos", "")})
    return movers


def fit_league_strength(movers, ridge=0.3, min_before=0.15):
    """Least-squares league STRENGTH index from movers' log production ratios (higher index = harder league):
    for a move from->to, log(after/before) ≈ strength[from] - strength[to]. Solved over the whole mover network,
    ridge-regularised for sparse leagues, centred to mean 0. -> {league: strength}. Style-blind by design."""
    m = [x for x in movers if x["before"] >= min_before and x["after"] >= min_before]
    leagues = sorted({x["from"] for x in m} | {x["to"] for x in m})
    if len(leagues) < 2 or len(m) < len(leagues):
        return {lg: 0.0 for lg in leagues}
    idx = {lg: i for i, lg in enumerate(leagues)}
    A = np.zeros((len(m) + 1, len(leagues))); y = np.zeros(len(m) + 1)
    for i, x in enumerate(m):
        A[i, idx[x["from"]]] = 1.0; A[i, idx[x["to"]]] = -1.0
        y[i] = np.log(x["after"] / x["before"])
    A[len(m), :] = 1.0 / len(leagues)   # anchor: mean strength = 0 (identifiability)
    A = np.vstack([A, np.sqrt(ridge) * np.eye(len(leagues))]); y = np.concatenate([y, np.zeros(len(leagues))])
    s, *_ = np.linalg.lstsq(A, y, rcond=None)
    s = s - s.mean()
    return {lg: round(float(s[idx[lg]]), 3) for lg in leagues}


def style_distance(a, b, style_vecs):
    """Playing-style distance between two leagues/teams from style vectors (possession, pressing, directness…),
    z-normalised, in [0,1]. `style_vecs` = {name: np.array}. Higher = less similar -> weaker translation prior."""
    if a not in style_vecs or b not in style_vecs:
        return None
    V = np.array(list(style_vecs.values())); mu, sd = V.mean(0), V.std(0) + 1e-6
    za, zb = (style_vecs[a] - mu) / sd, (style_vecs[b] - mu) / sd
    return round(float(np.tanh(np.linalg.norm(za - zb) / max(len(za), 1) ** 0.5)), 3)


def project(before, from_l, to_l, strength, style_vecs=None, style_pen=0.30):
    """Project a per-90 production from `from_l` into `to_l`. strength handles difficulty; style (if available)
    applies an extra discount for a stylistic mismatch (the reason similar leagues translate cleaner). -> projected."""
    gap = strength.get(from_l, 0.0) - strength.get(to_l, 0.0)   # >0 if moving to an easier league
    proj = before * np.exp(gap)
    if style_vecs is not None:
        d = style_distance(from_l, to_l, style_vecs)
        if d is not None:
            proj *= np.exp(-style_pen * d)
    return round(float(proj), 3)


def translation_board(movers, strength, style_vecs=None, min_before=0.15):
    """The outlier board: for each mover, residual = actual after-move production − projected. Positive = translated
    ABOVE expectation (the gem you didn't expect); negative = below (the bust). Ranked. Attacking output only —
    defensive translation is not reliably measurable from box score (role-preservation caveat)."""
    rows = []
    for x in movers:
        if x["before"] < min_before:
            continue
        proj = project(x["before"], x["from"], x["to"], strength, style_vecs)
        rows.append({"name": x["name"], "move": f"{x['from']} -> {x['to']}", "age": x.get("age", 0),
                     "before": round(x["before"], 2), "after": round(x["after"], 2), "projected": proj,
                     "residual": round(x["after"] - proj, 2), "harder": strength.get(x["to"], 0) > strength.get(x["from"], 0)})
    rows.sort(key=lambda r: -r["residual"])
    return {"over_translators": rows[:10], "under_translators": rows[-8:][::-1], "n_movers": len(rows)}


def translation_candidates(records, target_league, strength, season=None, style_vecs=None,
                           min_nineties=10.0, min_prod=0.10, top=15, prod=("gls90", "ast90")):
    """PROSPECTIVE projection: for players NOT already in `target_league` (optionally restricted to a `season`
    snapshot, e.g. the latest for recent-data use), predict their attacking output IF they moved to the target,
    ranked by projected production. This is the forward-looking scouting query — 'who would translate best to X'."""
    rows = []
    for r in records:
        if (season is not None and r.get("season") != season) or r.get("league") == target_league:
            continue
        if (r.get("nineties", 0) or 0) < min_nineties:
            continue
        before = _prod(r, prod)
        if before < min_prod:
            continue
        proj = project(before, r["league"], target_league, strength, style_vecs)
        rows.append({"name": r["name"], "from": r["league"], "season": r.get("season"),
                     "current": round(before, 2), "projected": proj, "delta": round(proj - before, 2)})
    rows.sort(key=lambda x: -x["projected"])
    return rows[:top]


def translation_report(records, style_vecs=None):
    """Consolidated cross-league translation: league strength table + the outlier board. `records` = multi-season
    player-season dicts; `style_vecs` optional league style vectors (Fulcrum's style-fit layer)."""
    movers = find_movers(records)
    strength = fit_league_strength(movers)
    return {"n_movers": len(movers), "league_strength": dict(sorted(strength.items(), key=lambda kv: -kv[1])),
            "style_layer": style_vecs is not None, **translation_board(movers, strength, style_vecs)}


def scout_report(players, feat=FBREF_FEATURES):
    """Consolidated recruitment report — archetypes + surplus-value boards. Cost proxied by age + minutes.
    If players carry tracking metrics (`sc`/`containment`), adds the off-ball `space_creators` board."""
    rep = {"n_players": len(players),
           "cost_proxy": "age + minutes (no headless market-value source) — captures upside, not fee",
           "archetypes": archetypes(players, feat),
           **boards(players, feat)}
    if any((p.get("sc") is not None or p.get("containment") is not None) for p in players):
        rep["off_ball"] = space_creators(players)
    return rep


def derive_insights(records, feat=FBREF_FEATURES, outputs=("gls90", "ast90"), k_style=3, ridge=1.0):
    """MODEL-DERIVED insight from AGGREGATE stats ALONE (no spatial data) — the structure the raw db doesn't show.
    Covers every player who has stats (the whole 15k), complementing the tracking/360 off-ball layer. Per player:
      residual[out] : standardized EXPECTED-vs-ACTUAL (over/under-performance the rest of the profile doesn't predict
                      — the model's 'surprise'; a clinical finisher scores high positive on gls90)
      style[:k]     : latent STYLE embedding (SVD of the z-scored profile) — powers 'players like X'
      typicality    : how ordinary the profile is (low = anomalous, hard to replace / system-specific)
    Plus `style_axes` = what each latent axis means (top-loading metrics). HONEST BOUNDARY: this only RECOMBINES the
    existing metrics — it cannot recover off-ball/spatial information the stats do not encode (that is the 360/tracking
    layer). `feat`/`outputs` are metric keys present on the records."""
    P = [r for r in records if all(r.get(k) is not None for k in feat)]
    if len(P) < 8:
        return {"players": [], "style_axes": [], "n": len(P)}
    X = np.array([[float(r[k]) for k in feat] for r in P], float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    resid = {}
    for o in outputs:
        if o not in feat:
            continue
        j = feat.index(o); cols = [i for i in range(len(feat)) if i != j]
        A = Z[:, cols]; y = Z[:, j]
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)   # ridge: predict o from the REST
        r = y - A @ w
        resid[o] = r / (r.std() + 1e-9)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    emb = U[:, :k_style] * S[:k_style]
    recon = U[:, :k_style] @ np.diag(S[:k_style]) @ Vt[:k_style]
    rec_err = np.sqrt(((Z - recon) ** 2).sum(1))
    typ = -(rec_err - rec_err.mean()) / (rec_err.std() + 1e-9)              # high = ordinary, low = anomalous
    players = [{"player": P[i].get("name", "?"), "team": P[i].get("team", ""),
                "residual": {o: round(float(resid[o][i]), 2) for o in resid},
                "style": [round(float(v), 2) for v in emb[i]], "typicality": round(float(typ[i]), 2)}
               for i in range(len(P))]
    axes = [{"axis": a, "top_features": [(feat[k], round(float(Vt[a, k]), 2))
             for k in np.argsort(-np.abs(Vt[a]))[:4]]} for a in range(k_style)]
    return {"players": players, "style_axes": axes, "n": len(P),
            "note": "model-derived from aggregate stats; recombines existing metrics, does not add spatial info"}


_METRIC_PHRASE = {"gls90": "goalscoring", "ast90": "chance creation", "sh90": "shot volume",
                  "sot90": "shooting accuracy", "finishing": "finishing", "tkl90": "tackling",
                  "int90": "interceptions", "crs90": "crossing", "fls90": "discipline"}


def _archetype_name(z, feat):
    """Role label from a PLAYER's OWN z-profile (not a cluster's), by RANKING candidate archetypes on that player's
    dominant traits — so a shooter in a mixed cluster is named a shooter, not inheriting the cluster's label."""
    def v(m): return float(z[feat.index(m)]) if m in feat else -9.0
    cand = {"ball-winner": v("tkl90") + v("int90"),
            "wide creator": v("crs90") + 0.5 * v("ast90"),
            "playmaker": v("ast90") - 0.3 * v("sh90"),
            "clinical poacher": v("finishing") + 0.5 * v("gls90") - 0.4 * v("sh90"),
            "volume scorer": v("gls90") + v("sh90") + v("sot90")}
    top = max(cand, key=cand.get)
    return top if cand[top] > 0.35 else "all-rounder"


_BBALL_PHRASE = {"pts": "scoring", "ast": "playmaking", "reb": "rebounding", "stl": "steals", "blk": "rim protection",
                 "fg3_pct": "3-point shooting", "fg_pct": "finishing", "tov": "ball security", "usg": "usage",
                 "ts_pct": "shooting efficiency", "ast_pct": "creation load", "min": "minutes"}


def _archetype_name_bball(z, feat):
    """Basketball role label from a PLAYER's OWN z-profile, RANKED on that player's dominant traits (so Curry reads as
    a shooter and Jokić as a playmaking hub, rather than both inheriting one cluster label)."""
    def v(m): return float(z[feat.index(m)]) if m in feat else -9.0
    cand = {"rim-protecting big": v("blk") + v("reb") - 0.5 * v("fg3_pct"),
            "primary playmaker": v("ast") + 0.4 * v("usg") - 0.4 * v("blk"),
            "3-and-D wing": v("fg3_pct") + v("stl") - 0.4 * v("usg"),
            "floor spacer": v("fg3_pct") - 0.3 * v("usg") - 0.4 * v("ast"),
            "high-usage scorer": v("pts") + v("usg") - 0.4 * v("ast"),
            "connector / glue": 0.6 * (v("ast") + v("stl")) - v("usg")}
    top = max(cand, key=cand.get)
    return top if cand[top] > 0.3 else "rotation piece"


# sport registries: same engine, swappable vocabulary (mirrors the platform's driver registry)
_ARCH = {"football": _archetype_name, "basketball": _archetype_name_bball}
_PHRASE = {"football": _METRIC_PHRASE, "basketball": _BBALL_PHRASE}


def _ord(n):
    n = int(n); suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _describe(p, ph=_METRIC_PHRASE):
    """Compose a scouting profile paragraph from the model's OWN derived signals (grounded, deterministic)."""
    s = []
    lead = f"A {p['archetype']}"
    if p["strengths"]:
        m, pc = p["strengths"][0]; lead += f" with {'elite' if pc >= 90 else 'strong'} {ph.get(m, m)} ({_ord(pc)} pct in role)"
    s.append(lead + ".")
    if len(p["strengths"]) > 1:
        m, pc = p["strengths"][1]; s.append(f"Also reads {_ord(pc)} for {ph.get(m, m)}.")
    for o, rv in p.get("residual", {}).items():
        if rv >= 1.0: s.append(f"Beats his expected {ph.get(o, o)} (+{rv:.1f}σ) — end product above what the profile predicts.")
        elif rv <= -1.0: s.append(f"Lags expected {ph.get(o, o)} ({rv:.1f}σ) — output trails the underlying play.")
    if p["weakness"]:
        m, pc = p["weakness"][0]; s.append(f"Limited {ph.get(m, m)} ({_ord(pc)}).")
    if p["typicality"] <= -1.0: s.append("An unusual, hard-to-replace shape.")
    elif p["typicality"] >= 1.2: s.append(f"A textbook {p['archetype']}.")
    if p["comparables"]: s.append(f"Profile echoes {', '.join(p['comparables'][:2])}.")
    return " ".join(s)


def player_profiles(records, feat=FBREF_FEATURES, outputs=("gls90", "ast90"), k_arch=6, top_compare=3, sport="football"):
    """Rich MODEL-DERIVED player intelligence from AGGREGATE stats — for the whole db, no spatial data. Per player:
    discovered archetype, ROLE-RELATIVE percentiles (vs same-role peers, far more meaningful than raw/global),
    signature strengths/weaknesses, over/under-performance residuals, latent style, style comparables, typicality,
    and a generated scouting DESCRIPTION composed from those signals (grounded in the numbers — not a lookup, not an
    LLM guess). Honest boundary: recombines existing metrics; adds no off-ball/spatial info (that is the 360/tracking layer)."""
    P = [r for r in records if all(r.get(k) is not None for k in feat)]
    if len(P) < 10:
        return {"players": [], "n": len(P)}
    A = _matrix(P, feat); Z, mu, sd = _z(A); lab, C = _kmeans(Z, k_arch)
    namer = _ARCH.get(sport, _archetype_name)
    di = derive_insights(P, feat, outputs)
    resid = {p["player"]: p["residual"] for p in di["players"]}
    style = {p["player"]: p["style"] for p in di["players"]}
    typ = {p["player"]: p["typicality"] for p in di["players"]}
    out = []
    for i, r in enumerate(P):
        j = int(lab[i]); peers = np.where(lab == j)[0]
        pct = {m: (round(100.0 * float((A[peers, feat.index(m)] < A[i, feat.index(m)]).mean()))
                   if len(peers) > 3 else 50) for m in feat}
        gz = {m: float(Z[i, feat.index(m)]) for m in feat}      # global z — a strength must also be high in absolute terms
        ranked = sorted(feat, key=lambda m: -pct[m])
        strengths = [(m, pct[m]) for m in ranked if pct[m] >= 70 and gz[m] >= 0.3][:2]
        weakness = [(m, pct[m]) for m in ranked[::-1] if pct[m] <= 30 and gz[m] <= -0.3][:1]
        d = np.linalg.norm(Z - Z[i], axis=1); d[i] = 1e9
        comps = [P[c].get("name", "?") for c in np.argsort(d)[:top_compare]]
        name = r.get("name", "?")
        prof = {"player": name, "team": r.get("team", ""), "archetype": namer(Z[i], feat),
                "percentiles": pct, "strengths": strengths, "weakness": weakness,
                "residual": resid.get(name, {}), "style": style.get(name, []),
                "typicality": typ.get(name, 0.0), "comparables": comps}
        prof["description"] = _describe(prof, _PHRASE.get(sport, _METRIC_PHRASE))
        out.append(prof)
    from collections import Counter
    return {"players": out, "archetypes": dict(Counter(p["archetype"] for p in out)), "n": len(P)}


# stats -> positional-spatial estimator (the 'estimated' coverage tier). Held-out (stats_to_geometry_bridge.json):
# ADVANCEMENT / defenders-beaten IS estimable (corr 0.57); space weak (0.23); DANGER is NOT (0.04) -> must be measured.
_SPATIAL_CONF = {"beaten": 0.57, "space": 0.23, "danger": 0.04}


def fit_spatial_estimator(records, stat_feat, target_key="beaten_measured"):
    """Fit ridge stats -> a MEASURED positional value on players who have BOTH box-score stats AND a 360/tracking
    label (`target_key`). Intended for ADVANCEMENT only — the one spatial signal recoverable from stats. Returns an
    estimator dict, or None if too few labelled players."""
    P = [r for r in records if r.get(target_key) is not None and all(r.get(k) is not None for k in stat_feat)]
    if len(P) < 20:
        return None
    X = np.array([[float(r[k]) for k in stat_feat] for r in P], float); y = np.array([float(r[target_key]) for r in P])
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xz = (X - mu) / sd
    ym, ysd = y.mean(), y.std() + 1e-9
    w = np.linalg.solve(Xz.T @ Xz + 1.0 * np.eye(Xz.shape[1]), Xz.T @ ((y - ym) / ysd))
    return {"w": w, "mu": mu, "sd": sd, "feat": list(stat_feat), "ym": float(ym), "ysd": float(ysd),
            "confidence": _SPATIAL_CONF["beaten"], "n": len(P)}


def estimate_spatial(records, estimator):
    """Apply a fitted estimator to STAT-ONLY players -> estimated advancement, TAGGED with held-out confidence. The
    'estimated' coverage tier between pure-stats and freeze-frame — never presented as a measurement (danger, the
    topological signal, is NOT estimable and is intentionally absent here)."""
    if not estimator:
        return []
    feat = estimator["feat"]; out = []
    for r in records:
        if not all(r.get(k) is not None for k in feat):
            continue
        x = (np.array([float(r[k]) for k in feat], float) - estimator["mu"]) / estimator["sd"]
        est = float(x @ estimator["w"]) * estimator["ysd"] + estimator["ym"]
        out.append({"player": r.get("name", "?"), "estimated_advancement": round(est, 2),
                    "confidence": estimator["confidence"], "status": "estimated (positional only; not measured)"})
    return out
