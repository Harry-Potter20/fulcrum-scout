"""Multi-SEASON football db — per-90 metrics for several seasons (25/26, 24/25, 23/24, ...), so Scout can compute
capability/projections on a SEASON basis, not just the current per-90 snapshot. Season IDs are auto-resolved per
league from Sofascore's /seasons endpoint (no hardcoding). Adds current market value/age/foot per unique player.
Local (residential IP, curl_cffi). Writes football_player_seasons.json (leaves moneyball_per90_25_26.json intact).

Env: SEASONS ("25/26,24/25,23/24"), TOPN (per league-season), LEAGUES (optional CSV of league names to include).
"""
import os, sys, json, time, datetime
from curl_cffi import requests as cr

B = "https://api.sofascore.com/api/v1"
# league -> unique-tournament id. IDs verified against Sofascore's /search + /standings (not guessed — a wrong id
# either 404s or, worse, silently pulls a same-named different competition, e.g. "Eliteserien"/"Allsvenskan" also
# name Norwegian/Swedish FLOORBALL and HANDBALL top divisions; ids 20/40 below are confirmed football via team names).
UT = {"Premier League": 17, "La Liga": 8, "Serie A": 23, "Bundesliga": 35, "Ligue 1": 34,
      "Championship": 18, "Eredivisie": 37, "Primeira Liga": 238,
      # moneyball expansion — proven cheap-talent / undervalued markets, less densely scouted than the big 5
      "Belgian Pro League": 38, "Austrian Bundesliga": 45, "Swiss Super League": 215,
      "Danish Superliga": 39, "Croatian HNL": 170,
      "Eliteserien": 20, "Allsvenskan": 40, "MLS": 242, "Brasileirao": 325, "Liga Profesional Argentina": 155}
# leagues whose season runs within one calendar year (Sofascore labels them "2026", not "25/26") — mapped to the
# double-year bucket they most overlap so they merge into the SAME season pool the rest of the product uses.
CALENDAR_YEAR_LEAGUES = {"Eliteserien", "Allsvenskan", "MLS", "Brasileirao", "Liga Profesional Argentina"}
SEASONS = os.environ.get("SEASONS", "26/27,25/26,24/25,23/24").split(",")   # 26/27 included proactively — season_ids()
                                                                            # silently skips labels Sofascore hasn't
                                                                            # created yet, so this is how a scheduled
                                                                            # run picks up a new season with no code change
TOPN = int(os.environ.get("TOPN", "120"))
LEAGUES = os.environ.get("LEAGUES", "").split(",") if os.environ.get("LEAGUES") else list(UT)


def get(u, tries=3):
    for _ in range(tries):
        try:
            r = cr.get(u, impersonate="chrome", timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.2)
    return {}


def season_ids(ut, want, calendar_year=False):
    """Resolve {label: season_id} for the wanted DOUBLE-YEAR labels (e.g. '25/26'). For calendar_year leagues,
    Sofascore stores seasons as a single year ('2026'); map calendar year Y -> the 'Y-1/Y' bucket it most overlaps
    (2026 -> '25/26') so these leagues land in the same season buckets as European ones for cross-league
    comparison, even though the actual playing calendars don't align exactly. Unmatched labels are just skipped
    (empty result), never mismatched to the wrong season."""
    ss = get(f"{B}/unique-tournament/{ut}/seasons").get("seasons", [])
    if calendar_year:
        by_cy = {s.get("year"): s.get("id") for s in ss}
        out = {}
        for lab in want:
            cy = "20" + lab.split("/")[1]                    # "25/26" -> "2026"
            if cy in by_cy:
                out[lab] = by_cy[cy]
        return out
    by_year = {s.get("year"): s.get("id") for s in ss}
    return {lab: by_year[lab] for lab in want if lab in by_year}


def per90_record(name, pid, league, label, st):
    """Every field here is a REAL Sofascore statistics/overall key (verified against a live response — that API has
    no 'accurateProgressivePasses' field, so the old prog_pass90 silently always fell back to bigChancesCreated;
    fixed below to accurateFinalThirdPasses, the closest real progression signal it actually has). Comprehensive
    by design (spec: "comprehensive to a tee") — Attacking/Creativity/Carrying/Defending/Duels/Discipline, ~30
    per-90 fields plus 3 rate stats that are already per-match averages (rating, pass%, dribble%) and must NOT be
    divided by n90 again."""
    mp = st.get("minutesPlayed") or 0
    if mp < 450:                                              # >= 5 full matches
        return None
    n90 = mp / 90.0
    g = lambda k: (st.get(k, 0) or 0) / n90                    # per-90 helper
    return {
        "name": name, "player_id": pid, "league": league, "season": label, "nineties": round(n90, 1),
        # ---- attacking ----
        "gls90": g("goals"), "sh90": g("totalShots"), "sot90": g("shotsOnTarget"), "xg90": g("expectedGoals"),
        "finishing": (st.get("goals", 0) or 0) / max(st.get("expectedGoals", 0) or 0.1, 0.1),
        "goal_conv_pct": st.get("goalConversionPercentage", 0) or 0,
        "big_ch_missed90": g("bigChancesMissed"), "headed_gls90": g("headedGoals"),
        # ---- creativity / passing ----
        "ast90": g("assists"), "xa90": g("expectedAssists"), "keypass90": g("keyPasses"),
        "big_ch_created90": g("bigChancesCreated"), "prog_pass90": g("accurateFinalThirdPasses"),
        "pass_pct": st.get("accuratePassesPercentage", 0) or 0, "long_ball90": g("accurateLongBalls"),
        "crs90": g("accurateCrosses"), "cross_pct": st.get("accurateCrossesPercentage", 0) or 0,
        # ---- carrying / possession ----
        "dribble90": g("successfulDribbles"), "dribble_pct": st.get("successfulDribblesPercentage", 0) or 0,
        "touches90": g("touches"), "dispossessed90": g("dispossessed"), "was_fouled90": g("wasFouled"),
        # ---- defending ----
        "tkl90": g("tackles"), "tkl_won_pct": st.get("tacklesWonPercentage", 0) or 0, "int90": g("interceptions"),
        "clearances90": g("clearances"), "blocks90": g("blockedShots"), "recoveries90": g("ballRecovery"),
        # ---- duels ----
        "duels_won90": g("totalDuelsWon"), "duel_won_pct": st.get("totalDuelsWonPercentage", 0) or 0,
        "aerial_won90": g("aerialDuelsWon"), "aerial_won_pct": st.get("aerialDuelsWonPercentage", 0) or 0,
        "dribbled_past90": g("dribbledPast"),
        # ---- discipline ----
        "fls90": g("fouls"), "yellow90": g("yellowCards"), "offsides90": g("offsides"),
        # ---- overall ----
        "rating": st.get("rating", 0) or 0,
    }


def enrich(recs):
    """Enrich records in place with a CURRENT snapshot per unique player: market value/age/height/foot (already
    had these) plus club/nationality/photo — all from the SAME /player/{id} call already being made, just more
    fields extracted from a response we were already fetching. Photo/club-logo aren't fields Sofascore returns —
    they're served from a predictable, verified CDN pattern (api.sofascore.com/api/v1/{player,team}/{id}/image)."""
    now = time.time(); ids = sorted({r["player_id"] for r in recs}); meta = {}
    print(f"[ms] enriching {len(ids)} unique players ...", flush=True)
    for i, pid in enumerate(ids):
        p = get(f"{B}/player/{pid}").get("player", {})
        dob = p.get("dateOfBirthTimestamp")
        team = p.get("team") or {}; country = p.get("country") or {}
        meta[pid] = {"market_value": (p.get("proposedMarketValueRaw") or {}).get("value"),
                     "age": round((now - dob) / (365.25 * 86400), 1) if dob else None,
                     "height": p.get("height"), "foot": p.get("preferredFoot"),
                     "club": team.get("name"), "club_id": team.get("id"),
                     "nationality": country.get("name"), "nationality_code": country.get("alpha3"),
                     "nationality_alpha2": country.get("alpha2"),
                     "photo_url": f"{B}/player/{pid}/image",
                     "club_logo_url": f"{B}/team/{team['id']}/image" if team.get("id") else None}
        if (i + 1) % 100 == 0: print(f"[ms] enrich {i+1}/{len(ids)}", flush=True)
        time.sleep(0.12)
    for r in recs:
        r.update(meta.get(r["player_id"], {}))
    return recs


def enrich_only():
    """Fast path: re-enrich an ALREADY-SCRAPED dataset (club/nationality/photo added after the fact) without
    redoing the slow per-league-season stats crawl — same unique players, just re-fetching /player/{id} for the
    new fields. Loads from the bucket (canonical store), enriches, re-uploads."""
    tok = os.environ["HF_TOKEN"]
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Chucks90/football-sofascore-data", "football_player_seasons.json", repo_type="dataset", token=tok)
    out = json.load(open(p))
    recs = out["records"]
    before = sum(1 for r in recs if r.get("market_value"))
    print(f"[ms] enrich-only: {len(recs)} existing records, {len(set(r['player_id'] for r in recs))} unique players, "
          f"{before} already have market_value", flush=True)
    enrich(recs)
    after = sum(1 for r in recs if r.get("market_value"))
    print(f"[ms] enrich-only: {after} records have market_value after enrichment (was {before})", flush=True)
    if after < before:                                             # enrichment must never make coverage WORSE
        print(f"[ms] ABORT: market_value coverage regressed ({before} -> {after}) — looks like Sofascore is "
              f"blocking /player/{{id}} again. Refusing to overwrite good data with degraded data.", flush=True)
        print("FB_MULTISEASON_DONE", flush=True)
        return
    out["generated"] = datetime.date.today().isoformat()
    _upload(out)


def _upload(out):
    json.dump(out, open("/tmp/fb_multiseason.json", "w"))
    tok = os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import HfApi
        HfApi(token=tok).upload_file(path_or_fileobj="/tmp/fb_multiseason.json",
            path_in_repo="football_player_seasons.json", repo_id="Chucks90/football-sofascore-data", repo_type="dataset")
        print("[ms] persisted -> dataset football_player_seasons.json", flush=True)
    except Exception as e:
        print("[ms] dataset persist skip:", str(e)[:60], flush=True)
    try:                                                        # canonical Fulcrum store = the HF bucket
        from huggingface_hub import HfFileSystem
        HfFileSystem(token=tok).put("/tmp/fb_multiseason.json",
                                    "hf://buckets/Chucks90/fulcrum-data/sofascore/football_player_seasons.json")
        print("[ms] persisted -> bucket fulcrum-data/sofascore/football_player_seasons.json", flush=True)
    except Exception as e:
        print("[ms] bucket persist skip:", str(e)[:60], flush=True)
    print("FB_MULTISEASON_DONE", flush=True)


def main():
    if os.environ.get("ENRICH_ONLY") == "1":
        return enrich_only()
    recs = []
    for name in LEAGUES:
        ut = UT.get(name)
        if not ut:
            continue
        sids = season_ids(ut, SEASONS, calendar_year=name in CALENDAR_YEAR_LEAGUES)
        for label, sid in sids.items():
            pool, off, pages = {}, 0, 1
            while off < pages * 100:
                d = get(f"{B}/unique-tournament/{ut}/season/{sid}/statistics?limit=100&offset={off}&order=-goals&group=summary&accumulation=total")
                rs = d.get("results", []); pages = d.get("pages", 0)
                for r in rs:
                    pid = r["player"]["id"]
                    pool[pid] = (r["player"]["name"], (r.get("goals", 0) or 0) + (r.get("assists", 0) or 0))
                off += 100
                if not rs:
                    break
                time.sleep(0.35)
            top = sorted(pool.items(), key=lambda kv: -kv[1][1])[:TOPN]
            for pid, (pname, _) in top:
                st = get(f"{B}/player/{pid}/unique-tournament/{ut}/season/{sid}/statistics/overall").get("statistics", {})
                rec = per90_record(pname, pid, name, label, st)
                if rec:
                    recs.append(rec)
                time.sleep(0.3)
            print(f"[ms] {name} {label} (sid {sid}): cum {len(recs)}", flush=True)

    if len(recs) < 500:                                            # sanity floor: a real run yields ~2800+
        print(f"[ms] ABORT: only {len(recs)} records scraped (expected 2000+) — looks like a blocked/failed run "
              f"(Sofascore challenge, rate limit, etc), not a real empty season. Refusing to overwrite the "
              f"existing dataset with this.", flush=True)
        print("FB_MULTISEASON_DONE", flush=True)
        return

    enrich(recs)
    seasons_present = sorted({r["season"] for r in recs})
    print(f"[ms] {len(recs)} player-seasons across {seasons_present}", flush=True)
    out = {"n": len(recs), "seasons": seasons_present, "leagues": LEAGUES, "topn": TOPN,
           "generated": datetime.date.today().isoformat(), "records": recs}
    _upload(out)


if __name__ == "__main__":
    main()
