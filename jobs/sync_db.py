"""sync_db — the AUTO-INGEST job. Idempotent: pulls the latest player-season pool (public HF dataset, kept fresh by
jobs/fb_multiseason.py running on a schedule against Sofascore) and the latest tracked Measured caps (private HF
bucket, produced by app/data/tracked.py's precompute) and upserts both into Fulcrum Scout's own Postgres db.

This is the honest version of "auto-ingest when the season starts": there is no feed that announces a season's
start, so instead this runs on a fixed cadence (daily, via the host's scheduler — Pxxl Cron Jobs locally, or any
cron) and no-ops cheaply when nothing changed. When Sofascore populates a new season id (e.g. 26/27), it starts
appearing in fb_multiseason.py's output automatically (season labels are resolved dynamically, not hardcoded) and
this job picks it up on its next run — so "the season starting" is absorbed within one ingest cycle, not detected
in advance.

Safe to run anywhere DATABASE_URL + HF_TOKEN are set. Every run is recorded to ingest_log (see app/db.py) so the
ingest history is inspectable from the Evidence page, not a black box.

    python jobs/sync_db.py                # sync everything
    python jobs/sync_db.py --players-only  # skip measured caps
"""
from __future__ import annotations
import os, sys, json, argparse, traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import db
from app.config import settings as S


def sync_players() -> int:
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    p = hf_hub_download(S.RECORDS_REPO, S.RECORDS_FILE, repo_type="dataset", token=tok)
    data = json.load(open(p))
    by_season = {}
    for r in data["records"]:
        by_season.setdefault(r.get("season"), []).append(r)
    total = 0
    for season, recs in by_season.items():
        n = db.upsert_players(recs, season)
        total += n
        print(f"[sync] players {season}: {n} upserted", flush=True)
    return total


def sync_measured() -> int:
    from app.data import tracked as T
    total = 0
    for seq in T.available_sequences_from_bucket():           # bucket-direct — see its docstring for why
        data = T.load_measured_from_bucket(seq)
        n = db.upsert_measured(seq, data["players"])
        total += n
        print(f"[sync] measured {seq}: {n} upserted", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players-only", action="store_true")
    ap.add_argument("--measured-only", action="store_true")
    a = ap.parse_args()
    run(players_only=a.players_only, measured_only=a.measured_only)


def run(players_only: bool = False, measured_only: bool = False) -> int:
    """The actual sync — importable and callable directly (e.g. from app.py's HTTP-triggered ingest), independent
    of CLI argument parsing. Returns the number of records upserted; raises on failure after logging it."""
    if not db.available():
        print("[sync] DATABASE_URL not set — nothing to do", flush=True)
        return 0
    db.init_schema()

    n = 0
    try:
        if not measured_only:
            n += sync_players()
        if not players_only:
            n += sync_measured()
        db.log_ingest("sync_db", n, "ok")
        print(f"[sync] SYNC_DB_DONE total_upserted={n}", flush=True)
        return n
    except Exception as e:
        db.log_ingest("sync_db", n, "error", traceback.format_exc())
        print(f"[sync] SYNC_DB_FAILED: {e}", flush=True)
        raise


if __name__ == "__main__":
    main()
