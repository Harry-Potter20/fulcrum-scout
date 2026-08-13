"""app.db — Fulcrum Scout's own database (Postgres). Everywhere else in this codebase, "the bucket" (HF datasets)
is the canonical Fulcrum store for research artifacts; this module is the PRODUCT's own store — the thing a deployed
Scout instance actually queries, refreshed on a schedule by jobs/sync_db.py (§ auto-ingest).

Connects via DATABASE_URL (Postgres — Pxxl's managed engine injects this at runtime, no auto-injection so it must be
set explicitly in project env vars; a local dev Postgres works identically). If DATABASE_URL is unset, `available()`
returns False and the data layer falls back to reading HF directly (today's behaviour) — the app never hard-fails
just because the db hasn't been provisioned yet.

Schema is intentionally small and upsert-only: players (season pool cache), measured_caps (per-track geometry
capabilities from tracked sequences), ingest_log (an auditable history of every sync run, so "auto-ingest" is
inspectable, not a black box).
"""
from __future__ import annotations
import os, json, functools
from contextlib import contextmanager

DDL = """
CREATE TABLE IF NOT EXISTS players (
    player_id   BIGINT NOT NULL,
    season      TEXT NOT NULL,
    name        TEXT NOT NULL,
    league      TEXT,
    record      JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season)
);
CREATE INDEX IF NOT EXISTS idx_players_season ON players (season);
CREATE INDEX IF NOT EXISTS idx_players_name   ON players (name);

CREATE TABLE IF NOT EXISTS measured_caps (
    seq         TEXT NOT NULL,
    tid         INTEGER NOT NULL,
    team        INTEGER,
    space_creation DOUBLE PRECISION,
    containment    DOUBLE PRECISION,
    shape_influence DOUBLE PRECISION,
    mean_x         DOUBLE PRECISION,
    frames         INTEGER,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (seq, tid)
);
ALTER TABLE measured_caps ADD COLUMN IF NOT EXISTS shape_influence DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS ingest_log (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    n_upserted  INTEGER,
    status      TEXT,
    detail      TEXT
);
"""


def database_url() -> str | None:
    return os.environ.get("DATABASE_URL")


def available() -> bool:
    return bool(database_url())


@functools.lru_cache(maxsize=1)
def _pool():
    import psycopg2.pool
    return psycopg2.pool.ThreadedConnectionPool(1, 5, dsn=database_url())


@contextmanager
def connect():
    """A pooled connection, committed on success / rolled back on error. Raises if DATABASE_URL is unset — callers
    should check `available()` first (the data layer does, to decide HF-fallback vs db-read)."""
    if not available():
        raise RuntimeError("DATABASE_URL not set — no database configured")
    pool = _pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback(); raise
    finally:
        pool.putconn(conn)


def init_schema():
    with connect() as conn, conn.cursor() as cur:
        cur.execute(DDL)


# ---------------- writers (used by jobs/sync_db.py) ----------------
def upsert_players(records: list, season: str) -> int:
    """Upsert a season's player records. `record` stores the FULL original dict (so the app never loses a field the
    ingest source adds later) alongside indexed player_id/season/name/league for fast lookups."""
    if not records:
        return 0
    with connect() as conn, conn.cursor() as cur:
        for r in records:
            cur.execute(
                """INSERT INTO players (player_id, season, name, league, record, updated_at)
                   VALUES (%s, %s, %s, %s, %s, now())
                   ON CONFLICT (player_id, season) DO UPDATE
                   SET name = EXCLUDED.name, league = EXCLUDED.league, record = EXCLUDED.record, updated_at = now()""",
                (r["player_id"], season, r["name"], r.get("league"), json.dumps(r)))
    return len(records)


def upsert_measured(seq: str, players: list) -> int:
    if not players:
        return 0
    with connect() as conn, conn.cursor() as cur:
        for p in players:
            cur.execute(
                """INSERT INTO measured_caps (seq, tid, team, space_creation, containment, shape_influence, mean_x, frames, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                   ON CONFLICT (seq, tid) DO UPDATE
                   SET team=EXCLUDED.team, space_creation=EXCLUDED.space_creation, containment=EXCLUDED.containment,
                       shape_influence=EXCLUDED.shape_influence, mean_x=EXCLUDED.mean_x, frames=EXCLUDED.frames, updated_at=now()""",
                (seq, p["tid"], p.get("team"), p.get("space_creation"), p.get("containment"),
                 p.get("shape_influence"), p.get("mean_x"), p.get("frames")))
    return len(players)


def log_ingest(source: str, n_upserted: int, status: str, detail: str = ""):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO ingest_log (source, finished_at, n_upserted, status, detail) VALUES (%s, now(), %s, %s, %s)",
                    (source, n_upserted, status, detail[:2000]))


def last_ingest() -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT source, started_at, finished_at, n_upserted, status, detail FROM ingest_log "
                    "ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        return {"source": row[0], "started_at": str(row[1]), "finished_at": str(row[2]),
                "n_upserted": row[3], "status": row[4], "detail": row[5]}


# ---------------- readers (used by app/data/loaders.py + tracked.py) ----------------
def read_players(season: str) -> list:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT record FROM players WHERE season = %s", (season,))
        return [row[0] for row in cur.fetchall()]


def read_measured(seq: str) -> list:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT tid, team, space_creation, containment, shape_influence, mean_x, frames FROM measured_caps "
                    "WHERE seq = %s ORDER BY tid", (seq,))
        return [{"tid": r[0], "team": r[1], "space_creation": r[2], "containment": r[3], "shape_influence": r[4],
                "mean_x": r[5], "frames": r[6]} for r in cur.fetchall()]


def measured_sequences() -> list:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT seq FROM measured_caps ORDER BY seq")
        return [r[0] for r in cur.fetchall()]
