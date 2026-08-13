"""app.ingest_server — the auto-ingest WEBHOOK. Deployed as a second, separate Pxxl service from the same repo
(Pxxl's documented "multi-service" pattern: one project, different start commands for web vs. background work —
see app/DEPLOY.md). This exists because Streamlit only runs app.py per browser WEBSOCKET session; a scheduler's
plain HTTP request (Pxxl Cron Jobs fire "scheduled HTTP requests", not shell commands) never reaches a session-bound
script. This is a genuine stdlib HTTP server — no framework dependency, minimal attack surface, easy to reason about.

    GET /sync?key=<FULCRUM_SYNC_KEY>    -> runs jobs/sync_db.run() (HF -> Postgres), 200 {"status":"ok","upserted":N}
    GET /scrape?key=<FULCRUM_SYNC_KEY>  -> runs jobs/fb_multiseason.main() (Sofascore -> HF dataset), 200 {"status":"ok"}
    GET /healthz                        -> 200 "ok" (no auth; for the platform's own liveness checks)

/scrape is the upstream half of "auto-ingest when the season starts": it refreshes the public
football_player_seasons.json dataset that /sync then pulls into Postgres. Run /scrape on a slower cron (e.g. daily)
than /sync isn't required — /sync just re-reads whatever's currently in the dataset, so scrape-then-sync same-day
is fine, or wire them as two separate Pxxl cron schedules hitting this one service.

Auth is the key itself (FULCRUM_SYNC_KEY, distinct from the human-facing FULCRUM_APP_PASSWORD). Unset -> every
protected request is rejected (503), so this can be deployed with the endpoints present but inert until configured.

    python app/ingest_server.py            # serves on $PORT (default 8080)
"""
from __future__ import annotations
import os, sys, json, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jobs")))

SYNC_KEY = os.environ.get("FULCRUM_SYNC_KEY", "")

# /scrape runs fb_multiseason.main() — several minutes of rate-limited sequential API calls (8 leagues x up to
# 4 seasons x up to 120 players). Run it in a background thread so a slow platform-side HTTP timeout can't kill
# it mid-scrape; the busy-flag stops a second cron tick from starting an overlapping run.
_scrape_lock = threading.Lock()
_scrape_state = {"running": False, "last_status": None, "last_error": None}


def _run_scrape():
    try:
        import fb_multiseason
        fb_multiseason.main()
        _scrape_state["last_status"] = "ok"; _scrape_state["last_error"] = None
    except Exception:
        _scrape_state["last_status"] = "error"; _scrape_state["last_error"] = traceback.format_exc()[-2000:]
        print("[ingest_server] scrape failed:\n" + _scrape_state["last_error"], flush=True)
    finally:
        _scrape_state["running"] = False


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self, qs):
        if not SYNC_KEY:
            self._json(503, {"error": "FULCRUM_SYNC_KEY not configured"}); return False
        if qs.get("key", [""])[0] != SYNC_KEY:
            self._json(401, {"error": "unauthorized"}); return False
        return True

    def do_GET(self):
        path = urlparse(self.path)
        qs = parse_qs(path.query)

        if path.path == "/healthz":
            return self._json(200, {"status": "ok"})

        if path.path == "/sync":
            if not self._authed(qs):
                return
            try:
                import sync_db
                n = sync_db.run()
                return self._json(200, {"status": "ok", "upserted": n})
            except Exception as e:
                return self._json(500, {"status": "error", "detail": str(e)})

        if path.path == "/scrape":
            if not self._authed(qs):
                return
            with _scrape_lock:
                if _scrape_state["running"]:
                    return self._json(409, {"status": "already_running"})
                _scrape_state["running"] = True
                threading.Thread(target=_run_scrape, daemon=True).start()
            return self._json(202, {"status": "started"})

        if path.path == "/scrape/status":
            return self._json(200, {k: v for k, v in _scrape_state.items()})

        return self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        print(f"[ingest_server] {self.address_string()} {fmt % args}", flush=True)


def main():
    port = int(os.environ.get("PORT", "8080"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[ingest_server] listening on 0.0.0.0:{port} (sync key {'set' if SYNC_KEY else 'NOT SET'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
