# Deploying Fulcrum Scout

## Live now (ephemeral, this machine)
A Cloudflare quick-tunnel serves the running local instance behind an access key. It stays up only while this
machine is awake and the tunnel process runs; the URL rotates on every tunnel restart. Good for a quick preview,
not for always-on.

- App: `FULCRUM_APP_PASSWORD=<key> streamlit run app/app.py --server.enableCORS false --server.enableXsrfProtection false`
- Tunnel: `cloudflared tunnel --url http://localhost:8533`

## Always-on, stable URL — Pxxl (recommended)
Repo: **github.com/Harry-Potter20/fulcrum-scout** (public — trimmed to just the app + the `fulcrum` package needed
to run it; no research code, no checkpoints, no private data). Pxxl is a GitHub-connected PaaS with managed
Postgres, auto-detects Python from `requirements.txt` + `pxxl.toml` (no Dockerfile needed) — docs.pxxl.app.

**This deploys as TWO Pxxl projects from the same repo** (Pxxl's documented "multi-service" pattern). This is not
optional complexity — it's forced by how the pieces actually work: Streamlit only runs `app.py`'s script per
browser **websocket** session, so a scheduler's plain HTTP hit (which is what Pxxl Cron Jobs send) would never
reach it. `app/ingest_server.py` is a genuine minimal HTTP server built specifically to be a valid cron target.

### One-time setup (dashboard — the one step no CLI/token can skip)
Project creation and database provisioning happen through the Pxxl dashboard (confirmed: the CLI can deploy an
*existing* project non-interactively with an API key, but creating a new one needs the dashboard first).

1. Sign in at pxxl.app, connect GitHub, **Import Project** → `Harry-Potter20/fulcrum-scout`. This becomes the
   **web** project; it auto-detects `pxxl.toml` at the repo root.
2. **Dashboard → Database → Create Database** → PostgreSQL. Wait for "active", copy `DATABASE_URL`.
3. On the **web** project's env vars, set:
   - `DATABASE_URL` — from step 2
   - `HF_TOKEN` — read access to `Chucks90/football-gsr-data` (the Measured page + Postgres sync source)
   - `FULCRUM_APP_PASSWORD` — the human access key for the app
4. **Import the same repo a second time** as a new project (the **ingest** project). Override its build config
   (project settings, since a single `pxxl.toml` describes one service):
   - Install command: `pip install -r requirements.txt`
   - Start command: `python app/ingest_server.py`
   Set its env vars: `DATABASE_URL` (same as above), `HF_TOKEN` (same), `FULCRUM_SYNC_KEY` — a *different* secret
   from `FULCRUM_APP_PASSWORD`, used to authorize the sync webhook.
5. Deploy both. Confirm the ingest project's health: `GET https://<ingest-url>/healthz` → `{"status":"ok"}`.
6. **Dashboard → Cron Jobs → Create** (or `pxxl cron create`): schedule a daily HTTP GET at
   `https://<ingest-url>/sync?key=<FULCRUM_SYNC_KEY>`. This is the auto-ingest — see below for what "auto" means.
7. Trigger it once manually (visit the URL, or `pxxl cron` run-now) to seed the database before first real use.

### What "auto-ingest when the season starts" actually means here
There is no feed that announces a season's start, so the honest version is: the cron hits `/sync` daily, which
runs `jobs/sync_db.py` — idempotent upserts of the player pool (from the public `football-sofascore-data` HF
dataset) and the tracked Measured caps (from the private `gsr` bucket) into Postgres. `fb_multiseason.py` (the
actual Sofascore scraper feeding that public dataset) resolves season labels dynamically and already includes
`26/27` in its default list — Sofascore blocks datacenter IPs, so that scraper runs from a **residential** machine
on its own schedule (see "Local ingest source" below), not from Pxxl. Once Sofascore populates a new season, the
next scrape picks it up, uploads it, and the next daily `/sync` absorbs it into Postgres — a season "starting" is
absorbed within one ingest cycle, not detected in advance.

### Local ingest source (residential IP required)
`jobs/fb_multiseason.py` scrapes Sofascore and uploads to the public HF dataset the DB syncs from. Sofascore
blocks datacenter IPs, so this must run from a residential connection — schedule it via `launchd`/cron on a home
machine (see the repo's `Football_Research/jobs/` for the script; it already uploads its output to HF on success).

## Cloudflare — Workers isn't a fit, but two things are worth pairing in
**Workers/Pages cannot run this app.** Workers is a V8-isolate runtime (JS/Wasm, or Python via Pyodide) — it has
no long-running Python process, no `pip install torch`/`psycopg2`, no websocket-driven Streamlit session model.
Trying to force this stack onto Workers would silently fail, not just be awkward.

What *is* worth using from Cloudflare:
- **A named Cloudflare Tunnel** (a different product from Workers) bound to a real domain gives a **stable**
  public URL for the ephemeral/local deployment path above, instead of a quick-tunnel's rotating one.
- **Cloudflare Pages** could host a lightweight static landing/teaser page that links to the real (Pxxl-hosted)
  app — reasonable if a marketing front door is wanted later; not a substitute for hosting the app itself.

## Fallback: plain container (Fly/Render/Railway/VPS)
`app/Dockerfile` + `app/requirements.txt` still work if Pxxl isn't the right fit later — same two-service split
applies (`CMD` for the web image, override `CMD python app/ingest_server.py` for a second ingest image/service).

```bash
docker build -f app/Dockerfile -t fulcrum-scout .
docker run -d -p 80:8501 -e HF_TOKEN=hf_xxx -e DATABASE_URL=postgres://... -e FULCRUM_APP_PASSWORD=yourkey \
  --restart unless-stopped fulcrum-scout
```

## Note on the gates
Both `FULCRUM_APP_PASSWORD` and `FULCRUM_SYNC_KEY` are app-level checks, not hardened HTTP auth — they hide things
behind a key but aren't a full auth system. For real multi-user auth, front the web project with the host's access
control (e.g. Cloudflare Access) in addition.
