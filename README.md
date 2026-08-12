# Fulcrum Scout

Tactical intelligence over Fulcrum, a player-agnostic geometric football world model. Built to
[`app_build.md`](https://github.com/Harry-Potter20/fulcrum-scout) (the product spec): capability, not production —
scouting by what a player's geometry and behaviour demonstrate, with every claim carrying a machine-readable
scientific tier (`fulcrum/registry.py`) so the UI can never silently over-claim.

**Pages:** Home · Discover (capability market) · Player (dossier, evidence drawer, live signing-impact simulation)
· Tactical Fit · Measured (real geometry-derived capabilities from tracked broadcast) · Evidence.

**Two services, one repo** (see [`app/DEPLOY.md`](app/DEPLOY.md)):
- `app/app.py` — the Streamlit web app.
- `app/ingest_server.py` — a minimal stdlib HTTP webhook that runs the Postgres auto-ingest (`jobs/sync_db.py`) on
  a schedule. Separate because Streamlit only executes per browser websocket session; a cron's plain HTTP hit needs
  a real target.

**Data:** public player-season pool (Sofascore via `Chucks90/football-sofascore-data`) + private tracked-geometry
Measured caps (`Chucks90/football-gsr-data`), synced into the app's own Postgres database.

**The counterfactual is live**: `app/services/counterfactual_service.py` rolls real tracked broadcast phases
through the validated twin (attack mechanism corr 0.994, defend corr −0.865) on request. The *mechanism* is
validated; the signing-impact *number* stays labelled research preview until sim-to-real (G3) — see the Evidence
page and `fulcrum/registry.py`.

**Deploy:** Pxxl (managed Postgres + cron) — full walkthrough in [`app/DEPLOY.md`](app/DEPLOY.md). Also runs as a
plain Docker container (`app/Dockerfile`) on Fly/Render/any VPS.
