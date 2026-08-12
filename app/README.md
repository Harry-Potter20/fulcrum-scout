# Fulcrum Scout — MVP

Streamlit product surface over the Fulcrum geometric world model. Built to `app_build.md` (the 69-section product
spec). Scouting by **capability**, not production.

## Run

```bash
cd Football_Research
streamlit run app/app.py
```

Needs an HF token (`~/.cache/huggingface/token` or `HF_TOKEN`) to load the player pool
(`Chucks90/football-sofascore-data/football_player_seasons.json`, 2865 player-seasons, 8 leagues, 3 seasons).

## Architecture (spec §48/§49 — pages call services, services read data + registry; no `model()` in UI)

```
app/
  app.py                     shell + pages (Home · Discover · Player · Tactical Fit · Evidence)
  config/settings.py         palette, season pool, the capability model (axis → per-90 proxies → registry key)
  data/loaders.py            records (cached) → percentiles → capability_profile / archetype / anomaly
  services/scout_service.py  Discover, dossier, similarity, market anomalies  (§17/§19/§27/§42/§43)
  services/fit_service.py    Tactical Fit — named need → ranked, decomposable matches  (§20/§31)
  components/ui.py           CSS + badge/scorecard/capability primitives (read tiers from the registry)
  visualization/charts.py    inline-SVG capability radar + market anomaly map (§46, self-contained)
```

The single source of scientific truth is **`fulcrum/registry.py`** (spec §60). Every tier/claim the UI shows resolves
from it, so no screen can silently over-claim, and a retraction changes one line.

## The honesty model (spec §9/§36/§63)

Each capability carries **two independent facts**:
- the **method's** scientific tier — from the registry (`danger`/`space_creation` **validated**, `containment`
  **face-valid**, `signing_impact` **unproven**);
- **this player's** evidence grade — Measured / **Estimated (from production proxies)** / Insufficient.

For the named-player universe we have box-score production, not per-player tracking, so capabilities are **Estimated**
from per-90 proxies. When tracking is attached the same axis switches to **Measured (geometry)**. Missing ≠ 0.

No opaque master score: every headline is **FIT / VALUE / UPSIDE / EVIDENCE**, each explainable. The counterfactual
is present but marked **EXPERIMENTAL · signing-impact unproven** until sim-to-real (G3).
