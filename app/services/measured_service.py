"""app.services.measured_service — exposes the MEASURED-from-geometry capabilities (spec §9) to the UI. These are
computed by the topology stack on real tracked states (anonymous track ids), so their evidence grade is
`measured_geometry` (b-val), not the named-player `estimated`. This is the product's differentiator made concrete:
the real Fulcrum measurement, not a production proxy.
"""
from __future__ import annotations
from app.data import tracked as T
from app.config import settings as S
from fulcrum import registry as R


def sequences() -> list:
    return T.available_sequences()


def measured_players(seq: str) -> dict:
    """Tracked players of a sequence with their Measured capability percentiles + role hints, split by team."""
    data = T.load_measured(seq)
    rows = []
    for p in data["players"]:
        rows.append({
            "tid": p["tid"], "team": p["team"], "role": T.role_hint(p["mean_x"], p["team"]),
            "space_creation": p["space_creation"], "containment": p["containment"],
            "shape_influence": p.get("shape_influence"), "frames": p["frames"],
        })
    return {"seq": seq, "n_frames": data["n_frames"], "players": rows,
            "sc_status": R.status_of("space_creation"), "cont_status": R.status_of("containment"),
            "shape_status": R.status_of("shape_influence")}


def measured_profile(row: dict) -> dict:
    """Adapt a tracked player's Measured values into the same profile shape the capability panel/radar consume, with
    evidence = measured_geometry (the axes we can actually measure from tracking; others stay insufficient)."""
    prof = {}
    mapping = {"space_creation": ("space_creation", "Space creation"),
               "containment": ("containment", "Defensive containment"),
               "shape_influence": ("shape_influence", "Shape influence")}
    for ax, spec in S.CAP_AXES.items():
        val = row.get(ax) if ax in mapping else None
        prof[ax] = {"label": spec["label"], "pct": val,
                    "evidence": "measured_geometry" if val is not None else "insufficient_data",
                    "registry_status": R.status_of(spec["registry_key"]), "registry_key": spec["registry_key"],
                    "drivers": ["topology"] if val is not None else []}
    return prof
