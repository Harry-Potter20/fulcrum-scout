"""fulcrum.registry — the single, machine-readable source of scientific truth for every capability Scout surfaces.

Per app_build.md §36/§60/§63: NO component hard-codes a claim or a status. Every UI element, report line, and
narration reads its tier + evidence from here, so a screen can never silently over-claim, and a retraction changes
one line. This is a STABLE PRIMITIVE (fulcrum/, §59) — product code (scout/, app/) composes over it, never edits it.

Status ladder (§36), strongest to weakest:
  VALIDATED         held-out outcome test with scale/CI — decision-adjacent
  FACE_VALID        mechanistically sound, outcome test pending — describe, don't decide on it
  EXPERIMENTAL      research capability, not decision-grade (e.g. the counterfactual mechanism)
  UNPROVEN          a claim we deliberately do NOT make yet (e.g. signing impact) — gated
  INSUFFICIENT_DATA too little coverage on THIS subject to report (per-player, decided at runtime)

The numbers are sourced (evidence/source), not asserted. Update evidence when a gate report changes; nothing else.
"""
from __future__ import annotations

VALIDATED, FACE_VALID, EXPERIMENTAL, UNPROVEN, INSUFFICIENT_DATA = \
    "validated", "face_valid", "experimental", "unproven", "insufficient_data"

# UI treatment per tier (label + the artifact's badge class + one-line meaning). The app reads this — no per-page copy.
TIERS = {
    VALIDATED:         {"label": "Validated",    "badge": "b-val",  "means": "held-out outcome test · scale + CI"},
    FACE_VALID:        {"label": "Face-valid",   "badge": "b-face", "means": "mechanistically sound · outcome test pending"},
    EXPERIMENTAL:      {"label": "Experimental", "badge": "b-exp",  "means": "research capability · not decision-grade"},
    UNPROVEN:          {"label": "Unproven",     "badge": "b-exp",  "means": "claim deliberately not made yet · gated"},
    INSUFFICIENT_DATA: {"label": "Insufficient", "badge": "b-mut",  "means": "too little coverage on this subject"},
}

# Every capability/claim Scout can show. `metric` is the validated readout; `evidence` says where it comes from.
CAPABILITIES = {
    # ---- attacking geometry (the differentiators) ----
    "danger": {
        "status": VALIDATED, "headline": "Exploitable space / tactical danger",
        "metric": "chance-soon lift 2.2–3.0×", "method": "find_holes topology (not the neural latent)",
        "evidence": "held-out match data, per-modality", "say": "captures exploitable spatial configuration",
        "dont_say": "goal probability / guaranteed chance"},
    "space_creation": {
        "status": VALIDATED, "headline": "Space creation",
        "metric": "chance lift 1.4–1.6×", "method": "per-player off-ball space opened (M.space_creation)",
        "evidence": "held-out match data", "say": "creates exploitable geometry for teammates",
        "dont_say": "assists / xA / progressive passes (a different phenomenon)"},
    "phase_value": {
        "status": VALIDATED, "headline": "Phase value",
        "metric": "value ↔ xG  ρ 0.64", "method": "linear probe on the v3/predictive encoder",
        "evidence": "held-out xG regression (contrastive 0.61, v3 0.64)", "say": "captures value in the pre-shot configuration",
        "dont_say": "predicts goals"},
    "progressive_intent": {
        "status": VALIDATED, "headline": "Progressive intent",
        "metric": "η² 0.70 (latent trait)", "method": "latent-trait ANOVA over players",
        "evidence": "trait-variance study", "say": "movement/positioning indicating intent to advance state",
        "dont_say": "progressive passes (broader than the event)"},
    "off_ball_penetration": {
        "status": VALIDATED, "headline": "Off-ball penetration",
        "metric": "η² 0.38 (latent trait)", "method": "latent-trait ANOVA over players",
        "evidence": "trait-variance study", "say": "attacks exploitable space without needing possession",
        "dont_say": "runs / touches in the box"},
    # ---- pressure & defence (mechanistically sound, outcome test pending) ----
    "press_resistance": {
        "status": FACE_VALID, "headline": "Press resistance",
        "metric": "G1d pressure-gated retention", "method": "state retained under spatial constraint",
        "evidence": "twin G1d gate (mechanism)", "say": "maintains useful state when constrained by pressure",
        "dont_say": "successful dribbles / turnovers"},
    "containment": {
        "status": FACE_VALID, "headline": "Defensive containment",
        "metric": "per-defender danger suppressed", "method": "M.containment positional value",
        "evidence": "face-valid, not outcome-validated", "say": "occupies structurally valuable defensive positions",
        "dont_say": "best defender"},
    "structural_exposure": {
        "status": FACE_VALID, "headline": "Structural exposure",
        "metric": "durable defensive gap (not transient)", "method": "persistence-filtered topology",
        "evidence": "face-valid, outcome validation pending", "say": "durable exploitable gaps in a defensive block",
        "dont_say": "proven outcome predictor"},
    "shape_influence": {
        "status": FACE_VALID, "headline": "Shape influence",
        "metric": "|Δcompactness| + |Δwidth| on removal", "method": "M.shape_influence — remove-and-recompute on "
                  "the player's own team's positional spread (possession-independent, no find_holes call)",
        "evidence": "face-valid mechanism, not outcome-validated — no production proxy exists, Measured-only",
        "say": "how much this player's positioning holds their team's shape together",
        "dont_say": "a stable measure of skill independent of tactical setup — it is influenced by role and system"},
    # ---- the counterfactual: mechanism validated, SIGNING IMPACT is the gated claim (§32/62/63) ----
    "counterfactual_mechanism_attack": {
        "status": EXPERIMENTAL, "headline": "Counterfactual · attack (mechanism)",
        "metric": "corr 0.994 · localization 7.4× · horizon ≤2s",
        "method": "per-player forward-intent perturbation → that runner's own space_creation (twin, defenders react)",
        "evidence": "gate_reports/cf_perplayer.json", "say": "a different simulated short-horizon trajectory",
        "dont_say": "signing impact / a goals figure"},
    "counterfactual_mechanism_defend": {
        "status": EXPERIMENTAL, "headline": "Counterfactual · defend (mechanism)",
        "metric": "corr −0.865 · beats random null · horizon ≤2s",
        "method": "cover-the-hole defender perturbation → team danger reduced (twin)",
        "evidence": "gate_reports/cf_defender_v2.json", "say": "covering the threatened space lowers modelled danger",
        "dont_say": "guaranteed defensive improvement"},
    "counterfactual_signing_impact": {
        "status": UNPROVEN, "headline": "Signing impact",
        "metric": "—", "method": "Mode-A twin replacement over a team's real phases",
        "evidence": "GATED — needs G3 sim-to-real; washes out on 576p tracking",
        "say": "research-grade projection, not a decision", "dont_say": "signing X adds Y% goals"},
}

# System-level facts (shown on the Evidence screen; not per-player capabilities).
SYSTEM = {
    "twin_oos":          {"status": VALIDATED,    "metric": "+25.6%", "of": "attribute-conditioned twin, out-of-sample"},
    "tracking_1080p":    {"status": VALIDATED,    "metric": "0.46 m", "of": "recognition-free tracking, real 1080p broadcast"},
    "team_assign_1080p": {"status": VALIDATED,    "metric": "0.999",  "of": "kit-colour team assignment at 1080p (0.60 at 576p — resolution-gated)"},
    "danger_retention":  {"status": VALIDATED,    "metric": "0.767 ± 0.006", "of": "danger retention, seed-stable"},
    "retrieval":         {"status": VALIDATED,    "metric": "1.0",    "of": "temporal retrieval"},
    "cf_null_ratio":     {"status": EXPERIMENTAL, "metric": "2.16×",  "of": "coherent vs incoherent counterfactual delta (cf_2_5 null test)"},
    "twin_horizon":      {"status": EXPERIMENTAL, "metric": "~2 s",   "of": "validated world-model horizon — do not simulate full possessions"},
    "identity_in_core":  {"status": VALIDATED,    "metric": "NONE",   "of": "identity dependence in the backbone (agnosticism law)"},
}


def get(cap: str) -> dict:
    """Capability record + its resolved UI tier. Raises on unknown key — a typo must never silently become 'validated'."""
    if cap not in CAPABILITIES:
        raise KeyError(f"unknown capability {cap!r}; known: {sorted(CAPABILITIES)}")
    rec = dict(CAPABILITIES[cap]); rec["tier"] = TIERS[rec["status"]]; rec["key"] = cap
    return rec


def status_of(cap: str) -> str:
    return CAPABILITIES[cap]["status"]


def badge(cap: str) -> str:
    """The artifact badge class for a capability's tier (so the UI never maps status→style itself)."""
    return TIERS[CAPABILITIES[cap]["status"]]["badge"]


def is_decision_grade(cap: str) -> bool:
    """Only VALIDATED capabilities may drive a recommendation number; everything else is describe-only (§63)."""
    return CAPABILITIES[cap]["status"] == VALIDATED


def as_dict() -> dict:
    """Full registry as plain data — for JSON export, the LLM narration contract (§58), and the app's data layer."""
    return {"tiers": TIERS, "capabilities": CAPABILITIES, "system": SYSTEM}


if __name__ == "__main__":
    import json
    print(json.dumps(as_dict(), indent=2))
    print("\n# decision-grade capabilities:",
          [k for k in CAPABILITIES if is_decision_grade(k)])
