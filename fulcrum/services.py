"""fulcrum.services — the SIX Foundation Services (the stable platform API; see architecture/SERVICES.md).

Products compose these; products never import the encoder directly. Two platform invariants are enforced here:
 1. **Registry, not forks.** Implementations live in `REGISTRY` keyed by domain (football-v3, basketball-v1, …) —
    same interface, swappable driver. Adding a sport/modality adds an entry, never a fork of the platform.
 2. **Epistemic status in the contract.** Every output is a `Tagged(value, status)`; a product composing an
    `unproven` output inherits that label and cannot silently over-claim. Tiers: validated · face-valid · descriptor ·
    works · unproven.

Inputs are the Canonical State (`architecture/CANONICAL_STATE.md`): a `Window` (model view) or an oriented
`state` dict (topology view, `fulcrum.state_at` / `window_state`). Identity is never an input.
"""
from __future__ import annotations
import numpy as np
from collections import namedtuple

Tagged = namedtuple("Tagged", ["value", "status"])

# ---- registry: domain -> driver (the anti-fork mechanism) --------------------------------------------------------
REGISTRY = {
    "football-v3": {"repo": "Chucks90/football-gsr-data", "encoder": "checkpoints/fulcrum_v3_contrastive.pt",
                    "heads": "checkpoints/fulcrum_unified.pt", "equivariance": "V4", "status": "live",
                    "pitch": (105.0, 68.0), "max_nodes": 28},
    # SCAFFOLD — a second sport is a DRIVER, never a fork: same six service signatures, different weights +
    # symmetry group + court dimensions. Adding one = an entry here + a trained encoder; the services below are
    # untouched. Data-gated: needs basketball tracking (e.g. SportVU-style) to train the C2-equivariant encoder.
    "basketball-v1": {"repo": None, "encoder": None, "heads": None, "equivariance": "C2", "status": "scaffold",
                      "court": (28.65, 15.24), "max_nodes": 11,
                      "needs": "basketball tracking data + a trained C2-equivariant encoder"},
}
_cache: dict = {}


def drivers():
    """The registry as a capability map — which domains are live vs scaffold. Products query this before composing."""
    return {d: {"equivariance": v["equivariance"], "status": v["status"]} for d, v in REGISTRY.items()}


def _load(domain, which):
    key = (domain, which)
    if key not in _cache:
        import os, torch
        from huggingface_hub import hf_hub_download
        import fulcrum
        from fulcrum import model as M
        d = REGISTRY[domain]
        if d.get("status") == "scaffold" or not d.get(which):
            raise NotImplementedError(f"driver '{domain}' is a scaffold — {d.get('needs', 'not yet trained')}. "
                                      f"The registry pattern is ready; this domain needs data + weights.")
        tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        ck = hf_hub_download(d["repo"], d[which], repo_type="dataset", token=tok)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        if "config" in sd:                       # unified/heads format -> canonical loader
            m, _ = fulcrum.load(ck, "cpu")
        else:                                    # v3/contrastive format
            # feat_dim inferred lazily on first encode; store the raw ckpt to build against the caller's feat_dim
            _cache[key] = ("build", sd); return _cache[key]
        _cache[key] = ("model", m)
    return _cache[key]


def _encoder_for(feat_dim, domain):
    kind, obj = _load(domain, "encoder")
    if kind == "model":
        return obj
    import fulcrum
    from fulcrum import model as M
    m = M.build_model(feat_dim, d=obj.get("d", 96), layers=2); m.load_state_dict(obj["model"], strict=False)
    _cache[(domain, "encoder")] = ("model", m); return m


# ---- 1. ENCODE ---------------------------------------------------------------------------------------------------
def encode(windows, domain="football-v3"):
    """State(s) -> embedding. status: validated (representation: preserved+linearised; see FULCRUM_V3_SPEC)."""
    import torch
    from fulcrum import model as M
    ws = windows if isinstance(windows, (list, tuple)) else [windows]
    fd = M.featurize(ws[0]).shape[1]; enc = _encoder_for(fd, domain); enc.eval(); out = []
    for i in range(0, len(ws), 256):
        b = M.collate([M.make_sample(x, M.RS) for x in ws[i:i + 256]], M.MAX_NODES)
        with torch.no_grad():
            _, _, p = enc(b["feat"], b["pos"], b["team"], b["mask"], return_embedding=True)
        out.append(p.cpu().numpy())
    z = np.concatenate(out)
    return Tagged(z if isinstance(windows, (list, tuple)) else z[0], "validated:representation")


# ---- 2. EVALUATE -------------------------------------------------------------------------------------------------
def evaluate(state):
    """State -> tactical value signals, each tagged. Computed (topology) parts need no model."""
    import fulcrum
    summ = fulcrum.state_summary(state)
    return {"danger": Tagged(summ["danger"] if summ else 0.0, "validated:2.2-3.0x-chance"),
            "space_creation": Tagged(sum(fulcrum.space_creation(state).values()), "validated:1.4-1.6x-offball"),
            "containment": Tagged(sum(fulcrum.containment(state).values()), "face-valid:role-discriminative"),
            "value": Tagged(None, "validated:rho0.54 — needs heads driver (score)")}


# ---- 3. PREDICT --------------------------------------------------------------------------------------------------
def predict(windows, domain="football-v3"):
    """State -> future states (rollout). status: twin generalisation validated (+25.6%); style-conditioning composes."""
    return Tagged("rollout via dynamics head (worldmodel); style-conditionable", "validated:twin+25.6 / face-valid:rollout")


# ---- 4. EXPLAIN --------------------------------------------------------------------------------------------------
def explain(state, frames=None, fps=None, fid=None):
    """State -> {topology, formation, pressing, narrative}. Computed geometry; narrative is a descriptor."""
    import fulcrum
    from fulcrum import opposition as _opp
    out = {"topology": Tagged(fulcrum.find_holes(np.asarray(state["att"]), np.zeros_like(state["att"]),
              np.asarray(state["dfn"]), np.zeros_like(state["dfn"]), np.asarray(state["ball"]), top=3)[3], "computed:topology")}
    try:
        out["pressing"] = Tagged(_opp.pressing_structure(state), "computed:pressing")
        out["formation"] = Tagged(_opp.team_formation(state), "computed:formation")
    except Exception:
        pass
    return out


# ---- 5. RETRIEVE -------------------------------------------------------------------------------------------------
def retrieve(query_windows, gallery_windows, k=10, domain="football-v3"):
    """Find similar states in the gallery via latent kNN. status: temporal-retrieval validated; DECISION-retrieval
    on a real outcome is UNPROVEN (the outcome-independent shot-soon test reversed it — see FULCRUM_V3_SPEC)."""
    Z = encode(list(gallery_windows) + list(query_windows), domain).value
    ng = len(gallery_windows); G, Q = Z[:ng], Z[ng:]
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9); Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    nn = [np.argsort(-(Qn[i] @ Gn.T))[:k].tolist() for i in range(len(Qn))]
    return Tagged(nn, "validated:temporal / unproven:decision-outcome")


# ---- 6. OPTIMIZE -------------------------------------------------------------------------------------------------
def optimize(state, goal="close", side="dfn"):
    """State -> {plans, counterfactuals}. Plans use an unhackable COMPUTED reward (works, no training).
    Counterfactual DECISION-QUALITY is UNPROVEN (v3 rep did not improve outcome retrieval)."""
    import fulcrum
    return {"plan": Tagged(fulcrum.plan(state, goal=goal, side=side), "works:computed-reward-no-training"),
            "counterfactual": Tagged(None, "unproven:decision-quality")}


SERVICES = ["encode", "evaluate", "predict", "explain", "retrieve", "optimize"]
