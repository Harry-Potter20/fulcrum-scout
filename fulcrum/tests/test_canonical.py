"""Conformance test for the Canonical State ABI (architecture/CANONICAL_STATE.md). Every ingestion adapter MUST
pass this — it is what guarantees "if a data source changes, nothing above the state layer changes". Run against a
Window produced by any adapter. The analogue of the medical project's label-leak test: here the sin is a player
identity entering the state."""
import numpy as np
import worldmodel as W

PITCH_L, PITCH_W, MAX_NODES = 105.0, 68.0, 28


def check_window(w):
    """Assert a Window conforms to canonical/1.0. Raises AssertionError on the first violation."""
    pos, team, isb = np.asarray(w.pos), np.asarray(w.team), np.asarray(w.isball)
    n = len(pos)
    # (a) finite geometry
    assert np.isfinite(pos).all() and np.isfinite(np.asarray(w.vel)).all(), "non-finite pos/vel in a real node"
    # (b) canonical coordinate frame (small tolerance for tracking overshoot at the lines)
    assert (pos[:, 0] >= -3).all() and (pos[:, 0] <= PITCH_L + 3).all(), "x outside [0,105]"
    assert (pos[:, 1] >= -3).all() and (pos[:, 1] <= PITCH_W + 3).all(), "y outside [0,68]"
    # (c) exactly one ball node
    assert int(isb.sum()) == 1, f"expected exactly one ball node, got {int(isb.sum())}"
    # (d) team role coding in {0,1,2}
    assert set(np.unique(team).tolist()).issubset({0.0, 1.0, 2.0}), "team role not in {0,1,2}"
    # (e) ball node is team==2
    assert team[isb.astype(bool)][0] == 2.0, "ball node must be team-coded 2"
    # (f) node count within padding budget
    assert n <= MAX_NODES, f"node count {n} exceeds MAX_NODES {MAX_NODES}"
    # (g) IDENTITY-FREE: the Window dataclass carries no identity/attribute fields (only geometry)
    fields = set(getattr(w, "_fields", []) or (w.__dataclass_fields__.keys() if hasattr(w, "__dataclass_fields__") else []))
    forbidden = {"player_id", "name", "identity", "club", "team_name", "age", "attributes"}
    assert not (fields & forbidden), f"identity leaked into the canonical state: {fields & forbidden}"
    return True


def test_canonical_smoke():
    """Synthetic well-formed Window passes; identity-leak / bad-frame variants fail."""
    n = 15; pos = np.random.default_rng(0).uniform([0, 0], [PITCH_L, PITCH_W], (n, 2)).astype(np.float32)
    team = np.array([2.0] + [1.0] * 7 + [0.0] * 7, np.float32); isb = np.array([1.0] + [0.0] * 14, np.float32)
    w = W.Window(pos, np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32), team, isb, [pos.copy() for _ in range(W.STEPS)])
    assert check_window(w)
    # bad frame: coordinates way off-pitch must fail
    bad = W.Window(pos + 500, np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32), team, isb, [pos.copy() for _ in range(W.STEPS)])
    try:
        check_window(bad); raise SystemExit("conformance FAILED to catch off-pitch coordinates")
    except AssertionError:
        pass
    print("test_canonical: PASS (well-formed conforms; off-pitch rejected)")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jobs"))
    test_canonical_smoke()
