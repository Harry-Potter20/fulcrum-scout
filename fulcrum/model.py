"""fulcrum.model — Fulcrum's learned WORLD MODEL, exposed as the package's stable inference interface.

The world model: a shared relational spatiotemporal encoder over the ball+players graph, with task heads
(dynamics / state value / structured concepts / retrieval), trained under the covtoken coverage-constrained
objective, with Klein-4 pitch equivariance and relational attention. This module is the SUPPORTED surface for
loading and running it — depend on `fulcrum.model`, not on where the implementation file sits.

Vendored as an interface: the implementation is currently sourced from the training module `worldmodel`
(Football_Research/jobs/worldmodel.py); the full physical relocation into this file is a tracked follow-up and
is behaviour-identical. This module also removes the need for callers to manage sys.path.
"""
from __future__ import annotations
import os as _os
import sys as _sys

# Make the implementation importable without callers touching sys.path.
_JOBS = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", "jobs"))
if _JOBS not in _sys.path:
    _sys.path.insert(0, _JOBS)

from worldmodel import (  # noqa: E402  (path is set up above)
    Window, build_model, load_checkpoint, featurize, featurize_torch, collate, make_sample, symmetry_augment,
    PITCH_L, PITCH_W, VEL_WINDOW_S, HORIZON_S, STEPS, MAX_NODES, RS, EVENT_VOCAB, EVENT_ID, N_EVENTS,
)

__all__ = [
    "Window", "build_model", "load_checkpoint", "featurize", "featurize_torch", "collate", "make_sample",
    "symmetry_augment", "PITCH_L", "PITCH_W", "VEL_WINDOW_S", "HORIZON_S", "STEPS", "MAX_NODES", "RS",
    "EVENT_VOCAB", "EVENT_ID", "N_EVENTS",
]
