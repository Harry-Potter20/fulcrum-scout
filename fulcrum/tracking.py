"""fulcrum.tracking — a shared player tracker for the ingestion pipeline, decoupled from Ultralytics' built-in
`model.track(tracker="bytetrack.yaml")` call that was previously duplicated identically across five job scripts
(footballia_process, lean_ingest, youtube_flip, end2end_dropin, learned_teams) with no shared implementation.

Default tracker is BoT-SORT (via the `trackers` package, github.com/roboflow/trackers) with Camera Motion
Compensation (CMC) enabled. Broadcast football video pans/zooms constantly following play; plain ByteTrack's Kalman
filter assumes a static camera and predicts each track's next position from its own past pixel-space velocity, so a
camera pan looks like every player suddenly moving in the same direction at once — this can break IoU association
and cause ID switches exactly during the moments (camera following a break/counter) that matter most for danger
detection. BoT-SORT estimates the frame-to-frame camera motion (default method 'sparseOptFlow') and compensates
predicted track boxes for it before association. `kind="bytetrack"` reproduces the prior behaviour for comparison.

Detection and tracking are run as two separate steps here (`ymodel(img, ...)` then `tracker.update(det, frame=img)`)
because BoT-SORT's CMC needs the raw frame pixels, which Ultralytics' bundled `.track()` doesn't expose per-call.
"""
from __future__ import annotations
import numpy as np

PERSON, BALL = 0, 32


def make_tracker(kind: str = "botsort", frame_rate: float = 10.0):
    """kind: 'botsort' (default, CMC-enabled), 'bytetrack' (prior behaviour, no CMC), or 'ocsort' (occlusion-robust,
    no appearance features, also no CMC — a middle ground worth benchmarking alongside the other two)."""
    import trackers as trk
    if kind == "botsort":
        return trk.BoTSORTTracker(frame_rate=frame_rate)
    if kind == "bytetrack":
        return trk.ByteTrackTracker(frame_rate=frame_rate)
    if kind == "ocsort":
        return trk.OCSORTTracker(frame_rate=frame_rate)
    raise ValueError(f"unknown tracker kind: {kind!r} (want botsort/bytetrack/ocsort)")


def track_persons(ymodel, tracker, img: np.ndarray, *, pconf: float, bconf: float,
                  imgsz: int) -> tuple[list, tuple | None]:
    """Runs plain YOLO detection (PERSON+BALL) then hands PERSON boxes to `tracker`. Returns
    (players: list[(tid:int, foot_xy:(float,float), ltwh:(float,float,float,float))], full_ball: (conf, xyxy)|None)
    — the exact shape the existing per-job loops already build from `model.track()`'s output, so call sites swap
    in with a minimal diff. Ball is deliberately NOT tracked (mirrors the prior pipeline's own finding: tracking
    the ball lost recall vs a dedicated per-frame detect pass — see jobs/footballia_process.py's GRID/BALL_MAXGAP
    docstring — this module only changes PERSON tracking).

    Two thresholds, matching the established pipeline pattern exactly (not a single `conf`): YOLO itself runs at
    min(pconf, bconf) so nothing is dropped before classification, then PERSON boxes are filtered to >=pconf and
    BALL boxes to >=bconf separately — ball detection is deliberately more permissive (faint/small target)."""
    import supervision as sv
    r = ymodel(img, classes=[PERSON, BALL], conf=min(pconf, bconf), imgsz=imgsz, verbose=False)[0]
    det = sv.Detections.from_ultralytics(r)

    full_ball = None
    if len(det) and BALL in det.class_id:
        ball_det = det[det.class_id == BALL]
        ball_det = ball_det[ball_det.confidence >= bconf]
        if len(ball_det):
            bi = int(np.argmax(ball_det.confidence))
            full_ball = (float(ball_det.confidence[bi]), tuple(float(v) for v in ball_det.xyxy[bi]))

    person_det = det[det.class_id == PERSON] if len(det) else det
    person_det = person_det[person_det.confidence >= pconf] if len(person_det) else person_det
    tracked = tracker.update(person_det, frame=img)

    players = []
    if tracked.tracker_id is not None:
        for j in range(len(tracked)):
            tid = tracked.tracker_id[j]
            if tid is None or tid < 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in tracked.xyxy[j]]
            players.append((int(tid), ((x1 + x2) / 2, y2), (x1, y1, x2 - x1, y2 - y1)))
    return players, full_ball
