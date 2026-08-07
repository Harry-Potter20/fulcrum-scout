"""fulcrum.products.broadcast — the BROADCAST product. Evaluate → Explain → Render/Narrate.

Turns a match state into a grounded tactical read: danger + space (Evaluate), formation + pressing + exposed
topology (Explain), and a generated tactical NARRATIVE composed from that read (deterministic, grounded in the
numbers — not invented). The Present/Render layer (`applications.renderer`) turns it into an annotated clip.

This is the platform proof: a DIFFERENT product from Scout, over the SAME six services and the SAME canonical state —
no new model, no fork. Scout reads players; Broadcast reads the moment; both are compositions of the engine.
"""
from fulcrum.applications import tactical

PITCH_L, PITCH_W = 105.0, 68.0


def _zone(x, y):
    zx = "in the box" if x > 92 else ("the final third" if x > 75 else ("midfield" if x > 52 else "deep areas"))
    zy = "the left channel" if y < PITCH_W * 0.33 else ("the right channel" if y > PITCH_W * 0.67 else "a central pocket")
    return f"{zx}, {zy}"


def _narrate(read):
    press = (read.get("pressing") or {}).get("press_type", "a set block")
    form = read.get("formation") or "an unsettled shape"
    d = read.get("danger", 0) or 0
    lvl = "high" if d >= 0.6 else ("building" if d >= 0.25 else "contained")
    s = ["%s in %s." % (form, press), "Threat is %s (%.2f)." % (lvl, d)]
    holes = read.get("top_holes") or []
    if holes:
        x, y, sc = holes[0]
        s.append("The defence is most exposed %s — a %.2f-value pocket." % (_zone(x, y), sc))
    if (read.get("space_creation", 0) or 0) > 0.2:
        s.append("Off-ball movement is prising open secondary lanes.")
    return " ".join(s)


def broadcast(state):
    """State -> a broadcast tactical report. `render` points to the Present layer that draws the annotated clip."""
    read = tactical.interpret(state)
    return {"product": "broadcast",
            "danger": read.get("danger"), "space_creation": read.get("space_creation"),
            "formation": read.get("formation"), "pressing": read.get("pressing"),
            "key_spaces": read.get("top_holes"),
            "narrative": _narrate(read),
            "contract": {"danger": "validated:2.2-3.0x-chance", "space_creation": "validated:1.4-1.6x-offball",
                         "formation/pressing/topology": "computed geometry",
                         "narrative": "descriptor — composed from the read, grounded in the numbers"},
            "render": "applications.renderer.render(frames, fps, fid) -> annotated clip (the Present concern)"}
