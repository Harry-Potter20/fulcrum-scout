"""Renderer (the PRESENT layer) — turns state + service outputs into the shareable viz. status: presentation.
Consumes Evaluate/Explain; feeds the Broadcast product. Not a state-service (analysis->pixels, not state->analysis)."""
import fulcrum
def render(frames, fps, fid, out_path, hero=True, **kw):
    fn = fulcrum.render_hero if hero else fulcrum.render_frame
    return {"application": "renderer", "holes": fn(frames, fps, fid, out_path, **kw), "out": out_path,
            "status": "presentation"}
