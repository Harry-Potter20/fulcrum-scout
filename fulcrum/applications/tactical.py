"""Tactical interpretation — composes Explain (formation/pressing/topology) + Evaluate (danger/space). A grounded
read of a state. status: computed geometry + validated danger."""
import fulcrum
from fulcrum import services as S
def interpret(state):
    ev, ex = S.evaluate(state), S.explain(state)
    return {"application": "tactical-interpretation",
            "danger": ev["danger"].value, "space_creation": ev["space_creation"].value,
            "formation": (ex.get("formation").value if "formation" in ex else None),
            "pressing": (ex.get("pressing").value if "pressing" in ex else None),
            "top_holes": [(round(h["x"], 1), round(h["y"], 1), round(h["score"], 2)) for h in ex["topology"].value[:3]],
            "status": "computed:geometry + validated:danger"}
