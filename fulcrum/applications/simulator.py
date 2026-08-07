"""Simulator — composes Optimize (plan/plan_multi/plan_dynamic) + Predict. Plays out tactical reorganisations
against an UNHACKABLE computed reward. status: works (computed reward, no training); aggressive counterfactuals unproven."""
import fulcrum
def simulate(state, goal="exploit", side="att", depth=3, beam=4):
    return {"application": "simulator",
            "single_edit": fulcrum.plan(state, goal=goal, side=side),
            "coordinated": fulcrum.plan_multi(state, goal=goal, side=side, depth=depth, beam=beam),
            "status": "works:computed-reward-no-training / unproven:aggressive-counterfactual"}
