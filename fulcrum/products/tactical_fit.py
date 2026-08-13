"""fulcrum.products.tactical_fit — the SIMULATE step of Scout: Counterfactual Tactical Fit.

Turns a player's capability into a tactical-value delta on a real phase, by putting that capability into the world
model and playing it forward. The mechanism is the one validated in architecture/COUNTERFACTUAL_RL.md (G1 gates +
G2 monotonicity corr 0.925 + G2.5 no-op/seed-stable): condition an attacker's motion on capability, roll the twin so
DEFENDERS REACT (shared attention), and read the reward from the INDEPENDENT find_holes topology.

Faithful + training-free per call: kinematic capabilities (pace, forward-intent, width) enter as an INPUT velocity
perturbation (the twin then predicts everyone's reaction); the relational press_resistance enters as the G1d
pressure-gated retention on displacement. Output is a Tactical Fit delta with HONEST tags — "changes the model's
predicted danger within the ~2s validated horizon", NEVER "signing impact" (see the claim ladder in the spec).

    engine = TacticalFit()                       # loads the frozen base twin
    engine.simulate(window, {"forward_intent": 1.5, "pace": 1.0})   -> {delta_danger, per_axis, validity, ...}
    engine.fit_report(deficiency_axis, capability, anchors)          -> the product card (mean delta + CI over phases)
"""
import numpy as np

DT = 0.5
PSCALE = 6.0
SPEED_UNIT = 1.2   # m/s of input velocity per unit of a kinematic capability
PITCH_L, PITCH_W = 105.0, 68.0
AXES = ("forward_intent", "pace", "width", "press_resistance")


class TacticalFit:
    def __init__(self, checkpoint="checkpoints/fulcrum_unified.pt", repo="Chucks90/football-gsr-data", device=None, steps=3):
        import torch, os, sys
        from huggingface_hub import hf_hub_download
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jobs"))
        import worldmodel as W
        self.W = W; self.torch = torch; self.steps = steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        ckp = hf_hub_download(repo, checkpoint, repo_type="dataset", token=tok)
        self.base, self.ck = W.load_checkpoint(ckp, self.device)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.w_pr = 0.6                                        # relational strength (validated in G1d)

    # ---- one conditioned multi-step rollout of a Window -> attacker/defender/ball final positions ----
    def _rollout(self, window, cap, return_start=False, return_trajectory=False):
        import torch
        W = self.W
        b = W.collate([W.make_sample(window, W.RS)], W.MAX_NODES)
        b = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in b.items()}
        pos, vel, team, mask, isball = b["pos"].clone(), b["vel"].clone(), b["team"], b["mask"], b["isball"]
        n = int(mask[0].sum()); tt = team[0]; att = (tt == 1.0)
        fwd = float(cap.get("forward_intent", 0.0)); pac = float(cap.get("pace", 0.0))
        wid = float(cap.get("width", 0.0)); prs = float(cap.get("press_resistance", 0.0))
        if return_start:                                        # positions BEFORE the capability perturbs velocity —
            p0 = pos[0].detach().cpu().numpy()[:n]               # i.e. the shared starting frame (§ pitch viz)
        traj = [pos[0].detach().cpu().numpy()[:n]] if return_trajectory else None   # step 0 = the real start frame
        # kinematic capability -> INPUT velocity perturbation on the attackers (the twin reacts to it)
        vel[0, att, 0] += fwd * SPEED_UNIT                      # forward_intent -> +x drive toward goal
        vel[0, att, 1] += wid * SPEED_UNIT                      # width -> lateral spread
        vel[0, att] *= (1.0 + pac * 0.4)                        # pace -> faster overall
        for _ in range(self.steps):
            feat = W.featurize_torch(pos, vel, torch.zeros_like(pos), team, isball, mask, b["event_ctx"])
            mu = self.base(feat, pos, team, mask)[0]
            disp = vel * DT + mu
            if prs != 0.0:                                      # relational: retain path under pressure (G1d)
                dif = pos[:, :, None, :] - pos[:, None, :, :]; dist = dif.norm(dim=-1)
                opp = (team[:, :, None] != team[:, None, :]).float() * (mask[:, None, :] > 0).float() * (team[:, None, :] < 2).float()
                pressure = torch.exp(-(dist + (1 - opp) * 1e6).min(dim=2).values / PSCALE)
                m = torch.zeros_like(pressure); m[0, att] = prs
                disp = disp * (1 - (self.w_pr * m * pressure).clamp(-0.95, 0.95))[..., None]
            newpos = pos + disp; vel = (newpos - pos) / DT; pos = newpos
            if return_trajectory:
                traj.append(pos[0].detach().cpu().numpy()[:n])   # a REAL intermediate rollout position, not interpolated
        p = pos[0].detach().cpu().numpy()[:n]; tm = tt.detach().cpu().numpy()[:n]
        out = (p[tm == 1.0], p[tm == 0.0], p[tm == 2.0])
        if return_trajectory:
            return out + (([step[tm == 1.0] for step in traj], [step[tm == 0.0] for step in traj], [step[tm == 2.0] for step in traj]),)
        if return_start:
            return out + ((p0[tm == 1.0], p0[tm == 0.0], p0[tm == 2.0]),)
        return out

    def _reward(self, window, cap):
        import fulcrum
        a, f, bl = self._rollout(window, cap)
        if len(a) < 5 or len(f) < 4 or len(bl) == 0: return None
        ss = fulcrum.state_summary({"att": a, "dfn": f, "ball": bl[0]})
        if not ss: return None
        return {"danger": ss["danger"], "space": ss.get("sc", 0.0)}

    def simulate_runner(self, window, capability, unit_ms=2.18):
        """The VALIDATED per-player counterfactual (cf_perplayer: corr 0.994, localization 7.4x). Perturb ONE off-ball
        runner's capability (in real m/s units), roll the twin (everyone reacts), reward = THAT runner's own
        space_creation (validated 1.4-1.6x). This is the correct Tactical Fit signal — a specific player's run into
        space, not a team-wide danger sweep (which grounding showed is flat)."""
        import torch, numpy as np, fulcrum
        from fulcrum import metrics as M
        W = self.W
        b0 = W.collate([W.make_sample(window, W.RS)], W.MAX_NODES); n = int(b0["mask"][0].sum())
        t0 = b0["team"][0].numpy()[:n]; pos0 = b0["pos"][0].numpy()[:n]; ball0 = pos0[t0 == 2.0]
        att = [i for i in range(n) if t0[i] == 1.0]
        if len(att) < 5 or len(ball0) == 0: return None
        fdir = 1.0 if pos0[att, 0].mean() < 52.5 else -1.0
        carrier = min(att, key=lambda i: np.linalg.norm(pos0[i] - ball0[0]))
        runner = max([i for i in att if i != carrier], key=lambda i: pos0[i, 0] * fdir)
        def space(cval):
            b = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in W.collate([W.make_sample(window, W.RS)], W.MAX_NODES).items()}
            pos, vel, tm, mask, isball = b["pos"].clone(), b["vel"].clone(), b["team"], b["mask"], b["isball"]
            vel[0, runner, 0] += cval * unit_ms * fdir
            for _ in range(self.steps):
                feat = W.featurize_torch(pos, vel, torch.zeros_like(pos), tm, isball, mask, b["event_ctx"])
                mu = self.base(feat, pos, tm, mask)[0]; disp = vel * DT + mu
                newpos = pos + disp; vel = (newpos - pos) / DT; pos = newpos
            p = pos[0].detach().cpu().numpy()[:n]; t = tm[0].cpu().numpy()[:n]
            a, f, bl = p[t == 1.0], p[t == 0.0], p[t == 2.0]
            if len(a) < 5 or len(f) < 4 or len(bl) == 0: return None
            sc = M.space_creation({"att": a, "dfn": f, "ball": bl[0], "att_ids": [i for i in range(len(t)) if t[i] == 1.0]})
            return sc.get(runner)
        base_s = space(0.0); cf_s = space(float(capability.get("forward_intent", 1.0)))
        if base_s is None or cf_s is None: return None
        return {"runner_node": int(runner), "delta_space": round(cf_s - base_s, 4), "baseline_space": round(base_s, 4)}

    def simulate_defender(self, window, cover=2.0, unit_ms=2.18):
        """The VALIDATED defensive counterfactual (cf_defender_v2: cover-hole -> danger reduction, corr -0.865, beats
        random null). Move the defender nearest the top exposed hole TOWARD it and measure the danger REDUCED. Values
        a ball-winner/coverer — the defensive half of Simulate. Returns delta_danger (NEGATIVE = danger suppressed)."""
        import torch, numpy as np, fulcrum
        W = self.W
        b0 = W.collate([W.make_sample(window, W.RS)], W.MAX_NODES); n = int(b0["mask"][0].sum())
        t0 = b0["team"][0].numpy()[:n]; pos0 = b0["pos"][0].numpy()[:n]
        dfn = [i for i in range(n) if t0[i] == 0.0]
        a0, f0, bl0 = pos0[t0 == 1.0], pos0[t0 == 0.0], pos0[t0 == 2.0]
        if len(dfn) < 5 or len(a0) < 5 or len(bl0) == 0: return None
        _, _, _, holes = fulcrum.find_holes(a0, np.zeros_like(a0), f0, np.zeros_like(f0), bl0[0], top=1)
        if not holes: return None
        h = holes[0]; hole = np.array([h.get("x", h.get("cx", 52.5)), h.get("y", h.get("cy", 34.0))])
        defender = min(dfn, key=lambda i: np.linalg.norm(pos0[i] - hole))
        direction = (hole - pos0[defender]); direction = direction / (np.linalg.norm(direction) + 1e-6)
        def danger(c):
            b = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in W.collate([W.make_sample(window, W.RS)], W.MAX_NODES).items()}
            pos, vel, tm, mask, isball = b["pos"].clone(), b["vel"].clone(), b["team"], b["mask"], b["isball"]
            vel[0, defender] += torch.tensor(c * unit_ms * direction, dtype=vel.dtype, device=self.device)
            for _ in range(self.steps):
                feat = W.featurize_torch(pos, vel, torch.zeros_like(pos), tm, isball, mask, b["event_ctx"])
                mu = self.base(feat, pos, tm, mask)[0]; disp = vel * DT + mu
                newpos = pos + disp; vel = (newpos - pos) / DT; pos = newpos
            p = pos[0].detach().cpu().numpy()[:n]; t = tm[0].cpu().numpy()[:n]
            aa, ff, bb = p[t == 1.0], p[t == 0.0], p[t == 2.0]
            if len(aa) < 5 or len(ff) < 4 or len(bb) == 0: return None
            ss = fulcrum.state_summary({"att": aa, "dfn": ff, "ball": bb[0]}); return ss["danger"] if ss else None
        d0, dc = danger(0.0), danger(cover)
        if d0 is None or dc is None: return None
        return {"defender_node": int(defender), "delta_danger": round(dc - d0, 4), "baseline_danger": round(d0, 4)}

    def simulate(self, window, capability):
        """One phase: reward under `capability` minus reward at baseline (all-zero capability). -> deltas + per-axis."""
        base = self._reward(window, {})
        cf = self._reward(window, capability)
        if base is None or cf is None:
            return None
        per_axis = {}
        for ax in AXES:
            if capability.get(ax):
                one = self._reward(window, {ax: capability[ax]})
                if one is not None:
                    per_axis[ax] = round(one["danger"] - base["danger"], 3)
        return {"delta_danger": round(cf["danger"] - base["danger"], 3),
                "delta_space": round(cf["space"] - base["space"], 3),
                "baseline_danger": round(base["danger"], 3), "per_axis_danger": per_axis}

    def fit_report(self, capability, anchors, boot=300):
        """Aggregate the fit over many real phases (anchors = list of Windows) with a bootstrap 95% CI on Δdanger.
        Returns the Tactical Fit readout — HONEST language, within-horizon, not a signing claim."""
        deltas, per_axis_all = [], {ax: [] for ax in AXES}
        for w in anchors:
            s = self.simulate(w, capability)
            if s is None: continue
            deltas.append(s["delta_danger"])
            for ax, v in s["per_axis_danger"].items(): per_axis_all[ax].append(v)
        if not deltas:
            return {"error": "no valid phases"}
        d = np.array(deltas); rng = np.random.default_rng(0)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(boot)]
        pa = {ax: round(float(np.mean(v)), 3) for ax, v in per_axis_all.items() if v}
        return {"capability": capability, "phases": len(deltas),
                "mean_delta_danger": round(float(d.mean()), 3),
                "delta_danger_CI95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                "per_axis_mean_danger": pa,
                "simulation_validity": ("high" if len(deltas) > 100 else "moderate" if len(deltas) > 30 else "low"),
                "epistemic": "counterfactual tactical fit — changes the model's predicted danger within the ~2s "
                             "validated horizon; NOT a signing/results claim",
                "status": "face-valid:counterfactual (G2 monotonic corr 0.925, G2.5 no-op/seed-stable; specificity moderate)"}
