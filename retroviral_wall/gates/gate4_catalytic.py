from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult


class CatalyticCompetenceGate(AbstractGate):
    def __init__(self):
        super().__init__("catalytic")
        self.weights = {
            "motif_present": 0.30,
            "triad_geometry": 0.30,
            "motif_ss": 0.15,
            "metal_coordination": 0.15,
            "esm_if_quality": 0.10,
        }
        self.ref_d1_d2 = 6.5
        self.ref_d2_d3 = 3.8
        self.tol = 3.0

    def _metal_coord(self, d1: float, d2: float) -> float:
        if np.isnan(d1) or np.isnan(d2):
            return 0.2
        if d1 < 10 and d2 < 10:
            return 0.8
        return 0.3

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            d1 = float(hc.get("D1_D2_dist", np.nan))
            d2 = float(hc.get("D2_D3_dist", np.nan))
            motif_present = 0.0 if np.isnan(d1) and np.isnan(d2) else 1.0

            if motif_present > 0:
                dev1 = abs(d1 - self.ref_d1_d2) if not np.isnan(d1) else self.tol
                dev2 = abs(d2 - self.ref_d2_d3) if not np.isnan(d2) else self.tol
                triad_geo = float(np.exp(-(dev1**2 + dev2**2) / (2 * self.tol**2)))
            else:
                triad_geo = 0.0

            motif_vals = [
                float(hc.get("yxdd_y_is_strand", np.nan)),
                float(hc.get("yxdd_d2_is_strand", np.nan)),
            ]
            motif_vals = [v for v in motif_vals if not np.isnan(v)]
            if motif_vals:
                motif_ss = float(np.mean(motif_vals))
            else:
                motif_ss = 0.5 if motif_present else 0.1

            metal = self._metal_coord(d1, d2)

            esm_perp = float(hc.get("perplexity", np.nan))
            esm_if_quality = float(np.exp(-max(esm_perp - 5, 0) / 10.0)) if not np.isnan(esm_perp) else 0.5

            sub = {
                "motif_present": motif_present,
                "triad_geometry": triad_geo,
                "motif_ss": float(motif_ss),
                "metal_coordination": float(metal),
                "esm_if_quality": float(esm_if_quality),
            }
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            failure = "" if motif_present > 0 else "No catalytic motif detected"
            conf = 0.9 if motif_present > 0 else 0.4
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, conf, failure))
        return results
