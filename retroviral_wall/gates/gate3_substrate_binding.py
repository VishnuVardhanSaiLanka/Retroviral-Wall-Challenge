from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.structural_features import substrate_path_features


class SubstrateBindingGate(AbstractGate):
    def __init__(self):
        super().__init__("substrate_binding")
        self.weights = {
            "path_detected": 0.12,
            "path_positive_continuity": 0.22,
            "path_openness": 0.16,
            "path_extension": 0.14,
            "catalytic_accessibility": 0.18,
            "steric_crowding_penalty": 0.08,
            "hydrophobic_interrupt": 0.05,
            "track_to_thumb": 0.05,
        }

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            path = substrate_path_features(pdb_path)
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            triad_found = float(hc.get("triad_found_bin", 0.0))
            qg_found = float(hc.get("qg_motif_found", 0.0))
            motif_present = max(triad_found, qg_found)
            d1 = float(hc.get("D1_D2_dist", np.nan))
            d2 = float(hc.get("D2_D3_dist", np.nan))
            if motif_present > 0 and (not np.isnan(d1) or not np.isnan(d2)):
                dev1 = abs(d1 - 6.5) if not np.isnan(d1) else 3.0
                dev2 = abs(d2 - 3.8) if not np.isnan(d2) else 3.0
                motif_context = float(np.exp(-(dev1**2 + dev2**2) / (2 * 3.0**2)))
            else:
                motif_context = 0.35 if motif_present > 0 else 0.15

            sub = {
                "path_detected": 1.0 if path["positive_continuity"] >= 0.45 and path["path_openness"] >= 0.5 else 0.35,
                "path_positive_continuity": float(path["positive_continuity"]),
                "path_openness": float(path["path_openness"]),
                "path_extension": float(path["path_extension"]),
                "catalytic_accessibility": float(0.7 * path["catalytic_accessibility"] + 0.3 * motif_context),
                "steric_crowding_penalty": float(path["steric_crowding_penalty"]),
                "hydrophobic_interrupt": float(path["hydrophobic_interrupt"]),
                "track_to_thumb": float(path["track_to_thumb"]),
            }
            confidence = float(np.clip(0.35 + 0.65 * np.mean([sub["path_positive_continuity"], sub["catalytic_accessibility"]]), 0, 1))
            failure = "" if sub["path_detected"] >= 1.0 and sub["catalytic_accessibility"] >= 0.4 else "No confident catalytic-adjacent substrate path"
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, confidence, failure))
        return results
