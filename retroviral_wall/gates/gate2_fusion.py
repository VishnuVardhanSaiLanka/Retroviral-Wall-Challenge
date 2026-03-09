from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid


class FusionCompatibilityGate(AbstractGate):
    def __init__(self):
        super().__init__("fusion_compat")
        self.weights = {
            "clash_score": 0.30,
            "alignment_quality": 0.15,
            "active_site_access": 0.25,
            "size_penalty": 0.15,
            "terminus_flex": 0.15,
        }

    @staticmethod
    def _size_score(length: float) -> float:
        if length <= 800:
            return 1.0
        return float(np.exp(-(length - 800) / 200.0))

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            length = float(row.get("protein_length_aa", 500))

            align = float(hc.get("foldseek_TM_MMLV", np.nan))
            if np.isnan(align):
                align = float(hc.get("foldseek_best_TM", 0.4))

            # Proxy for steric burden: more contacts per residue + size.
            hydrophobic_per_res = float(hc.get("hydrophobic_per_res", 0.5))
            clash_score = float(np.clip(np.exp(-(hydrophobic_per_res * length) / 500.0), 0, 1))

            triad = float(hc.get("triad_found_bin", 0.0))
            active_site_access = 0.3 + 0.7 * triad * align

            size_penalty = self._size_score(length)
            term_plddt = float(hc.get("best_confidence", 0.7))
            terminus_flex = float(np.clip(1.0 - term_plddt / 2.0, 0, 1))

            sub = {
                "clash_score": clash_score,
                "alignment_quality": float(np.clip(align, 0, 1)),
                "active_site_access": float(np.clip(active_site_access, 0, 1)),
                "size_penalty": float(np.clip(size_penalty, 0, 1)),
                "terminus_flex": terminus_flex,
            }
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            confidence = float(np.clip(0.4 + 0.6 * sub["alignment_quality"], 0, 1))
            failure = "" if sub["clash_score"] >= 0.25 else "Likely steric incompatibility with Cas9 context"
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, confidence, failure))
        return results
