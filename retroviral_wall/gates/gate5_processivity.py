from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid


class ProcessivityGate(AbstractGate):
    def __init__(self):
        super().__init__("processivity")
        self.weights = {
            "thumb_detected": 0.25,
            "thumb_quality": 0.20,
            "structural_sim": 0.25,
            "hairpin": 0.15,
            "length_proxy": 0.15,
        }

    def _length_score(self, seq_len: float) -> float:
        if seq_len < 200:
            return 0.1
        if 300 <= seq_len <= 700:
            return 0.9
        return float(max(0.4, 0.9 * np.exp(-(seq_len - 700) / 500)))

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        processivity_lit = external_data.get("rt_processivity", {})
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            seq_len = float(row["protein_length_aa"])

            thumb_charge = float(hc.get("thumb_surface_net_charge", np.nan))
            thumb_detected = 0.0 if np.isnan(thumb_charge) else 1.0
            thumb_quality = float(sigmoid(thumb_charge / 5.0)) if thumb_detected else 0.15

            tm_mmlv = float(hc.get("foldseek_TM_MMLV", np.nan))
            tm_hiv = float(hc.get("foldseek_TM_HIV1", np.nan))
            if np.isnan(tm_hiv):
                tm_hiv = float(hc.get("foldseek_TM_HIV1RT", 0.0))
            if np.isnan(tm_mmlv):
                tm_mmlv = 0.0
            structural_sim = float(max(tm_mmlv, tm_hiv))

            hairpin = float(hc.get("hairpin_pass", np.nan))
            if np.isnan(hairpin):
                hairpin = 0.3

            length_proxy = self._length_score(seq_len)

            sub = {
                "thumb_detected": thumb_detected,
                "thumb_quality": thumb_quality,
                "structural_sim": structural_sim,
                "hairpin": float(hairpin),
                "length_proxy": length_proxy,
            }

            lit = processivity_lit.get(rt_name)
            if lit is not None:
                lit_score = float(min(lit / 50.0, 1.0))
                sub["structural_sim"] = 0.5 * sub["structural_sim"] + 0.5 * lit_score

            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            conf = 0.8 if thumb_detected else 0.4
            failure = "" if thumb_detected else "No thumb-domain proxy detected"
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, conf, failure))
        return results
