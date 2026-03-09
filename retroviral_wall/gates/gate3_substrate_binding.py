from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid


class SubstrateBindingGate(AbstractGate):
    def __init__(self):
        super().__init__("substrate_binding")
        self.weights = {
            "groove_detected": 0.25,
            "groove_geometry": 0.25,
            "groove_charge": 0.20,
            "sasa_active_site": 0.15,
            "thumb_contact": 0.15,
        }

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]

            triad_found = float(hc.get("triad_found_bin", 0.0))
            motif_found = float(hc.get("qg_motif_found", 0.0))
            groove_detected = float(np.clip(max(triad_found, motif_found), 0, 1))

            hb = float(hc.get("pocket_hbonds_per_res", np.nan))
            hyd = float(hc.get("pocket_hydrophobic_per_res", np.nan))
            if np.isnan(hb):
                hb = 0.3
            if np.isnan(hyd):
                hyd = 0.3
            groove_geometry = float(np.clip(np.exp(-abs(hb - 0.6) * 2.0) * np.exp(-abs(hyd - 0.35) * 1.8), 0, 1))

            net_charge = float(hc.get("native_net_charge", 0.0))
            groove_charge = float(sigmoid(net_charge / 5.0))

            sasa = float(hc.get("sasa_avg", np.nan))
            sasa_score = float(sigmoid((sasa - 110) / 25)) if not np.isnan(sasa) else 0.5

            thumb_detected = 0.0 if np.isnan(float(hc.get("thumb_total_residues", np.nan))) else 1.0
            thumb_contact = 0.75 if thumb_detected else 0.2

            sub = {
                "groove_detected": groove_detected,
                "groove_geometry": groove_geometry,
                "groove_charge": groove_charge,
                "sasa_active_site": sasa_score,
                "thumb_contact": thumb_contact,
            }
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            confidence = float(0.3 + 0.7 * groove_detected)
            failure = "" if groove_detected > 0 else "No confident substrate groove proxy"
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, confidence, failure))
        return results
