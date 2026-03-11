from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.pdb_utils import foldability_structure_features


class FoldabilityGate(AbstractGate):
    def __init__(self):
        super().__init__("foldability")
        self.weights = {
            "plddt_mean": 0.18,
            "plddt_core": 0.12,
            "high_conf_frac": 0.18,
            "low_conf_frac": 0.10,
            "longest_conf_segment": 0.12,
            "contact_density": 0.10,
            "thermo_37": 0.10,
            "solubility": 0.06,
            "instability": 0.04,
        }

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            sf = foldability_structure_features(pdb_path)
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]

            thermo_37 = np.nanmean([hc.get("t40_raw", np.nan), hc.get("t45_raw", np.nan)])
            if np.isnan(thermo_37):
                thermo_37 = 0.5

            camsol = hc.get("camsol_score", np.nan)
            solubility = sigmoid(camsol) if not np.isnan(camsol) else 0.5

            instability_idx = hc.get("instability_index", np.nan)
            instability = sigmoid(-(instability_idx - 40) / 10) if not np.isnan(instability_idx) else 0.5

            sub = {
                "plddt_mean": sf["plddt_mean"],
                "plddt_core": sf["plddt_core"],
                "high_conf_frac": sf["high_conf_frac"],
                "low_conf_frac": float(1.0 - sf["low_conf_frac"]),
                "longest_conf_segment": sf["longest_conf_segment"],
                "contact_density": sf["contact_density"],
                "thermo_37": float(thermo_37),
                "solubility": float(solubility),
                "instability": float(instability),
            }
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            miss = sum(1 for v in sub.values() if abs(v - 0.5) < 1e-8)
            conf = float(max(0.2, 1 - 0.12 * miss))
            failure = ""
            if sf["plddt_mean"] < 0.55:
                failure = "Low global pLDDT"
            elif sf["high_conf_frac"] < 0.25:
                failure = "Insufficient high-confidence folded core"
            elif thermo_37 < 0.35:
                failure = "Low thermostability proxy"

            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, conf, failure))
        return results
