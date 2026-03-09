from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.pdb_utils import mean_and_core_plddt


class FoldabilityGate(AbstractGate):
    def __init__(self):
        super().__init__("foldability")
        self.weights = {
            "plddt_mean": 0.30,
            "plddt_core": 0.20,
            "thermo_37": 0.25,
            "solubility": 0.15,
            "instability": 0.10,
        }

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            plddt_mean, plddt_core = mean_and_core_plddt(pdb_path)
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]

            thermo_37 = np.nanmean([hc.get("t40_raw", np.nan), hc.get("t45_raw", np.nan)])
            if np.isnan(thermo_37):
                thermo_37 = 0.5

            camsol = hc.get("camsol_score", np.nan)
            solubility = sigmoid(camsol) if not np.isnan(camsol) else 0.5

            instability_idx = hc.get("instability_index", np.nan)
            instability = sigmoid(-(instability_idx - 40) / 10) if not np.isnan(instability_idx) else 0.5

            sub = {
                "plddt_mean": plddt_mean,
                "plddt_core": plddt_core,
                "thermo_37": float(thermo_37),
                "solubility": float(solubility),
                "instability": float(instability),
            }
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            miss = sum(1 for v in sub.values() if abs(v - 0.5) < 1e-8)
            conf = float(max(0.2, 1 - 0.12 * miss))
            failure = ""
            if plddt_mean < 0.55:
                failure = "Low global pLDDT"
            elif thermo_37 < 0.35:
                failure = "Low thermostability proxy"

            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, conf, failure))
        return results
