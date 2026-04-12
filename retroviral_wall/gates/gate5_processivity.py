from __future__ import annotations

import numpy as np

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.structural_features import substrate_path_features


class ProcessivityGate(AbstractGate):
    def __init__(self, use_similarity_support: bool = True, use_literature_prior: bool = True):
        super().__init__("processivity")
        self.use_similarity_support = use_similarity_support
        self.use_literature_prior = use_literature_prior
        self.weights = {
            "track_detected": 0.12,
            "path_positive_density": 0.16,
            "path_extension": 0.18,
            "path_catalytic_proximity": 0.14,
            "structural_sim": 0.10,
            "thumb_support": 0.12,
            "path_openness": 0.08,
            "steric_crowding_penalty": 0.05,
            "hydrophobic_interrupt": 0.05,
        }

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        processivity_lit = external_data.get("rt_processivity", {})
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            path = substrate_path_features(pdb_path)

            if self.use_similarity_support:
                tm_mmlv = float(hc.get("foldseek_TM_MMLV", np.nan))
                tm_hiv = float(hc.get("foldseek_TM_HIV1", np.nan))
                if np.isnan(tm_hiv):
                    tm_hiv = float(hc.get("foldseek_TM_HIV1RT", 0.0))
                if np.isnan(tm_mmlv):
                    tm_mmlv = 0.0
                structural_sim = float(max(tm_mmlv, tm_hiv))
            else:
                structural_sim = 0.5
            thumb_charge = float(hc.get("thumb_surface_net_charge", np.nan))
            thumb_context = float(sigmoid(thumb_charge / 6.0)) if not np.isnan(thumb_charge) else 0.35

            lit = processivity_lit.get(rt_name)
            if lit is not None and self.use_literature_prior:
                lit_score = float(min(lit / 50.0, 1.0))
                structural_sim = 0.6 * structural_sim + 0.4 * lit_score

            sub = {
                "track_detected": 1.0 if path["positive_continuity"] >= 0.45 and path["path_extension"] >= 0.5 else 0.35,
                "path_positive_density": float(path["positive_continuity"]),
                "path_extension": float(path["path_extension"]),
                "path_catalytic_proximity": float(path["catalytic_accessibility"]),
                "structural_sim": float(structural_sim),
                "thumb_support": float(0.7 * path["track_to_thumb"] + 0.3 * thumb_context),
                "path_openness": float(path["path_openness"]),
                "steric_crowding_penalty": float(path["steric_crowding_penalty"]),
                "hydrophobic_interrupt": float(path["hydrophobic_interrupt"]),
            }
            conf = float(np.clip(0.35 + 0.65 * np.mean([sub["track_detected"], sub["path_extension"], sub["thumb_support"]]), 0, 1))
            failure = "" if sub["track_detected"] >= 1.0 and sub["path_catalytic_proximity"] >= 0.35 else "No confident processive contact path"

            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, conf, failure))
        return results
