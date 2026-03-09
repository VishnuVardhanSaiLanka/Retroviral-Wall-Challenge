from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.pdb_utils import extract_ca_residues


class ProcessivityGate(AbstractGate):
    def __init__(self):
        super().__init__("processivity")
        self.weights = {
            "path_detected": 0.15,
            "path_positive_density": 0.20,
            "path_extension": 0.20,
            "path_catalytic_proximity": 0.15,
            "structural_sim": 0.20,
            "thumb_support": 0.10,
        }
        self.positive_residues = {"LYS": 1.0, "ARG": 1.0, "HIS": 0.5}
        self.polar_residues = {"ASN", "GLN", "SER", "THR", "TYR", "LYS", "ARG", "HIS"}
        self.acidic_residues = {"ASP", "GLU"}

    def _best_processive_path(self, pdb_path: str) -> dict[str, float] | None:
        residues = [r for r in extract_ca_residues(pdb_path) if np.isfinite(r["coord"]).all()]
        if len(residues) < 20:
            return None

        coords = np.array([r["coord"] for r in residues], dtype=float)
        tree = cKDTree(coords)
        contacts = np.array([len(tree.query_ball_point(coord, 10.0)) - 1 for coord in coords], dtype=float)
        exposure = 1.0 / (1.0 + np.exp((contacts - 14.0) / 2.5))
        positive = np.array([self.positive_residues.get(r["resname"], 0.0) for r in residues], dtype=float)
        polar = np.array([r["resname"] in self.polar_residues for r in residues], dtype=bool)
        acidic_idx = [i for i, r in enumerate(residues) if r["resname"] in self.acidic_residues]
        acidic_center = coords[acidic_idx].mean(axis=0) if acidic_idx else coords.mean(axis=0)

        eligible = (exposure >= 0.42) & ((positive > 0) | polar)
        eligible_idx = np.where(eligible)[0]
        if len(eligible_idx) == 0:
            return None

        visited: set[int] = set()
        best: dict[str, float] | None = None
        for root in eligible_idx:
            if root in visited:
                continue
            queue = [root]
            component: list[int] = []
            visited.add(root)
            while queue:
                idx = queue.pop()
                component.append(idx)
                for nbr in tree.query_ball_point(coords[idx], 9.0):
                    if nbr in visited or not eligible[nbr]:
                        continue
                    visited.add(nbr)
                    queue.append(nbr)

            comp = np.array(component, dtype=int)
            if len(comp) < 3:
                continue
            comp_coords = coords[comp]
            comp_centroid = comp_coords.mean(axis=0)
            centered = comp_coords - comp_centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            projections = centered @ axis
            extension = float(projections.max() - projections.min())
            path_extension = float(sigmoid((extension - 18.0) / 5.0))
            pos_density = float(positive[comp].sum() / max(len(comp), 1))
            path_positive_density = float(sigmoid((pos_density - 0.22) * 7.0))
            centroid_dist = float(np.linalg.norm(comp_centroid - acidic_center))
            path_catalytic_proximity = float(np.exp(-((centroid_dist - 16.0) ** 2) / (2 * 8.0**2)))
            thumb_support = float(sigmoid((np.percentile(projections, 90) - 6.0) / 3.0))

            path = {
                "path_detected": 1.0 if pos_density >= 0.18 and extension >= 14.0 else 0.35,
                "path_positive_density": path_positive_density,
                "path_extension": path_extension,
                "path_catalytic_proximity": path_catalytic_proximity,
                "thumb_support": thumb_support,
                "_rank_score": 0.30 * path_positive_density + 0.30 * path_extension + 0.25 * path_catalytic_proximity + 0.15 * thumb_support,
            }
            if best is None or path["_rank_score"] > best["_rank_score"]:
                best = path
        return best

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        processivity_lit = external_data.get("rt_processivity", {})
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            path = self._best_processive_path(pdb_path)

            tm_mmlv = float(hc.get("foldseek_TM_MMLV", np.nan))
            tm_hiv = float(hc.get("foldseek_TM_HIV1", np.nan))
            if np.isnan(tm_hiv):
                tm_hiv = float(hc.get("foldseek_TM_HIV1RT", 0.0))
            if np.isnan(tm_mmlv):
                tm_mmlv = 0.0
            structural_sim = float(max(tm_mmlv, tm_hiv))
            thumb_charge = float(hc.get("thumb_surface_net_charge", np.nan))
            thumb_context = float(sigmoid(thumb_charge / 6.0)) if not np.isnan(thumb_charge) else 0.35

            lit = processivity_lit.get(rt_name)
            if lit is not None:
                lit_score = float(min(lit / 50.0, 1.0))
                structural_sim = 0.6 * structural_sim + 0.4 * lit_score

            if path is None:
                sub = {
                    "path_detected": 0.25,
                    "path_positive_density": 0.5,
                    "path_extension": 0.5,
                    "path_catalytic_proximity": 0.5,
                    "structural_sim": float(structural_sim),
                    "thumb_support": float(thumb_context),
                }
                conf = 0.35
                failure = "No confident processive contact path"
            else:
                sub = {k: float(v) for k, v in path.items() if not k.startswith("_")}
                sub["structural_sim"] = float(structural_sim)
                sub["thumb_support"] = float(0.7 * sub["thumb_support"] + 0.3 * thumb_context)
                conf = float(np.clip(0.35 + 0.65 * np.mean([sub["path_detected"], sub["path_extension"]]), 0, 1))
                failure = "" if sub["path_detected"] >= 1.0 and sub["path_catalytic_proximity"] >= 0.35 else "No confident processive contact path"

            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, conf, failure))
        return results
