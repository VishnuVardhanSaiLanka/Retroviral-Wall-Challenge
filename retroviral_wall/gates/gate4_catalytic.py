from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.pdb_utils import extract_ca_residues


class CatalyticCompetenceGate(AbstractGate):
    def __init__(self):
        super().__init__("catalytic")
        self.weights = {
            "acidic_cluster_detected": 0.15,
            "acidic_cluster_compactness": 0.25,
            "acidic_cluster_exposure": 0.20,
            "catalytic_neighborhood": 0.20,
            "motif_context": 0.10,
            "esm_if_quality": 0.10,
        }
        self.ref_d1_d2 = 6.5
        self.ref_d2_d3 = 3.8
        self.tol = 3.0
        self.acidic_residues = {"ASP", "GLU"}
        self.aromatic_residues = {"TYR", "PHE", "TRP"}
        self.basic_residues = {"LYS", "ARG", "HIS"}

    def _best_acidic_cluster(self, pdb_path: str) -> dict[str, float] | None:
        residues = [r for r in extract_ca_residues(pdb_path) if np.isfinite(r["coord"]).all()]
        if len(residues) < 20:
            return None

        coords = np.array([r["coord"] for r in residues], dtype=float)
        tree = cKDTree(coords)
        contacts = np.array([len(tree.query_ball_point(coord, 10.0)) - 1 for coord in coords], dtype=float)
        exposure = 1.0 / (1.0 + np.exp((contacts - 14.0) / 2.5))

        acidic_idx = [i for i, r in enumerate(residues) if r["resname"] in self.acidic_residues]
        if len(acidic_idx) < 2:
            return None

        acidic_set = set(acidic_idx)
        visited: set[int] = set()
        best: dict[str, float] | None = None
        for root in acidic_idx:
            if root in visited:
                continue
            queue = [root]
            component: list[int] = []
            visited.add(root)
            while queue:
                idx = queue.pop()
                component.append(idx)
                for nbr in tree.query_ball_point(coords[idx], 8.5):
                    if nbr in visited or nbr not in acidic_set:
                        continue
                    visited.add(nbr)
                    queue.append(nbr)

            comp = np.array(component, dtype=int)
            cluster_coords = coords[comp]
            centroid = cluster_coords.mean(axis=0)
            rms = float(np.sqrt(np.mean(np.sum((cluster_coords - centroid) ** 2, axis=1))))
            compactness = float(np.exp(-(max(rms - 4.5, 0.0) ** 2) / (2 * 2.5**2)))
            cluster_exposure = float(exposure[comp].mean())

            neighbors = tree.query_ball_point(centroid, 12.0)
            aromatic = sum(1 for idx in neighbors if residues[idx]["resname"] in self.aromatic_residues)
            basic = sum(1 for idx in neighbors if residues[idx]["resname"] in self.basic_residues)
            catalytic_neighborhood = float(
                0.55 * sigmoid((basic - 1.5) / 1.0) + 0.45 * sigmoid((aromatic - 0.5) / 1.0)
            )

            cluster = {
                "acidic_cluster_detected": 1.0 if len(comp) >= 2 else 0.3,
                "acidic_cluster_compactness": compactness,
                "acidic_cluster_exposure": cluster_exposure,
                "catalytic_neighborhood": catalytic_neighborhood,
                "_rank_score": 0.40 * compactness + 0.25 * cluster_exposure + 0.20 * catalytic_neighborhood + 0.15 * min(len(comp) / 3.0, 1.0),
            }
            if best is None or cluster["_rank_score"] > best["_rank_score"]:
                best = cluster
        return best

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            cluster = self._best_acidic_cluster(pdb_path)

            d1 = float(hc.get("D1_D2_dist", np.nan))
            d2 = float(hc.get("D2_D3_dist", np.nan))
            motif_present = 0.0 if np.isnan(d1) and np.isnan(d2) else 1.0
            if motif_present > 0:
                dev1 = abs(d1 - self.ref_d1_d2) if not np.isnan(d1) else self.tol
                dev2 = abs(d2 - self.ref_d2_d3) if not np.isnan(d2) else self.tol
                motif_context = float(np.exp(-(dev1**2 + dev2**2) / (2 * self.tol**2)))
            else:
                motif_context = 0.0

            motif_vals = [
                float(hc.get("yxdd_y_is_strand", np.nan)),
                float(hc.get("yxdd_d1_is_turn", np.nan)),
                float(hc.get("yxdd_d2_is_strand", np.nan)),
            ]
            motif_vals = [v for v in motif_vals if not np.isnan(v)]
            if motif_vals:
                motif_context = float(0.6 * motif_context + 0.4 * np.mean(motif_vals))
            else:
                motif_context = float(0.7 * motif_context + 0.3 * (0.35 if motif_present else 0.1))

            esm_perp = float(hc.get("perplexity", np.nan))
            esm_if_quality = float(np.exp(-max(esm_perp - 5, 0) / 10.0)) if not np.isnan(esm_perp) else 0.5

            if cluster is None:
                sub = {
                    "acidic_cluster_detected": 0.25,
                    "acidic_cluster_compactness": 0.5,
                    "acidic_cluster_exposure": 0.5,
                    "catalytic_neighborhood": 0.5,
                    "motif_context": float(max(motif_context, 0.15)),
                    "esm_if_quality": float(esm_if_quality),
                }
                conf = 0.35
                failure = "No confident catalytic acidic cluster"
            else:
                sub = {k: float(v) for k, v in cluster.items() if not k.startswith("_")}
                sub["motif_context"] = float(max(motif_context, 0.15))
                sub["esm_if_quality"] = float(esm_if_quality)
                conf = float(np.clip(0.35 + 0.65 * np.mean([sub["acidic_cluster_detected"], sub["acidic_cluster_compactness"]]), 0, 1))
                failure = "" if sub["acidic_cluster_detected"] >= 1.0 and sub["acidic_cluster_compactness"] >= 0.4 else "Catalytic cluster geometry not convincing"
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, conf, failure))
        return results
