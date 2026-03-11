from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.pdb_utils import extract_ca_residues


class SubstrateBindingGate(AbstractGate):
    def __init__(self):
        super().__init__("substrate_binding")
        self.weights = {
            "patch_detected": 0.15,
            "patch_positive_density": 0.35,
            "patch_exposure": 0.20,
            "patch_size": 0.05,
            "patch_contiguity": 0.05,
            "active_site_proximity": 0.20,
        }
        self.positive_residues = {"LYS": 1.0, "ARG": 1.0, "HIS": 0.5}
        self.polar_residues = {"ASN", "GLN", "SER", "THR", "TYR", "LYS", "ARG", "HIS"}
        self.acidic_residues = {"ASP", "GLU"}

    def _acidic_cluster_centroids(self, residues: list[dict], tree: cKDTree, coords: np.ndarray) -> list[np.ndarray]:
        acidic_idx = [i for i, r in enumerate(residues) if r["resname"] in self.acidic_residues]
        if not acidic_idx:
            return []

        acidic_set = set(acidic_idx)
        visited: set[int] = set()
        centroids: list[np.ndarray] = []
        for root in acidic_idx:
            if root in visited:
                continue
            queue = [root]
            component: list[int] = []
            visited.add(root)
            while queue:
                idx = queue.pop()
                component.append(idx)
                for nbr in tree.query_ball_point(coords[idx], 9.0):
                    if nbr in visited or nbr not in acidic_set:
                        continue
                    visited.add(nbr)
                    queue.append(nbr)
            if len(component) < 2:
                continue
            centroids.append(coords[np.array(component, dtype=int)].mean(axis=0))
        return centroids

    def _best_electropositive_patch(self, pdb_path: str) -> dict[str, float] | None:
        residues = extract_ca_residues(pdb_path)
        residues = [r for r in residues if np.isfinite(r["coord"]).all()]
        if len(residues) < 20:
            return None

        coords = np.array([r["coord"] for r in residues], dtype=float)
        tree = cKDTree(coords)
        acidic_centroids = self._acidic_cluster_centroids(residues, tree, coords)
        contacts = np.array([len(tree.query_ball_point(coord, 10.0)) - 1 for coord in coords], dtype=float)

        # Low local contact count is a practical proxy for surface exposure.
        exposure = 1.0 / (1.0 + np.exp((contacts - 14.0) / 2.5))
        surface_mask = exposure >= 0.45
        if not surface_mask.any():
            top = np.argsort(exposure)[-10:]
            surface_mask[top] = True

        positive_charge = np.array([self.positive_residues.get(r["resname"], 0.0) for r in residues], dtype=float)
        polar_mask = np.array([r["resname"] in self.polar_residues for r in residues], dtype=bool)
        eligible = surface_mask & ((positive_charge > 0) | polar_mask)
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
                for nbr in tree.query_ball_point(coords[idx], 8.0):
                    if nbr in visited or not eligible[nbr]:
                        continue
                    visited.add(nbr)
                    queue.append(nbr)

            comp = np.array(component, dtype=int)
            comp_exposure = exposure[comp]
            comp_charge = positive_charge[comp]
            mean_positive_density = float(comp_charge.sum() / max(len(comp), 1))
            patch_size = len(comp)
            mean_exposure = float(comp_exposure.mean())
            centroid = coords[comp].mean(axis=0)
            neighbor_counts = []
            for idx in comp:
                same_patch = [nbr for nbr in tree.query_ball_point(coords[idx], 8.0) if nbr in component and nbr != idx]
                neighbor_counts.append(len(same_patch))
            mean_neighbors = float(np.mean(neighbor_counts)) if neighbor_counts else 0.0
            contiguity = float(sigmoid((mean_neighbors - 2.5) / 1.5))
            if acidic_centroids:
                dists = [float(np.linalg.norm(centroid - acid_centroid)) for acid_centroid in acidic_centroids]
                best_dist = min(dists)
                active_site_proximity = float(np.exp(-((best_dist - 12.0) ** 2) / (2 * 6.0**2)))
            else:
                active_site_proximity = 0.5

            patch = {
                "patch_detected": 1.0 if mean_positive_density >= 0.18 and mean_exposure >= 0.55 else 0.35,
                "patch_positive_density": float(sigmoid((mean_positive_density - 0.28) * 8.0)),
                "patch_exposure": mean_exposure,
                "patch_size": float(sigmoid((8.0 - patch_size) / 2.0)),
                "patch_contiguity": float(1.0 - contiguity),
                "active_site_proximity": active_site_proximity,
                "_rank_score": 0.45 * mean_positive_density + 0.20 * mean_exposure + 0.25 * active_site_proximity + 0.10 * max(0.0, 1.0 - min(patch_size / 12.0, 1.0)),
            }
            if best is None or patch["_rank_score"] > best["_rank_score"]:
                best = patch
        return best

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            pdb_path = f"{structures_dir}/{rt_name}.pdb"
            patch = self._best_electropositive_patch(pdb_path)
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
            if patch is None:
                sub = {
                    "patch_detected": 0.25,
                    "patch_positive_density": 0.5,
                    "patch_exposure": 0.5,
                    "patch_size": 0.5,
                    "patch_contiguity": 0.5,
                    "active_site_proximity": motif_context,
                }
                confidence = 0.3
                failure = "No confident electropositive surface patch"
            else:
                sub = {k: float(v) for k, v in patch.items() if not k.startswith("_")}
                sub["active_site_proximity"] = float(0.7 * sub["active_site_proximity"] + 0.3 * motif_context)
                confidence = float(np.clip(0.35 + 0.65 * np.mean([sub["patch_detected"], sub["patch_exposure"]]), 0, 1))
                failure = "" if sub["patch_detected"] >= 1.0 and sub["active_site_proximity"] >= 0.4 else "No confident active-site-adjacent electropositive patch"
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, confidence, failure))
        return results
