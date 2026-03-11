from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from retroviral_wall.gates.base import AbstractGate, GateResult
from retroviral_wall.utils.alignment import structural_align
from retroviral_wall.utils.geometry import compute_steric_clashes, sigmoid
from retroviral_wall.utils.pdb_utils import extract_atom_records


class FusionCompatibilityGate(AbstractGate):
    def __init__(self):
        super().__init__("fusion_compat")
        self.weights = {
            "clash_score": 0.35,
            "alignment_quality": 0.15,
            "active_site_access": 0.25,
            "linker_feasibility": 0.15,
            "size_penalty": 0.10,
        }
        self._reference_context: dict | None = None

    @staticmethod
    def _size_score(length: float) -> float:
        if length <= 800:
            return 1.0
        return float(np.exp(-(length - 800) / 200.0))

    def _load_reference_context(self, external_data: dict) -> dict:
        if self._reference_context is not None:
            return self._reference_context

        pe_path = external_data.get("pe2_complex", {}).get("path")
        if not pe_path:
            self._reference_context = {}
            return self._reference_context

        cas9_atoms = extract_atom_records(pe_path, chains={"A"})
        rt_atoms = extract_atom_records(pe_path, chains={"E"})
        env_atoms = extract_atom_records(pe_path, chains={"A", "B", "C", "D", "F"})
        rt_ca = [a for a in rt_atoms if a["atom"] == "CA"]
        cas9_ca = [a for a in cas9_atoms if a["atom"] == "CA"]
        nuc_atoms = extract_atom_records(pe_path, chains={"B", "C", "D", "F"})

        anchor_coord = np.zeros(3)
        if cas9_ca and rt_ca:
            cas9_coords = np.array([a["coord"] for a in cas9_ca], dtype=float)
            rt_coords = np.array([a["coord"] for a in rt_ca], dtype=float)
            rt_tree = cKDTree(rt_coords)
            dists, idx = rt_tree.query(cas9_coords, k=1)
            anchor_coord = cas9_coords[int(np.argmin(dists))]

        acidic = np.array([a["coord"] for a in rt_ca if a["resname"] in {"ASP", "GLU"}], dtype=float)
        if len(acidic) >= 3:
            center = acidic.mean(axis=0)
            dists = np.linalg.norm(acidic - center, axis=1)
            ref_active = acidic[dists <= np.quantile(dists, 0.35)].mean(axis=0)
        elif len(rt_ca) > 0:
            ref_active = np.array([a["coord"] for a in rt_ca], dtype=float).mean(axis=0)
        else:
            ref_active = np.zeros(3)

        nuc_coords = np.array([a["coord"] for a in nuc_atoms], dtype=float) if nuc_atoms else np.zeros((0, 3))
        self._reference_context = {
            "pe_path": pe_path,
            "rt_ca": rt_ca,
            "env_coords": np.array([a["coord"] for a in env_atoms], dtype=float) if env_atoms else np.zeros((0, 3)),
            "nuc_coords": nuc_coords,
            "anchor_coord": anchor_coord,
            "ref_active_site": ref_active,
        }
        return self._reference_context

    @staticmethod
    def _linker_score(candidate_coords: np.ndarray, anchor_coord: np.ndarray) -> float:
        if candidate_coords.size == 0:
            return 0.4
        n_term = candidate_coords[0]
        c_term = candidate_coords[-1]
        best = min(float(np.linalg.norm(n_term - anchor_coord)), float(np.linalg.norm(c_term - anchor_coord)))
        return float(np.exp(-max(best - 35.0, 0.0) / 20.0))

    def compute_scores(self, sequences, structures_dir, handcrafted, external_data):
        ref = self._load_reference_context(external_data)
        results = []
        for _, row in sequences.iterrows():
            rt_name = row["rt_name"]
            hc = handcrafted[handcrafted["rt_name"] == rt_name].iloc[0]
            length = float(row.get("protein_length_aa", 500))
            pdb_path = f"{structures_dir}/{rt_name}.pdb"

            align_res = structural_align(pdb_path, target_atoms=ref.get("rt_ca"))
            fallback_align = float(hc.get("foldseek_TM_MMLV", np.nan))
            if np.isnan(fallback_align):
                fallback_align = float(hc.get("foldseek_best_TM", 0.4))
            align = max(float(align_res.tm_score), float(np.clip(fallback_align, 0, 1)))

            n_clashes, n_contacts = compute_steric_clashes(
                query_coords=align_res.aligned_atoms,
                env_coords=ref.get("env_coords", np.zeros((0, 3))),
                clash_dist=2.5,
                contact_dist=4.5,
            )
            clash_density = n_clashes / max(len(align_res.aligned_atoms), 1)
            clash_score = float(np.exp(-clash_density / 0.08)) if len(align_res.aligned_atoms) else 0.35

            active_site_access = 0.35
            if align_res.active_site_center_transformed is not None:
                ref_active = ref.get("ref_active_site", np.zeros(3))
                as_dist = float(np.linalg.norm(align_res.active_site_center_transformed - ref_active))
                nuc_coords = ref.get("nuc_coords", np.zeros((0, 3)))
                if nuc_coords.size:
                    nuc_tree = cKDTree(nuc_coords)
                    nuc_dist, _ = nuc_tree.query(align_res.active_site_center_transformed, k=1)
                    active_site_access = float(np.exp(-as_dist / 12.0) * np.exp(-max(nuc_dist - 10.0, 0.0) / 18.0))
                else:
                    active_site_access = float(np.exp(-as_dist / 12.0))
            triad = float(hc.get("triad_found_bin", 0.0))
            active_site_access = float(np.clip(0.75 * active_site_access + 0.25 * (0.3 + 0.7 * triad), 0, 1))

            candidate_ca = [a["coord"] for a in extract_atom_records(pdb_path, atom_name="CA") if np.isfinite(a["coord"]).all()]
            if candidate_ca:
                candidate_ca_t = np.array([coord @ align_res.rotation.T + align_res.translation for coord in candidate_ca], dtype=float)
            else:
                candidate_ca_t = np.zeros((0, 3))
            linker_feasibility = self._linker_score(candidate_ca_t, ref.get("anchor_coord", np.zeros(3)))
            size_penalty = self._size_score(length)

            sub = {
                "clash_score": clash_score,
                "alignment_quality": float(np.clip(align, 0, 1)),
                "active_site_access": float(np.clip(active_site_access, 0, 1)),
                "linker_feasibility": float(np.clip(linker_feasibility, 0, 1)),
                "size_penalty": float(np.clip(size_penalty, 0, 1)),
            }
            score = float(np.exp(sum(self.weights[k] * np.log(max(v, 1e-6)) for k, v in sub.items())))
            confidence = float(np.clip(0.3 + 0.35 * sub["alignment_quality"] + 0.35 * sub["active_site_access"], 0, 1))
            if sub["clash_score"] < 0.35:
                failure = "Likely steric incompatibility with Cas9 context"
            elif sub["linker_feasibility"] < 0.4:
                failure = "Poor linker geometry to Cas9 anchor"
            elif sub["active_site_access"] < 0.35:
                failure = "Active site poorly positioned in PE context"
            else:
                failure = ""
            results.append(GateResult(rt_name, self.name, float(np.clip(score, 0, 1)), sub, confidence, failure))
        return results
