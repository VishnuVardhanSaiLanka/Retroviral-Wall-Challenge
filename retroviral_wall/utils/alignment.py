from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retroviral_wall.utils.pdb_utils import extract_atom_records, extract_ca_residues


@dataclass
class AlignmentResult:
    tm_score: float
    rmsd: float
    rotation: np.ndarray
    translation: np.ndarray
    aligned_atoms: np.ndarray
    active_site_center_transformed: np.ndarray | None


def structural_align(mobile_pdb: str, target_pdb: str | None = None, target_atoms: list | None = None, method: str = "proxy") -> AlignmentResult:
    def _sample_coords(coords: np.ndarray, n_points: int) -> np.ndarray:
        if len(coords) == 0:
            return np.zeros((0, 3))
        if len(coords) <= n_points:
            return coords
        idx = np.linspace(0, len(coords) - 1, n_points).astype(int)
        return coords[idx]

    mobile_res = [r for r in extract_ca_residues(mobile_pdb) if np.isfinite(r["coord"]).all()]
    if target_pdb is not None:
        target_res = [r for r in extract_ca_residues(target_pdb) if np.isfinite(r["coord"]).all()]
    else:
        target_res = [a for a in (target_atoms or []) if np.isfinite(a["coord"]).all()]

    if len(mobile_res) < 10 or len(target_res) < 10:
        return AlignmentResult(
            tm_score=0.2,
            rmsd=8.0,
            rotation=np.eye(3),
            translation=np.zeros(3),
            aligned_atoms=np.zeros((0, 3)),
            active_site_center_transformed=None,
        )

    mob = np.array([r["coord"] for r in mobile_res], dtype=float)
    tgt = np.array([r["coord"] for r in target_res], dtype=float)
    n = int(min(80, len(mob), len(tgt)))
    mob_s = _sample_coords(mob, n)
    tgt_s = _sample_coords(tgt, n)

    mob_cent = mob_s.mean(axis=0)
    tgt_cent = tgt_s.mean(axis=0)
    mob0 = mob_s - mob_cent
    tgt0 = tgt_s - tgt_cent
    h = mob0.T @ tgt0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = tgt_cent - mob_cent @ r.T

    all_mobile_atoms = [a for a in extract_atom_records(mobile_pdb) if np.isfinite(a["coord"]).all()]
    aligned_atoms = np.array([a["coord"] @ r.T + t for a in all_mobile_atoms], dtype=float) if all_mobile_atoms else np.zeros((0, 3))
    aligned_ca = mob_s @ r.T + t
    rmsd = float(np.sqrt(np.mean(np.sum((aligned_ca - tgt_s) ** 2, axis=1))))
    scale = max(len(mob), len(tgt), 1)
    tm_score = float(np.clip(1.0 / (1.0 + rmsd / 6.0) * min(len(mob), len(tgt)) / scale, 0.0, 1.0))

    acidic = [r0["coord"] for r0 in mobile_res if r0["resname"] in {"ASP", "GLU"}]
    active_site = None
    if acidic:
        acidic_arr = np.array(acidic, dtype=float)
        center = acidic_arr.mean(axis=0)
        dists = np.linalg.norm(acidic_arr - center, axis=1)
        core = acidic_arr[dists <= np.quantile(dists, 0.35)] if len(acidic_arr) >= 3 else acidic_arr
        active_site = core.mean(axis=0) @ r.T + t

    return AlignmentResult(
        tm_score=tm_score,
        rmsd=rmsd,
        rotation=r,
        translation=t,
        aligned_atoms=aligned_atoms,
        active_site_center_transformed=active_site,
    )
