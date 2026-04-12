from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from retroviral_wall.utils.geometry import sigmoid
from retroviral_wall.utils.pdb_utils import extract_ca_plddt, extract_ca_residues

POSITIVE_RESIDUES = {"LYS": 1.0, "ARG": 1.0, "HIS": 0.5}
POLAR_RESIDUES = {"ASN", "GLN", "SER", "THR", "TYR", "LYS", "ARG", "HIS"}
ACIDIC_RESIDUES = {"ASP", "GLU"}
HYDROPHOBIC_RESIDUES = {"VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "ALA"}


def _residues_and_tree(pdb_path: str | Path) -> tuple[list[dict], np.ndarray, cKDTree] | tuple[None, None, None]:
    residues = [r for r in extract_ca_residues(pdb_path) if np.isfinite(r["coord"]).all()]
    if len(residues) < 20:
        return None, None, None
    coords = np.array([r["coord"] for r in residues], dtype=float)
    return residues, coords, cKDTree(coords)


def _surface_exposure(coords: np.ndarray, tree: cKDTree) -> np.ndarray:
    contacts = np.array([len(tree.query_ball_point(coord, 10.0)) - 1 for coord in coords], dtype=float)
    return 1.0 / (1.0 + np.exp((contacts - 14.0) / 2.5))


def find_acidic_cluster_center(pdb_path: str | Path) -> np.ndarray | None:
    residues, coords, tree = _residues_and_tree(pdb_path)
    if residues is None:
        return None
    acidic_idx = [i for i, r in enumerate(residues) if r["resname"] in ACIDIC_RESIDUES]
    if len(acidic_idx) < 2:
        return None

    acidic_set = set(acidic_idx)
    visited: set[int] = set()
    best_center = None
    best_score = -1.0
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
        if len(comp) < 2:
            continue
        comp_coords = coords[comp]
        center = comp_coords.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum((comp_coords - center) ** 2, axis=1))))
        score = len(comp) - 0.5 * rms
        if score > best_score:
            best_score = score
            best_center = center
    return best_center


def substrate_path_features(pdb_path: str | Path) -> dict[str, float]:
    residues, coords, tree = _residues_and_tree(pdb_path)
    if residues is None:
        return {
            "positive_continuity": 0.35,
            "path_openness": 0.5,
            "path_extension": 0.5,
            "catalytic_accessibility": 0.35,
            "steric_crowding_penalty": 0.5,
            "hydrophobic_interrupt": 0.5,
            "track_to_thumb": 0.35,
        }

    exposure = _surface_exposure(coords, tree)
    positive = np.array([POSITIVE_RESIDUES.get(r["resname"], 0.0) for r in residues], dtype=float)
    polar = np.array([r["resname"] in POLAR_RESIDUES for r in residues], dtype=bool)
    hydrophobic = np.array([r["resname"] in HYDROPHOBIC_RESIDUES for r in residues], dtype=bool)
    acidic_center = find_acidic_cluster_center(pdb_path)
    if acidic_center is None:
        acidic_center = coords.mean(axis=0)

    eligible = (exposure >= 0.42) & ((positive > 0) | polar)
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) == 0:
        return {
            "positive_continuity": 0.35,
            "path_openness": 0.5,
            "path_extension": 0.5,
            "catalytic_accessibility": 0.35,
            "steric_crowding_penalty": 0.5,
            "hydrophobic_interrupt": 0.5,
            "track_to_thumb": 0.35,
        }

    visited: set[int] = set()
    best = None
    best_rank = -1.0
    for root in eligible_idx:
        if root in visited:
            continue
        queue = [root]
        component: list[int] = []
        visited.add(root)
        while queue:
            idx = queue.pop()
            component.append(idx)
            for nbr in tree.query_ball_point(coords[idx], 8.5):
                if nbr in visited or not eligible[nbr]:
                    continue
                visited.add(nbr)
                queue.append(nbr)
        comp = np.array(component, dtype=int)
        if len(comp) < 3:
            continue
        comp_coords = coords[comp]
        centroid = comp_coords.mean(axis=0)
        centered = comp_coords - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        proj = centered @ axis
        extension = float(proj.max() - proj.min())
        open_score = float(np.mean(exposure[comp]))
        pos_density = float(positive[comp].sum() / max(len(comp), 1))
        hydro_interrupt = float(hydrophobic[comp].mean())
        catalytic_dist = float(np.linalg.norm(centroid - acidic_center))
        catalytic_accessibility = float(np.exp(-((catalytic_dist - 13.0) ** 2) / (2 * 6.5**2)))
        crowding = np.array([len(tree.query_ball_point(coord, 6.0)) - 1 for coord in comp_coords], dtype=float)
        crowding_penalty = float(sigmoid((crowding.mean() - 8.0) / 2.0))
        track_to_thumb = float(sigmoid((np.percentile(proj, 85) - 6.0) / 3.0))

        positive_continuity = float(sigmoid((pos_density - 0.22) * 7.0))
        path_extension = float(sigmoid((extension - 16.0) / 4.0))
        hydrophobic_interrupt = float(1.0 - sigmoid((hydro_interrupt - 0.28) * 8.0))
        steric_crowding_penalty = float(1.0 - crowding_penalty)

        rank = (
            0.24 * positive_continuity
            + 0.20 * open_score
            + 0.18 * path_extension
            + 0.18 * catalytic_accessibility
            + 0.10 * steric_crowding_penalty
            + 0.10 * hydrophobic_interrupt
        )
        if rank > best_rank:
            best_rank = rank
            best = {
                "positive_continuity": positive_continuity,
                "path_openness": open_score,
                "path_extension": path_extension,
                "catalytic_accessibility": catalytic_accessibility,
                "steric_crowding_penalty": steric_crowding_penalty,
                "hydrophobic_interrupt": hydrophobic_interrupt,
                "track_to_thumb": track_to_thumb,
            }
    return best if best is not None else {
        "positive_continuity": 0.35,
        "path_openness": 0.5,
        "path_extension": 0.5,
        "catalytic_accessibility": 0.35,
        "steric_crowding_penalty": 0.5,
        "hydrophobic_interrupt": 0.5,
        "track_to_thumb": 0.35,
    }


def fusion_context_features(pdb_path: str | Path) -> dict[str, float]:
    residues, coords, tree = _residues_and_tree(pdb_path)
    if residues is None:
        return {
            "terminal_accessibility": 0.5,
            "core_compactness": 0.5,
            "active_site_exposure": 0.5,
            "terminal_disorder_proxy": 0.5,
        }

    exposure = _surface_exposure(coords, tree)
    plddt = np.array(extract_ca_plddt(pdb_path)[: len(residues)], dtype=float)
    if len(plddt) != len(residues):
        plddt = np.full(len(residues), 70.0, dtype=float)
    plddt = plddt / 100.0

    term_n = exposure[: min(20, len(exposure))]
    term_c = exposure[-min(20, len(exposure)) :]
    terminal_accessibility = float(max(term_n.mean(), term_c.mean()))
    core_mask = plddt >= 0.7
    if core_mask.sum() >= 10:
        core_coords = coords[core_mask]
        core_centroid = core_coords.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum((core_coords - core_centroid) ** 2, axis=1))))
        core_compactness = float(np.exp(-(max(rms - 20.0, 0.0) ** 2) / (2 * 8.0**2)))
    else:
        core_compactness = 0.5

    acidic_center = find_acidic_cluster_center(pdb_path)
    if acidic_center is None:
        active_site_exposure = 0.5
    else:
        idx = int(np.argmin(np.linalg.norm(coords - acidic_center, axis=1)))
        active_site_exposure = float(exposure[idx])

    term_plddt = np.concatenate([plddt[: min(25, len(plddt))], plddt[-min(25, len(plddt)) :]])
    terminal_disorder_proxy = float(1.0 - term_plddt.mean())

    return {
        "terminal_accessibility": terminal_accessibility,
        "core_compactness": core_compactness,
        "active_site_exposure": active_site_exposure,
        "terminal_disorder_proxy": terminal_disorder_proxy,
    }


def catalytic_to_thumb_corridor_features(pdb_path: str | Path) -> dict[str, float]:
    residues, coords, tree = _residues_and_tree(pdb_path)
    if residues is None:
        return {
            "corridor_basic_density": 0.35,
            "corridor_openness": 0.5,
            "corridor_bottleneck": 0.35,
            "corridor_hydrophobic_tolerance": 0.5,
            "corridor_order": 0.5,
            "corridor_catalytic_shell_basicity": 0.35,
        }

    path = substrate_path_features(pdb_path)
    exposure = _surface_exposure(coords, tree)
    positive = np.array([POSITIVE_RESIDUES.get(r["resname"], 0.0) for r in residues], dtype=float)
    hydrophobic = np.array([r["resname"] in HYDROPHOBIC_RESIDUES for r in residues], dtype=bool)
    acidic_center = find_acidic_cluster_center(pdb_path)
    if acidic_center is None:
        acidic_center = coords.mean(axis=0)

    # Find a target point that follows the best exposed positive track away from the catalytic center.
    dists_from_active = np.linalg.norm(coords - acidic_center, axis=1)
    candidate_mask = (dists_from_active >= 8.0) & (exposure >= 0.4)
    if candidate_mask.sum() == 0:
        target = coords[np.argmax(exposure)]
    else:
        candidate_scores = (
            0.45 * exposure[candidate_mask]
            + 0.35 * np.clip(positive[candidate_mask], 0.0, 1.0)
            + 0.20 * path["track_to_thumb"]
        )
        target = coords[np.where(candidate_mask)[0][int(np.argmax(candidate_scores))]]

    axis = target - acidic_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return {
            "corridor_basic_density": 0.35,
            "corridor_openness": 0.5,
            "corridor_bottleneck": 0.35,
            "corridor_hydrophobic_tolerance": 0.5,
            "corridor_order": 0.5,
            "corridor_catalytic_shell_basicity": 0.35,
        }
    axis = axis / axis_norm

    rel = coords - acidic_center
    longitudinal = rel @ axis
    radial = np.linalg.norm(rel - np.outer(longitudinal, axis), axis=1)
    corridor_mask = (longitudinal >= -2.0) & (longitudinal <= axis_norm + 3.0) & (radial <= 7.5)

    if corridor_mask.sum() < 4:
        corridor_mask = (longitudinal >= -2.0) & (longitudinal <= axis_norm + 4.0) & (radial <= 10.0)

    corridor_idx = np.where(corridor_mask)[0]
    if len(corridor_idx) == 0:
        return {
            "corridor_basic_density": 0.35,
            "corridor_openness": 0.5,
            "corridor_bottleneck": 0.35,
            "corridor_hydrophobic_tolerance": 0.5,
            "corridor_order": 0.5,
            "corridor_catalytic_shell_basicity": 0.35,
        }

    corridor_coords = coords[corridor_idx]
    corridor_exposure = exposure[corridor_idx]
    corridor_positive = positive[corridor_idx]
    corridor_hydrophobic = hydrophobic[corridor_idx]
    plddt = np.array(extract_ca_plddt(pdb_path)[: len(residues)], dtype=float)
    if len(plddt) != len(residues):
        plddt = np.full(len(residues), 70.0, dtype=float)
    plddt = plddt / 100.0
    corridor_order = float(plddt[corridor_idx].mean())

    corridor_basic_density = float(sigmoid((corridor_positive.mean() - 0.18) * 8.0))
    corridor_openness = float(corridor_exposure.mean())
    hydrophobic_fraction = float(corridor_hydrophobic.mean())
    corridor_hydrophobic_tolerance = float(1.0 - sigmoid((hydrophobic_fraction - 0.30) * 8.0))

    # Evaluate bottlenecks by binning the corridor and asking whether any segment collapses in openness.
    bins = np.linspace(max(-1.0, longitudinal[corridor_idx].min()), longitudinal[corridor_idx].max() + 1.0, 5)
    bin_scores = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = corridor_mask & (longitudinal >= left) & (longitudinal < right)
        if mask.sum() == 0:
            continue
        local_score = 0.6 * float(exposure[mask].mean()) + 0.4 * float(sigmoid((positive[mask].mean() - 0.15) * 8.0))
        bin_scores.append(local_score)
    corridor_bottleneck = float(min(bin_scores)) if bin_scores else 0.35

    shell_mask = (dists_from_active >= 5.0) & (dists_from_active <= 12.0)
    corridor_catalytic_shell_basicity = float(sigmoid((positive[shell_mask].mean() - 0.16) * 8.0)) if shell_mask.sum() else 0.35

    return {
        "corridor_basic_density": corridor_basic_density,
        "corridor_openness": corridor_openness,
        "corridor_bottleneck": corridor_bottleneck,
        "corridor_hydrophobic_tolerance": corridor_hydrophobic_tolerance,
        "corridor_order": corridor_order,
        "corridor_catalytic_shell_basicity": corridor_catalytic_shell_basicity,
    }


def literature_pe_features(pdb_path: str | Path) -> dict[str, float]:
    residues, coords, tree = _residues_and_tree(pdb_path)
    if residues is None:
        return {
            "priming_shell_readiness": 0.35,
            "template_grip_positive_groove": 0.35,
        }

    acidic_center = find_acidic_cluster_center(pdb_path)
    if acidic_center is None:
        acidic_center = coords.mean(axis=0)
    dists = np.linalg.norm(coords - acidic_center, axis=1)
    exposure = _surface_exposure(coords, tree)
    plddt = np.array(extract_ca_plddt(pdb_path)[: len(residues)], dtype=float)
    if len(plddt) != len(residues):
        plddt = np.full(len(residues), 70.0, dtype=float)
    plddt = plddt / 100.0
    positive = np.array([POSITIVE_RESIDUES.get(r["resname"], 0.0) for r in residues], dtype=float)
    hydrophobic = np.array([r["resname"] in HYDROPHOBIC_RESIDUES for r in residues], dtype=bool)

    # Literature-inspired feature 1:
    # NAR 2023 suggests RT priming/initiation depends on local active-site architecture.
    # We score an ordered, moderately exposed basic shell around the catalytic cluster.
    priming_shell = (dists >= 4.0) & (dists <= 10.0)
    if priming_shell.sum():
        shell_basic = float(positive[priming_shell].mean())
        shell_exposure = float(exposure[priming_shell].mean())
        shell_order = float(plddt[priming_shell].mean())
        shell_hydrophobic = float(hydrophobic[priming_shell].mean())
        priming_shell_readiness = float(
            np.clip(
                0.40 * sigmoid((shell_basic - 0.12) * 9.0)
                + 0.25 * shell_exposure
                + 0.25 * shell_order
                + 0.10 * (1.0 - sigmoid((shell_hydrophobic - 0.32) * 8.0)),
                0,
                1,
            )
        )
    else:
        priming_shell_readiness = 0.35

    # Literature-inspired feature 2:
    # Nature 2024 shows the PBS–NTS / RTT–DNA heteroduplex is engaged through a positively charged groove.
    # We score a positive, exposed, ordered groove extending outward from the catalytic center.
    corridor = catalytic_to_thumb_corridor_features(pdb_path)
    target_idx = int(np.argmax(exposure + 0.5 * np.clip(positive, 0.0, 1.0)))
    target = coords[target_idx]
    axis = target - acidic_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-6:
        axis = axis / axis_norm
        rel = coords - acidic_center
        longitudinal = rel @ axis
        radial = np.linalg.norm(rel - np.outer(longitudinal, axis), axis=1)
        groove_mask = (longitudinal >= 0.0) & (longitudinal <= max(axis_norm, 12.0)) & (radial <= 8.0)
    else:
        groove_mask = dists <= 12.0

    if groove_mask.sum():
        groove_positive = float(positive[groove_mask].mean())
        groove_exposure = float(exposure[groove_mask].mean())
        groove_order = float(plddt[groove_mask].mean())
        groove_hydrophobic = float(hydrophobic[groove_mask].mean())
        template_grip_positive_groove = float(
            np.clip(
                0.35 * sigmoid((groove_positive - 0.16) * 8.0)
                + 0.20 * groove_exposure
                + 0.20 * groove_order
                + 0.15 * corridor["corridor_bottleneck"]
                + 0.10 * (1.0 - sigmoid((groove_hydrophobic - 0.28) * 8.0)),
                0,
                1,
            )
        )
    else:
        template_grip_positive_groove = 0.35

    return {
        "priming_shell_readiness": priming_shell_readiness,
        "template_grip_positive_groove": template_grip_positive_groove,
    }
