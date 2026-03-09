from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def extract_atom_records(
    pdb_path: str | Path,
    chains: set[str] | None = None,
    atom_name: str | None = None,
) -> list[dict]:
    atoms: list[dict] = []
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                chain = line[21].strip() or "_"
                if chains is not None and chain not in chains:
                    continue
                atom = line[12:16].strip()
                if atom_name is not None and atom != atom_name:
                    continue
                try:
                    coord = np.array(
                        [
                            float(line[30:38].strip()),
                            float(line[38:46].strip()),
                            float(line[46:54].strip()),
                        ],
                        dtype=float,
                    )
                except ValueError:
                    continue
                atoms.append(
                    {
                        "atom": atom,
                        "resname": line[17:20].strip(),
                        "chain": chain,
                        "resseq": line[22:26].strip(),
                        "icode": line[26].strip() or "_",
                        "coord": coord,
                    }
                )
    except FileNotFoundError:
        return []
    return atoms


def extract_ca_residues(pdb_path: str | Path) -> list[dict]:
    residues: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for atom in extract_atom_records(pdb_path, atom_name="CA"):
        key = (atom["chain"], atom["resseq"], atom["icode"])
        if key in seen:
            continue
        seen.add(key)
        residues.append(
            {
                "resname": atom["resname"],
                "chain": atom["chain"],
                "resseq": atom["resseq"],
                "icode": atom["icode"],
                "coord": atom["coord"],
            }
        )
    return residues


def extract_ca_plddt(pdb_path: str | Path) -> list[float]:
    values: list[float] = []
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                atom = line[12:16].strip()
                if atom != "CA":
                    continue
                try:
                    bfactor = float(line[60:66].strip())
                    values.append(bfactor)
                except ValueError:
                    continue
    except FileNotFoundError:
        return []
    return values


def mean_and_core_plddt(pdb_path: str | Path, trim: int = 20) -> tuple[float, float]:
    plddt = extract_ca_plddt(pdb_path)
    if not plddt:
        return 0.5, 0.5
    arr = np.array(plddt, dtype=float) / 100.0
    if len(arr) > 2 * trim:
        core = arr[trim:-trim]
    else:
        core = arr
    return float(arr.mean()), float(core.mean())


def foldability_structure_features(pdb_path: str | Path) -> dict[str, float]:
    residues = [r for r in extract_ca_residues(pdb_path) if np.isfinite(r["coord"]).all()]
    plddt = extract_ca_plddt(pdb_path)
    if not residues or not plddt:
        return {
            "plddt_mean": 0.5,
            "plddt_core": 0.5,
            "high_conf_frac": 0.5,
            "low_conf_frac": 0.5,
            "longest_conf_segment": 0.5,
            "contact_density": 0.5,
        }

    arr = np.array(plddt[: len(residues)], dtype=float) / 100.0
    if len(arr) > 40:
        core = arr[20:-20]
    else:
        core = arr

    high_conf_frac = float((arr >= 0.70).mean())
    low_conf_frac = float((arr < 0.50).mean())

    longest = 0
    current = 0
    for val in arr:
        if val >= 0.70:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    longest_conf_segment = float(longest / max(len(arr), 1))

    coords = np.array([r["coord"] for r in residues], dtype=float)
    if len(coords) >= 3:
        tree = cKDTree(coords)
        contacts = np.array([len(tree.query_ball_point(coord, 8.0)) - 1 for coord in coords], dtype=float)
        # More contacts per residue suggests a more compact fold.
        contact_density = float(1.0 / (1.0 + np.exp(-(contacts.mean() - 8.0) / 2.0)))
    else:
        contact_density = 0.5

    return {
        "plddt_mean": float(arr.mean()),
        "plddt_core": float(core.mean()),
        "high_conf_frac": high_conf_frac,
        "low_conf_frac": low_conf_frac,
        "longest_conf_segment": longest_conf_segment,
        "contact_density": contact_density,
    }
