from __future__ import annotations

from pathlib import Path

import numpy as np


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
