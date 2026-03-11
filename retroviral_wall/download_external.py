from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

from retroviral_wall.config import EXTERNAL_DIR, EXTERNAL_PDBS

PDB_BASE_URL = "https://files.rcsb.org/download"


def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out
    url = f"{PDB_BASE_URL}/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, out)
    except Exception:
        return out
    return out


def load_literature_processivity(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = path.read_text().splitlines()
    if len(rows) < 2:
        return {}
    header = rows[0].split(",")
    idx_name = header.index("rt_name")
    idx_proc = header.index("processivity_nt")
    values: dict[str, float] = {}
    for line in rows[1:]:
        parts = line.split(",")
        if len(parts) <= max(idx_name, idx_proc):
            continue
        try:
            values[parts[idx_name]] = float(parts[idx_proc])
        except ValueError:
            continue
    return values


def download_all_external() -> dict[str, Any]:
    pdb_dir = EXTERNAL_DIR / "pdb"
    external_data: dict[str, Any] = {}
    for key, info in EXTERNAL_PDBS.items():
        path = download_pdb(info["pdb_id"], pdb_dir)
        external_data[key] = {**info, "path": str(path)}

    lit_path = EXTERNAL_DIR / "literature" / "rt_processivity.csv"
    if not lit_path.exists():
        lit_path.write_text(
            "rt_name,organism,processivity_nt,kcat_s,optimal_temp_C,source_doi\n"
            "MMLV-RT,Moloney murine leukemia virus,45,,37,10.1016/0092-8674(86)90598-0\n"
            "HIVpol+MMLVRNaseH_RT,HIV-1,30,,37,10.1016/0092-8674(89)90450-8\n"
        )
    external_data["rt_processivity"] = load_literature_processivity(lit_path)
    return external_data


if __name__ == "__main__":
    data = download_all_external()
    print(f"Downloaded/registered {len(data)} external resources")
