from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = PACKAGE_ROOT / "external"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
STRUCTURES_DIR = DATA_DIR / "structures"

for path in [
    OUTPUT_DIR,
    OUTPUT_DIR / "gate_scores",
    OUTPUT_DIR / "predictions",
    OUTPUT_DIR / "figures",
    EXTERNAL_DIR / "pdb",
    EXTERNAL_DIR / "literature",
]:
    path.mkdir(parents=True, exist_ok=True)

EXTERNAL_PDBS = {
    "pe2_complex": {"pdb_id": "8W8H", "description": "Prime editor cryo-EM structure"},
    "mmlv_rt_substrate": {"pdb_id": "7UVO", "description": "MMLV-RT with template-primer"},
    "hiv1_rt_substrate": {"pdb_id": "1RTD", "description": "HIV-1 RT with RNA:DNA hybrid"},
    "cas9_alone": {"pdb_id": "4ZT0", "description": "SpCas9 crystal structure"},
}

GATE_PARAMS = {
    "clash_distance_A": 2.0,
    "contact_distance_A": 4.0,
    "min_processive_length_aa": 200,
    "max_size_no_penalty_aa": 800,
}
