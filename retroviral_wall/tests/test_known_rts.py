from __future__ import annotations

import pandas as pd

from retroviral_wall.config import DATA_DIR, STRUCTURES_DIR
from retroviral_wall.download_external import download_all_external
from retroviral_wall.gates import (
    CatalyticCompetenceGate,
    FoldabilityGate,
    FusionCompatibilityGate,
    ProcessivityGate,
    SubstrateBindingGate,
)


def _compute_scores_for(rt_name: str) -> dict[str, float]:
    seq = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    hc = pd.read_csv(DATA_DIR / "handcrafted_features.csv")
    ext = download_all_external()
    gates = [FoldabilityGate(), FusionCompatibilityGate(), SubstrateBindingGate(), CatalyticCompetenceGate(), ProcessivityGate()]

    out = {}
    for gate in gates:
        res = gate.compute_scores(seq, str(STRUCTURES_DIR), hc, ext)
        s = {r.rt_name: r.score for r in res}
        out[f"{gate.name}_score"] = s[rt_name]
    return out


def test_scores_bounded_for_mmlv():
    scores = _compute_scores_for("MMLV-RT")
    assert all(0 <= v <= 1 for v in scores.values())


def test_inactive_has_one_weak_gate():
    scores = _compute_scores_for("A.platensis-Cas1-RT")
    assert min(scores.values()) < 0.5
