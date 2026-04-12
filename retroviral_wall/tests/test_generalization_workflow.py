from __future__ import annotations

from pathlib import Path

from experiments.generalization_workflow import load_bundle, select_low_leakage_features
from retroviral_wall.analysis.gate_audit import audit_gates


def test_gate_audit_has_expected_sections():
    full, gate_scores = load_bundle(
        Path("data"),
        Path("retroviral_wall/outputs/gate_scores/all_gate_scores.csv"),
    )
    report = audit_gates(
        gate_scores.set_index("rt_name"),
        full.set_index("rt_name")["active"],
        full.set_index("rt_name")["rt_family"],
    )

    assert "per_gate" in report
    assert "pairwise_score_correlation" in report
    assert "foldability" in report["per_gate"]


def test_low_leakage_feature_selection_returns_candidates():
    full, _ = load_bundle(
        Path("data"),
        Path("retroviral_wall/outputs/gate_scores/all_gate_scores.csv"),
    )
    handcrafted_columns = ["rt_name"] + [
        c for c in full.columns if c not in {"rt_name", "active", "rt_family", "sequence"} and not c.startswith(("foldability_", "fusion_", "substrate_", "catalytic_", "processivity_"))
    ]
    selected, report = select_low_leakage_features(full, handcrafted_columns)

    assert len(report) > 0
    assert len(selected) > 0
