from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from retroviral_wall.analysis.gate_diagnostics import diagnose_gates
from retroviral_wall.calibration.evaluation import evaluate_lofo_predictions, print_results
from retroviral_wall.calibration.integrator import GateIntegrator
from retroviral_wall.config import DATA_DIR, OUTPUT_DIR, STRUCTURES_DIR
from retroviral_wall.download_external import download_all_external
from retroviral_wall.gates import (
    CatalyticCompetenceGate,
    FoldabilityGate,
    FusionCompatibilityGate,
    ProcessivityGate,
    SubstrateBindingGate,
)


def select_residual_features(handcrafted: pd.DataFrame, gate_scores: pd.DataFrame, max_features: int = 5) -> pd.DataFrame | None:
    hc = handcrafted.set_index("rt_name")
    candidates = []
    gate_cols = [c for c in gate_scores.columns if c.endswith("_score")]
    for col in hc.columns:
        vals = pd.to_numeric(hc[col], errors="coerce")
        if vals.isna().mean() > 0.25:
            continue
        vals = vals.fillna(vals.median())
        if vals.std() < 1e-6:
            continue
        max_corr = max(abs(vals.corr(gate_scores[g])) for g in gate_cols)
        if pd.isna(max_corr) or max_corr > 0.75:
            continue
        candidates.append((col, vals.var()))
    if not candidates:
        return None
    top = [c for c, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:max_features]]
    return hc[top].loc[gate_scores.index]


def run_pipeline() -> dict:
    print("=" * 70)
    print("Mechanistic Gate Pipeline")
    print(datetime.now().isoformat())
    print("=" * 70)

    sequences = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    handcrafted = pd.read_csv(DATA_DIR / "handcrafted_features.csv")
    external = download_all_external()

    gates = [
        FoldabilityGate(),
        FusionCompatibilityGate(),
        SubstrateBindingGate(),
        CatalyticCompetenceGate(),
        ProcessivityGate(),
    ]

    gate_scores = pd.DataFrame(index=sequences["rt_name"])
    for gate in gates:
        results = gate.compute_scores(sequences, str(STRUCTURES_DIR), handcrafted, external)
        gdf = gate.score_matrix(results).set_index("rt_name")
        gate_scores = gate_scores.join(gdf)

    gate_scores.to_csv(OUTPUT_DIR / "gate_scores" / "all_gate_scores.csv")

    labels = sequences.set_index("rt_name")["active"]
    families = sequences.set_index("rt_name")["rt_family"]

    diagnostics = diagnose_gates(gate_scores, labels, families)
    with open(OUTPUT_DIR / "gate_scores" / "gate_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    best_results = None
    best_preds = None
    best_strategy = None
    best_f1 = -1.0

    for strategy in ["multiplicative", "bayesian_lr", "bart"]:
        integrator = GateIntegrator(strategy=strategy)
        preds = integrator.fit_predict_lofo(gate_scores, labels, families)
        results = evaluate_lofo_predictions(preds, sequences)
        print(f"\nStrategy: {strategy}")
        print_results(results)
        f1 = results["primary_metric"]["LOFO_macro_F1_4_folds"]
        if f1 > best_f1:
            best_f1 = f1
            best_strategy = strategy
            best_results = results
            best_preds = preds

    residuals = select_residual_features(handcrafted, gate_scores)
    if residuals is not None and best_strategy is not None:
        integrator = GateIntegrator(strategy=best_strategy)
        preds = integrator.fit_predict_lofo(gate_scores, labels, families, handcrafted_residuals=residuals)
        res = evaluate_lofo_predictions(preds, sequences)
        if res["primary_metric"]["LOFO_macro_F1_4_folds"] > best_f1:
            best_results = res
            best_preds = preds
            best_f1 = res["primary_metric"]["LOFO_macro_F1_4_folds"]

    assert best_preds is not None and best_results is not None
    out_pred = OUTPUT_DIR / "predictions" / "submission.csv"
    best_preds[["rt_name", "predicted_active", "predicted_score"]].to_csv(out_pred, index=False)
    with open(OUTPUT_DIR / "predictions" / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)

    print(f"\nSaved predictions to {out_pred}")
    print(f"Best LOFO macro-F1: {best_f1:.3f}")
    return best_results


if __name__ == "__main__":
    run_pipeline()
