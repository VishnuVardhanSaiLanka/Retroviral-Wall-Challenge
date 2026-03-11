from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

INFORMATIVE_FAMILIES = ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]


def evaluate_lofo_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    merged = predictions.merge(ground_truth[["rt_name", "active", "pe_efficiency_pct", "rt_family"]], on="rt_name", how="inner")
    y_true = merged["active"].to_numpy()
    y_pred = merged["predicted_active"].to_numpy()
    y_score = merged["predicted_score"].to_numpy()

    overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    overall_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fam_results = {}
    informative = []
    for fam in merged["held_out_family"].unique():
        mask = merged["held_out_family"] == fam
        yt, yp, ys = merged.loc[mask, "active"].to_numpy(), merged.loc[mask, "predicted_active"].to_numpy(), merged.loc[mask, "predicted_score"].to_numpy()
        fam_f1 = f1_score(yt, yp, zero_division=0) if len(np.unique(yt)) > 1 else None
        fam_results[fam] = {
            "n": int(len(yt)),
            "n_active": int(yt.sum()),
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
            "tn": int(((yt == 0) & (yp == 0)).sum()),
            "f1": fam_f1,
        }
        if fam in INFORMATIVE_FAMILIES and fam_f1 is not None:
            informative.append(fam_f1)

    active_mask = merged["active"] == 1
    if active_mask.sum() > 2:
        sp, _ = spearmanr(merged.loc[active_mask, "pe_efficiency_pct"], merged.loc[active_mask, "predicted_score"])
        kd, _ = kendalltau(merged.loc[active_mask, "pe_efficiency_pct"], merged.loc[active_mask, "predicted_score"])
    else:
        sp, kd = None, None

    return {
        "primary_metric": {"LOFO_macro_F1_4_folds": float(np.mean(informative)) if informative else 0.0},
        "secondary_metrics": {
            "retroviral_TP_of_12": fam_results.get("Retroviral", {}).get("tp", 0),
            "overall_F1": float(overall_f1),
            "overall_AUC": float(overall_auc) if overall_auc is not None else None,
            "total_TP": int(tp),
            "total_FP": int(fp),
            "total_FN": int(fn),
            "total_TN": int(tn),
        },
        "ranking_quality": {
            "spearman_rho": float(sp) if sp is not None and not np.isnan(sp) else None,
            "kendall_tau": float(kd) if kd is not None and not np.isnan(kd) else None,
        },
        "per_family": fam_results,
    }


def print_results(results: dict) -> None:
    print("=" * 65)
    print("  LOFO EVALUATION RESULTS")
    print("=" * 65)
    print(f"Primary LOFO Macro-F1 (4 informative folds): {results['primary_metric']['LOFO_macro_F1_4_folds']:.3f}")
    sm = results["secondary_metrics"]
    print(f"Retroviral TP: {sm['retroviral_TP_of_12']}/12")
    print(f"Overall F1: {sm['overall_F1']:.3f}")
    if sm["overall_AUC"] is not None:
        print(f"Overall AUC: {sm['overall_AUC']:.3f}")
    print(f"Confusion TP={sm['total_TP']} FP={sm['total_FP']} FN={sm['total_FN']} TN={sm['total_TN']}")
