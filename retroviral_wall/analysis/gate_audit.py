from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

INFORMATIVE_FAMILIES = ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    return float(f1_score(y_true, y_pred, zero_division=0))


def audit_gates(gate_scores: pd.DataFrame, labels: pd.Series, families: pd.Series) -> dict:
    score_cols = [c for c in gate_scores.columns if c.endswith("_score")]
    conf_cols = [c for c in gate_scores.columns if c.endswith("_confidence")]
    y = labels.loc[gate_scores.index].astype(int).to_numpy()
    fam = families.loc[gate_scores.index].to_numpy()

    per_gate: dict[str, dict] = {}
    weakest_gate_counts: dict[str, int] = {}

    for col in score_cols:
        s = gate_scores[col].astype(float).fillna(0.5)
        preds = (s >= 0.5).astype(int)
        gate_name = col.replace("_score", "")

        fam_groups = [s.loc[fam == f].to_numpy() for f in np.unique(fam)]
        fam_groups = [g for g in fam_groups if len(g) > 1]
        if fam_groups:
            ss_between = sum(len(g) * (float(g.mean()) - float(s.mean())) ** 2 for g in fam_groups)
            ss_total = float(np.sum((s.to_numpy() - float(s.mean())) ** 2))
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
        else:
            eta_sq = 0.0

        per_family = {}
        informative_f1 = []
        for family in sorted(np.unique(fam)):
            mask = fam == family
            fam_auc = _safe_auc(y[mask], s.loc[mask].to_numpy())
            fam_f1 = _safe_f1(y[mask], preds.loc[mask].to_numpy())
            per_family[family] = {
                "n": int(mask.sum()),
                "active_n": int(y[mask].sum()),
                "auc": fam_auc,
                "f1_at_0.5": fam_f1,
                "mean_score": float(s.loc[mask].mean()),
            }
            if family in INFORMATIVE_FAMILIES and fam_f1 is not None:
                informative_f1.append(fam_f1)

        conf_col = f"{gate_name}_confidence"
        confidence = gate_scores[conf_col].astype(float).fillna(0.5) if conf_col in conf_cols else None
        fail_col = f"{gate_name}_failure_reason"
        failure_rate = None
        if fail_col in gate_scores.columns:
            failure_rate = float((gate_scores[fail_col].fillna("").astype(str) != "").mean())

        per_gate[gate_name] = {
            "overall_auc": _safe_auc(y, s.to_numpy()),
            "overall_f1_at_0.5": _safe_f1(y, preds.to_numpy()),
            "informative_macro_f1_at_0.5": float(np.mean(informative_f1)) if informative_f1 else None,
            "family_eta_squared": eta_sq,
            "active_mean": float(s.loc[y == 1].mean()),
            "inactive_mean": float(s.loc[y == 0].mean()),
            "confidence_mean": float(confidence.mean()) if confidence is not None else None,
            "confidence_active_mean": float(confidence.loc[y == 1].mean()) if confidence is not None else None,
            "confidence_inactive_mean": float(confidence.loc[y == 0].mean()) if confidence is not None else None,
            "failure_rate": failure_rate,
            "per_family": per_family,
        }

    weakest = gate_scores[score_cols].astype(float).idxmin(axis=1).str.replace("_score", "", regex=False)
    for gate_name, count in weakest.value_counts().items():
        weakest_gate_counts[str(gate_name)] = int(count)

    pairwise_corr = {}
    numeric_scores = gate_scores[score_cols].astype(float).fillna(0.5)
    for left, right in combinations(score_cols, 2):
        corr = float(numeric_scores[left].corr(numeric_scores[right]))
        pairwise_corr[f"{left}|{right}"] = corr

    return {
        "per_gate": per_gate,
        "weakest_gate_counts": weakest_gate_counts,
        "pairwise_score_correlation": pairwise_corr,
    }
