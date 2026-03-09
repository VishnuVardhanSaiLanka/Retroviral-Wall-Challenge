from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def diagnose_gates(gate_scores: pd.DataFrame, labels: pd.Series, families: pd.Series) -> dict:
    out = {}
    score_cols = [c for c in gate_scores.columns if c.endswith("_score")]
    y = labels.loc[gate_scores.index].to_numpy()
    fam = families.loc[gate_scores.index].to_numpy()

    for col in score_cols:
        s = gate_scores[col].astype(float).fillna(0.5).to_numpy()
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else None
        retro = fam == "Retroviral"
        retro_auc = roc_auc_score(y[retro], s[retro]) if retro.any() and len(np.unique(y[retro])) > 1 else None

        # family eta-squared
        fam_groups = [s[fam == f] for f in np.unique(fam)]
        fam_groups = [g for g in fam_groups if len(g) > 1]
        if fam_groups:
            ss_between = sum(len(g) * (g.mean() - s.mean()) ** 2 for g in fam_groups)
            ss_total = np.sum((s - s.mean()) ** 2)
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
        else:
            eta_sq = 0.0

        out[col] = {
            "overall_auc": float(auc) if auc is not None else None,
            "retroviral_within_auc": float(retro_auc) if retro_auc is not None else None,
            "family_eta_squared": eta_sq,
            "active_mean": float(s[y == 1].mean()),
            "inactive_mean": float(s[y == 0].mean()),
        }
    return out
