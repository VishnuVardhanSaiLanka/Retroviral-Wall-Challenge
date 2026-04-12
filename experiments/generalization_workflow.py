#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroviral_wall.analysis.gate_audit import audit_gates

INFORMATIVE_FAMILIES = ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def load_bundle(data_dir: Path, gate_score_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequences = pd.read_csv(data_dir / "rt_sequences.csv")
    handcrafted = pd.read_csv(data_dir / "handcrafted_features.csv")
    gate_scores = pd.read_csv(gate_score_path)
    full = sequences.merge(handcrafted, on="rt_name", how="inner").merge(gate_scores, on="rt_name", how="inner")
    return full, gate_scores


def select_low_leakage_features(df: pd.DataFrame, handcrafted_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in {"rt_name", "active", "rt_family", "sequence"}]
    hand_cols = [c for c in feature_cols if c in set(handcrafted_columns) and c != "rt_name"]
    y = df["active"].astype(int).to_numpy()
    fam_codes = pd.Categorical(df["rt_family"]).codes
    rows = []
    for col in hand_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().mean() > 0.35:
            continue
        vals = vals.fillna(vals.median())
        if vals.std() < 1e-6:
            continue
        auc = safe_auc(y, vals.to_numpy())
        active_auc_abs = max(auc, 1.0 - auc) if auc is not None else 0.5
        family_mi = float(mutual_info_classif(vals.to_frame(), fam_codes, discrete_features=False, random_state=42)[0])
        within_family_aucs = []
        for family in INFORMATIVE_FAMILIES:
            mask = df["rt_family"] == family
            fam_auc = safe_auc(y[mask], vals.loc[mask].to_numpy())
            if fam_auc is not None:
                within_family_aucs.append(max(fam_auc, 1.0 - fam_auc))
        stable_signal = float(np.mean(within_family_aucs)) if within_family_aucs else 0.5
        rows.append(
            {
                "feature": col,
                "active_auc_abs": active_auc_abs,
                "family_mi": family_mi,
                "stable_signal": stable_signal,
                "leakage_adjusted_score": stable_signal + 0.5 * active_auc_abs - 0.75 * family_mi,
                "is_foldseek": col.startswith("foldseek_"),
            }
        )
    report = pd.DataFrame(rows).sort_values("leakage_adjusted_score", ascending=False)
    selected = report[
        (report["family_mi"] <= 0.55)
        & (report["active_auc_abs"] >= 0.60)
        & (report["stable_signal"] >= 0.55)
        & (~report["is_foldseek"])
    ].copy()
    if len(selected) < 8:
        selected = report[~report["is_foldseek"]].head(min(12, len(report))).copy()
    return selected, report


def build_model(num_cols: list[str], model_kind: str) -> Pipeline:
    if model_kind == "lr":
        clf = LogisticRegression(
            C=0.3,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=42,
        )
        prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    elif model_kind == "et":
        clf = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
        prep = SimpleImputer(strategy="median")
    else:
        raise ValueError(model_kind)

    return Pipeline(
        steps=[
            ("prep", ColumnTransformer([("num", prep, num_cols)], remainder="drop")),
            ("clf", clf),
        ]
    )


def optimise_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_f1 = -1.0
    best_t = 0.5
    for t in np.arange(0.1, 0.91, 0.02):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def evaluate_lofo(df: pd.DataFrame, feature_cols: list[str], model_kind: str) -> tuple[dict, pd.DataFrame]:
    model = build_model(feature_cols, model_kind)
    y = df["active"].astype(int).to_numpy()
    fam = df["rt_family"].to_numpy()
    rows = []
    for held_out in sorted(np.unique(fam)):
        train_mask = fam != held_out
        test_mask = fam == held_out
        X_train = df.loc[train_mask, feature_cols]
        y_train = y[train_mask]
        X_test = df.loc[test_mask, feature_cols]
        clf = clone(model)
        clf.fit(X_train, y_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)
        for rt_name, score in zip(df.loc[test_mask, "rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )
    pred_df = pd.DataFrame(rows).merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    y_true = pred_df["active"].to_numpy()
    y_pred = pred_df["predicted_active"].to_numpy()
    y_score = pred_df["predicted_score"].to_numpy()
    per_family = {}
    informative = []
    for family in sorted(pred_df["held_out_family"].unique()):
        mask = pred_df["held_out_family"] == family
        fam_f1 = float(f1_score(y_true[mask], y_pred[mask], zero_division=0))
        per_family[family] = {
            "n": int(mask.sum()),
            "n_active": int(y_true[mask].sum()),
            "f1": fam_f1,
            "auc": safe_auc(y_true[mask], y_score[mask]),
            "tp": int(((y_true[mask] == 1) & (y_pred[mask] == 1)).sum()),
            "fp": int(((y_true[mask] == 0) & (y_pred[mask] == 1)).sum()),
            "fn": int(((y_true[mask] == 1) & (y_pred[mask] == 0)).sum()),
            "tn": int(((y_true[mask] == 0) & (y_pred[mask] == 0)).sum()),
        }
        if family in INFORMATIVE_FAMILIES:
            informative.append(fam_f1)
    metrics = {
        "lofo_macro_f1_informative": float(np.mean(informative)) if informative else 0.0,
        "overall_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "overall_auc": safe_auc(y_true, y_score),
        "retroviral_tp_of_12": per_family.get("Retroviral", {}).get("tp", 0),
        "n_features": len(feature_cols),
        "per_family": per_family,
    }
    return metrics, pred_df


def run_generalization_workflow(data_dir: Path, gate_score_path: Path, output_dir: Path) -> dict:
    full, gate_scores = load_bundle(data_dir, gate_score_path)
    handcrafted_columns = pd.read_csv(data_dir / "handcrafted_features.csv", nrows=1).columns.tolist()
    gate_scores_idx = gate_scores.set_index("rt_name")
    seq_idx = full.set_index("rt_name")
    gate_audit = audit_gates(
        gate_scores_idx,
        seq_idx["active"],
        seq_idx["rt_family"],
    )

    selected_features, feature_report = select_low_leakage_features(full, handcrafted_columns)
    selected_cols = selected_features["feature"].tolist()
    gate_cols = [c for c in gate_scores.columns if c.endswith("_score")]
    gate_cols = [c for c in gate_cols if c != "rt_name"]
    handcrafted_cols = [c for c in selected_cols if c in full.columns]
    no_foldseek_cols = [c for c in handcrafted_columns if c not in {"rt_name"} and not c.startswith("foldseek_")]

    model_specs = {
        "gates_lr": gate_cols,
        "gates_et": gate_cols,
        "robust_handcrafted_lr": handcrafted_cols,
        "robust_handcrafted_et": handcrafted_cols,
        "handcrafted_no_foldseek_lr": no_foldseek_cols,
        "hybrid_lr": gate_cols + handcrafted_cols,
        "hybrid_et": gate_cols + handcrafted_cols,
    }

    summary_rows = []
    detailed = {"gate_audit": gate_audit, "selected_features": selected_features.to_dict(orient="records"), "models": {}}
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, cols in model_specs.items():
        if not cols:
            continue
        kind = "et" if model_name.endswith("_et") else "lr"
        metrics, preds = evaluate_lofo(full, cols, kind)
        summary_rows.append(
            {
                "model": model_name,
                "lofo_macro_f1_informative": metrics["lofo_macro_f1_informative"],
                "overall_f1": metrics["overall_f1"],
                "overall_auc": metrics["overall_auc"],
                "retroviral_tp_of_12": metrics["retroviral_tp_of_12"],
                "n_features": metrics["n_features"],
            }
        )
        detailed["models"][model_name] = metrics
        preds.to_csv(output_dir / f"{model_name}_lofo_predictions.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("lofo_macro_f1_informative", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    feature_report.to_csv(output_dir / "feature_leakage_report.csv", index=False)
    with open(output_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    return {"summary": summary.to_dict(orient="records"), "selected_feature_count": len(selected_cols)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run family-robust generalization workflow.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--gate-scores", type=Path, default=Path("retroviral_wall/outputs/gate_scores/all_gate_scores.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generalization"))
    args = parser.parse_args()

    result = run_generalization_workflow(args.data_dir, args.gate_scores, args.output_dir)
    print(pd.DataFrame(result["summary"]).to_string(index=False, float_format=lambda x: f"{x:.3f}" if x is not None else "nan"))
    print(f"\nSelected robust handcrafted features: {result['selected_feature_count']}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
