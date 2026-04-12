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
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroviral_wall.config import DATA_DIR, STRUCTURES_DIR
from retroviral_wall.download_external import download_all_external
from retroviral_wall.gates import (
    CatalyticCompetenceGate,
    FoldabilityGate,
    FusionCompatibilityGate,
    ProcessivityGate,
    SubstrateBindingGate,
)

INFORMATIVE_FAMILIES = ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]
BIOLOGY_FEATURES = [
    "triad_found_bin",
    "triad_best_rmsd",
    "D1_D2_dist",
    "D2_D3_dist",
    "sasa_avg",
    "sasa_high_pct",
    "pocket_hbonds_per_res",
    "thumb_surface_net_charge",
    "thumb_charge_ratio",
    "instability_index",
    "camsol_score",
    "t40_raw",
    "t45_raw",
    "protein_length_aa",
]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def optimise_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_f1 = -1.0
    best_t = 0.5
    for t in np.arange(0.1, 0.91, 0.02):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def build_lr_model(feature_cols: list[str], c: float = 0.3) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), feature_cols)],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def lofo_score_within_train(df: pd.DataFrame, feature_cols: list[str], c: float) -> float:
    if not feature_cols:
        return -1.0
    model = build_lr_model(feature_cols, c=c)
    fam = df["rt_family"].to_numpy()
    informative = []
    for held_out in sorted(np.unique(fam)):
        train_mask = fam != held_out
        test_mask = fam == held_out
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        clf = clone(model)
        clf.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, "active"].astype(int).to_numpy())
        train_scores = clf.predict_proba(df.loc[train_mask, feature_cols])[:, 1]
        test_scores = clf.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
        threshold = optimise_threshold(train_scores, df.loc[train_mask, "active"].astype(int).to_numpy())
        preds = (test_scores >= threshold).astype(int)
        fam_f1 = float(f1_score(df.loc[test_mask, "active"].astype(int).to_numpy(), preds, zero_division=0))
        if held_out in INFORMATIVE_FAMILIES:
            informative.append(fam_f1)
    return float(np.mean(informative)) if informative else -1.0


def compute_gate_scores(clean: bool = False) -> pd.DataFrame:
    sequences = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    handcrafted = pd.read_csv(DATA_DIR / "handcrafted_features.csv")
    external = download_all_external()
    gates = [
        FoldabilityGate(),
        FusionCompatibilityGate(use_similarity_fallback=not clean),
        SubstrateBindingGate(),
        CatalyticCompetenceGate(),
        ProcessivityGate(use_similarity_support=not clean, use_literature_prior=not clean),
    ]

    gate_scores = pd.DataFrame(index=sequences["rt_name"])
    for gate in gates:
        results = gate.compute_scores(sequences, str(STRUCTURES_DIR), handcrafted, external)
        gate_scores = gate_scores.join(gate.score_matrix(results).set_index("rt_name"))
    return gate_scores.reset_index()


def nested_select_features(train_df: pd.DataFrame, candidate_cols: list[str], top_k: int = 8) -> list[str]:
    y = train_df["active"].astype(int).to_numpy()
    fam_codes = pd.Categorical(train_df["rt_family"]).codes
    rows = []
    for col in candidate_cols:
        vals = pd.to_numeric(train_df[col], errors="coerce")
        if vals.isna().mean() > 0.35:
            continue
        vals = vals.fillna(vals.median())
        if vals.std() < 1e-6:
            continue
        auc = safe_auc(y, vals.to_numpy())
        active_auc_abs = max(auc, 1.0 - auc) if auc is not None else 0.5
        family_mi = float(mutual_info_classif(vals.to_frame(), fam_codes, discrete_features=False, random_state=42)[0])
        within_family = []
        for family in sorted(train_df["rt_family"].unique()):
            mask = train_df["rt_family"] == family
            fam_auc = safe_auc(y[mask], vals.loc[mask].to_numpy())
            if fam_auc is not None:
                within_family.append(max(fam_auc, 1.0 - fam_auc))
        stable_signal = float(np.mean(within_family)) if within_family else 0.5
        rows.append(
            {
                "feature": col,
                "score": stable_signal + 0.5 * active_auc_abs - 0.75 * family_mi,
            }
        )
    ranked = pd.DataFrame(rows).sort_values("score", ascending=False)
    return ranked["feature"].head(top_k).tolist()


def evaluate_nested_lofo(df: pd.DataFrame, model_name: str, feature_builder) -> tuple[dict, pd.DataFrame]:
    y = df["active"].astype(int).to_numpy()
    fam = df["rt_family"].to_numpy()
    rows = []
    feature_usage: dict[str, int] = {}
    for held_out in sorted(np.unique(fam)):
        train_mask = fam != held_out
        test_mask = fam == held_out
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        feature_cols, c = feature_builder(train_df)
        for col in feature_cols:
            feature_usage[col] = feature_usage.get(col, 0) + 1
        model = build_lr_model(feature_cols, c=c)
        clf = clone(model)
        clf.fit(train_df[feature_cols], train_df["active"].astype(int).to_numpy())
        train_scores = clf.predict_proba(train_df[feature_cols])[:, 1]
        test_scores = clf.predict_proba(test_df[feature_cols])[:, 1]
        threshold = optimise_threshold(train_scores, train_df["active"].astype(int).to_numpy())
        for rt_name, score in zip(test_df["rt_name"], test_scores):
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
    informative = []
    per_family = {}
    for family in sorted(pred_df["held_out_family"].unique()):
        mask = pred_df["held_out_family"] == family
        fam_f1 = float(f1_score(y_true[mask], y_pred[mask], zero_division=0))
        per_family[family] = {
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
        "model": model_name,
        "lofo_macro_f1_informative": float(np.mean(informative)) if informative else 0.0,
        "overall_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "overall_auc": safe_auc(y_true, y_score),
        "retroviral_tp_of_12": per_family.get("Retroviral", {}).get("tp", 0),
        "feature_usage": feature_usage,
        "per_family": per_family,
    }
    return metrics, pred_df


def run_search(output_dir: Path) -> dict:
    sequences = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    handcrafted = pd.read_csv(DATA_DIR / "handcrafted_features.csv")
    gate_scores = compute_gate_scores(clean=False)
    clean_gate_scores = compute_gate_scores(clean=True)

    full = sequences.merge(handcrafted, on="rt_name", how="inner").merge(gate_scores, on="rt_name", how="inner")
    full_clean = sequences.merge(handcrafted, on="rt_name", how="inner").merge(clean_gate_scores, on="rt_name", how="inner")
    gate_cols = [c for c in gate_scores.columns if c.endswith("_score")]
    gate_cols_clean = [c for c in clean_gate_scores.columns if c.endswith("_score")]
    hybrid_candidates = [
        c
        for c in handcrafted.columns
        if c not in {"rt_name"}
        and not c.startswith("foldseek_")
        and c not in {"perplexity", "avg_log_likelihood"}
    ]

    def bio_builder(train_df: pd.DataFrame):
        bio_sets = {
            "catalytic_core": ["triad_found_bin", "triad_best_rmsd", "D1_D2_dist", "D2_D3_dist"],
            "bio_small": ["triad_found_bin", "triad_best_rmsd", "sasa_avg", "pocket_hbonds_per_res", "thumb_surface_net_charge", "instability_index", "camsol_score", "protein_length_aa"],
            "bio_full": [c for c in BIOLOGY_FEATURES if c in train_df.columns],
        }
        candidates = []
        for c_val in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            for name, cols in bio_sets.items():
                valid_cols = [c for c in cols if c in train_df.columns]
                score = lofo_score_within_train(train_df, valid_cols, c_val)
                candidates.append((score, c_val, valid_cols, name))
        best = max(candidates, key=lambda x: x[0])
        return best[2], best[1]

    def gates_builder(train_df: pd.DataFrame):
        candidates = []
        for c_val in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            score = lofo_score_within_train(train_df, gate_cols, c_val)
            candidates.append((score, c_val))
        best = max(candidates, key=lambda x: x[0])
        return gate_cols, best[1]

    def clean_gates_builder(train_df: pd.DataFrame):
        candidates = []
        for c_val in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            score = lofo_score_within_train(train_df, gate_cols_clean, c_val)
            candidates.append((score, c_val))
        best = max(candidates, key=lambda x: x[0])
        return gate_cols_clean, best[1]

    def sparse_hybrid_builder(train_df: pd.DataFrame):
        candidates = []
        for top_k in (4, 6, 8, 10, 12):
            selected = nested_select_features(train_df, hybrid_candidates, top_k=top_k)
            cols = gate_cols_clean + selected
            for c_val in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
                score = lofo_score_within_train(train_df, cols, c_val)
                candidates.append((score, c_val, cols, top_k))
        best = max(candidates, key=lambda x: x[0])
        return best[2], best[1]

    model_runs = [
        ("biology_lr", full, bio_builder),
        ("gates_default_lr", full, gates_builder),
        ("gates_clean_lr", full_clean, clean_gates_builder),
        ("sparse_hybrid_clean_lr", full_clean, sparse_hybrid_builder),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    details = {}
    for name, dataset, builder in model_runs:
        metrics, preds = evaluate_nested_lofo(dataset, name, builder)
        summaries.append(
            {
                "model": name,
                "lofo_macro_f1_informative": metrics["lofo_macro_f1_informative"],
                "overall_f1": metrics["overall_f1"],
                "overall_auc": metrics["overall_auc"],
                "retroviral_tp_of_12": metrics["retroviral_tp_of_12"],
            }
        )
        details[name] = metrics
        preds.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    summary_df = pd.DataFrame(summaries).sort_values("lofo_macro_f1_informative", ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    gate_scores.to_csv(output_dir / "gate_scores_default.csv", index=False)
    clean_gate_scores.to_csv(output_dir / "gate_scores_clean.csv", index=False)
    return {"summary": summary_df.to_dict(orient="records"), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simplified nested-LOFO search.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simplified_search"))
    args = parser.parse_args()
    result = run_search(args.output_dir)
    print(pd.DataFrame(result["summary"]).to_string(index=False, float_format=lambda x: f"{x:.3f}" if x is not None else "nan"))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
