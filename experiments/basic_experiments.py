#!/usr/bin/env python3
"""Basic experiment setup for Retroviral Wall challenge.

Implements simple, reproducible baselines for:
- Leave-One-Family-Out (LOFO)
- Leave-One-Out (LOO)

Models:
- all_inactive (dummy)
- handcrafted_logreg
- handcrafted_rf
- esm_ridge
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INFORMATIVE_FAMILIES = {
    "Retroviral",
    "Retron",
    "LTR_Retrotransposon",
    "Group_II_Intron",
}


@dataclass
class EvalResult:
    name: str
    lofo_f1_pos: float
    lofo_auc: float
    lofo_macro_f1_informative: float
    lofo_tp: int
    lofo_fp: int
    lofo_fn: int
    lofo_tn: int
    lofo_retroviral_tp: int
    loo_f1_pos: float
    loo_auc: float


class AllInactiveClassifier:
    def get_params(self, deep: bool = True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2), dtype=float)
        probs[:, 0] = 1.0
        return probs


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def evaluate_lofo(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    families: pd.Series,
) -> tuple[float, float, float, int, int, int, int, int, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []

    for family in families.unique():
        train_idx = families != family
        test_idx = families == family

        clf = clone(model)
        clf.fit(X.loc[train_idx], y.loc[train_idx])

        preds = clf.predict(X.loc[test_idx])
        scores = clf.predict_proba(X.loc[test_idx])[:, 1]

        fold_df = pd.DataFrame(
            {
                "family": family,
                "y_true": y.loc[test_idx].to_numpy(),
                "y_pred": preds,
                "y_score": scores,
            },
            index=X.loc[test_idx].index,
        )
        all_rows.append(fold_df)

    pred_df = pd.concat(all_rows).sort_index()

    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    y_score = pred_df["y_score"].to_numpy()

    f1_pos = f1_score(y_true, y_pred, zero_division=0)
    auc = safe_auc(y_true, y_score)
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)

    informative = pred_df[pred_df["family"].isin(INFORMATIVE_FAMILIES)].copy()
    informative_macro = []
    for family in sorted(INFORMATIVE_FAMILIES):
        fam = informative[informative["family"] == family]
        informative_macro.append(
            f1_score(fam["y_true"], fam["y_pred"], average="macro", zero_division=0)
        )
    informative_macro_f1 = float(np.mean(informative_macro))

    retroviral = pred_df[pred_df["family"] == "Retroviral"]
    retroviral_tp = int(np.sum((retroviral["y_true"] == 1) & (retroviral["y_pred"] == 1)))

    return f1_pos, auc, informative_macro_f1, tp, fp, fn, tn, retroviral_tp, pred_df


def evaluate_loo(model, X: pd.DataFrame, y: pd.Series) -> tuple[float, float, pd.DataFrame]:
    loo = LeaveOneOut()
    preds = np.zeros(len(X), dtype=int)
    scores = np.zeros(len(X), dtype=float)

    for train_idx, test_idx in loo.split(X):
        clf = clone(model)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = clf.predict(X.iloc[test_idx])
        scores[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    y_true = y.to_numpy()
    f1_pos = f1_score(y_true, preds, zero_division=0)
    auc = safe_auc(y_true, scores)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": preds, "y_score": scores}, index=X.index)
    return f1_pos, auc, pred_df


def build_handcrafted_logreg(feature_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_cols,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=0.01,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_handcrafted_rf(feature_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), feature_cols),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )


def build_esm_ridge(feature_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_cols,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=0.001,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def load_data(data_dir: Path) -> pd.DataFrame:
    seq = pd.read_csv(data_dir / "rt_sequences.csv")
    hand = pd.read_csv(data_dir / "handcrafted_features.csv")

    emb_npz = np.load(data_dir / "esm2_embeddings.npz", allow_pickle=True)
    emb_names = emb_npz["names"]
    emb = emb_npz["embeddings"]
    emb_cols = [f"esm_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, "rt_name", emb_names)

    df = seq[["rt_name", "active", "rt_family", "pe_efficiency_pct"]].merge(hand, on="rt_name", how="inner")
    df = df.merge(emb_df, on="rt_name", how="inner")

    if len(df) != len(seq):
        raise ValueError(f"Merged dataset has {len(df)} rows, expected {len(seq)}")

    return df


def run_all(df: pd.DataFrame, output_dir: Path, hand_cols: list[str]) -> list[EvalResult]:
    y = df["active"].astype(int)
    families = df["rt_family"]

    esm_cols = [c for c in df.columns if c.startswith("esm_")]

    model_specs: list[tuple[str, Callable[[], object], list[str]]] = [
        ("all_inactive", lambda: AllInactiveClassifier(), hand_cols),
        ("handcrafted_logreg", lambda: build_handcrafted_logreg(hand_cols), hand_cols),
        ("handcrafted_rf", lambda: build_handcrafted_rf(hand_cols), hand_cols),
        ("esm_ridge", lambda: build_esm_ridge(esm_cols), esm_cols),
    ]

    results: list[EvalResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model_fn, cols in model_specs:
        model = model_fn()
        X = df[cols].copy()

        lofo = evaluate_lofo(model, X, y, families)
        loo = evaluate_loo(model, X, y)

        results.append(
            EvalResult(
                name=name,
                lofo_f1_pos=lofo[0],
                lofo_auc=lofo[1],
                lofo_macro_f1_informative=lofo[2],
                lofo_tp=lofo[3],
                lofo_fp=lofo[4],
                lofo_fn=lofo[5],
                lofo_tn=lofo[6],
                lofo_retroviral_tp=lofo[7],
                loo_f1_pos=loo[0],
                loo_auc=loo[1],
            )
        )

        lofo_df = lofo[8].copy()
        lofo_df.insert(0, "rt_name", df.loc[lofo_df.index, "rt_name"].values)
        lofo_df.to_csv(output_dir / f"{name}_lofo_predictions.csv", index=False)

        loo_df = loo[2].copy()
        loo_df.insert(0, "rt_name", df["rt_name"].values)
        loo_df.to_csv(output_dir / f"{name}_loo_predictions.csv", index=False)

    return results


def format_results(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": r.name,
                "lofo_f1_pos": r.lofo_f1_pos,
                "lofo_auc": r.lofo_auc,
                "lofo_macro_f1_informative": r.lofo_macro_f1_informative,
                "lofo_tp": r.lofo_tp,
                "lofo_fp": r.lofo_fp,
                "lofo_fn": r.lofo_fn,
                "lofo_tn": r.lofo_tn,
                "lofo_retroviral_tp": r.lofo_retroviral_tp,
                "loo_f1_pos": r.loo_f1_pos,
                "loo_auc": r.loo_auc,
            }
            for r in results
        ]
    ).sort_values("lofo_macro_f1_informative", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic RT activity experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/basic_experiments"))
    args = parser.parse_args()

    hand_cols = [c for c in pd.read_csv(args.data_dir / "handcrafted_features.csv", nrows=1).columns if c != "rt_name"]
    df = load_data(args.data_dir)
    results = run_all(df, args.output_dir, hand_cols)
    results_df = format_results(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_dir / "summary_metrics.csv", index=False)

    print("\n=== Basic Experiment Summary ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
