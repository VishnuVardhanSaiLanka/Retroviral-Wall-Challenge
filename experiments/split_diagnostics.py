#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INFORMATIVE_FAMILIES = {
    "Retroviral",
    "Retron",
    "LTR_Retrotransposon",
    "Group_II_Intron",
}


@dataclass
class RunResult:
    model: str
    split: str
    f1_pos: float
    auc: float | None
    informative_macro_f1: float | None
    retroviral_tp: int | None
    tp: int
    fp: int
    fn: int
    tn: int


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def load_data(data_dir: Path) -> pd.DataFrame:
    seq = pd.read_csv(data_dir / "rt_sequences.csv")
    hand = pd.read_csv(data_dir / "handcrafted_features.csv")
    emb_npz = np.load(data_dir / "esm2_embeddings.npz", allow_pickle=True)
    emb_cols = [f"esm_{i}" for i in range(emb_npz["embeddings"].shape[1])]
    emb_df = pd.DataFrame(emb_npz["embeddings"], columns=emb_cols)
    emb_df.insert(0, "rt_name", emb_npz["names"])

    df = seq[["rt_name", "active", "rt_family", "protein_length_aa"]].merge(hand, on="rt_name", how="inner")
    df = df.merge(emb_df, on="rt_name", how="inner")
    return df


def build_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    hand_cols = [c for c in df.columns if c not in {"rt_name", "active", "rt_family"} and not c.startswith("esm_")]
    emb_cols = [c for c in df.columns if c.startswith("esm_")]

    return {
        "hand_lr": Pipeline(
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
                                hand_cols,
                            )
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hand_family_lr": Pipeline(
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
                                hand_cols,
                            ),
                            ("family", OneHotEncoder(handle_unknown="ignore"), ["rt_family"]),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hand_family_rf": Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            ("num", SimpleImputer(strategy="median"), hand_cols),
                            ("family", OneHotEncoder(handle_unknown="ignore"), ["rt_family"]),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=800,
                        max_depth=None,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "esm_family_lr": Pipeline(
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
                                emb_cols + ["protein_length_aa"],
                            ),
                            ("family", OneHotEncoder(handle_unknown="ignore"), ["rt_family"]),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    families: np.ndarray,
) -> tuple[float, float | None, float | None, int | None, int, int, int, int]:
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    informative_macro_f1 = None
    retroviral_tp = None

    if families is not None:
        informative_scores = []
        for family in sorted(INFORMATIVE_FAMILIES):
            mask = families == family
            if mask.sum() == 0:
                continue
            informative_scores.append(f1_score(y_true[mask], y_pred[mask], zero_division=0))
        if informative_scores:
            informative_macro_f1 = float(np.mean(informative_scores))

        retroviral_mask = families == "Retroviral"
        if retroviral_mask.sum() > 0:
            retroviral_tp = int(np.sum((y_true[retroviral_mask] == 1) & (y_pred[retroviral_mask] == 1)))

    return (
        float(f1_score(y_true, y_pred, zero_division=0)),
        safe_auc(y_true, y_score),
        informative_macro_f1,
        retroviral_tp,
        tp,
        fp,
        fn,
        tn,
    )


def evaluate_lofo(model: Pipeline, df: pd.DataFrame) -> RunResult:
    rows = []
    families = df["rt_family"]
    y = df["active"].astype(int)
    X = df.drop(columns=["active"])

    for held_out in sorted(families.unique()):
        train_mask = families != held_out
        test_mask = families == held_out
        clf = clone(model)
        clf.fit(X.loc[train_mask], y.loc[train_mask])
        preds = clf.predict(X.loc[test_mask])
        scores = clf.predict_proba(X.loc[test_mask])[:, 1]
        rows.append(
            pd.DataFrame(
                {
                    "y_true": y.loc[test_mask].to_numpy(),
                    "y_pred": preds,
                    "y_score": scores,
                    "family": families.loc[test_mask].to_numpy(),
                }
            )
        )

    pred_df = pd.concat(rows, ignore_index=True)
    metrics = evaluate_predictions(
        pred_df["y_true"].to_numpy(),
        pred_df["y_pred"].to_numpy(),
        pred_df["y_score"].to_numpy(),
        pred_df["family"].to_numpy(),
    )
    return RunResult("pending", "LOFO", *metrics)


def evaluate_loo(model: Pipeline, df: pd.DataFrame) -> RunResult:
    loo = LeaveOneOut()
    X = df.drop(columns=["active"])
    y = df["active"].astype(int).to_numpy()
    preds = np.zeros(len(df), dtype=int)
    scores = np.zeros(len(df), dtype=float)

    for train_idx, test_idx in loo.split(X):
        clf = clone(model)
        clf.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X.iloc[test_idx])
        scores[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    metrics = evaluate_predictions(y, preds, scores, df["rt_family"].to_numpy())
    return RunResult("pending", "LOO", *metrics)


def evaluate_random_cv(model: Pipeline, df: pd.DataFrame, seed: int = 42) -> RunResult:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X = df.drop(columns=["active"])
    y = df["active"].astype(int).to_numpy()
    preds = np.zeros(len(df), dtype=int)
    scores = np.zeros(len(df), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        clf = clone(model)
        clf.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X.iloc[test_idx])
        scores[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    metrics = evaluate_predictions(y, preds, scores, df["rt_family"].to_numpy())
    return RunResult("pending", "Random5Fold", *metrics)


def evaluate_resubstitution(model: Pipeline, df: pd.DataFrame) -> RunResult:
    X = df.drop(columns=["active"])
    y = df["active"].astype(int).to_numpy()
    clf = clone(model)
    clf.fit(X, y)
    preds = clf.predict(X)
    scores = clf.predict_proba(X)[:, 1]
    metrics = evaluate_predictions(y, preds, scores, df["rt_family"].to_numpy())
    return RunResult("pending", "TrainOnTrain", *metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare score regimes across validation splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/split_diagnostics.csv"))
    args = parser.parse_args()

    df = load_data(args.data_dir)
    models = build_models(df)
    rows: list[RunResult] = []

    for model_name, model in models.items():
        for evaluator in (evaluate_lofo, evaluate_loo, evaluate_random_cv, evaluate_resubstitution):
            result = evaluator(model, df)
            result.model = model_name
            rows.append(result)

    out_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["split", "f1_pos"], ascending=[True, False])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if x is not None else "nan"))
    print(f"\nSaved split diagnostics to {args.output}")


if __name__ == "__main__":
    main()
