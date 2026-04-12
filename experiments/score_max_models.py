#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


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


def score_max_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    hand_cols = [c for c in df.columns if c not in {"rt_name", "active", "rt_family"} and not c.startswith("esm_")]
    foldseek_cols = [c for c in hand_cols if c.startswith("foldseek_")]
    structure_proxy_cols = [
        c
        for c in hand_cols
        if c.startswith(("sp_", "thumb_", "t6", "t7", "n_hairpins", "best_turn"))
    ]
    esm_cols = [c for c in df.columns if c.startswith("esm_")]

    return {
        "hand_all": hand_cols,
        "hand_foldseek": sorted(set(foldseek_cols + structure_proxy_cols + ["protein_length_aa"])),
        "full_plus_esm": hand_cols + esm_cols,
    }


def build_model(num_cols: list[str], family: bool, kind: str) -> Pipeline:
    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=(kind == "lr"))),
                ]
            ),
            num_cols,
        )
    ]
    if family:
        transformers.append(("family", OneHotEncoder(handle_unknown="ignore"), ["rt_family"]))

    if kind == "lr":
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=42,
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
        )
    elif kind == "et":
        clf = ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(kind)

    return Pipeline(
        steps=[
            ("prep", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("clf", clf),
        ]
    )


def evaluate_cv(model: Pipeline, X: pd.DataFrame, y: np.ndarray, cv) -> tuple[float, float | None]:
    preds = np.zeros(len(X), dtype=int)
    scores = np.zeros(len(X), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        clf = clone(model)
        clf.fit(X.iloc[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
        preds[test_idx] = (scores[test_idx] >= 0.5).astype(int)

    return float(f1_score(y, preds, zero_division=0)), safe_auc(y, scores)


def evaluate_trainfit(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float | None]:
    clf = clone(model)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    preds = (scores >= 0.5).astype(int)
    return float(f1_score(y, preds, zero_division=0)), safe_auc(y, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score-max model hunt for in-distribution evaluation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/score_max_summary.csv"))
    args = parser.parse_args()

    df = load_data(args.data_dir)
    y = df["active"].astype(int).to_numpy()
    X = df.drop(columns=["active"])

    rows = []
    for feat_name, feat_cols in score_max_feature_sets(df).items():
        for family in (False, True):
            for kind in ("lr", "rf", "et"):
                name = f"{feat_name}|family={int(family)}|{kind}"
                model = build_model(feat_cols, family=family, kind=kind)
                loo_f1, loo_auc = evaluate_cv(model, X, y, LeaveOneOut())
                rand_f1, rand_auc = evaluate_cv(model, X, y, StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
                fit_f1, fit_auc = evaluate_trainfit(model, X, y)
                rows.append(
                    {
                        "model": name,
                        "loo_f1": loo_f1,
                        "loo_auc": loo_auc,
                        "rand5_f1": rand_f1,
                        "rand5_auc": rand_auc,
                        "train_f1": fit_f1,
                        "train_auc": fit_auc,
                    }
                )

    out_df = pd.DataFrame(rows).sort_values(["rand5_f1", "loo_f1", "train_f1"], ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(out_df.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}" if x is not None else "nan"))
    print(f"\nSaved score-max summary to {args.output}")


if __name__ == "__main__":
    main()
