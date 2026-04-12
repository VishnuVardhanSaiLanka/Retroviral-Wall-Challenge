#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroviral_wall.utils.structural_features import catalytic_to_thumb_corridor_features, literature_pe_features

DATA_DIR = PROJECT_ROOT / "data"
STRUCTURES_DIR = DATA_DIR / "structures"
GENERALIZATION_DIR = PROJECT_ROOT / "outputs" / "generalization"
INFORMATIVE_FAMILIES = ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]
BENCHMARK_GATE_COLS = [
    "foldability_score",
    "fusion_compat_score",
    "fusion_compat_clash_score",
    "substrate_binding_score",
    "catalytic_score",
    "processivity_score",
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


def load_frozen_feature_set() -> list[str]:
    details = json.loads((GENERALIZATION_DIR / "details.json").read_text())
    residuals = [item["feature"] for item in details["selected_features"]]
    return BENCHMARK_GATE_COLS + residuals


def compute_corridor_df() -> pd.DataFrame:
    sequences = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    rows = []
    for rt_name in sequences["rt_name"]:
        feats = catalytic_to_thumb_corridor_features(STRUCTURES_DIR / f"{rt_name}.pdb")
        rows.append({"rt_name": rt_name, **feats})
    return pd.DataFrame(rows)


def compute_literature_df() -> pd.DataFrame:
    sequences = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    rows = []
    for rt_name in sequences["rt_name"]:
        feats = literature_pe_features(STRUCTURES_DIR / f"{rt_name}.pdb")
        rows.append({"rt_name": rt_name, **feats})
    return pd.DataFrame(rows)


def build_model(feature_cols: list[str], kind: str = "lr") -> Pipeline:
    if kind == "lr":
        clf = LogisticRegression(
            C=0.2,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=42,
        )
        prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    else:
        clf = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
        prep = SimpleImputer(strategy="median")
    return Pipeline(
        steps=[
            ("prep", ColumnTransformer([("num", prep, feature_cols)], remainder="drop")),
            ("clf", clf),
        ]
    )


def evaluate_lofo(df: pd.DataFrame, feature_cols: list[str], kind: str) -> tuple[dict, pd.DataFrame]:
    model = build_model(feature_cols, kind)
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


def retroviral_rescue_audit(df: pd.DataFrame, preds: pd.DataFrame, corridor_cols: list[str]) -> dict:
    merged = preds.merge(df[["rt_name"] + corridor_cols], on="rt_name", how="left")
    retro = merged[merged["held_out_family"] == "Retroviral"].copy()
    tp = retro[(retro["active"] == 1) & (retro["predicted_active"] == 1)]
    fn = retro[(retro["active"] == 1) & (retro["predicted_active"] == 0)]
    rows = []
    if len(tp) and len(fn):
        for col in corridor_cols:
            vals_tp = pd.to_numeric(tp[col], errors="coerce").dropna()
            vals_fn = pd.to_numeric(fn[col], errors="coerce").dropna()
            if len(vals_tp) == 0 or len(vals_fn) == 0:
                continue
            rows.append(
                {
                    "feature": col,
                    "tp_mean": float(vals_tp.mean()),
                    "fn_mean": float(vals_fn.mean()),
                    "delta_tp_minus_fn": float(vals_tp.mean() - vals_fn.mean()),
                }
            )
    rows = sorted(rows, key=lambda x: abs(x["delta_tp_minus_fn"]), reverse=True)
    return {"n_tp": int(len(tp)), "n_fn": int(len(fn)), "feature_deltas": rows}


def run(output_dir: Path) -> dict:
    frozen_features = load_frozen_feature_set()
    seq = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    hc = pd.read_csv(DATA_DIR / "handcrafted_features.csv")
    gate_scores = pd.read_csv(PROJECT_ROOT / "retroviral_wall" / "outputs" / "gate_scores" / "all_gate_scores.csv")
    corridor = compute_corridor_df()
    literature = compute_literature_df()
    base = seq.merge(hc, on="rt_name", how="inner").merge(gate_scores, on="rt_name", how="inner").merge(corridor, on="rt_name", how="inner").merge(literature, on="rt_name", how="inner")

    corridor_cols = [c for c in corridor.columns if c != "rt_name"]
    literature_cols = [c for c in literature.columns if c != "rt_name"]
    model_specs = {
        "frozen_hybrid_lr": (frozen_features, "lr"),
        "frozen_plus_corridor_lr": (frozen_features + corridor_cols, "lr"),
        "frozen_plus_corridor_et": (frozen_features + corridor_cols, "et"),
        "corridor_only_lr": (corridor_cols, "lr"),
        "frozen_plus_literature_lr": (frozen_features + literature_cols, "lr"),
        "frozen_plus_literature_et": (frozen_features + literature_cols, "et"),
        "literature_only_lr": (literature_cols, "lr"),
    }
    for size in (1, 2, 3):
        for subset in itertools.combinations(corridor_cols, size):
            name = "frozen_plus_" + "_".join(subset) + "_lr"
            model_specs[name] = (frozen_features + list(subset), "lr")
    for size in (1, 2):
        for subset in itertools.combinations(literature_cols, size):
            name = "frozen_plus_" + "_".join(subset) + "_lr"
            model_specs[name] = (frozen_features + list(subset), "lr")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    details = {"frozen_features": frozen_features, "corridor_features": corridor_cols, "literature_features": literature_cols, "models": {}}
    best_name = None
    best_score = -1.0
    best_preds = None
    for name, (features, kind) in model_specs.items():
        metrics, preds = evaluate_lofo(base, features, kind)
        summaries.append(
            {
                "model": name,
                "lofo_macro_f1_informative": metrics["lofo_macro_f1_informative"],
                "overall_f1": metrics["overall_f1"],
                "overall_auc": metrics["overall_auc"],
                "retroviral_tp_of_12": metrics["retroviral_tp_of_12"],
                "n_features": metrics["n_features"],
            }
        )
        details["models"][name] = metrics
        preds.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        if metrics["lofo_macro_f1_informative"] > best_score:
            best_score = metrics["lofo_macro_f1_informative"]
            best_name = name
            best_preds = preds
    assert best_name is not None and best_preds is not None
    summary_df = pd.DataFrame(summaries).sort_values("lofo_macro_f1_informative", ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    audit_cols = corridor_cols + literature_cols
    with open(output_dir / "retroviral_rescue_audit.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, **retroviral_rescue_audit(base, best_preds, audit_cols)}, f, indent=2)
    return {"summary": summary_df.to_dict(orient="records"), "best_model": best_name, "best_score": best_score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment frozen benchmark with corridor features.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/augment_frozen_benchmark"))
    args = parser.parse_args()
    result = run(args.output_dir)
    print(pd.DataFrame(result["summary"]).to_string(index=False, float_format=lambda x: f"{x:.3f}" if x is not None else "nan"))
    print(f"\nBest model: {result['best_model']} score={result['best_score']:.3f}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
