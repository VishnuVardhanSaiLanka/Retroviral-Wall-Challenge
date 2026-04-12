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
BIO_RESIDUAL_CANDIDATES = [
    "triad_found_bin",
    "D1_D2_dist",
    "D2_D3_dist",
    "triad_best_rmsd",
    "n_salt_bridges",
    "salt_per_res",
    "hbonds_per_res",
    "pocket_hbonds",
    "pocket_hbonds_per_res",
    "pocket_salt_bridges",
    "hydrophobic_per_res",
    "thumb_surface_net_charge",
    "thumb_charge_ratio",
    "thumb_surface_positive",
    "thumb_surface_negative",
    "instability_index",
    "camsol_score",
    "camsol_min_profile",
    "t40_raw",
    "t45_raw",
    "protein_length_aa",
    "sasa_avg",
    "sasa_high_pct",
    "sasa_low_pct",
    "g_factor_overall",
    "ramachandran_outliers",
    "ramachandran_allowed",
    "ramachandran_favoured",
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


def build_model(feature_cols: list[str], model_kind: str = "lr") -> Pipeline:
    if model_kind == "lr":
        clf = LogisticRegression(
            C=0.25,
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


def rank_residual_features(df: pd.DataFrame, candidate_cols: list[str], top_k: int = 14) -> list[str]:
    y = df["active"].astype(int).to_numpy()
    fam_codes = pd.Categorical(df["rt_family"]).codes
    rows = []
    for col in candidate_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().mean() > 0.35:
            continue
        vals = vals.fillna(vals.median())
        if vals.std() < 1e-6:
            continue
        auc = safe_auc(y, vals.to_numpy())
        active_auc_abs = max(auc, 1.0 - auc) if auc is not None else 0.5
        family_mi = float(mutual_info_classif(vals.to_frame(), fam_codes, discrete_features=False, random_state=42)[0])
        within_family = []
        for family in INFORMATIVE_FAMILIES:
            mask = df["rt_family"] == family
            fam_auc = safe_auc(y[mask], vals.loc[mask].to_numpy())
            if fam_auc is not None:
                within_family.append(max(fam_auc, 1.0 - fam_auc))
        stable_signal = float(np.mean(within_family)) if within_family else 0.5
        rows.append((col, stable_signal + 0.55 * active_auc_abs - 0.55 * family_mi))
    ranked = [name for name, _ in sorted(rows, key=lambda x: x[1], reverse=True)]
    return ranked[:top_k]


def evaluate_lofo(df: pd.DataFrame, feature_cols: list[str], model_kind: str) -> tuple[dict, pd.DataFrame]:
    model = build_model(feature_cols, model_kind=model_kind)
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


def retroviral_mechanism_audit(df: pd.DataFrame, preds: pd.DataFrame, feature_cols: list[str]) -> dict:
    merged = preds.merge(df[["rt_name"] + feature_cols], on="rt_name", how="left")
    retro = merged[merged["held_out_family"] == "Retroviral"].copy()
    tp = retro[(retro["active"] == 1) & (retro["predicted_active"] == 1)]
    fn = retro[(retro["active"] == 1) & (retro["predicted_active"] == 0)]
    if len(tp) == 0 or len(fn) == 0:
        return {"n_tp": int(len(tp)), "n_fn": int(len(fn)), "feature_deltas": []}
    rows = []
    for col in feature_cols:
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
    return {"n_tp": int(len(tp)), "n_fn": int(len(fn)), "feature_deltas": rows[:20]}


def run(output_dir: Path) -> dict:
    sequences = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    handcrafted = pd.read_csv(DATA_DIR / "handcrafted_features.csv")
    gate_default = compute_gate_scores(clean=False)
    gate_clean = compute_gate_scores(clean=True)

    gate_feature_cols = [
        "foldability_score",
        "fusion_compat_score",
        "fusion_compat_active_site_access",
        "fusion_compat_linker_feasibility",
        "fusion_compat_terminal_accessibility",
        "fusion_compat_core_compactness",
        "fusion_compat_fusion_readiness",
        "substrate_binding_score",
        "substrate_binding_path_positive_continuity",
        "substrate_binding_path_openness",
        "substrate_binding_path_extension",
        "substrate_binding_catalytic_accessibility",
        "substrate_binding_track_to_thumb",
        "catalytic_score",
        "catalytic_acidic_cluster_compactness",
        "catalytic_catalytic_neighborhood",
        "catalytic_motif_context",
        "processivity_score",
        "processivity_path_positive_density",
        "processivity_path_extension",
        "processivity_path_catalytic_proximity",
        "processivity_thumb_support",
        "processivity_path_openness",
    ]

    full_default = sequences.merge(handcrafted, on="rt_name", how="inner").merge(gate_default, on="rt_name", how="inner")
    full_clean = sequences.merge(handcrafted, on="rt_name", how="inner").merge(gate_clean, on="rt_name", how="inner")

    residuals_default = rank_residual_features(full_default, [c for c in BIO_RESIDUAL_CANDIDATES if c in full_default.columns], top_k=14)
    residuals_clean = rank_residual_features(full_clean, [c for c in BIO_RESIDUAL_CANDIDATES if c in full_clean.columns], top_k=14)

    model_specs = {
        "bio_augmented_default_lr": (full_default, [c for c in gate_feature_cols if c in full_default.columns] + residuals_default, "lr"),
        "bio_augmented_clean_lr": (full_clean, [c for c in gate_feature_cols if c in full_clean.columns] + residuals_clean, "lr"),
        "bio_augmented_default_et": (full_default, [c for c in gate_feature_cols if c in full_default.columns] + residuals_default, "et"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    details = {"selected_residuals": {"default": residuals_default, "clean": residuals_clean}, "models": {}}
    best_name = None
    best_score = -1.0
    best_df = None
    best_feats = None
    best_preds = None
    for name, (dataset, features, kind) in model_specs.items():
        metrics, preds = evaluate_lofo(dataset, features, kind)
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
            best_df = dataset
            best_feats = features
            best_preds = preds

    summary_df = pd.DataFrame(summaries).sort_values("lofo_macro_f1_informative", ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    assert best_df is not None and best_feats is not None and best_preds is not None and best_name is not None
    audit = retroviral_mechanism_audit(best_df, best_preds, best_feats)
    with open(output_dir / "retroviral_audit.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, **audit}, f, indent=2)
    with open(output_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    gate_default.to_csv(output_dir / "gate_scores_default.csv", index=False)
    gate_clean.to_csv(output_dir / "gate_scores_clean.csv", index=False)
    return {"summary": summary_df.to_dict(orient="records"), "best_model": best_name, "best_score": best_score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run biology-intensive search around the frozen benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bio_intensive"))
    args = parser.parse_args()
    result = run(args.output_dir)
    print(pd.DataFrame(result["summary"]).to_string(index=False, float_format=lambda x: f"{x:.3f}" if x is not None else "nan"))
    print(f"\nBest model: {result['best_model']} score={result['best_score']:.3f}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
