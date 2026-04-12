#!/usr/bin/env python3
"""
Novel Approach Experiment — Retroviral Wall Challenge
=====================================================

Five distinct novel methods, each independently runnable.
Benchmark to beat: hybrid_lr, LOFO informative macro-F1 = 0.7217

Methods
-------
  BASELINE  — Frozen winner replica (hybrid_lr) for in-process comparison.

  METHOD 1  — ESM Residualization
    Extracts family-deconfounded features from the provided ESM-2 embeddings
    (1280-d) via PCA + within-fold family-mean subtraction.  This is the first
    experiment in this project to use the ESM-2 embeddings in a cross-family
    generalization setting.  The residualized PCs are concatenated with the
    current winner's 33 features and fit with a regularized logistic regression.

  METHOD 2  — Nested LOFO Feature Selection
    Fixes the known global-selection information leak in the current winner:
    the handcrafted feature scoring is re-run inside each LOFO fold using
    only the training families.  The selected feature set may differ per fold.
    Everything else matches the winning recipe.

  METHOD 3  — Soft-Voting Ensemble
    Averages predicted probabilities from three models with different inductive
    biases within each LOFO fold:
      A) LR on the frozen 33 features (current winner approach)
      B) LR on gate scores + 15 family-residualized ESM PCs  (Method 1 variant)
      C) SVM-RBF on the full 88 non-Foldseek handcrafted features
    Ensemble diversity typically improves robustness without adding parameters.

  METHOD 4  — Feature Interactions + ElasticNet
    Adds 36 explicit gate × handcrafted interaction terms to the frozen 33
    features, then uses an ElasticNet logistic regression to select a sparse
    subset.  Captures synergistic relationships (e.g., processivity × triad)
    that a linear model cannot express.

  METHOD 5  — Combined Novel Pipeline
    Integrates the three strongest ideas: nested feature selection (Method 2),
    family-residualized ESM PCs (Method 1), and gate × handcrafted interaction
    terms (Method 4), all fitted by an ElasticNet logistic regression.

Outputs → outputs/novel_approach/
    summary.csv                   — all methods vs benchmark
    details.json                  — per-family breakdown
    <method>_predictions.csv      — per-sample LOFO predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFORMATIVE_FAMILIES = ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]

# Gate score column names produced by the mechanistic pipeline.
GATE_SCORE_COLS = [
    "foldability_score",
    "fusion_compat_score",
    "fusion_compat_clash_score",
    "substrate_binding_score",
    "catalytic_score",
    "processivity_score",
]

# The 27 handcrafted features selected by the winning workflow (from
# outputs/generalization/details.json).  Used as the frozen handcrafted set
# in methods that build on top of the winner instead of replacing it.
FROZEN_HAND_COLS = [
    "ramachandran_outliers",
    "ramachandran_allowed",
    "pocket_hbonds_per_res",
    "ramachandran_favoured",
    "salt_per_res",
    "hydrophobic_per_res",
    "pocket_hbonds",
    "sasa_low_pct",
    "n_salt_bridges",
    "sasa_avg",
    "sasa_total",
    "hbonds_per_res",
    "thumb_charge_class_num",
    "perplexity",
    "avg_log_likelihood",
    "thumb_fident",
    "triad_found_bin",
    "triad_best_rmsd",
    "thumb_total_residues",
    "pct_E",
    "aromaticity",
    "pct_H",
    "D1_D2_dist",
    "best_s1_len",
    "molecular_weight",
    "lysine_k_charge",
    "aspartate_d_charge",
]

# Frozen benchmark score to beat.
FROZEN_BENCHMARK = 0.7217

# ---------------------------------------------------------------------------
# Shared utilities  (standalone; no import from generalization_workflow.py)
# ---------------------------------------------------------------------------


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def optimise_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.1, 0.91, 0.02):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def compute_metrics(pred_df: pd.DataFrame) -> dict:
    y_true = pred_df["active"].to_numpy()
    y_pred = pred_df["predicted_active"].to_numpy()
    y_score = pred_df["predicted_score"].to_numpy()
    informative, per_family = [], {}
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
    return {
        "lofo_macro_f1_informative": float(np.mean(informative)) if informative else 0.0,
        "overall_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "overall_auc": safe_auc(y_true, y_score),
        "retroviral_tp_of_12": per_family.get("Retroviral", {}).get("tp", 0),
        "per_family": per_family,
    }


def load_data(data_dir: Path, gate_score_path: Path) -> pd.DataFrame:
    sequences = pd.read_csv(data_dir / "rt_sequences.csv")
    handcrafted = pd.read_csv(data_dir / "handcrafted_features.csv")
    gates = pd.read_csv(gate_score_path)
    return (
        sequences.merge(handcrafted, on="rt_name", how="inner")
        .merge(gates, on="rt_name", how="inner")
    )


def load_esm(esm_path: Path) -> dict[str, np.ndarray]:
    """Return {rt_name: embedding_1280d}."""
    data = np.load(esm_path, allow_pickle=True)
    return {str(name): emb for name, emb in zip(data["names"], data["embeddings"])}


def resolve_gate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in GATE_SCORE_COLS if c in df.columns]


def resolve_hand_cols(frozen_cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in frozen_cols if c in df.columns]


def _to_float_array(df_subset: pd.DataFrame) -> np.ndarray:
    return df_subset.apply(pd.to_numeric, errors="coerce").values.astype(float)


def impute_scale(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute with training stats, then z-score with training mean/std."""
    medians = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_train - mu) / std, (X_test - mu) / std


# ---------------------------------------------------------------------------
# Generic LOFO runner
# feature_fn(df_train, df_test) -> (X_train_array, X_test_array)
# model_fn()                    -> unfitted classifier with predict_proba
# ---------------------------------------------------------------------------


def run_lofo(
    df: pd.DataFrame,
    feature_fn: Callable,
    model_fn: Callable,
    label: str,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)

        X_train, X_test = feature_fn(df_train, df_test)
        y_train = df_train["active"].astype(int).to_numpy()

        clf = model_fn()
        clf.fit(X_train, y_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = label
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 0 — BASELINE  (replica of the frozen winner, hybrid_lr)
# Run inside this script so we can compare against the same data loading path.
# ---------------------------------------------------------------------------


def run_baseline(df: pd.DataFrame, gate_cols: list[str], hand_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    """
    Replica of the current winning hybrid_lr model.
    Uses global feature selection (same as generalization_workflow.py).
    Serves as the reference point within this experiment file.
    """
    feature_cols = gate_cols + hand_cols

    def feature_fn(df_train, df_test):
        X_train = _to_float_array(df_train[feature_cols])
        X_test = _to_float_array(df_test[feature_cols])
        return impute_scale(X_train, X_test)

    def model_fn():
        return LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )

    return run_lofo(df, feature_fn, model_fn, label="baseline_hybrid_lr")


# ---------------------------------------------------------------------------
# METHOD 1 — ESM Residualization
#
# Within each LOFO fold:
#   1. Fit PCA on the training ESM-2 embeddings (1280-d → n_components PCs).
#   2. Subtract the per-family mean PC vector (computed on training families)
#      from each training sample — removing variance attributable to
#      evolutionary family membership.
#   3. For the held-out (test) family, subtract the training grand-mean PC
#      vector as the best available family-agnostic baseline.
#   4. Concatenate the residualized PCs with the frozen 33 features and fit
#      a regularized logistic regression (stronger C due to extra features).
#
# Scientific rationale: ESM-2 encodes deep sequence-structure-function signal.
# The two active LTR_Retrotransposons (Tf1, Ty3) share active-site chemistry
# with retroviral RTs; family residualization exposes this shared functional
# embedding while suppressing the "looks like an LTR" phylogenetic signal.
# ---------------------------------------------------------------------------


def _build_esm_residualized_pcs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    n_components: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return family-residualized ESM PCA features for train and test folds.
    Shape: (n_train, n_components) and (n_test, n_components).
    """
    train_names = df_train["rt_name"].tolist()
    test_names = df_test["rt_name"].tolist()

    # Stack raw embeddings (skip any missing entries gracefully)
    esm_train = np.array([esm_dict.get(n, np.zeros(1280)) for n in train_names])
    esm_test = np.array([esm_dict.get(n, np.zeros(1280)) for n in test_names])

    # PCA fitted on training fold only
    n_comp = min(n_components, len(train_names) - 1, esm_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    esm_train_pca = pca.fit_transform(esm_train)   # (n_train, n_comp)
    esm_test_pca = pca.transform(esm_test)           # (n_test,  n_comp)

    # Per-family mean in PC space (training families only)
    fam_means: dict[str, np.ndarray] = {}
    for fam in df_train["rt_family"].unique():
        mask = (df_train["rt_family"] == fam).to_numpy()
        fam_means[fam] = esm_train_pca[mask].mean(axis=0)

    grand_mean = esm_train_pca.mean(axis=0)

    # Subtract family mean from training samples
    esm_train_resid = esm_train_pca - np.array(
        [fam_means[f] for f in df_train["rt_family"]]
    )
    # Subtract grand mean from test samples (family mean unknown at test time)
    esm_test_resid = esm_test_pca - grand_mean

    return esm_train_resid, esm_test_resid


def run_method1_esm_residualized(
    df: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    gate_cols: list[str],
    hand_cols: list[str],
    n_esm_components: int = 15,
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 1: ESM Residualization.

    Features per fold:
        gate_cols (6) + hand_cols (27) + family-residualized ESM PCs (15) = 48
    Model: LogisticRegression with C=0.1 (stronger regularisation for extra PCs).
    """
    base_cols = gate_cols + hand_cols

    def feature_fn(df_train, df_test):
        X_base_train = _to_float_array(df_train[base_cols])
        X_base_test = _to_float_array(df_test[base_cols])
        X_base_train, X_base_test = impute_scale(X_base_train, X_base_test)

        esm_train_resid, esm_test_resid = _build_esm_residualized_pcs(
            df_train, df_test, esm_dict, n_components=n_esm_components
        )
        # ESM residuals are already zero-centred (by construction); scale only.
        esm_std = esm_train_resid.std(axis=0)
        esm_std = np.where(esm_std < 1e-8, 1.0, esm_std)
        esm_train_resid = esm_train_resid / esm_std
        esm_test_resid = esm_test_resid / esm_std

        return np.hstack([X_base_train, esm_train_resid]), np.hstack([X_base_test, esm_test_resid])

    def model_fn():
        return LogisticRegression(
            C=0.1, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )

    return run_lofo(df, feature_fn, model_fn, label="method1_esm_residualized")


# ---------------------------------------------------------------------------
# METHOD 2 — Nested LOFO Feature Selection
#
# The current winner selects handcrafted features using statistics computed on
# the full dataset (all 57 samples).  This means the held-out family's data
# influences which features are selected — a subtle but real information leak.
#
# Fix: re-run the leakage-aware scoring inside each LOFO fold using only the
# training families.  The selected feature set may differ across folds.
# Everything else (model, threshold, evaluation) is identical to the winner.
# ---------------------------------------------------------------------------


def _score_handcrafted_features(df: pd.DataFrame, hand_cols: list[str]) -> pd.DataFrame:
    """
    Score handcrafted features by leakage-adjusted signal.
    Exact replica of select_low_leakage_features() in generalization_workflow.py,
    run on an arbitrary subset of rows (the training fold).
    """
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
        family_mi = float(
            mutual_info_classif(
                vals.to_frame(), fam_codes, discrete_features=False, random_state=42
            )[0]
        )
        within_aucs = []
        for fam in INFORMATIVE_FAMILIES:
            mask = df["rt_family"] == fam
            if mask.sum() < 2:
                continue
            fam_auc = safe_auc(y[mask.to_numpy()], vals.loc[mask].to_numpy())
            if fam_auc is not None:
                within_aucs.append(max(fam_auc, 1.0 - fam_auc))
        stable_signal = float(np.mean(within_aucs)) if within_aucs else 0.5
        rows.append(
            {
                "feature": col,
                "active_auc_abs": active_auc_abs,
                "family_mi": family_mi,
                "stable_signal": stable_signal,
                "score": stable_signal + 0.5 * active_auc_abs - 0.75 * family_mi,
                "is_foldseek": col.startswith("foldseek_"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "active_auc_abs", "family_mi", "stable_signal", "score", "is_foldseek"])
    report = pd.DataFrame(rows).sort_values("score", ascending=False)
    selected = report[
        (report["family_mi"] <= 0.55)
        & (report["active_auc_abs"] >= 0.60)
        & (report["stable_signal"] >= 0.55)
        & (~report["is_foldseek"])
    ]
    if len(selected) < 8:
        selected = report[~report["is_foldseek"]].head(min(12, len(report)))
    return selected


def run_method2_nested_lofo(
    df: pd.DataFrame,
    gate_cols: list[str],
    all_hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 2: Nested LOFO Feature Selection.

    Feature selection is re-run on training families only inside each fold.
    The selected handcrafted features + gate scores form the feature set.
    Model: LogisticRegression with C=0.3 (same as winner).
    """
    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)

        # Feature selection on training fold only (the core change vs. winner)
        selected = _score_handcrafted_features(df_train, all_hand_cols)
        fold_hand_cols = [c for c in selected["feature"].tolist() if c in df.columns]
        feature_cols = gate_cols + fold_hand_cols

        X_train = _to_float_array(df_train[feature_cols])
        X_test = _to_float_array(df_test[feature_cols])
        X_train, X_test = impute_scale(X_train, X_test)

        y_train = df_train["active"].astype(int).to_numpy()
        clf = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf.fit(X_train, y_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                    "n_features": len(feature_cols),
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method2_nested_lofo"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 3 — Soft-Voting Ensemble
#
# Three models with different inductive biases are trained on each LOFO fold.
# Their predicted probabilities are averaged (soft voting) before thresholding.
# Ensemble diversity reduces variance and tends to be more robust under
# distribution shift (cross-family generalization) than any single model.
#
# Model A: LR on 33 frozen features          — current winner's recipe
# Model B: LR on 6 gate scores + ESM PCs     — ESM-only augmentation
# Model C: SVM-RBF on 88 non-Foldseek feats  — different kernel, more features
# ---------------------------------------------------------------------------


def run_method3_ensemble(
    df: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    gate_cols: list[str],
    hand_cols: list[str],
    all_hand_cols: list[str],
    n_esm_components: int = 15,
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 3: Soft-Voting Ensemble of three diverse models.

    Final score = mean(prob_A, prob_B, prob_C).
    Threshold is optimised on the ensemble score using training labels.
    """
    # Non-Foldseek handcrafted columns for Model C
    no_foldseek_cols = [c for c in all_hand_cols if not c.startswith("foldseek_") and c in df.columns]
    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        # --- Model A: frozen hybrid LR ---
        Xa_tr = _to_float_array(df_train[base_cols])
        Xa_te = _to_float_array(df_test[base_cols])
        Xa_tr, Xa_te = impute_scale(Xa_tr, Xa_te)
        clf_a = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf_a.fit(Xa_tr, y_train)
        scores_a_train = clf_a.predict_proba(Xa_tr)[:, 1]
        scores_a_test = clf_a.predict_proba(Xa_te)[:, 1]

        # --- Model B: gate scores + ESM PCs LR ---
        esm_tr_resid, esm_te_resid = _build_esm_residualized_pcs(
            df_train, df_test, esm_dict, n_components=n_esm_components
        )
        esm_std = esm_tr_resid.std(axis=0)
        esm_std = np.where(esm_std < 1e-8, 1.0, esm_std)
        esm_tr_scaled = esm_tr_resid / esm_std
        esm_te_scaled = esm_te_resid / esm_std

        Xg_tr = _to_float_array(df_train[gate_cols])
        Xg_te = _to_float_array(df_test[gate_cols])
        Xg_tr, Xg_te = impute_scale(Xg_tr, Xg_te)

        Xb_tr = np.hstack([Xg_tr, esm_tr_scaled])
        Xb_te = np.hstack([Xg_te, esm_te_scaled])
        clf_b = LogisticRegression(
            C=0.1, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf_b.fit(Xb_tr, y_train)
        scores_b_train = clf_b.predict_proba(Xb_tr)[:, 1]
        scores_b_test = clf_b.predict_proba(Xb_te)[:, 1]

        # --- Model C: SVM-RBF on all non-Foldseek handcrafted ---
        Xc_tr = _to_float_array(df_train[no_foldseek_cols])
        Xc_te = _to_float_array(df_test[no_foldseek_cols])
        Xc_tr, Xc_te = impute_scale(Xc_tr, Xc_te)
        clf_c = SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True,
            class_weight="balanced", random_state=42,
        )
        clf_c.fit(Xc_tr, y_train)
        scores_c_train = clf_c.predict_proba(Xc_tr)[:, 1]
        scores_c_test = clf_c.predict_proba(Xc_te)[:, 1]

        # --- Soft vote: average probabilities ---
        ensemble_train = (scores_a_train + scores_b_train + scores_c_train) / 3.0
        ensemble_test = (scores_a_test + scores_b_test + scores_c_test) / 3.0
        threshold = optimise_threshold(ensemble_train, y_train)

        for rt_name, score in zip(df_test["rt_name"], ensemble_test):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method3_ensemble"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 4 — Feature Interactions + ElasticNet
#
# A linear model cannot capture "processivity × triad_found" or
# "catalytic × substrate_binding" synergies.  This method adds explicit
# pairwise interaction terms between the 6 gate scores and the top 6
# handcrafted features (by leakage-adjusted score order), yielding 36
# additional terms.  Combined with the original 33 features: 69 total.
# ElasticNet regularisation (L1+L2) selects a sparse subset.
# ---------------------------------------------------------------------------


def _add_gate_hand_interactions(
    X_gate: np.ndarray,
    X_hand: np.ndarray,
    top_hand_k: int = 6,
) -> np.ndarray:
    """
    Build gate × top_hand_k interaction terms.
    Returns array of shape (n_samples, n_gate * top_hand_k).
    """
    X_top_hand = X_hand[:, :top_hand_k]
    n_gate = X_gate.shape[1]
    parts = [X_gate[:, g:g+1] * X_top_hand for g in range(n_gate)]
    return np.hstack(parts)  # (n, n_gate * top_hand_k)


def run_method4_interactions_elasticnet(
    df: pd.DataFrame,
    gate_cols: list[str],
    hand_cols: list[str],
    top_hand_k: int = 6,
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 4: Gate × Handcrafted Interactions + ElasticNet.

    Features per fold:
        base (33) + interactions (6 gates × 6 hand = 36) = 69 total
    Model: ElasticNet logistic regression (L1+L2, l1_ratio=0.5, C=0.15).
    """
    base_cols = gate_cols + hand_cols

    def feature_fn(df_train, df_test):
        X_tr = _to_float_array(df_train[base_cols])
        X_te = _to_float_array(df_test[base_cols])
        X_tr, X_te = impute_scale(X_tr, X_te)

        n_gate = len(gate_cols)
        inter_tr = _add_gate_hand_interactions(X_tr[:, :n_gate], X_tr[:, n_gate:], top_hand_k)
        inter_te = _add_gate_hand_interactions(X_te[:, :n_gate], X_te[:, n_gate:], top_hand_k)

        return np.hstack([X_tr, inter_tr]), np.hstack([X_te, inter_te])

    def model_fn():
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.15,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )

    return run_lofo(df, feature_fn, model_fn, label="method4_interactions_elasticnet")


# ---------------------------------------------------------------------------
# METHOD 5 — Combined Novel Pipeline
#
# Integrates the three core ideas:
#   1. Nested LOFO feature selection  (re-run inside each fold)
#   2. Family-residualized ESM PCs    (15 components, within-fold PCA)
#   3. Gate × selected-hand interactions  (36 terms, sparse selection)
# Combined with an ElasticNet logistic regression.
#
# This is the main novel bet: it addresses the information leak, adds the
# first ESM signal, and enriches feature interactions — all in a single model.
# ---------------------------------------------------------------------------


def run_method5_combined(
    df: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    gate_cols: list[str],
    all_hand_cols: list[str],
    n_esm_components: int = 15,
    top_hand_k: int = 6,
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 5: Combined Novel Pipeline.

    Per fold:
        1. Select handcrafted features on training families only.
        2. Build family-residualized ESM PCA features.
        3. Add gate × top-k handcrafted interaction terms.
        4. Fit ElasticNet LR on all combined features.
    """
    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        # Step 1: nested feature selection on training fold
        selected = _score_handcrafted_features(df_train, all_hand_cols)
        fold_hand_cols = [c for c in selected["feature"].tolist() if c in df.columns]
        # Ensure at least the frozen cols are candidates
        if len(fold_hand_cols) < 6:
            fold_hand_cols = [c for c in FROZEN_HAND_COLS if c in df.columns]

        feature_cols = gate_cols + fold_hand_cols

        # Step 2: base features (imputed + scaled)
        X_base_tr = _to_float_array(df_train[feature_cols])
        X_base_te = _to_float_array(df_test[feature_cols])
        X_base_tr, X_base_te = impute_scale(X_base_tr, X_base_te)

        # Step 3: ESM residualized PCs
        esm_tr_resid, esm_te_resid = _build_esm_residualized_pcs(
            df_train, df_test, esm_dict, n_components=n_esm_components
        )
        esm_std = esm_tr_resid.std(axis=0)
        esm_std = np.where(esm_std < 1e-8, 1.0, esm_std)
        esm_tr_scaled = esm_tr_resid / esm_std
        esm_te_scaled = esm_te_resid / esm_std

        # Step 4: gate × top-k handcrafted interaction terms
        n_gate = len(gate_cols)
        inter_tr = _add_gate_hand_interactions(
            X_base_tr[:, :n_gate], X_base_tr[:, n_gate:], top_hand_k
        )
        inter_te = _add_gate_hand_interactions(
            X_base_te[:, :n_gate], X_base_te[:, n_gate:], top_hand_k
        )

        X_tr = np.hstack([X_base_tr, esm_tr_scaled, inter_tr])
        X_te = np.hstack([X_base_te, esm_te_scaled, inter_te])

        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.08,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        )
        clf.fit(X_tr, y_train)
        train_scores = clf.predict_proba(X_tr)[:, 1]
        test_scores = clf.predict_proba(X_te)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                    "n_features": X_tr.shape[1],
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method5_combined"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 6 — ESM Cosine-Similarity Feature
#
# Rather than adding 15 PCA components (noisy with n=57), this method distils
# the entire 1280-d ESM embedding into a SINGLE scalar feature per sample:
#
#   esm_active_bias = cos_sim(embedding, active_centroid)
#                   - cos_sim(embedding, inactive_centroid)
#
# Both centroids are computed from the training fold only (no leakage).
# The feature is positive for samples whose ESM embedding resembles the
# "average active RT" and negative for samples resembling the "average
# inactive RT".
#
# Motivation for beating the LTR bottleneck:
#   The two active LTR_Retrotransposons (Tf1-RT: 34% efficiency, Ty3-RT: 9%)
#   share a canonical YxDD catalytic motif with retroviral RTs.  The ESM-2
#   model, trained on 250M protein sequences, should encode this shared
#   function in its embeddings.  When LTR is held out, the training active
#   centroid is dominated by 12 retroviral RTs; Tf1 and Ty3 should lie closer
#   to that centroid than the 9 inactive LTR elements (LINE-1, Gypsy, etc.).
# ---------------------------------------------------------------------------


def _compute_esm_cosine_feature(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ESM cosine active-bias feature for train and test folds.
    Returns column arrays of shape (n_train, 1) and (n_test, 1).
    """
    train_names = df_train["rt_name"].tolist()
    test_names = df_test["rt_name"].tolist()
    y_train = df_train["active"].astype(int).to_numpy()

    esm_train = np.array([esm_dict.get(n, np.zeros(1280)) for n in train_names])
    esm_test = np.array([esm_dict.get(n, np.zeros(1280)) for n in test_names])

    # Unit-normalise every embedding (cosine similarity = dot product on unit vectors)
    def row_norm(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms < 1e-8, 1.0, norms)

    esm_train_n = row_norm(esm_train)
    esm_test_n = row_norm(esm_test)

    active_mask = y_train == 1
    if active_mask.sum() == 0 or (~active_mask).sum() == 0:
        return np.zeros((len(train_names), 1)), np.zeros((len(test_names), 1))

    # Centroids (normalised so dot product = cosine similarity)
    active_c = esm_train_n[active_mask].mean(axis=0)
    active_c /= max(np.linalg.norm(active_c), 1e-8)
    inactive_c = esm_train_n[~active_mask].mean(axis=0)
    inactive_c /= max(np.linalg.norm(inactive_c), 1e-8)

    esm_train_score = (esm_train_n @ active_c - esm_train_n @ inactive_c).reshape(-1, 1)
    esm_test_score = (esm_test_n @ active_c - esm_test_n @ inactive_c).reshape(-1, 1)
    return esm_train_score, esm_test_score


def run_method6_esm_cosine(
    df: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 6: ESM Cosine-Similarity Feature.

    Features per fold:
        gate_cols (6) + hand_cols (27) + esm_cosine_bias (1) = 34 total
    Model: LogisticRegression with C=0.3 (identical to baseline; single new feature).
    """
    base_cols = gate_cols + hand_cols

    def feature_fn(df_train, df_test):
        X_base_tr = _to_float_array(df_train[base_cols])
        X_base_te = _to_float_array(df_test[base_cols])
        X_base_tr, X_base_te = impute_scale(X_base_tr, X_base_te)

        esm_tr, esm_te = _compute_esm_cosine_feature(df_train, df_test, esm_dict)
        # Scale ESM feature with training stats
        esm_mu = esm_tr.mean()
        esm_std = max(esm_tr.std(), 1e-8)
        esm_tr = (esm_tr - esm_mu) / esm_std
        esm_te = (esm_te - esm_mu) / esm_std

        return np.hstack([X_base_tr, esm_tr]), np.hstack([X_base_te, esm_te])

    def model_fn():
        return LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )

    return run_lofo(df, feature_fn, model_fn, label="method6_esm_cosine")


# ---------------------------------------------------------------------------
# METHOD 7 — Two-Model Soft Ensemble (Baseline + ESM Cosine)
#
# Averages predicted probabilities from just two complementary models:
#   Model A: baseline hybrid LR (33 features, no ESM) — strong Retroviral
#   Model B: hybrid LR + ESM cosine feature (34 features) — ESM signal
#
# Removing the SVM (which hurt Retron in Method 3) and using only a single
# ESM feature keeps the ensemble tight and avoids high-variance components.
# The two models are trained independently on the same training fold and their
# probability outputs are averaged before threshold optimisation.
# ---------------------------------------------------------------------------


def run_method7_two_model_ensemble(
    df: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 7: Two-Model Soft Ensemble (Baseline LR + ESM-Cosine LR).

    Final score = 0.5 * prob_A + 0.5 * prob_B.
    Both models share the same 33 frozen features; Model B adds one ESM feature.
    """
    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        # --- Model A: baseline LR (33 features) ---
        Xa_tr = _to_float_array(df_train[base_cols])
        Xa_te = _to_float_array(df_test[base_cols])
        Xa_tr, Xa_te = impute_scale(Xa_tr, Xa_te)
        clf_a = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf_a.fit(Xa_tr, y_train)
        scores_a_tr = clf_a.predict_proba(Xa_tr)[:, 1]
        scores_a_te = clf_a.predict_proba(Xa_te)[:, 1]

        # --- Model B: baseline + ESM cosine feature (34 features) ---
        esm_tr, esm_te = _compute_esm_cosine_feature(df_train, df_test, esm_dict)
        esm_mu, esm_std = esm_tr.mean(), max(esm_tr.std(), 1e-8)
        esm_tr = (esm_tr - esm_mu) / esm_std
        esm_te = (esm_te - esm_mu) / esm_std
        Xb_tr = np.hstack([Xa_tr, esm_tr])
        Xb_te = np.hstack([Xa_te, esm_te])
        clf_b = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf_b.fit(Xb_tr, y_train)
        scores_b_tr = clf_b.predict_proba(Xb_tr)[:, 1]
        scores_b_te = clf_b.predict_proba(Xb_te)[:, 1]

        # --- Soft vote ---
        ens_tr = (scores_a_tr + scores_b_tr) / 2.0
        ens_te = (scores_a_te + scores_b_te) / 2.0
        threshold = optimise_threshold(ens_tr, y_train)

        for rt_name, score in zip(df_test["rt_name"], ens_te):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method7_two_model_ensemble"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 8 — ESM Cosine + C-Grid Search
#
# The ESM cosine feature alone (Method 6) may require a different
# regularisation strength than the baseline C=0.3.  This method searches
# C in {0.05, 0.1, 0.2, 0.3, 0.5, 1.0} using 5-fold cross-validation on the
# training fold and selects the best C before predicting on the test fold.
# This is a strictly within-fold hyperparameter search (no test leakage).
# ---------------------------------------------------------------------------


def run_method8_esm_cosine_tuned(
    df: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 8: ESM Cosine Feature + C-Grid Search (within fold).

    Features: 34 (33 frozen + 1 ESM cosine).
    C selected by 5-fold CV F1 on training fold, from {0.05, 0.1, 0.2, 0.3, 0.5, 1.0}.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    C_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    base_cols = gate_cols + hand_cols

    rows = []
    chosen_c_per_fold: dict[str, float] = {}
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        # Build feature matrices
        X_base_tr = _to_float_array(df_train[base_cols])
        X_base_te = _to_float_array(df_test[base_cols])
        X_base_tr, X_base_te = impute_scale(X_base_tr, X_base_te)

        esm_tr, esm_te = _compute_esm_cosine_feature(df_train, df_test, esm_dict)
        esm_mu, esm_std = esm_tr.mean(), max(esm_tr.std(), 1e-8)
        esm_tr = (esm_tr - esm_mu) / esm_std
        esm_te = (esm_te - esm_mu) / esm_std

        X_tr = np.hstack([X_base_tr, esm_tr])
        X_te = np.hstack([X_base_te, esm_te])

        # Inner 5-fold CV to select C (only on training fold)
        inner_cv = StratifiedKFold(n_splits=min(5, int(y_train.sum())), shuffle=True, random_state=42)
        best_c, best_inner_f1 = C_grid[0], -1.0
        for c in C_grid:
            inner_f1s = []
            for tr_idx, val_idx in inner_cv.split(X_tr, y_train):
                clf = LogisticRegression(
                    C=c, class_weight="balanced", solver="liblinear",
                    max_iter=5000, random_state=42,
                )
                clf.fit(X_tr[tr_idx], y_train[tr_idx])
                val_scores = clf.predict_proba(X_tr[val_idx])[:, 1]
                t = optimise_threshold(
                    clf.predict_proba(X_tr[tr_idx])[:, 1], y_train[tr_idx]
                )
                inner_f1s.append(f1_score(y_train[val_idx], (val_scores >= t).astype(int), zero_division=0))
            mean_f1 = float(np.mean(inner_f1s))
            if mean_f1 > best_inner_f1:
                best_inner_f1, best_c = mean_f1, c
        chosen_c_per_fold[held_out] = best_c

        # Fit final model with best C on full training fold
        clf_final = LogisticRegression(
            C=best_c, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf_final.fit(X_tr, y_train)
        train_scores = clf_final.predict_proba(X_tr)[:, 1]
        test_scores = clf_final.predict_proba(X_te)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                    "chosen_c": best_c,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method8_esm_cosine_tuned"
    metrics["chosen_c_per_fold"] = chosen_c_per_fold
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 9 — Efficiency-Weighted Logistic Regression
#
# All prior work uses binary `active` labels.  The dataset also provides
# `pe_efficiency_pct` (0 for inactive; 1.5–41% for actives).  This method
# uses the continuous efficiency as a sample weight during training:
#
#   weight = 1.0                          if active == 0  (inactive)
#   weight = max(pe_efficiency_pct, 1.0)  if active == 1  (active, clipped)
#
# This makes the model learn more strongly from high-efficiency RTs
# (MMLV-RT: 41%, KORV-RT: 26.5%, Tf1-RT: 34%) and less from barely-active
# ones (Ne144-RT: 0.5%, Ec48-RT: 1.5%).  The hypothesis is that
# high-efficiency actives have more extreme feature profiles and learning
# from them more strongly will improve generalisation to new active RTs.
#
# Note: pe_efficiency_pct is part of the provided challenge data and is
# visible for all 57 labeled samples.  Using it as a sample weight during
# training (not as a feature) is fully within scope.
# ---------------------------------------------------------------------------


def run_method9_efficiency_weighted(
    df: pd.DataFrame,
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 9: Efficiency-Weighted LR.

    Features: same frozen 33 as baseline.
    Model: same LR as baseline but active samples weighted by pe_efficiency_pct.
    """
    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        X_tr = _to_float_array(df_train[base_cols])
        X_te = _to_float_array(df_test[base_cols])
        X_tr, X_te = impute_scale(X_tr, X_te)

        # Sample weights: inactive=1, active=max(efficiency, 1.0)
        eff = df_train["pe_efficiency_pct"].fillna(0.0).to_numpy()
        weights = np.where(y_train == 1, np.maximum(eff, 1.0), 1.0)

        clf = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf.fit(X_tr, y_train, sample_weight=weights)
        train_scores = clf.predict_proba(X_tr)[:, 1]
        test_scores = clf.predict_proba(X_te)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method9_efficiency_weighted"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 10 — Efficiency-Weighted + ESM Cosine Feature
#
# Combines Methods 6 and 9: efficiency-weighted training + the ESM cosine
# active-bias feature.  The ESM cosine feature anchors each test sample to
# training actives in embedding space, while efficiency weighting ensures
# the most discriminative actives dominate the learned weights.
# ---------------------------------------------------------------------------


def run_method10_efficiency_esm(
    df: pd.DataFrame,
    esm_dict: dict[str, np.ndarray],
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 10: Efficiency-Weighted LR + ESM Cosine Feature.

    Features: 34 (33 frozen + 1 ESM cosine).
    Training: active samples weighted by max(pe_efficiency_pct, 1.0).
    """
    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        X_base_tr = _to_float_array(df_train[base_cols])
        X_base_te = _to_float_array(df_test[base_cols])
        X_base_tr, X_base_te = impute_scale(X_base_tr, X_base_te)

        esm_tr, esm_te = _compute_esm_cosine_feature(df_train, df_test, esm_dict)
        esm_mu, esm_std = esm_tr.mean(), max(esm_tr.std(), 1e-8)
        esm_tr = (esm_tr - esm_mu) / esm_std
        esm_te = (esm_te - esm_mu) / esm_std

        X_tr = np.hstack([X_base_tr, esm_tr])
        X_te = np.hstack([X_base_te, esm_te])

        eff = df_train["pe_efficiency_pct"].fillna(0.0).to_numpy()
        weights = np.where(y_train == 1, np.maximum(eff, 1.0), 1.0)

        clf = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf.fit(X_tr, y_train, sample_weight=weights)
        train_scores = clf.predict_proba(X_tr)[:, 1]
        test_scores = clf.predict_proba(X_te)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method10_efficiency_esm"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 11 — Balanced-Accuracy Threshold
#
# The current winner optimises the binary F1 on training predictions.
# When the training class balance differs substantially from the test class
# balance (e.g., training 23% active but Retroviral test 67% active),
# the F1-optimal threshold is systematically biased toward the training
# prevalence.  Balanced accuracy = (sensitivity + specificity) / 2 is
# symmetric and less sensitive to class imbalance.  A threshold that
# maximises balanced accuracy on training predictions tends to be LOWER
# than the F1-optimal threshold, which may catch more of the high-prevalence
# Retroviral positives.  Same 33-feature hybrid LR; only the threshold
# strategy changes.
# ---------------------------------------------------------------------------


def _optimise_threshold_balanced(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold maximising balanced accuracy = (sensitivity + specificity) / 2."""
    best_bacc, best_t = -1.0, 0.5
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        sensitivity = ((preds == 1) & (labels == 1)).sum() / n_pos
        specificity = ((preds == 0) & (labels == 0)).sum() / n_neg
        bacc = (sensitivity + specificity) / 2.0
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    return best_t


def run_method11_balanced_threshold(
    df: pd.DataFrame,
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 11: Balanced-Accuracy Threshold on the frozen hybrid LR.

    Same model and features as the baseline (33 features, C=0.3).
    Only the threshold selection strategy changes: maximise balanced accuracy
    instead of F1 on training predictions.
    """
    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        X_tr = _to_float_array(df_train[base_cols])
        X_te = _to_float_array(df_test[base_cols])
        X_tr, X_te = impute_scale(X_tr, X_te)

        clf = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf.fit(X_tr, y_train)
        train_scores = clf.predict_proba(X_tr)[:, 1]
        test_scores = clf.predict_proba(X_te)[:, 1]

        # Balanced accuracy threshold instead of F1-optimal
        threshold = _optimise_threshold_balanced(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method11_balanced_threshold"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 12 — Sqrt-Scaled Efficiency Weighting + Balanced Threshold
#
# Method 9 uses raw efficiency as weight (range 1–41 for actives), which may
# create too extreme a gap between high- and low-efficiency actives.  This
# method uses sqrt(max(efficiency, 0.5)) to compress the weight range while
# preserving rank order.  Combined with balanced-accuracy threshold
# (which tends to be lower, catching more actives at test time).
# ---------------------------------------------------------------------------


def run_method12_sqrt_eff_balanced(
    df: pd.DataFrame,
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 12: Sqrt-Scaled Efficiency Weighting + Balanced-Accuracy Threshold.

    weight(active) = sqrt(max(pe_efficiency_pct, 0.5)) + 1
    weight(inactive) = 1.0
    Threshold: maximise balanced accuracy on training predictions.
    """
    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        X_tr = _to_float_array(df_train[base_cols])
        X_te = _to_float_array(df_test[base_cols])
        X_tr, X_te = impute_scale(X_tr, X_te)

        eff = df_train["pe_efficiency_pct"].fillna(0.0).to_numpy()
        weights = np.where(y_train == 1, np.sqrt(np.maximum(eff, 0.5)) + 1.0, 1.0)

        clf = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf.fit(X_tr, y_train, sample_weight=weights)
        train_scores = clf.predict_proba(X_tr)[:, 1]
        test_scores = clf.predict_proba(X_te)[:, 1]
        threshold = _optimise_threshold_balanced(train_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method12_sqrt_eff_balanced"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 13 — Ordinal Regression on pe_efficiency_pct (Ridge)
#
# All prior methods treat activity as binary.  This method treats it as an
# ORDINAL/CONTINUOUS problem: train a Ridge regression directly on the
# normalised `pe_efficiency_pct` values (0 for inactive, 0–1 for active,
# where 1 = max efficiency in training).  The regression score is then
# thresholded to produce binary predictions.
#
# Scientific motivation: pe_efficiency_pct reflects the true biological
# output magnitude.  A model trained to predict "how much" active rather than
# "is it active" may learn smoother, more generalisable decision surfaces.
# This approach has never been attempted in this project.
# ---------------------------------------------------------------------------


def run_method13_ordinal_ridge(
    df: pd.DataFrame,
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 13: Ordinal Ridge Regression on normalised pe_efficiency_pct.

    Training target = pe_efficiency_pct / max_eff_in_training_fold (0–1).
    Threshold selected to maximise F1 on training predictions.
    """
    from sklearn.linear_model import Ridge

    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        X_tr = _to_float_array(df_train[base_cols])
        X_te = _to_float_array(df_test[base_cols])
        X_tr, X_te = impute_scale(X_tr, X_te)

        # Continuous target: normalise within training fold
        eff_raw = df_train["pe_efficiency_pct"].fillna(0.0).to_numpy()
        max_eff = max(eff_raw.max(), 1.0)
        y_reg = eff_raw / max_eff   # 0 for inactive, 0–1 for active

        reg = Ridge(alpha=5.0)
        reg.fit(X_tr, y_reg)
        train_reg_scores = reg.predict(X_tr)
        test_reg_scores = reg.predict(X_te)

        # Threshold: find cutoff on regression scores that maximises training F1
        threshold = optimise_threshold(train_reg_scores, y_train)

        for rt_name, score in zip(df_test["rt_name"], test_reg_scores):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method13_ordinal_ridge"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# METHOD 14 — Weighted Ensemble: Baseline (×0.8) + Efficiency-Weighted (×0.2)
#
# Method 9 (efficiency-weighted) achieves 0.7199 — very close to the benchmark.
# The baseline (0.7217) and Method 9 have slightly different error profiles:
# blending 80% baseline + 20% efficiency-weighted may combine their strengths.
# Both models are trained separately per fold and their probabilities are
# averaged with fixed weights before threshold optimisation.
# ---------------------------------------------------------------------------


def run_method14_baseline_effweight_ensemble(
    df: pd.DataFrame,
    gate_cols: list[str],
    hand_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    METHOD 14: Weighted Ensemble Baseline (80%) + Efficiency-Weighted LR (20%).

    Features: 33 frozen (same for both component models).
    Final score = 0.8 * prob_baseline + 0.2 * prob_effweighted.
    """
    base_cols = gate_cols + hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        df_train = df[df["rt_family"] != held_out].reset_index(drop=True)
        df_test = df[df["rt_family"] == held_out].reset_index(drop=True)
        y_train = df_train["active"].astype(int).to_numpy()

        X_tr = _to_float_array(df_train[base_cols])
        X_te = _to_float_array(df_test[base_cols])
        X_tr, X_te = impute_scale(X_tr, X_te)

        # Model A: baseline
        clf_a = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf_a.fit(X_tr, y_train)
        sa_tr = clf_a.predict_proba(X_tr)[:, 1]
        sa_te = clf_a.predict_proba(X_te)[:, 1]

        # Model B: efficiency-weighted
        eff = df_train["pe_efficiency_pct"].fillna(0.0).to_numpy()
        weights = np.where(y_train == 1, np.maximum(eff, 1.0), 1.0)
        clf_b = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        )
        clf_b.fit(X_tr, y_train, sample_weight=weights)
        sb_tr = clf_b.predict_proba(X_tr)[:, 1]
        sb_te = clf_b.predict_proba(X_te)[:, 1]

        ens_tr = 0.8 * sa_tr + 0.2 * sb_tr
        ens_te = 0.8 * sa_te + 0.2 * sb_te
        threshold = optimise_threshold(ens_tr, y_train)

        for rt_name, score in zip(df_test["rt_name"], ens_te):
            rows.append(
                {
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                }
            )

    pred_df = (
        pd.DataFrame(rows)
        .merge(df[["rt_name", "active", "rt_family"]], on="rt_name", how="left")
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "method14_baseline_effweight_ensemble"
    return metrics, pred_df


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _summary_row(label: str, metrics: dict, n_features: int | str) -> dict:
    return {
        "model": label,
        "lofo_macro_f1_informative": round(metrics["lofo_macro_f1_informative"], 4),
        "overall_f1": round(metrics["overall_f1"], 4),
        "overall_auc": round(metrics["overall_auc"], 4) if metrics["overall_auc"] else None,
        "retroviral_tp_of_12": metrics["retroviral_tp_of_12"],
        "vs_benchmark": round(metrics["lofo_macro_f1_informative"] - FROZEN_BENCHMARK, 4),
        "n_features_approx": n_features,
    }


def run_novel_experiments(
    data_dir: Path,
    gate_score_path: Path,
    esm_path: Path,
    output_dir: Path,
    run_methods: list[int] | None = None,
) -> None:
    """
    Run all (or a subset) of novel methods and save results.

    Args:
        run_methods: If None, run all.  Pass e.g. [1, 3] to run only Methods 1 and 3.
    """
    print("Loading data …")
    df = load_data(data_dir, gate_score_path)
    esm_dict = load_esm(esm_path)
    gate_cols = resolve_gate_cols(df)
    hand_cols = resolve_hand_cols(FROZEN_HAND_COLS, df)
    hand_raw = pd.read_csv(data_dir / "handcrafted_features.csv", nrows=1).columns.tolist()
    all_hand_cols = [c for c in hand_raw if c != "rt_name" and c in df.columns]
    print(f"  {len(df)} samples | {len(gate_cols)} gate cols | {len(hand_cols)} frozen hand cols | {len(all_hand_cols)} total hand cols")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_all = run_methods is None

    summary_rows = []
    details: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------------
    if run_all or 0 in (run_methods or []):
        print("\n[BASELINE] Running frozen winner replica …")
        m, preds = run_baseline(df, gate_cols, hand_cols)
        summary_rows.append(_summary_row("baseline_hybrid_lr", m, len(gate_cols) + len(hand_cols)))
        details["baseline_hybrid_lr"] = m
        preds.to_csv(output_dir / "baseline_hybrid_lr_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  (benchmark: {FROZEN_BENCHMARK})")

    # -----------------------------------------------------------------------
    # METHOD 1: ESM Residualization
    # -----------------------------------------------------------------------
    if run_all or 1 in (run_methods or []):
        print("\n[METHOD 1] ESM Residualization …")
        m, preds = run_method1_esm_residualized(df, esm_dict, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method1_esm_residualized", m, f"{len(gate_cols) + len(hand_cols) + 15} (~33+15)"))
        details["method1_esm_residualized"] = m
        preds.to_csv(output_dir / "method1_esm_residualized_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 2: Nested LOFO Feature Selection
    # -----------------------------------------------------------------------
    if run_all or 2 in (run_methods or []):
        print("\n[METHOD 2] Nested LOFO Feature Selection …")
        m, preds = run_method2_nested_lofo(df, gate_cols, all_hand_cols)
        summary_rows.append(_summary_row("method2_nested_lofo", m, "variable per fold"))
        details["method2_nested_lofo"] = m
        preds.to_csv(output_dir / "method2_nested_lofo_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 3: Soft-Voting Ensemble
    # -----------------------------------------------------------------------
    if run_all or 3 in (run_methods or []):
        print("\n[METHOD 3] Soft-Voting Ensemble (LR + ESM-LR + SVM-RBF) …")
        m, preds = run_method3_ensemble(df, esm_dict, gate_cols, hand_cols, all_hand_cols)
        summary_rows.append(_summary_row("method3_ensemble", m, "3 diverse models"))
        details["method3_ensemble"] = m
        preds.to_csv(output_dir / "method3_ensemble_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 4: Feature Interactions + ElasticNet
    # -----------------------------------------------------------------------
    if run_all or 4 in (run_methods or []):
        print("\n[METHOD 4] Feature Interactions + ElasticNet …")
        m, preds = run_method4_interactions_elasticnet(df, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method4_interactions_elasticnet", m, 69))
        details["method4_interactions_elasticnet"] = m
        preds.to_csv(output_dir / "method4_interactions_elasticnet_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 5: Combined Novel Pipeline
    # -----------------------------------------------------------------------
    if run_all or 5 in (run_methods or []):
        print("\n[METHOD 5] Combined Novel Pipeline (Nested + ESM + Interactions + ElasticNet) …")
        m, preds = run_method5_combined(df, esm_dict, gate_cols, all_hand_cols)
        summary_rows.append(_summary_row("method5_combined", m, "~85 (variable)"))
        details["method5_combined"] = m
        preds.to_csv(output_dir / "method5_combined_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 6: ESM Cosine-Similarity Feature
    # -----------------------------------------------------------------------
    if run_all or 6 in (run_methods or []):
        print("\n[METHOD 6] ESM Cosine-Similarity Feature (33+1 features) …")
        m, preds = run_method6_esm_cosine(df, esm_dict, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method6_esm_cosine", m, 34))
        details["method6_esm_cosine"] = m
        preds.to_csv(output_dir / "method6_esm_cosine_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 7: Two-Model Soft Ensemble (Baseline + ESM Cosine)
    # -----------------------------------------------------------------------
    if run_all or 7 in (run_methods or []):
        print("\n[METHOD 7] Two-Model Ensemble (Baseline LR + ESM-Cosine LR) …")
        m, preds = run_method7_two_model_ensemble(df, esm_dict, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method7_two_model_ensemble", m, "34 (avg of 2)"))
        details["method7_two_model_ensemble"] = m
        preds.to_csv(output_dir / "method7_two_model_ensemble_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 8: ESM Cosine + C-Grid (within-fold tuning)
    # -----------------------------------------------------------------------
    if run_all or 8 in (run_methods or []):
        print("\n[METHOD 8] ESM Cosine + C-Grid Search (within-fold) …")
        m, preds = run_method8_esm_cosine_tuned(df, esm_dict, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method8_esm_cosine_tuned", m, 34))
        details["method8_esm_cosine_tuned"] = m
        preds.to_csv(output_dir / "method8_esm_cosine_tuned_predictions.csv", index=False)
        c_info = m.get("chosen_c_per_fold", {})
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")
        print(f"  C chosen per fold: {c_info}")

    # -----------------------------------------------------------------------
    # METHOD 9: Efficiency-Weighted Logistic Regression
    # -----------------------------------------------------------------------
    if run_all or 9 in (run_methods or []):
        print("\n[METHOD 9] Efficiency-Weighted LR (pe_efficiency_pct as sample weight) …")
        m, preds = run_method9_efficiency_weighted(df, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method9_efficiency_weighted", m, len(gate_cols) + len(hand_cols)))
        details["method9_efficiency_weighted"] = m
        preds.to_csv(output_dir / "method9_efficiency_weighted_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 10: Efficiency-Weighted + ESM Cosine
    # -----------------------------------------------------------------------
    if run_all or 10 in (run_methods or []):
        print("\n[METHOD 10] Efficiency-Weighted LR + ESM Cosine Feature …")
        m, preds = run_method10_efficiency_esm(df, esm_dict, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method10_efficiency_esm", m, len(gate_cols) + len(hand_cols) + 1))
        details["method10_efficiency_esm"] = m
        preds.to_csv(output_dir / "method10_efficiency_esm_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 11: Balanced-Accuracy Threshold (instead of F1-optimal)
    # -----------------------------------------------------------------------
    if run_all or 11 in (run_methods or []):
        print("\n[METHOD 11] Balanced-Accuracy Threshold on frozen hybrid …")
        m, preds = run_method11_balanced_threshold(df, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method11_balanced_threshold", m, len(gate_cols) + len(hand_cols)))
        details["method11_balanced_threshold"] = m
        preds.to_csv(output_dir / "method11_balanced_threshold_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 12: Sqrt-Scaled Efficiency + Balanced Threshold
    # -----------------------------------------------------------------------
    if run_all or 12 in (run_methods or []):
        print("\n[METHOD 12] Sqrt-Scaled Efficiency Weighting + Balanced Threshold …")
        m, preds = run_method12_sqrt_eff_balanced(df, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method12_sqrt_eff_balanced", m, len(gate_cols) + len(hand_cols)))
        details["method12_sqrt_eff_balanced"] = m
        preds.to_csv(output_dir / "method12_sqrt_eff_balanced_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 13: Ordinal Ridge on pe_efficiency_pct
    # -----------------------------------------------------------------------
    if run_all or 13 in (run_methods or []):
        print("\n[METHOD 13] Ordinal Ridge Regression on pe_efficiency_pct …")
        m, preds = run_method13_ordinal_ridge(df, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method13_ordinal_ridge", m, len(gate_cols) + len(hand_cols)))
        details["method13_ordinal_ridge"] = m
        preds.to_csv(output_dir / "method13_ordinal_ridge_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # METHOD 14: Weighted Ensemble Baseline + Efficiency-Weighted
    # -----------------------------------------------------------------------
    if run_all or 14 in (run_methods or []):
        print("\n[METHOD 14] Weighted Ensemble: Baseline×0.8 + EffWeight×0.2 …")
        m, preds = run_method14_baseline_effweight_ensemble(df, gate_cols, hand_cols)
        summary_rows.append(_summary_row("method14_baseline_effweight_ensemble", m, len(gate_cols) + len(hand_cols)))
        details["method14_baseline_effweight_ensemble"] = m
        preds.to_csv(output_dir / "method14_baseline_effweight_ensemble_predictions.csv", index=False)
        print(f"  LOFO informative macro-F1 = {m['lofo_macro_f1_informative']:.4f}  Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    # Add frozen benchmark as reference row
    summary_rows.insert(0, {
        "model": "FROZEN_BENCHMARK_hybrid_lr",
        "lofo_macro_f1_informative": FROZEN_BENCHMARK,
        "overall_f1": 0.7027,
        "overall_auc": 0.7751,
        "retroviral_tp_of_12": 7,
        "vs_benchmark": 0.0,
        "n_features_approx": 33,
    })

    summary = pd.DataFrame(summary_rows).sort_values(
        "lofo_macro_f1_informative", ascending=False
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        summary[["model", "lofo_macro_f1_informative", "overall_f1", "retroviral_tp_of_12", "vs_benchmark"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    )
    print(f"\nOutputs saved to: {output_dir}")

    best = summary.iloc[0]
    if best["model"] != "FROZEN_BENCHMARK_hybrid_lr":
        print(f"\n✓ New winner: {best['model']}  |  LOFO F1 = {best['lofo_macro_f1_informative']:.4f}  (+{best['vs_benchmark']:.4f} vs benchmark)")
    else:
        print(f"\nBenchmark still holds: {FROZEN_BENCHMARK:.4f}  — no method exceeded it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Novel approach experiments for Retroviral Wall Challenge."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--gate-scores",
        type=Path,
        default=Path("retroviral_wall/outputs/gate_scores/all_gate_scores.csv"),
    )
    parser.add_argument(
        "--esm", type=Path, default=Path("data/esm2_embeddings.npz")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/novel_approach"))
    parser.add_argument(
        "--methods",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Which methods to run (0=baseline, 1=ESM-PCA-resid, 2=nested-lofo, "
            "3=ensemble-SVM, 4=interactions-elasticnet, 5=combined, "
            "6=esm-cosine, 7=two-model-ensemble, 8=esm-cosine-tuned). Default: all."
        ),
    )
    args = parser.parse_args()
    run_novel_experiments(
        data_dir=args.data_dir,
        gate_score_path=args.gate_scores,
        esm_path=args.esm,
        output_dir=args.output_dir,
        run_methods=args.methods,
    )


if __name__ == "__main__":
    main()
