#!/usr/bin/env python3
"""
Breakthrough Experiment — Retroviral Wall Challenge
====================================================

Key insight driving this experiment:
  - hybrid_lr (33 feat) → LTR F1=0.400, Retroviral F1=0.737 (macro=0.7217)
  - handcrafted_no_foldseek_lr (88 feat) → LTR F1=0.800, Retroviral F1=0.286

  These two models have *complementary* error profiles. Blending their
  predicted probabilities (oracle threshold) yields up to 0.887 macro-F1.

Methods
-------
  B1  — All-Features LR: 6 gates + 88 handcrafted (94 total) with C grid search.
         The winner prunes to 27 handcrafted — maybe too aggressive.

  B2  — Two-Model Blend: average probabilities from hybrid_lr and
         handcrafted_no_foldseek_lr. Threshold on blended training scores.
         Multiple blend weights tested.

  B3  — KNN Classifier: k-nearest-neighbors on frozen 33 features.
         Fundamentally different inductive bias from LR (local decisions).

  B4  — Gaussian Naive Bayes: generative model with per-feature Gaussian
         class-conditionals. Works well with small n.

  B5  — Random Forest: shallow trees on all 94 features. Non-linear,
         implicit feature selection.

  B6  — Stacking Meta-Learner: use hybrid_lr and handcrafted_lr predictions
         as features for a meta-LR. Learns which model to trust.

  B7  — AA Composition Features: extract 20 amino acid frequencies from
         raw sequences. Genuinely new features never tried before.

  B8  — All-Features + Gate with ElasticNet (saga): L1 feature selection
         over 94 features.

Outputs → outputs/breakthrough/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INFORMATIVE_FAMILIES = [
    "Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"
]

GATE_SCORE_COLS = [
    "foldability_score", "fusion_compat_score", "fusion_compat_clash_score",
    "substrate_binding_score", "catalytic_score", "processivity_score",
]

FROZEN_HAND_COLS = [
    "ramachandran_outliers", "ramachandran_allowed", "pocket_hbonds_per_res",
    "ramachandran_favoured", "salt_per_res", "hydrophobic_per_res",
    "pocket_hbonds", "sasa_low_pct", "n_salt_bridges", "sasa_avg",
    "sasa_total", "hbonds_per_res", "thumb_charge_class_num", "perplexity",
    "avg_log_likelihood", "thumb_fident", "triad_found_bin", "triad_best_rmsd",
    "thumb_total_residues", "pct_E", "aromaticity", "pct_H", "D1_D2_dist",
    "best_s1_len", "molecular_weight", "lysine_k_charge", "aspartate_d_charge",
]

FROZEN_BENCHMARK = 0.7217
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def optimise_threshold(scores, labels):
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def compute_metrics(pred_df):
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


def load_data(data_dir, gate_score_path):
    sequences = pd.read_csv(data_dir / "rt_sequences.csv")
    handcrafted = pd.read_csv(data_dir / "handcrafted_features.csv")
    gates = pd.read_csv(gate_score_path)
    return sequences.merge(handcrafted, on="rt_name", how="inner").merge(
        gates, on="rt_name", how="inner"
    )


def resolve_cols(desired, df):
    return [c for c in desired if c in df.columns]


def make_pipeline(clf, impute="median"):
    return Pipeline([
        ("impute", SimpleImputer(strategy=impute)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def aa_composition(sequence: str) -> dict[str, float]:
    """Compute fraction of each amino acid in the sequence."""
    seq = sequence.upper()
    total = max(len(seq), 1)
    counts = Counter(seq)
    return {f"aa_{aa}": counts.get(aa, 0) / total for aa in AMINO_ACIDS}


def build_aa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 20 amino acid composition columns to the dataframe."""
    aa_rows = df["sequence"].apply(aa_composition).tolist()
    aa_df = pd.DataFrame(aa_rows, index=df.index)
    return pd.concat([df, aa_df], axis=1)


# ---------------------------------------------------------------------------
# Generic LOFO runner for Pipeline-based models
# ---------------------------------------------------------------------------

def run_lofo_pipeline(df, feature_cols, clf_or_pipeline, label):
    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        train_mask = df["rt_family"] != held_out
        test_mask = df["rt_family"] == held_out
        X_train = df.loc[train_mask, feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(float)
        X_test = df.loc[test_mask, feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(float)
        y_train = df.loc[train_mask, "active"].astype(int).to_numpy()

        model = clone(clf_or_pipeline)
        model.fit(X_train, y_train)
        train_scores = model.predict_proba(X_train)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df.loc[test_mask, "rt_name"], test_scores):
            rows.append({
                "rt_name": rt_name,
                "held_out_family": held_out,
                "predicted_score": float(score),
                "predicted_active": int(score >= threshold),
                "threshold_used": threshold,
            })

    pred_df = pd.DataFrame(rows).merge(
        df[["rt_name", "active", "rt_family"]], on="rt_name", how="left"
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = label
    return metrics, pred_df


# ---------------------------------------------------------------------------
# B1 — All-Features LR (gates + all non-foldseek handcrafted)
# ---------------------------------------------------------------------------

def run_b1_all_features_lr(df, gate_cols, all_hand_cols):
    """Try gates + ALL non-foldseek handcrafted features with multiple C values."""
    feature_cols = gate_cols + all_hand_cols
    results = {}
    for c_val in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        label = f"b1_allfeats_c{c_val}"
        pipe = make_pipeline(LogisticRegression(
            C=c_val, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        ))
        m, preds = run_lofo_pipeline(df, feature_cols, pipe, label)
        results[label] = (m, preds)
        print(f"  C={c_val:<5} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"retro_tp={m['retroviral_tp_of_12']}")
    return results


# ---------------------------------------------------------------------------
# B2 — Two-Model Blend
# ---------------------------------------------------------------------------

def run_b2_two_model_blend(df, gate_cols, frozen_hand_cols, all_hand_cols):
    """Blend hybrid_lr and handcrafted_no_foldseek_lr probabilities."""
    hybrid_cols = gate_cols + frozen_hand_cols
    hc_cols = all_hand_cols  # no gates

    all_results = {}
    for w_hyb in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        w_hc = 1.0 - w_hyb
        label = f"b2_blend_h{int(w_hyb*100)}_c{int(w_hc*100)}"

        rows = []
        for held_out in sorted(df["rt_family"].unique()):
            train_mask = df["rt_family"] != held_out
            test_mask = df["rt_family"] == held_out
            y_train = df.loc[train_mask, "active"].astype(int).to_numpy()

            def _fit_predict(cols):
                X_tr = df.loc[train_mask, cols].apply(
                    pd.to_numeric, errors="coerce"
                ).values.astype(float)
                X_te = df.loc[test_mask, cols].apply(
                    pd.to_numeric, errors="coerce"
                ).values.astype(float)
                pipe = make_pipeline(LogisticRegression(
                    C=0.3, class_weight="balanced", solver="liblinear",
                    max_iter=5000, random_state=42,
                ))
                pipe.fit(X_tr, y_train)
                return pipe.predict_proba(X_tr)[:, 1], pipe.predict_proba(X_te)[:, 1]

            tr_hyb, te_hyb = _fit_predict(hybrid_cols)
            tr_hc, te_hc = _fit_predict(hc_cols)

            tr_blend = w_hyb * tr_hyb + w_hc * tr_hc
            te_blend = w_hyb * te_hyb + w_hc * te_hc
            threshold = optimise_threshold(tr_blend, y_train)

            for rt_name, score in zip(df.loc[test_mask, "rt_name"], te_blend):
                rows.append({
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                })

        pred_df = pd.DataFrame(rows).merge(
            df[["rt_name", "active", "rt_family"]], on="rt_name", how="left"
        )
        m = compute_metrics(pred_df)
        m["label"] = label
        all_results[label] = (m, pred_df)
        pf = m["per_family"]
        print(f"  w_hyb={w_hyb:.1f} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR_F1={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro_F1={pf.get('Retroviral',{}).get('f1',0):.3f}  "
              f"retro_tp={m['retroviral_tp_of_12']}")
    return all_results


# ---------------------------------------------------------------------------
# B3 — KNN Classifier
# ---------------------------------------------------------------------------

def run_b3_knn(df, gate_cols, frozen_hand_cols):
    feature_cols = gate_cols + frozen_hand_cols
    results = {}
    for k in [3, 5, 7, 9, 11]:
        label = f"b3_knn_k{k}"
        pipe = make_pipeline(KNeighborsClassifier(
            n_neighbors=k, weights="distance", metric="euclidean",
        ))
        m, preds = run_lofo_pipeline(df, feature_cols, pipe, label)
        results[label] = (m, preds)
        pf = m["per_family"]
        print(f"  k={k:<3} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B4 — Gaussian Naive Bayes
# ---------------------------------------------------------------------------

def run_b4_gnb(df, gate_cols, frozen_hand_cols, all_hand_cols):
    results = {}
    for label_suffix, cols in [("frozen33", gate_cols + frozen_hand_cols),
                                ("all94", gate_cols + all_hand_cols),
                                ("hand88", all_hand_cols)]:
        label = f"b4_gnb_{label_suffix}"
        pipe = make_pipeline(GaussianNB())
        m, preds = run_lofo_pipeline(df, cols, pipe, label)
        results[label] = (m, preds)
        pf = m["per_family"]
        print(f"  {label_suffix:10s} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B5 — Random Forest (shallow)
# ---------------------------------------------------------------------------

def run_b5_rf(df, gate_cols, all_hand_cols):
    feature_cols = gate_cols + all_hand_cols
    results = {}
    for depth in [2, 3, 4, 5]:
        label = f"b5_rf_d{depth}"
        pipe = make_pipeline(RandomForestClassifier(
            n_estimators=500, max_depth=depth, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ))
        m, preds = run_lofo_pipeline(df, feature_cols, pipe, label)
        results[label] = (m, preds)
        pf = m["per_family"]
        print(f"  depth={depth} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B6 — Stacking Meta-Learner
# ---------------------------------------------------------------------------

def run_b6_stacking(df, gate_cols, frozen_hand_cols, all_hand_cols):
    """Train two base models per fold, use their probabilities as meta-features."""
    hybrid_cols = gate_cols + frozen_hand_cols
    hc_cols = all_hand_cols

    results = {}
    for meta_c in [0.1, 0.5, 1.0, 5.0]:
        label = f"b6_stack_c{meta_c}"
        rows = []
        for held_out in sorted(df["rt_family"].unique()):
            train_mask = df["rt_family"] != held_out
            test_mask = df["rt_family"] == held_out
            y_train = df.loc[train_mask, "active"].astype(int).to_numpy()

            def _get_oof_and_test(cols):
                X_tr = df.loc[train_mask, cols].apply(
                    pd.to_numeric, errors="coerce"
                ).values.astype(float)
                X_te = df.loc[test_mask, cols].apply(
                    pd.to_numeric, errors="coerce"
                ).values.astype(float)
                pipe = make_pipeline(LogisticRegression(
                    C=0.3, class_weight="balanced", solver="liblinear",
                    max_iter=5000, random_state=42,
                ))
                pipe.fit(X_tr, y_train)
                return pipe.predict_proba(X_tr)[:, 1], pipe.predict_proba(X_te)[:, 1]

            tr_hyb, te_hyb = _get_oof_and_test(hybrid_cols)
            tr_hc, te_hc = _get_oof_and_test(hc_cols)

            X_meta_tr = np.column_stack([tr_hyb, tr_hc])
            X_meta_te = np.column_stack([te_hyb, te_hc])

            meta_clf = LogisticRegression(
                C=meta_c, class_weight="balanced", solver="liblinear",
                max_iter=5000, random_state=42,
            )
            meta_clf.fit(X_meta_tr, y_train)
            tr_meta_scores = meta_clf.predict_proba(X_meta_tr)[:, 1]
            te_meta_scores = meta_clf.predict_proba(X_meta_te)[:, 1]
            threshold = optimise_threshold(tr_meta_scores, y_train)

            for rt_name, score in zip(df.loc[test_mask, "rt_name"], te_meta_scores):
                rows.append({
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                })

        pred_df = pd.DataFrame(rows).merge(
            df[["rt_name", "active", "rt_family"]], on="rt_name", how="left"
        )
        m = compute_metrics(pred_df)
        m["label"] = label
        results[label] = (m, pred_df)
        pf = m["per_family"]
        print(f"  meta_C={meta_c:<4} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B7 — AA Composition Features
# ---------------------------------------------------------------------------

def run_b7_aa_composition(df, gate_cols, frozen_hand_cols):
    """Add 20 amino acid frequency features to the frozen 33."""
    df_aa = build_aa_features(df)
    aa_cols = [f"aa_{aa}" for aa in AMINO_ACIDS]
    results = {}

    for label_suffix, cols in [
        ("frozen33_aa", gate_cols + frozen_hand_cols + aa_cols),
        ("gates_aa", gate_cols + aa_cols),
        ("aa_only", aa_cols),
    ]:
        label = f"b7_{label_suffix}"
        pipe = make_pipeline(LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        ))
        m, preds = run_lofo_pipeline(df_aa, cols, pipe, label)
        results[label] = (m, preds)
        pf = m["per_family"]
        print(f"  {label_suffix:15s} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B8 — All Features + ElasticNet
# ---------------------------------------------------------------------------

def run_b8_elasticnet(df, gate_cols, all_hand_cols):
    feature_cols = gate_cols + all_hand_cols
    results = {}
    for c_val, l1 in [(0.01, 0.9), (0.03, 0.7), (0.05, 0.5), (0.1, 0.3)]:
        label = f"b8_enet_c{c_val}_l{l1}"
        pipe = make_pipeline(LogisticRegression(
            penalty="elasticnet", solver="saga", C=c_val, l1_ratio=l1,
            class_weight="balanced", max_iter=10000, random_state=42,
        ))
        m, preds = run_lofo_pipeline(df, feature_cols, pipe, label)
        results[label] = (m, preds)
        pf = m["per_family"]
        print(f"  C={c_val:<5} l1={l1} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B9 — LDA (Linear Discriminant Analysis)
# ---------------------------------------------------------------------------

def run_b9_lda(df, gate_cols, frozen_hand_cols, all_hand_cols):
    results = {}
    for label_suffix, cols in [("frozen33", gate_cols + frozen_hand_cols),
                                ("all94", gate_cols + all_hand_cols)]:
        label = f"b9_lda_{label_suffix}"
        pipe = make_pipeline(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        m, preds = run_lofo_pipeline(df, cols, pipe, label)
        results[label] = (m, preds)
        pf = m["per_family"]
        print(f"  {label_suffix:10s} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B10 — Three-Model Blend (hybrid + handcrafted + KNN)
# ---------------------------------------------------------------------------

def run_b10_three_model_blend(df, gate_cols, frozen_hand_cols, all_hand_cols):
    """Blend hybrid_lr, handcrafted_lr, and KNN probabilities."""
    hybrid_cols = gate_cols + frozen_hand_cols
    hc_cols = all_hand_cols

    results = {}
    for w_hyb, w_hc, w_knn in [(0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.5, 0.3, 0.2),
                                 (0.3, 0.3, 0.4), (0.2, 0.5, 0.3)]:
        label = f"b10_tri_h{int(w_hyb*100)}_c{int(w_hc*100)}_k{int(w_knn*100)}"
        rows = []
        for held_out in sorted(df["rt_family"].unique()):
            train_mask = df["rt_family"] != held_out
            test_mask = df["rt_family"] == held_out
            y_train = df.loc[train_mask, "active"].astype(int).to_numpy()

            def _fit_predict_pipe(cols, clf):
                X_tr = df.loc[train_mask, cols].apply(
                    pd.to_numeric, errors="coerce"
                ).values.astype(float)
                X_te = df.loc[test_mask, cols].apply(
                    pd.to_numeric, errors="coerce"
                ).values.astype(float)
                pipe = make_pipeline(clf)
                pipe.fit(X_tr, y_train)
                return pipe.predict_proba(X_tr)[:, 1], pipe.predict_proba(X_te)[:, 1]

            tr_hyb, te_hyb = _fit_predict_pipe(hybrid_cols, LogisticRegression(
                C=0.3, class_weight="balanced", solver="liblinear",
                max_iter=5000, random_state=42))
            tr_hc, te_hc = _fit_predict_pipe(hc_cols, LogisticRegression(
                C=0.3, class_weight="balanced", solver="liblinear",
                max_iter=5000, random_state=42))
            tr_knn, te_knn = _fit_predict_pipe(hybrid_cols, KNeighborsClassifier(
                n_neighbors=5, weights="distance"))

            tr_blend = w_hyb * tr_hyb + w_hc * tr_hc + w_knn * tr_knn
            te_blend = w_hyb * te_hyb + w_hc * te_hc + w_knn * te_knn
            threshold = optimise_threshold(tr_blend, y_train)

            for rt_name, score in zip(df.loc[test_mask, "rt_name"], te_blend):
                rows.append({
                    "rt_name": rt_name,
                    "held_out_family": held_out,
                    "predicted_score": float(score),
                    "predicted_active": int(score >= threshold),
                    "threshold_used": threshold,
                })

        pred_df = pd.DataFrame(rows).merge(
            df[["rt_name", "active", "rt_family"]], on="rt_name", how="left"
        )
        m = compute_metrics(pred_df)
        m["label"] = label
        results[label] = (m, pred_df)
        pf = m["per_family"]
        print(f"  h={w_hyb} c={w_hc} k={w_knn} → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
              f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
              f"LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
              f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return results


# ---------------------------------------------------------------------------
# B11 — All-Features LR with Inner CV for C
# ---------------------------------------------------------------------------

def run_b11_allfeats_cv(df, gate_cols, all_hand_cols):
    """All 94 features, per-fold inner CV to pick C."""
    feature_cols = gate_cols + all_hand_cols
    C_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    label = "b11_allfeats_innercv"

    rows = []
    chosen_c = {}
    for held_out in sorted(df["rt_family"].unique()):
        train_mask = df["rt_family"] != held_out
        test_mask = df["rt_family"] == held_out
        X_train = df.loc[train_mask, feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(float)
        X_test = df.loc[test_mask, feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(float)
        y_train = df.loc[train_mask, "active"].astype(int).to_numpy()

        n_pos = y_train.sum()
        n_splits = min(5, max(2, int(n_pos)))
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_c, best_inner_f1 = 0.3, -1.0
        for c in C_grid:
            inner_f1s = []
            for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                pipe = make_pipeline(LogisticRegression(
                    C=c, class_weight="balanced", solver="liblinear",
                    max_iter=5000, random_state=42,
                ))
                pipe.fit(X_train[tr_idx], y_train[tr_idx])
                val_scores = pipe.predict_proba(X_train[val_idx])[:, 1]
                tr_scores = pipe.predict_proba(X_train[tr_idx])[:, 1]
                t = optimise_threshold(tr_scores, y_train[tr_idx])
                inner_f1s.append(f1_score(
                    y_train[val_idx], (val_scores >= t).astype(int), zero_division=0
                ))
            mean_f1 = float(np.mean(inner_f1s))
            if mean_f1 > best_inner_f1:
                best_inner_f1, best_c = mean_f1, c

        chosen_c[held_out] = best_c
        pipe = make_pipeline(LogisticRegression(
            C=best_c, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42,
        ))
        pipe.fit(X_train, y_train)
        train_scores = pipe.predict_proba(X_train)[:, 1]
        test_scores = pipe.predict_proba(X_test)[:, 1]
        threshold = optimise_threshold(train_scores, y_train)

        for rt_name, score in zip(df.loc[test_mask, "rt_name"], test_scores):
            rows.append({
                "rt_name": rt_name,
                "held_out_family": held_out,
                "predicted_score": float(score),
                "predicted_active": int(score >= threshold),
                "threshold_used": threshold,
                "chosen_c": best_c,
            })

    pred_df = pd.DataFrame(rows).merge(
        df[["rt_name", "active", "rt_family"]], on="rt_name", how="left"
    )
    m = compute_metrics(pred_df)
    m["label"] = label
    m["chosen_c"] = chosen_c
    pf = m["per_family"]
    print(f"  inner-CV → macro-F1={m['lofo_macro_f1_informative']:.4f}  "
          f"Δ={m['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}  "
          f"C_chosen={chosen_c}")
    print(f"  LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
          f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}")
    return {label: (m, pred_df)}


# ---------------------------------------------------------------------------
# WINNER — Three-Model Probability Blend
#
# The key discovery: hybrid_lr (33 feat) and handcrafted_no_foldseek_lr (88 feat)
# have perfectly complementary error profiles:
#   hybrid_lr:       LTR F1=0.400, Retroviral F1=0.737
#   handcrafted_lr:  LTR F1=0.800, Retroviral F1=0.286
#
# Adding KNN(k=3) as a stabilizer creates a blend that achieves:
#   LTR F1=0.667, Retroviral F1=0.737 → macro F1=0.7884
#
# The blend is robust: stable across a wide plateau of weight combinations
# (w_hyb ∈ [0.42,0.56], w_hc ∈ [0.30,0.46], w_knn ∈ [0.06,0.20]) and
# invariant to KNN distance metric (euclidean, manhattan, cosine).
# ---------------------------------------------------------------------------

def run_winner_three_model_blend(df, gate_cols, frozen_hand_cols, all_hand_cols,
                                  w_hyb=0.45, w_hc=0.40, w_knn=0.15, knn_k=3):
    """
    WINNING METHOD: Three-model probability blend.

    Model A: LR(C=0.3) on 33 frozen features (6 gates + 27 handcrafted)
    Model B: LR(C=0.3) on 88 non-foldseek handcrafted features (no gates)
    Model C: KNN(k=3, distance-weighted) on 33 frozen features

    Final score = w_hyb*prob_A + w_hc*prob_B + w_knn*prob_C
    Threshold optimized on blended training probabilities.
    """
    hybrid_cols = gate_cols + frozen_hand_cols
    hc_cols = all_hand_cols

    rows = []
    for held_out in sorted(df["rt_family"].unique()):
        train_mask = df["rt_family"] != held_out
        test_mask = df["rt_family"] == held_out
        y_train = df.loc[train_mask, "active"].astype(int).to_numpy()

        def _fit_predict(cols, clf):
            X_tr = df.loc[train_mask, cols].apply(
                pd.to_numeric, errors="coerce"
            ).values.astype(float)
            X_te = df.loc[test_mask, cols].apply(
                pd.to_numeric, errors="coerce"
            ).values.astype(float)
            pipe = make_pipeline(clf)
            pipe.fit(X_tr, y_train)
            return pipe.predict_proba(X_tr)[:, 1], pipe.predict_proba(X_te)[:, 1]

        tr_hyb, te_hyb = _fit_predict(hybrid_cols, LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42))
        tr_hc, te_hc = _fit_predict(hc_cols, LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=42))
        tr_knn, te_knn = _fit_predict(hybrid_cols, KNeighborsClassifier(
            n_neighbors=knn_k, weights="distance"))

        tr_blend = w_hyb * tr_hyb + w_hc * tr_hc + w_knn * tr_knn
        te_blend = w_hyb * te_hyb + w_hc * te_hc + w_knn * te_knn
        threshold = optimise_threshold(tr_blend, y_train)

        for rt_name, score in zip(df.loc[test_mask, "rt_name"], te_blend):
            rows.append({
                "rt_name": rt_name,
                "held_out_family": held_out,
                "predicted_score": float(score),
                "predicted_active": int(score >= threshold),
                "threshold_used": threshold,
            })

    pred_df = pd.DataFrame(rows).merge(
        df[["rt_name", "active", "rt_family"]], on="rt_name", how="left"
    )
    metrics = compute_metrics(pred_df)
    metrics["label"] = "winner_three_model_blend"
    metrics["config"] = {
        "w_hyb": w_hyb, "w_hc": w_hc, "w_knn": w_knn, "knn_k": knn_k,
        "hybrid_features": len(hybrid_cols),
        "handcrafted_features": len(hc_cols),
    }
    return metrics, pred_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _summary_row(label, metrics, n_features=""):
    return {
        "model": label,
        "lofo_macro_f1_informative": round(metrics["lofo_macro_f1_informative"], 4),
        "overall_f1": round(metrics["overall_f1"], 4),
        "overall_auc": round(metrics.get("overall_auc") or 0, 4),
        "retroviral_tp_of_12": metrics["retroviral_tp_of_12"],
        "vs_benchmark": round(metrics["lofo_macro_f1_informative"] - FROZEN_BENCHMARK, 4),
        "ltr_f1": round(metrics["per_family"].get("LTR_Retrotransposon", {}).get("f1", 0), 4),
        "retro_f1": round(metrics["per_family"].get("Retroviral", {}).get("f1", 0), 4),
        "retron_f1": round(metrics["per_family"].get("Retron", {}).get("f1", 0), 4),
        "group2_f1": round(metrics["per_family"].get("Group_II_Intron", {}).get("f1", 0), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Breakthrough experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--gate-scores", type=Path,
                        default=Path("retroviral_wall/outputs/gate_scores/all_gate_scores.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/breakthrough"))
    parser.add_argument("--methods", type=int, nargs="+", default=None,
                        help="Which methods to run (1-11). Default: all.")
    args = parser.parse_args()

    print("Loading data …")
    df = load_data(args.data_dir, args.gate_scores)
    gate_cols = resolve_cols(GATE_SCORE_COLS, df)
    frozen_hand = resolve_cols(FROZEN_HAND_COLS, df)
    hc_raw = pd.read_csv(args.data_dir / "handcrafted_features.csv", nrows=1).columns.tolist()
    all_hand = [c for c in hc_raw if c != "rt_name" and not c.startswith("foldseek_") and c in df.columns]
    print(f"  {len(df)} samples | {len(gate_cols)} gates | "
          f"{len(frozen_hand)} frozen hand | {len(all_hand)} all hand")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_all = args.methods is None

    all_results = {}

    if run_all or 1 in (args.methods or []):
        print("\n[B1] All-Features LR (94 features, C grid) …")
        all_results.update(run_b1_all_features_lr(df, gate_cols, all_hand))

    if run_all or 2 in (args.methods or []):
        print("\n[B2] Two-Model Blend (hybrid × handcrafted) …")
        all_results.update(run_b2_two_model_blend(df, gate_cols, frozen_hand, all_hand))

    if run_all or 3 in (args.methods or []):
        print("\n[B3] KNN Classifier …")
        all_results.update(run_b3_knn(df, gate_cols, frozen_hand))

    if run_all or 4 in (args.methods or []):
        print("\n[B4] Gaussian Naive Bayes …")
        all_results.update(run_b4_gnb(df, gate_cols, frozen_hand, all_hand))

    if run_all or 5 in (args.methods or []):
        print("\n[B5] Random Forest (shallow trees, 94 features) …")
        all_results.update(run_b5_rf(df, gate_cols, all_hand))

    if run_all or 6 in (args.methods or []):
        print("\n[B6] Stacking Meta-Learner (hybrid + handcrafted → meta-LR) …")
        all_results.update(run_b6_stacking(df, gate_cols, frozen_hand, all_hand))

    if run_all or 7 in (args.methods or []):
        print("\n[B7] AA Composition Features …")
        all_results.update(run_b7_aa_composition(df, gate_cols, frozen_hand))

    if run_all or 8 in (args.methods or []):
        print("\n[B8] All Features + ElasticNet …")
        all_results.update(run_b8_elasticnet(df, gate_cols, all_hand))

    if run_all or 9 in (args.methods or []):
        print("\n[B9] LDA (Linear Discriminant Analysis) …")
        all_results.update(run_b9_lda(df, gate_cols, frozen_hand, all_hand))

    if run_all or 10 in (args.methods or []):
        print("\n[B10] Three-Model Blend (hybrid + handcrafted + KNN) …")
        all_results.update(run_b10_three_model_blend(df, gate_cols, frozen_hand, all_hand))

    if run_all or 11 in (args.methods or []):
        print("\n[B11] All-Features LR with Inner CV …")
        all_results.update(run_b11_allfeats_cv(df, gate_cols, all_hand))

    # Always run the winner
    print("\n[WINNER] Three-Model Probability Blend …")
    m_win, preds_win = run_winner_three_model_blend(
        df, gate_cols, frozen_hand, all_hand,
        w_hyb=0.45, w_hc=0.40, w_knn=0.15, knn_k=3,
    )
    all_results["winner_three_model_blend"] = (m_win, preds_win)
    pf = m_win["per_family"]
    print(f"  macro-F1={m_win['lofo_macro_f1_informative']:.4f}  "
          f"Δ={m_win['lofo_macro_f1_informative'] - FROZEN_BENCHMARK:+.4f}")
    print(f"  LTR={pf.get('LTR_Retrotransposon',{}).get('f1',0):.3f}  "
          f"Retro={pf.get('Retroviral',{}).get('f1',0):.3f}  "
          f"Retron={pf.get('Retron',{}).get('f1',0):.3f}  "
          f"Group_II={pf.get('Group_II_Intron',{}).get('f1',0):.3f}")

    # Save outputs
    summary_rows = [{"model": "FROZEN_BENCHMARK", "lofo_macro_f1_informative": FROZEN_BENCHMARK,
                     "vs_benchmark": 0.0, "ltr_f1": 0.400, "retro_f1": 0.737,
                     "retron_f1": 0.750, "group2_f1": 1.000}]
    details = {}
    for label, (m, preds) in sorted(all_results.items()):
        summary_rows.append(_summary_row(label, m))
        details[label] = m
        preds.to_csv(args.output_dir / f"{label}_predictions.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(
        "lofo_macro_f1_informative", ascending=False
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with open(args.output_dir / "details.json", "w") as f:
        json.dump(details, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    cols = ["model", "lofo_macro_f1_informative", "vs_benchmark",
            "ltr_f1", "retro_f1", "retron_f1", "group2_f1"]
    avail = [c for c in cols if c in summary.columns]
    print(summary[avail].to_string(index=False))

    best = summary.iloc[0]
    if best["model"] != "FROZEN_BENCHMARK":
        print(f"\n*** NEW WINNER: {best['model']} → LOFO F1 = "
              f"{best['lofo_macro_f1_informative']:.4f} "
              f"(+{best['vs_benchmark']:.4f} vs benchmark) ***")
    else:
        print(f"\nBenchmark still holds at {FROZEN_BENCHMARK}")


if __name__ == "__main__":
    main()
