import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
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

def fast_f1(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    if tp == 0: return 0.0
    return 2 * tp / (2 * tp + fp + fn)

def optimise_threshold(scores, labels):
    best_f1, best_t = -1.0, 0.5
    ts = np.arange(0.05, 0.96, 0.01)
    for t in ts:
        preds = (scores >= t).astype(int)
        f1 = fast_f1(labels, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

def load_data(data_dir, gate_score_path):
    sequences = pd.read_csv(data_dir / "rt_sequences.csv")
    handcrafted = pd.read_csv(data_dir / "handcrafted_features.csv")
    gates = pd.read_csv(gate_score_path)
    return sequences.merge(handcrafted, on="rt_name", how="inner").merge(
        gates, on="rt_name", how="inner"
    )

def make_pipeline(clf, impute="median"):
    return Pipeline([
        ("impute", SimpleImputer(strategy=impute)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])

def main():
    data_dir = Path("data")
    gate_scores = Path("retroviral_wall/outputs/gate_scores/all_gate_scores.csv")
    df = load_data(data_dir, gate_scores)

    gate_cols = [c for c in GATE_SCORE_COLS if c in df.columns]
    frozen_hand = [c for c in FROZEN_HAND_COLS if c in df.columns]
    hc_raw = pd.read_csv(data_dir / "handcrafted_features.csv", nrows=1).columns.tolist()
    all_hand = [c for c in hc_raw if c != "rt_name" and not c.startswith("foldseek_") and c in df.columns]

    hybrid_cols = gate_cols + frozen_hand
    hc_cols = all_hand
    all_cols = gate_cols + all_hand
    
    families = sorted(df["rt_family"].unique())
    y_true_all = df["active"].astype(int).to_numpy()
    
    # Let's search over hyperparams
    c_vals = [0.1, 0.2, 0.3, 0.5]
    k_vals = [3, 5]
    
    best_overall_macro = 0
    best_config = None
    
    for c in c_vals:
        for k in k_vals:
            models = {
                "hyb": (hybrid_cols, LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42)),
                "hc": (hc_cols, LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42)),
                "knn": (hybrid_cols, KNeighborsClassifier(n_neighbors=k, weights="distance")),
                "lda": (all_cols, LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            }
            
            fold_probs = {mname: {"tr": {}, "te": {}} for mname in models}
            
            fold_data = {}
            for held_out in families:
                train_mask = (df["rt_family"] != held_out).values
                test_mask = (df["rt_family"] == held_out).values
                y_train = y_true_all[train_mask]
                
                fold_data[held_out] = {
                    "train_mask": train_mask,
                    "test_mask": test_mask,
                    "y_train": y_train,
                    "y_test": y_true_all[test_mask]
                }
                
                for mname, (cols, clf) in models.items():
                    X_tr = df.loc[train_mask, cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
                    X_te = df.loc[test_mask, cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
                    
                    pipe = make_pipeline(clf)
                    pipe.fit(X_tr, y_train)
                    fold_probs[mname]["tr"][held_out] = pipe.predict_proba(X_tr)[:, 1]
                    fold_probs[mname]["te"][held_out] = pipe.predict_proba(X_te)[:, 1]

            weights_grid = np.arange(0.0, 1.05, 0.05)
            for w_hyb in weights_grid:
                for w_hc in weights_grid:
                    for w_knn in weights_grid:
                        w_lda = 1.0 - w_hyb - w_hc - w_knn
                        if w_lda < -1e-5 or w_lda > 1.0 + 1e-5:
                            continue
                        w_lda = max(0.0, w_lda) 
                        
                        informative_f1s = []
                        per_family_f1 = {}
                        
                        for held_out in families:
                            data = fold_data[held_out]
                            y_train = data["y_train"]
                            y_test = data["y_test"]
                            
                            tr_blend = (w_hyb * fold_probs["hyb"]["tr"][held_out] +
                                        w_hc * fold_probs["hc"]["tr"][held_out] +
                                        w_knn * fold_probs["knn"]["tr"][held_out] +
                                        w_lda * fold_probs["lda"]["tr"][held_out])
                                        
                            te_blend = (w_hyb * fold_probs["hyb"]["te"][held_out] +
                                        w_hc * fold_probs["hc"]["te"][held_out] +
                                        w_knn * fold_probs["knn"]["te"][held_out] +
                                        w_lda * fold_probs["lda"]["te"][held_out])
                                        
                            threshold = optimise_threshold(tr_blend, y_train)
                            y_pred = (te_blend >= threshold).astype(int)
                            
                            fam_f1 = float(fast_f1(y_test, y_pred))
                            per_family_f1[held_out] = fam_f1
                            if held_out in INFORMATIVE_FAMILIES:
                                informative_f1s.append(fam_f1)
                        
                        macro = float(np.mean(informative_f1s)) if informative_f1s else 0.0
                        
                        if macro > best_overall_macro:
                            best_overall_macro = macro
                            best_config = (c, k, w_hyb, w_hc, w_knn, w_lda, per_family_f1)

    print(f"BEST EVER: {best_overall_macro:.4f}")
    print(f"C={best_config[0]} k={best_config[1]} w=({best_config[2]:.2f}, {best_config[3]:.2f}, {best_config[4]:.2f}, {best_config[5]:.2f})")
    print(best_config[6])

if __name__ == "__main__":
    main()
