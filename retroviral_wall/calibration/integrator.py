from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class GateIntegrator:
    def __init__(self, strategy: str = "bayesian_lr", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state

    def _prepare_features(self, gate_scores: pd.DataFrame, handcrafted_residuals: pd.DataFrame | None = None) -> np.ndarray:
        gate_cols = [c for c in gate_scores.columns if c.endswith("_score")]
        X = gate_scores[gate_cols].astype(float).fillna(0.5).to_numpy()
        if handcrafted_residuals is not None:
            X = np.hstack([X, handcrafted_residuals.astype(float).fillna(handcrafted_residuals.median()).to_numpy()])
        return X

    def fit_predict_lofo(self, gate_scores: pd.DataFrame, labels: pd.Series, families: pd.Series, handcrafted_residuals: pd.DataFrame | None = None) -> pd.DataFrame:
        X = self._prepare_features(gate_scores, handcrafted_residuals)
        y = labels.to_numpy().astype(int)
        fam = families.to_numpy()
        names = gate_scores.index.to_numpy()

        rows: list[dict] = []
        for held_out in np.unique(fam):
            test_mask = fam == held_out
            train_mask = ~test_mask
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            scores_test, scores_train = self._fit_predict_strategy(X_train_s, y_train, X_test_s)
            threshold = self._optimise_threshold(scores_train, y_train)
            for n, s in zip(names[test_mask], scores_test):
                rows.append({
                    "rt_name": n,
                    "held_out_family": held_out,
                    "predicted_score": float(s),
                    "predicted_active": int(s >= threshold),
                    "threshold_used": float(threshold),
                })
        return pd.DataFrame(rows)

    def _fit_predict_strategy(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.strategy == "multiplicative":
            train_scores = np.exp(np.mean(np.log(np.clip(X_train, 1e-6, 1.0)), axis=1))
            test_scores = np.exp(np.mean(np.log(np.clip(X_test, 1e-6, 1.0)), axis=1))
            return test_scores, train_scores

        if self.strategy == "bayesian_lr":
            # Practical fallback: strong regularized logistic if PyMC unavailable.
            clf = LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", random_state=self.random_state)
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1], clf.predict_proba(X_train)[:, 1]

        if self.strategy == "bart":
            clf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, class_weight="balanced_subsample", random_state=self.random_state)
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1], clf.predict_proba(X_train)[:, 1]

        raise ValueError(f"Unknown strategy: {self.strategy}")

    @staticmethod
    def _optimise_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
        best_f1 = -1.0
        best_t = 0.5
        for t in np.arange(0.1, 0.91, 0.02):
            preds = (scores >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return best_t
