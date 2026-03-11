from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class GateIntegrator:
    GATE_COLUMNS = [
        "foldability_score",
        "fusion_compat_score",
        "substrate_binding_score",
        "catalytic_score",
        "processivity_score",
    ]

    def __init__(self, strategy: str = "bayesian_lr", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state

    @classmethod
    def available_strategies(cls) -> list[str]:
        return [
            "strict_veto",
            "harmonic_mean",
            "weaklink_support",
            "two_stage_triage",
            "majority_support",
            "bayesian_lr",
            "bart",
        ]

    def _prepare_features(self, gate_scores: pd.DataFrame, handcrafted_residuals: pd.DataFrame | None = None) -> np.ndarray:
        gate_cols = [c for c in self.GATE_COLUMNS if c in gate_scores.columns]
        X = gate_scores[gate_cols].astype(float).fillna(0.5).to_numpy()
        if handcrafted_residuals is not None:
            X = np.hstack([X, handcrafted_residuals.astype(float).fillna(handcrafted_residuals.median()).to_numpy()])
        return X

    def fit_predict_lofo(self, gate_scores: pd.DataFrame, labels: pd.Series, families: pd.Series, handcrafted_residuals: pd.DataFrame | None = None) -> pd.DataFrame:
        X_raw = self._prepare_features(gate_scores, handcrafted_residuals)
        y = labels.to_numpy().astype(int)
        fam = families.to_numpy()
        names = gate_scores.index.to_numpy()
        gate_cols = [c for c in self.GATE_COLUMNS if c in gate_scores.columns]
        gate_matrix = gate_scores[gate_cols].astype(float).fillna(0.5)

        rows: list[dict] = []
        for held_out in np.unique(fam):
            test_mask = fam == held_out
            train_mask = ~test_mask
            X_train, y_train = X_raw[train_mask], y[train_mask]
            X_test = X_raw[test_mask]
            gate_test = gate_matrix.loc[test_mask]

            scores_test, scores_train = self._fit_predict_strategy(X_train, y_train, X_test)
            threshold = self._optimise_threshold(scores_train, y_train)
            for idx, (n, s) in enumerate(zip(names[test_mask], scores_test)):
                gate_row = gate_test.iloc[idx]
                weakest_gate = str(gate_row.idxmin()).replace("_score", "")
                weakest_gate_score = float(gate_row.min())
                low_gate_count = int((gate_row < 0.5).sum())
                rows.append({
                    "rt_name": n,
                    "held_out_family": held_out,
                    "predicted_score": float(s),
                    "predicted_active": int(s >= threshold),
                    "threshold_used": float(threshold),
                    "weakest_gate": weakest_gate,
                    "weakest_gate_score": weakest_gate_score,
                    "low_gate_count": low_gate_count,
                })
        return pd.DataFrame(rows)

    def _fit_predict_strategy(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.strategy == "strict_veto":
            train_scores = self._strict_veto(X_train)
            test_scores = self._strict_veto(X_test)
            return test_scores, train_scores

        if self.strategy == "harmonic_mean":
            train_scores = self._harmonic_mean(X_train)
            test_scores = self._harmonic_mean(X_test)
            return test_scores, train_scores

        if self.strategy == "weaklink_support":
            train_scores = self._weaklink_support(X_train)
            test_scores = self._weaklink_support(X_test)
            return test_scores, train_scores

        if self.strategy == "two_stage_triage":
            train_scores = self._two_stage_triage(X_train)
            test_scores = self._two_stage_triage(X_test)
            return test_scores, train_scores

        if self.strategy == "majority_support":
            train_scores = self._majority_support(X_train)
            test_scores = self._majority_support(X_test)
            return test_scores, train_scores

        if self.strategy == "bayesian_lr":
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            clf = LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", random_state=self.random_state)
            clf.fit(X_train_s, y_train)
            return clf.predict_proba(X_test_s)[:, 1], clf.predict_proba(X_train_s)[:, 1]

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

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _strict_veto(self, X: np.ndarray) -> np.ndarray:
        gate_x = np.clip(X[:, : len(self.GATE_COLUMNS)], 1e-6, 1.0)
        gmean = np.exp(np.mean(np.log(gate_x), axis=1))
        min_gate = gate_x.min(axis=1)
        veto = self._sigmoid((min_gate - 0.33) * 16.0)
        return np.clip(0.55 * gmean + 0.45 * (min_gate * veto), 0.0, 1.0)

    def _harmonic_mean(self, X: np.ndarray) -> np.ndarray:
        gate_x = np.clip(X[:, : len(self.GATE_COLUMNS)], 1e-6, 1.0)
        hm = gate_x.shape[1] / np.sum(1.0 / gate_x, axis=1)
        min_gate = gate_x.min(axis=1)
        soft_veto = self._sigmoid((min_gate - 0.28) * 10.0)
        return np.clip(0.85 * hm + 0.15 * soft_veto, 0.0, 1.0)

    def _weaklink_support(self, X: np.ndarray) -> np.ndarray:
        gate_x = np.clip(X[:, : len(self.GATE_COLUMNS)], 1e-6, 1.0)
        min_gate = gate_x.min(axis=1)
        mean_gate = gate_x.mean(axis=1)
        support = (gate_x >= 0.6).mean(axis=1)
        catastrophe = self._sigmoid((min_gate - 0.24) * 14.0)
        return np.clip(0.35 * min_gate + 0.25 * mean_gate + 0.20 * support + 0.20 * catastrophe, 0.0, 1.0)

    def _two_stage_triage(self, X: np.ndarray) -> np.ndarray:
        gate_x = np.clip(X[:, : len(self.GATE_COLUMNS)], 1e-6, 1.0)
        foldability = gate_x[:, 0]
        fusion = gate_x[:, 1]
        substrate = gate_x[:, 2]
        catalytic = gate_x[:, 3]
        processivity = gate_x[:, 4]

        filter_score = 0.5 * np.minimum.reduce([fusion, substrate, catalytic]) + 0.5 * np.mean([foldability, fusion, substrate, catalytic], axis=0)
        filter_pass = self._sigmoid((filter_score - 0.34) * 10.0)
        support = np.mean(np.sort(gate_x, axis=1)[:, -3:], axis=1)
        ranking = 0.45 * support + 0.35 * processivity + 0.20 * gate_x.mean(axis=1)
        return np.clip(0.25 * filter_pass + 0.60 * ranking * np.sqrt(filter_pass) + 0.15 * np.minimum(fusion, substrate), 0.0, 1.0)

    def _majority_support(self, X: np.ndarray) -> np.ndarray:
        gate_x = np.clip(X[:, : len(self.GATE_COLUMNS)], 1e-6, 1.0)
        support = (gate_x >= 0.55).mean(axis=1)
        mean_gate = gate_x.mean(axis=1)
        catastrophic = self._sigmoid((gate_x.min(axis=1) - 0.22) * 18.0)
        return np.clip(0.40 * support + 0.35 * mean_gate + 0.25 * catastrophic, 0.0, 1.0)
