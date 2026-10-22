"""Detector model implementations for HW2 v2.

Detectors:
- T1: Error / repetition rule heuristic
- T2: Primary R3 kNN normal distance
- T3: Supervised L2 Logistic Regression on frozen embeddings
- Control NLP: R1 TF-IDF + Logistic Regression
- Control Tabular: Isolation Forest on R0-clean (normal-only fit)
- Control Tabular: Inductive ECOD on R0-clean (normal-only fit)
- Legacy Control: R0 Autoencoder (PyTorch, normal-only fit)
- Legacy Control: R0 XGBoost (supervised)
- Context Controls: Always-alert, Never-alert
- Secondary: Shallow DAE on R0-clean
- Secondary: Conditional residual baseline (behavior -> patch)
- Validation Threshold Selector (FPR <= 20% maximizing recall)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import IsolationForest
import xgboost as xgb

from Model.tabular_features_v2 import (
    BEHAVIOR_FEATURES,
    PATCH_FEATURES,
    R0_CLEAN_BINARY_COLUMNS,
    R0_CLEAN_COLUMNS,
    InductiveECODVariant,
)
from tools.build_text_embeddings import compute_channel_distance, compute_dual_distance


def select_validation_threshold(
    val_y: np.ndarray,
    val_scores: np.ndarray,
    max_fpr: float = 0.20,
) -> Tuple[float, Dict[str, float]]:
    """Frozen validation policy from Section 5:

    Enumerate all unique validation scores and +/- infinity.
    Predict anomaly when score >= tau.
    Feasible set: empirical validation FPR <= 20%.
    Among feasible: maximize failure recall.
    Tiebreaker 1: maximize balanced accuracy.
    Tiebreaker 2: higher threshold tau.
    """
    unique_scores = np.unique(val_scores)
    # Include bounds
    candidates = np.concatenate([[-np.inf], unique_scores, [np.inf]])

    best_tau = -np.inf
    best_recall = -1.0
    best_bal_acc = -1.0
    best_metrics: Dict[str, float] = {}

    feasible_candidates = []
    all_candidates_info = []

    for tau in candidates:
        pred = (val_scores >= tau).astype(int)
        tn = int(np.sum((val_y == 0) & (pred == 0)))
        fp = int(np.sum((val_y == 0) & (pred == 1)))
        fn = int(np.sum((val_y == 1) & (pred == 0)))
        tp = int(np.sum((val_y == 1) & (pred == 1)))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = 0.5 * (recall + spec)

        metrics = {
            "tau": float(tau),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "fpr": float(fpr),
            "recall": float(recall),
            "specificity": float(spec),
            "balanced_accuracy": float(bal_acc),
        }
        all_candidates_info.append(metrics)

        if fpr <= max_fpr:
            feasible_candidates.append(metrics)

    if feasible_candidates:
        # Sort by: recall (desc), balanced_acc (desc), tau (desc)
        feasible_candidates.sort(
            key=lambda m: (m["recall"], m["balanced_accuracy"], m["tau"]),
            reverse=True,
        )
        best = feasible_candidates[0]
    else:
        # Fallback: candidate with lowest FPR
        all_candidates_info.sort(key=lambda m: (m["fpr"], -m["recall"], -m["tau"]))
        best = all_candidates_info[0]

    return best["tau"], best


class KnnNormalDistanceDetector:
    """kNN normal distance detector supporting dual, cosine, or euclidean distance."""

    def __init__(self, k: int = 3, metric: str = "dual"):
        self.k = k
        self.metric = metric
        self.normal_log_embeds: Optional[np.ndarray] = None
        self.normal_patch_embeds: Optional[np.ndarray] = None
        self.normal_patch_empty: Optional[np.ndarray] = None
        self.normal_features: Optional[np.ndarray] = None

    def fit_embeddings(
        self,
        normal_log_embeds: np.ndarray,
        normal_patch_embeds: Optional[np.ndarray] = None,
        normal_patch_empty: Optional[np.ndarray] = None,
    ) -> KnnNormalDistanceDetector:
        self.normal_log_embeds = normal_log_embeds
        self.normal_patch_embeds = normal_patch_embeds
        self.normal_patch_empty = normal_patch_empty
        return self

    def fit_tabular(self, normal_features: np.ndarray) -> KnnNormalDistanceDetector:
        self.normal_features = normal_features
        return self

    def score_dual(
        self,
        query_log_embeds: np.ndarray,
        query_patch_embeds: np.ndarray,
        query_patch_empty: np.ndarray,
    ) -> np.ndarray:
        n_queries = query_log_embeds.shape[0]
        n_ref = self.normal_log_embeds.shape[0]
        scores = np.zeros(n_queries, dtype=np.float32)

        for i in range(n_queries):
            q_log = query_log_embeds[i]
            q_patch = query_patch_embeds[i]
            q_empty = query_patch_empty[i]

            dists = np.zeros(n_ref, dtype=np.float32)
            for j in range(n_ref):
                r_log = self.normal_log_embeds[j]
                r_patch = self.normal_patch_embeds[j]
                r_empty = self.normal_patch_empty[j]
                dists[j] = compute_dual_distance(q_log, r_log, q_patch, r_patch, q_empty, r_empty)

            dists.sort()
            effective_k = min(self.k, len(dists))
            scores[i] = dists[:effective_k].mean()

        return scores

    def score_single_cosine(
        self,
        query_embeds: np.ndarray,
        query_empty: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_queries = query_embeds.shape[0]
        n_ref = self.normal_log_embeds.shape[0]
        scores = np.zeros(n_queries, dtype=np.float32)

        for i in range(n_queries):
            q_vec = query_embeds[i]
            q_empty = query_empty[i] if query_empty is not None else False

            dists = np.zeros(n_ref, dtype=np.float32)
            for j in range(n_ref):
                r_vec = self.normal_log_embeds[j]
                r_empty = self.normal_patch_empty[j] if self.normal_patch_empty is not None else False
                dists[j] = compute_channel_distance(q_vec, r_vec, q_empty, r_empty)

            dists.sort()
            effective_k = min(self.k, len(dists))
            scores[i] = dists[:effective_k].mean()

        return scores

    def score_euclidean(self, query_features: np.ndarray) -> np.ndarray:
        n_queries = query_features.shape[0]
        scores = np.zeros(n_queries, dtype=np.float32)

        # Standard Euclidean distance to normal reference
        diffs = query_features[:, None, :] - self.normal_features[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)  # (N_query, N_ref)
        dists.sort(axis=1)

        effective_k = min(self.k, dists.shape[1])
        scores = dists[:, :effective_k].mean(axis=1)
        return scores.astype(np.float32)


class SupervisedLogisticDetector:
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
            max_iter=1000,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> SupervisedLogisticDetector:
        self.clf.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        probs = self.clf.predict_proba(X)
        # Class 1 is anomaly/unresolved
        if 1 in self.clf.classes_:
            idx = list(self.clf.classes_).index(1)
            return probs[:, idx].astype(np.float32)
        return np.zeros(X.shape[0], dtype=np.float32)


class IsolationForestDetector:
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None

    def fit(self, X_train_normal: np.ndarray) -> IsolationForestDetector:
        n_samples = X_train_normal.shape[0]
        max_samples = min(256, n_samples)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=False,
            random_state=self.random_state,
        )
        self.model.fit(X_train_normal)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Negative normality score: higher = more anomalous
        return (-self.model.score_samples(X)).astype(np.float32)


class ShallowDAEModule(nn.Module):
    def __init__(self, input_dim: int = 23, hidden_dims: List[int] = [16, 4, 16]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class ShallowDAEDetector:
    """Shallow DAE on R0-clean: 23 -> 16 -> 4 -> 16 -> 23 with separate behavior/patch losses."""

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 150,
        batch_size: int = 16,
        masking_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.masking_rate = masking_rate
        self.random_state = random_state
        self.model: Optional[ShallowDAEModule] = None

        # Feature indices
        self.behavior_indices = [R0_CLEAN_COLUMNS.index(c) for c in BEHAVIOR_FEATURES]
        self.patch_indices = [R0_CLEAN_COLUMNS.index(c) for c in PATCH_FEATURES]
        self.binary_indices = [R0_CLEAN_COLUMNS.index(c) for c in R0_CLEAN_BINARY_COLUMNS]
        self.numeric_indices = [i for i in range(23) if i not in self.binary_indices]

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Numeric loss: Huber delta=1
        huber = nn.HuberLoss(delta=1.0, reduction="none")
        bce = nn.BCEWithLogitsLoss(reduction="none")

        loss_per_col = torch.zeros_like(pred)
        if self.numeric_indices:
            loss_per_col[:, self.numeric_indices] = huber(
                pred[:, self.numeric_indices], target[:, self.numeric_indices]
            )
        if self.binary_indices:
            loss_per_col[:, self.binary_indices] = bce(
                pred[:, self.binary_indices], target[:, self.binary_indices]
            )

        # Equal-weight average of behavior (15) and patch (8)
        loss_behav = loss_per_col[:, self.behavior_indices].mean(dim=1)
        loss_patch = loss_per_col[:, self.patch_indices].mean(dim=1)
        total_loss = 0.5 * (loss_behav + loss_patch)
        return total_loss

    def fit(self, X_train_normal: np.ndarray) -> ShallowDAEDetector:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model = ShallowDAEModule(input_dim=23)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        n_samples = X_train_normal.shape[0]
        data = torch.tensor(X_train_normal, dtype=torch.float32)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            for b_start in range(0, n_samples, self.batch_size):
                batch_idx = perm[b_start : b_start + self.batch_size]
                target_b = data[batch_idx]

                # Independent noise mask (masking rate 0.1)
                corrupt_mask = torch.rand_like(target_b) < self.masking_rate
                input_b = target_b.clone()
                input_b[corrupt_mask] = 0.0

                optimizer.zero_grad()
                pred_b = self.model(input_b)
                loss = self._compute_loss(pred_b, target_b).mean()
                loss.backward()
                optimizer.step()

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        data = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(data)
            scores = self._compute_loss(pred, data).cpu().numpy()
        return scores.astype(np.float32)


class ConditionalResidualDetector:
    """Predicts 8 patch features from 15 behavior features on normal fitting rows."""

    def __init__(self, ridge_alpha: float = 10.0, logistic_c: float = 0.1):
        self.ridge_alpha = ridge_alpha
        self.logistic_c = logistic_c
        self.behavior_indices = [R0_CLEAN_COLUMNS.index(c) for c in BEHAVIOR_FEATURES]
        self.patch_indices = [R0_CLEAN_COLUMNS.index(c) for c in PATCH_FEATURES]

        self.models: Dict[str, Any] = {}

    def fit(self, X_train_normal: np.ndarray) -> ConditionalResidualDetector:
        X_b = X_train_normal[:, self.behavior_indices]

        for p_idx in self.patch_indices:
            col_name = R0_CLEAN_COLUMNS[p_idx]
            y_col = X_train_normal[:, p_idx]

            if col_name in R0_CLEAN_BINARY_COLUMNS:
                y_bin = (y_col > 0.5).astype(int)
                unique_y = np.unique(y_bin)
                if len(unique_y) > 1:
                    clf = LogisticRegression(
                        C=self.logistic_c, solver="liblinear"
                    )
                    clf.fit(X_b, y_bin)
                    self.models[col_name] = ("logistic", clf)
                else:
                    # Laplace smoothed constant probability
                    pos = int(np.sum(y_bin == 1))
                    n = len(y_bin)
                    p_const = (pos + 1.0) / (n + 2.0)
                    self.models[col_name] = ("constant", p_const)
            else:
                reg = Ridge(alpha=self.ridge_alpha)
                reg.fit(X_b, y_col)
                self.models[col_name] = ("ridge", reg)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_b = X[:, self.behavior_indices]
        n_samples = X.shape[0]
        residuals = np.zeros((n_samples, len(self.patch_indices)), dtype=np.float32)

        for i, p_idx in enumerate(self.patch_indices):
            col_name = R0_CLEAN_COLUMNS[p_idx]
            y_col = X[:, p_idx]
            m_type, model = self.models[col_name]

            if m_type == "ridge":
                pred = model.predict(X_b)
                # Huber loss with delta=1.0
                diff = np.abs(pred - y_col)
                huber = np.where(diff <= 1.0, 0.5 * (diff**2), diff - 0.5)
                residuals[:, i] = huber
            elif m_type == "logistic":
                probs = model.predict_proba(X_b)[:, 1]
                probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
                # BCE: - [y * log(p) + (1-y) * log(1-p)]
                bce = -(y_col * np.log(probs) + (1.0 - y_col) * np.log(1.0 - probs))
                residuals[:, i] = bce
            elif m_type == "constant":
                p = model
                bce = -(y_col * np.log(p) + (1.0 - y_col) * np.log(1.0 - p))
                residuals[:, i] = bce

        # Anomaly score is mean residual across 8 patch targets
        return residuals.mean(axis=1).astype(np.float32)


class RuleHeuristicDetector:
    """T1: Rule-based composite score from process heuristics."""

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> RuleHeuristicDetector:
        return self

    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        err_rate = df["error_observation_rate"].values
        streak = df["max_repeated_tool_streak"].values
        last_err = df["last_step_error"].values
        no_repro = (df["reproduce_script_created"] == 0).astype(float).values

        score = err_rate + 0.1 * streak + 1.0 * last_err + 0.5 * no_repro
        return score.astype(np.float32)


class LegacyXGBoostDetector:
    """R0 XGBoost supervised control."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=0.235,  # 54 / 230
            eval_metric="logloss",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LegacyXGBoostDetector:
        self.model.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1].astype(np.float32)


class LegacyAEDetector:
    """R0 Autoencoder legacy control (24 -> 16 -> 8 -> 16 -> 24)."""

    def __init__(self, epochs: int = 100, lr: float = 0.001, random_state: int = 42):
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.model: Optional[nn.Sequential] = None

    def fit(self, X_train_normal: np.ndarray) -> LegacyAEDetector:
        torch.manual_seed(self.random_state)
        self.model = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        data = torch.tensor(X_train_normal, dtype=torch.float32)
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(data)
            loss = criterion(out, data)
            loss.backward()
            optimizer.step()

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        data = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(data)
            err = torch.mean((out - data) ** 2, dim=1).cpu().numpy()
        return err.astype(np.float32)
