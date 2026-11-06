"""Tier 2 Control Models.

1. InductiveECOD: Non-parametric empirical cumulative distribution function scoring across tabular features.
2. StandardizedDualKNN: Distance to k-nearest normal neighbors in normalized embedding space.
3. TabularIsolationForest: Tree-based ensemble anomaly isolation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


class InductiveECOD:
    """Inductive ECOD (Li et al., TKDE 2022) for non-parametric outlier scoring."""
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.train_features_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None):
        # Expects 2D numpy array of shape (N, D)
        self.train_features_ = np.sort(X_train.copy(), axis=0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        N, D = self.train_features_.shape
        M = X.shape[0]
        scores = np.zeros(M, dtype=float)

        for d in range(D):
            col_train = self.train_features_[:, d]
            col_test = X[:, d]
            
            # Left tail: P(X <= x)
            left_counts = np.searchsorted(col_train, col_test, side="right")
            p_left = np.clip(left_counts / N, self.eps, 1.0 - self.eps)
            
            # Right tail: P(X >= x)
            right_counts = N - np.searchsorted(col_train, col_test, side="left")
            p_right = np.clip(right_counts / N, self.eps, 1.0 - self.eps)
            
            scores += (-np.log(p_left) - np.log(p_right))

        return scores / D


class StandardizedDualKNN:
    """Dual-space k-NN measuring distance to k nearest normal neighbors in multimodal space."""
    def __init__(self, k: int = 15):
        self.k = k
        self.nn_log_ = None
        self.nn_patch_ = None

    def fit(self, log_embs_train: np.ndarray, patch_embs_train: np.ndarray):
        self.nn_log_ = NearestNeighbors(n_neighbors=self.k, metric="cosine", algorithm="brute")
        self.nn_log_.fit(log_embs_train)

        self.nn_patch_ = NearestNeighbors(n_neighbors=self.k, metric="cosine", algorithm="brute")
        self.nn_patch_.fit(patch_embs_train)
        return self

    def score(self, log_embs: np.ndarray, patch_embs: np.ndarray) -> np.ndarray:
        dist_log, _ = self.nn_log_.kneighbors(log_embs)
        dist_patch, _ = self.nn_patch_.kneighbors(patch_embs)
        # Average distance to k nearest neighbors
        mean_log = np.mean(dist_log, axis=1)
        mean_patch = np.mean(dist_patch, axis=1)
        # Combined anomaly score
        return (mean_log + mean_patch) / 2.0


class TabularIsolationForest:
    """Isolation Forest fitted on normal tabular features."""
    def __init__(self, n_estimators: int = 200, contamination: float = 0.037, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None):
        self.model.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # Lower decision_function means more abnormal; negate to return anomaly score
        return -self.model.decision_function(X)
