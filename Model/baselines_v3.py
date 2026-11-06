"""Tier 1 Naive Baseline Heuristics.

Includes:
1. Predict-all-normal (T1-A): Always predicts 0 / anomaly score 0.
2. 3-Sigma Patch Churn (T1-B): Standardizes total edit lines against normal training mean/std.
3. Unverified-Edit Heuristic (T1-C): Computes unverified edit ratio * cross-module dispersion.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class AllNormalBaseline:
    """Predicts all instances as normal (y_hat = 0, score = 0)."""
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray = None):
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


class PatchChurnBaseline:
    """Flags anomalies based on total modified lines exceeding 3 standard deviations above normal mean."""
    def __init__(self, sigma_threshold: float = 3.0):
        self.sigma_threshold = sigma_threshold
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray = None):
        churn = X_train["edit_lines_total"].values
        self.mean_ = float(np.mean(churn))
        self.std_ = float(np.std(churn)) if float(np.std(churn)) > 1e-6 else 1.0
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        churn = X["edit_lines_total"].values
        z_scores = (churn - self.mean_) / self.std_
        return np.maximum(0.0, z_scores)

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        thresh = self.sigma_threshold if threshold is None else threshold
        scores = self.score(X)
        return (scores >= thresh).astype(int)


class UnverifiedEditHeuristic:
    """Multiplies unverified edit ratio by cross-module dispersion."""
    def __init__(self):
        self.max_score_ = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray = None):
        scores = X_train["unverified_edit_ratio"].values * X_train["cross_module_dispersion"].values
        self.max_score_ = float(np.max(scores)) if float(np.max(scores)) > 0 else 1.0
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raw_scores = X["unverified_edit_ratio"].values * X["cross_module_dispersion"].values
        return raw_scores / self.max_score_

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        scores = self.score(X)
        return (scores >= threshold).astype(int)
