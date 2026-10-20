"""3-Tier Anomaly Detection Models for AI Agent Ghostwork Detector.

Tier 1: Naive Heuristic Baseline (Error count & Tool loop streak threshold)
Tier 2: Primary Method (Deep Bottleneck Autoencoder trained on Normal-only)
Tier 3: Strong Reference Benchmark (Supervised XGBoost Classifier)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ==============================================================================
# Tier 1: Naive Heuristic Baseline
# ==============================================================================

class NaiveHeuristicDetector:
    """Rule-based heuristic detector adapted from ATFD benchmark.

    Flags an anomaly if:
    - Number of error observations exceeds error_threshold (default: >= 3) OR
    - Maximum tool repeat streak exceeds loop_threshold (default: > 5)
    """

    def __init__(self, error_threshold: int = 3, loop_threshold: int = 5) -> None:
        self.error_threshold = error_threshold
        self.loop_threshold = loop_threshold

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NaiveHeuristicDetector:
        """Heuristic baseline has no trainable parameters."""
        return self

    def score_samples(self, X_df: Any) -> np.ndarray:
        """Anomaly score is a composite error count and loop severity score."""
        errors = X_df["num_error_observations"].values
        streaks = X_df["max_repeated_tool_streak"].values
        scores = (errors / (self.error_threshold + 1e-5)) * 0.5 + (
            streaks / (self.loop_threshold + 1e-5)
        ) * 0.5
        return scores

    def predict(self, X_df: Any) -> np.ndarray:
        scores = self.score_samples(X_df)
        preds = (scores >= 1.0).astype(int)
        return preds


# ==============================================================================
# Tier 2: Deep Bottleneck Autoencoder (Primary Method)
# ==============================================================================

class AutoencoderModule(nn.Module):
    """PyTorch Bottleneck Autoencoder architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, latent_dim: int = 6) -> None:
        super().__init__()
        # Adjust dimensions if input_dim is small
        hidden_dim = min(hidden_dim, max(4, input_dim))
        latent_dim = max(2, min(latent_dim, hidden_dim // 2))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector:
    """Primary Anomaly Detector trained exclusively on normal (resolved) trajectories."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 6,
        lr: float = 0.005,
        epochs: int = 150,
        batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model: AutoencoderModule | None = None
        self.scaler: StandardScaler | None = None
        self.threshold: float = 1.0

    def fit(self, X_train: np.ndarray) -> AutoencoderDetector:
        """Fit scaler and train autoencoder strictly on normal data."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = AutoencoderModule(
            self.input_dim, self.hidden_dim, self.latent_dim
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.batch_size, len(X_train)), shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score s(x) = mean squared reconstruction error."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
            recon = self.model(tensor_x)
            diff = tensor_x - recon
            mse = torch.mean(diff**2, dim=1).numpy()
        return mse

    def tune_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray, beta: float = 2.0
    ) -> float:
        """Tune decision threshold tau strictly on validation set to maximize F_beta."""
        scores = self.score_samples(X_val)
        best_fbeta = -1.0
        best_tau = float(np.median(scores))

        threshold_candidates = np.percentile(scores, np.linspace(5, 95, 91))
        for tau in threshold_candidates:
            preds = (scores >= tau).astype(int)
            fb = fbeta_score(y_val, preds, beta=beta, zero_division=0)
            if fb > best_fbeta:
                best_fbeta = fb
                best_tau = float(tau)

        self.threshold = best_tau
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return (scores >= self.threshold).astype(int)


# ==============================================================================
# Tier 3: Strong Reference Benchmark (Supervised XGBoost)
# ==============================================================================

class SupervisedReferenceDetector:
    """Strong supervised benchmark trained on both normal and anomaly classes."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=1.0,
            random_state=seed,
            eval_metric="logloss",
        )
        self.threshold = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SupervisedReferenceDetector:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score is the predicted probability of failure / ghostwork."""
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return probs

    def tune_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray, beta: float = 2.0
    ) -> float:
        """Tune decision threshold strictly on validation set."""
        probs = self.score_samples(X_val)
        best_fbeta = -1.0
        best_tau = 0.5
        for tau in np.linspace(0.1, 0.9, 81):
            preds = (probs >= tau).astype(int)
            fb = fbeta_score(y_val, preds, beta=beta, zero_division=0)
            if fb > best_fbeta:
                best_fbeta = fb
                best_tau = float(tau)
        self.threshold = best_tau
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.score_samples(X)
        return (probs >= self.threshold).astype(int)
