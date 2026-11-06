"""Tier 2 Primary Method (HERO): Calibrated Multimodal Bottleneck Autoencoder (CM-BAE).

Key Innovations:
1. Variance-Equalized Multimodal Concatenation (structural R0 + MiniLM logs + CodeBERT diffs).
2. Robust Huber Loss training strictly on y=0 normal candidates.
3. Additive Channel Attribution (struct, log, patch) for full analyst interpretability.
4. Platt Posterior Calibration and Cost-Optimal CI/CD Triage Thresholding.
5. Deterministic reproducibility across 5 seeds (42, 123, 456, 789, 1011).
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BAEArchitecture(nn.Module):
    """Bottleneck Autoencoder with batch normalization and leaky ReLU."""
    def __init__(self, input_dim: int = 1164, bottleneck_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class CMBAE:
    """Calibrated Multimodal Bottleneck Autoencoder detector."""
    def __init__(
        self,
        bottleneck_dim: int = 64,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
        delta: float = 1.0,
        seed: int = 42,
    ):
        self.bottleneck_dim = bottleneck_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.delta = delta
        self.seed = seed

        self.scaler_struct = StandardScaler()
        self.alpha_struct = 1.0
        self.alpha_log = 1.0
        self.alpha_patch = 1.0

        self.dim_struct = 12
        self.dim_log = 384
        self.dim_patch = 768
        self.total_dim = 12 + 384 + 768

        self.model: BAEArchitecture | None = None
        self.calibrator: LogisticRegression | None = None
        self.optimal_threshold: float = 0.5
        self.cost_optimal_threshold: float = 0.5

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _prepare_inputs(
        self,
        X_struct: np.ndarray,
        X_log: np.ndarray,
        X_patch: np.ndarray,
        is_fit: bool = False,
    ) -> np.ndarray:
        if is_fit:
            s_scaled = self.scaler_struct.fit_transform(X_struct)
            # Variance equalization factors
            var_s = np.mean(np.var(s_scaled, axis=0))
            var_l = np.mean(np.var(X_log, axis=0))
            var_p = np.mean(np.var(X_patch, axis=0))

            self.alpha_struct = 1.0 / np.sqrt(max(var_s, 1e-6))
            self.alpha_log = 1.0 / np.sqrt(max(var_l, 1e-6))
            self.alpha_patch = 1.0 / np.sqrt(max(var_p, 1e-6))
        else:
            s_scaled = self.scaler_struct.transform(X_struct)

        # Apply modality scaling
        s_part = s_scaled * self.alpha_struct
        l_part = X_log * self.alpha_log
        p_part = X_patch * self.alpha_patch

        return np.hstack([s_part, l_part, p_part]).astype(np.float32)

    def fit(
        self,
        X_struct_norm: np.ndarray,
        X_log_norm: np.ndarray,
        X_patch_norm: np.ndarray,
    ):
        """Fits autoencoder strictly on y=0 normal instances."""
        self._set_seed()
        self.dim_struct = X_struct_norm.shape[1]
        self.dim_log = X_log_norm.shape[1]
        self.dim_patch = X_patch_norm.shape[1]
        self.total_dim = self.dim_struct + self.dim_log + self.dim_patch

        Z = self._prepare_inputs(X_struct_norm, X_log_norm, X_patch_norm, is_fit=True)

        dataset = TensorDataset(torch.from_numpy(Z))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model = BAEArchitecture(input_dim=self.total_dim, bottleneck_dim=self.bottleneck_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.HuberLoss(delta=self.delta)

        self.model.train()
        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.model.eval()
        return self

    def score_detailed(
        self,
        X_struct: np.ndarray,
        X_log: np.ndarray,
        X_patch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Computes reconstruction error with additive channel decomposition."""
        self.model.eval()
        Z = self._prepare_inputs(X_struct, X_log, X_patch, is_fit=False)
        with torch.no_grad():
            Z_t = torch.from_numpy(Z)
            Z_recon = self.model(Z_t).cpu().numpy()

        diff_sq = (Z - Z_recon) ** 2
        
        # Split into structural, log, and patch modalities
        d_s = self.dim_struct
        d_l = self.dim_log
        d_p = self.dim_patch

        err_struct = np.mean(diff_sq[:, :d_s], axis=1)
        err_log = np.mean(diff_sq[:, d_s : d_s + d_l], axis=1)
        err_patch = np.mean(diff_sq[:, d_s + d_l :], axis=1)

        total_err = err_struct + err_log + err_patch

        return {
            "total_score": total_err,
            "err_struct": err_struct,
            "err_log": err_log,
            "err_patch": err_patch,
        }

    def score(
        self,
        X_struct: np.ndarray,
        X_log: np.ndarray,
        X_patch: np.ndarray,
    ) -> np.ndarray:
        return self.score_detailed(X_struct, X_log, X_patch)["total_score"]

    def calibrate(
        self,
        X_struct_val: np.ndarray,
        X_log_val: np.ndarray,
        X_patch_val: np.ndarray,
        y_val: np.ndarray,
        c_escaped: float = 1200.0,
        c_audit: float = 30.0,
    ):
        """Calibrates scores and selects cost-optimal threshold on validation split."""
        raw_scores = self.score(X_struct_val, X_log_val, X_patch_val)
        
        # Logistic calibration
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        self.calibrator.fit(raw_scores.reshape(-1, 1), y_val)

        probs = self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

        # Scan candidate thresholds
        cand_thresholds = np.linspace(0.01, 0.99, 100)
        best_f2 = -1.0
        best_f2_thresh = 0.5
        best_savings = -float("inf")
        best_cost_thresh = 0.5

        for th in cand_thresholds:
            preds = (probs >= th).astype(int)
            tp = int(np.sum((preds == 1) & (y_val == 1)))
            fp = int(np.sum((preds == 1) & (y_val == 0)))
            fn = int(np.sum((preds == 0) & (y_val == 1)))

            # F2 score: beta=2 weights recall 4x more than precision
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f2 = (5.0 * prec * rec) / max(4.0 * prec + rec, 1e-6)

            if f2 > best_f2:
                best_f2 = f2
                best_f2_thresh = th

            # Net CI/CD triage savings
            # Baseline cost with no detection: c_escaped * (TP + FN)
            # Cost with detection: c_audit * (TP + FP) + c_escaped * FN
            # Net savings: c_escaped * TP - c_audit * (TP + FP)
            savings = c_escaped * tp - c_audit * (tp + fp)
            if savings > best_savings:
                best_savings = savings
                best_cost_thresh = th

        self.optimal_threshold = best_f2_thresh
        self.cost_optimal_threshold = best_cost_thresh
        return self

    def predict_proba(
        self,
        X_struct: np.ndarray,
        X_log: np.ndarray,
        X_patch: np.ndarray,
    ) -> np.ndarray:
        raw_scores = self.score(X_struct, X_log, X_patch)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        # Min-max scale if not calibrated
        return (raw_scores - np.min(raw_scores)) / max(np.max(raw_scores) - np.min(raw_scores), 1e-6)
