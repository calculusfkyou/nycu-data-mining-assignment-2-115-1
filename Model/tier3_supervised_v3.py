"""Tier 3 Strong Supervised Reference Models.

1. Supervised XGBoost with class-weight compensation (scale_pos_weight ~ 26.0).
2. Multimodal Supervised MLP with weighted Binary Cross-Entropy loss.
"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


class SupervisedXGBoost:
    """Supervised gradient boosted decision trees on fused multimodal features."""
    def __init__(self, scale_pos_weight: float = 25.9, seed: int = 42):
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            eval_metric="logloss",
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class SupervisedMultimodalMLP:
    """Supervised PyTorch MLP trained with class-weighted BCE loss."""
    def __init__(self, input_dim: int = 1164, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, pos_weight: float = 25.9, seed: int = 42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pos_weight = pos_weight
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        X_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        y_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)

        dataset = TensorDataset(torch.from_numpy(X_scaled), y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_scaled))
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return probs
