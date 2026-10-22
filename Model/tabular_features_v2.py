"""Tabular feature preparation for HW2 v2 (R0 legacy and R0-clean) and inductive ECOD.

Specifications from Section 11:
1. R0 legacy: 24 process & output features as defined in v1.
2. R0-clean:
   - Drop redundant 'num_tool_calls' (leaving 23 features).
   - Apply log1p to 16 non-negative long-tailed features:
     num_steps, num_unique_tools, max_repeated_tool_streak, num_error_observations,
     consecutive_errors_max, stalled_turn_count, test_suite_run_count, total_cost_usd,
     tokens_sent, tokens_received, tokens_per_step, patch_char_length, patch_lines_added,
     patch_lines_removed, patch_num_files_changed, patch_hunk_count.
   - Retain 4 binary features as 0/1:
     reproduce_script_created, last_step_error, patch_is_empty, patch_touches_tests.
   - Retain 3 bounded features as original:
     tool_entropy, error_observation_rate, patch_edit_ratio.
   - Normal-only median/IQR scaling for non-binary features (fallback to SD if IQR==0, 1.0 if SD==0).
3. Inductive ECOD Variant:
   - Frozen fitting ECDF & skewness on normal training reference.
   - Finite sample smoothing: pL=(1+count(train<=x))/(n+1), pR=(1+count(train>=x))/(n+1).
   - Invariant to test batch size, order, and composition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


FEATURES_CSV = Path("Dataset/sweagent_gpt4_features.csv")

R0_LEGACY_COLUMNS = [
    "num_steps",
    "num_tool_calls",
    "num_unique_tools",
    "tool_entropy",
    "max_repeated_tool_streak",
    "num_error_observations",
    "error_observation_rate",
    "consecutive_errors_max",
    "stalled_turn_count",
    "reproduce_script_created",
    "test_suite_run_count",
    "last_step_error",
    "total_cost_usd",
    "tokens_sent",
    "tokens_received",
    "tokens_per_step",
    "patch_char_length",
    "patch_lines_added",
    "patch_lines_removed",
    "patch_num_files_changed",
    "patch_is_empty",
    "patch_touches_tests",
    "patch_edit_ratio",
    "patch_hunk_count",
]

R0_CLEAN_LOG1P_COLUMNS = [
    "num_steps",
    "num_unique_tools",
    "max_repeated_tool_streak",
    "num_error_observations",
    "consecutive_errors_max",
    "stalled_turn_count",
    "test_suite_run_count",
    "total_cost_usd",
    "tokens_sent",
    "tokens_received",
    "tokens_per_step",
    "patch_char_length",
    "patch_lines_added",
    "patch_lines_removed",
    "patch_num_files_changed",
    "patch_hunk_count",
]

R0_CLEAN_BINARY_COLUMNS = [
    "reproduce_script_created",
    "last_step_error",
    "patch_is_empty",
    "patch_touches_tests",
]

R0_CLEAN_BOUNDED_COLUMNS = [
    "tool_entropy",
    "error_observation_rate",
    "patch_edit_ratio",
]

# Total 16 + 4 + 3 = 23 columns
R0_CLEAN_COLUMNS = (
    R0_CLEAN_LOG1P_COLUMNS + R0_CLEAN_BINARY_COLUMNS + R0_CLEAN_BOUNDED_COLUMNS
)

# 15 behavior features and 8 patch features for conditional baseline & DAE feature grouping
BEHAVIOR_FEATURES = [
    "num_steps",
    "num_unique_tools",
    "max_repeated_tool_streak",
    "num_error_observations",
    "consecutive_errors_max",
    "stalled_turn_count",
    "test_suite_run_count",
    "total_cost_usd",
    "tokens_sent",
    "tokens_received",
    "tokens_per_step",
    "reproduce_script_created",
    "last_step_error",
    "tool_entropy",
    "error_observation_rate",
]

PATCH_FEATURES = [
    "patch_char_length",
    "patch_lines_added",
    "patch_lines_removed",
    "patch_num_files_changed",
    "patch_hunk_count",
    "patch_is_empty",
    "patch_touches_tests",
    "patch_edit_ratio",
]


def load_raw_features_df() -> pd.DataFrame:
    """Load raw features DataFrame sorted by instance_id."""
    df = pd.read_csv(FEATURES_CSV)
    df = df.sort_values("instance_id").reset_index(drop=True)
    return df


def get_r0_legacy_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Return R0 legacy feature matrix (24 columns)."""
    X = df[R0_LEGACY_COLUMNS].to_numpy(dtype=np.float32)
    return X, list(R0_LEGACY_COLUMNS)


def get_r0_clean_unscaled_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Return R0-clean unscaled feature matrix (23 columns) with log1p applied."""
    clean_df = pd.DataFrame(index=df.index)

    # 1. Apply log1p to non-negative skewed features
    for col in R0_CLEAN_LOG1P_COLUMNS:
        val = df[col].clip(lower=0)
        clean_df[col] = np.log1p(val)

    # 2. Keep binary features as 0/1
    for col in R0_CLEAN_BINARY_COLUMNS:
        clean_df[col] = df[col].astype(np.float32)

    # 3. Keep bounded features as original
    for col in R0_CLEAN_BOUNDED_COLUMNS:
        clean_df[col] = df[col].astype(np.float32)

    # Reorder by R0_CLEAN_COLUMNS
    clean_df = clean_df[R0_CLEAN_COLUMNS]
    X = clean_df.to_numpy(dtype=np.float32)
    return X, list(R0_CLEAN_COLUMNS)


class NormalOnlyRobustScaler:
    """Robust scaler fitted strictly on normal training rows using median and IQR.

    Fallback rule: if IQR == 0, use SD. If SD == 0, scale = 1.0.
    Binary features are NOT scaled (remain 0/1).
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.medians: Optional[np.ndarray] = None
        self.scales: Optional[np.ndarray] = None
        self.is_binary: np.ndarray = np.array(
            [col in R0_CLEAN_BINARY_COLUMNS for col in feature_names], dtype=bool
        )

    def fit(self, X_train_normal: np.ndarray) -> NormalOnlyRobustScaler:
        assert X_train_normal.shape[1] == len(self.feature_names)
        n = X_train_normal.shape[0]
        assert n > 0, "Cannot fit scaler on empty normal training set"

        q25 = np.percentile(X_train_normal, 25, axis=0)
        q50 = np.percentile(X_train_normal, 50, axis=0)
        q75 = np.percentile(X_train_normal, 75, axis=0)
        iqr = q75 - q25

        sd = np.std(X_train_normal, axis=0)

        medians = q50.copy()
        scales = np.ones_like(medians)

        for j in range(len(self.feature_names)):
            if self.is_binary[j]:
                medians[j] = 0.0
                scales[j] = 1.0
            else:
                if iqr[j] > 1e-8:
                    scales[j] = iqr[j]
                elif sd[j] > 1e-8:
                    scales[j] = sd[j]
                else:
                    scales[j] = 1.0

        self.medians = medians.astype(np.float32)
        self.scales = scales.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.medians is not None and self.scales is not None
        X_scaled = (X - self.medians) / self.scales
        return X_scaled.astype(np.float32)

    def fit_transform(self, X_train_normal: np.ndarray) -> np.ndarray:
        return self.fit(X_train_normal).transform(X_train_normal)


class InductiveECODVariant:
    """Inductive ECOD variant with frozen ECDF and skewness fitted strictly on normal reference.

    Score formulation:
      For each column j:
        pL = (1 + sum(train_normal <= x)) / (n + 1)
        pR = (1 + sum(train_normal >= x)) / (n + 1)
        UL = -log(pL)
        UR = -log(pR)
        selected = UR if skewness > 0 else (UL if skewness < 0 else UL + UR)
        col_score = max(UL, UR, selected)
      total_score = sum(col_score)

    Strict property: Invariant to test batch size, order, and composition.
    """

    def __init__(self):
        self.train_data: Optional[np.ndarray] = None
        self.skewness: Optional[np.ndarray] = None
        self.n_samples: int = 0
        self.n_features: int = 0

    def fit(self, X_train_normal: np.ndarray) -> InductiveECODVariant:
        assert X_train_normal.ndim == 2
        self.n_samples, self.n_features = X_train_normal.shape
        assert self.n_samples > 0, "Training reference cannot be empty"

        # Store sorted columns for fast ECDF evaluation
        self.train_data = X_train_normal.copy()
        # Compute empirical skewness per column
        # stats.skew with bias=False (unbiased) or default
        col_skew = stats.skew(X_train_normal, axis=0, nan_policy="omit")
        # Replace NaN skewness (constant columns) with 0.0
        col_skew = np.nan_to_num(col_skew, nan=0.0)
        self.skewness = col_skew
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score for each row in X. Higher score = more anomalous."""
        assert self.train_data is not None and self.skewness is not None
        n = self.n_samples
        n_test = X.shape[0]

        total_scores = np.zeros(n_test, dtype=np.float32)

        for j in range(self.n_features):
            train_col = self.train_data[:, j]
            test_col = X[:, j]
            skew_j = self.skewness[j]

            # Vectorized ECDF evaluation:
            # count(train <= x) = sum(train_col[None, :] <= test_col[:, None], axis=1)
            # count(train >= x) = sum(train_col[None, :] >= test_col[:, None], axis=1)
            count_le = np.sum(train_col[None, :] <= test_col[:, None], axis=1)
            count_ge = np.sum(train_col[None, :] >= test_col[:, None], axis=1)

            pL = (1.0 + count_le) / (n + 1.0)
            pR = (1.0 + count_ge) / (n + 1.0)

            UL = -np.log(pL)
            UR = -np.log(pR)

            if skew_j > 1e-6:
                selected = UR
            elif skew_j < -1e-6:
                selected = UL
            else:
                selected = UL + UR

            col_score = np.maximum(np.maximum(UL, UR), selected)
            total_scores += col_score.astype(np.float32)

        return total_scores
