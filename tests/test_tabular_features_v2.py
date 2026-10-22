"""Unit tests for R0 tabular features, normal-only robust scaling, and Inductive ECOD invariance."""

import numpy as np
import pytest
from Model.tabular_features_v2 import (
    BEHAVIOR_FEATURES,
    PATCH_FEATURES,
    R0_CLEAN_COLUMNS,
    R0_LEGACY_COLUMNS,
    InductiveECODVariant,
    NormalOnlyRobustScaler,
    get_r0_clean_unscaled_features,
    get_r0_legacy_features,
    load_raw_features_df,
)


def test_tabular_column_counts():
    df = load_raw_features_df()
    X_legacy, leg_cols = get_r0_legacy_features(df)
    assert X_legacy.shape == (284, 24)
    assert len(leg_cols) == 24

    X_clean, clean_cols = get_r0_clean_unscaled_features(df)
    assert X_clean.shape == (284, 23)
    assert len(clean_cols) == 23
    assert "num_tool_calls" not in clean_cols

    # Verify grouping
    assert len(BEHAVIOR_FEATURES) == 15
    assert len(PATCH_FEATURES) == 8
    assert set(BEHAVIOR_FEATURES + PATCH_FEATURES) == set(clean_cols)


def test_normal_only_robust_scaler():
    df = load_raw_features_df()
    X_clean, clean_cols = get_r0_clean_unscaled_features(df)

    # 54 normal instances (label == 0, is_resolved == 1)
    normal_mask = (df["is_resolved"] == 1).to_numpy()
    X_normal = X_clean[normal_mask]

    scaler = NormalOnlyRobustScaler(clean_cols)
    scaler.fit(X_normal)

    # Transform
    X_scaled = scaler.transform(X_clean)
    assert X_scaled.shape == (284, 23)

    # Verify binary columns are unchanged
    binary_indices = [i for i, col in enumerate(clean_cols) if col in ["patch_is_empty", "last_step_error"]]
    for idx in binary_indices:
        assert np.array_equal(X_scaled[:, idx], X_clean[:, idx])


def test_inductive_ecod_batch_invariance():
    """Verify that inductive ECOD variant scores are completely invariant to batch size and order."""
    df = load_raw_features_df()
    X_clean, _ = get_r0_clean_unscaled_features(df)
    normal_mask = (df["is_resolved"] == 1).to_numpy()
    X_normal = X_clean[normal_mask]

    ecod = InductiveECODVariant()
    ecod.fit(X_normal)

    # Pick 10 test samples
    test_subset = X_clean[:10]

    # Evaluate batch together
    batch_scores = ecod.decision_function(test_subset)

    # Evaluate each sample individually
    individual_scores = np.array([ecod.decision_function(test_subset[i : i + 1])[0] for i in range(10)])

    # Evaluate reversed batch
    reversed_scores = ecod.decision_function(test_subset[::-1])[::-1]

    # Assert exact invariance
    np.testing.assert_allclose(batch_scores, individual_scores, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(batch_scores, reversed_scores, rtol=1e-5, atol=1e-5)
