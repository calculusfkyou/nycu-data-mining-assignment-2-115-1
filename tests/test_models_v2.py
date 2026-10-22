"""Unit tests for models_v2 detectors and validation threshold policy."""

import numpy as np
import pytest
from Model.models_v2 import (
    ConditionalResidualDetector,
    IsolationForestDetector,
    KnnNormalDistanceDetector,
    ShallowDAEDetector,
    SupervisedLogisticDetector,
    select_validation_threshold,
)


def test_validation_threshold_policy():
    val_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 10 normal, 5 anomaly
    # Normal scores: mostly low (0.1 to 0.4) with one high (0.8)
    # Anomaly scores: mostly high (0.6 to 0.9)
    val_scores = np.array([
        0.1, 0.15, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.42, 0.8,
        0.5, 0.65, 0.7, 0.85, 0.9,
    ])

    best_tau, metrics = select_validation_threshold(val_y, val_scores, max_fpr=0.20)

    # 10 normal samples: FPR <= 20% means FP <= 2
    assert metrics["fpr"] <= 0.20
    assert metrics["tn"] >= 8
    # Anomaly recall should be maximized
    assert metrics["recall"] >= 0.80
    assert best_tau > 0.4


def test_knn_detector_missing_patch_distance():
    # 2 normal reference items
    normal_log = np.ones((2, 384), dtype=np.float32) / np.sqrt(384)
    normal_patch = np.ones((2, 768), dtype=np.float32) / np.sqrt(768)
    normal_empty = np.array([False, False])

    knn = KnnNormalDistanceDetector(k=1, metric="dual")
    knn.fit_embeddings(normal_log, normal_patch, normal_empty)

    # Query 1: identical to normal -> distance should be 0.0
    q_log_same = normal_log[0:1]
    q_patch_same = normal_patch[0:1]
    q_empty_false = np.array([False])
    s_same = knn.score_dual(q_log_same, q_patch_same, q_empty_false)
    assert np.isclose(s_same[0], 0.0, atol=1e-5)

    # Query 2: log identical, but patch is empty -> d_log = 0.0, d_patch = 1.0 -> d_dual = 0.5
    s_patch_empty = knn.score_dual(q_log_same, np.zeros((1, 768), dtype=np.float32), np.array([True]))
    assert np.isclose(s_patch_empty[0], 0.5, atol=1e-5)


def test_supervised_logistic_detector():
    X = np.random.randn(20, 10).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)
    clf = SupervisedLogisticDetector(C=1.0)
    clf.fit(X, y)
    scores = clf.decision_function(X)
    assert scores.shape == (20,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_shallow_dae_smoke():
    X_normal = np.random.randn(15, 23).astype(np.float32)
    # Ensure binary columns (indices 16, 17, 18, 19) are 0 or 1
    for b_idx in [16, 17, 18, 19]:
        X_normal[:, b_idx] = np.random.choice([0.0, 1.0], size=15)

    dae = ShallowDAEDetector(epochs=5, batch_size=8)
    dae.fit(X_normal)
    scores = dae.decision_function(X_normal)
    assert scores.shape == (15,)
    assert (scores >= 0.0).all()


def test_conditional_residual_smoke():
    X_normal = np.random.randn(15, 23).astype(np.float32)
    for b_idx in [16, 17, 18, 19]:
        X_normal[:, b_idx] = np.random.choice([0.0, 1.0], size=15)

    cond = ConditionalResidualDetector(ridge_alpha=1.0, logistic_c=1.0)
    cond.fit(X_normal)
    scores = cond.decision_function(X_normal)
    assert scores.shape == (15,)
    assert (scores >= 0.0).all()
