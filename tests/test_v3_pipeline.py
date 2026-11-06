"""Phase 1 Pipeline & Hygiene Verification Tests.

Verifies:
1. Manifest integrity and candidate filtering
2. Feature CSV row counts and column completeness
3. Embedding shapes and non-triviality
4. 5-Fold Stratified Group K-Fold split hygiene (zero group leakage, valid proportions)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_manifest_and_counts():
    manifest_path = Path("Dataset/verified/manifest.json")
    assert manifest_path.exists(), "Manifest file missing"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["total_records_count"] == 3998
    assert manifest["candidate_records_count"] == 3023
    assert manifest["deceptive_regression_count"] == 114


def test_features_and_labels():
    csv_path = Path("Dataset/verified/candidate_features.csv")
    assert csv_path.exists(), "Feature CSV missing"
    df = pd.read_csv(csv_path)
    assert len(df) == 2663, f"Expected 2663 candidate rows, got {len(df)}"
    assert "label" in df.columns
    n_anom = int(df["label"].sum())
    n_norm = int((df["label"] == 0).sum())
    assert n_anom == 99, f"Expected 99 anomalies, got {n_anom}"
    assert n_norm == 2564, f"Expected 2564 normals, got {n_norm}"
    prevalence = n_anom / len(df)
    assert 0.035 <= prevalence <= 0.040, f"Anomaly rate {prevalence:.4f} not within 3.5%-4.0%"


def test_embedding_dimensions():
    minilm_path = Path("Dataset/verified/embeddings/minilm_logs.npy")
    codebert_path = Path("Dataset/verified/embeddings/codebert_diffs.npy")
    assert minilm_path.exists(), "MiniLM embeddings missing"
    assert codebert_path.exists(), "CodeBERT embeddings missing"
    
    m_emb = np.load(minilm_path)
    c_emb = np.load(codebert_path)
    
    assert m_emb.shape == (2663, 384), f"Wrong MiniLM shape: {m_emb.shape}"
    assert c_emb.shape == (2663, 768), f"Wrong CodeBERT shape: {c_emb.shape}"
    assert not np.isnan(m_emb).any(), "NaN in MiniLM embeddings"
    assert not np.isnan(c_emb).any(), "NaN in CodeBERT embeddings"


def test_grouped_splits_hygiene():
    splits_path = Path("Dataset/verified/splits.json")
    assert splits_path.exists(), "Splits file missing"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    assert splits["n_splits"] == 5
    assert splits["total_samples"] == 2663
    
    csv_path = Path("Dataset/verified/candidate_features.csv")
    df = pd.read_csv(csv_path)
    
    all_test_indices = []
    for fold in splits["folds"]:
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]
        test_idx = fold["test_indices"]
        train_norm_idx = fold["train_normal_indices"]
        
        # Test disjointness of sample indices within fold
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(set(val_idx).intersection(set(test_idx))) == 0
        
        # Test normal training indices subset
        assert set(train_norm_idx).issubset(set(train_idx))
        for idx in train_norm_idx:
            assert df.iloc[idx]["label"] == 0, f"Index {idx} in train_normal has label {df.iloc[idx]['label']}"
            
        # Test zero instance_id leakage across splits
        train_groups = set(df.iloc[train_idx]["instance_id"])
        val_groups = set(df.iloc[val_idx]["instance_id"])
        test_groups = set(df.iloc[test_idx]["instance_id"])
        
        assert len(train_groups.intersection(val_groups)) == 0, "Train-Val group leakage!"
        assert len(train_groups.intersection(test_groups)) == 0, "Train-Test group leakage!"
        assert len(val_groups.intersection(test_groups)) == 0, "Val-Test group leakage!"
        
        all_test_indices.extend(test_idx)
        
    # Test partition covers all samples
    assert len(all_test_indices) == 2663
    assert len(set(all_test_indices)) == 2663
