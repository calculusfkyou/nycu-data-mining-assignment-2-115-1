"""Unit tests for v2 experiment splits and text manifests."""

import json
from pathlib import Path
import pytest


def test_splits_integrity():
    splits_path = Path("Dataset/v2/splits.json")
    assert splits_path.exists()

    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["split_seed"] == 42
    assert data["n_outer_folds"] == 5
    outer_folds = data["outer_folds"]
    assert len(outer_folds) == 5

    all_test_ids = []
    for fold in outer_folds:
        train = set(fold["train_instance_ids"])
        val = set(fold["val_instance_ids"])
        test = set(fold["test_instance_ids"])

        # Pairwise disjoint
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

        # Total partition size
        assert len(train) + len(val) + len(test) == 284

        # Normal fitting instances must be a subset of train
        train_normal = set(fold["train_normal_instance_ids"])
        assert train_normal.issubset(train)

        all_test_ids.extend(fold["test_instance_ids"])

    # Outer test sets are disjoint and cover all 284 instances exactly once
    assert len(all_test_ids) == 284
    assert len(set(all_test_ids)) == 284


def test_manifest_integrity():
    manifest_path = Path("Dataset/v2/text_manifest.jsonl")
    assert manifest_path.exists()

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 284

    # Check labels and counts
    resolved_count = sum(1 for r in records if r["label"] == 0)
    unresolved_count = sum(1 for r in records if r["label"] == 1)
    assert resolved_count == 54
    assert unresolved_count == 230

    # Ensure all have valid sha256 (64 hex characters)
    for r in records:
        assert len(r["sha256"]) == 64
        assert 0 <= r["outer_fold"] <= 4
