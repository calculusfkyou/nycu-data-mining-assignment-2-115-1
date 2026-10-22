"""Generate Dataset/v2/splits.json and Dataset/v2/text_manifest.jsonl.

Strict requirements:
1. 5-fold stratified outer CV on all 284 instances (split seed = 42).
2. Per-fold inner validation split: 25% of training partition (seed = 100 + fold), stratified.
3. Every instance has exactly one outer-test assignment.
4. Repository-held-out split definitions.
5. text_manifest.jsonl recording metadata, raw SHA-256, character counts, and missingness masks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROVENANCE_PATH = Path("Dataset/provenance.json")
FEATURES_CSV = Path("Dataset/sweagent_gpt4_features.csv")
RAW_TRAJ_DIR = Path("Dataset/raw/sweagent_gpt4")
SPLITS_PATH = Path("Dataset/v2/splits.json")
MANIFEST_PATH = Path("Dataset/v2/text_manifest.jsonl")


def build_manifests() -> None:
    print("Building v2 splits and text manifest...")
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov_map = {item["instance_id"]: item["sha256"] for item in prov["trajectories"]}

    df = pd.read_csv(FEATURES_CSV)
    # Ensure consistent ordering
    df = df.sort_values("instance_id").reset_index(drop=True)
    n = len(df)
    assert n == 284, f"Expected 284 instances, got {n}"

    # Label: y=0 for resolved (success, normal), y=1 for unresolved (failure, anomaly)
    labels = np.array([0 if r == 1 else 1 for r in df["is_resolved"].values])
    instance_ids = df["instance_id"].values
    repos = df["repo"].values

    # 1. 5-Fold Stratified Outer CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_folds = []
    outer_test_assignment = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(instance_ids, labels)):
        # Inner validation split (25% of train_val_idx)
        train_val_labels = labels[train_val_idx]
        train_sub_idx, val_sub_idx = train_test_split(
            train_val_idx,
            test_size=0.25,
            random_state=100 + fold_idx,
            stratify=train_val_labels,
        )

        # Normal-only subsets (y=0)
        train_normal_idx = [i for i in train_sub_idx if labels[i] == 0]
        val_normal_idx = [i for i in val_sub_idx if labels[i] == 0]

        fold_dict = {
            "fold": fold_idx,
            "train_instance_ids": sorted([instance_ids[i] for i in train_sub_idx]),
            "val_instance_ids": sorted([instance_ids[i] for i in val_sub_idx]),
            "test_instance_ids": sorted([instance_ids[i] for i in test_idx]),
            "train_normal_instance_ids": sorted([instance_ids[i] for i in train_normal_idx]),
            "val_normal_instance_ids": sorted([instance_ids[i] for i in val_normal_idx]),
            "counts": {
                "train_total": len(train_sub_idx),
                "train_normal": len(train_normal_idx),
                "train_anomaly": len(train_sub_idx) - len(train_normal_idx),
                "val_total": len(val_sub_idx),
                "val_normal": len(val_normal_idx),
                "val_anomaly": len(val_sub_idx) - len(val_normal_idx),
                "test_total": len(test_idx),
                "test_normal": sum(1 for i in test_idx if labels[i] == 0),
                "test_anomaly": sum(1 for i in test_idx if labels[i] == 1),
            },
        }
        outer_folds.append(fold_dict)

        for i in test_idx:
            outer_test_assignment[instance_ids[i]] = fold_idx

    # 2. Repository-held-out splits
    unique_repos = sorted(list(set(repos)))
    repo_folds = []
    for repo_name in unique_repos:
        repo_test_idx = [i for i, r in enumerate(repos) if r == repo_name]
        repo_train_val_idx = [i for i, r in enumerate(repos) if r != repo_name]

        test_labels = labels[repo_test_idx]
        train_val_labels = labels[repo_train_val_idx]

        has_both_test_classes = (0 in test_labels) and (1 in test_labels)
        normal_fit_count = sum(1 for i in repo_train_val_idx if labels[i] == 0)
        is_evaluable = has_both_test_classes and normal_fit_count >= 10

        # Split remaining into train/val
        try:
            r_train_idx, r_val_idx = train_test_split(
                repo_train_val_idx,
                test_size=0.25,
                random_state=42,
                stratify=train_val_labels,
            )
        except Exception:
            # If stratification fails due to class counts
            r_train_idx, r_val_idx = train_test_split(
                repo_train_val_idx,
                test_size=0.25,
                random_state=42,
            )

        repo_folds.append({
            "repo": repo_name,
            "test_instance_ids": sorted([instance_ids[i] for i in repo_test_idx]),
            "train_instance_ids": sorted([instance_ids[i] for i in r_train_idx]),
            "val_instance_ids": sorted([instance_ids[i] for i in r_val_idx]),
            "is_evaluable": is_evaluable,
            "test_count": len(repo_test_idx),
            "test_normal_count": sum(1 for l in test_labels if l == 0),
            "test_anomaly_count": sum(1 for l in test_labels if l == 1),
            "train_normal_count": sum(1 for i in r_train_idx if labels[i] == 0),
        })

    splits_data = {
        "split_seed": 42,
        "n_outer_folds": 5,
        "outer_folds": outer_folds,
        "repository_held_out_folds": repo_folds,
    }

    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    # 3. Text manifest
    manifest_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        iid = row["instance_id"]
        sha256 = prov_map.get(iid, "")
        traj_path = RAW_TRAJ_DIR / f"{iid}.traj"

        with open(traj_path, "r", encoding="utf-8", errors="replace") as f:
            tdata = json.load(f)

        traj = tdata.get("trajectory", [])
        info = tdata.get("info", {})
        submission = info.get("submission", "") or ""

        log_chars = sum(len(str(s.get("action", ""))) + len(str(s.get("observation", ""))) for s in traj)
        patch_chars = len(submission.strip())
        patch_is_empty = patch_chars == 0
        log_is_missing = len(traj) == 0

        manifest_records.append({
            "instance_id": iid,
            "repo": row["repo"],
            "sha256": sha256,
            "is_resolved": int(row["is_resolved"]),
            "label": 0 if int(row["is_resolved"]) == 1 else 1,
            "outer_fold": outer_test_assignment[iid],
            "num_steps": len(traj),
            "log_char_length": log_chars,
            "patch_char_length": patch_chars,
            "patch_is_empty": patch_is_empty,
            "log_is_missing": log_is_missing,
        })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(outer_folds)} CV folds to {SPLITS_PATH}")
    print(f"Saved {len(manifest_records)} trajectory manifests to {MANIFEST_PATH}")


if __name__ == "__main__":
    build_manifests()
