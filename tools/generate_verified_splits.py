"""Generate strict 5-Fold Stratified Group K-Fold splits for SWE-bench Verified candidates.

Hygiene guarantees:
1. Grouped by instance_id: all models evaluating the same issue are in the exact same fold.
2. Zero leakage: no instance_id in test/val is ever seen in train.
3. Stratified: anomaly prevalence (~3.7%) is preserved across folds.
4. Normal-only training subset: saves train_normal_indices (label == 0) for pure-normal Tier 2 fitting.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

FEATURES_PATH = Path("Dataset/verified/candidate_features.csv")
OUTPUT_PATH = Path("Dataset/verified/splits.json")


def main():
    print("=" * 70)
    print("GENERATING 5-FOLD STRATIFIED GROUP K-FOLD SPLITS")
    print("=" * 70)
    
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} candidate rows.")
    
    groups = df["instance_id"].values
    y = df["label"].values
    unique_groups = len(np.unique(groups))
    print(f"Unique instance_id groups: {unique_groups}")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    splits_data = {
        "n_splits": 5,
        "seed": 42,
        "total_samples": len(df),
        "total_groups": unique_groups,
        "folds": [],
    }
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(sgkf.split(df, y, groups)):
        test_df = df.iloc[test_idx]
        train_val_df = df.iloc[train_val_idx]
        
        # Split train_val into train (75%) and val (25%) using inner StratifiedGroupKFold
        inner_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42 + fold_idx)
        inner_splits = list(inner_sgkf.split(train_val_df, train_val_df["label"].values, train_val_df["instance_id"].values))
        train_sub_idx, val_sub_idx = inner_splits[0]
        
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Verify zero group leakage
        train_groups = set(train_df["instance_id"])
        val_groups = set(val_df["instance_id"])
        test_groups = set(test_df["instance_id"])
        
        assert len(train_groups.intersection(val_groups)) == 0, f"Fold {fold_idx}: Train-Val group leakage!"
        assert len(train_groups.intersection(test_groups)) == 0, f"Fold {fold_idx}: Train-Test group leakage!"
        assert len(val_groups.intersection(test_groups)) == 0, f"Fold {fold_idx}: Val-Test group leakage!"
        
        # Extract train normal indices
        train_normal_idx = [int(i) for i in train_idx if df.iloc[i]["label"] == 0]
        
        n_train = len(train_idx)
        n_val = len(val_idx)
        n_test = len(test_idx)
        
        train_anom = int(train_df["label"].sum())
        val_anom = int(val_df["label"].sum())
        test_anom = int(test_df["label"].sum())
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {n_train} runs (normal: {len(train_normal_idx)}, anom: {train_anom} = {train_anom/n_train*100:.2f}%) across {len(train_groups)} groups")
        print(f"  Val:   {n_val} runs (normal: {n_val-val_anom}, anom: {val_anom} = {val_anom/n_val*100:.2f}%) across {len(val_groups)} groups")
        print(f"  Test:  {n_test} runs (normal: {n_test-test_anom}, anom: {test_anom} = {test_anom/n_test*100:.2f}%) across {len(test_groups)} groups")
        
        splits_data["folds"].append({
            "fold": fold_idx,
            "train_indices": [int(i) for i in train_idx],
            "train_normal_indices": train_normal_idx,
            "val_indices": [int(i) for i in val_idx],
            "test_indices": [int(i) for i in test_idx],
            "train_groups": sorted(list(train_groups)),
            "val_groups": sorted(list(val_groups)),
            "test_groups": sorted(list(test_groups)),
            "stats": {
                "train_total": n_train,
                "train_normal": len(train_normal_idx),
                "train_anomaly": train_anom,
                "val_total": n_val,
                "val_normal": n_val - val_anom,
                "val_anomaly": val_anom,
                "test_total": n_test,
                "test_normal": n_test - test_anom,
                "test_anomaly": test_anom,
            }
        })
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
    print("\n" + "=" * 70)
    print(f"SPLITS SAVED TO {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
