"""Controlled Ablation Experiments for AI Agent Ghostwork Detector.

Ablation 1 (AD-Specific): Training Contamination Ratio (0%, 1%, 5%, 10% anomalies in training).
Ablation 2 (Core Research Question): Trajectory Process Dynamics vs. Static Final Patch Output Features.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score

from Model.models import AutoencoderDetector


TABLES_DIR = Path("Model/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 2026]


def run_ablation_contamination() -> pd.DataFrame:
    """Ablate training set contamination by injecting 0%, 1%, 5%, 10% anomalies."""
    train_df = pd.read_csv("Dataset/train.csv")
    val_df = pd.read_csv("Dataset/val.csv")
    test_df = pd.read_csv("Dataset/test.csv")

    meta_cols = ["instance_id", "repo", "label", "is_resolved", "exit_status"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    clean_normal = train_df[train_df["label"] == 0][feature_cols].values
    val_anomalies = val_df[val_df["label"] == 1][feature_cols].values

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    contamination_levels = [0.0, 0.01, 0.05, 0.10]
    records = []

    for contam in contamination_levels:
        seed_pr = []
        seed_roc = []
        seed_f2 = []

        for seed in SEEDS:
            np.random.seed(seed)
            if contam == 0.0:
                X_train = clean_normal.copy()
            else:
                n_inject = max(1, int(len(clean_normal) * contam))
                inject_idx = np.random.choice(len(val_anomalies), size=n_inject, replace=False)
                injected = val_anomalies[inject_idx]
                X_train = np.vstack([clean_normal, injected])

            detector = AutoencoderDetector(input_dim=len(feature_cols), seed=seed)
            detector.fit(X_train)
            detector.tune_threshold(X_val, y_val, beta=2.0)

            scores = detector.score_samples(X_test)
            preds = detector.predict(X_test)

            seed_pr.append(average_precision_score(y_test, scores))
            seed_roc.append(roc_auc_score(y_test, scores))
            seed_f2.append(fbeta_score(y_test, preds, beta=2.0, zero_division=0))

        records.append({
            "Contamination_Ratio": f"{int(contam*100)}%",
            "PR_AUC_mean": np.mean(seed_pr),
            "PR_AUC_std": np.std(seed_pr),
            "ROC_AUC_mean": np.mean(seed_roc),
            "ROC_AUC_std": np.std(seed_roc),
            "F2_mean": np.mean(seed_f2),
            "F2_std": np.std(seed_f2),
        })

    df_contam = pd.DataFrame(records)
    base_pr = df_contam.loc[0, "PR_AUC_mean"]
    base_f2 = df_contam.loc[0, "F2_mean"]
    df_contam["Delta_PR_AUC"] = df_contam["PR_AUC_mean"] - base_pr
    df_contam["Delta_F2"] = df_contam["F2_mean"] - base_f2

    df_contam.to_csv(TABLES_DIR / "ablation_contamination.csv", index=False)
    print("\n=== Ablation 1: Training Contamination Ratio ===")
    print(df_contam[["Contamination_Ratio", "PR_AUC_mean", "PR_AUC_std", "Delta_PR_AUC", "F2_mean", "Delta_F2"]].to_string())
    return df_contam


def run_ablation_features() -> pd.DataFrame:
    """Ablate Trajectory Process Dynamics vs Static Final Patch Output features."""
    train_df = pd.read_csv("Dataset/train.csv")
    val_df = pd.read_csv("Dataset/val.csv")
    test_df = pd.read_csv("Dataset/test.csv")

    meta_cols = ["instance_id", "repo", "label", "is_resolved", "exit_status"]
    all_feature_cols = [c for c in train_df.columns if c not in meta_cols]
    process_cols = all_feature_cols[:16]
    patch_cols = all_feature_cols[16:]

    feature_sets = {
        "Full Representation (Process + Patch, D=24)": all_feature_cols,
        "Trajectory Process Dynamics Only (D=16)": process_cols,
        "Static Final Patch Output Only (D=8)": patch_cols,
    }

    records = []
    for name, cols in feature_sets.items():
        seed_pr = []
        seed_roc = []
        seed_f1 = []
        seed_f2 = []

        X_train_norm = train_df[train_df["label"] == 0][cols].values
        X_val = val_df[cols].values
        y_val = val_df["label"].values
        X_test = test_df[cols].values
        y_test = test_df["label"].values

        for seed in SEEDS:
            detector = AutoencoderDetector(input_dim=len(cols), seed=seed)
            detector.fit(X_train_norm)
            detector.tune_threshold(X_val, y_val, beta=2.0)

            scores = detector.score_samples(X_test)
            preds = detector.predict(X_test)

            seed_pr.append(average_precision_score(y_test, scores))
            seed_roc.append(roc_auc_score(y_test, scores))
            seed_f1.append(average_precision_score(y_test, preds))
            seed_f2.append(fbeta_score(y_test, preds, beta=2.0, zero_division=0))

        records.append({
            "Feature_Set": name,
            "Feature_Count": len(cols),
            "PR_AUC_mean": np.mean(seed_pr),
            "PR_AUC_std": np.std(seed_pr),
            "ROC_AUC_mean": np.mean(seed_roc),
            "ROC_AUC_std": np.std(seed_roc),
            "F2_mean": np.mean(seed_f2),
            "F2_std": np.std(seed_f2),
        })

    df_feat = pd.DataFrame(records)
    base_pr = df_feat.loc[0, "PR_AUC_mean"]
    base_f2 = df_feat.loc[0, "F2_mean"]
    df_feat["Delta_PR_AUC"] = df_feat["PR_AUC_mean"] - base_pr
    df_feat["Delta_F2"] = df_feat["F2_mean"] - base_f2

    df_feat.to_csv(TABLES_DIR / "ablation_features.csv", index=False)
    print("\n=== Ablation 2: Trajectory Process vs. Final Patch Output ===")
    print(df_feat[["Feature_Set", "Feature_Count", "PR_AUC_mean", "PR_AUC_std", "Delta_PR_AUC", "F2_mean", "Delta_F2"]].to_string())
    return df_feat


def main() -> None:
    run_ablation_contamination()
    run_ablation_features()


if __name__ == "__main__":
    main()
