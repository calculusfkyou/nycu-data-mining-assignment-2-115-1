"""5-Seed Multi-Tier Benchmark Evaluation with Statistical Significance Testing.

Adheres strictly to DM-HW2 Section 6 and Section 7 requirements:
- Shared test split across all 3 tiers
- 5 random seeds reporting mean +/- std
- Paired t-tests and non-parametric bootstrap 95% CI
- Training time and inference latency (ms / sample)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from Model.models import (
    AutoencoderDetector,
    NaiveHeuristicDetector,
    SupervisedReferenceDetector,
)


TABLES_DIR = Path("Model/tables")
OUTPUTS_DIR = Path("Model/outputs")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]


def measure_scoring(model, data):
    """Median of 20 warmed batch score calls, divided by batch size (not request latency)."""
    model.score_samples(data)
    durations = []
    for _ in range(20):
        t0 = time.perf_counter()
        scores = model.score_samples(data)
        durations.append(time.perf_counter() - t0)
    return scores, float(np.median(durations) * 1000 / len(data))


def run_benchmark() -> pd.DataFrame:
    # Load frozen data splits
    train_df = pd.read_csv("Dataset/train.csv")
    val_df = pd.read_csv("Dataset/val.csv")
    test_df = pd.read_csv("Dataset/test.csv")

    meta_cols = ["instance_id", "repo", "label", "is_resolved", "exit_status"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train_normal = train_df[train_df["label"] == 0][feature_cols].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # All fitting uses the shared training partition; validation is calibration only.
    X_train_sup = train_df[feature_cols].values
    y_train_sup = train_df['label'].values

    results = []
    preds_dict = {"instance_id": test_df["instance_id"], "label": y_test}

    for seed in SEEDS:
        # ----------------------------------------------------------------------
        # Tier 1: Naive Baseline
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        t1_model = NaiveHeuristicDetector()
        t1_model.fit(X_train_normal)
        t1_train_time = time.perf_counter() - t0

        t1_scores, t1_latency = measure_scoring(t1_model, test_df)
        t1_preds = t1_model.predict(test_df)

        results.append({
            "seed": seed,
            "method": "Tier 1: Naive Heuristic",
            "PR_AUC": average_precision_score(y_test, t1_scores),
            "ROC_AUC": roc_auc_score(y_test, t1_scores),
            "F1": f1_score(y_test, t1_preds, zero_division=0),
            "F2": fbeta_score(y_test, t1_preds, beta=2.0, zero_division=0),
            "Precision": precision_score(y_test, t1_preds, zero_division=0),
            "Recall": recall_score(y_test, t1_preds, zero_division=0),
            "Latency_ms": t1_latency,
            "Train_Time_s": t1_train_time,
            "Threshold_tau": 1.0,
        })

        # ----------------------------------------------------------------------
        # Tier 2: Primary Method (Deep Autoencoder, Normal-Only)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        t2_model = AutoencoderDetector(input_dim=len(feature_cols), seed=seed)
        t2_model.fit(X_train_normal)
        t2_train_time = time.perf_counter() - t0

        t2_model.tune_threshold(X_val, y_val, beta=2.0)

        t2_scores, t2_latency = measure_scoring(t2_model, X_test)
        t2_preds = t2_model.predict(X_test)

        results.append({
            "seed": seed,
            "method": "Tier 2: Deep Autoencoder (Ours)",
            "PR_AUC": average_precision_score(y_test, t2_scores),
            "ROC_AUC": roc_auc_score(y_test, t2_scores),
            "F1": f1_score(y_test, t2_preds, zero_division=0),
            "F2": fbeta_score(y_test, t2_preds, beta=2.0, zero_division=0),
            "Precision": precision_score(y_test, t2_preds, zero_division=0),
            "Recall": recall_score(y_test, t2_preds, zero_division=0),
            "Latency_ms": t2_latency,
            "Train_Time_s": t2_train_time,
            "Threshold_tau": t2_model.threshold,
        })

        # ----------------------------------------------------------------------
        # Tier 3: Strong Reference (Supervised XGBoost)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        t3_model = SupervisedReferenceDetector(seed=seed)
        t3_model.fit(X_train_sup, y_train_sup)
        t3_train_time = time.perf_counter() - t0

        t3_model.tune_threshold(X_val, y_val, beta=2.0)

        t3_scores, t3_latency = measure_scoring(t3_model, X_test)
        t3_preds = t3_model.predict(X_test)

        results.append({
            "seed": seed,
            "method": "Tier 3: Supervised XGBoost",
            "PR_AUC": average_precision_score(y_test, t3_scores),
            "ROC_AUC": roc_auc_score(y_test, t3_scores),
            "F1": f1_score(y_test, t3_preds, zero_division=0),
            "F2": fbeta_score(y_test, t3_preds, beta=2.0, zero_division=0),
            "Precision": precision_score(y_test, t3_preds, zero_division=0),
            "Recall": recall_score(y_test, t3_preds, zero_division=0),
            "Latency_ms": t3_latency,
            "Train_Time_s": t3_train_time,
            "Threshold_tau": t3_model.threshold,
        })

        pd.DataFrame({"instance_id": test_df.instance_id, "label": y_test,
                      "T1_score": t1_scores, "T1_pred": t1_preds,
                      "T2_score": t2_scores, "T2_pred": t2_preds,
                      "T3_score": t3_scores, "T3_pred": t3_preds}).to_csv(
                          OUTPUTS_DIR / f'test_predictions_seed_{seed}.csv', index=False)
        pd.DataFrame({"instance_id": val_df.instance_id, "label": y_val,
                      "T2_score": t2_model.score_samples(X_val)}).to_csv(
                          OUTPUTS_DIR / f'val_predictions_seed_{seed}.csv', index=False)
        if seed == 42:
            preds_dict["T1_score"] = t1_scores
            preds_dict["T1_pred"] = t1_preds
            preds_dict["T2_score"] = t2_scores
            preds_dict["T2_pred"] = t2_preds
            preds_dict["T3_score"] = t3_scores
            preds_dict["T3_pred"] = t3_preds

    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLES_DIR / "all_seed_runs.csv", index=False)
    pd.DataFrame(preds_dict).to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    # Aggregate Main Results Table
    metrics = ["PR_AUC", "ROC_AUC", "F1", "F2", "Precision", "Recall", "Latency_ms", "Train_Time_s"]
    summary_rows = []
    for method in ["Tier 1: Naive Heuristic", "Tier 2: Deep Autoencoder (Ours)", "Tier 3: Supervised XGBoost"]:
        sub = df_res[df_res["method"] == method]
        row = {"Method": method}
        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std()
            row[m] = f"{sub[m].mean():.3f} ± {sub[m].std():.3f}"
        summary_rows.append(row)

    main_table = pd.DataFrame(summary_rows)
    main_table.to_csv(TABLES_DIR / "main_results_table.csv", index=False)
    print("\n=== Main Results Table (5 Seeds) ===")
    print(main_table[["Method", "PR_AUC", "ROC_AUC", "F1", "F2", "Latency_ms", "Train_Time_s"]].to_string())

    # Statistical Significance Testing (Tier 2 vs Tier 1 and Tier 2 vs Tier 3)
    t2_pr = df_res[df_res["method"] == "Tier 2: Deep Autoencoder (Ours)"]["PR_AUC"].values
    t1_pr = df_res[df_res["method"] == "Tier 1: Naive Heuristic"]["PR_AUC"].values
    t3_pr = df_res[df_res["method"] == "Tier 3: Supervised XGBoost"]["PR_AUC"].values

    t2_f2 = df_res[df_res["method"] == "Tier 2: Deep Autoencoder (Ours)"]["F2"].values
    t1_f2 = df_res[df_res["method"] == "Tier 1: Naive Heuristic"]["F2"].values
    t3_f2 = df_res[df_res["method"] == "Tier 3: Supervised XGBoost"]["F2"].values

    # Paired t-test
    ttest_t2_t1 = stats.ttest_rel(t2_pr, t1_pr)
    ttest_t2_t3 = stats.ttest_rel(t2_pr, t3_pr)

    # Bootstrap 95% CI of PR-AUC gap (1000 resamples on seed 42 test predictions)
    np.random.seed(42)
    boot_gaps_t1 = []
    boot_gaps_t3 = []
    N = len(y_test)
    for _ in range(1000):
        b_idx = np.random.choice(N, size=N, replace=True)
        if len(np.unique(y_test[b_idx])) < 2:
            continue
        pr_t2 = average_precision_score(y_test[b_idx], preds_dict["T2_score"][b_idx])
        pr_t1 = average_precision_score(y_test[b_idx], preds_dict["T1_score"][b_idx])
        pr_t3 = average_precision_score(y_test[b_idx], preds_dict["T3_score"][b_idx])
        boot_gaps_t1.append(pr_t2 - pr_t1)
        boot_gaps_t3.append(pr_t2 - pr_t3)

    ci_t1 = (np.percentile(boot_gaps_t1, 2.5), np.percentile(boot_gaps_t1, 97.5))
    ci_t3 = (np.percentile(boot_gaps_t3, 2.5), np.percentile(boot_gaps_t3, 97.5))

    sig_records = [
        {
            "Comparison": "Tier 2 (Ours) vs. Tier 1 (Naive)",
            "Metric": "PR-AUC",
            "T2_Mean": t2_pr.mean(),
            "Competitor_Mean": t1_pr.mean(),
            "Gap": t2_pr.mean() - t1_pr.mean(),
            "Paired_ttest_pval": float(ttest_t2_t1.pvalue),
            "Bootstrap_95CI_Lower": float(ci_t1[0]),
            "Bootstrap_95CI_Upper": float(ci_t1[1]),
        },
        {
            "Comparison": "Tier 2 (Ours) vs. Tier 3 (Supervised)",
            "Metric": "PR-AUC",
            "T2_Mean": t2_pr.mean(),
            "Competitor_Mean": t3_pr.mean(),
            "Gap": t2_pr.mean() - t3_pr.mean(),
            "Paired_ttest_pval": float(ttest_t2_t3.pvalue),
            "Bootstrap_95CI_Lower": float(ci_t3[0]),
            "Bootstrap_95CI_Upper": float(ci_t3[1]),
        },
    ]
    sig_df = pd.DataFrame(sig_records)
    sig_df.to_csv(TABLES_DIR / "significance_tests.csv", index=False)
    print("\n=== Statistical Significance Tests ===")
    print(sig_df[["Comparison", "Gap", "Paired_ttest_pval", "Bootstrap_95CI_Lower", "Bootstrap_95CI_Upper"]].to_string())

    return main_table


if __name__ == "__main__":
    run_benchmark()
