"""Detailed error analysis, case studies, and stratified paired bootstrap for HW2 v2.

Requirements from Section 8 & Section 6:
1. Rule-based case selection on OOF predictions of primary detector (T2: knn_R3_T2):
   - Highest-scoring False Positive (true resolved/success flagged as anomaly)
   - Lowest-scoring False Negative (true unresolved/failure missed as normal)
   - Borderline error (closest absolute margin to threshold tau)
2. Case inspection:
   - instance_id, official label, prediction, score, tau, nearest normal neighbor ID
   - log distance, patch distance, raw action/observation excerpts, git diff hunk excerpts
   - SHA-256 hash tracking
   - Leave-One-Out (LOO) sensitivity analysis on the most influential step/hunk
3. Stratified paired bootstrap:
   - 2,000 resamples, seed 2026, instance-stratified
   - 95% confidence intervals for Balanced Accuracy, FPR, Recall, Unflagged Success Fraction, ROC-AUC, AP
   - Pairwise difference CI vs. T2 primary model
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from Model.models_v2 import KnnNormalDistanceDetector
from tools.build_text_embeddings import compute_channel_distance, compute_dual_distance


CONFIG_PATH = Path("configs/experiment_v2.json")
SPLITS_PATH = Path("Dataset/v2/splits.json")
MANIFEST_PATH = Path("Dataset/v2/text_manifest.jsonl")
EMBEDDINGS_DIR = Path("Dataset/v2/embeddings")
RAW_TRAJ_DIR = Path("Dataset/raw/sweagent_gpt4")
OUTPUTS_DIR = Path("Model/outputs/v2")
TABLES_DIR = Path("Model/tables/v2")


def run_error_case_analysis() -> Dict[str, Any]:
    print("Running error case studies and sensitivity analysis...")

    # Load splits & manifest
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_json = json.load(f)
    outer_folds = splits_json["outer_folds"]

    manifest_df = pd.read_json(MANIFEST_PATH, lines=True).sort_values("instance_id").reset_index(drop=True)
    all_instance_ids = manifest_df["instance_id"].tolist()
    all_labels = manifest_df["label"].to_numpy()
    sha_map = dict(zip(manifest_df["instance_id"], manifest_df["sha256"]))
    id_to_idx = {iid: idx for idx, iid in enumerate(all_instance_ids)}

    # Load cached embeddings
    r2_data = np.load(EMBEDDINGS_DIR / "r2_minilm.npz")
    r3_data = np.load(EMBEDDINGS_DIR / "r3_codebert.npz")
    with open(EMBEDDINGS_DIR / "step_embeddings.json", "r", encoding="utf-8") as f:
        step_embeddings_dict = json.load(f)

    r2_embed_map = {iid: r2_data["embeddings"][i] for i, iid in enumerate(r2_data["instance_ids"])}
    r3_embed_map = {iid: r3_data["embeddings"][i] for i, iid in enumerate(r3_data["instance_ids"])}
    r3_empty_map = {iid: r3_data["is_empty"][i] for i, iid in enumerate(r3_data["instance_ids"])}

    # Load OOF predictions
    oof_df = pd.read_csv(OUTPUTS_DIR / "oof_predictions.csv")
    t2_scores = oof_df["knn_R3_T2_seed_42_score"].to_numpy()
    t2_preds = oof_df["knn_R3_T2_seed_42_pred"].to_numpy()

    # Fold parameters for tau
    fold_params_df = pd.read_csv(TABLES_DIR / "fold_parameters.csv")
    fold_tau_map = {}
    for _, row in fold_params_df[fold_params_df["model"] == "knn_R3_T2"].iterrows():
        fold_tau_map[int(row["fold"])] = float(row["tau"])

    # Map each instance to its test fold tau
    inst_fold_map = dict(zip(manifest_df["instance_id"], manifest_df["outer_fold"]))
    inst_tau = np.array([fold_tau_map[inst_fold_map[iid]] for iid in all_instance_ids])

    # 1. Identify cases
    # FP: true label == 0 and pred == 1
    fp_indices = [i for i in range(284) if all_labels[i] == 0 and t2_preds[i] == 1]
    # FN: true label == 1 and pred == 0
    fn_indices = [i for i in range(284) if all_labels[i] == 1 and t2_preds[i] == 0]
    # All errors
    err_indices = fp_indices + fn_indices

    # Highest score FP
    highest_fp_idx = max(fp_indices, key=lambda i: t2_scores[i]) if fp_indices else None
    # Lowest score FN
    lowest_fn_idx = min(fn_indices, key=lambda i: t2_scores[i]) if fn_indices else None
    # Borderline error: smallest |score - tau|
    borderline_idx = min(err_indices, key=lambda i: abs(t2_scores[i] - inst_tau[i])) if err_indices else None

    case_indices = {
        "highest_fp": (highest_fp_idx, "False Positive (Resolved run flagged as failure)"),
        "lowest_fn": (lowest_fn_idx, "False Negative (Unresolved run missed as success)"),
        "borderline_error": (borderline_idx, "Borderline Error (Closest to decision threshold)"),
    }

    case_studies = {}

    for c_key, (idx, c_desc) in case_indices.items():
        if idx is None:
            case_studies[c_key] = {"description": c_desc, "found": False}
            continue

        iid = all_instance_ids[idx]
        fold_id = inst_fold_map[iid]
        fold_info = outer_folds[fold_id]
        train_norm_ids = fold_info["train_normal_instance_ids"]

        q_log = r2_embed_map[iid]
        q_patch = r3_embed_map[iid]
        q_empty = r3_empty_map[iid]

        # Find nearest normal neighbor in fold training reference
        best_ref_id = None
        min_dual_dist = 1e9
        best_log_dist = 1.0
        best_patch_dist = 1.0

        for r_id in train_norm_ids:
            r_log = r2_embed_map[r_id]
            r_patch = r3_embed_map[r_id]
            r_empty = r3_empty_map[r_id]

            d_l = compute_channel_distance(q_log, r_log)
            d_p = compute_channel_distance(q_patch, r_patch, q_empty, r_empty)
            d_dual = 0.5 * (d_l + d_p)

            if d_dual < min_dual_dist:
                min_dual_dist = d_dual
                best_ref_id = r_id
                best_log_dist = d_l
                best_patch_dist = d_p

        # Load raw excerpts
        traj_file = RAW_TRAJ_DIR / f"{iid}.traj"
        with open(traj_file, "r", encoding="utf-8", errors="replace") as f:
            tdata = json.load(f)

        traj = tdata.get("trajectory", [])
        submission = tdata.get("info", {}).get("submission", "") or ""

        # Step distance to nearest normal reference
        step_vecs = [np.array(v, dtype=np.float32) for v in step_embeddings_dict.get(iid, [])]
        ref_log_vec = r2_embed_map[best_ref_id]

        step_distances = []
        for s_i, s_vec in enumerate(step_vecs):
            sd = compute_channel_distance(s_vec, ref_log_vec)
            step_distances.append(sd)

        # Leave-one-out sensitivity: remove the most distant step and recompute
        if step_vecs and len(step_vecs) > 1:
            worst_step_idx = int(np.argmax(step_distances))
            remaining_steps = [s for s_i, s in enumerate(step_vecs) if s_i != worst_step_idx]
            loo_log = np.mean(remaining_steps, axis=0)
            loo_log = loo_log / np.linalg.norm(loo_log)

            loo_d_log = compute_channel_distance(loo_log, ref_log_vec)
            loo_dual = 0.5 * (loo_d_log + best_patch_dist)
            loo_delta = loo_dual - min_dual_dist
        else:
            worst_step_idx = 0
            loo_delta = 0.0

        sample_step = traj[worst_step_idx] if worst_step_idx < len(traj) else {}
        action_excerpt = str(sample_step.get("action", ""))[:200]
        obs_excerpt = str(sample_step.get("observation", ""))[:200]
        patch_excerpt = submission[:200]

        case_studies[c_key] = {
            "description": c_desc,
            "found": True,
            "instance_id": iid,
            "sha256": sha_map.get(iid, ""),
            "official_label": int(all_labels[idx]),
            "prediction": int(t2_preds[idx]),
            "score": float(t2_scores[idx]),
            "tau": float(inst_tau[idx]),
            "margin_to_tau": float(t2_scores[idx] - inst_tau[idx]),
            "nearest_normal_reference_id": best_ref_id,
            "nearest_normal_sha256": sha_map.get(best_ref_id, ""),
            "dual_distance_to_nearest": float(min_dual_dist),
            "log_distance_to_nearest": float(best_log_dist),
            "patch_distance_to_nearest": float(best_patch_dist),
            "num_steps": len(traj),
            "patch_is_empty": bool(q_empty),
            "most_distant_step_index": worst_step_idx,
            "step_action_excerpt": action_excerpt,
            "step_obs_excerpt": obs_excerpt,
            "patch_excerpt": patch_excerpt,
            "leave_one_out_score_delta": float(loo_delta),
        }

        print(f"\nCase [{c_key}]: {iid} (Label={all_labels[idx]}, Pred={t2_preds[idx]})")
        print(f"  Score: {t2_scores[idx]:.4f} vs Tau: {inst_tau[idx]:.4f} (Margin: {t2_scores[idx] - inst_tau[idx]:.4f})")
        print(f"  Nearest Normal: {best_ref_id} (d_log={best_log_dist:.4f}, d_patch={best_patch_dist:.4f})")
        print(f"  Leave-One-Out Step {worst_step_idx} Delta: {loo_delta:+.4f}")

    with open(TABLES_DIR / "case_studies.json", "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2)

    return case_studies


def run_stratified_paired_bootstrap(
    n_bootstraps: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Compute 2000-iteration stratified paired bootstrap confidence intervals on pooled OOF predictions."""
    print("\nRunning stratified paired bootstrap (N=2000 resamples)...")
    np.random.seed(seed)

    oof_df = pd.read_csv(OUTPUTS_DIR / "oof_predictions.csv")
    manifest_df = pd.read_json(MANIFEST_PATH, lines=True).sort_values("instance_id").reset_index(drop=True)
    all_labels = manifest_df["label"].to_numpy()

    # Models to evaluate in bootstrap
    models = [
        "knn_R3_T2_seed_42",
        "supervised_R3_logistic_T3_seed_42",
        "supervised_R1_logistic_seed_42",
        "inductive_ecod_R0_clean_seed_42",
        "isolation_forest_R0_clean_seed_42",
        "legacy_ae_R0_seed_42",
        "legacy_xgboost_R0_seed_42",
        "heuristic_T1_seed_42",
    ]

    normal_indices = np.where(all_labels == 0)[0]
    anomaly_indices = np.where(all_labels == 1)[0]
    n_norm = len(normal_indices)
    n_anom = len(anomaly_indices)

    # Collect predictions and scores
    model_preds = {m: oof_df[f"{m}_pred"].to_numpy() for m in models}
    model_scores = {m: oof_df[f"{m}_score"].to_numpy() for m in models}

    bootstrap_metrics: Dict[str, Dict[str, List[float]]] = {
        m: {
            "balanced_accuracy": [],
            "fpr": [],
            "failure_recall": [],
            "unflagged_success_fraction": [],
            "roc_auc": [],
            "average_precision": [],
            "diff_bal_acc_vs_t2": [],
            "diff_fpr_vs_t2": [],
        }
        for m in models
    }

    t2_name = "knn_R3_T2_seed_42"

    for b in range(n_bootstraps):
        # Stratified sampling with replacement
        boot_norm = np.random.choice(normal_indices, size=n_norm, replace=True)
        boot_anom = np.random.choice(anomaly_indices, size=n_anom, replace=True)
        boot_idx = np.concatenate([boot_norm, boot_anom])

        y_b = all_labels[boot_idx]

        # Compute for T2 first
        t2_p = model_preds[t2_name][boot_idx]
        t2_s = model_scores[t2_name][boot_idx]
        t2_m = compute_metrics(y_b, t2_s, t2_p)

        for m in models:
            m_p = model_preds[m][boot_idx]
            m_s = model_scores[m][boot_idx]
            m_m = compute_metrics(y_b, m_s, m_p)

            bootstrap_metrics[m]["balanced_accuracy"].append(m_m["balanced_accuracy"])
            bootstrap_metrics[m]["fpr"].append(m_m["fpr"])
            bootstrap_metrics[m]["failure_recall"].append(m_m["failure_recall"])
            bootstrap_metrics[m]["unflagged_success_fraction"].append(m_m["unflagged_success_fraction"])
            bootstrap_metrics[m]["roc_auc"].append(m_m["roc_auc"])
            bootstrap_metrics[m]["average_precision"].append(m_m["average_precision"])

            # Differences vs T2
            bootstrap_metrics[m]["diff_bal_acc_vs_t2"].append(m_m["balanced_accuracy"] - t2_m["balanced_accuracy"])
            bootstrap_metrics[m]["diff_fpr_vs_t2"].append(m_m["fpr"] - t2_m["fpr"])

    summary_rows = []
    for m in models:
        row = {"model": m}
        for metric, vals in bootstrap_metrics[m].items():
            arr = np.array(vals)
            # Remove NaNs if any
            clean_arr = arr[~np.isnan(arr)]
            if len(clean_arr) == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_ci_lower"] = float("nan")
                row[f"{metric}_ci_upper"] = float("nan")
            else:
                row[f"{metric}_mean"] = float(np.mean(clean_arr))
                row[f"{metric}_ci_lower"] = float(np.percentile(clean_arr, 2.5))
                row[f"{metric}_ci_upper"] = float(np.percentile(clean_arr, 97.5))
        summary_rows.append(row)

    ci_df = pd.DataFrame(summary_rows)
    ci_df.to_csv(TABLES_DIR / "bootstrap_ci.csv", index=False)
    print(f"Bootstrap CIs saved to {TABLES_DIR / 'bootstrap_ci.csv'}")
    return ci_df


if __name__ == "__main__":
    run_error_case_analysis()
    run_stratified_paired_bootstrap()
