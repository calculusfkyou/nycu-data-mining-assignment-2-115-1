"""Controlled ablation experiments for HW2 v2.

Ablations:
1. Representation Ablation: R0 (Euclidean on standardized normal fit) vs. R1 (cosine on TF-IDF) vs. R2 (cosine on MiniLM) vs. R3 (dual distance) under kNN normal distance.
2. Channel Ablation: Log-only (MiniLM) vs. Patch-only (CodeBERT) vs. Equal-weight Dual (R3) under kNN.
3. Normal Reference Contamination Ablation:
   - Contamination fraction p in {0.0, 0.05, 0.10} across seeds {42, 123, 456}.
   - Evaluated under both clean threshold tau and recalibrated tau.
4. Auxiliary Diagnostics:
   - Last-step-only observation vs. full trajectory.
   - Keyword ablation: masking error/traceback/failed keywords.
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
from sklearn.preprocessing import StandardScaler

from Model.experiment_v2 import compute_metrics, fit_r1_tfidf, load_raw_text_corpus, transform_r1_tfidf
from Model.models_v2 import KnnNormalDistanceDetector, select_validation_threshold
from Model.tabular_features_v2 import (
    NormalOnlyRobustScaler,
    get_r0_clean_unscaled_features,
    get_r0_legacy_features,
    load_raw_features_df,
)


CONFIG_PATH = Path("configs/experiment_v2.json")
SPLITS_PATH = Path("Dataset/v2/splits.json")
MANIFEST_PATH = Path("Dataset/v2/text_manifest.jsonl")
EMBEDDINGS_DIR = Path("Dataset/v2/embeddings")
RAW_TRAJ_DIR = Path("Dataset/raw/sweagent_gpt4")
TABLES_DIR = Path("Model/tables/v2")


def run_ablations() -> None:
    print("=" * 70)
    print("STARTING HW2 V2 CONTROLLED ABLATIONS")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_json = json.load(f)
    outer_folds = splits_json["outer_folds"]

    manifest_df = pd.read_json(MANIFEST_PATH, lines=True).sort_values("instance_id").reset_index(drop=True)
    all_instance_ids = manifest_df["instance_id"].tolist()
    all_labels = manifest_df["label"].to_numpy()
    id_to_idx = {iid: idx for idx, iid in enumerate(all_instance_ids)}

    raw_df = load_raw_features_df()
    X_r0_clean_unscaled, clean_col_names = get_r0_clean_unscaled_features(raw_df)

    # Load cached embeddings
    r2_data = np.load(EMBEDDINGS_DIR / "r2_minilm.npz")
    r3_data = np.load(EMBEDDINGS_DIR / "r3_codebert.npz")

    r2_embed_map = {iid: r2_data["embeddings"][i] for i, iid in enumerate(r2_data["instance_ids"])}
    r3_embed_map = {iid: r3_data["embeddings"][i] for i, iid in enumerate(r3_data["instance_ids"])}
    r3_empty_map = {iid: r3_data["is_empty"][i] for i, iid in enumerate(r3_data["instance_ids"])}

    all_log_embeds = np.array([r2_embed_map[iid] for iid in all_instance_ids], dtype=np.float32)
    all_patch_embeds = np.array([r3_embed_map[iid] for iid in all_instance_ids], dtype=np.float32)
    all_patch_empty = np.array([r3_empty_map[iid] for iid in all_instance_ids], dtype=bool)

    corpus_logs, corpus_patches = load_raw_text_corpus(all_instance_ids)

    # =========================================================================
    # ABLATION 1: REPRESENTATION COMPARISON (R0, R1, R2, R3)
    # =========================================================================
    print("\n--- Ablation 1: Representations (R0 vs R1 vs R2 vs R3) under kNN ---")
    rep_variants = ["R0_clean_euclidean", "R1_tfidf_cosine", "R2_minilm_cosine", "R3_dual_distance"]
    rep_oof_scores = {v: np.zeros(284, dtype=np.float32) for v in rep_variants}
    rep_oof_preds = {v: np.zeros(284, dtype=int) for v in rep_variants}

    for fold_info in outer_folds:
        f_idx = fold_info["fold"]
        train_idx = [id_to_idx[i] for i in fold_info["train_instance_ids"]]
        val_idx = [id_to_idx[i] for i in fold_info["val_instance_ids"]]
        test_idx = [id_to_idx[i] for i in fold_info["test_instance_ids"]]
        train_normal_idx = [id_to_idx[i] for i in fold_info["train_normal_instance_ids"]]

        y_val = all_labels[val_idx]

        # 1. R0-clean Euclidean
        scaler = NormalOnlyRobustScaler(clean_col_names)
        scaler.fit(X_r0_clean_unscaled[train_normal_idx])
        X_c_norm_train = scaler.transform(X_r0_clean_unscaled[train_normal_idx])
        X_c_val = scaler.transform(X_r0_clean_unscaled[val_idx])
        X_c_test = scaler.transform(X_r0_clean_unscaled[test_idx])

        best_k_r0 = 1
        best_acc_r0 = -1.0
        best_tau_r0 = -np.inf
        for k_cand in [1, 3, 5]:
            knn = KnnNormalDistanceDetector(k=k_cand, metric="euclidean")
            knn.fit_tabular(X_c_norm_train)
            v_scores = knn.score_euclidean(X_c_val)
            t_cand, m_cand = select_validation_threshold(y_val, v_scores)
            if m_cand["balanced_accuracy"] > best_acc_r0:
                best_acc_r0 = m_cand["balanced_accuracy"]
                best_k_r0 = k_cand
                best_tau_r0 = t_cand

        knn_r0 = KnnNormalDistanceDetector(k=best_k_r0, metric="euclidean")
        knn_r0.fit_tabular(X_c_norm_train)
        test_scores_r0 = knn_r0.score_euclidean(X_c_test)
        rep_oof_scores["R0_clean_euclidean"][test_idx] = test_scores_r0
        rep_oof_preds["R0_clean_euclidean"][test_idx] = (test_scores_r0 >= best_tau_r0).astype(int)

        # 2. R1 TF-IDF Cosine
        train_log_corpus = [corpus_logs[i] for i in train_idx]
        train_patch_corpus = [corpus_patches[i] for i in train_idx]
        val_log_corpus = [corpus_logs[i] for i in val_idx]
        val_patch_corpus = [corpus_patches[i] for i in val_idx]
        test_log_corpus = [corpus_logs[i] for i in test_idx]
        test_patch_corpus = [corpus_patches[i] for i in test_idx]

        vec_log, vec_patch = fit_r1_tfidf(train_log_corpus, train_patch_corpus)
        X_log_tr, X_patch_tr, _ = transform_r1_tfidf(vec_log, vec_patch, train_log_corpus, train_patch_corpus)
        X_log_val, X_patch_val, _ = transform_r1_tfidf(vec_log, vec_patch, val_log_corpus, val_patch_corpus)
        X_log_test, X_patch_test, _ = transform_r1_tfidf(vec_log, vec_patch, test_log_corpus, test_patch_corpus)

        # Filter normal training
        norm_in_train = [train_idx.index(i) for i in train_normal_idx]
        X_log_norm_tr = X_log_tr[norm_in_train]
        X_patch_norm_tr = X_patch_tr[norm_in_train]
        norm_empty_tr = np.array([len(corpus_patches[i].strip()) == 0 for i in train_normal_idx], dtype=bool)

        val_empty = np.array([len(corpus_patches[i].strip()) == 0 for i in val_idx], dtype=bool)
        test_empty = np.array([len(corpus_patches[i].strip()) == 0 for i in test_idx], dtype=bool)

        best_k_r1 = 1
        best_acc_r1 = -1.0
        best_tau_r1 = -np.inf
        for k_cand in [1, 3, 5]:
            knn = KnnNormalDistanceDetector(k=k_cand, metric="dual")
            knn.fit_embeddings(X_log_norm_tr, X_patch_norm_tr, norm_empty_tr)
            v_scores = knn.score_dual(X_log_val, X_patch_val, val_empty)
            t_cand, m_cand = select_validation_threshold(y_val, v_scores)
            if m_cand["balanced_accuracy"] > best_acc_r1:
                best_acc_r1 = m_cand["balanced_accuracy"]
                best_k_r1 = k_cand
                best_tau_r1 = t_cand

        knn_r1 = KnnNormalDistanceDetector(k=best_k_r1, metric="dual")
        knn_r1.fit_embeddings(X_log_norm_tr, X_patch_norm_tr, norm_empty_tr)
        test_scores_r1 = knn_r1.score_dual(X_log_test, X_patch_test, test_empty)
        rep_oof_scores["R1_tfidf_cosine"][test_idx] = test_scores_r1
        rep_oof_preds["R1_tfidf_cosine"][test_idx] = (test_scores_r1 >= best_tau_r1).astype(int)

        # 3. R2 MiniLM Cosine (Log-only)
        r2_norm_tr = all_log_embeds[train_normal_idx]
        r2_val = all_log_embeds[val_idx]
        r2_test = all_log_embeds[test_idx]

        best_k_r2 = 1
        best_acc_r2 = -1.0
        best_tau_r2 = -np.inf
        for k_cand in [1, 3, 5]:
            knn = KnnNormalDistanceDetector(k=k_cand, metric="cosine")
            knn.fit_embeddings(r2_norm_tr)
            v_scores = knn.score_single_cosine(r2_val)
            t_cand, m_cand = select_validation_threshold(y_val, v_scores)
            if m_cand["balanced_accuracy"] > best_acc_r2:
                best_acc_r2 = m_cand["balanced_accuracy"]
                best_k_r2 = k_cand
                best_tau_r2 = t_cand

        knn_r2 = KnnNormalDistanceDetector(k=best_k_r2, metric="cosine")
        knn_r2.fit_embeddings(r2_norm_tr)
        test_scores_r2 = knn_r2.score_single_cosine(r2_test)
        rep_oof_scores["R2_minilm_cosine"][test_idx] = test_scores_r2
        rep_oof_preds["R2_minilm_cosine"][test_idx] = (test_scores_r2 >= best_tau_r2).astype(int)

        # 4. R3 Dual Distance
        r3_norm_log = all_log_embeds[train_normal_idx]
        r3_norm_patch = all_patch_embeds[train_normal_idx]
        r3_norm_empty = all_patch_empty[train_normal_idx]

        r3_val_log = all_log_embeds[val_idx]
        r3_val_patch = all_patch_embeds[val_idx]
        r3_val_empty = all_patch_empty[val_idx]

        r3_test_log = all_log_embeds[test_idx]
        r3_test_patch = all_patch_embeds[test_idx]
        r3_test_empty = all_patch_empty[test_idx]

        best_k_r3 = 1
        best_acc_r3 = -1.0
        best_tau_r3 = -np.inf
        for k_cand in [1, 3, 5]:
            knn = KnnNormalDistanceDetector(k=k_cand, metric="dual")
            knn.fit_embeddings(r3_norm_log, r3_norm_patch, r3_norm_empty)
            v_scores = knn.score_dual(r3_val_log, r3_val_patch, r3_val_empty)
            t_cand, m_cand = select_validation_threshold(y_val, v_scores)
            if m_cand["balanced_accuracy"] > best_acc_r3:
                best_acc_r3 = m_cand["balanced_accuracy"]
                best_k_r3 = k_cand
                best_tau_r3 = t_cand

        knn_r3 = KnnNormalDistanceDetector(k=best_k_r3, metric="dual")
        knn_r3.fit_embeddings(r3_norm_log, r3_norm_patch, r3_norm_empty)
        test_scores_r3 = knn_r3.score_dual(r3_test_log, r3_test_patch, r3_test_empty)
        rep_oof_scores["R3_dual_distance"][test_idx] = test_scores_r3
        rep_oof_preds["R3_dual_distance"][test_idx] = (test_scores_r3 >= best_tau_r3).astype(int)

    rep_summary = []
    for v in rep_variants:
        m = compute_metrics(all_labels, rep_oof_scores[v], rep_oof_preds[v])
        rep_summary.append({"representation": v, **m})
        print(f"Rep: {v:<25} | BalAcc: {m['balanced_accuracy']:.3f} | FPR: {m['fpr']:.3f} | Recall: {m['failure_recall']:.3f}")

    pd.DataFrame(rep_summary).to_csv(TABLES_DIR / "ablation_representations.csv", index=False)

    # =========================================================================
    # ABLATION 2: CHANNEL CONTRIBUTION (LOG-ONLY vs PATCH-ONLY vs DUAL)
    # =========================================================================
    print("\n--- Ablation 2: Channel Contribution (Log vs Patch vs Dual) ---")
    chan_variants = ["log_only_minilm", "patch_only_codebert", "dual_channel_R3"]
    chan_oof_scores = {c: np.zeros(284, dtype=np.float32) for c in chan_variants}
    chan_oof_preds = {c: np.zeros(284, dtype=int) for c in chan_variants}
    chan_chosen_params: List[Dict[str, Any]] = []

    for fold_info in outer_folds:
        f_idx = fold_info["fold"]
        val_idx = [id_to_idx[i] for i in fold_info["val_instance_ids"]]
        test_idx = [id_to_idx[i] for i in fold_info["test_instance_ids"]]
        train_normal_idx = [id_to_idx[i] for i in fold_info["train_normal_instance_ids"]]

        y_val = all_labels[val_idx]

        # Log-only
        r2_norm = all_log_embeds[train_normal_idx]
        best_k_log = 1
        best_acc_log = -1.0
        best_tau_log = -np.inf
        for k_cand in [1, 3, 5]:
            knn = KnnNormalDistanceDetector(k=k_cand, metric="cosine")
            knn.fit_embeddings(r2_norm)
            v_scores = knn.score_single_cosine(all_log_embeds[val_idx])
            t_cand, m_cand = select_validation_threshold(y_val, v_scores)
            if m_cand["balanced_accuracy"] > best_acc_log:
                best_acc_log = m_cand["balanced_accuracy"]
                best_k_log = k_cand
                best_tau_log = t_cand

        knn_log = KnnNormalDistanceDetector(k=best_k_log, metric="cosine")
        knn_log.fit_embeddings(r2_norm)
        t_sc = knn_log.score_single_cosine(all_log_embeds[test_idx])
        chan_oof_scores["log_only_minilm"][test_idx] = t_sc
        chan_oof_preds["log_only_minilm"][test_idx] = (t_sc >= best_tau_log).astype(int)

        # Patch-only
        cb_norm = all_patch_embeds[train_normal_idx]
        cb_norm_empty = all_patch_empty[train_normal_idx]
        best_k_patch = 1
        best_acc_patch = -1.0
        best_tau_patch = -np.inf
        for k_cand in [1, 3, 5]:
            knn = KnnNormalDistanceDetector(k=k_cand, metric="cosine")
            knn.fit_embeddings(cb_norm, normal_patch_empty=cb_norm_empty)
            v_scores = knn.score_single_cosine(all_patch_embeds[val_idx], all_patch_empty[val_idx])
            t_cand, m_cand = select_validation_threshold(y_val, v_scores)
            if m_cand["balanced_accuracy"] > best_acc_patch:
                best_acc_patch = m_cand["balanced_accuracy"]
                best_k_patch = k_cand
                best_tau_patch = t_cand

        knn_patch = KnnNormalDistanceDetector(k=best_k_patch, metric="cosine")
        knn_patch.fit_embeddings(cb_norm, normal_patch_empty=cb_norm_empty)
        t_sc = knn_patch.score_single_cosine(all_patch_embeds[test_idx], all_patch_empty[test_idx])
        chan_oof_scores["patch_only_codebert"][test_idx] = t_sc
        chan_oof_preds["patch_only_codebert"][test_idx] = (t_sc >= best_tau_patch).astype(int)

        # Dual
        chan_oof_scores["dual_channel_R3"][test_idx] = rep_oof_scores["R3_dual_distance"][test_idx]
        chan_oof_preds["dual_channel_R3"][test_idx] = rep_oof_preds["R3_dual_distance"][test_idx]

        chan_chosen_params.append({
            "fold": f_idx,
            "best_k_log": best_k_log,
            "tau_log": float(best_tau_log),
            "best_k_patch": best_k_patch,
            "tau_patch": float(best_tau_patch),
        })

    chan_summary = []
    for c in chan_variants:
        m = compute_metrics(all_labels, chan_oof_scores[c], chan_oof_preds[c])
        chan_summary.append({"channel": c, **m})
        print(f"Chan: {c:<22} | BalAcc: {m['balanced_accuracy']:.3f} | FPR: {m['fpr']:.3f} | Recall: {m['failure_recall']:.3f}")

    pd.DataFrame(chan_summary).to_csv(TABLES_DIR / "ablation_channels.csv", index=False)
    pd.DataFrame(chan_chosen_params).to_csv(TABLES_DIR / "ablation_channels_params.csv", index=False)

    # =========================================================================
    # ABLATION 3: NORMAL REFERENCE CONTAMINATION
    # =========================================================================
    print("\n--- Ablation 3: Normal Reference Contamination ---")
    from Model.contamination_v2 import main as run_controlled_contamination
    run_controlled_contamination()
    print("Ablations complete.")


if __name__ == "__main__":
    run_ablations()
