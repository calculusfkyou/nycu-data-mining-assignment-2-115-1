"""Main benchmark experiment runner for HW2 v2.

Implements strict 5-fold outer cross-validation with inner validation model selection.
Outputs:
- OOF predictions for all 284 instances per model
- Pooled confusion matrices, ROC-AUC, AP, FPR, Failure Recall, Balanced Accuracy
- Precision@K (10, 20, 50)
- Repository-held-out domain sensitivity analysis
- Timing and memory profiling
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from Model.models_v2 import (
    ConditionalResidualDetector,
    InductiveECODVariant,
    IsolationForestDetector,
    KnnNormalDistanceDetector,
    LegacyAEDetector,
    LegacyXGBoostDetector,
    RuleHeuristicDetector,
    ShallowDAEDetector,
    SupervisedLogisticDetector,
    select_validation_threshold,
)
from Model.tabular_features_v2 import (
    NormalOnlyRobustScaler,
    get_r0_clean_unscaled_features,
    get_r0_legacy_features,
    load_raw_features_df,
)
from tools.normalize_text import normalize_log_text, normalize_patch_text


CONFIG_PATH = Path("configs/experiment_v2.json")
SPLITS_PATH = Path("Dataset/v2/splits.json")
MANIFEST_PATH = Path("Dataset/v2/text_manifest.jsonl")
EMBEDDINGS_DIR = Path("Dataset/v2/embeddings")
RAW_TRAJ_DIR = Path("Dataset/raw/sweagent_gpt4")

OUTPUTS_DIR = Path("Model/outputs/v2")
TABLES_DIR = Path("Model/tables/v2")


def load_raw_text_corpus(instance_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Load normalized log and patch strings for given instances."""
    df_raw = load_raw_features_df().set_index("instance_id")
    log_texts = []
    patch_texts = []

    for iid in instance_ids:
        repo = str(df_raw.loc[iid, "repo"]) if iid in df_raw.index else ""
        traj_file = RAW_TRAJ_DIR / f"{iid}.traj"

        with open(traj_file, "r", encoding="utf-8", errors="replace") as f:
            tdata = json.load(f)

        traj = tdata.get("trajectory", [])
        submission = tdata.get("info", {}).get("submission")

        # Combine steps sequentially
        log_parts = []
        for s in traj:
            act = normalize_log_text(str(s.get("action", "")), iid, repo)
            obs = normalize_log_text(str(s.get("observation", "")), iid, repo)
            if act:
                log_parts.append(act)
            if obs:
                log_parts.append(obs)
        log_str = "\n".join(log_parts)
        log_texts.append(log_str)

        patch_clean, _ = normalize_patch_text(submission, iid, repo)
        patch_texts.append(patch_clean)

    return log_texts, patch_texts


def fit_r1_tfidf(
    train_log_texts: List[str],
    train_patch_texts: List[str],
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Fit character TF-IDF strictly on training partition."""
    # Analyzer char, ngrams 3-5, min_df=2, sublinear_tf=True, max_features=20000
    vec_log = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=20000,
        norm="l2",
    )
    vec_patch = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=20000,
        norm="l2",
    )

    vec_log.fit(train_log_texts)
    # Ensure at least one non-empty string to avoid empty vocabulary error
    valid_patch = [p for p in train_patch_texts if p.strip()]
    if valid_patch:
        vec_patch.fit(valid_patch)
    else:
        vec_patch.fit(["dummy diff @@ -1 +1 @@"])

    return vec_log, vec_patch


def transform_r1_tfidf(
    vec_log: TfidfVectorizer,
    vec_patch: TfidfVectorizer,
    log_texts: List[str],
    patch_texts: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform log and patch texts into L2-normalized TF-IDF and concatenated representation."""
    X_log = vec_log.transform(log_texts).toarray().astype(np.float32)
    X_patch = vec_patch.transform(patch_texts).toarray().astype(np.float32)

    # Supervised concatenation: (1 / sqrt(2)) * log || (1 / sqrt(2)) * patch
    scale = float(1.0 / np.sqrt(2.0))
    X_concat = np.hstack([scale * X_log, scale * X_patch]).astype(np.float32)

    return X_log, X_patch, X_concat


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics including confusion matrix, balanced accuracy, and ranking metrics."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    n = len(y_true)

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (recall + spec)

    unflagged_count = tn + fn
    unflagged_success_frac = float(tn / unflagged_count) if unflagged_count > 0 else float("nan")
    unflagged_coverage = float(unflagged_count / n)
    review_fraction = float((tp + fp) / n)

    # Ranking metrics (ROC-AUC and AP)
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.5

    try:
        ap = float(average_precision_score(y_true, y_scores))
    except Exception:
        ap = float(np.mean(y_true))

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "failure_recall": recall,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "unflagged_success_fraction": unflagged_success_frac,
        "unflagged_coverage": unflagged_coverage,
        "review_fraction": review_fraction,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "prevalence": float(np.mean(y_true)),
    }


def compute_precision_at_k(
    instance_ids: List[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    k_list: List[int] = [10, 20, 50],
) -> Dict[str, float]:
    """Compute Precision@K with deterministic tiebreak on instance_id."""
    order = sorted(
        range(len(scores)),
        key=lambda i: (-scores[i], instance_ids[i]),
    )
    y_sorted = y_true[order]
    results = {}
    for k in k_list:
        top_k = y_sorted[:k]
        prec = float(np.sum(top_k == 1) / k)
        results[f"precision@{k}"] = prec
    return results


def run_experiment(seeds: List[int] = [42, 123, 456]) -> None:
    print("=" * 70)
    print("STARTING HW2 V2 BENCHMARK EXPERIMENT")
    print("=" * 70)

    start_clock = time.perf_counter()
    process = psutil.Process()
    mem_start = process.memory_info().rss / (1024 * 1024)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_json = json.load(f)
    outer_folds = splits_json["outer_folds"]
    repo_folds = splits_json["repository_held_out_folds"]

    # Load manifest
    manifest_df = pd.read_json(MANIFEST_PATH, lines=True).sort_values("instance_id").reset_index(drop=True)
    all_instance_ids = manifest_df["instance_id"].tolist()
    all_labels = manifest_df["label"].to_numpy()
    id_to_idx = {iid: idx for idx, iid in enumerate(all_instance_ids)}

    # Load tabular features
    raw_df = load_raw_features_df()
    X_r0_legacy, _ = get_r0_legacy_features(raw_df)
    X_r0_clean_unscaled, clean_col_names = get_r0_clean_unscaled_features(raw_df)

    # Load cached embeddings
    print("Loading cached frozen embeddings...")
    r2_data = np.load(EMBEDDINGS_DIR / "r2_minilm.npz")
    r3_data = np.load(EMBEDDINGS_DIR / "r3_codebert.npz")

    r2_embed_map = {iid: r2_data["embeddings"][i] for i, iid in enumerate(r2_data["instance_ids"])}
    r3_embed_map = {iid: r3_data["embeddings"][i] for i, iid in enumerate(r3_data["instance_ids"])}
    r3_empty_map = {iid: r3_data["is_empty"][i] for i, iid in enumerate(r3_data["instance_ids"])}

    all_log_embeds = np.array([r2_embed_map[iid] for iid in all_instance_ids], dtype=np.float32)
    all_patch_embeds = np.array([r3_embed_map[iid] for iid in all_instance_ids], dtype=np.float32)
    all_patch_empty = np.array([r3_empty_map[iid] for iid in all_instance_ids], dtype=bool)

    # R3 concatenation: 1/sqrt(2)*log || 1/sqrt(2)*patch
    scale = float(1.0 / np.sqrt(2.0))
    all_r3_concat = np.hstack([scale * all_log_embeds, scale * all_patch_embeds]).astype(np.float32)

    # Define all evaluation models
    # Multi-seed models run across seeds [42, 123, 456]; deterministic models run on seed 42 only
    model_definitions = [
        {"name": "always_alert", "multi_seed": False, "category": "context"},
        {"name": "never_alert", "multi_seed": False, "category": "context"},
        {"name": "heuristic_T1", "multi_seed": False, "category": "heuristic"},
        {"name": "knn_R3_T2", "multi_seed": False, "category": "primary"},
        {"name": "supervised_R3_logistic_T3", "multi_seed": False, "category": "supervised_reference"},
        {"name": "supervised_R1_logistic", "multi_seed": False, "category": "nlp_control"},
        {"name": "isolation_forest_R0_clean", "multi_seed": True, "category": "tabular_control"},
        {"name": "inductive_ecod_R0_clean", "multi_seed": False, "category": "tabular_control"},
        {"name": "legacy_ae_R0", "multi_seed": True, "category": "legacy_control"},
        {"name": "legacy_xgboost_R0", "multi_seed": False, "category": "legacy_control"},
        {"name": "shallow_dae_R0_clean", "multi_seed": True, "category": "secondary"},
        {"name": "conditional_residual_R0_clean", "multi_seed": False, "category": "secondary"},
    ]

    # Pre-extract raw text corpus for R1 TF-IDF
    print("Loading normalized raw text corpus for R1 TF-IDF...")
    corpus_logs, corpus_patches = load_raw_text_corpus(all_instance_ids)

    # Storage for OOF predictions across folds: model_name -> seed -> array of 284 scores & preds
    oof_results: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    fold_details_records: List[Dict[str, Any]] = []

    for m_def in model_definitions:
        m_name = m_def["name"]
        m_seeds = seeds if m_def["multi_seed"] else [42]
        oof_results[m_name] = {}
        for s in m_seeds:
            oof_results[m_name][s] = {
                "scores": np.zeros(284, dtype=np.float32),
                "preds": np.zeros(284, dtype=int),
                "tau": np.zeros(5, dtype=np.float32),
            }

    # =========================================================================
    # 5-FOLD STRATIFIED OUTER CROSS-VALIDATION
    # =========================================================================
    timing_records = []

    for fold_info in outer_folds:
        f_idx = fold_info["fold"]
        print(f"\n--- Running Fold {f_idx + 1} / 5 ---")

        train_ids = fold_info["train_instance_ids"]
        val_ids = fold_info["val_instance_ids"]
        test_ids = fold_info["test_instance_ids"]
        train_normal_ids = fold_info["train_normal_instance_ids"]

        train_idx = [id_to_idx[i] for i in train_ids]
        val_idx = [id_to_idx[i] for i in val_ids]
        test_idx = [id_to_idx[i] for i in test_ids]
        train_normal_idx = [id_to_idx[i] for i in train_normal_ids]

        y_train = all_labels[train_idx]
        y_val = all_labels[val_idx]
        y_test = all_labels[test_idx]

        # 1. R0 Legacy Scaler
        scaler_r0_leg = StandardScaler()
        X_r0_leg_train = scaler_r0_leg.fit_transform(X_r0_legacy[train_idx])
        X_r0_leg_val = scaler_r0_leg.transform(X_r0_legacy[val_idx])
        X_r0_leg_test = scaler_r0_leg.transform(X_r0_legacy[test_idx])

        # Normal-only legacy for AE
        scaler_r0_leg_norm = StandardScaler()
        X_r0_leg_norm_train = scaler_r0_leg_norm.fit_transform(X_r0_legacy[train_normal_idx])
        X_r0_leg_norm_val = scaler_r0_leg_norm.transform(X_r0_legacy[val_idx])
        X_r0_leg_norm_test = scaler_r0_leg_norm.transform(X_r0_legacy[test_idx])

        # 2. R0-Clean Scaler (normal-only median/IQR)
        scaler_clean = NormalOnlyRobustScaler(clean_col_names)
        scaler_clean.fit(X_r0_clean_unscaled[train_normal_idx])
        X_clean_train_norm = scaler_clean.transform(X_r0_clean_unscaled[train_normal_idx])
        X_clean_val = scaler_clean.transform(X_r0_clean_unscaled[val_idx])
        X_clean_test = scaler_clean.transform(X_r0_clean_unscaled[test_idx])
        X_clean_train_all = scaler_clean.transform(X_r0_clean_unscaled[train_idx])

        # 3. R1 TF-IDF Vectorization
        t_vec0 = time.perf_counter()
        train_log_corpus = [corpus_logs[i] for i in train_idx]
        train_patch_corpus = [corpus_patches[i] for i in train_idx]
        val_log_corpus = [corpus_logs[i] for i in val_idx]
        val_patch_corpus = [corpus_patches[i] for i in val_idx]
        test_log_corpus = [corpus_logs[i] for i in test_idx]
        test_patch_corpus = [corpus_patches[i] for i in test_idx]

        vec_log, vec_patch = fit_r1_tfidf(train_log_corpus, train_patch_corpus)
        _, _, X_r1_concat_train = transform_r1_tfidf(vec_log, vec_patch, train_log_corpus, train_patch_corpus)
        _, _, X_r1_concat_val = transform_r1_tfidf(vec_log, vec_patch, val_log_corpus, val_patch_corpus)
        _, _, X_r1_concat_test = transform_r1_tfidf(vec_log, vec_patch, test_log_corpus, test_patch_corpus)
        t_vec = time.perf_counter() - t_vec0

        # Embeddings slices
        r3_norm_log = all_log_embeds[train_normal_idx]
        r3_norm_patch = all_patch_embeds[train_normal_idx]
        r3_norm_empty = all_patch_empty[train_normal_idx]

        r3_val_log = all_log_embeds[val_idx]
        r3_val_patch = all_patch_embeds[val_idx]
        r3_val_empty = all_patch_empty[val_idx]

        r3_test_log = all_log_embeds[test_idx]
        r3_test_patch = all_patch_embeds[test_idx]
        r3_test_empty = all_patch_empty[test_idx]

        # ---------------------------------------------------------------------
        # RUN EACH MODEL ON FOLD
        # ---------------------------------------------------------------------

        # A. Context: Always-alert & Never-alert
        for m_name, val_val in [("always_alert", 1.0), ("never_alert", 0.0)]:
            val_scores = np.full(len(val_idx), val_val, dtype=np.float32)
            test_scores = np.full(len(test_idx), val_val, dtype=np.float32)
            tau = 0.5
            pred_test = (test_scores >= tau).astype(int)
            oof_results[m_name][42]["scores"][test_idx] = test_scores
            oof_results[m_name][42]["preds"][test_idx] = pred_test
            oof_results[m_name][42]["tau"][f_idx] = tau

        # B. Heuristic T1
        heur = RuleHeuristicDetector()
        val_df = raw_df.iloc[val_idx]
        test_df = raw_df.iloc[test_idx]
        val_scores = heur.decision_function(val_df)
        test_scores = heur.decision_function(test_df)
        best_tau, _ = select_validation_threshold(y_val, val_scores)
        pred_test = (test_scores >= best_tau).astype(int)
        oof_results["heuristic_T1"][42]["scores"][test_idx] = test_scores
        oof_results["heuristic_T1"][42]["preds"][test_idx] = pred_test
        oof_results["heuristic_T1"][42]["tau"][f_idx] = best_tau

        # C. Primary kNN R3 (T2)
        # Select k in {1, 3, 5} on validation fold
        t0_knn = time.perf_counter()
        best_k = 1
        best_bal_acc = -1.0
        best_tau = -np.inf

        for k_cand in [1, 3, 5]:
            knn = KnnNormalDistanceDetector(k=k_cand, metric="dual")
            knn.fit_embeddings(r3_norm_log, r3_norm_patch, r3_norm_empty)
            v_scores = knn.score_dual(r3_val_log, r3_val_patch, r3_val_empty)
            tau_cand, metrics_cand = select_validation_threshold(y_val, v_scores)

            if (metrics_cand["balanced_accuracy"] > best_bal_acc) or (
                np.isclose(metrics_cand["balanced_accuracy"], best_bal_acc) and k_cand < best_k
            ):
                best_bal_acc = metrics_cand["balanced_accuracy"]
                best_k = k_cand
                best_tau = tau_cand

        knn_final = KnnNormalDistanceDetector(k=best_k, metric="dual")
        knn_final.fit_embeddings(r3_norm_log, r3_norm_patch, r3_norm_empty)
        test_scores = knn_final.score_dual(r3_test_log, r3_test_patch, r3_test_empty)
        pred_test = (test_scores >= best_tau).astype(int)
        t_knn = time.perf_counter() - t0_knn

        oof_results["knn_R3_T2"][42]["scores"][test_idx] = test_scores
        oof_results["knn_R3_T2"][42]["preds"][test_idx] = pred_test
        oof_results["knn_R3_T2"][42]["tau"][f_idx] = best_tau

        fold_details_records.append({
            "fold": f_idx,
            "model": "knn_R3_T2",
            "seed": 42,
            "chosen_param": f"k={best_k}",
            "tau": float(best_tau),
            "val_bal_acc": float(best_bal_acc),
        })

        # D. Supervised R3 Logistic (T3)
        # Grid C in {0.01, 0.1, 1.0, 10.0}
        best_c = 0.01
        best_bal_acc = -1.0
        best_tau = -np.inf
        X_r3_concat_train = all_r3_concat[train_idx]
        X_r3_concat_val = all_r3_concat[val_idx]
        X_r3_concat_test = all_r3_concat[test_idx]

        for c_cand in [0.01, 0.1, 1.0, 10.0]:
            clf = SupervisedLogisticDetector(C=c_cand)
            clf.fit(X_r3_concat_train, y_train)
            v_scores = clf.decision_function(X_r3_concat_val)
            tau_cand, metrics_cand = select_validation_threshold(y_val, v_scores)

            if (metrics_cand["balanced_accuracy"] > best_bal_acc) or (
                np.isclose(metrics_cand["balanced_accuracy"], best_bal_acc) and c_cand < best_c
            ):
                best_bal_acc = metrics_cand["balanced_accuracy"]
                best_c = c_cand
                best_tau = tau_cand

        clf_final = SupervisedLogisticDetector(C=best_c)
        clf_final.fit(X_r3_concat_train, y_train)
        test_scores = clf_final.decision_function(X_r3_concat_test)
        pred_test = (test_scores >= best_tau).astype(int)

        oof_results["supervised_R3_logistic_T3"][42]["scores"][test_idx] = test_scores
        oof_results["supervised_R3_logistic_T3"][42]["preds"][test_idx] = pred_test
        oof_results["supervised_R3_logistic_T3"][42]["tau"][f_idx] = best_tau

        fold_details_records.append({
            "fold": f_idx,
            "model": "supervised_R3_logistic_T3",
            "seed": 42,
            "chosen_param": f"C={best_c}",
            "tau": float(best_tau),
            "val_bal_acc": float(best_bal_acc),
        })

        # E. Necessary NLP Control: R1 TF-IDF Logistic
        best_c_r1 = 0.01
        best_bal_acc_r1 = -1.0
        best_tau_r1 = -np.inf

        for c_cand in [0.01, 0.1, 1.0, 10.0]:
            clf_r1 = SupervisedLogisticDetector(C=c_cand)
            clf_r1.fit(X_r1_concat_train, y_train)
            v_scores = clf_r1.decision_function(X_r1_concat_val)
            tau_cand, metrics_cand = select_validation_threshold(y_val, v_scores)

            if (metrics_cand["balanced_accuracy"] > best_bal_acc_r1) or (
                np.isclose(metrics_cand["balanced_accuracy"], best_bal_acc_r1) and c_cand < best_c_r1
            ):
                best_bal_acc_r1 = metrics_cand["balanced_accuracy"]
                best_c_r1 = c_cand
                best_tau_r1 = tau_cand

        clf_r1_final = SupervisedLogisticDetector(C=best_c_r1)
        clf_r1_final.fit(X_r1_concat_train, y_train)
        test_scores = clf_r1_final.decision_function(X_r1_concat_test)
        pred_test = (test_scores >= best_tau_r1).astype(int)

        oof_results["supervised_R1_logistic"][42]["scores"][test_idx] = test_scores
        oof_results["supervised_R1_logistic"][42]["preds"][test_idx] = pred_test
        oof_results["supervised_R1_logistic"][42]["tau"][f_idx] = best_tau_r1

        # F. Necessary Tabular Control: Inductive ECOD (R0-clean)
        ecod = InductiveECODVariant()
        ecod.fit(X_clean_train_norm)
        v_scores = ecod.decision_function(X_clean_val)
        best_tau_ecod, _ = select_validation_threshold(y_val, v_scores)
        test_scores = ecod.decision_function(X_clean_test)
        pred_test = (test_scores >= best_tau_ecod).astype(int)

        oof_results["inductive_ecod_R0_clean"][42]["scores"][test_idx] = test_scores
        oof_results["inductive_ecod_R0_clean"][42]["preds"][test_idx] = pred_test
        oof_results["inductive_ecod_R0_clean"][42]["tau"][f_idx] = best_tau_ecod

        # G. Necessary Tabular Control: Isolation Forest (R0-clean, multi-seed)
        for s in seeds:
            ifo = IsolationForestDetector(n_estimators=200, random_state=s)
            ifo.fit(X_clean_train_norm)
            v_scores = ifo.decision_function(X_clean_val)
            best_tau_if, _ = select_validation_threshold(y_val, v_scores)
            test_scores = ifo.decision_function(X_clean_test)
            pred_test = (test_scores >= best_tau_if).astype(int)

            oof_results["isolation_forest_R0_clean"][s]["scores"][test_idx] = test_scores
            oof_results["isolation_forest_R0_clean"][s]["preds"][test_idx] = pred_test
            oof_results["isolation_forest_R0_clean"][s]["tau"][f_idx] = best_tau_if

        # H. Legacy Controls: XGBoost and Autoencoder
        # XGBoost (supervised on R0 legacy)
        xgb_det = LegacyXGBoostDetector(random_state=42)
        xgb_det.fit(X_r0_leg_train, y_train)
        v_scores = xgb_det.decision_function(X_r0_leg_val)
        best_tau_xgb, _ = select_validation_threshold(y_val, v_scores)
        test_scores = xgb_det.decision_function(X_r0_leg_test)
        pred_test = (test_scores >= best_tau_xgb).astype(int)

        oof_results["legacy_xgboost_R0"][42]["scores"][test_idx] = test_scores
        oof_results["legacy_xgboost_R0"][42]["preds"][test_idx] = pred_test
        oof_results["legacy_xgboost_R0"][42]["tau"][f_idx] = best_tau_xgb

        # Autoencoder (normal-only on R0 legacy, multi-seed)
        for s in seeds:
            ae = LegacyAEDetector(epochs=100, lr=0.001, random_state=s)
            ae.fit(X_r0_leg_norm_train)
            v_scores = ae.decision_function(X_r0_leg_norm_val)
            best_tau_ae, _ = select_validation_threshold(y_val, v_scores)
            test_scores = ae.decision_function(X_r0_leg_norm_test)
            pred_test = (test_scores >= best_tau_ae).astype(int)

            oof_results["legacy_ae_R0"][s]["scores"][test_idx] = test_scores
            oof_results["legacy_ae_R0"][s]["preds"][test_idx] = pred_test
            oof_results["legacy_ae_R0"][s]["tau"][f_idx] = best_tau_ae

        # I. Secondary Extensions: Shallow DAE and Conditional Residual
        # Shallow DAE (multi-seed)
        for s in seeds:
            dae = ShallowDAEDetector(epochs=150, batch_size=16, random_state=s)
            dae.fit(X_clean_train_norm)
            v_scores = dae.decision_function(X_clean_val)
            best_tau_dae, _ = select_validation_threshold(y_val, v_scores)
            test_scores = dae.decision_function(X_clean_test)
            pred_test = (test_scores >= best_tau_dae).astype(int)

            oof_results["shallow_dae_R0_clean"][s]["scores"][test_idx] = test_scores
            oof_results["shallow_dae_R0_clean"][s]["preds"][test_idx] = pred_test
            oof_results["shallow_dae_R0_clean"][s]["tau"][f_idx] = best_tau_dae

        # Conditional Residual
        cond = ConditionalResidualDetector(ridge_alpha=10.0, logistic_c=0.1)
        cond.fit(X_clean_train_norm)
        v_scores = cond.decision_function(X_clean_val)
        best_tau_cond, _ = select_validation_threshold(y_val, v_scores)
        test_scores = cond.decision_function(X_clean_test)
        pred_test = (test_scores >= best_tau_cond).astype(int)

        oof_results["conditional_residual_R0_clean"][42]["scores"][test_idx] = test_scores
        oof_results["conditional_residual_R0_clean"][42]["preds"][test_idx] = pred_test
        oof_results["conditional_residual_R0_clean"][42]["tau"][f_idx] = best_tau_cond

    # =========================================================================
    # POOLED OUT-OF-FOLD (OOF) EVALUATION ACROSS ALL 284 INSTANCES
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPUTING POOLED OUT-OF-FOLD (OOF) METRICS (N=284)")
    print("=" * 70)

    summary_rows = []
    p_at_k_rows = []

    # Save OOF prediction records
    oof_predictions_df = pd.DataFrame({"instance_id": all_instance_ids, "true_label": all_labels})

    for m_def in model_definitions:
        m_name = m_def["name"]
        m_seeds = seeds if m_def["multi_seed"] else [42]

        for s in m_seeds:
            res = oof_results[m_name][s]
            scores = res["scores"]
            preds = res["preds"]

            # Invariant assertion: exact 284 instances
            assert len(scores) == 284
            assert len(preds) == 284

            metrics = compute_metrics(all_labels, scores, preds)
            p_k = compute_precision_at_k(all_instance_ids, all_labels, scores, [10, 20, 50])

            row = {
                "model": m_name,
                "seed": s,
                "category": m_def["category"],
                **metrics,
            }
            summary_rows.append(row)

            p_k_row = {
                "model": m_name,
                "seed": s,
                **p_k,
                "prevalence": float(np.mean(all_labels)),
            }
            p_at_k_rows.append(p_k_row)

            # Store in predictions dataframe
            oof_predictions_df[f"{m_name}_seed_{s}_score"] = scores
            oof_predictions_df[f"{m_name}_seed_{s}_pred"] = preds

            if s == 42:
                print(
                    f"{m_name:<30} | BalAcc: {metrics['balanced_accuracy']:.3f} | "
                    f"FPR: {metrics['fpr']:.3f} | Recall: {metrics['failure_recall']:.3f} | "
                    f"UnflagSucc: {metrics['unflagged_success_fraction']:.3f} | "
                    f"TN/FP/FN/TP: {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}"
                )

    # Save OOF predictions CSV
    oof_predictions_df.to_csv(OUTPUTS_DIR / "oof_predictions.csv", index=False)
    print(f"Saved complete OOF predictions to {OUTPUTS_DIR / 'oof_predictions.csv'}")

    # Save Main Results Table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "main_results.csv", index=False)

    p_at_k_df = pd.DataFrame(p_at_k_rows)
    p_at_k_df.to_csv(TABLES_DIR / "precision_at_k.csv", index=False)

    # Save Fold Details Table
    fold_details_df = pd.DataFrame(fold_details_records)
    fold_details_df.to_csv(TABLES_DIR / "fold_parameters.csv", index=False)

    # =========================================================================
    # REPOSITORY-HELD-OUT DOMAIN SHIFT SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING REPOSITORY-HELD-OUT SENSITIVITY ANALYSIS")
    print("=" * 70)

    repo_results = []
    for r_fold in repo_folds:
        r_name = r_fold["repo"]
        is_eval = r_fold["is_evaluable"]
        test_r_ids = r_fold["test_instance_ids"]
        train_r_ids = r_fold["train_instance_ids"]
        val_r_ids = r_fold["val_instance_ids"]

        if not is_eval:
            repo_results.append({
                "repo": r_name,
                "is_evaluable": False,
                "test_count": r_fold["test_count"],
                "reason": "Missing classes or insufficient normal fitting rows",
                "knn_R3_bal_acc": float("nan"),
                "knn_R3_fpr": float("nan"),
                "knn_R3_recall": float("nan"),
            })
            continue

        test_r_idx = [id_to_idx[i] for i in test_r_ids]
        train_r_idx = [id_to_idx[i] for i in train_r_ids]
        val_r_idx = [id_to_idx[i] for i in val_r_ids]

        y_r_val = all_labels[val_r_idx]
        y_r_test = all_labels[test_r_idx]

        # Normal fitting rows in train
        train_r_norm_idx = [i for i in train_r_idx if all_labels[i] == 0]

        r3_r_norm_log = all_log_embeds[train_r_norm_idx]
        r3_r_norm_patch = all_patch_embeds[train_r_norm_idx]
        r3_r_norm_empty = all_patch_empty[train_r_norm_idx]

        knn_repo = KnnNormalDistanceDetector(k=3, metric="dual")
        knn_repo.fit_embeddings(r3_r_norm_log, r3_r_norm_patch, r3_r_norm_empty)

        v_scores = knn_repo.score_dual(all_log_embeds[val_r_idx], all_patch_embeds[val_r_idx], all_patch_empty[val_r_idx])
        tau_r, _ = select_validation_threshold(y_r_val, v_scores)

        t_scores = knn_repo.score_dual(all_log_embeds[test_r_idx], all_patch_embeds[test_r_idx], all_patch_empty[test_r_idx])
        pred_r = (t_scores >= tau_r).astype(int)

        m_r = compute_metrics(y_r_test, t_scores, pred_r)

        repo_results.append({
            "repo": r_name,
            "is_evaluable": True,
            "test_count": r_fold["test_count"],
            "test_normal_count": r_fold["test_normal_count"],
            "test_anomaly_count": r_fold["test_anomaly_count"],
            "tau": float(tau_r),
            "knn_R3_bal_acc": m_r["balanced_accuracy"],
            "knn_R3_fpr": m_r["fpr"],
            "knn_R3_recall": m_r["failure_recall"],
        })

    repo_df = pd.DataFrame(repo_results)
    repo_df.to_csv(TABLES_DIR / "repo_held_out_sensitivity.csv", index=False)
    print(f"Saved repository sensitivity analysis to {TABLES_DIR / 'repo_held_out_sensitivity.csv'}")

    total_time = time.perf_counter() - start_clock
    mem_peak = process.memory_info().rss / (1024 * 1024)

    timing_summary = {
        "total_experiment_wall_clock_seconds": total_time,
        "peak_rss_memory_mb": float(mem_peak),
        "total_instances_evaluated": 284,
        "n_folds": 5,
    }
    with open(TABLES_DIR / "timing_profile.json", "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"\nExperiment complete in {total_time:.1f}s. Peak memory: {mem_peak:.1f}MB.")


if __name__ == "__main__":
    run_experiment()
