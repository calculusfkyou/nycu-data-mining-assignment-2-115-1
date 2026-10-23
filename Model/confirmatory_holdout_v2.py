"""Run once-only untouched confirmatory evaluation on 20240402_sweagent_claude3opus (N=271).

Evaluates:
1. Frozen Cross-Run Transfer: Models, scalers, reference banks, hyperparameters (k, C), and thresholds tau
   frozen strictly from the 5 outer folds of 20240402_sweagent_gpt4, evaluated directly on all 271
   untouched Claude-3-Opus runs (reporting mean +/- SD across the 5 frozen fold models).
2. Locked-Protocol 5-Fold Confirmatory Replication: The exact same 5-fold stratified outer CV protocol
   (seed 42 outer, seed 100+f inner validation, FPR <= 20% recall-maximizing rule) run once on the
   271 Claude-3-Opus runs with zero hyperparameter or architecture changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from Model.experiment_v2 import compute_metrics
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


def main() -> None:
    # Load primary GPT-4 data & splits
    gpt_folds = json.loads(Path("Dataset/v2/splits.json").read_text())["outer_folds"]
    gpt_df = load_raw_features_df()
    gpt_ids = gpt_df["instance_id"].tolist()
    gpt_idx_map = {x: i for i, x in enumerate(gpt_ids)}
    y_gpt = gpt_df["label"].to_numpy()
    X_gpt_clean, clean_cols = get_r0_clean_unscaled_features(gpt_df)
    X_gpt_leg, _ = get_r0_legacy_features(gpt_df)

    gpt_r2 = np.load("Dataset/v2/embeddings/r2_minilm.npz")
    gpt_r3 = np.load("Dataset/v2/embeddings/r3_codebert.npz")
    gli = {str(x): i for i, x in enumerate(gpt_r2["instance_ids"])}
    gpi = {str(x): i for i, x in enumerate(gpt_r3["instance_ids"])}
    gpt_log = np.array([gpt_r2["embeddings"][gli[x]] for x in gpt_ids], dtype=np.float32)
    gpt_patch = np.array([gpt_r3["embeddings"][gpi[x]] for x in gpt_ids], dtype=np.float32)
    gpt_empty = np.array([gpt_r3["is_empty"][gpi[x]] for x in gpt_ids], dtype=bool)
    scale = float(1.0 / np.sqrt(2.0))
    gpt_r3_concat = np.hstack([scale * gpt_log, scale * gpt_patch]).astype(np.float32)

    # Load untouched Claude-3-Opus data & embeddings
    opus_df = pd.read_csv("Dataset/v2/claude3opus_features.csv").sort_values("instance_id").reset_index(drop=True)
    opus_ids = opus_df["instance_id"].tolist()
    y_opus = opus_df["label"].to_numpy()
    X_opus_clean, _ = get_r0_clean_unscaled_features(opus_df)
    X_opus_leg, _ = get_r0_legacy_features(opus_df)

    opus_r2 = np.load("Dataset/v2/embeddings/claude3opus_r2_minilm.npz")
    opus_r3 = np.load("Dataset/v2/embeddings/claude3opus_r3_codebert.npz")
    oli = {str(x): i for i, x in enumerate(opus_r2["instance_ids"])}
    opi = {str(x): i for i, x in enumerate(opus_r3["instance_ids"])}
    opus_log = np.array([opus_r2["embeddings"][oli[x]] for x in opus_ids], dtype=np.float32)
    opus_patch = np.array([opus_r3["embeddings"][opi[x]] for x in opus_ids], dtype=np.float32)
    opus_empty = np.array([opus_r3["is_empty"][opi[x]] for x in opus_ids], dtype=bool)
    opus_r3_concat = np.hstack([scale * opus_log, scale * opus_patch]).astype(np.float32)

    fold_params = pd.read_csv("Model/tables/v2/fold_parameters.csv")

    # =========================================================================
    # MODE A: ZERO-SHOT CROSS-AGENT TRANSFER (FROZEN GPT-4 FOLD MODELS -> OPUS)
    # =========================================================================
    transfer_rows = []
    for f in gpt_folds:
        fi = f["fold"]
        tr_i = [gpt_idx_map[x] for x in f["train_instance_ids"]]
        tr_norm_i = [gpt_idx_map[x] for x in f["train_normal_instance_ids"]]
        val_i = [gpt_idx_map[x] for x in f["val_instance_ids"]]
        y_val = y_gpt[val_i]

        # 1. T1 Heuristic
        heur = RuleHeuristicDetector()
        tau_t1, _ = select_validation_threshold(y_val, heur.decision_function(gpt_df.iloc[val_i]))
        s_t1 = heur.decision_function(opus_df)
        p_t1 = (s_t1 >= tau_t1).astype(int)
        transfer_rows.append({"model": "heuristic_T1", "fold": fi, **compute_metrics(y_opus, s_t1, p_t1)})

        # 2. T2 Dual Encoder kNN
        par_t2 = fold_params[(fold_params.fold == fi) & (fold_params.model == "knn_R3_T2")].iloc[0]
        k_t2 = int(par_t2.chosen_param.split("=")[1])
        tau_t2 = float(par_t2.tau)
        knn = KnnNormalDistanceDetector(k=k_t2, metric="dual").fit_embeddings(
            gpt_log[tr_norm_i], gpt_patch[tr_norm_i], gpt_empty[tr_norm_i]
        )
        s_t2 = knn.score_dual(opus_log, opus_patch, opus_empty)
        p_t2 = (s_t2 >= tau_t2).astype(int)
        transfer_rows.append({"model": "knn_R3_T2", "fold": fi, **compute_metrics(y_opus, s_t2, p_t2)})

        # 3. CodeBERT Patch-Only kNN
        ch_params = pd.read_csv("Model/tables/v2/ablation_channels_params.csv").iloc[fi]
        k_po = int(ch_params.best_k_patch)
        tau_po = float(ch_params.tau_patch)
        knn_po = KnnNormalDistanceDetector(k=k_po, metric="cosine").fit_embeddings(
            gpt_patch[tr_norm_i], normal_patch_empty=gpt_empty[tr_norm_i]
        )
        s_po = knn_po.score_single_cosine(opus_patch, opus_empty)
        p_po = (s_po >= tau_po).astype(int)
        transfer_rows.append({"model": "patch_only_codebert", "fold": fi, **compute_metrics(y_opus, s_po, p_po)})

        # 4. T3 Supervised Encoder Logistic
        par_t3 = fold_params[(fold_params.fold == fi) & (fold_params.model == "supervised_R3_logistic_T3")].iloc[0]
        c_t3 = float(par_t3.chosen_param.split("=")[1])
        tau_t3 = float(par_t3.tau)
        clf_t3 = SupervisedLogisticDetector(C=c_t3).fit(gpt_r3_concat[tr_i], y_gpt[tr_i])
        s_t3 = clf_t3.decision_function(opus_r3_concat)
        p_t3 = (s_t3 >= tau_t3).astype(int)
        transfer_rows.append({"model": "supervised_R3_logistic_T3", "fold": fi, **compute_metrics(y_opus, s_t3, p_t3)})

        # 5. Inductive ECOD (R0-clean)
        sc_clean = NormalOnlyRobustScaler(clean_cols).fit(X_gpt_clean[tr_norm_i])
        ecod = InductiveECODVariant().fit(sc_clean.transform(X_gpt_clean[tr_norm_i]))
        tau_ec, _ = select_validation_threshold(y_val, ecod.decision_function(sc_clean.transform(X_gpt_clean[val_i])))
        s_ec = ecod.decision_function(sc_clean.transform(X_opus_clean))
        p_ec = (s_ec >= tau_ec).astype(int)
        transfer_rows.append({"model": "inductive_ecod_R0_clean", "fold": fi, **compute_metrics(y_opus, s_ec, p_ec)})

        # 6. Isolation Forest (R0-clean)
        ifo = IsolationForestDetector(n_estimators=200, random_state=42).fit(sc_clean.transform(X_gpt_clean[tr_norm_i]))
        tau_if, _ = select_validation_threshold(y_val, ifo.decision_function(sc_clean.transform(X_gpt_clean[val_i])))
        s_if = ifo.decision_function(sc_clean.transform(X_opus_clean))
        p_if = (s_if >= tau_if).astype(int)
        transfer_rows.append({"model": "isolation_forest_R0_clean", "fold": fi, **compute_metrics(y_opus, s_if, p_if)})

        # 7. Conditional Residual (R0-clean)
        cond = ConditionalResidualDetector(ridge_alpha=10.0, logistic_c=0.1).fit(sc_clean.transform(X_gpt_clean[tr_norm_i]))
        tau_cd, _ = select_validation_threshold(y_val, cond.decision_function(sc_clean.transform(X_gpt_clean[val_i])))
        s_cd = cond.decision_function(sc_clean.transform(X_opus_clean))
        p_cd = (s_cd >= tau_cd).astype(int)
        transfer_rows.append({"model": "conditional_residual_R0_clean", "fold": fi, **compute_metrics(y_opus, s_cd, p_cd)})

        # 8. XGBoost (R0 legacy)
        sc_leg = StandardScaler().fit(X_gpt_leg[tr_i])
        xgb_m = LegacyXGBoostDetector(random_state=42).fit(sc_leg.transform(X_gpt_leg[tr_i]), y_gpt[tr_i])
        tau_xg, _ = select_validation_threshold(y_val, xgb_m.decision_function(sc_leg.transform(X_gpt_leg[val_i])))
        s_xg = xgb_m.decision_function(sc_leg.transform(X_opus_leg))
        p_xg = (s_xg >= tau_xg).astype(int)
        transfer_rows.append({"model": "legacy_xgboost_R0", "fold": fi, **compute_metrics(y_opus, s_xg, p_xg)})

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv("Model/tables/v2/confirmatory_claude3opus_transfer.csv", index=False)

    # =========================================================================
    # MODE B: LOCKED-PROTOCOL 5-FOLD CONFIRMATORY REPLICATION ON CLAUDE-3-OPUS
    # =========================================================================
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_rep = [
        "heuristic_T1",
        "knn_R3_T2",
        "patch_only_codebert",
        "supervised_R3_logistic_T3",
        "inductive_ecod_R0_clean",
        "isolation_forest_R0_clean",
        "conditional_residual_R0_clean",
        "legacy_xgboost_R0",
    ]
    oof_scores = {m: np.zeros(len(opus_ids), dtype=np.float32) for m in models_rep}
    oof_preds = {m: np.zeros(len(opus_ids), dtype=int) for m in models_rep}
    fold_rep_rows = []

    for fi, (dev_idx, test_idx) in enumerate(skf_outer.split(opus_ids, y_opus)):
        skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + fi)
        tr_sub, val_sub = next(skf_inner.split(dev_idx, y_opus[dev_idx]))
        tr_idx = dev_idx[tr_sub]
        val_idx = dev_idx[val_sub]
        tr_norm_idx = np.array([i for i in tr_idx if y_opus[i] == 0], dtype=int)

        y_tr, y_val, y_te = y_opus[tr_idx], y_opus[val_idx], y_opus[test_idx]

        # 1. T1
        heur = RuleHeuristicDetector()
        tau, _ = select_validation_threshold(y_val, heur.decision_function(opus_df.iloc[val_idx]))
        s = heur.decision_function(opus_df.iloc[test_idx])
        p = (s >= tau).astype(int)
        oof_scores["heuristic_T1"][test_idx], oof_preds["heuristic_T1"][test_idx] = s, p

        # 2. T2 Dual Encoder kNN
        best_k, best_ba, best_tau = 1, -1.0, -np.inf
        for kc in [1, 3, 5]:
            det = KnnNormalDistanceDetector(k=kc, metric="dual").fit_embeddings(
                opus_log[tr_norm_idx], opus_patch[tr_norm_idx], opus_empty[tr_norm_idx]
            )
            tc, mc = select_validation_threshold(
                y_val, det.score_dual(opus_log[val_idx], opus_patch[val_idx], opus_empty[val_idx])
            )
            if mc["balanced_accuracy"] > best_ba or (np.isclose(mc["balanced_accuracy"], best_ba) and kc < best_k):
                best_ba, best_k, best_tau = mc["balanced_accuracy"], kc, tc
        det_t2 = KnnNormalDistanceDetector(k=best_k, metric="dual").fit_embeddings(
            opus_log[tr_norm_idx], opus_patch[tr_norm_idx], opus_empty[tr_norm_idx]
        )
        s = det_t2.score_dual(opus_log[test_idx], opus_patch[test_idx], opus_empty[test_idx])
        oof_scores["knn_R3_T2"][test_idx], oof_preds["knn_R3_T2"][test_idx] = s, (s >= best_tau).astype(int)

        # 3. CodeBERT Patch-only kNN
        best_k, best_ba, best_tau = 1, -1.0, -np.inf
        for kc in [1, 3, 5]:
            det = KnnNormalDistanceDetector(k=kc, metric="cosine").fit_embeddings(
                opus_patch[tr_norm_idx], normal_patch_empty=opus_empty[tr_norm_idx]
            )
            tc, mc = select_validation_threshold(
                y_val, det.score_single_cosine(opus_patch[val_idx], opus_empty[val_idx])
            )
            if mc["balanced_accuracy"] > best_ba or (np.isclose(mc["balanced_accuracy"], best_ba) and kc < best_k):
                best_ba, best_k, best_tau = mc["balanced_accuracy"], kc, tc
        det_po = KnnNormalDistanceDetector(k=best_k, metric="cosine").fit_embeddings(
            opus_patch[tr_norm_idx], normal_patch_empty=opus_empty[tr_norm_idx]
        )
        s = det_po.score_single_cosine(opus_patch[test_idx], opus_empty[test_idx])
        oof_scores["patch_only_codebert"][test_idx], oof_preds["patch_only_codebert"][test_idx] = s, (s >= best_tau).astype(int)

        # 4. T3 Logistic
        best_c, best_ba, best_tau = 0.01, -1.0, -np.inf
        for cc in [0.01, 0.1, 1.0, 10.0]:
            clf = SupervisedLogisticDetector(C=cc).fit(opus_r3_concat[tr_idx], y_tr)
            tc, mc = select_validation_threshold(y_val, clf.decision_function(opus_r3_concat[val_idx]))
            if mc["balanced_accuracy"] > best_ba or (np.isclose(mc["balanced_accuracy"], best_ba) and cc < best_c):
                best_ba, best_c, best_tau = mc["balanced_accuracy"], cc, tc
        clf_t3 = SupervisedLogisticDetector(C=best_c).fit(opus_r3_concat[tr_idx], y_tr)
        s = clf_t3.decision_function(opus_r3_concat[test_idx])
        oof_scores["supervised_R3_logistic_T3"][test_idx], oof_preds["supervised_R3_logistic_T3"][test_idx] = s, (s >= best_tau).astype(int)

        # 5. ECOD, IF, Conditional Residual
        sc_clean = NormalOnlyRobustScaler(clean_cols).fit(X_opus_clean[tr_norm_idx])
        X_cn = sc_clean.transform(X_opus_clean[tr_norm_idx])
        X_cv = sc_clean.transform(X_opus_clean[val_idx])
        X_cte = sc_clean.transform(X_opus_clean[test_idx])

        ecod = InductiveECODVariant().fit(X_cn)
        tau, _ = select_validation_threshold(y_val, ecod.decision_function(X_cv))
        s = ecod.decision_function(X_cte)
        oof_scores["inductive_ecod_R0_clean"][test_idx], oof_preds["inductive_ecod_R0_clean"][test_idx] = s, (s >= tau).astype(int)

        ifo = IsolationForestDetector(n_estimators=200, random_state=42).fit(X_cn)
        tau, _ = select_validation_threshold(y_val, ifo.decision_function(X_cv))
        s = ifo.decision_function(X_cte)
        oof_scores["isolation_forest_R0_clean"][test_idx], oof_preds["isolation_forest_R0_clean"][test_idx] = s, (s >= tau).astype(int)

        cond = ConditionalResidualDetector(ridge_alpha=10.0, logistic_c=0.1).fit(X_cn)
        tau, _ = select_validation_threshold(y_val, cond.decision_function(X_cv))
        s = cond.decision_function(X_cte)
        oof_scores["conditional_residual_R0_clean"][test_idx], oof_preds["conditional_residual_R0_clean"][test_idx] = s, (s >= tau).astype(int)

        sc_leg = StandardScaler().fit(X_opus_leg[tr_idx])
        xgb_m = LegacyXGBoostDetector(random_state=42).fit(sc_leg.transform(X_opus_leg[tr_idx]), y_tr)
        tau, _ = select_validation_threshold(y_val, xgb_m.decision_function(sc_leg.transform(X_opus_leg[val_idx])))
        s = xgb_m.decision_function(sc_leg.transform(X_opus_leg[test_idx]))
        oof_scores["legacy_xgboost_R0"][test_idx], oof_preds["legacy_xgboost_R0"][test_idx] = s, (s >= tau).astype(int)

        for m in models_rep:
            fold_rep_rows.append({"model": m, "fold": fi, **compute_metrics(y_te, oof_scores[m][test_idx], oof_preds[m][test_idx])})

    rep_fold_df = pd.DataFrame(fold_rep_rows)
    rep_summary = []
    for m in models_rep:
        pooled = compute_metrics(y_opus, oof_scores[m], oof_preds[m])
        sub = rep_fold_df[rep_fold_df.model == m]
        t2_sub = rep_fold_df[rep_fold_df.model == "knn_R3_T2"]
        if m == "knn_R3_T2":
            p_val = 1.0
        else:
            _, p_val = stats.ttest_rel(sub["balanced_accuracy"], t2_sub["balanced_accuracy"])
        rep_summary.append({
            "model": m,
            "fold_ba_mean": float(sub["balanced_accuracy"].mean()),
            "fold_ba_sd": float(sub["balanced_accuracy"].std(ddof=1)),
            "fold_ap_mean": float(sub["average_precision"].mean()),
            "fold_ap_sd": float(sub["average_precision"].std(ddof=1)),
            "paired_t_p_vs_t2": float(p_val),
            **pooled,
        })
    rep_df = pd.DataFrame(rep_summary)
    rep_df.to_csv("Model/tables/v2/confirmatory_claude3opus_oof.csv", index=False)

    print("=== MODE A: ZERO-SHOT TRANSFER (GPT-4 -> CLAUDE-3-OPUS, 5 FOLD MODELS MEAN +/- SD) ===")
    for m in models_rep:
        sub = transfer_df[transfer_df.model == m]
        print(
            f"{m:<30} | BA: {sub.balanced_accuracy.mean():.3f} +/- {sub.balanced_accuracy.std(ddof=1):.3f} | "
            f"FPR: {sub.fpr.mean():.3f} +/- {sub.fpr.std(ddof=1):.3f} | "
            f"Rec: {sub.failure_recall.mean():.3f} +/- {sub.failure_recall.std(ddof=1):.3f} | "
            f"AP: {sub.average_precision.mean():.3f}"
        )

    print("\n=== MODE B: LOCKED-PROTOCOL 5-FOLD CONFIRMATORY OOF ON CLAUDE-3-OPUS (N=271) ===")
    for r in rep_df.itertuples():
        print(
            f"{r.model:<30} | Pooled BA: {r.balanced_accuracy:.3f} (Fold {r.fold_ba_mean:.3f} +/- {r.fold_ba_sd:.3f}) | "
            f"FPR: {r.fpr:.3f} | Rec: {r.failure_recall:.3f} | TN/FP/FN/TP: {r.tn}/{r.fp}/{r.fn}/{r.tp} | "
            f"Fold AP: {r.fold_ap_mean:.3f} | p vs T2: {r.paired_t_p_vs_t2:.4f}"
        )


if __name__ == "__main__":
    main()
