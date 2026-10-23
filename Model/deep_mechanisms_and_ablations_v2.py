"""Compute mechanistic verifications, threshold policy ablations, latency profiles, and confirmatory holdout evaluation for HW2 v2."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from Model.experiment_v2 import (
    compute_metrics,
    fit_r1_tfidf,
    load_raw_text_corpus,
    transform_r1_tfidf,
)
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
    R0_CLEAN_COLUMNS,
    NormalOnlyRobustScaler,
    get_r0_clean_unscaled_features,
    get_r0_legacy_features,
    load_raw_features_df,
)
from tools.build_text_embeddings import compute_channel_distance, compute_dual_distance


def select_threshold_by_rule(
    val_y: np.ndarray,
    val_scores: np.ndarray,
    rule: str,
) -> float:
    """Select validation threshold tau according to specified policy rule."""
    unique_scores = np.unique(val_scores)
    candidates = np.concatenate([[-np.inf], unique_scores, [np.inf]])

    if rule == "fpr_le_20":
        tau, _ = select_validation_threshold(val_y, val_scores, max_fpr=0.20)
        return float(tau)

    best_tau = float(candidates[0])
    best_obj = -np.inf

    for tau in candidates:
        pred = (val_scores >= tau).astype(int)
        tn = int(np.sum((val_y == 0) & (pred == 0)))
        fp = int(np.sum((val_y == 0) & (pred == 1)))
        fn = int(np.sum((val_y == 1) & (pred == 0)))
        tp = int(np.sum((val_y == 1) & (pred == 1)))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ba = 0.5 * (rec + spec)
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        f2 = (5 * tp) / (5 * tp + fp + 4 * fn) if (5 * tp + fp + 4 * fn) > 0 else 0.0
        # Cost per validation instance: $1200 * FN + $30 * (TP + FP)
        neg_cost = -(1200.0 * fn + 30.0 * (tp + fp))

        if rule == "max_ba":
            obj = (ba, -fpr, float(tau))
        elif rule == "max_f1":
            obj = (f1, ba, float(tau))
        elif rule == "max_f2":
            obj = (f2, ba, float(tau))
        elif rule == "min_cost":
            obj = (neg_cost, ba, float(tau))
        else:
            raise ValueError(f"Unknown rule: {rule}")

        if best_obj == -np.inf or obj > best_obj:
            best_obj = obj
            best_tau = float(tau)

    return best_tau


def main() -> None:
    folds = json.loads(Path("Dataset/v2/splits.json").read_text())["outer_folds"]
    raw_df = load_raw_features_df()
    raw_indexed = raw_df.set_index("instance_id")
    oof = pd.read_csv("Model/outputs/v2/oof_predictions.csv")
    ids = oof["instance_id"].tolist()
    id_to_idx = {x: i for i, x in enumerate(ids)}
    y_all = oof["true_label"].to_numpy()

    r2_data = np.load("Dataset/v2/embeddings/r2_minilm.npz")
    r3_data = np.load("Dataset/v2/embeddings/r3_codebert.npz")
    li = {str(x): i for i, x in enumerate(r2_data["instance_ids"])}
    pi = {str(x): i for i, x in enumerate(r3_data["instance_ids"])}
    all_log = np.array([r2_data["embeddings"][li[x]] for x in ids], dtype=np.float32)
    all_patch = np.array([r3_data["embeddings"][pi[x]] for x in ids], dtype=np.float32)
    all_empty = np.array([r3_data["is_empty"][pi[x]] for x in ids], dtype=bool)

    fold_params = pd.read_csv("Model/tables/v2/fold_parameters.csv")

    # =========================================================================
    # 1. CHANNEL DISTANCE VARIANCE DECOMPOSITION & CONTAMINATION MECHANISM
    # =========================================================================
    k_nearest_log_dists = []
    k_nearest_patch_dists = []
    k_nearest_dual_dists = []

    for f in folds:
        fi = f["fold"]
        par = fold_params[(fold_params.fold == fi) & (fold_params.model == "knn_R3_T2")].iloc[0]
        k = int(par.chosen_param.split("=")[1])
        ri = [id_to_idx[x] for x in f["train_normal_instance_ids"]]
        ti = [id_to_idx[x] for x in f["test_instance_ids"]]

        for q_idx in ti:
            d_logs = np.array([compute_channel_distance(all_log[q_idx], all_log[r]) for r in ri])
            d_patches = np.array([
                compute_channel_distance(all_patch[q_idx], all_patch[r], all_empty[q_idx], all_empty[r])
                for r in ri
            ])
            d_duals = 0.5 * (d_logs + d_patches)
            order = np.argsort(d_duals)[:k]
            k_nearest_log_dists.append(float(d_logs[order].mean()))
            k_nearest_patch_dists.append(float(d_patches[order].mean()))
            k_nearest_dual_dists.append(float(d_duals[order].mean()))

    arr_log = np.array(k_nearest_log_dists)
    arr_patch = np.array(k_nearest_patch_dists)
    arr_dual = np.array(k_nearest_dual_dists)
    var_log = float(np.var(0.5 * arr_log, ddof=1))
    var_patch = float(np.var(0.5 * arr_patch, ddof=1))
    cov_lp = float(np.cov(0.5 * arr_log, 0.5 * arr_patch, ddof=1)[0, 1])
    var_dual = float(np.var(arr_dual, ddof=1))

    # Check how often injected failures enter k-nearest neighbors at p=0.05 and p=0.10
    contam_records = json.loads(Path("Dataset/v2/splits.json").read_text())["outer_folds"]
    contam_fits = json.loads(Path("Model/tables/v2/contamination_fitting_ids.json").read_text())
    contam_mech = {}
    for p_val in [0.05, 0.10]:
        hit_rates = []
        fail_score_shifts = []
        norm_score_shifts = []
        for seed in [42, 123, 456]:
            hits = 0
            total_q = 0
            s_contam = np.zeros(284, dtype=np.float32)
            for f in folds:
                fi = f["fold"]
                rec = next(
                    r for r in contam_fits
                    if r["seed"] == seed and np.isclose(r["p_contamination"], p_val) and r["fold"] == fi
                )
                k = rec["k"]
                refs = [id_to_idx[x] for x in rec["fitting_ids"]]
                inj_set = set(id_to_idx[x] for x in rec["injected_ids"])
                ti = [id_to_idx[x] for x in f["test_instance_ids"]]
                for q_idx in ti:
                    d_duals = np.array([
                        compute_dual_distance(
                            all_log[q_idx], all_log[r],
                            all_patch[q_idx], all_patch[r],
                            all_empty[q_idx], all_empty[r],
                        )
                        for r in refs
                    ])
                    order = np.argsort(d_duals)[:k]
                    chosen_refs = [refs[o] for o in order]
                    if any(cr in inj_set for cr in chosen_refs):
                        hits += 1
                    total_q += 1
                    s_contam[q_idx] = float(d_duals[order].mean())
            hit_rates.append(hits / total_q)
            clean_s = oof["knn_R3_T2_seed_42_score"].to_numpy()
            fail_score_shifts.append(float(np.mean(s_contam[y_all == 1] - clean_s[y_all == 1])))
            norm_score_shifts.append(float(np.mean(s_contam[y_all == 0] - clean_s[y_all == 0])))
        contam_mech[str(p_val)] = {
            "knn_injected_hit_rate_mean": float(np.mean(hit_rates)),
            "knn_injected_hit_rate_sd": float(np.std(hit_rates, ddof=1)),
            "fail_score_shift_mean": float(np.mean(fail_score_shifts)),
            "norm_score_shift_mean": float(np.mean(norm_score_shifts)),
        }

    # Also test contamination on Inductive ECOD (R0-clean) at p=0, 0.05, 0.10 to contrast with R3!
    X_r0_clean_unscaled, clean_col_names = get_r0_clean_unscaled_features(raw_df)
    ecod_contam_rows = []
    for seed in [42, 123, 456]:
        for p_val in [0.0, 0.05, 0.10]:
            s_ecod = np.zeros(284, dtype=np.float32)
            p_ecod_recal = np.zeros(284, dtype=int)
            for f in folds:
                fi = f["fold"]
                rec = next(
                    r for r in contam_fits
                    if r["seed"] == seed and np.isclose(r["p_contamination"], p_val) and r["fold"] == fi
                )
                refs = [id_to_idx[x] for x in rec["fitting_ids"]]
                vi = [id_to_idx[x] for x in f["val_instance_ids"]]
                ti = [id_to_idx[x] for x in f["test_instance_ids"]]
                scaler = NormalOnlyRobustScaler(clean_col_names).fit(X_r0_clean_unscaled[refs])
                ecod = InductiveECODVariant().fit(scaler.transform(X_r0_clean_unscaled[refs]))
                v_sc = ecod.decision_function(scaler.transform(X_r0_clean_unscaled[vi]))
                tau_c, _ = select_validation_threshold(y_all[vi], v_sc)
                t_sc = ecod.decision_function(scaler.transform(X_r0_clean_unscaled[ti]))
                s_ecod[ti] = t_sc
                p_ecod_recal[ti] = (t_sc >= tau_c).astype(int)
            m_ec = compute_metrics(y_all, s_ecod, p_ecod_recal)
            ecod_contam_rows.append({"seed": seed, "p_contamination": p_val, **m_ec})
    ecod_contam_df = pd.DataFrame(ecod_contam_rows)
    ecod_contam_df.to_csv("Model/tables/v2/ablation_contamination_ecod.csv", index=False)

    # =========================================================================
    # 2. THRESHOLD POLICY ABLATION (ON T2 AND ECOD)
    # =========================================================================
    thresh_rows = []
    for model_key in ["knn_R3_T2", "inductive_ecod_R0_clean"]:
        for rule in ["fpr_le_20", "max_ba", "max_f1", "max_f2", "min_cost"]:
            oof_p = np.zeros(284, dtype=int)
            oof_s = oof[f"{model_key}_seed_42_score"].to_numpy()
            taus = []
            for f in folds:
                fi = f["fold"]
                ri = [id_to_idx[x] for x in f["train_normal_instance_ids"]]
                vi = [id_to_idx[x] for x in f["val_instance_ids"]]
                ti = [id_to_idx[x] for x in f["test_instance_ids"]]
                if model_key == "knn_R3_T2":
                    par = fold_params[(fold_params.fold == fi) & (fold_params.model == "knn_R3_T2")].iloc[0]
                    k = int(par.chosen_param.split("=")[1])
                    det = KnnNormalDistanceDetector(k=k, metric="dual").fit_embeddings(
                        all_log[ri], all_patch[ri], all_empty[ri]
                    )
                    v_scores = det.score_dual(all_log[vi], all_patch[vi], all_empty[vi])
                else:
                    scaler = NormalOnlyRobustScaler(clean_col_names).fit(X_r0_clean_unscaled[ri])
                    det = InductiveECODVariant().fit(scaler.transform(X_r0_clean_unscaled[ri]))
                    v_scores = det.decision_function(scaler.transform(X_r0_clean_unscaled[vi]))

                tau_r = select_threshold_by_rule(y_all[vi], v_scores, rule)
                taus.append(tau_r)
                oof_p[ti] = (oof_s[ti] >= tau_r).astype(int)

            m = compute_metrics(y_all, oof_s, oof_p)
            f1 = (2 * m["tp"]) / (2 * m["tp"] + m["fp"] + m["fn"]) if (2 * m["tp"] + m["fp"] + m["fn"]) > 0 else 0.0
            f2 = (5 * m["tp"]) / (5 * m["tp"] + m["fp"] + 4 * m["fn"]) if (5 * m["tp"] + m["fp"] + 4 * m["fn"]) > 0 else 0.0
            # Monthly cost projection on 500 tasks at natural prevalence (230/284 = 80.986% -> 404.93 failures, 95.07 normals)
            scale_500 = 500.0 / 284.0
            monthly_escapes = m["fn"] * scale_500
            monthly_reviews = (m["tp"] + m["fp"]) * scale_500
            monthly_cost = 1200.0 * monthly_escapes + 30.0 * monthly_reviews
            thresh_rows.append({
                "model": model_key,
                "rule": rule,
                "mean_tau": float(np.mean(taus)),
                "f1": float(f1),
                "f2": float(f2),
                "monthly_escapes_500": float(monthly_escapes),
                "monthly_reviews_500": float(monthly_reviews),
                "monthly_cost_usd_500": float(monthly_cost),
                **m,
            })

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv("Model/tables/v2/ablation_thresholds.csv", index=False)

    # =========================================================================
    # 3. PER-METHOD TRAINING TIME (s) & WARMED INFERENCE LATENCY (ms/sample)
    # =========================================================================
    X_r0_legacy, _ = get_r0_legacy_features(raw_df)
    f0 = folds[0]
    tr_idx = [id_to_idx[x] for x in f0["train_instance_ids"]]
    tr_norm_idx = [id_to_idx[x] for x in f0["train_normal_instance_ids"]]
    te_idx = [id_to_idx[x] for x in f0["test_instance_ids"]]
    n_te = len(te_idx)

    scaler_clean = NormalOnlyRobustScaler(clean_col_names).fit(X_r0_clean_unscaled[tr_norm_idx])
    X_c_norm = scaler_clean.transform(X_r0_clean_unscaled[tr_norm_idx])
    X_c_te = scaler_clean.transform(X_r0_clean_unscaled[te_idx])

    scaler_leg = StandardScaler().fit(X_r0_legacy[tr_idx])
    X_l_tr = scaler_leg.transform(X_r0_legacy[tr_idx])
    X_l_te = scaler_leg.transform(X_r0_legacy[te_idx])

    scaler_leg_n = StandardScaler().fit(X_r0_legacy[tr_norm_idx])
    X_ln_tr = scaler_leg_n.transform(X_r0_legacy[tr_norm_idx])
    X_ln_te = scaler_leg_n.transform(X_r0_legacy[te_idx])

    scale_concat = float(1.0 / np.sqrt(2.0))
    all_r3_concat = np.hstack([scale_concat * all_log, scale_concat * all_patch]).astype(np.float32)

    corpus_logs, corpus_patches = load_raw_text_corpus(ids)
    tr_logs = [corpus_logs[i] for i in tr_idx]
    tr_patches = [corpus_patches[i] for i in tr_idx]
    te_logs = [corpus_logs[i] for i in te_idx]
    te_patches = [corpus_patches[i] for i in te_idx]

    def bench(fit_fn, score_fn, repeats: int = 20) -> Tuple[float, float]:
        t0 = time.perf_counter()
        fit_fn()
        train_s = time.perf_counter() - t0
        # Warmup
        score_fn()
        times = []
        for _ in range(repeats):
            t1 = time.perf_counter()
            score_fn()
            times.append((time.perf_counter() - t1) * 1000.0 / n_te)
        return float(train_s), float(np.median(times))

    heur_det = RuleHeuristicDetector()
    te_df = raw_df.iloc[te_idx]
    t_heur = bench(lambda: None, lambda: heur_det.decision_function(te_df))

    knn_det = KnnNormalDistanceDetector(k=3, metric="dual")
    t_knn = bench(
        lambda: knn_det.fit_embeddings(all_log[tr_norm_idx], all_patch[tr_norm_idx], all_empty[tr_norm_idx]),
        lambda: knn_det.score_dual(all_log[te_idx], all_patch[te_idx], all_empty[te_idx]),
    )

    t3_det = SupervisedLogisticDetector(C=1.0)
    t_t3 = bench(
        lambda: t3_det.fit(all_r3_concat[tr_idx], y_all[tr_idx]),
        lambda: t3_det.decision_function(all_r3_concat[te_idx]),
    )

    # For R1 TF-IDF, include vectorizer fit/transform
    state_r1 = {}
    def fit_r1_all():
        vl, vp = fit_r1_tfidf(tr_logs, tr_patches)
        _, _, Xtr = transform_r1_tfidf(vl, vp, tr_logs, tr_patches)
        clf = SupervisedLogisticDetector(C=1.0).fit(Xtr, y_all[tr_idx])
        state_r1["vl"], state_r1["vp"], state_r1["clf"] = vl, vp, clf

    def score_r1_all():
        _, _, Xte = transform_r1_tfidf(state_r1["vl"], state_r1["vp"], te_logs, te_patches)
        state_r1["clf"].decision_function(Xte)

    t_r1 = bench(fit_r1_all, score_r1_all)

    ecod_det = InductiveECODVariant()
    t_ecod = bench(
        lambda: ecod_det.fit(X_c_norm),
        lambda: ecod_det.decision_function(X_c_te),
    )

    if_det = IsolationForestDetector(n_estimators=200, random_state=42)
    t_if = bench(
        lambda: if_det.fit(X_c_norm),
        lambda: if_det.decision_function(X_c_te),
    )

    cond_det = ConditionalResidualDetector(ridge_alpha=10.0, logistic_c=0.1)
    t_cond = bench(
        lambda: cond_det.fit(X_c_norm),
        lambda: cond_det.decision_function(X_c_te),
    )

    dae_det = ShallowDAEDetector(epochs=150, batch_size=16, random_state=42)
    t_dae = bench(
        lambda: dae_det.fit(X_c_norm),
        lambda: dae_det.decision_function(X_c_te),
    )

    ae_det = LegacyAEDetector(epochs=100, lr=0.001, random_state=42)
    t_ae = bench(
        lambda: ae_det.fit(X_ln_tr),
        lambda: ae_det.decision_function(X_ln_te),
    )

    xgb_det = LegacyXGBoostDetector(random_state=42)
    t_xgb = bench(
        lambda: xgb_det.fit(X_l_tr, y_all[tr_idx]),
        lambda: xgb_det.decision_function(X_l_te),
    )

    latency_map = {
        "always_alert": {"train_s": 0.0, "infer_ms": 0.0, "embed_ms": 0.0},
        "never_alert": {"train_s": 0.0, "infer_ms": 0.0, "embed_ms": 0.0},
        "heuristic_T1": {"train_s": t_heur[0], "infer_ms": t_heur[1], "embed_ms": 0.0},
        "knn_R3_T2": {"train_s": t_knn[0], "infer_ms": t_knn[1], "embed_ms": 3150.7},
        "supervised_R3_logistic_T3": {"train_s": t_t3[0], "infer_ms": t_t3[1], "embed_ms": 3150.7},
        "supervised_R1_logistic": {"train_s": t_r1[0], "infer_ms": t_r1[1], "embed_ms": 0.0},
        "inductive_ecod_R0_clean": {"train_s": t_ecod[0], "infer_ms": t_ecod[1], "embed_ms": 0.0},
        "isolation_forest_R0_clean": {"train_s": t_if[0], "infer_ms": t_if[1], "embed_ms": 0.0},
        "conditional_residual_R0_clean": {"train_s": t_cond[0], "infer_ms": t_cond[1], "embed_ms": 0.0},
        "shallow_dae_R0_clean": {"train_s": t_dae[0], "infer_ms": t_dae[1], "embed_ms": 0.0},
        "legacy_ae_R0": {"train_s": t_ae[0], "infer_ms": t_ae[1], "embed_ms": 0.0},
        "legacy_xgboost_R0": {"train_s": t_xgb[0], "infer_ms": t_xgb[1], "embed_ms": 0.0},
    }
    Path("Model/tables/v2/method_latencies.json").write_text(json.dumps(latency_map, indent=2) + "\n")

    # =========================================================================
    # 4. CASE STUDY ENRICHMENT & ECOD PER-FEATURE ATTRIBUTION
    # =========================================================================
    cases = json.loads(Path("Model/tables/v2/case_studies.json").read_text())
    manifest_df = pd.read_json("Dataset/v2/text_manifest.jsonl", lines=True).set_index("instance_id")
    enriched_cases = {}
    for k_case, c_info in cases.items():
        iid = c_info["instance_id"]
        f_idx = int(manifest_df.loc[iid, "outer_fold"])
        f_norm_ids = folds[f_idx]["train_normal_instance_ids"]
        repo = str(raw_indexed.loc[iid, "repo"])
        same_repo_normals = [x for x in f_norm_ids if str(raw_indexed.loc[x, "repo"]) == repo]
        r_row = raw_indexed.loc[iid]
        enriched_cases[k_case] = {
            **c_info,
            "outer_fold": f_idx,
            "repo": repo,
            "fold_normal_ref_count": len(f_norm_ids),
            "same_repo_normal_ref_count": len(same_repo_normals),
            "same_repo_normal_refs": same_repo_normals,
            "num_error_observations": int(r_row["num_error_observations"]),
            "error_observation_rate": float(r_row["error_observation_rate"]),
            "tokens_received": int(r_row["tokens_received"]),
            "total_cost_usd": float(r_row["total_cost_usd"]),
            "stalled_turn_count": int(r_row["stalled_turn_count"]),
            "reproduce_script_created": int(r_row["reproduce_script_created"]),
            "patch_char_length": int(r_row["patch_char_length"]),
            "patch_lines_added": int(r_row["patch_lines_added"]),
            "patch_lines_removed": int(r_row["patch_lines_removed"]),
            "patch_touches_tests": int(r_row["patch_touches_tests"]),
        }

    # Also compute per-feature ECOD attribution on a flagged true-positive failure (e.g. django__django-11630 or highest ECOD TP)
    ecod_scores = oof["inductive_ecod_R0_clean_seed_42_score"].to_numpy()
    ecod_preds = oof["inductive_ecod_R0_clean_seed_42_pred"].to_numpy()
    tp_flag_id = "django__django-11630"
    tp_idx = id_to_idx[tp_flag_id]
    tp_fold = int(manifest_df.loc[tp_flag_id, "outer_fold"])
    tp_ri = [id_to_idx[x] for x in folds[tp_fold]["train_normal_instance_ids"]]
    scaler_tp = NormalOnlyRobustScaler(clean_col_names).fit(X_r0_clean_unscaled[tp_ri])
    X_tp_norm = scaler_tp.transform(X_r0_clean_unscaled[tp_ri])
    x_tp_vec = scaler_tp.transform(X_r0_clean_unscaled[[tp_idx]])[0]
    ecod_tp = InductiveECODVariant().fit(X_tp_norm)
    col_contribs = []
    n_ref_tp = X_tp_norm.shape[0]
    for j, cname in enumerate(clean_col_names):
        tr_col = X_tp_norm[:, j]
        val_j = x_tp_vec[j]
        sk_j = ecod_tp.skewness[j]
        pL = (1.0 + np.sum(tr_col <= val_j)) / (n_ref_tp + 1.0)
        pR = (1.0 + np.sum(tr_col >= val_j)) / (n_ref_tp + 1.0)
        uL, uR = -np.log(pL), -np.log(pR)
        sel = uR if sk_j > 1e-6 else (uL if sk_j < -1e-6 else uL + uR)
        c_score = float(max(uL, uR, sel))
        col_contribs.append({"feature": cname, "score": c_score, "raw_value": float(raw_indexed.loc[tp_flag_id, cname])})
    tot_c = sum(x["score"] for x in col_contribs)
    for x in col_contribs:
        x["pct"] = 100.0 * x["score"] / tot_c
    col_contribs.sort(key=lambda x: x["score"], reverse=True)

    diagnostics = {
        "channel_variance": {
            "d_log_mean": float(arr_log.mean()),
            "d_log_sd": float(arr_log.std(ddof=1)),
            "d_patch_mean": float(arr_patch.mean()),
            "d_patch_sd": float(arr_patch.std(ddof=1)),
            "d_dual_mean": float(arr_dual.mean()),
            "d_dual_sd": float(arr_dual.std(ddof=1)),
            "d_dual_fail_mean": float(arr_dual[y_all == 1].mean()),
            "d_dual_fail_sd": float(arr_dual[y_all == 1].std(ddof=1)),
            "d_dual_norm_mean": float(arr_dual[y_all == 0].mean()),
            "d_dual_norm_sd": float(arr_dual[y_all == 0].std(ddof=1)),
            "var_half_log": var_log,
            "var_half_patch": var_patch,
            "cov_half_log_patch": cov_lp,
            "var_dual": var_dual,
            "log_variance_share_pct": float(100.0 * var_log / (var_log + var_patch)),
        },
        "contamination_mechanism": contam_mech,
        "ecod_contamination_summary": ecod_contam_df.groupby("p_contamination")[["balanced_accuracy", "fpr", "failure_recall"]].mean().to_dict(),
        "enriched_cases": enriched_cases,
        "ecod_flagged_attribution": {
            "instance_id": tp_flag_id,
            "outer_fold": tp_fold,
            "total_score": float(tot_c),
            "top_features": col_contribs[:8],
        },
    }
    Path("Model/tables/v2/mechanistic_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("Saved mechanistic_diagnostics.json, ablation_thresholds.csv, and method_latencies.json.")


if __name__ == "__main__":
    main()
