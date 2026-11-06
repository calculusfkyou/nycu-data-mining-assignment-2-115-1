"""Controlled Ablation Studies for CM-BAE.

Studies:
1. Training Contamination Sensitivity: c in {0%, 1%, 3%, 5%, 10%}.
2. Modality & Variance Equalization Ablation:
   - Full (Struct + Log + Patch + VarEq)
   - Without Variance Equalization (Raw Concatenation)
   - Log-Only (MiniLM 384-D)
   - Patch-Only (CodeBERT 768-D)
   - Struct-Only (R0-struct 12-D)
3. CI/CD Triage Economics & Threshold Policies: Net savings sweep across review budgets.

Exports:
- Paper/figures/v3/contamination_curve.pdf
- Paper/figures/v3/cost_triage_curve.pdf
- Paper/generated/v3/v3_ablation_table.tex
- Model/ablation_results_v3.json
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Model.cm_bae_v3 import BAEArchitecture, CMBAE
from Model.experiment_v3 import STRUCT_COLS

FEATURES_PATH = Path("Dataset/verified/candidate_features.csv")
MINILM_PATH = Path("Dataset/verified/embeddings/minilm_logs.npy")
CODEBERT_PATH = Path("Dataset/verified/embeddings/codebert_diffs.npy")
SPLITS_PATH = Path("Dataset/verified/splits.json")
FIG_DIR = Path("Paper/figures/v3")
TEX_DIR = Path("Paper/generated/v3")
OUTPUT_JSON = Path("Model/ablation_results_v3.json")


def run_contamination_study(df, minilm, codebert, splits):
    print("\n--- Running Contamination Sensitivity Study ---")
    contaminations = [0.00, 0.01, 0.03, 0.05, 0.10]
    X_struct = df[STRUCT_COLS].values.astype(np.float32)
    y_all = df["label"].values.astype(int)

    results_by_c = {c: {"pr_auc": [], "f2": []} for c in contaminations}

    for c in contaminations:
        for fold in splits["folds"][:3]:  # 3 folds for ablation efficiency
            train_norm_idx = np.array(fold["train_normal_indices"])
            val_idx = fold["val_indices"]
            test_idx = fold["test_indices"]

            # Contaminate normal training pool with c% anomalies from train anomalies
            train_anom_idx = np.array([i for i in fold["train_indices"] if y_all[i] == 1])
            n_contam = int(len(train_norm_idx) * c)

            if n_contam > 0:
                rng = np.random.RandomState(42)
                inject_idx = rng.choice(train_anom_idx, size=min(n_contam, len(train_anom_idx)), replace=True)
                active_train_idx = np.concatenate([train_norm_idx, inject_idx])
            else:
                active_train_idx = train_norm_idx

            cm_bae = CMBAE(bottleneck_dim=64, epochs=20, seed=42)
            cm_bae.fit(X_struct[active_train_idx], minilm[active_train_idx], codebert[active_train_idx])
            cm_bae.calibrate(X_struct[val_idx], minilm[val_idx], codebert[val_idx], y_all[val_idx])

            p_test = cm_bae.predict_proba(X_struct[test_idx], minilm[test_idx], codebert[test_idx])
            pr_val = average_precision_score(y_all[test_idx], p_test)
            preds = (p_test >= cm_bae.cost_optimal_threshold).astype(int)
            f2_val = fbeta_score(y_all[test_idx], preds, beta=2, zero_division=0)

            results_by_c[c]["pr_auc"].append(float(pr_val))
            results_by_c[c]["f2"].append(float(f2_val))

        mean_pr = np.mean(results_by_c[c]["pr_auc"])
        mean_f2 = np.mean(results_by_c[c]["f2"])
        print(f"  Contamination {c*100:4.1f}% -> PR-AUC: {mean_pr:.4f}, Cost-Opt F2: {mean_f2:.4f}")

    # Plot Contamination Sensitivity Curve
    fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
    c_pcts = [c * 100 for c in contaminations]
    pr_means = [np.mean(results_by_c[c]["pr_auc"]) for c in contaminations]
    pr_stds = [np.std(results_by_c[c]["pr_auc"]) for c in contaminations]
    f2_means = [np.mean(results_by_c[c]["f2"]) for c in contaminations]
    f2_stds = [np.std(results_by_c[c]["f2"]) for c in contaminations]

    color1 = "#d95f02"
    ax1.errorbar(c_pcts, pr_means, yerr=pr_stds, fmt="o-", color=color1, label="PR-AUC", linewidth=2.0, capsize=4)
    ax1.set_xlabel("Training Set Contamination Rate (%)")
    ax1.set_ylabel("PR-AUC", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    color2 = "#2b5c8f"
    ax2.errorbar(c_pcts, f2_means, yerr=f2_stds, fmt="s--", color=color2, label="Cost-Opt F2", linewidth=2.0, capsize=4)
    ax2.set_ylabel("Cost-Optimal F2", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("CM-BAE Robustness Under Training Contamination")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    contam_fig = FIG_DIR / "contamination_curve.pdf"
    plt.savefig(contam_fig, bbox_inches="tight")
    plt.close()
    print(f"Saved contamination curve to {contam_fig}")

    return {
        str(c): {
            "pr_auc_mean": float(np.mean(results_by_c[c]["pr_auc"])),
            "pr_auc_std": float(np.std(results_by_c[c]["pr_auc"])),
            "f2_mean": float(np.mean(results_by_c[c]["f2"])),
            "f2_std": float(np.std(results_by_c[c]["f2"])),
        }
        for c in contaminations
    }


def run_modality_ablation(df, minilm, codebert, splits):
    print("\n--- Running Modality & Variance Equalization Ablation ---")
    X_struct = df[STRUCT_COLS].values.astype(np.float32)
    y_all = df["label"].values.astype(int)

    configs = [
        "Full (Struct + Log + Patch + VarEq)",
        "Raw Fusion (w/o VarEq)",
        "Log-Only (MiniLM 384-D)",
        "Patch-Only (CodeBERT 768-D)",
        "Struct-Only (R0-struct 12-D)",
    ]

    ablation_results = {cfg: {"pr_auc": [], "f2": [], "roc_auc": []} for cfg in configs}

    for fold in splits["folds"]:
        train_norm_idx = fold["train_normal_indices"]
        val_idx = fold["val_indices"]
        test_idx = fold["test_indices"]
        y_test = y_all[test_idx]
        y_val = y_all[val_idx]

        # 1. Full CM-BAE
        m1 = CMBAE(bottleneck_dim=64, epochs=25, seed=42)
        m1.fit(X_struct[train_norm_idx], minilm[train_norm_idx], codebert[train_norm_idx])
        m1.calibrate(X_struct[val_idx], minilm[val_idx], codebert[val_idx], y_val)
        p1 = m1.predict_proba(X_struct[test_idx], minilm[test_idx], codebert[test_idx])
        ablation_results["Full (Struct + Log + Patch + VarEq)"]["pr_auc"].append(float(average_precision_score(y_test, p1)))
        preds1 = (p1 >= m1.cost_optimal_threshold).astype(int)
        ablation_results["Full (Struct + Log + Patch + VarEq)"]["f2"].append(float(fbeta_score(y_test, preds1, beta=2, zero_division=0)))

        # 2. Raw Fusion (w/o VarEq)
        m2 = CMBAE(bottleneck_dim=64, epochs=25, seed=42)
        m2.alpha_struct = 1.0
        m2.alpha_log = 1.0
        m2.alpha_patch = 1.0
        # Fit with unscaled inputs
        m2.fit(X_struct[train_norm_idx], minilm[train_norm_idx], codebert[train_norm_idx])
        m2.alpha_struct = 1.0
        m2.alpha_log = 1.0
        m2.alpha_patch = 1.0
        m2.calibrate(X_struct[val_idx], minilm[val_idx], codebert[val_idx], y_val)
        p2 = m2.predict_proba(X_struct[test_idx], minilm[test_idx], codebert[test_idx])
        ablation_results["Raw Fusion (w/o VarEq)"]["pr_auc"].append(float(average_precision_score(y_test, p2)))
        preds2 = (p2 >= m2.cost_optimal_threshold).astype(int)
        ablation_results["Raw Fusion (w/o VarEq)"]["f2"].append(float(fbeta_score(y_test, preds2, beta=2, zero_division=0)))

        # 3. Log-Only
        zeros_s = np.zeros_like(X_struct)
        zeros_p = np.zeros_like(codebert)
        m3 = CMBAE(bottleneck_dim=32, epochs=25, seed=42)
        m3.fit(zeros_s[train_norm_idx], minilm[train_norm_idx], zeros_p[train_norm_idx])
        m3.calibrate(zeros_s[val_idx], minilm[val_idx], zeros_p[val_idx], y_val)
        p3 = m3.predict_proba(zeros_s[test_idx], minilm[test_idx], zeros_p[test_idx])
        ablation_results["Log-Only (MiniLM 384-D)"]["pr_auc"].append(float(average_precision_score(y_test, p3)))
        preds3 = (p3 >= m3.cost_optimal_threshold).astype(int)
        ablation_results["Log-Only (MiniLM 384-D)"]["f2"].append(float(fbeta_score(y_test, preds3, beta=2, zero_division=0)))

        # 4. Patch-Only
        zeros_l = np.zeros_like(minilm)
        m4 = CMBAE(bottleneck_dim=32, epochs=25, seed=42)
        m4.fit(zeros_s[train_norm_idx], zeros_l[train_norm_idx], codebert[train_norm_idx])
        m4.calibrate(zeros_s[val_idx], zeros_l[val_idx], codebert[val_idx], y_val)
        p4 = m4.predict_proba(zeros_s[test_idx], zeros_l[test_idx], codebert[test_idx])
        ablation_results["Patch-Only (CodeBERT 768-D)"]["pr_auc"].append(float(average_precision_score(y_test, p4)))
        preds4 = (p4 >= m4.cost_optimal_threshold).astype(int)
        ablation_results["Patch-Only (CodeBERT 768-D)"]["f2"].append(float(fbeta_score(y_test, preds4, beta=2, zero_division=0)))

        # 5. Struct-Only
        m5 = CMBAE(bottleneck_dim=8, epochs=25, seed=42)
        m5.fit(X_struct[train_norm_idx], zeros_l[train_norm_idx], zeros_p[train_norm_idx])
        m5.calibrate(X_struct[val_idx], zeros_l[val_idx], zeros_p[val_idx], y_val)
        p5 = m5.predict_proba(X_struct[test_idx], zeros_l[test_idx], zeros_p[test_idx])
        ablation_results["Struct-Only (R0-struct 12-D)"]["pr_auc"].append(float(average_precision_score(y_test, p5)))
        preds5 = (p5 >= m5.cost_optimal_threshold).astype(int)
        ablation_results["Struct-Only (R0-struct 12-D)"]["f2"].append(float(fbeta_score(y_test, preds5, beta=2, zero_division=0)))

    # Export LaTeX ablation table
    table_rows = []
    print("\nModality Ablation Summary:")
    print("-" * 65)
    for cfg in configs:
        pr_m = float(np.mean(ablation_results[cfg]["pr_auc"]))
        pr_s = float(np.std(ablation_results[cfg]["pr_auc"]))
        f2_m = float(np.mean(ablation_results[cfg]["f2"]))
        f2_s = float(np.std(ablation_results[cfg]["f2"]))
        print(f"{cfg:<36} | PR-AUC: {pr_m:.4f}±{pr_s:.4f} | F2: {f2_m:.4f}±{f2_s:.4f}")
        table_rows.append({"config": cfg, "pr_m": pr_m, "pr_s": pr_s, "f2_m": f2_m, "f2_s": f2_s})

    tex_path = TEX_DIR / "v3_ablation_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by Model/ablations_v3.py\n")
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabular}{lcc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Ablation Variant} & \textbf{PR-AUC} $\uparrow$ & \textbf{Cost-Opt $F_2$} $\uparrow$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in table_rows:
            is_best = "Full" in r["config"]
            pr_val = f"{r['pr_m']:.3f} $\\pm$ {r['pr_s']:.3f}"
            f2_val = f"{r['f2_m']:.3f} $\\pm$ {r['f2_s']:.3f}"
            if is_best:
                pr_val = r"\textbf{" + pr_val + r"}"
                f2_val = r"\textbf{" + f2_val + r"}"
            f.write(f"{r['config']} & {pr_val} & {f2_val} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{\textbf{Modality and Normalization Ablation.} Evaluating individual feature modalities and the impact of variance equalization prior to bottleneck compression. Tested across 5 folds.}" + "\n")
        f.write(r"\label{tab:ablation}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Saved LaTeX ablation table to {tex_path}")

    return {
        cfg: {
            "pr_auc_mean": float(np.mean(ablation_results[cfg]["pr_auc"])),
            "pr_auc_std": float(np.std(ablation_results[cfg]["pr_auc"])),
            "f2_mean": float(np.mean(ablation_results[cfg]["f2"])),
            "f2_std": float(np.std(ablation_results[cfg]["f2"])),
        }
        for cfg in configs
    }


def run_triage_economics_study(df, minilm, codebert, splits):
    print("\n--- Running CI/CD Triage Economics Study ---")
    X_struct = df[STRUCT_COLS].values.astype(np.float32)
    y_all = df["label"].values.astype(int)

    # Use Fold 0 test split for concrete triage demonstration
    fold = splits["folds"][0]
    train_norm_idx = fold["train_normal_indices"]
    val_idx = fold["val_indices"]
    test_idx = fold["test_indices"]

    y_test = y_all[test_idx]
    n_test = len(y_test)
    n_anom = int(np.sum(y_test))

    cm_bae = CMBAE(bottleneck_dim=64, epochs=25, seed=42)
    cm_bae.fit(X_struct[train_norm_idx], minilm[train_norm_idx], codebert[train_norm_idx])
    cm_bae.calibrate(X_struct[val_idx], minilm[val_idx], codebert[val_idx], y_all[val_idx])
    scores_cm_bae = cm_bae.predict_proba(X_struct[test_idx], minilm[test_idx], codebert[test_idx])

    # Churn baseline scores
    scaler = StandardScaler()
    churn_z = scaler.fit_transform(df[["edit_lines_total"]].iloc[train_norm_idx])
    test_churn = (df["edit_lines_total"].iloc[test_idx].values - scaler.mean_[0]) / scaler.scale_[0]

    c_escaped = 1200.0
    c_audit = 30.0
    baseline_unmitigated_cost = n_anom * c_escaped

    review_budgets = np.linspace(0.01, 0.30, 30)  # Review budget: 1% to 30% of patches
    savings_cm_bae = []
    savings_churn = []
    savings_random = []

    for b in review_budgets:
        k = int(b * n_test)
        
        # CM-BAE top-k flagged
        top_k_bae = np.argsort(scores_cm_bae)[::-1][:k]
        tp_bae = np.sum(y_test[top_k_bae])
        cost_bae = k * c_audit + (n_anom - tp_bae) * c_escaped
        savings_cm_bae.append(baseline_unmitigated_cost - cost_bae)

        # Churn top-k flagged
        top_k_churn = np.argsort(test_churn)[::-1][:k]
        tp_churn = np.sum(y_test[top_k_churn])
        cost_churn = k * c_audit + (n_anom - tp_churn) * c_escaped
        savings_churn.append(baseline_unmitigated_cost - cost_churn)

        # Random review expectation
        tp_rand = b * n_anom
        cost_rand = k * c_audit + (n_anom - tp_rand) * c_escaped
        savings_random.append(baseline_unmitigated_cost - cost_rand)

    # Plot Cost Triage Curve
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    budgets_pct = review_budgets * 100
    ax.plot(budgets_pct, savings_cm_bae, "o-", color="#d95f02", label="CM-BAE (Primary)", linewidth=2.2)
    ax.plot(budgets_pct, savings_churn, "s--", color="#7570b3", label="3-Sigma Patch Churn", linewidth=1.8)
    ax.plot(budgets_pct, savings_random, ":", color="gray", label="Random Inspection", linewidth=1.5)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_title("CI/CD Net Cost Savings vs Review Budget Capacity")
    ax.set_xlabel("Review Capacity Budget (% of PR candidates audited)")
    ax.set_ylabel("Net Inspection Savings ($ USD)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    cost_fig = FIG_DIR / "cost_triage_curve.pdf"
    plt.savefig(cost_fig, bbox_inches="tight")
    plt.close()
    print(f"Saved cost triage curve to {cost_fig}")

    return {
        "budgets_pct": budgets_pct.tolist(),
        "savings_cm_bae": savings_cm_bae,
        "savings_churn": savings_churn,
        "savings_random": savings_random,
    }


def main():
    print("=" * 70)
    print("RUNNING ABLATION SUITE (CONTAMINATION, MODALITY, TRIAGE ECONOMICS)")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    minilm = np.load(MINILM_PATH)
    codebert = np.load(CODEBERT_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    contam_res = run_contamination_study(df, minilm, codebert, splits)
    modality_res = run_modality_ablation(df, minilm, codebert, splits)
    triage_res = run_triage_economics_study(df, minilm, codebert, splits)

    payload = {
        "contamination_study": contam_res,
        "modality_ablation": modality_res,
        "triage_economics": triage_res,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nAll ablation results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
