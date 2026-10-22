"""Generate publication-grade figures for HW2 v2.

Figures:
1. fig_confusion_matrices.png / .pdf:
   Side-by-side raw integer counts and row-normalized heatmaps for key models:
   - T2 Primary: kNN R3
   - T3 Supervised Reference: Logistic R3
   - Tabular Control: Inductive ECOD R0-clean
   - Tabular Control: Isolation Forest R0-clean
   - Legacy Control: R0 AE
   - Legacy Control: R0 XGBoost
2. fig_roc_pr_curves.png / .pdf:
   - Panel A: Receiver Operating Characteristic (ROC) with diagonal baseline
   - Panel B: Precision-Recall (PR) with prevalence horizontal baseline (0.810)
3. fig_tradeoff_curves.png / .pdf:
   - Empirical FPR vs Failure Recall tradeoff curves across decision thresholds
4. fig_risk_coverage.png / .pdf:
   - Unflagged success fraction vs Unflagged coverage
5. fig_ablations.png / .pdf:
   - Representation ablation, Channel ablation, and Contamination curves
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


OUTPUTS_DIR = Path("Model/outputs/v2")
TABLES_DIR = Path("Model/tables/v2")
FIGURES_DIR = Path("Model/figures/v2")
MANIFEST_PATH = Path("Dataset/v2/text_manifest.jsonl")


# Global plotting aesthetics matching paper_style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
})


def plot_confusion_matrices() -> None:
    print("Plotting confusion matrices...")
    oof_df = pd.read_csv(OUTPUTS_DIR / "oof_predictions.csv")
    y_true = oof_df["true_label"].to_numpy()

    models_to_plot = [
        ("knn_R3_T2_seed_42_pred", "T2 Primary: kNN (R3 Dual)"),
        ("supervised_R3_logistic_T3_seed_42_pred", "T3 Reference: Logistic (R3)"),
        ("inductive_ecod_R0_clean_seed_42_pred", "Control: Inductive ECOD (R0-clean)"),
        ("isolation_forest_R0_clean_seed_42_pred", "Control: Isolation Forest (R0-clean)"),
        ("legacy_ae_R0_seed_42_pred", "Legacy Control: R0 AE"),
        ("legacy_xgboost_R0_seed_42_pred", "Legacy Control: R0 XGBoost"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (col_name, title) in enumerate(models_to_plot):
        ax = axes[idx]
        y_pred = oof_df[col_name].to_numpy()

        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        cm_counts = np.array([[tn, fp], [fn, tp]])
        cm_norm = np.array([
            [tn / (tn + fp) if (tn + fp) > 0 else 0, fp / (tn + fp) if (tn + fp) > 0 else 0],
            [fn / (fn + tp) if (fn + tp) > 0 else 0, tp / (fn + tp) if (fn + tp) > 0 else 0],
        ])

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        # Annotate text
        for r in range(2):
            for c in range(2):
                cnt = cm_counts[r, c]
                pct = cm_norm[r, c] * 100
                txt = f"{cnt}\n({pct:.1f}%)"
                color = "white" if cm_norm[r, c] > 0.5 else "black"
                ax.text(c, r, txt, ha="center", va="center", color=color, fontweight="bold", fontsize=11)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal (0)", "Pred Anomaly (1)"])
        ax.set_yticklabels(["True Success (0)", "True Failure (1)"])
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_confusion_matrices.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig_confusion_matrices.pdf")
    plt.close(fig)


def plot_roc_and_pr_curves() -> None:
    print("Plotting ROC and PR curves...")
    oof_df = pd.read_csv(OUTPUTS_DIR / "oof_predictions.csv")
    y_true = oof_df["true_label"].to_numpy()
    prevalence = float(np.mean(y_true))

    models = [
        ("knn_R3_T2_seed_42_score", "T2 kNN R3 (Dual)", "#1f77b4", "-"),
        ("supervised_R3_logistic_T3_seed_42_score", "T3 Logistic R3", "#ff7f0e", "--"),
        ("supervised_R1_logistic_seed_42_score", "Control: R1 TF-IDF Logistic", "#2ca02c", ":"),
        ("inductive_ecod_R0_clean_seed_42_score", "Control: Inductive ECOD", "#d62728", "-."),
        ("isolation_forest_R0_clean_seed_42_score", "Control: Isolation Forest", "#9467bd", "--"),
        ("legacy_ae_R0_seed_42_score", "Legacy: R0 AE", "#8c564b", ":"),
        ("legacy_xgboost_R0_seed_42_score", "Legacy: R0 XGBoost", "#e377c2", "-"),
    ]

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: ROC
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance (AUC = 0.500)")
    for col, label, color, ls in models:
        scores = oof_df[col].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax_roc.plot(fpr, tpr, label=f"{label} ({auc:.3f})", color=color, linestyle=ls, linewidth=2)

    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("(a) Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # Panel B: PR
    ax_pr.axhline(prevalence, color="k", linestyle="--", alpha=0.6, label=f"Prevalence Baseline ({prevalence:.3f})")
    for col, label, color, ls in models:
        scores = oof_df[col].to_numpy()
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax_pr.plot(rec, prec, label=f"{label} (AP = {ap:.3f})", color=color, linestyle=ls, linewidth=2)

    ax_pr.set_xlabel("Recall (Failure Detection Rate)")
    ax_pr.set_ylabel("Precision (Failure Proportion in Alarmed)")
    ax_pr.set_title("(b) Precision-Recall (PR) Curves", fontweight="bold")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_roc_pr_curves.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig_roc_pr_curves.pdf")
    plt.close(fig)


def plot_tradeoff_curves() -> None:
    print("Plotting tradeoff curves...")
    oof_df = pd.read_csv(OUTPUTS_DIR / "oof_predictions.csv")
    y_true = oof_df["true_label"].to_numpy()

    models = [
        ("knn_R3_T2_seed_42_score", "T2 kNN R3 (Dual)", "#1f77b4"),
        ("supervised_R3_logistic_T3_seed_42_score", "T3 Logistic R3", "#ff7f0e"),
        ("inductive_ecod_R0_clean_seed_42_score", "Control: Inductive ECOD", "#d62728"),
        ("isolation_forest_R0_clean_seed_42_score", "Control: Isolation Forest", "#9467bd"),
        ("legacy_ae_R0_seed_42_score", "Legacy: R0 AE", "#8c564b"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(0.20, color="gray", linestyle=":", label="Validation FPR Budget (<=20%)")
    ax.axhline(0.90, color="green", linestyle=":", label="Practical Recall Gate (>=90%)")

    for col, label, color in models:
        scores = oof_df[col].to_numpy()
        thresholds = np.unique(scores)
        fpr_list, rec_list = [], []
        for t in thresholds:
            pred = (scores >= t).astype(int)
            tn = np.sum((y_true == 0) & (pred == 0))
            fp = np.sum((y_true == 0) & (pred == 1))
            fn = np.sum((y_true == 1) & (pred == 0))
            tp = np.sum((y_true == 1) & (pred == 1))
            fpr_list.append(fp / (fp + tn))
            rec_list.append(tp / (tp + fn))

        ax.plot(fpr_list, rec_list, label=label, color=color, linewidth=2)

    ax.set_xlabel("Empirical False Positive Rate (FPR)")
    ax.set_ylabel("Empirical Failure Recall")
    ax.set_title("Failure Recall vs. False Positive Rate Tradeoff", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_tradeoff_curves.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig_tradeoff_curves.pdf")
    plt.close(fig)


def plot_ablations() -> None:
    print("Plotting ablation figures...")
    rep_df = pd.read_csv(TABLES_DIR / "ablation_representations.csv")
    chan_df = pd.read_csv(TABLES_DIR / "ablation_channels.csv")
    contam_df = pd.read_csv(TABLES_DIR / "ablation_contamination.csv")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Representations
    reps = rep_df["representation"].tolist()
    bal_accs = rep_df["balanced_accuracy"].tolist()
    fprs = rep_df["fpr"].tolist()
    recalls = rep_df["failure_recall"].tolist()

    x = np.arange(len(reps))
    w = 0.25
    ax1.bar(x - w, bal_accs, width=w, label="Bal Acc", color="#1f77b4")
    ax1.bar(x, fprs, width=w, label="FPR", color="#d62728")
    ax1.bar(x + w, recalls, width=w, label="Recall", color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["R0 Euclidean", "R1 TF-IDF", "R2 MiniLM", "R3 Dual"], rotation=20, ha="right")
    ax1.set_title("(a) Representation Ablation", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Channels
    chans = chan_df["channel"].tolist()
    c_bal = chan_df["balanced_accuracy"].tolist()
    c_fpr = chan_df["fpr"].tolist()
    c_rec = chan_df["failure_recall"].tolist()

    x_c = np.arange(len(chans))
    ax2.bar(x_c - w, c_bal, width=w, label="Bal Acc", color="#1f77b4")
    ax2.bar(x_c, c_fpr, width=w, label="FPR", color="#d62728")
    ax2.bar(x_c + w, c_rec, width=w, label="Recall", color="#2ca02c")
    ax2.set_xticks(x_c)
    ax2.set_xticklabels(["Log Only", "Patch Only", "Dual (R3)"], rotation=15)
    ax2.set_title("(b) Channel Contribution", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Contamination
    # Group by p_contamination
    contam_group = contam_df.groupby("p_contamination")[["balanced_accuracy", "fpr", "failure_recall"]].mean().reset_index()
    p_vals = contam_group["p_contamination"] * 100
    ax3.plot(p_vals, contam_group["balanced_accuracy"], "o-", label="Balanced Accuracy", color="#1f77b4", linewidth=2)
    ax3.plot(p_vals, contam_group["fpr"], "s--", label="FPR", color="#d62728", linewidth=2)
    ax3.plot(p_vals, contam_group["failure_recall"], "^-.", label="Recall", color="#2ca02c", linewidth=2)
    ax3.set_xlabel("Contamination Fraction p (%)")
    ax3.set_ylabel("Metric Value")
    ax3.set_title("(c) Reference Contamination Robustness", fontweight="bold")
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="center right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ablations.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig_ablations.pdf")
    plt.close(fig)


def generate_all_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices()
    plot_roc_and_pr_curves()
    plot_tradeoff_curves()
    plot_ablations()
    print(f"All publication figures generated in {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
