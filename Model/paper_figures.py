"""Generate publication-ready figures for Model sections (§5, §7, §8).

Figures:
- fig0_pipeline: 6-checkpoint End-to-End Pipeline for Tier 2 Primary Method
- fig4_pr_roc_curves: Precision-Recall and ROC curves for Tier 1, Tier 2, Tier 3
- fig5_ablation_curves: Contamination curve and Feature set ablation comparison
- fig6_feature_attribution: Feature reconstruction error attribution for flagged sample
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

import paper_style


FIGURES_DIR = Path("Model/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_pipeline_diagram() -> None:
    """Draw the required 6-checkpoint End-to-End Pipeline Diagram for Tier 2 Primary Method."""
    paper_style.apply()

    fig, ax = plt.subplots(figsize=(6.8, 2.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    stages = [
        {"num": "STAGE 1", "title": "Input Tensor", "sub": "Raw Trajectory\nx in R^24\nProcess + Patch"},
        {"num": "STAGE 2", "title": "Train Scaling", "sub": "Fit Scaler on\nNormal-Only Train\nTrain statistics only"},
        {"num": "STAGE 3", "title": "Core Model", "sub": "Bottleneck Autoencoder\n24 -> 16 -> 6 -> 16 -> 24\nLayerNorm + ReLU"},
        {"num": "STAGE 4", "title": "Scoring s(x)", "sub": "Standardized Residuals\nMean Squared Error\nDimensionless Score"},
        {"num": "STAGE 5", "title": "Threshold tau", "sub": "Val-Tuned Cutoff\nCost-Sensitive F2\nvalidation F2 cutoff"},
        {"num": "STAGE 6", "title": "Operational Alert", "sub": "Binary Decision\ny_hat = 1[s(x) >= tau]\nHuman Trace Review"},
    ]

    box_w = 13.8
    box_h = 24.0
    start_x = 2.0
    gap = 2.6
    y_pos = 6.0

    for i, st in enumerate(stages):
        x = start_x + i * (box_w + gap)
        # Background box
        is_hero = i in (2, 3)
        edge_col = paper_style.BLUE_MAIN if is_hero else paper_style.INK2
        fill_col = "#EDF3F9" if is_hero else "#F8F9FA"

        rect = patches.FancyBboxPatch(
            (x, y_pos),
            box_w,
            box_h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            linewidth=1.0 if is_hero else 0.8,
            edgecolor=edge_col,
            facecolor=fill_col,
        )
        ax.add_patch(rect)

        # Stage Number header
        ax.text(
            x + box_w / 2,
            y_pos + box_h - 2.8,
            st["num"],
            ha="center",
            va="center",
            fontsize=6.5,
            weight="bold",
            color=paper_style.BLUE_MAIN if is_hero else paper_style.REF_GRAY,
        )
        # Title
        ax.text(
            x + box_w / 2,
            y_pos + box_h - 7.2,
            st["title"],
            ha="center",
            va="center",
            fontsize=7.5,
            weight="bold",
            color=paper_style.INK,
        )
        # Horizontal dividing line inside box
        ax.plot(
            [x + 1.2, x + box_w - 1.2],
            [y_pos + box_h - 9.8, y_pos + box_h - 9.8],
            color=paper_style.LIGHT_GRAY,
            lw=0.6,
        )
        # Subtext details
        ax.text(
            x + box_w / 2,
            y_pos + 6.5,
            st["sub"],
            ha="center",
            va="center",
            fontsize=6.0,
            color=paper_style.INK2,
            linespacing=1.25,
        )

        # Arrow to next stage
        if i < len(stages) - 1:
            arr_x = x + box_w + 0.3
            arr_y = y_pos + box_h / 2
            ax.annotate(
                "",
                xy=(arr_x + gap - 0.6, arr_y),
                xytext=(arr_x, arr_y),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.25,head_length=0.35",
                    lw=1.0,
                    color=paper_style.BLUE_MAIN if i == 2 else paper_style.REF_GRAY,
                ),
            )

    plt.tight_layout()
    paper_style.save(fig, FIGURES_DIR / "fig0_pipeline")
    print(f"Saved pipeline figure to {FIGURES_DIR / 'fig0_pipeline'}")


def plot_pr_roc_curves() -> None:
    """Plot Precision-Recall and ROC curves for Tier 1, Tier 2, Tier 3."""
    paper_style.apply()

    preds_df = pd.read_csv("Model/outputs/test_predictions.csv")
    y_true = preds_df["label"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9), dpi=300)

    # 1. Precision-Recall Curve
    p1, r1, _ = precision_recall_curve(y_true, preds_df["T1_score"].values)
    p2, r2, _ = precision_recall_curve(y_true, preds_df["T2_score"].values)
    p3, r3, _ = precision_recall_curve(y_true, preds_df["T3_score"].values)

    ax1.plot(r1, p1, color=paper_style.REF_GRAY, lw=1.2, linestyle="--", label=f"T1 AP ({average_precision_score(y_true, preds_df['T1_score']):.3f})")
    ax1.plot(r2, p2, color=paper_style.BLUE_MAIN, lw=1.8, label=f"T2 AP ({average_precision_score(y_true, preds_df['T2_score']):.3f})")
    ax1.plot(r3, p3, color=paper_style.RED, lw=1.2, linestyle="-.", label=f"T3 AP ({average_precision_score(y_true, preds_df['T3_score']):.3f})")
    ax1.axhline(y_true.mean(), color=paper_style.LIGHT_GRAY, linestyle=":", label=f"Prior {y_true.mean():.3f}")
    ax1.set_xlabel("Recall (caught failures)")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curves", fontsize=8, pad=4)
    ax1.legend(loc="lower left", frameon=False, fontsize=6.5)

    # 2. ROC Curve
    fpr1, tpr1, _ = roc_curve(y_true, preds_df["T1_score"].values)
    fpr2, tpr2, _ = roc_curve(y_true, preds_df["T2_score"].values)
    fpr3, tpr3, _ = roc_curve(y_true, preds_df["T3_score"].values)

    ax2.plot(fpr1, tpr1, color=paper_style.REF_GRAY, lw=1.2, linestyle="--", label=f"T1 ROC ({roc_auc_score(y_true, preds_df['T1_score']):.3f})")
    ax2.plot(fpr2, tpr2, color=paper_style.BLUE_MAIN, lw=1.8, label=f"T2 ROC ({roc_auc_score(y_true, preds_df['T2_score']):.3f})")
    ax2.plot(fpr3, tpr3, color=paper_style.RED, lw=1.2, linestyle="-.", label=f"T3 ROC ({roc_auc_score(y_true, preds_df['T3_score']):.3f})")
    ax2.plot([0, 1], [0, 1], color=paper_style.LIGHT_GRAY, lw=0.8, linestyle=":")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves", fontsize=8, pad=4)
    ax2.legend(loc="lower right", frameon=False, fontsize=6.5)

    plt.tight_layout()
    paper_style.panel_labels_row(fig, [ax1, ax2], ["(a)", "(b)"])
    paper_style.save(fig, FIGURES_DIR / "fig4_pr_roc_curves")
    print(f"Saved PR and ROC curves to {FIGURES_DIR / 'fig4_pr_roc_curves'}")


def plot_ablation_figures() -> None:
    """Plot training contamination degradation curve and feature subset ablation bars."""
    paper_style.apply()

    contam_df = pd.read_csv("Model/tables/ablation_contamination.csv")
    feat_df = pd.read_csv("Model/tables/ablation_features.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8), dpi=300)

    # Panel A: Contamination Curve
    x_ratios = [0, 3.125, 6.25, 9.375]
    ax1.errorbar(
        x_ratios,
        contam_df["PR_AUC_mean"],
        yerr=contam_df["PR_AUC_std"],
        fmt="-o",
        color=paper_style.BLUE_MAIN,
        ecolor=paper_style.BLUE_2,
        elinewidth=1.0,
        capsize=3,
        markersize=4,
        lw=1.5,
    )
    ax1.set_xlabel("Training Contamination Ratio (%)")
    ax1.set_ylabel("Test PR-AUC")
    ax1.set_title("Contamination Sensitivity (Ablation 1)", fontsize=8, pad=4)
    ax1.set_xticks(x_ratios)
    ax1.set_xticklabels(["0%", "3.125%", "6.25%", "9.375%"])

    # Panel B: Feature Subset Bar Chart
    labels = ["Full (D=24)", "Process (D=16)", "Patch (D=8)"]
    means = feat_df["PR_AUC_mean"].values
    stds = feat_df["PR_AUC_std"].values
    bars = ax2.bar(
        labels,
        means,
        yerr=stds,
        capsize=3,
        color=[paper_style.BLUE_MAIN, paper_style.TEAL, paper_style.REF_GRAY],
        width=0.45,
        edgecolor=paper_style.INK2,
        linewidth=0.6,
    )
    ax2.set_ylabel("Test PR-AUC")
    ax2.set_ylim(0.94, 1.0)
    ax2.set_title("Trajectory vs. Patch Output (Ablation 2)", fontsize=8, pad=4)

    plt.tight_layout()
    paper_style.panel_labels_row(fig, [ax1, ax2], ["(a)", "(b)"])
    paper_style.save(fig, FIGURES_DIR / "fig5_ablation_curves")
    print(f"Saved ablation curves to {FIGURES_DIR / 'fig5_ablation_curves'}")


def plot_interpretability() -> None:
    """Plot per-feature reconstruction error attribution for flagged ghostwork sample."""
    paper_style.apply()

    attr_df = pd.read_csv("Model/tables/feature_attribution_flagged_sample.csv").head(6)

    fig, ax = plt.subplots(figsize=(4.8, 2.6), dpi=300)
    y_pos = np.arange(len(attr_df))
    bars = ax.barh(
        y_pos,
        attr_df["Contribution_Percent"],
        color=paper_style.BLUE_MAIN,
        height=0.55,
        edgecolor=paper_style.INK2,
        linewidth=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(attr_df["Feature"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Reconstruction Error Contribution (%)")
    ax.set_title(f"Flagged Sample Attribution\n({attr_df.Instance_ID.iloc[0]})", fontsize=8, pad=4)

    for i, bar in enumerate(bars):
        pct = attr_df.loc[i, "Contribution_Percent"]
        raw = attr_df.loc[i, "Raw_Value"]
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% (raw={raw:.0f})",
            va="center",
            fontsize=6.5,
            color=paper_style.INK,
        )

    ax.set_xlim(0, 95)
    plt.tight_layout()
    paper_style.save(fig, FIGURES_DIR / "fig6_feature_attribution")
    print(f"Saved interpretability attribution to {FIGURES_DIR / 'fig6_feature_attribution'}")


def main() -> None:
    plot_pipeline_diagram()
    plot_pr_roc_curves()
    plot_ablation_figures()
    plot_interpretability()
    print("All Model publication figures generated successfully!")


if __name__ == "__main__":
    main()
