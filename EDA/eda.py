"""Exploratory Feature Analysis (EDA) for AI Agent Ghostwork Detector.

Strict Rule: Uses the training and validation portion only so the test set remains unseen!
Produces:
- Tables: feature_separability.csv, correlation_ranking.csv, summary_table.csv
- Figures: fig1_distribution_contrast (pdf/png), fig2_correlation_heatmap (pdf/png), fig3_latent_manifold (pdf/png)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

import paper_style


TABLES_DIR = Path("EDA/tables")
FIGURES_DIR = Path("EDA/figures")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_development_data() -> tuple[pd.DataFrame, list[str]]:
    """Load training and validation sets strictly without touching the test set."""
    train_df = pd.read_csv("Dataset/train.csv")
    val_df = pd.read_csv("Dataset/val.csv")

    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    meta_cols = ["instance_id", "repo", "label", "is_resolved", "exit_status"]
    feature_cols = [c for c in dev_df.columns if c not in meta_cols]
    return dev_df, feature_cols


def compute_separability(dev_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute KS distance, Wasserstein distance, and Mutual Information for each feature."""
    norm_df = dev_df[dev_df["label"] == 0]
    anom_df = dev_df[dev_df["label"] == 1]

    results = []
    # Standardize data for scale-invariant Wasserstein comparison
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dev_df[feature_cols].values)
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    scaled_df["label"] = dev_df["label"].values

    mi_scores = mutual_info_classif(
        dev_df[feature_cols].values, dev_df["label"].values, random_state=42
    )

    for idx, col in enumerate(feature_cols):
        x_norm = norm_df[col].dropna().values
        x_anom = anom_df[col].dropna().values

        # KS Test
        ks_stat, ks_pval = stats.ks_2samp(x_norm, x_anom)

        # Scale-invariant Wasserstein distance on standardized values
        x_norm_std = scaled_df[scaled_df["label"] == 0][col].values
        x_anom_std = scaled_df[scaled_df["label"] == 1][col].values
        w_dist = stats.wasserstein_distance(x_norm_std, x_anom_std)

        # Group type
        feat_type = "Trajectory Process" if idx < 16 else "Static Patch"

        results.append({
            "Feature": col,
            "Type": feat_type,
            "Normal_Mean": float(np.mean(x_norm)),
            "Normal_Std": float(np.std(x_norm)),
            "Anomaly_Mean": float(np.mean(x_anom)),
            "Anomaly_Std": float(np.std(x_anom)),
            "Wasserstein_Std": float(w_dist),
            "KS_Stat": float(ks_stat),
            "KS_Pval": float(ks_pval),
            "Mutual_Info": float(mi_scores[idx]),
        })

    sep_df = pd.DataFrame(results)
    sep_df = sep_df.sort_values(by="Wasserstein_Std", ascending=False).reset_index(drop=True)
    sep_df.to_csv(TABLES_DIR / "feature_separability.csv", index=False)
    return sep_df


def plot_distribution_contrast(dev_df: pd.DataFrame) -> None:
    """Plot 4-panel distribution contrast for top process features and patch characteristics."""
    paper_style.apply()

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), dpi=300)
    axes = axes.flatten()

    norm_mask = dev_df["label"] == 0
    anom_mask = dev_df["label"] == 1

    # Panel A: Error Observation Rate
    ax = axes[0]
    sns.kdeplot(
        dev_df.loc[norm_mask, "error_observation_rate"],
        ax=ax,
        color=paper_style.BLUE_MAIN,
        label="Normal (Resolved)",
        fill=True,
        alpha=0.3,
        linewidth=1.5,
    )
    sns.kdeplot(
        dev_df.loc[anom_mask, "error_observation_rate"],
        ax=ax,
        color=paper_style.RED,
        label="Anomaly (Ghostwork)",
        fill=True,
        alpha=0.3,
        linewidth=1.5,
    )
    ax.set_xlabel("Error Observation Rate (errors / step)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Compounding Error Cascades", fontsize=8, pad=3)

    # Panel B: Max Consecutive Error Streak
    ax = axes[1]
    norm_consec = dev_df.loc[norm_mask, "consecutive_errors_max"]
    anom_consec = dev_df.loc[anom_mask, "consecutive_errors_max"]
    bins = np.arange(0, max(norm_consec.max(), anom_consec.max()) + 2) - 0.5
    ax.hist(
        norm_consec,
        bins=bins,
        density=True,
        alpha=0.6,
        color=paper_style.BLUE_MAIN,
        label="Normal",
        edgecolor="white",
    )
    ax.hist(
        anom_consec,
        bins=bins,
        density=True,
        alpha=0.6,
        color=paper_style.RED,
        label="Anomaly",
        edgecolor="white",
    )
    ax.set_xlabel("Max Consecutive Error Streak (turns)")
    ax.set_ylabel("Relative Frequency")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Repeated Execution Stalls", fontsize=8, pad=3)

    # Panel C: Stalled Turns Count
    ax = axes[2]
    sns.boxplot(
        x="label",
        y="stalled_turn_count",
        hue="label",
        data=dev_df,
        ax=ax,
        palette=[paper_style.BLUE_MAIN, paper_style.RED],
        width=0.4,
        fliersize=2,
        legend=False,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal (Resolved)", "Anomaly (Ghostwork)"])
    ax.set_xlabel("")
    ax.set_ylabel("Stalled / Repetitive Turns")
    ax.set_title("Workflow Thrashing Turns", fontsize=8, pad=3)

    # Panel D: Patch Lines Added vs Removed Scatter
    ax = axes[3]
    ax.scatter(
        dev_df.loc[norm_mask, "patch_lines_added"],
        dev_df.loc[norm_mask, "patch_lines_removed"],
        c=paper_style.BLUE_MAIN,
        label="Normal",
        alpha=0.7,
        s=18,
        edgecolors="none",
    )
    ax.scatter(
        dev_df.loc[anom_mask, "patch_lines_added"],
        dev_df.loc[anom_mask, "patch_lines_removed"],
        c=paper_style.RED,
        label="Anomaly",
        alpha=0.7,
        s=18,
        edgecolors="none",
        marker="^",
    )
    ax.set_xlabel("Patch Lines Added (+)")
    ax.set_ylabel("Patch Lines Removed (-)")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Patch Complexity Profile", fontsize=8, pad=3)

    plt.tight_layout()
    paper_style.panel_labels_row(fig, axes[:2], ["(a)", "(b)"])
    paper_style.panel_labels_row(fig, axes[2:], ["(c)", "(d)"])
    paper_style.save(fig, FIGURES_DIR / "fig1_distribution_contrast")
    print(f"Saved distribution contrast figure to {FIGURES_DIR / 'fig1_distribution_contrast'}")


def plot_correlation_heatmap(dev_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Plot Spearman correlation matrix across candidate features."""
    paper_style.apply()

    corr = dev_df[feature_cols].corr(method="spearman")
    corr.to_csv(TABLES_DIR / "feature_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=300)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr,
        cmap=cmap,
        center=0,
        vmin=-0.8,
        vmax=0.8,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.7, "label": "Spearman Correlation (ρ)"},
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )
    ax.tick_params(axis="both", labelsize=5.5)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Candidate Feature Correlation & Redundancy Matrix", fontsize=8, pad=6)

    plt.tight_layout()
    paper_style.save(fig, FIGURES_DIR / "fig2_correlation_heatmap")
    print(f"Saved correlation heatmap to {FIGURES_DIR / 'fig2_correlation_heatmap'}")


def plot_latent_manifold(dev_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Plot PCA projection showing normal manifold vs anomaly dispersion."""
    paper_style.apply()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(dev_df[feature_cols].values)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_std)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=300)

    norm_mask = dev_df["label"] == 0
    anom_mask = dev_df["label"] == 1

    ax.scatter(
        X_pca[norm_mask, 0],
        X_pca[norm_mask, 1],
        c=paper_style.BLUE_MAIN,
        label=f"Normal (Resolved, N={norm_mask.sum()})",
        alpha=0.8,
        s=24,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.scatter(
        X_pca[anom_mask, 0],
        X_pca[anom_mask, 1],
        c=paper_style.RED,
        label=f"Anomaly (Ghostwork, N={anom_mask.sum()})",
        alpha=0.6,
        s=20,
        marker="^",
        edgecolors="none",
    )

    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% variance)")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% variance)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Trajectory Representation Space (PCA Projection)", fontsize=8, pad=5)

    plt.tight_layout()
    paper_style.save(fig, FIGURES_DIR / "fig3_latent_manifold")
    print(f"Saved latent manifold projection to {FIGURES_DIR / 'fig3_latent_manifold'}")


def main() -> None:
    print("=== Step 1: Loading Development Partition (Train + Val strictly) ===")
    dev_df, feature_cols = load_development_data()
    print(f"Development samples: {len(dev_df)} (Normal: {(dev_df['label'] == 0).sum()}, Anomaly: {(dev_df['label'] == 1).sum()})")

    print("=== Step 2: Computing Statistical Separability Metrics ===")
    sep_df = compute_separability(dev_df, feature_cols)
    print("\nTop 5 Separating Features:")
    print(sep_df[["Feature", "Type", "Wasserstein_Std", "KS_Stat", "Mutual_Info"]].head())

    print("\n=== Step 3: Generating Exploratory Visualizations ===")
    plot_distribution_contrast(dev_df)
    plot_correlation_heatmap(dev_df, feature_cols)
    plot_latent_manifold(dev_df, feature_cols)
    print("\nEDA processing complete!")


if __name__ == "__main__":
    main()
