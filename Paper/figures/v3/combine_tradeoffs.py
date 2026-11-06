"""Combine Contamination Sensitivity and Cost Triage into a single publication-ready 2-panel figure."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

RESULTS_PATH = Path("Model/ablation_results_v3.json")
OUTPUT_FIG = Path("Paper/figures/v3/tradeoffs.pdf")


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    contam = data["contamination_study"]
    triage = data["triage_economics"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    # Panel A: Contamination
    c_vals = [float(c) * 100 for c in contam.keys()]
    pr_means = [contam[c]["pr_auc_mean"] for c in contam.keys()]
    pr_stds = [contam[c]["pr_auc_std"] for c in contam.keys()]
    f2_means = [contam[c]["f2_mean"] for c in contam.keys()]
    f2_stds = [contam[c]["f2_std"] for c in contam.keys()]

    ax1 = axes[0]
    color1 = "#d95f02"
    ax1.errorbar(c_vals, pr_means, yerr=pr_stds, fmt="o-", color=color1, label="PR-AUC", linewidth=1.6, capsize=3)
    ax1.set_xlabel("Contamination Rate (%)")
    ax1.set_ylabel("PR-AUC", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax1_twin = ax1.twinx()
    color2 = "#2b5c8f"
    ax1_twin.errorbar(c_vals, f2_means, yerr=f2_stds, fmt="s--", color=color2, label="Cost-Opt F2", linewidth=1.6, capsize=3)
    ax1_twin.set_ylabel("Cost-Optimal F2", color=color2)
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("(a) Robustness to Contamination")

    # Panel B: Cost Triage
    ax2 = axes[1]
    budgets = triage["budgets_pct"]
    sav_bae = triage["savings_cm_bae"]
    sav_churn = triage["savings_churn"]
    sav_rand = triage["savings_random"]

    ax2.plot(budgets, sav_bae, "o-", color="#d95f02", label="CM-BAE", linewidth=1.8, markersize=3.5)
    ax2.plot(budgets, sav_churn, "s--", color="#7570b3", label="3-Sigma Churn", linewidth=1.4, markersize=3.0)
    ax2.plot(budgets, sav_rand, ":", color="gray", label="Random", linewidth=1.2)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.5)

    ax2.set_xlabel("Review Budget (% candidates)")
    ax2.set_ylabel("Net Savings ($ USD)")
    ax2.set_title("(b) CI/CD Net Triage Savings")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Saved combined tradeoffs figure to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
