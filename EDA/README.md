# Exploratory Feature Analysis (EDA)

This directory implements the exploratory feature analysis for the **AI Agent Ghostwork Detector**, corresponding to §4 of the DM-HW2 specification.

## 1. Objectives & Protocol Hygiene
- **Strict Leakage Prevention Protocol:** All exploratory statistics, distribution plots, and correlation matrices are computed **strictly on the development portion** (`train.csv` + `val.csv`, $N=122$). The held-out test split (`test.csv`, $N=162$) remains completely unseen until §7 evaluation.
- **Goal:** Quantify the separability of healthy, productive issue-solving workflows (Normal) versus anomalous task failures and false completion claims (Ghostwork), and formulate rigorous hypotheses to test during modeling.

---

## 2. Directory Structure

```
EDA/
├── README.md                          # Comprehensive EDA report and hypotheses documentation
├── eda.py                             # Reproducible script producing all figures and tables
├── figures/
│   ├── fig1_distribution_contrast     # 4-panel distribution contrast (.pdf, .png)
│   ├── fig2_correlation_heatmap       # Spearman candidate feature correlation matrix (.pdf, .png)
│   └── fig3_latent_manifold           # 2D PCA latent representation projection (.pdf, .png)
└── tables/
    ├── feature_separability.csv       # KS distance, Wasserstein distance, Mutual Information ranking
    └── feature_correlation_matrix.csv  # Full pairwise Spearman correlation matrix
```

---

## 3. Key Empirical Findings

### A. Dominance of Trajectory Interaction Dynamics over Static Patches
As shown in `tables/feature_separability.csv`, interaction-level process features demonstrate substantially higher statistical separability than static patch size metrics:
- `tokens_received`: Standardized Wasserstein distance = $0.910$, KS statistic = $0.440$ ($p = 2.4 \times 10^{-5}$).
- `total_cost_usd`: Wasserstein distance = $0.828$, KS statistic = $0.383$ ($p = 3.8 \times 10^{-4}$).
- `num_steps` & `num_tool_calls`: Wasserstein distance = $0.810$, KS statistic = $0.371$ ($p = 6.6 \times 10^{-4}$).

Successful runs resolve bugs with concise, high-efficiency tool trajectories (mean $21.4$ steps, $\$2.41$ cost). Ghostwork and task failures, by contrast, exhibit prolonged exploration thrashing (mean $37.8$ steps, $\$4.52$ cost), repeatedly re-reading files and re-running failed commands.

### B. Manifold Geometry & Compactness
In `figures/fig3_latent_manifold.pdf`, 2D PCA projection reveals that Normal trajectories form a tight, coherent cluster around the origin in latent space. Anomalies, conversely, disperse outward into high-variance error subspaces, validating the fundamental premise of **one-class reconstruction-based anomaly detection** (Tier 2 Autoencoder).

---

## 4. Formal Exploratory Hypotheses

Based strictly on the development distribution contrast, we establish three explicit hypotheses to be rigorously evaluated in §7 (Results) and §8 (Analysis):

- **Hypothesis 1 ($H_1$: Workflow Process Superiority):**
  *Detectors utilizing full trajectory dynamics (tool streaks, error cascades, token burn rate) will achieve statistically significant gains in PR-AUC and $F_2$ score over detectors restricted to final patch diff features alone.*

- **Hypothesis 2 ($H_2$: Training Contamination Brittleness):**
  *In semi-supervised Autoencoder reconstruction, injecting unlabeled task failures into the normal training pool beyond $5\%$ contamination will distort the healthy manifold, causing a measurable degradation in anomaly reconstruction error gap and test PR-AUC.*

- **Hypothesis 3 ($H_3$: Cost-Asymmetric Threshold Calibration):**
  *Calibrating the anomaly decision threshold $\tau$ on validation data via cost-sensitive $F_2$ optimization (valuing failure recall over precision) will suppress silent ghostwork code integrations while maintaining bounded human audit costs.*
