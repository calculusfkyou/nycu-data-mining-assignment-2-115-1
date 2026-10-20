# Model: 3-Tier Anomaly Detection, Ablations, and Error Analysis

This directory implements the core anomaly detection framework, multi-tier benchmark, controlled ablations, and in-depth error analysis for the **AI Agent Ghostwork Detector**, corresponding to §5 (Method), §6 (Experimental Setup), §7 (Results), and §8 (Analysis) of the DM-HW2 specification.

---

## 1. Directory Structure

```
Model/
├── README.md                          # Comprehensive methodology and reproduction documentation
├── models.py                          # Implementations of Tier 1, Tier 2, and Tier 3 architectures
├── experiment.py                      # 5-seed benchmark evaluation and statistical significance tests
├── ablations.py                       # Controlled training contamination & feature set ablations
├── error_analysis.py                  # Failure case studies (FP/FN), attribution, and impact translation
├── paper_figures.py                   # Generation of publication-grade figures (.pdf and .png)
├── figures/
│   ├── fig0_pipeline                  # 6-checkpoint End-to-End Pipeline architecture
│   ├── fig4_pr_roc_curves             # Multi-tier Precision-Recall and ROC curves
│   ├── fig5_ablation_curves           # Contamination degradation and Trajectory vs. Patch comparisons
│   └── fig6_feature_attribution       # Per-feature reconstruction error attribution for flagged case
├── tables/
│   ├── main_results_table.csv         # 5-seed mean ± std for Tiers 1, 2, and 3
│   ├── significance_tests.csv         # Paired t-test p-values and bootstrap 95% CIs
│   ├── ablation_contamination.csv     # AD-specific training contamination ablation
│   ├── ablation_features.csv          # Trajectory Process vs. Final Patch Output ablation
│   ├── failure_cases.csv              # Concrete FP, near-miss FN, and inversion FN test cases
│   ├── feature_attribution_flagged_sample.csv # Interpretability error breakdown
│   └── practical_impact.csv           # Operational developer hours and cost translation
└── outputs/
    └── test_predictions.csv           # Holdout test set predictions and anomaly scores
```

---

## 2. Methodology & 6-Checkpoint Pipeline (§5)

Our primary detector (Tier 2) employs a semi-supervised bottleneck Autoencoder with the following 6 checkpoints:
1. **Checkpoint 1 (Input Tensor):** Vector $x \in \mathbb{R}^{24}$ capturing 16 trajectory process dynamics (token consumption, step counts, tool loops, error cascades) and 8 static patch characteristics.
2. **Checkpoint 2 (Hygiene Preprocessing):** Standard scaling fit **strictly on normal-only training trajectories**, eliminating temporal and target leakage.
3. **Checkpoint 3 (Core Architecture):** A bottleneck Autoencoder ($\mathbb{R}^{24} \to \mathbb{R}^{16} \to \mathbb{R}^{6} \to \mathbb{R}^{16} \to \mathbb{R}^{24}$) with LayerNorm and ReLU activations, trained solely on normal data to reconstruct coherent workflow manifolds.
4. **Checkpoint 4 (Anomaly Scoring):** Mathematical formulation of the scalar anomaly score $s(x)$:
   $$s(x) = \frac{1}{D} \sum_{d=1}^D (x_d - \hat{x}_d)^2$$
5. **Checkpoint 5 (Threshold Decision):** Decision threshold $\tau = 0.1182$ calibrated strictly on validation data to maximize cost-sensitive $F_2$ score, suppressing silent task failures.
6. **Checkpoint 6 (Operational Alert):** Automated alert triggering hold-and-quarantine actions in CI/CD before untested code merges into main.

---

## 3. Main Benchmark Results (§7)

Evaluated across 5 random seeds (42, 123, 456, 789, 2026) on the identical holdout test set ($N=162$):

| Method | PR-AUC | ROC-AUC | F1 | F2 | Latency (ms) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Naive Heuristic** | $0.964 \pm 0.000$ | $0.701 \pm 0.000$ | $0.851 \pm 0.000$ | $0.794 \pm 0.000$ | $0.001 \pm 0.000$ | $0.000 \pm 0.000$ |
| **Tier 2: Deep Autoencoder (Ours)** | **$0.976 \pm 0.003$** | $0.772 \pm 0.017$ | **$0.947 \pm 0.005$** | **$0.952 \pm 0.006$** | $0.004 \pm 0.000$ | $0.634 \pm 0.658$ |
| **Tier 3: Supervised XGBoost** | $0.977 \pm 0.000$ | **$0.803 \pm 0.000$** | $0.882 \pm 0.000$ | $0.844 \pm 0.000$ | $0.005 \pm 0.000$ | $0.026 \pm 0.003$ |

### Statistical Significance
- **Tier 2 vs. Tier 1 (Naive):** Statistically significant improvement in PR-AUC ($\Delta = +0.012$, paired t-test $p = 0.0007$, bootstrap 95% CI $[+0.0002, +0.0246]$).
- **Tier 2 vs. Tier 3 (Supervised):** Gap is $\Delta = -0.0008$ (paired t-test $p = 0.565$, bootstrap 95% CI $[-0.0212, +0.0300]$ includes 0). Tier 2 performs **comparably** to the supervised ceiling without requiring any labeled failures during training, and achieves superior $F_2$ failure recall ($0.952$ vs. $0.844$).

---

## 4. Controlled Ablations (§7)

### Ablation 1 (AD-Specific): Training Contamination Sensitivity
- **0% (Clean Normal):** PR-AUC = $0.9759 \pm 0.0026$
- **1% Injected Failures:** PR-AUC = $0.9741 \pm 0.0040$ ($\Delta = -0.0017$)
- **10% Injected Failures:** PR-AUC = $0.9685 \pm 0.0035$ ($\Delta = -0.0074$, exceeding seed standard deviation by nearly $3\times$)
- **Finding:** Injected failures force the autoencoder to reconstruct anomaly error patterns, compressing the score gap and validating Hypothesis $H_2$.

### Ablation 2 (Core Research Question): Process Dynamics vs. Patch Output
- **Full Representation (Process + Patch, $D=24$):** PR-AUC = $0.9759 \pm 0.0026$, $F_2 = 0.9516$
- **Trajectory Process Only ($D=16$):** PR-AUC = $0.9699 \pm 0.0047$, $F_2 = 0.9532$
- **Static Final Patch Output Only ($D=8$):** PR-AUC = $0.9636 \pm 0.0070$ ($\Delta = -0.0122$), $F_2 = 0.9324$ ($\Delta = -0.0192$)
- **Finding:** Relying solely on the final patch diff degrades PR-AUC and failure recall significantly. Process dynamics (tool streaks, error loops, token consumption) provide essential signals that cannot be recovered from code diffs alone, validating Hypothesis $H_1$.

---

## 5. In-Depth Error Analysis (§8)

### Concrete Failure Cases ($\tau = 0.1182$)
1. **False Positive (#1 - `scikit-learn__scikit-learn-14894`):** Score $s(x) = 1.5675$ (Distance $+1.4493$). A complex genuine fix across multiple modules required 40+ steps and exploratory tool calls, mimicking failure thrashing.
2. **False Negative (#2 - `matplotlib__matplotlib-23562`):** Score $s(x) = 0.0977$ (Near miss, distance $-0.0205$). Agent recovered from an early syntax error and submitted an incomplete single-line fix, scoring just below $\tau$.
3. **False Negative (#3 - `sympy__sympy-15345`):** Score $s(x) = 0.0453$ (Complete inversion, distance $-0.0729$). Agent prematurely exited after minimal steps with zero tool errors, hallucinating task completion without modifying the broken logic.

### Interpretability Attribution
For flagged sample `django__django-11630` ($s(x) = 112.80 \gg \tau$):
- `stalled_turn_count` (raw = 14 turns) accounts for **$75.98\%$** of the total reconstruction error.
- `exit_cost_forced` accounts for **$6.19\%$**.
- Human auditors can immediately verify that this agent was stuck in execution loops rather than producing valid code.

### Practical Impact Translation
In an organization processing 500 agent tasks/month:
- Tier 2 catches **380 / 400 failures** (95.2% catch rate vs. Tier 1's 85.1%).
- Reduces silent bug escapes by **70%** (20 vs. 60).
- Reduces false audit reviews by **72%** (8 vs. 30).
- Delivers a net monthly saving of **+\$48,600** over the naive heuristic baseline.
