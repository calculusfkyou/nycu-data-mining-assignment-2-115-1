# System Architecture & Design Specification: AI Coding Agent Deceptive Regression Anomaly Detector

- **Date:** 2026-11-07
- **Team ID:** G04 (NYCU Data Mining Fall 2026, HW2 Anomaly Detection)
- **Status:** Validated Architectural Design Specification
- **Subject Category:** Category 4: Text / Logs (Software Engineering Coding Agent Interaction Traces, Terminal Logs, and Code Diffs)

---

## 1. Overview & Problem Formulation

### 1.1 Engineering Stakeholder & Operational Context
Autonomous coding agents (e.g., SWE-agent, mini-SWE-agent, OpenHands) iteratively edit code, inspect terminal outputs, and execute unit tests to solve complex software issues. In modern CI/CD pull-request (PR) integration pipelines, candidate patches must undergo rigorous validation before merging into release branches.
Running full-repository regression suites (e.g., thousands of test cases in Django, SymPy, Matplotlib) takes 30–45 minutes and incurs ~$150 in compute/runner costs per run.
To optimize CI capacity and prevent silent repository corruption, a **CI/CD Release Engineer** deploys our anomaly detector as a **Stage-2 Candidate PR Gatekeeper**:
- **Operational Trigger:** When an agent finishes execution and its candidate patch passes the local target issue reproduction test (`FAIL_TO_PASS` test suite, runtime ~1 second).
- **Detector Decision:** The detector inspects the agent's interaction trajectory (terminal logs) and final git diff to calculate an anomaly score $s(x)$.
  - **Flagged as Anomaly ($s(x) \ge \tau^*$):** The candidate patch is flagged as a high-risk *Deceptive Regression* and held in quarantine for mandatory human review ($C_{\text{audit}} = \$30$).
  - **Unflagged ($s(x) < \tau^*$):** The candidate patch proceeds to the automated full-repository regression suite. An escaped regression that slips past CI into production incurs catastrophic remediation costs ($C_{\text{escaped}} = \$1,200$).

### 1.2 The "Deceptive Fix-with-Regression" Phenomenon
In benchmark evaluations, agents frequently achieve "apparent success" on target bugs while silently breaking existing passing functionality. SWE-bench evaluates patches in isolated Docker containers against two orthogonal test splits:
1. `FAIL_TO_PASS` (F2P): Tests that reproduce the specific issue. These must switch from FAIL to PASS.
2. `PASS_TO_PASS` (P2P): Pre-existing repository regression tests. These must remain PASS.

When evaluating candidate patches that pass `FAIL_TO_PASS`, the vast majority are cleanly resolved ($y=0$). However, a critical minority of patches pass `FAIL_TO_PASS` yet break existing `PASS_TO_PASS` tests ($y=1$). This constitutes a textbook, naturally rare, high-stakes anomaly: **Deceptive Fix-with-Regression**.

### 1.3 Dataset Specification & Ground-Truth Definitions
All data originate from official SWE-bench Verified evaluation runs archived in `SWE-bench/experiments` and stored on the public S3 bucket `swe-bench-submissions.s3.amazonaws.com`.

- **Benchmark Split:** `SWE-bench Verified` (500 human-validated real-world software engineering tasks).
- **Core Experimental Population ($N = 2,667$ candidate runs across 7 frontier models):**
  We audit all complete 500/500 `mini-SWE-agent v2.0.0` runs from February 2026:
  1. `20260217_mini-v2.0.0_claude-4-6-opus` (Resolved: 378, F2P-passing: 388, Deceptive Regression: 10)
  2. `20260217_mini-v2.0.0_claude-4-5-opus-high` (Resolved: 384, F2P-passing: 395, Deceptive Regression: 11)
  3. `20260217_mini-v2.0.0_gemini-3-flash-high` (Resolved: 379, F2P-passing: 391, Deceptive Regression: 12)
  4. `20260217_mini-v2.0.0_gpt-5-2-high` (Resolved: 364, F2P-passing: 381, Deceptive Regression: 17)
  5. `20260219_mini-v2.0.0_gpt-5-2-codex` (Resolved: 364, F2P-passing: 374, Deceptive Regression: 10)
  6. `20260226_mini-v2.0.0_gemini-3-pro-high` (Resolved: 353, F2P-passing: 368, Deceptive Regression: 15)
  7. `20260217_mini-v2.0.0_deepseek-3-2-high` (Resolved: 350, F2P-passing: 370, Deceptive Regression: 20)

- **Class Distribution:**
  - **Total Evaluated Patches Passing F2P:** $N = 2,667$
  - **Normal Class ($y = 0$, Cleanly Resolved):** $n_0 = 2,572$ (**96.44%**)
  - **Anomaly Class ($y = 1$, Deceptive Regression):** $n_1 = 95$ (**3.56% natural prevalence**, ratio 1:27)
  - *5-Model Core Subset (Opus 4.6, Opus 4.5, GPT-5.2, Gemini 3 Pro, DeepSeek V3.2):* $N = 1,902$, $n_0 = 1,829$ (96.16%), $n_1 = 73$ (**3.84%**).
  - *Full Unfiltered Population ($N = 3,500$ runs):* Any `PASS_TO_PASS` breakage occurs in $176 / 3,500$ runs (**5.03%**).
  - *External Untouched Holdout (`SWE-agent + Claude 4 Sonnet`, 20250522):* $N = 360$ F2P-passing runs, $n_0 = 345$ (95.83%), $n_1 = 15$ (**4.17%**).

- **Objective, Non-Circular Ground Truth:**
  Labels are derived strictly from Docker execution outputs in `logs/<instance_id>/report.json`:
  $$y = \begin{cases} 1 & \text{if } \mathrm{len}(\mathrm{report}[\text{'tests\_status'}][\text{'PASS\_TO\_PASS'}][\text{'failure'}]) > 0 \\ 0 & \text{if } \mathrm{len}(\mathrm{report}[\text{'tests\_status'}][\text{'PASS\_TO\_PASS'}][\text{'failure'}]) == 0 \end{cases}$$
  Because `report.json` is generated by an isolated external evaluation harness after trajectory completion, the agent has zero visibility into regression tests. The ground truth is 100% physical and non-circular.

---

## 2. Leakage-Free Validation Protocol (§6)

### 2.1 Grouped Stratified Partitioning
Because multiple models evaluate the same underlying SWE-bench tasks, random row shuffling would cause same-issue data leakage across folds.
- **Protocol:** 5-Fold Stratified Group Cross-Validation grouped strictly by `instance_id` (the software issue ID).
- **Guarantee:** All runs for a given `instance_id` across all agent models reside in the exact same fold. No test issue or test repository file is ever seen during training or validation.
- **Split Ratio:** Each outer fold provides ~60% train ($y=0$ normal fitting pool), ~20% inner validation (for hyperparameter tuning and threshold $\tau^*$ selection), and ~20% test holdout (evaluated once).

### 2.2 Normal-Only Training Paradigm
- **Training Set:** Tier 2 one-class reconstruction models fit strictly on $y=0$ (cleanly resolved) training samples ($n_{\text{fit}} \approx 1,540$ per fold).
- **Preprocessing Hygiene:** All tokenizers, TF-IDF vocabularies, robust scalers, and channel normalization parameters are computed strictly on $y=0$ training instances. Held-out validation and test rows are never accessed during feature transformation fitting.
- **Validation Set:** Validation rows (containing labeled normal and anomalous instances) are used exclusively to select hyperparameters and calibrate decision thresholds $\tau^*$.
- **Test Set:** Evaluated once per fold with frozen models and frozen thresholds.

---

## 3. Multimodal Feature Engineering (§4: R0–R3)

### 3.1 R0-struct: Execution Dynamics & Verification-Gap Features (20 features)
Unlike standard token counts, we introduce domain-grounded **Verification-Gap Features** that detect when an agent altered code outside the scope of its verified test executions:
1. `edit_lines_total`, `edit_files_count`, `edit_hunks_count`: Scale of the patch diff.
2. `unverified_edit_ratio`: Fraction of files modified in the git diff that were never referenced in any `pytest` or test execution observation in the terminal logs.
3. `cross_module_dispersion`: Entropy of directories modified (agents that touch multiple unrelated packages have high regression risk).
4. `post_test_edit_steps`: Number of code modification steps performed *after* the agent's last test suite execution.
5. `reproduction_test_ratio`: Frequency of test execution commands relative to total bash actions.
6. `terminal_error_streak`, `total_tracebacks`, `repeated_command_count`: Interaction instability indicators.
7. `total_steps`, `total_tokens_received`, `total_cost_usd`: Trajectory complexity metrics.

### 3.2 R1: Character n-gram TF-IDF Baseline
- Sublinear term-frequency character 3–5 grams extracted separately from terminal commands/logs and unified patch diffs.
- Vocabulary truncated to top 10,000 features per channel, fit strictly on $y=0$ training folds.

### 3.3 R2: Dense Text Embeddings (Terminal Logs)
- Frozen `sentence-transformers/all-MiniLM-L6-v2` (384-D).
- Terminal actions and observations chunked into 224-token windows with 32-token overlap. Step embeddings are mean-pooled across execution steps and $\ell_2$-normalized to $z_{\text{log}} \in \mathbb{R}^{384}$.

### 3.4 R3: Multimodal Dense Representation with Variance Equalization
- Code diffs chunked into 480-token windows with 64-token overlap and encoded via frozen `microsoft/codebert-base` (768-D), yielding $z_{\text{patch}} \in \mathbb{R}^{768}$.
- **Variance Equalization:** To eliminate the channel dominance flaw diagnosed in v2 (where log distance variance drowned patch signals by 56x), each channel is standardized by its empirical trace covariance on $y=0$ training data before concatenation:
  $$z_{\text{fused}} = \left[ \frac{z_{\text{log}}}{\sigma_{\text{log}}}, \frac{z_{\text{patch}}}{\sigma_{\text{patch}}}, \frac{z_{\text{struct}}}{\sigma_{\text{struct}}} \right]$$

---

## 4. 3-Tier Method Comparison Architecture (§5 & §7)

```
[Candidate Agent Run x: (Logs, Diff, Dynamics)]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 [Tier 1: Baselines] [Feature Extraction: R0-struct, MiniLM, CodeBERT]
   - Predict Normal        │
   - 3-sigma Churn         ▼
   - Unverified Edit   [Variance Equalization (fit on Train y=0)]
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 [Tier 2: Primary Method (HERO)]     [Tier 3: SOTA References]
  Calibrated Multimodal Bottleneck    - Supervised XGBoost (class-weighted)
  Autoencoder (CM-BAE)                - Multimodal MLP / Logistic Regression
  Score s(x) = ||z - z_hat||^2                │
        │                                     ▼
        ▼                              [Upper Bound Reference]
 [Validation Threshold Tuning tau*]
        │
        ▼
 [Deployable Binary Alert & Interpretability Attribution]
```

### 4.1 Tier 1: Naive Baseline Heuristics
- **T1-A (Predict-all-normal):** Demonstrates the class-imbalance trap where 96.44% accuracy yields 0% anomaly recall and 0.000 PR-AUC.
- **T1-B (3-sigma Patch Churn):** Flags patches whose total changed lines exceed $\mu_0 + 3\sigma_0$ of normal training runs.
- **T1-C (Unverified-Edit Rule):** Flags runs where `unverified_edit_ratio` $> 0.5$ or `post_test_edit_steps` $\ge 2$.

### 4.2 Tier 2: Primary Method — Calibrated Multimodal Bottleneck Autoencoder (CM-BAE)
- **Architecture:** Symmetric bottleneck neural autoencoder ($D_{\text{in}} \to 128 \to 32 \to 16 \to 32 \to 128 \to D_{\text{in}}$) with LeakyReLU activations and dropout ($p=0.1$).
- **Training:** Trained exclusively on $y=0$ training instances for 100 epochs using AdamW ($\mathrm{lr}=10^{-3}$, weight decay $10^{-4}$) with Huber loss ($\delta=1.0$) to prevent gradient explosion on long diffs.
- **Anomaly Scoring:** Reconstruction squared Euclidean error:
  $$s(x) = \|z_{\text{fused}} - \hat{z}_{\text{fused}}\|_2^2 = s_{\text{log}}(x) + s_{\text{patch}}(x) + s_{\text{struct}}(x)$$
- **Stochastic Rigor:** Evaluated across 5 random seeds (42, 123, 456, 789, 1011) to report exact `mean ± std`.
- **Auxiliary Tier 2 Controls:**
  - *Standardized Dual-Channel k-NN* ($k \in \{1, 3, 5\}$ on equalized cosine distances).
  - *Parameter-free Inductive ECOD* on clean structural features.
  - *Isolation Forest* (200 trees, contamination set to validation estimate).

### 4.3 Tier 3: Strong Supervised Reference
- **Supervised XGBoost:** Class-weighted (`scale_pos_weight` $= 27.0$), max depth 3, 150 trees, trained on full training partition ($y \in \{0, 1\}$).
- **Role:** Serves as the empirical upper bound to demonstrate how close an unsupervised normal-only detector can approach fully supervised discrimination.

---

## 5. Decision Threshold Calibration & Economic Cost Model (§5 & §8)

### 5.1 Operational Cost Matrix
- Escaped Regression to Production ($C_{\text{FN}}$): **$1,200** (emergency patch, post-mortem, downtime).
- False Alarm Quarantine Review ($C_{\text{FP}}$): **$30** (15 minutes of senior engineer triage).
- True Negative Clean Automated Release ($C_{\text{TN}}$): **$0**.
- True Positive Caught Regression ($C_{\text{TP}}$): **$30** (review cost).
- **Break-Even Prevalence:**
  $$\pi^* = \frac{C_{\text{FP}}}{C_{\text{FN}} - C_{\text{TN}} + C_{\text{FP}}} = \frac{30}{1200 + 30} \approx 2.44\%$$
  Because our natural anomaly rate is **3.56%** ($\pi > \pi^*$), automated gating is economically viable and dominates both "never-alert" and "always-review".

### 5.2 Threshold Selection Strategies Compared in Validation
1. **Min-Cost Expected Loss ($\tau_{\text{cost}}^*$):** Minimizes total dollar loss on validation split:
   $$\tau^* = \arg\min_{\tau} \left( 30 \cdot \mathrm{FP}(\tau) + 1200 \cdot \mathrm{FN}(\tau) \right)$$
2. **Review Capacity Cap ($\tau_{\text{cap}}^*$):** Enforces operational constraint $\mathrm{FPR}_{\text{val}} \le 10\%$ while maximizing anomaly recall.
3. **Max $F_2$ Score:** Prioritizes recall with twice the weight of precision.

---

## 6. Single-Factor Controlled Ablations (§7.2)

1. **Ablation 1 (AD-Specific Training Contamination):**
   Inject 0%, 1%, 3%, 5%, and 10% unlabeled deceptive regressions into the normal training pool $\mathcal{R}_{\text{norm}}$. Measure PR-AUC and $F_2$ degradation to test robustness against imperfect training labels.
2. **Ablation 2 (Representation & Variance Equalization):**
   Evaluate CM-BAE on (a) Log-only, (b) Patch-only, (c) Struct-only, (d) Raw unweighted concatenation, and (e) Variance-equalized fusion. Measure the specific lift provided by the Verification-Gap features.
3. **Ablation 3 (Operational Threshold Policy):**
   Compare Fixed 95th Percentile vs. Max $F_1$ vs. Max $F_2$ vs. Min-Cost calibration across all five folds.

---

## 7. Concrete Failure Cases & Interpretability (§8)

### 7.1 Traceable Case Studies
- **True Positive Caught (e.g., `django__django-13590`):** Claude 4.6 Opus resolves the target query issue, but collaterally breaks 5 existing tests including `test_complex_expressions_do_not_introduce_sql_injection`. The detector flags this due to high `unverified_edit_ratio` ($0.75$) and elevated patch reconstruction error $s_{\text{patch}}$.
- **False Positive (Boundary Clean Fix):** A legitimate large refactoring that touches multiple configuration files without breaking existing tests.
- **False Negative (Subtle 1-line semantic flaw):** A subtle off-by-one boundary edit that escapes terminal and diff notice.

### 7.2 Channel & Feature Attribution
CM-BAE computes exact additive reconstruction loss per channel:
$$s(x) = s_{\text{log}}(x) + s_{\text{patch}}(x) + s_{\text{struct}}(x)$$
For any flagged run, the system outputs an automated explanation card displaying which channel triggered the anomaly threshold, restoring developer trust and reducing triage time from 15 minutes to <3 minutes.

---

## 8. Verification & Deliverables Plan

1. **Dataset Pipeline:** `tools/fetch_verified_trajectories.py` to ingest official `report.json` and `.traj.json` for all 7 frontier models directly from S3 into `Dataset/verified/`.
2. **Feature Extraction:** `tools/extract_verified_features.py` computing R0-struct, MiniLM, and CodeBERT embeddings with full provenance hashes.
3. **Experiment Runner:** `Model/experiment_v3.py` executing 5-fold grouped CV across Tier 1, 2, and 3 with 5 random seeds, bootstrap CIs, and paired t-tests.
4. **Paper & Slides:**
   - 6-page ACL LaTeX report (`HW2_G04_report.pdf` & anonymized version) compiling via Tectonic/LaTeX without overfull warnings.
   - 11-slide presentation (`HW2_G04_presentation.pptx`) with speaker notes totaling 480 seconds.
   - Submission ZIP `HW2_G04.zip`.
5. **Verification Suite:** `tests/test_v3_pipeline.py` verifying zero leakage, exact numbers, and rubric compliance.
