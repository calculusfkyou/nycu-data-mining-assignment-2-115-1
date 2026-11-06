# AI Coding Agent Deceptive Regression Anomaly Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end Anomaly Detection system on real SWE-bench Verified coding agent runs to detect **Deceptive Fixes-with-Regression** (patches passing local bug reproduction tests that silently break repository regression tests, natural prevalence 3.56%), evaluated via 5-fold grouped CV and a 5-seed Multimodal Bottleneck Autoencoder, and generate publication deliverables (6-page ACL LaTeX report, 11-slide presentation, verification suite, and submission zip).

**Architecture:**
- **Category:** Category 4: Text / Logs (Coding agent execution logs, terminal stdout/stderr, and unified git diffs).
- **Unit of Detection:** Candidate coding agent run $x = (\text{terminal\_trace}, \text{patch\_diff}, \text{dynamics}) \in \mathcal{X}$.
- **Population & Ground Truth:** All candidate patches passing target issue reproduction tests ($N = 2,667$ across 7 frontier `mini-SWE-agent v2.0.0` models); $y=0$ (cleanly resolved, $n_0 = 2,572$, 96.44%) vs $y=1$ (deceptive regression, $n_1 = 95$, 3.56% natural prevalence), derived objectively from Docker execution `logs/<id>/report.json`.
- **Validation Protocol:** 5-Fold Stratified Group K-Fold partitioned strictly by `instance_id` (zero issue/repo leakage across folds). Normal-only training on $y=0$ training instances.
- **Features:** R0-struct (20 features with domain Verification-Gap metrics), R1 (Character 3-5 gram TF-IDF), R2 (Frozen MiniLM 384-D), R3 (Variance-Equalized Multimodal Fusion: MiniLM + CodeBERT 768-D + R0-struct).
- **3-Tier Methods:**
  - *Tier 1 (Naive Baselines):* Predict-all-normal, 3-sigma Patch Churn, Unverified-Edit Heuristic.
  - *Tier 2 (Primary Method - HERO):* Calibrated Multimodal Bottleneck Autoencoder (CM-BAE), trained on $y=0$ normals with Huber loss across 5 random seeds (reporting `mean ± std`), accompanied by Variance-Standardized Dual $k$-NN, Inductive ECOD, and Isolation Forest.
  - *Tier 3 (Strong Reference):* Class-weighted Supervised XGBoost and Multimodal MLP.
- **Evaluation:** Primary metric PR-AUC (due to 1:27 class imbalance), secondary metric Cost-Optimal $F_2$ ($C_{\text{escaped}}=\$1,200$, $C_{\text{audit}}=\$30$, break-even $\pi^*=2.44\%$), ROC-AUC, FPR, Recall, and inference latency (ms/run).
- **Deliverables:** Verified code, data manifests, EDA figures, ablation curves, 6-page ACL report (`HW2_G04_report.pdf` & anonymized build), 11-slide presentation (`HW2_G04_presentation.pptx`), and `HW2_G04.zip`.

---

## Explicit Feedback & Lessons Learned Cross-Reference Table

Every design decision in this implementation directly resolves specific reviewer comments from the Oct 24 run (Mean: 8.58, R1: 7.5, R2: 9.6, R3: 8.6):

| Rubric / Reviewer | Reviewer Concern in Prior Draft (Oct 24) | Exact Architectural Resolution in v3 | Verified In |
|---|---|---|---|
| **D1 & D6 (R1, R2, R3)** | 81% failure rate is not a rare anomaly; breaks anomaly detection paradigm | Re-anchored to **Deceptive Fix-with-Regression**: $n_1=95$ out of $N=2,667$ candidate runs (**3.56% natural prevalence**, 1:27 imbalance) | §1, §3 |
| **D1 & D5 (R1, R2, R3)** | FPR $\le 20\%$ review cap lacked domain justification; conflicted with $F_2$/min-cost | Replaced with explicit **CI/CD triage economics**: $C_{\text{escaped}}=\$1,200$ vs $C_{\text{audit}}=\$30$; break-even $\pi^*=2.44\%$; calibration maximizes net savings | §1, §5, §7.2 |
| **D1 & D3 (R1, R2)** | 1:1 unweighted log/patch fusion read as unjustified default before ablation | Fused features are **variance-equalized** on $y=0$ training data a priori to guarantee equal information entropy | §4, §5 |
| **D2 & D5 (R1, R2, R3)** | Small numeric discrepancies (e.g. caption vs table; sympy error counts) | `tools/verify_v3.py` executes **end-to-end regex & numeric cross-validation** across LaTeX, figures, and slides | §8, tests |
| **D3 (R1, R2)** | Interpretability analyzed an auxiliary model TP rather than primary T2; thin analyst trust | CM-BAE outputs **direct additive channel attribution** ($s_{\text{log}} + s_{\text{patch}} + s_{\text{struct}}$) for primary flagged runs (e.g. `django-13590`) | §8, Fig 5 |
| **D3 (R2)** | Error cases read as 3 isolated anecdotes without a unifying blind spot | Explicitly defines and diagnoses the **shared blind spot**: *Unverified Edit Gap* (agents modifying files outside their executed test scope) | §8 |
| **D4 (R1, R3)** | Validation set too small (~10 normals) causing discrete quantile jumps in $\tau^*$ | Population expanded to $N=2,667$; each fold validation split has **~514 normal runs and ~19 anomalies** (50x increase!) | §3, §6 |
| **D4 (R1, R3)** | Balanced Accuracy as headline metric is inappropriate under severe skew | **PR-AUC (threshold-free)** and **Cost-Optimal $F_2$** elevated to primary headline metrics; BA demoted to descriptive baseline | Abstract, Table 2 |
| **D4 (R1, R2, R3)** | Deterministic models had no seed variation; needed paired tests and std | CM-BAE runs over **5 random seeds** (42, 123, 456, 789, 1011) reporting `mean ± std`, paired $t$-tests against baselines, and 2000-sample bootstrap CIs | Table 2 |
| **D4 (R2)** | Need issue/repo-level stratification justification to prevent cross-fold leakage | 5-fold cross-validation is **grouped by `instance_id`**, guaranteeing zero issue or repository file leakage across splits | §6 |
| **D5 (R1, R2)** | Soften claim of "exact hierarchy" replication on holdout | Accurately describes external holdout (`SWE-agent + Claude 4 Sonnet`) as **directional rank preservation** with explicit $p$-values | §7.1 |

---

## Proposed File Structure

```
test-nycu-data-mining-assignment-2-115-1/
├── Dataset/
│   └── verified/
│       ├── raw/                       # Cached .traj.json and report.json per model
│       ├── manifest.json              # Full 3,500 run provenance & Docker execution verdicts
│       ├── candidate_features.csv     # Extracted R0-struct features & labels (N=2,667)
│       ├── embeddings/                # Cached MiniLM and CodeBERT embeddings
│       └── splits.json                # Stratified Group K-Fold indices
├── EDA/
│   └── eda_v3.py                      # Training-only feature distribution contrast & statistics
├── Model/
│   ├── baselines_v3.py                # Tier 1 Heuristics
│   ├── cm_bae_v3.py                   # Tier 2 Primary Method (CM-BAE PyTorch model & trainer)
│   ├── tier2_controls_v3.py           # Standardized Dual k-NN, Inductive ECOD, Isolation Forest
│   ├── tier3_supervised_v3.py         # Supervised XGBoost & Multimodal MLP
│   ├── experiment_v3.py               # Main 5-fold 5-seed benchmark runner & statistical tests
│   ├── ablations_v3.py                # Contamination, modality, and threshold ablations
│   └── error_analysis_v3.py           # Traceable cases, channel attribution, & cost analysis
├── Paper/
│   ├── figures/v3/                    # Generated vector PDF figures
│   ├── generated/v3/                  # Generated LaTeX table inputs
│   ├── v3_body.tex                    # Core 6-page report manuscript
│   ├── HW2_G04_report.tex             # Named ACL entrypoint
│   └── build_v3_artifacts.py          # PDF compiler and LaTeX table generator
├── Presentation/
│   ├── content_v3.json                # 11 slide contents with 480s speaker notes
│   └── generate_deck_v3.py            # PowerPoint generator
├── tests/
│   └── test_v3_pipeline.py            # Strict verification unit tests
└── tools/
    ├── fetch_verified_trajectories.py # Real S3 downloader & Docker report parser
    ├── extract_verified_features.py   # R0-struct & embedding extraction
    ├── generate_verified_splits.py    # Grouped Stratified 5-fold generator
    └── verify_v3.py                   # Final submission integrity screen & artifact validator
```

---

## Tasks

### Phase 1: Real Data Ingestion & Grouped Leakage-Free Partitioning

- [ ] **Task 1.1: Fetch official evaluation logs & report manifests**
  - Create `tools/fetch_verified_trajectories.py` using `urllib.request` to stream `report.json` and `.traj.json` directly from public S3 `https://swe-bench-submissions.s3.amazonaws.com/bash-only/` across all 7 `mini-v2.0.0` frontier models:
    - `20260217_mini-v2.0.0_claude-4-6-opus`
    - `20260217_mini-v2.0.0_claude-4-5-opus-high`
    - `20260217_mini-v2.0.0_gemini-3-flash-high`
    - `20260217_mini-v2.0.0_gpt-5-2-high`
    - `20260219_mini-v2.0.0_gpt-5-2-codex`
    - `20260226_mini-v2.0.0_gemini-3-pro-high`
    - `20260217_mini-v2.0.0_deepseek-3-2-high`
    - Plus external holdout: `verified/20250522_sweagent_claude-4-sonnet-20250514`.
  - Parse `tests_status['FAIL_TO_PASS']` and `tests_status['PASS_TO_PASS']`.
  - Verify exact counts: $N_{\text{total}} = 3,500$, $N_{\text{candidate}} = 2,667$ (patches passing F2P), $n_0 = 2,572$ (resolved), $n_1 = 95$ (deceptive regression, 3.56%).
  - Export `Dataset/verified/manifest.json`.

- [ ] **Task 1.2: Feature Extraction Pipeline (R0-struct + Embeddings)**
  - Create `tools/extract_verified_features.py`.
  - Extract 20 structural & verification-gap features:
    - Code churn: `edit_lines_added`, `edit_lines_deleted`, `edit_files_count`, `edit_hunks_count`.
    - Verification gap: `unverified_edit_ratio`, `cross_module_dispersion`, `post_test_edit_steps`, `reproduction_test_ratio`.
    - Execution dynamics: `total_steps`, `total_tracebacks`, `terminal_error_streak`, `repeated_command_count`, `total_tokens_received`, `total_cost_usd`.
  - Compute and cache dense embeddings:
    - MiniLM 384-D on terminal command & observation chunks (`Dataset/verified/embeddings/minilm_logs.npy`).
    - CodeBERT 768-D on unified patch diffs (`Dataset/verified/embeddings/codebert_diffs.npy`).
  - Export `Dataset/verified/candidate_features.csv`.

- [ ] **Task 1.3: Grouped Stratified Partitioning (Issue-level isolation)**
  - Create `tools/generate_verified_splits.py`.
  - Implement 5-Fold Stratified Group Cross-Validation grouped by `instance_id` with seed 42.
  - Verify that no `instance_id` spans multiple folds.
  - Export `Dataset/verified/splits.json` and log fold counts (each fold: ~533 test runs, ~19 regressions).

- [ ] **Task 1.4: Unit Tests for Phase 1**
  - Create `tests/test_v3_pipeline.py`.
  - Test manifest validity, SHA-256 consistency, group disjointness across folds, and class balance.
  - Commit and push Phase 1.

---

### Phase 2: Exploratory Feature Analysis (§4)

- [ ] **Task 2.1: Training-Only Feature Contrast & Statistics**
  - Create `EDA/eda_v3.py`.
  - Restrict strictly to Fold 0 training rows ($y=0$ vs $y=1$).
  - Calculate median (IQR), standardized Wasserstein distance $W$, two-sample Kolmogorov-Smirnov distance $D_{\mathrm{KS}}$, and $p$-values.
  - Export LaTeX summary table to `Paper/generated/v3/v3_stats.tex`.

- [ ] **Task 2.2: Publication-Quality Distribution Figures**
  - Generate `Paper/figures/v3/eda.pdf` (boxplots of `unverified_edit_ratio`, `cross_module_dispersion`, and patch churn).
  - Generate `Paper/figures/v3/correlation_heatmap.pdf` checking feature redundancy and collinearity.

- [ ] **Task 2.3: Formulate Falsifiable Hypotheses**
  - Formulate H1 (Verification-gap features separate deceptive regressions from clean repairs significantly better than pure token counts).
  - Formulate H2 (Variance equalization prevents log channel from overwhelming patch diff signals in autoencoder reconstruction).
  - Formulate H3 (CM-BAE anomaly detection performance degrades monotonically under training set contamination).
  - Commit and push Phase 2.

---

### Phase 3: 3-Tier Model Implementation & Main Benchmark (§5, §6, §7)

- [ ] **Task 3.1: Tier 1 Naive Baseline Heuristics**
  - Create `Model/baselines_v3.py`.
  - Implement T1-A (Predict-all-normal), T1-B (3-sigma Patch Churn), T1-C (Unverified-Edit Rule).

- [ ] **Task 3.2: Tier 2 Primary Method — Calibrated Multimodal Bottleneck Autoencoder (CM-BAE)**
  - Create `Model/cm_bae_v3.py`.
  - PyTorch symmetric bottleneck architecture ($D \to 128 \to 32 \to 16 \to 32 \to 128 \to D$).
  - Training strictly on $y=0$ normal training instances with Huber loss and AdamW.
  - Additive reconstruction anomaly score $s(x) = s_{\text{log}}(x) + s_{\text{patch}}(x) + s_{\text{struct}}(x)$.
  - Multi-seed execution wrapper running seeds 42, 123, 456, 789, 1011.

- [ ] **Task 3.3: Tier 2 Auxiliary Controls**
  - Create `Model/tier2_controls_v3.py`.
  - Standardized Dual-Channel $k$-NN ($k \in \{1, 3, 5\}$ on normalized cosine distances).
  - Inductive ECOD (parameter-free tail CDF estimator fit on training $y=0$).
  - Isolation Forest (200 trees, subsampling 64).

- [ ] **Task 3.4: Tier 3 Strong Supervised Reference**
  - Create `Model/tier3_supervised_v3.py`.
  - Class-weighted XGBoost (`scale_pos_weight=27.0`) and Supervised Multimodal MLP.

- [ ] **Task 3.5: Full 5-Fold Benchmark Runner & Significance Tests**
  - Create `Model/experiment_v3.py`.
  - Run all models across all 5 folds under identical test splits.
  - Compute PR-AUC (primary), Cost-Optimal $F_2$, ROC-AUC, Balanced Accuracy, FPR, Recall, and Latency (ms/run).
  - Perform paired $t$-tests across folds against Tier 2 Primary Method and compute 2000-sample bootstrap 95% CIs.
  - Export `Paper/generated/v3/v3_main.tex` and raw CSV logs.

- [ ] **Task 3.6: Untouched External Holdout Evaluation**
  - Evaluate frozen models on untouched `SWE-agent + Claude 4 Sonnet` ($N=360$, $n_1=15$).
  - Confirm rank preservation in Mode A (zero-shot transfer) and Mode B (confirmatory replication).
  - Commit and push Phase 3.

---

### Phase 4: Controlled Ablations & Error Analysis (§7.2, §8)

- [ ] **Task 4.1: Controlled Ablation 1 — AD Training Contamination**
  - Create `Model/ablations_v3.py`.
  - Invalidate normal training pool with 0%, 1%, 3%, 5%, and 10% injected regressions.
  - Measure PR-AUC and $F_2$ degradation curves.
  - Export `Paper/generated/v3/v3_contamination.tex` and `Paper/figures/v3/contamination.pdf`.

- [ ] **Task 4.2: Controlled Ablation 2 — Modality & Variance Equalization**
  - Evaluate Log-only, Patch-only, Struct-only, Raw Concat, and Variance-Equalized Fusion.
  - Prove that variance equalization prevents channel drowning.
  - Export `Paper/generated/v3/v3_ablation.tex` and `Paper/figures/v3/ablations.pdf`.

- [ ] **Task 4.3: Controlled Ablation 3 — Operational Threshold Calibration**
  - Compare Fixed Quantile (95th) vs. Validation Max-$F_1$ vs. Review Capacity ($\mathrm{FPR} \le 10\%$) vs. Min-Cost Expected Loss.
  - Translate metric changes into operational dollar outcomes ($C_{\text{FN}}=\$1,200$, $C_{\text{FP}}=\$30$).
  - Export `Paper/generated/v3/v3_threshold.tex`.

- [ ] **Task 4.4: In-Depth Error Analysis & Interpretability**
  - Create `Model/error_analysis_v3.py`.
  - Audit at least 3 concrete test cases (TP `django-13590`, boundary FP, subtle FN) with sample IDs, raw feature values, $s(x)$, and $\tau^*$.
  - Generate CM-BAE additive per-channel reconstruction attribution plot (`Paper/figures/v3/interpretability.pdf`).
  - Assess analyst trust and compute developer review time savings.
  - Commit and push Phase 4.

---

### Phase 5: 6-Page ACL Paper Report, Oral Presentation & Deliverables (§1–§9)

- [ ] **Task 5.1: Build End-to-End 6-Checkpoint Pipeline Diagram**
  - Generate `Paper/figures/v3/pipeline.pdf` covering: (1) Candidate Run Input, (2) Feature Hygiene & Variance Equalization, (3) Normal-Only Bottleneck Training, (4) Additive Reconstruction Scoring, (5) Validation Cost-Calibrated Thresholding, (6) Quarantine Alert & Explanation Card.

- [ ] **Task 5.2: Draft and Compile ACL LaTeX Paper Report**
  - Update `Paper/v3_body.tex` with all verified v3 numbers, tables, citations, and sections:
    - §1 Abstract & Introduction (Stakeholder, CI cost model, Stage-2 gatekeeper, 3.56% natural anomaly definition).
    - §2 Related Work (SWE-bench, CodeBERT, Sentence-BERT, Autoencoders, ECOD, XGBoost).
    - §3 Task Definition & Data (Candidate population, objective Docker label, 1:27 imbalance).
    - §4 Exploratory Feature Analysis (Verification-gap contrast, KS/Wasserstein stats, H1-H3).
    - §5 Method (Pipeline figure, CM-BAE architecture, variance equalization, scoring $s(x)$).
    - §6 Experimental Setup (Grouped Stratified 5-Fold, pure-normal fitting, validation $\tau^*$).
    - §7 Results (Main benchmark table, 5-seed std, paired t-tests, bootstrap CIs, 3 ablations).
    - §8 Analysis (3 case studies, channel attribution, operational impact, deployment triggers).
    - §9 Conclusion & Limitations (Honest findings, deployment limitations, concept drift).
    - Contributions of Each Member & AI Usage Statement.
  - Compile `HW2_G04_report.pdf` (named) and `HW2_G04_report_anonymized.pdf` (anonymous review build).
  - Verify page length: exactly 6 body pages + separate references page. 0 overfull hbox errors.

- [ ] **Task 5.3: Oral Presentation Slides (11 Slides with 480s Budget)**
  - Create `Presentation/generate_deck_v3.py`.
  - Produce editable PowerPoint `HW2_G04_presentation.pptx` with embedded vector figures, native results tables, and 480-second speaker rehearsal notes.

- [ ] **Task 5.4: Final Verification, Packaging & E3 Archive**
  - Create `tools/verify_v3.py` executing over 1,000 automated checks:
    - Ground-truth label cross-check against official S3 Docker reports.
    - Zero issue overlap across cross-validation folds.
    - Consistency of all numbers across abstract, text, tables, figures, and slides.
    - Integrity screen check (zero score solicitation or criteria rewriting text).
    - Package final submission zip `HW2_G04.zip` containing exactly `HW2_G04_report.pdf` and `HW2_G04_presentation.pptx`.
  - Update root `README.md` with complete v3 reproduction instructions.
  - Final commit and push.
