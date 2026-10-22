# HW2 v2: Detecting Coding-Agent Failure with Local NLP Representations Implementation Plan

## Overview

This implementation plan executes the complete v2 experimental specification for detecting SWE-agent trajectory failures. The study investigates whether frozen local language and code encoders (Sentence-MiniLM and CodeBERT) combined with normal-only kNN distance outperform 24 tabular features, TF-IDF baselines, and legacy models on 284 SWE-agent GPT-4 trajectories (54 resolved/success, 230 unresolved/failure), specifically aiming to reduce false positive alarms on successful runs.

All models and representations are evaluated strictly using a 5-fold stratified outer cross-validation with inner validation model selection (seed 42, ~60/20/20 effective split), accompanied by leave-repository-out domain sensitivity analysis, rigorous error analysis, and 2,000-sample paired bootstrap confidence intervals.

## Proposed File Structure

```
test-nycu-data-mining-assignment-2-115-1/
├── configs/
│   └── experiment_v2.json             # Single source of truth: models, seeds, grids, metrics, flags
├── Dataset/
│   ├── v2/
│   │   ├── audit_report.json          # Audit of 10 normal & 10 failure trajectories JSON/timeline
│   │   ├── splits.json                # 5-fold outer CV + inner validation + repo-held-out split IDs
│   │   ├── text_manifest.jsonl        # Per-instance text hashes, timestamps, normalization flags
│   │   └── embeddings/                # Local cache of R1, R2, R3 representations & manifest
├── tools/
│   ├── audit_trajectories.py          # Data timeline and observation leakage audit tool
│   ├── normalize_text.py              # Text sanitization (ANSI removal, path/repo masking)
│   └── build_text_embeddings.py       # MiniLM, CodeBERT, and TF-IDF extraction & local caching
├── Model/
│   ├── tabular_features_v2.py         # R0 legacy and R0-clean tabular feature preparation
│   ├── models_v2.py                   # Core detectors: kNN distance, Logistic, ECOD, IF, DAE, Residual
│   ├── experiment_v2.py               # Main 5-fold CV runner across all primary & control models
│   ├── ablations_v2.py                # Representation, channel, contamination, and keyword ablations
│   ├── error_analysis_v2.py           # FP, FN, threshold-boundary case studies & LOO sensitivity
│   ├── figures_v2.py                  # Confusion matrices, ROC/PR, tradeoff & risk-coverage plots
│   └── verify_v2.py                   # Full invariant assertion test suite
├── Model/outputs/v2/                  # Predictions, raw scores, threshold tau records
├── Model/tables/v2/                   # Main results, fold details, ablations, bootstrap CIs, timing
└── Model/figures/v2/                  # High-resolution PNG and PDF publication figures
```

## Task Decomposition

### Task 1: Trajectory Audit & Text Sanitization Engine

**Files:**
- Create: `tools/audit_trajectories.py`
- Create: `tools/normalize_text.py`
- Output: `Dataset/v2/audit_report.json`

**Actions:**
1. Implement audit script inspecting at least 10 resolved (normal, y=0) and 10 unresolved (failure, y=1) trajectories:
   - Check JSON structure: `trajectory[].action`, `trajectory[].observation`, `info.submission` (patch).
   - Check observation timeline to verify observations were produced during agent execution and are not post-evaluation outputs.
   - Verify absence of evaluation leakage: no `resolved`, gold patch, gold tests, or grader outputs in inputs.
2. Implement fixed text normalization in `tools/normalize_text.py`:
   - Strip ANSI escape sequences (`re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')`).
   - Replace absolute working directory paths with `<WORKDIR>`.
   - Replace repository root names and instance IDs with `<REPO_ROOT>` and `<INSTANCE_ID>`.
   - Preserve programming symbols, syntax, traceback formatting, and relative paths.
   - Handle empty patches via explicit `empty-patch` mask without substituting arbitrary placeholder text.
3. Run audit on all 284 trajectories and save detailed inspection results.
4. Verify audit results: zero evaluator leakage confirmed, normalization handles edge cases.
5. Commit and push: `feat(data): implement trajectory audit and text normalization engine`.

---

### Task 2: Experiment Configuration & Stratified Split Manifests

**Files:**
- Create: `configs/experiment_v2.json`
- Create: `Dataset/v2/splits.json`
- Create: `Dataset/v2/text_manifest.jsonl`

**Actions:**
1. Create `configs/experiment_v2.json` freezing:
   - Random seeds: split seed 42, inner validation seeds (100+fold), optimization seeds (42, 123, 456), bootstrap seed 2026.
   - Models & revisions: `sentence-transformers/all-MiniLM-L6-v2`, `microsoft/codebert-base`.
   - Chunking rules: MiniLM (content <= 224, overlap 32, total <= 256), CodeBERT (content <= 480, overlap 64, total <= 512).
   - Hyperparameter grids: kNN k in {1, 3, 5}; Logistic C in {0.01, 0.1, 1.0, 10.0}.
   - Threshold policy: empirical validation FPR <= 20% maximizing failure recall, tie-breaking on higher threshold.
   - Secondary model flags: `enable_shallow_dae: true`, `enable_conditional_baseline: true`.
2. Generate 5-fold stratified outer CV splits in `Dataset/v2/splits.json`:
   - 5 outer folds on all 284 instances stratified by label (y=0: 54, y=1: 230).
   - Inner validation split: 25% of training partition per fold stratified by label (effective ~60% train / 20% val / 20% test).
   - Generate repository-held-out splits (grouping by repository prefix, e.g., `astropy`, `django`, `sympy`, etc.).
3. Generate `Dataset/v2/text_manifest.jsonl` recording:
   - `instance_id`, SHA-256 of raw file, label, outer fold ID, inner validation split ID.
   - Number of steps, total log characters, patch characters, empty patch mask, missing log mask.
4. Verification step: assert no instance overlap between train/val/test in any fold; confirm label proportions.
5. Commit and push: `feat(dataset): freeze v2 experiment config and stratified cross-validation splits`.

---

### Task 3: Local Text Representations & Embedding Pipeline (R1, R2, R3)

**Files:**
- Create: `tools/build_text_embeddings.py`
- Output: `Dataset/v2/embeddings/r2_minilm.npz`
- Output: `Dataset/v2/embeddings/r3_codebert.npz`
- Output: `Dataset/v2/embeddings/cache_manifest.json`

**Actions:**
1. Implement local tokenization and embedding pipeline:
   - **R1 (TF-IDF)**: Character TF-IDF (ngrams 3-5, min_df=2, sublinear TF, max 20,000 features, L2 normalized). Must provide a fold-aware fitting function fitted *only* on the training partition of the active fold.
   - **R2 (MiniLM Log Embedding)**: Chunk action and observation with step preservation. Content tokens <= 224, overlap 32, max 256. Step embedding = mean of step chunks, L2 normalized. Run embedding = equal-weight mean of steps, L2 normalized. Also cache per-step embeddings for error analysis.
   - **R3 (CodeBERT Patch Embedding)**: Split diff into hunks. Content tokens <= 480, overlap 64, max 512. Mean pool last-layer hidden states over non-padding/non-special tokens. Hunk-weighted average by content token count, L2 normalized. Empty patch gets zero vector.
   - **Channel Distance Metric**: `d = (d_log + d_patch) / 2`.
     - Log distance: `(1 - cosine) / 2`.
     - Patch distance: both empty -> 0; exactly one empty -> 1; both present -> `(1 - cosine) / 2`.
2. Run probe on 5 trajectories first: measure throughput, token counts, and memory usage.
3. Compute full local representations for all 284 trajectories without external APIs.
4. Store representations in `Dataset/v2/embeddings/` along with `cache_manifest.json` containing package versions, model revisions, and input hashes.
5. Verification step: verify shapes (R2: 284x384, R3 patch: 284x768), check L2 norms equal 1.0 (or 0 for empty patch), verify distance metric edge cases.
6. Commit and push: `feat(nlp): implement local MiniLM and CodeBERT embedding extraction pipeline`.

---

### Task 4: Tabular Data Preparation (R0 and R0-clean)

**Files:**
- Create: `Model/tabular_features_v2.py`
- Output: `Dataset/v2/r0_features.csv`, `Dataset/v2/r0_clean_features.csv`

**Actions:**
1. Extract and standardize tabular feature representations:
   - **R0 Legacy**: 24 tabular features as defined in v1.
   - **R0-clean**: 23 features (strictly drop redundant `num_tool_calls`). Apply `log1p` to 16 non-negative long-tailed features (`num_steps`, `num_unique_tools`, `max_repeated_tool_streak`, `num_error_observations`, `consecutive_errors_max`, `stalled_turn_count`, `test_suite_run_count`, `total_cost_usd`, `tokens_sent`, `tokens_received`, `tokens_per_step`, `patch_char_length`, `patch_lines_added`, `patch_lines_removed`, `patch_num_files_changed`, `patch_hunk_count`).
   - Preserve binary features as 0/1 (`reproduce_script_created`, `last_step_error`, `patch_is_empty`, `patch_touches_tests`).
   - Non-binary features scaled strictly on normal fitting rows using median/IQR (fallback to SD if IQR=0, fallback to 1 if SD=0).
2. Implement **Inductive ECOD Variant**:
   - Fit strictly on normal fitting rows: compute ECDF and skewness per column.
   - Scoring on test instances uses frozen training ECDF: $p_L = \frac{1 + \sum [x_{train} \le x]}{n+1}$, $p_R = \frac{1 + \sum [x_{train} \ge x]}{n+1}$, $U_L = -\log(p_L)$, $U_R = -\log(p_R)$.
   - Verify sample-level invariance: score for instance $x$ is identical whether evaluated individually or in varying batch sizes/orders.
3. Verification step: run tests verifying R0-clean has 23 features, scaler fits only on normal rows, inductive ECOD invariance holds.
4. Commit and push: `feat(tabular): implement clean tabular feature engineering and inductive ECOD variant`.

---

### Task 5: Main Benchmark Experiment Suite

**Files:**
- Create: `Model/models_v2.py`
- Create: `Model/experiment_v2.py`
- Output: `Model/outputs/v2/oof_predictions.csv`
- Output: `Model/tables/v2/main_results.csv`
- Output: `Model/tables/v2/fold_metrics.csv`
- Output: `Model/tables/v2/precision_at_k.csv`
- Output: `Model/tables/v2/timing_memory.csv`

**Actions:**
1. Implement detector implementations in `Model/models_v2.py`:
   - **T1**: Rule-based error/repetition heuristic.
   - **T2 (Primary)**: R3 + kNN normal distance ($k \in \{1, 3, 5\}$).
   - **T3 (Supervised Reference)**: R3 concatenation ($1/\sqrt{2} \times \text{log} \mathbin{\Vert} 1/\sqrt{2} \times \text{patch}$) + L2 Logistic Regression ($C \in \{0.01, 0.1, 1, 10\}$, balanced class weight).
   - **Necessary NLP Control**: R1 (TF-IDF) + L2 Logistic Regression.
   - **Necessary Tabular Controls**:
     - R0-clean + Isolation Forest (200 trees, seeds 42, 123, 456).
     - R0-clean + Inductive ECOD variant.
   - **Legacy Controls**: R0 Autoencoder (seeds 42, 123, 456), R0 XGBoost.
   - **Context Controls**: Always-alert, Never-alert.
   - **Pre-specified Secondary Extensions**:
     - Shallow DAE on R0-clean ($23 \to 16 \to 4 \to 16 \to 23$, Huber + BCE loss, seeds 42, 123, 456).
     - Conditional baseline (Ridge $\alpha=10$ + Logistic $C=0.1$ predicting 8 patch features from 15 behavior features).
2. Execute 5-fold outer cross-validation:
   - Inner validation selects best $k$ or $C$ and threshold $\tau$ under constraint: empirical validation FPR <= 20% maximizing failure recall, breaking ties on balanced accuracy, then smaller $k/C$.
   - Evaluate test partition using frozen model and threshold $\tau$.
   - Compute OOF predictions across all 284 instances.
   - Compute Precision@K ($K \in \{10, 20, 50\}$).
   - Record exact wall-clock time and peak memory per phase.
   - Execute leave-repository-out sensitivity analysis.
3. Verification step: verify total pooled OOF instances = 284; check metric calculations (balanced acc, FPR, recall, unflagged success fraction).
4. Commit and push: `feat(experiment): execute full v2 benchmark suite across all models and controls`.

---

### Task 6: Controlled Ablation Studies

**Files:**
- Create: `Model/ablations_v2.py`
- Output: `Model/tables/v2/ablation_representations.csv`
- Output: `Model/tables/v2/ablation_channels.csv`
- Output: `Model/tables/v2/ablation_contamination.csv`
- Output: `Model/tables/v2/ablation_keywords.csv`

**Actions:**
1. Implement and run **Ablation 1: Representations**:
   - Compare R0 (Euclidean on standardized normal fit), R1 (cosine on TF-IDF), R2 (cosine on MiniLM), R3 (dual distance) under identical kNN detector and validation policy.
2. Implement and run **Ablation 2: Channel Contribution**:
   - Compare Log-only (R2 MiniLM), Patch-only (CodeBERT), and Equal-weight Dual (R3) under kNN.
3. Implement and run **Ablation 3: Normal Reference Contamination**:
   - Contaminate normal training reference with $p \in \{0, 0.05, 0.10\}$ failure runs (seeds 42, 123, 456).
   - Evaluate under clean $\tau$ vs. recalibrated $\tau$ on validation fold.
4. Implement **Auxiliary Diagnostics**:
   - Last-step observation only vs. full trajectory mean.
   - Keyword ablation: mask frozen error keywords (`error`, `traceback`, `failed`, `exception`) and recompute embeddings.
5. Verification step: confirm all ablations use identical fold partitions and report complete results across all folds.
6. Commit and push: `feat(ablation): execute controlled representations, channels, and contamination ablations`.

---

### Task 7: Detailed Error Analysis, Interpretability & Bootstrap Statistics

**Files:**
- Create: `Model/error_analysis_v2.py`
- Create: `Model/figures_v2.py`
- Output: `Model/tables/v2/case_studies.json`
- Output: `Model/tables/v2/bootstrap_ci.csv`
- Output: `Model/figures/v2/fig_confusion_matrices.png`, `fig_roc_pr_curves.png`, `fig_tradeoff_curves.png`, `fig_risk_coverage.png`, `fig_ablations.png`

**Actions:**
1. Implement error case selection:
   - Highest-scoring False Positive (true success flagged as failure).
   - Lowest-scoring False Negative (true failure missed).
   - Borderline error (closest to threshold $\tau$).
   - For each case: extract instance ID, SHA-256, score, $\tau$, nearest normal neighbor ID, step-level distances, and leave-one-out sensitivity for the most influential step/hunk.
2. Implement stratified paired bootstrap (2,000 resamples, seed 2026):
   - Compute 95% bootstrap confidence intervals for Balanced Accuracy, FPR, Recall, Unflagged Success Fraction, ROC-AUC, AP, and pairwise differences vs. T2.
3. Generate publication-grade figures in `Model/figures/v2/`:
   - Confusion matrices: raw integer counts side-by-side with row-normalized percentages.
   - ROC & PR curves with prevalence baseline.
   - FPR-Recall tradeoff and Risk-Coverage curves.
   - Ablation performance charts.
4. Verification step: verify bootstrap reproducibility, assert all 3 case studies link to authentic SHA-256 and trajectory steps.
5. Commit and push: `feat(analysis): perform deep error analysis, bootstrap CIs, and generate publication figures`.

---

### Task 8: Verification Suite & Documentation

**Files:**
- Create: `Model/verify_v2.py`
- Update: `README.md`
- Output: `Model/tables/v2/practical_deployment_evaluation.csv`

**Actions:**
1. Implement automated test script `Model/verify_v2.py` asserting:
   - Fold disjointness (train/val/test disjoint per fold).
   - Exact 284 pooled OOF coverage.
   - Invariant check: confusion matrix numbers equal raw metric formulas.
   - Local execution confirmation: no remote API calls.
   - Real data fidelity: zero synthetic records.
2. Evaluate and state practical deployment criteria:
   - Check if OOF Failure Recall >= 90%, FPR <= 20%, Unflagged Success Fraction >= 80%, Coverage >= 5%.
   - Document findings transparently: whether NLP meets or fails the practical deployment threshold.
3. Update project `README.md` summarizing v2 methodology, findings, and deliverables.
4. Final verification and push: `docs: update documentation with comprehensive v2 experimental findings`.
