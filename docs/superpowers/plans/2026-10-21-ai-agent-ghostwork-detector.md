# AI Agent Ghostwork Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end Anomaly Detection system ("AI Agent Ghostwork Detector") on real SWE-bench agent trajectories to forecast task failure before final completion, and produce an ACL-format 6-page research paper and an 8-minute presentation deck adhering to all HW2 specifications and grading rubrics.

**Architecture:**
- **Data Category:** Text / Logs (Agent interaction traces, bash commands, error outputs, and patch diffs).
- **Unit of Detection:** End-to-end agent trajectory $x \in \mathbb{R}^D$ on a software engineering task.
- **3-Tier Comparison:**
  - *Tier 1 (Naive Baseline):* Rule-based heuristic (error count and tool loop thresholding).
  - *Tier 2 (Primary Method):* Deep Bottleneck Autoencoder trained on normal (successful) trajectories with MSE reconstruction error scoring $s(x) = \frac{1}{D}\sum_{d=1}^D (x_d - \hat{x}_d)^2$ and cost-sensitive threshold $\tau$.
  - *Tier 3 (Strong Reference):* Supervised XGBoost classifier.
- **Core Research Ablation:** Trajectory features (interaction dynamics) vs. Final Output features (static patch diffs only) to answer: *Does the full workflow trajectory detect failure better than the final patch?*
- **Deliverables:** Codebase, Dataset, EDA figures/tables, Model training & analysis notebooks, ACL LaTeX paper report (`HW2_<TeamID>_report.pdf`), and presentation slides (`HW2_<TeamID>_presentation.pptx` / `.pdf`).

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, scipy, torch, xgboost, matplotlib, seaborn, python-pptx, LaTeX (pdflatex / xelatex).

---

## Tasks

### Phase 1: Real Data Ingestion & Trajectory Feature Extraction
- [ ] Task 1.1: Download real SWE-bench trajectories and evaluation logs from official AWS S3 / SWE-bench experiments repository.
- [ ] Task 1.2: Build feature extraction pipeline (`extract_features.py`) transforming raw trajectories into structured feature vectors (sequential dynamics, tool loop counts, error frequencies, patch complexity, exit statuses).
- [ ] Task 1.3: Generate shared, leakage-free train/validation/test splits with fixed seed and export `Dataset/` CSV tables with summary statistics.
- [ ] Task 1.4: Commit and push Phase 1.

### Phase 2: Exploratory Feature Analysis (EDA) (§4)
- [ ] Task 2.1: Implement `eda.py` computing distribution contrast, Kolmogorov-Smirnov distance, Wasserstein distance, and mutual information.
- [ ] Task 2.2: Generate high-resolution figures: Feature distributions contrast, correlation heatmap, and PCA/t-SNE latent space projection.
- [ ] Task 2.3: Formulate explicit hypotheses (H1: Trajectory dynamics separability, H2: Contamination sensitivity, H3: Cost-sensitive threshold tuning) strictly on the training portion.
- [ ] Task 2.4: Commit and push Phase 2.

### Phase 3: 3-Tier Model Implementation & Experimental Setup (§5, §6)
- [ ] Task 3.1: Implement Tier 1 Naive Baseline (heuristic rule based on error logs and tool loop thresholding).
- [ ] Task 3.2: Implement Tier 2 Primary Method (Deep Bottleneck Autoencoder with strictly normal-only training and validation $\tau$ tuning).
- [ ] Task 3.3: Implement Tier 3 Strong Reference (Supervised XGBoost on the identical split).
- [ ] Task 3.4: Run 5-seed benchmark evaluation, paired t-tests, bootstrap 95% confidence intervals, and inference latency measurements.
- [ ] Task 3.5: Commit and push Phase 3.

### Phase 4: Controlled Ablations & In-Depth Error Analysis (§7, §8)
- [ ] Task 4.1: Controlled Ablation 1 (AD-specific): Training set contamination ratio (0%, 1%, 5%, 10% injected failures).
- [ ] Task 4.2: Controlled Ablation 2 (Core Research Question): Trajectory-level features vs. Final patch output features.
- [ ] Task 4.3: Failure Case Studies (§8.1): Detailed breakdown of at least 3 test cases (including False Positives and False Negatives) with sample IDs, raw features, $s(x)$, $\tau$, distance, and root causes.
- [ ] Task 4.4: Interpretability & Practical Impact (§8.2, §8.3): Feature reconstruction error attribution and operational developer time / cost savings translation.
- [ ] Task 4.5: Commit and push Phase 4.

### Phase 5: ACL LaTeX Paper Report Production (§1–§9)
- [ ] Task 5.1: Set up ACL LaTeX template with preprint and review modes.
- [ ] Task 5.2: Create the 6-checkpoint End-to-End Pipeline Figure for Tier 2 Primary Method.
- [ ] Task 5.3: Draft all 9 sections, Contributions, and AI Usage Statement according to the 100-point rubric and D1-D6 dimensions.
- [ ] Task 5.4: Verify page limit ($\le 6$ pages excluding references), citation formatting (\citet/\citep with BibTeX), and run strict integrity screen check.
- [ ] Task 5.5: Compile `HW2_<TeamID>_report.pdf`.
- [ ] Task 5.6: Commit and push Phase 5.

### Phase 6: Presentation Deck Production (Oral Presentation)
- [ ] Task 6.1: Structure an 8-minute presentation covering Title, Motivation, Problem Formulation, EDA Insight, Method Pipeline, 3-Tier Results, Ablations, Failure Cases, Interpretability, and Takeaways.
- [ ] Task 6.2: Generate professional slide deck (`HW2_<TeamID>_presentation.pptx` and `.pdf`) with clean typography, data charts, and pipeline diagrams.
- [ ] Task 6.3: Commit and push Phase 6.

### Phase 7: Verification & Final Packaging
- [ ] Task 7.1: Verify against all criteria in DM-HW2.pdf and LLM TA Judge rubric.
- [ ] Task 7.2: Generate clean `README.md` with full reproduction instructions.
- [ ] Task 7.3: Package final submission zip `HW2_<TeamID>.zip`.
- [ ] Task 7.4: Final commit and push.
