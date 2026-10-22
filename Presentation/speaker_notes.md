# Eight-minute rehearsal guide

## 1. Screening coding-agent task failures (25 s)

Open with the release engineer deciding which completed patch needs extra inspection. This is task-end screening, not early prediction. Repository: https://github.com/calculusfkyou/nycu-data-mining-assignment-2-115-1

## 2. Which completed patches need extra review? (40 s)

The detector is an experimental triage signal. Failure labels do not establish deceptive intent or an explicit false completion claim. Source: SWE-bench ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html

## 3. Failures are common in this dataset (40 s)

Of 300 Lite tasks, 16 have no generation; 284 generated trajectories are retained. 190 voluntary submissions include 140 failures; 94 budget-terminated submissions include 90 failures. Source: Dataset/evaluation_results.json and provenance.json. The application deviates from classical rare-event AD and this is a limitation.

## 4. Training logs suggest workflow signals (40 s)

W is standardized Wasserstein distance. Error patterns can be expected reproduction attempts; do not equate observed tracebacks with final failure. Summary statistics and Spearman matrix are computed only on training rows. Source: EDA/tables/feature_separability.csv.

## 5. Normal-only reconstruction gives an alert score (50 s)

Explain the six stages. AdamW lr .005, weight decay 1e-4, 150 epochs, batch 16. F2 weights false negatives more but is not the dollar-optimal objective. All features require the completed trajectory and final patch. Source: Model/models.py and Model/tables/all_seed_runs.csv.

## 6. Three tiers share disjoint fitting and calibration data (40 s)

The original test IDs were preserved during repairs. Fifty development failures moved to training. Validation never enters weight fitting. Test reuse is disclosed; these results require new runs for confirmation. No architecture search was performed. XGBoost defaults are deterministic across seeds. Sources: split_ids.json, environment.json. XGBoost: https://doi.org/10.1145/2939672.2939785

## 7. High AP does not establish a useful release gate (55 s)

Timing is median warmed batch scoring divided by 162, not single-request end-to-end latency. Train seconds: T1 essentially zero, T2 .641 ± .866, T3 .013 ± .001. Paired row bootstrap AP gap T2 minus T3: [-.0147,.0206], seed 42 model, 1000 resamples. Seeds characterize optimization variation on one split. Source: main_results_table.csv, significance_tests.csv, practical_impact.csv.

## 8. Ablations isolate three different decisions (55 s)

Feature masks preserve all 24 coordinates and network shape. Contamination replaces 0,1,2,3 rows for 0,3.125,6.25,9.375%, holding sample size and clean scaler fixed. The 9.375% AP drop exceeds baseline seed SD .0039 by 2.5x. A Normal95 validation threshold misses 144 of 150 failures with zero false alarms. Sources: ablation_runs.csv and summary tables. H1 ranking supported; H2 supported here; H3 not supported.

## 9. Noisy successes and quiet failures overlap (50 s)

Full IDs: scikit-learn__scikit-learn-14894, django__django-14155, sympy__sympy-15345. Scores minus tau: +1.3916,-.0065,-.1128. Patches add 8,15,2 lines respectively and all end in submit; do not claim aborts or empty patches. The separate flagged django__django-11630 has s=3.8673 and 14 stalled observations contributing 85.04% of reconstruction error; this supports trace inspection, not a causal verdict. Sources: failure_cases.csv, case_excerpts.json, feature_attribution_flagged_sample.csv.

## 10. Under these assumed costs, reviewing all wins (45 s)

These are conditional projections, not measured production savings. T2 consumes 227.4 review hours vs T1 163.5. Perfect review is a strong scenario assumption; real reviewer sensitivity and capacity are unknown. At low deployment prevalence, precision changes. Source: practical_impact.csv. Contrast with prior work: SWE-agent solves issues; Deep SVDD uses one-class compactness; our work audits task-end trace screening. Sources: https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html and https://proceedings.mlr.press/v80/ruff18a.html

## 11. Workflow signals help ranking; deployment remains unproven (40 s)

Team G04: Tun-Chieh Yang (315553010), Jia-Cheng Wu (314554059), Min-Quan Wang (315553008), Wei-Chia Chang (315553011), Chun-Ting Hou (315554022). Each member contribution is pending confirmation at the user’s request. Earlier materials mention Claude/Gemini but versions and scope remain unconfirmed. Do not imply team verification has already occurred. Full reproduction and source links are in README. Close with the research question: full workflow features improve AP over patch-only, but the alert trade-off is not a proven operational win.
