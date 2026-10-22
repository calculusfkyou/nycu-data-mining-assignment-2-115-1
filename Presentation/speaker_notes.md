# V2 eight-minute rehearsal guide

## 1. Can local text embeddings detect agent failures? (30 seconds)

Task-end failure screening. No generated trajectories or new LLM API calls. Official resolution labels are the target. Source: Dataset/evaluation_results.json. Full author list and student IDs appear in the report.

## 2. The decision is whether a completed run needs review (40 seconds)

Category 4: Text / Logs. Labels do not establish deception, reward hacking or false completion. No prefix prediction or autonomous release has been tested. SWE-bench and SWE-agent sources are in Paper/custom.bib.

## 3. Training-only analysis shows overlapping behavior (35 seconds)

No held-out row enters these boxplots. Numeric control has 23 columns, including 4 binary flags. The test-string flag is patch-wide and the command heuristic is noisy. Source: Paper/figures/v2/eda and v2_stats.tex.

## 4. Local encoders preserve more of the recorded text (45 seconds)

Encoders remain frozen. Log chunks:224 tokens,32 overlap. Patch chunks:480 tokens,64 overlap. Log and patch cosine distances have equal weight. k in {1,3,5}; only successful training rows form the reference. Pooling loses ordering and does not verify correctness.

## 5. Every method shares five outer folds (45 seconds)

Only around 32 successful training and 10 successful validation runs per fold. Empirical validation FPR is not a test guarantee. Shared repositories remain; additional held-out-repository sensitivity is limited. Deterministic methods run once, random methods have seeds42,123,456.

## 6. Numeric controls outperform the embedding detector (55 seconds)

All numbers derive from main_results.csv and bootstrap_ci.csv. Bootstrap resamples paired OOF instances within class2000times; shared training folds limit interpretation. Conditional residual is a secondary comparison. No method meets all operational gates.

## 7. The confusion matrices expose the missed failures (50 seconds)

Rows are actual success/failure. Columns are no alert/alert. Cell text is raw count; color is row fraction. T2 TN40FP14FN179TP51;ECOD TN47FP7FN128TP102;conditional TN44FP10FN107TP123. Prediction absence of alert is not a correctness guarantee.

## 8. Patch vectors help, but dual-channel averaging hurts (50 seconds)

All use kNN and validation model selection. Distances differ across representations, so this comparison includes geometry. Patch embeddings may encode surface similarity rather than functional correctness. Equal averaging is one tested fusion rule, not proof all text encoders fail.

## 9. Contamination does not show a consistent penalty (40 seconds)

Corrected after detecting a mismatched k=3 control in the earlier v2 draft. Fitting IDs and exact replacement fractions are archived. Three replacement seeds. H3 is not established if effects are small or nonmonotonic. Source:contamination_fitting_ids.json and ablation_contamination.csv.

## 10. Near-normal vectors can still describe failed patches (50 seconds)

Full IDs and raw SHA256s are in case_studies.json. Nearest reference log/patch distances respectively .20290/.00655, .04656/.00311, .14414/.00251. The saved leave-one-out values use a fixed nearest reference, not full kNN rescoring; do not call them complete score attribution.

## 11. These detectors cannot justify automatic release (40 seconds)

Gates:recall>=90%,FPR<=20%,unflagged success>=80%,coverage>=5%. Encoder memory reported ~1.19GiB by extractor profiling; not a whole-process-tree memory guarantee. Actual live rehearsal and E3 upload remain outstanding. AI assistance and all five author IDs appear in the report.
