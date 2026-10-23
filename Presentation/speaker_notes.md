# V2 eight-minute rehearsal guide

## 1. Can local text embeddings detect agent failures? (30 seconds)

Task-end failure screening on SWE-bench Lite. Zero new LLM API calls. Official execution verdicts define ground truth. Full author list and student IDs appear in the report.

## 2. The decision is whether a completed run needs review (40 seconds)

Category 4: Text / Logs. Task-end screening matches PR integration queues. We also isolate the rare 6.0% silent semantic failure subtype.

## 3. Training-only analysis shows overlapping behavior (35 seconds)

No held-out row enters these boxplots or Wasserstein/KS statistics. Source: Paper/figures/v2/eda and v2_stats.tex.

## 4. Six-stage primary pipeline with frozen encoders (45 seconds)

Encoders remain frozen. Validation selects k and threshold tau under empirical FPR <= 20% maximizing recall.

## 5. Strict 5-fold CV plus untouched confirmatory holdout (45 seconds)

Addresses prior test inspection on GPT-4 by evaluating frozen models once on the untouched 271 Claude-3-Opus holdout.

## 6. Numeric controls outperform the embedding detector (55 seconds)

All numbers derive from main_results.csv, bootstrap_ci.csv, and confirmatory_claude3opus_oof.csv.

## 7. The confusion matrices expose the missed failures (50 seconds)

T2 TN 40 FP 14 FN 179 TP 51; ECOD TN 47 FP 7 FN 128 TP 102; Conditional TN 44 FP 10 FN 107 TP 123.

## 8. Mechanistic ablation: log variance drowns out patch signal (50 seconds)

Checked directly in embedding distance data: d_log variance is 56x larger than d_patch variance.

## 9. Contamination & threshold policy ablations (40 seconds)

Threshold ablation table compares FPR<=20%, max-BA, max-F1, and max-F2/min-cost rules.

## 10. Error case studies and additive feature attribution (50 seconds)

Full IDs, SHA-256s, raw feature values, and repository overlap counts are reported in Table 6 and Figure 5.

## 11. Takeaways, deployment limits, and next steps (40 seconds)

Gates: recall>=90%, FPR<=20%, unflagged success>=80%, coverage>=5%. AI assistance and all five author IDs appear in the report.
