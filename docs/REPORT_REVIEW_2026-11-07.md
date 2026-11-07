# Report evidence and layout review

Reviewed saved repository evidence; did not independently repeat upstream Docker
evaluations or retrain models. This is not a fresh external scientific validation.

| Original claim | Evidence and correction |
| --- | --- |
| 2,663 candidates, 99 regressions, 3.72% | Consistent with saved candidate features and benchmark population. Conditional run-level prevalence, not industry prevalence. |
| Agent wrote and passed its own reproduction tests | Labels are benchmark F2P outcomes. They do not establish agent authorship or local execution of those tests. |
| 11 structural features | Implementation uses 12 structural features. |
| Never sees any anomalous labels | Autoencoder and preprocessing fit normal training runs; calibration and threshold selection use labelled validation regressions. |
| Cross-project grouped split | Split key is instance_id. Issue-held-out, not repository-held-out. |
| Beats traditional methods by 50% and supervised models | Contradicted by benchmark_results_v3.json: CM-BAE AP 0.0827; churn 0.1919; IF 0.1409; ECOD 0.1316; XGBoost 0.0964. It beats dual k-NN and MLP in mean AP. |
| Strong significance | p=0.033628 is an unadjusted paired comparison of five seed-42 fold APs against the all-normal baseline. Overlapping training folds and multiple comparisons limit inference. |
| Low false-alarm burden | CM-BAE FPR 0.8332 and recall 0.8286 at its saved cost-selected threshold. Several folds flag all clean candidates. |
| Equalization and multimodal synergy work | Ablation AP: full 0.0827, structural-only 0.1330, raw fusion 0.1233. H1/H2 unsupported. |
| Contamination proves robustness | Nonmonotonic AP near 0.041, with a clean endpoint inconsistent with the main benchmark. Comparability remains unresolved; H3 unsupported. |
| $13,800 realized savings per 1,000 PRs | Costs are hypothetical. Saved single-fold budget curve gives $6,840 at 25% review and negative savings at 1–3%; no measured production savings. Review-all is a necessary comparator. |
| 97.4% untested means the root cause | Command/path matching is a proxy, cannot measure actual coverage or establish causation. Both classes have median UER 1.0. |
| 0.384 AP in main-table caption | Stale number conflicts with saved results; current layout tables are generated from JSON. |
| All contributions allocated | team.json says pending confirmation; restored that status rather than invent allocations. |

## Layout work

- Rendered the original PDF and confirmed cross-column formulas, overflowing
  column tables, and small figure labels.
- Rebuilt the narrative as six deliberate pages: framing, data/EDA, method,
  benchmark, ablations/economics, and error analysis/conclusion/statements.
- Replaced long network formulas with explicit layer widths and short equations.
- Generated full-width tables from saved metrics and enlarged vector figures.
- Distinguished the diagnostic figure's two-fold curves from the five-fold table.
- Clarified that the attribution chart's percentile line is not the deployment
  probability threshold.
- Used matching anonymous/named ACL preprint layouts. Review line numbering was
  incompatible with balanced page panels and caused text outside the page.
- Rendered and visually inspected all seven pages in both final variants.

## Remaining work

Presentation claims were not revised in this PDF task. Audit slides and speaker
notes against corrected report before submission. Member allocations and external
submission remain unconfirmed. Passing local checks does not resolve these items.
