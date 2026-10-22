# HW2 audit and submission boundaries

Audit date: 2026-10-23 (Taiwan). Team G04 metadata follows the supplied HW1 author block. Member contributions intentionally remain pending confirmation at the user's request.

## Requirement coverage

| Assignment item | Current evidence | Boundary |
|---|---|---|
| Real data, documented labels | 284 raw trajectory SHA-256s and URLs, official generated/resolved JSON, 24-feature CSV | Does not establish actual deception/false completion intent |
| Explicit category, stakeholder, action | Report §§1–3: Text/Logs; release engineer; additional review | Task-end screening; no early prediction or autonomous release |
| Three comparable tiers | Same train/val/test IDs and test metrics; normal-only T2, mixed-train T3 | Shared repositories; earlier draft already inspected test |
| Train-only EDA | 82 training rows; KS, standardized Wasserstein, MI, Spearman, distribution plots | Uncorrected descriptive p-values; no confirmatory claim |
| Scoring and six-stage pipeline | Normal scaler → 24-16-6-16-24 AE → MSE → validation threshold → alert | F2 is not a dollar-optimal threshold |
| Five-seed results and uncertainty | Per-seed predictions, mean/sample SD, paired seed t-tests and row bootstrap | Optimization seeds on one split are not independent datasets |
| Controlled AD-specific ablation | Fixed-size contamination using training failures, exact 0/3.125/6.25/9.375% | Clean scaler held fixed; isolates weight contamination only |
| Second controlled ablation | Zero feature masks; fixed 24-dimensional network | Inactive columns can still carry reconstruction penalty; interpret as this masking protocol |
| Third ablation and hypotheses | Same scores, threshold rule F1/F2/Normal95; explicit H1/H2/H3 verdicts | H3 unsupported: F1 and F2 give identical cutoffs |
| Three errors and attribution | Real IDs, raw counts, score, threshold, signed distance, excerpts; additive per-feature MSE | Aggregate features cannot prove semantic root causes |
| Operational impact | Every seed's confusion matrix drives one explicit cost scenario; all alerts charged | Illustrative assumptions; review-all is cheaper under them |
| Paper and bibliography | Six body pages, references on separate page; ≥3 verified primary sources, BibTeX | No score or bonus is certified |
| Front matter and anonymity | Team G04, five names/IDs, repository URL on page one; separate review-mode anonymous PDF | Public repository access still needs confirmation |
| Contributions and AI disclosure | Five explicit pending entries; actual Codex audit disclosed | Member work cannot be inferred or fabricated |
| Presentation | 11 editable slides, embedded figures, native results table, 480-second speaker budgets | Actual PowerPoint/device rendering, live timing and Q&A untested |
| Exact E3 package | HW2_G04.zip contains only HW2_G04_report.pdf + HW2_G04_presentation.pptx | No external upload or receipt |

## Material corrections

The original draft was not ready for a high-confidence submission:

- It named Team G07 and four unrelated members. These were replaced using the user-supplied G04 metadata; no contributions were invented.
- The original paper described sample IDs and statistics that contradicted the actual CSVs, including an unresolved case called normal. Current tables/cases derive from actual data and model outputs.
- Tier 3 fitted on validation rows before using the same rows for calibration. Fitting and validation are now disjoint.
- Contamination drew from validation, misnamed a one-row injection as 1%, and changed sample size. Current replacements use only training rows, fixed total size and exact fractions.
- EDA used training + validation despite the assignment's train-only rule. It now uses only the mixed training partition.
- Tool frequency was mislabeled as the maximum consecutive streak. Extraction now computes a true streak.
- A forced-termination indicator was included in “static patch” features. It was replaced with patch hunk count.
- Feature ablation changed network dimensions. The revised ablation preserves the 24-dimensional network and masks inputs after scaling.
- Error-analysis stories included unsupported claims about syntax recovery, aborts and empty patches. Current interpretations separate recorded observations from hypotheses.
- Operational impact used hard-coded catch/FPR values inconsistent with predictions, confused F1/F2 with recall, and omitted true-positive review costs. It now derives all counts from predictions and includes a review-all baseline.
- Unsupported/fabricated ATFD reference entries, benchmark metrics and industry cost claims were removed. Primary SWE-bench, SWE-agent, Deep SVDD and XGBoost sources were checked.
- Claims that full trajectories forecast before submission or prove a 70% reduction in production escapes were removed. The dataset supports completed-run screening only.
- Small four-panel figures were illegible in one ACL column. Separate report figures are now drawn at final printed size; slides embed image bytes rather than unresolved file references.

## Reproduction and verification

See the root README for complete commands. `tools/verify_submission.py` independently checks data/provenance, official labels, exact split membership, train-only EDA statistics, per-seed metric replay, threshold replay, actual contamination fitting IDs, a fresh seed-42 normal-only model, error cases, additive attribution, all-alert cost accounting, report copies/page limits/anonymity, slide structure/notes/embedded images, the finalizer hash receipt, and exact ZIP contents. `docs/verification.json` records the latest count and remaining human items. Checks fail with a nonzero exit code.

The local raw cache was checked against committed hashes; official evaluation metadata was fetched and cross-checked. No original SWE-bench Docker evaluation was rerun. Report and slide PNGs were inspected locally; native PowerPoint execution and a physical presentation were not checked.

## Remaining human actions

- Fill confirmed contributions in `Paper/team.json`, regenerate tables, and rebuild the artifacts.
- Team-review the conclusions, rehearse within eight minutes, and prepare Q&A on class imbalance, false positives, holdout reuse, and review-all costs.
- Confirm the linked repository is accessible to graders and contains this revision. This checkout has no Git remote or upstream; local commits cannot establish publication.
- Submit the ZIP to E3 before 2026-11-16 23:59 Taiwan time and retain the receipt. Optional Judge submission uses the anonymous PDF. Neither submission was performed here.

The practical ceiling is evidence, not a promised score. The user-authorized contribution placeholders and actual presentation/submission remain external completion items.

## Final local verification

The final package passed 1,190 checks. Both PDFs compiled without undefined-reference or overfull-box warnings; six body pages precede a separate references page. All 11 slide previews and report pages were inspected locally. An independent read-only review found no additional material fitting leakage or reported-metric contradiction, and confirmed all 284 raw-to-feature rows. Its two findings (patch-wide test-string semantics and a Windows-only dependency marker) were corrected. This is local evidence, not a grading or upload receipt.
