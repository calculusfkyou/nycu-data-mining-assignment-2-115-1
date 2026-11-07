# Coding-Agent Failure Screening - NYCU Data Mining HW2

## Current v3 report revision (2026-11-07)

The PDFs now use `Paper/v3_body.tex`. Both variants contain six body pages and
references on page 7. Layout panels are generated directly from saved v3 JSON
results; no model rerun is required:

```powershell
.venv/Scripts/python.exe Paper/build_report_layout.py
tectonic Paper/HW2_G04_report.tex --keep-logs
tectonic Paper/HW2_G04_report_anonymized.tex --keep-logs
Copy-Item Paper/HW2_G04_report*.pdf .
.venv/Scripts/python.exe tools/package_submission.py
.venv/Scripts/python.exe tools/verify_v3.py
```

The corrected report does **not** claim that CM-BAE beats the strongest controls,
generalizes to held-out repositories, or provides demonstrated production savings.
Its AP is 0.083 and its cost-selected FPR is 83.3%. See
`docs/REPORT_REVIEW_2026-11-07.md` for the evidence audit. Slides are unchanged in
this report-layout task and still need a separate claim-consistency review before
submission. The ZIP is synchronized technically, not certified ready to submit.

The remaining sections below describe the historical v2 workflow and results.

Team **G04**. Current report, slides and submission ZIP use the **v2 local NLP experiments**. Five members and IDs follow the supplied HW1 author block; all contributions remain **pending confirmation** as requested.

## Current deliverables

- `HW2_G04_report.pdf`: named ACL report, six body pages plus a separate references page.
- `HW2_G04_report_anonymized.pdf`: Team G04 review build without member identities.
- `HW2_G04_presentation.pptx`: 11 editable slides with embedded figures and 480-second speaker budgets.
- `HW2_G04.zip`: exactly the named report and presentation, no parent folder.
- `docs/verification.json`: current local checks and final file hashes; no upload or grade receipt.

## What the experiment found

We screen completed agent runs before integration, using official issue-resolution labels. All 284 genuine SWE-agent GPT-4 trajectories are retained (54 successes,230 failures). Failures are common (81.0%), so this is normal-reference failure screening with a limited rare-anomaly interpretation. No generative LLM calls or new issue-solving runs are required.

MiniLM log vectors and CodeBERT patch vectors are computed locally with frozen weights. Dual-channel normal-reference kNN does not beat numeric controls. No evaluated model meets all practical targets for recall, false alarms, unflagged success and coverage. No automatic release is justified.

| Method | BA | FPR | Failure recall | TN/FP/FN/TP | Mean fold AP |
|---|---:|---:|---:|---|---:|
| Always alert | 0.500 | 1.000 | 1.000 | 0/54/0/230 | 0.810 |
| Never alert | 0.500 | 0.000 | 0.000 | 54/0/230/0 | 0.810 |
| T1: heuristic | 0.526 | 0.204 | 0.257 | 43/11/171/59 | 0.864 |
| T2: dual encoder kNN | 0.481 | 0.259 | 0.222 | 40/14/179/51 | 0.841 |
| T3: encoder logistic | 0.596 | 0.352 | 0.543 | 35/19/105/125 | 0.905 |
| TF-IDF logistic | 0.652 | 0.352 | 0.657 | 35/19/79/151 | 0.923 |
| Inductive ECOD variant | 0.657 | 0.130 | 0.443 | 47/7/128/102 | 0.913 |
| Isolation Forest | 0.635 | 0.148 | 0.417 | 46/8/134/96 | 0.914 |
| Conditional residual | 0.675 | 0.185 | 0.535 | 44/10/107/123 | 0.908 |
| Denoising AE | 0.648 | 0.148 | 0.443 | 46/8/128/102 | 0.909 |
| Numeric AE (v2 control) | 0.651 | 0.185 | 0.487 | 44/10/118/112 | 0.904 |
| XGBoost (v2 control) | 0.646 | 0.278 | 0.570 | 39/15/99/131 | 0.908 |

Operating-point metrics pool the 284 out-of-fold decisions at seed42. AP is computed within each fitted fold and averaged: raw scores from different fitted folds need not share a calibrated scale. Saved pooled raw AP/AUC are descriptive. AP prevalence baseline is about 0.810. Random numeric models also run seeds 123/456; deterministic models are not repeated as independent evidence.

## Protocol and corrections

Five stratified outer folds share identical IDs for all methods; inner validation gives approximately 60/20/20 fitting/validation/test. Normal-reference fitting uses only training successes. TF-IDF vocabulary/IDF never sees held-out text. Validation selects models and cutoffs under empirical FPR<=20%; it never fits weights. The data were seen in prior drafts, so v2 remains exploratory, not a fresh independent test. Repositories overlap.

The original v2 contamination experiment accidentally fixed k=3 instead of the clean selected k and omitted the frozen-threshold control. `Model/contamination_v2.py` repairs it using each clean fold k/tau, fixed reference sizes, nested replacements and archived fitting IDs. Zero contamination reproduces T2 exactly. Both clean and recalibrated cutoffs show small nonmonotonic effects; H3 is unsupported. `Model/ablations_v2.py` now invokes that corrected runner.

Nearest-reference channel distances are similarity explanations, not patch-correctness proofs. Saved step-removal diagnostics hold one reference fixed; they do not recompute the complete kNN score. Keyword masking/task-context models were not evaluated. Encoders were not originally revision-pinned, and timing is aggregate profiling rather than full per-method deployment benchmarking.

## Reproduce v2

Run from the repository root using the existing tested `.venv` (Python 3.11). On a new machine create a venv and install `requirements-lock.txt`; the encoder packages are included there. Raw trajectories are needed for text extraction/TF-IDF; restore their archived hashes with `tools/fetch_raw.py`. Encoder extraction downloads public model weights on first use and then runs locally.

```powershell
.venv/Scripts/python.exe tools/fetch_raw.py
.venv/Scripts/python.exe tools/audit_trajectories.py
.venv/Scripts/python.exe tools/generate_v2_manifests.py
.venv/Scripts/python.exe tools/build_text_embeddings.py
.venv/Scripts/python.exe Model/experiment_v2.py
.venv/Scripts/python.exe Model/ablations_v2.py
.venv/Scripts/python.exe Model/error_analysis_v2.py
.venv/Scripts/python.exe Model/verify_v2.py
.venv/Scripts/python.exe -m pytest tests -q
```

Existing embedding caches permit downstream evaluation without re-encoding. CPU extraction was profiled at about895 seconds for284 runs; actual timing varies. Raw source URLs and hashes are in `Dataset/provenance.json`; v2 partitions and audits are in `Dataset/v2/`. The current spec is `docs/EXPERIMENT_SPEC_V2.md`; the paper discloses implemented differences from historical models and diagnostic limitations.

## Rebuild submission artifacts

`Paper/build_v2_artifacts.py` generates current tables, final-size figures and slide content from saved v2 evidence. `Presentation/prepare_content.py` invokes it, so normal slide rebuilds cannot restore v1 content. `Paper/v2_body.tex` holds the report narrative. The shared named/anonymous front matter remains in `Paper/HW2_G04_report.tex`.

```powershell
.venv/Scripts/python.exe Paper/build_v2_artifacts.py
Set-Location Paper
../tmp/tectonic/tectonic.exe HW2_G04_report.tex --keep-logs
../tmp/tectonic/tectonic.exe HW2_G04_report_anonymized.tex --keep-logs
Set-Location ..
node Presentation/generate_deck.mjs
.venv/Scripts/python.exe tools/package_submission.py
.venv/Scripts/python.exe tools/verify_submission.py
```

Use an installed Tectonic/latexmk if the local ignored `tmp/tectonic` executable is absent. The JavaScript slide builder uses Codex Artifact Tool runtime dependencies; `HW2_RUNTIME_NODE_MODULES`, `HW2_PRESENTATION_SKILL_DIR` and `HW2_RUNTIME_PYTHON` override authoring-machine paths. Opening the exported PPTX needs no authoring runtime. Windows finalizer hard-link creation may need permission. Rebuilding artifacts changes hashes and requires packaging and verification again.

`Paper/generate_tables.py` and the original experiment scripts are v1 historical utilities and do not generate current v2 findings. Original results remain available for audit; do not mix v1 statistics with v2 reports. The broad historical figure suite in `Model/figures/v2` is diagnostic; current submission figures are in `Paper/figures/v2`.

## Remaining external steps

Confirm member contributions, team-review the report, rehearse in actual PowerPoint, confirm public GitHub accessibility, and upload the ZIP to E3 before 2026-11-16 23:59 Taiwan time. Optional Judge submission uses the anonymous PDF. No upload, Judge score or rehearsal has been performed by generating these artifacts. This checkout has no remote/upstream, so local commits are not publication evidence.
