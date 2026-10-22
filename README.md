# Coding-Agent Failure Screening — NYCU Data Mining HW2

Team **G04**. Real SWE-agent GPT-4 trajectories from SWE-bench Lite; no generated trajectories or invented evaluation labels.

This repository studies **task-end screening before integration**, not early/prefix prediction, proven deception detection, or a production release gate. The report follows the supplied HW1 ACL author layout with all five members and student IDs. Member contributions are **pending confirmation**, as requested.

## Deliverables

- `HW2_G04_report.pdf`: official named report, six body pages plus references.
- `HW2_G04_report_anonymized.pdf`: review-mode draft with Team G04 and no member identifiers.
- `HW2_G04_presentation.pptx`: 11 editable slides with speaker notes; rehearsal budgets total 480 seconds. Actual rehearsal is still needed.
- `HW2_G04.zip`: exactly the named report and presentation, with no parent directory.
- `docs/HW2_AUDIT.md`: requirement coverage, corrections, and remaining human actions.
- `docs/verification.json`: latest local validation results, not a grade or submission receipt.

No E3 upload, Judge submission, GitHub publication, live presentation, or production deployment is implied by local artifacts.

## Main findings

Five initialization seeds on the same 162 test runs; sample SD, not cross-validation uncertainty:

| Method | AP | ROC-AUC | F1 | F2 |
|---|---:|---:|---:|---:|
| T1: error/repetition heuristic | 0.962 ± 0.000 | 0.691 ± 0.000 | 0.817 ± 0.000 | 0.751 ± 0.000 |
| T2: normal-only autoencoder | 0.978 ± 0.004 | 0.782 ± 0.024 | 0.945 ± 0.004 | 0.948 ± 0.006 |
| T3: supervised XGBoost | 0.980 ± 0.000 | 0.794 ± 0.000 | 0.940 ± 0.000 | 0.940 ± 0.000 |
| Always alert (context) | 0.926 | 0.500 | 0.962 | 0.984 |

AP means `average_precision_score`, not trapezoidal PR area. Failures are the majority: 230/284 overall and 150/162 in test. AP's test prevalence baseline is already 0.926. T2's mean failure recall is 94.9%, but it flags 9/12 successful test runs (75% FPR). T2 versus T3 AP: paired seed t-test p=0.449; seed-42 paired row bootstrap CI [-0.0147, 0.0206]. This does not prove equivalence.

Full versus patch-only masked inputs gain 0.0143 AP, but patch-only F2 is slightly higher. Replacing 3/32 fitting rows with training failures drops AP by 0.0099. F1 and F2 calibration select identical thresholds across all seeds. Honest negative findings are retained.

## Data and split contract

`Dataset/sweagent_gpt4_features.csv` contains 284 real runs, 24 numeric features and five metadata columns. `evaluation_results.json` archives official generated/resolved labels; `provenance.json` records each raw S3 URL and SHA-256. Raw files are ignored because of size. Source: https://github.com/SWE-bench/experiments/tree/main/evaluation/lite/20240402_sweagent_gpt4

| Partition | Total | Success | Failure | Usage |
|---|---:|---:|---:|---|
| Train | 82 | 32 | 50 | EDA; T3 fit; T2 fits successes only |
| Validation | 40 | 10 | 30 | Threshold selection only |
| Test | 162 | 12 | 150 | Shared evaluation and diagnostic analysis |

IDs in `Dataset/split_ids.json` are fixed. Existing test membership was preserved during protocol repairs; the test was already inspected in an earlier draft, so this is not a pristine once-only evaluation. Shared repositories also limit generalization claims. No resolution, repository ID, or forced-exit indicator enters model inputs. The legacy column `patch_touches_tests` is a broad test-string regex across the whole diff; it is not proof that a changed file is a test.

## Reproduce the numerical pipeline

Run every command from the repository root. Python 3.11 and a CPU are sufficient. The tested environment is recorded in `Model/outputs/environment.json`; `requirements-lock.txt` records installed versions (with a Windows marker for pywinpty); only Windows execution was verified. No GPU, API key, or paid model call is required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python EDA/eda.py
.venv\Scripts\python Model/experiment.py
.venv\Scripts\python Model/ablations.py
.venv\Scripts\python Model/error_analysis.py
.venv\Scripts\python Model/paper_figures.py
.venv\Scripts\python Paper/compact_figures.py
.venv\Scripts\python Paper/generate_tables.py
.venv\Scripts\python Presentation/prepare_content.py
```

If a managed venv has no pip, use `uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt`. On macOS/Linux, replace the interpreter path with `.venv/bin/python`. Measured timing varies by machine and includes framework initialization in training; inference timing is warmed batch throughput, not end-to-end request latency.

Processed data are committed, so downloading is optional for model reproduction. For independent extraction/provenance verification:

```powershell
.venv\Scripts\python tools/fetch_raw.py
.venv\Scripts\python tools/prepare_data.py
```

Fetching checks hashes before saving; changed remote bytes fail rather than silently replace evidence. `prepare_data.py` needs all 284 raw files, retains fixed split IDs, and never manufactures missing rows. Official labels are independently checked by the final verifier. A blocked S3 connection does not prevent using the committed processed data, but full raw verification requires those downloads.

## Rebuild reports and slides

All numerical paper tables are generated from CSVs. Edit `Paper/team.json` to replace contribution placeholders, then regenerate tables. The report source, style files, BibTeX and figures live in `Paper/`.

Install a TeX distribution with `latexmk` or Tectonic, then from `Paper/`:

```powershell
tectonic HW2_G04_report.tex --keep-logs
tectonic HW2_G04_report_anonymized.tex --keep-logs
```

The first Tectonic run needs network access for TeX packages. With a conventional TeX installation, `latexmk -pdf HW2_G04_report.tex` and the analogous anonymous filename also work. Keep ACL margins/font sizes unchanged. Inspect both rendered PDFs after edits; contributions and the AI statement must finish within page six. References begin on a separate page. The abstract is under 150 words.

`Presentation/generate_deck.mjs` uses Codex's bundled Artifact Tool to produce editable PPTX shapes, a native results table, embedded plot images, and notes. Run inside a Codex desktop runtime:

```powershell
node Presentation/generate_deck.mjs
```

Outside the authoring machine, set `HW2_RUNTIME_NODE_MODULES` to the bundled Node modules directory, `HW2_PRESENTATION_SKILL_DIR` to the installed presentation skill directory, and `HW2_RUNTIME_PYTHON` to its Python executable. The builder needs that runtime, not only Python dependencies. The already-exported PPTX can be opened without Codex. Its structural/layout validation receipt is stored in `Presentation/validation.json`; source previews stay in ignored `tmp/slides/`. Rebuilding changes timing-independent file hashes, so copy the latest validation receipt alongside the deck.

## Package and verify

After compiling both PDFs and exporting slides:

```powershell
.venv\Scripts\python tools/package_submission.py
.venv\Scripts\python tools/verify_submission.py
```

The verifier exits nonzero on label/split/hash errors, mismatched metrics, validation-threshold replay errors, mismatched cases/attributions, broken artifact copies, anonymous identifier leaks, page-limit errors, missing slide images/notes, invalid layout receipts, or stale ZIP entries. It performs a fresh normal-only seed-42 fit. Raw checks require `tools/fetch_raw.py` first. Re-running model timing may change timing tables; regenerate report tables and slide content before rebuilding artifacts.

For visual checks with Poppler:

```powershell
pdftoppm -png -scale-to 1600 HW2_G04_report.pdf tmp/report
pdftoppm -png -scale-to 1600 HW2_G04_report_anonymized.pdf tmp/anonymous
```

Review every report page and all slide previews. A passing verifier does not prove a grade, public repository availability, actual presentation timing, or external submission.

## Remaining before submission

1. Confirm each member's contributions in `Paper/team.json`, then rebuild.
2. Team-review the revised findings and rehearse the submitted deck within 8 minutes.
3. Ensure the first-page repository URL is accessible to the TAs and contains the current code/data.
4. One member uploads `HW2_G04.zip` to E3 by **2026-11-16 23:59 Taiwan time**, retaining the receipt. The presentation is **2026-11-17**. Optional Judge upload uses the anonymous PDF.

No Git remote/upstream is configured in this checkout. Local commits do not establish publication of the linked repository.
