# HW2 v2 artifact audit

Updated 2026-10-23, Taiwan. This supersedes the v1 artifact audit; historical numeric outputs remain in Git for provenance.

## Current package

- Named and anonymous ACL PDFs: five body pages and one separate references page, original template margins/fonts, abstract 130 words.
- Named first page includes Team G04, all five supplied names/student IDs and GitHub URL. Anonymous build hides member identities.
- Eleven editable slides with embedded figures and native results table; planned speaker notes total 480 seconds. No actual live rehearsal is claimed.
- HW2_G04.zip contains exactly HW2_G04_report.pdf and HW2_G04_presentation.pptx, using the latest exported bytes.
- All five contributions remain Pending confirmation as explicitly requested.

## Evidence and substantive corrections

The report uses actual v2 OOF predictions over284 officially labeled runs, not the pasted narrative's inconsistent AUC/AP entries. Main decisions and all raw confusion counts were independently replayed. Ranking AP is averaged within each outer fold to avoid treating independently fitted raw-score scales as calibrated across folds. Random-seed variation is distinguished from fold variation and conditional bootstrap intervals.

The original v2 contamination runner used k=3 instead of clean-fold selected k, omitted the frozen-cutoff comparison, and did not reproduce the main detector at zero contamination. Model/contamination_v2.py now freezes clean k/tau, keeps reference size fixed, archives exact fitting/injected IDs and actual fractions, and compares frozen versus validation-recalibrated thresholds. Zero contamination exactly reproduces T2. Effects are small/nonmonotonic; H3 is unsupported.

All three error examples follow saved predictions, raw hashes and excerpts. Nearest-reference channel distances are not full k-neighbor attribution, and the saved fixed-reference step-removal check is not complete score recomputation. Unsupported stories about long debugging or proven semantic defect causes were removed.

No evaluated method meets all four practical screening targets. Failure prevalence is 81%, so the rare-anomaly interpretation remains limited. Reused data, overlapping repositories, few successes, unpinned encoder revisions and incomplete timing evidence are explicit limitations rather than hidden completion claims.

## Verification

- 14 existing focused unit tests pass.
- 1,306 independent current submission checks pass: official labels,284raw hashes/extractions, fold disjointness, normal-only references, all saved model confusion/metric replay, fresh primary validation k/tau and scoring replay, contamination controls and actual IDs, case evidence, PDF identity/page limits/citations, slide evidence/notes/hash/structure and exact ZIP bytes.
- Both PDFs compile without overfull boxes or undefined references. Tectonic reports normal underfull warnings and an anonymous-build lineno package encoding warning; rendered output was inspected.
- All pages of both PDFs and all 11 slides were visually inspected locally. This is not native PowerPoint or physical projection validation.
- Final artifact hashes are recorded in docs/verification.json.

## Remaining human/external items

Confirm member contributions, team-review the findings, rehearse in actual PowerPoint, confirm public repository accessibility, and upload the ZIP to E3 before 2026-11-16 23:59 Taiwan time. Optional Judge submission uses the anonymous PDF. No E3 upload, Judge score or receipt exists from this local artifact update. The checkout has no remote/upstream, so pushing is unavailable.
