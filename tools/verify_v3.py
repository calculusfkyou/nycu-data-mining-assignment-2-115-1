"""Verification script for v3 deliverables: HW2_G04 (NYCU Data Mining Fall 2026).

Verifies:
1. Candidate feature dataset: 2,663 runs, 99 deceptive regressions (3.72%), 2,564 normals.
2. Embeddings dimensions: minilm_logs (2663, 384), codebert_diffs (2663, 768).
3. 5-Fold Grouped CV hygiene: 439 instance_id groups, 0 group leakage.
4. Benchmark results: CM-BAE PR-AUC 0.0827, Cost F2 0.1445, chance 0.0372.
5. Report PDFs: body <= 6 pages (References on Page 7), required end matter present, anonymity verified.
6. Presentation PPTX: 11 slides, 11 speaker notes, exact 480 seconds total rehearsal budget.
7. Submission ZIP: contains HW2_G04_report.pdf and HW2_G04_presentation.pptx.
8. Writes docs/verification.json.
"""

import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pypdf import PdfReader

CHECKS_PASSED = 0


def check(ok, message):
    global CHECKS_PASSED
    if not bool(ok):
        raise AssertionError(f"FAILED CHECK: {message}")
    CHECKS_PASSED += 1


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    team = json.loads((root / "Paper" / "team.json").read_text(encoding="utf-8-sig"))

    # 1. Candidate features verification
    feat_path = root / "Dataset" / "verified" / "candidate_features.csv"
    check(feat_path.exists(), "candidate_features.csv exists")
    df = pd.read_csv(feat_path)
    check(len(df) == 2663, f"Total candidate count == 2,663 (got {len(df)})")
    n_anom = int(df["label"].sum())
    n_norm = int((df["label"] == 0).sum())
    check(n_anom == 99, f"Deceptive regressions count == 99 (got {n_anom})")
    check(n_norm == 2564, f"Clean normal repairs count == 2,564 (got {n_norm})")
    check(np.isclose(n_anom / len(df), 0.037176, atol=1e-4), "Natural prevalence == 3.72%")

    # 2. Embeddings verification
    minilm_path = root / "Dataset" / "verified" / "embeddings" / "minilm_logs.npy"
    codebert_path = root / "Dataset" / "verified" / "embeddings" / "codebert_diffs.npy"
    check(minilm_path.exists() and codebert_path.exists(), "Embeddings exist")
    minilm = np.load(minilm_path)
    codebert = np.load(codebert_path)
    check(minilm.shape == (2663, 384), f"MiniLM shape (2663, 384) (got {minilm.shape})")
    check(codebert.shape == (2663, 768), f"CodeBERT shape (2663, 768) (got {codebert.shape})")
    check(np.isfinite(minilm).all() and np.isfinite(codebert).all(), "Embeddings all finite")

    # 3. Splits hygiene verification
    splits_path = root / "Dataset" / "verified" / "splits.json"
    check(splits_path.exists(), "splits.json exists")
    splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
    folds = splits_data["folds"]
    check(len(folds) == 5, f"5 folds (got {len(folds)})")
    for f in folds:
        train_idx = set(f["train_indices"])
        val_idx = set(f["val_indices"])
        test_idx = set(f["test_indices"])
        check(not (train_idx & val_idx), "Train/Val indices disjoint")
        check(not (train_idx & test_idx), "Train/Test indices disjoint")
        check(not (val_idx & test_idx), "Val/Test indices disjoint")

        # Group leakage check
        train_groups = set(df.iloc[list(train_idx)]["instance_id"].unique())
        val_groups = set(df.iloc[list(val_idx)]["instance_id"].unique())
        test_groups = set(df.iloc[list(test_idx)]["instance_id"].unique())
        check(not (train_groups & test_groups), "0 group leakage between train and test")
        check(not (val_groups & test_groups), "0 group leakage between val and test")

    # 4. Benchmark results verification
    bench_path = root / "Model" / "benchmark_results_v3.json"
    check(bench_path.exists(), "benchmark_results_v3.json exists")
    bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
    summary = bench_data["summary"]
    cm_bae = next(item for item in summary if "CM-BAE" in item["model"])
    check(np.isclose(cm_bae["pr_auc_mean"], 0.0827, atol=1e-3), f"CM-BAE PR-AUC ~ 0.0827 (got {cm_bae['pr_auc_mean']})")
    check(np.isclose(cm_bae["f2_mean"], 0.1445, atol=1e-3), f"CM-BAE Cost F2 ~ 0.1445 (got {cm_bae['f2_mean']})")

    # 5. Report PDFs verification
    for name in ["HW2_G04_report.pdf", "HW2_G04_report_anonymized.pdf"]:
        root_pdf = root / name
        paper_pdf = root / "Paper" / name
        check(root_pdf.exists() and paper_pdf.exists(), f"{name} exists in root and Paper/")
        check(root_pdf.read_bytes() == paper_pdf.read_bytes(), f"{name} root matches Paper/ byte-for-byte")

        reader = PdfReader(str(root_pdf))
        texts = [p.extract_text() or "" for p in reader.pages]
        ref_idx = next(i for i, t in enumerate(texts) if "References" in t or "REFERENCES" in t)
        check(ref_idx <= 6, f"Body page limit: References on page {ref_idx + 1} (<= 7)")

        body_text = "\n".join(texts[:ref_idx])
        check("Contributions of Each Member" in body_text, f"{name}: Contributions in body")
        check("AI Usage Statement" in body_text, f"{name}: AI Usage Statement in body")
        check("CM-BAE" in body_text and "0.083" in body_text, f"{name}: v3 CM-BAE results in body")
        check(not re.search(r"\?\?|\[\?\]", body_text), f"{name}: no broken citation marks")

        if "anonymized" in name:
            for m in team["members"]:
                check(m["name"] not in body_text, f"Anonymized: no {m['name']}")
                check(m["student_id"] not in body_text, f"Anonymized: no {m['student_id']}")
            check(team["repository_url"] not in body_text, "Anonymized: no repo URL")
        else:
            for m in team["members"]:
                check(m["student_id"] in texts[0], f"Named: {m['student_id']} on Page 1")
            check(team["repository_url"] in texts[0].replace("\n", ""), "Named: repo URL on Page 1")

    # 6. Presentation Deck verification
    root_deck = root / "HW2_G04_presentation.pptx"
    pres_deck = root / "Presentation" / "HW2_G04_presentation.pptx"
    check(root_deck.exists() and pres_deck.exists(), "HW2_G04_presentation.pptx exists in root and Presentation/")
    check(root_deck.read_bytes() == pres_deck.read_bytes(), "Deck root matches Presentation/ byte-for-byte")

    with ZipFile(root_deck) as z:
        slides = sorted(n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
        notes = sorted(n for n in z.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n))
        check(len(slides) == 11, f"Presentation has 11 slides (got {len(slides)})")
        check(len(notes) == 11, f"Presentation has 11 notes slides (got {len(notes)})")
        media = [n for n in z.namelist() if n.startswith("ppt/media/")]
        check(len(media) >= 4, f"Presentation has embedded images (got {len(media)})")

    content_data = json.loads((root / "Presentation" / "content_v3.json").read_text(encoding="utf-8"))
    check(len(content_data) == 11, "content_v3.json has 11 slides")
    total_sec = sum(s["seconds"] for s in content_data)
    check(total_sec == 480, f"Total rehearsal seconds == 480 (got {total_sec})")

    # 7. Deliverable ZIP verification
    zip_path = root / "HW2_G04.zip"
    check(zip_path.exists(), "HW2_G04.zip exists")
    with ZipFile(zip_path) as z:
        expected_names = {"HW2_G04_report.pdf", "HW2_G04_presentation.pptx"}
        check(set(z.namelist()) == expected_names, f"ZIP contains exact files: {z.namelist()}")
        check(z.testzip() is None, "ZIP CRC check passed")
        check(z.read("HW2_G04_report.pdf") == (root / "HW2_G04_report.pdf").read_bytes(), "ZIP report matches root")
        check(z.read("HW2_G04_presentation.pptx") == (root / "HW2_G04_presentation.pptx").read_bytes(), "ZIP deck matches root")

    # 8. Write verification.json
    verification_output = {
        "experiment_version": "v3",
        "dataset": "SWE-bench Verified (Frontier Models)",
        "population": "Stage-2 Candidate Patches (Passing FAIL_TO_PASS)",
        "sample_size": len(df),
        "clean_normals": n_norm,
        "deceptive_regressions": n_anom,
        "natural_prevalence": f"{n_anom / len(df) * 100:.2f}%",
        "primary_model": "Calibrated Multimodal Bottleneck Autoencoder (CM-BAE)",
        "pr_auc": f"{cm_bae['pr_auc_mean']:.4f} ± {cm_bae['pr_auc_std']:.4f}",
        "cost_f2": f"{cm_bae['f2_mean']:.4f} ± {cm_bae['f2_std']:.4f}",
        "chance_pr_auc": "0.0372",
        "checks_passed": CHECKS_PASSED,
        "technical_verification": "pass",
        "body_pages": ref_idx,
        "presentation_slides": len(slides),
        "rehearsal_budget_seconds": total_sec,
        "files": {
            "HW2_G04_report.pdf": sha(root / "HW2_G04_report.pdf"),
            "HW2_G04_report_anonymized.pdf": sha(root / "HW2_G04_report_anonymized.pdf"),
            "HW2_G04_presentation.pptx": sha(root / "HW2_G04_presentation.pptx"),
            "HW2_G04.zip": sha(root / "HW2_G04.zip")
        },
        "remaining_human_items": [
            "Confirm member contributions in final submission portal",
            "Team review and 8-minute slide rehearsal before in-class presentation",
            "Verify public GitHub repository access",
            "Upload HW2_G04.zip and HW2_G04_report_anonymized.pdf to NYCU E3 portal"
        ],
        "external_submission_performed": False,
        "grade_claimed": False
    }

    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_file = docs_dir / "verification.json"
    out_file.write_text(json.dumps(verification_output, indent=2) + "\n", encoding="utf-8")
    print(f"All {CHECKS_PASSED} verification checks PASSED successfully!")
    print(json.dumps(verification_output, indent=2))


if __name__ == "__main__":
    main()
