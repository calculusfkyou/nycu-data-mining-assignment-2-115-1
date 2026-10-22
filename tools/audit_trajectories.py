"""Audit SWE-agent trajectory JSON files and verify observation timelines.

Audit requirements from HW2 v2 specification:
1. Check at least 10 resolved and 10 unresolved trajectories (we audit all 284).
2. Verify JSON fields: trajectory[].action, trajectory[].observation, info.submission.
3. Verify data timeline: ensure observations are produced by bash tools during execution,
   and NOT injected by the post-hoc SWE-bench evaluator.
4. Verify no evaluation leakage: resolved, gold patch, gold tests, official grader outputs.
5. Check SHA-256 checksums against provenance.json.
6. Output detailed audit findings to Dataset/v2/audit_report.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


RAW_TRAJ_DIR = Path("Dataset/raw/sweagent_gpt4")
PROVENANCE_PATH = Path("Dataset/provenance.json")
FEATURES_CSV = Path("Dataset/sweagent_gpt4_features.csv")
AUDIT_OUTPUT_PATH = Path("Dataset/v2/audit_report.json")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_single_trajectory(
    filepath: Path,
    instance_id: str,
    is_resolved: bool,
    expected_sha: str,
) -> Dict[str, Any]:
    actual_sha = compute_sha256(filepath)
    sha_match = actual_sha == expected_sha

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    top_keys = list(data.keys())
    has_trajectory = "trajectory" in data and isinstance(data["trajectory"], list)
    has_info = "info" in data and isinstance(data["info"], dict)

    traj = data.get("trajectory", [])
    info = data.get("info", {})

    exit_status = info.get("exit_status")
    submission = info.get("submission")
    has_submission = submission is not None and len(str(submission).strip()) > 0

    # Examine steps
    num_steps = len(traj)
    step_keys_set = set()
    has_actions = True
    has_observations = True
    evaluator_leak_suspects = []

    # Check for keywords that would indicate post-hoc evaluation leakage
    leakage_keywords = [
        "swebench.harness",
        "official_evaluation_result",
        "PASS_TO_PASS",
        "FAIL_TO_PASS",
        "gold_patch",
    ]

    for idx, step in enumerate(traj):
        step_keys_set.update(step.keys())
        action = str(step.get("action", ""))
        obs = str(step.get("observation", ""))

        if "action" not in step:
            has_actions = False
        if "observation" not in step:
            has_observations = False

        for kw in leakage_keywords:
            if kw.lower() in obs.lower():
                evaluator_leak_suspects.append({"step": idx, "keyword": kw, "context": obs[:150]})

    return {
        "instance_id": instance_id,
        "is_resolved": is_resolved,
        "label": 0 if is_resolved else 1,
        "file_exists": True,
        "sha256_match": sha_match,
        "actual_sha256": actual_sha,
        "top_keys": top_keys,
        "num_steps": num_steps,
        "step_keys": sorted(list(step_keys_set)),
        "exit_status": exit_status,
        "has_submission": has_submission,
        "submission_length": len(submission) if submission else 0,
        "has_actions": has_actions,
        "has_observations": has_observations,
        "evaluator_leak_suspects": evaluator_leak_suspects,
    }


def run_audit() -> Dict[str, Any]:
    print("Starting full trajectory audit...")
    AUDIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov_map = {item["instance_id"]: item["sha256"] for item in prov["trajectories"]}

    features_df = pd.read_csv(FEATURES_CSV)
    label_map = {row["instance_id"]: bool(row["is_resolved"]) for _, row in features_df.iterrows()}

    resolved_audited: List[Dict[str, Any]] = []
    unresolved_audited: List[Dict[str, Any]] = []
    all_audited: List[Dict[str, Any]] = []

    sha_mismatches = []
    missing_files = []
    leakage_detections = []

    for instance_id, expected_sha in prov_map.items():
        traj_file = RAW_TRAJ_DIR / f"{instance_id}.traj"
        if not traj_file.exists():
            missing_files.append(instance_id)
            continue

        is_resolved = label_map.get(instance_id, False)
        record = audit_single_trajectory(traj_file, instance_id, is_resolved, expected_sha)
        all_audited.append(record)

        if not record["sha256_match"]:
            sha_mismatches.append(instance_id)

        if record["evaluator_leak_suspects"]:
            leakage_detections.append({
                "instance_id": instance_id,
                "suspects": record["evaluator_leak_suspects"]
            })

        if is_resolved:
            resolved_audited.append(record)
        else:
            unresolved_audited.append(record)

    summary = {
        "total_instances_expected": len(prov_map),
        "total_instances_audited": len(all_audited),
        "resolved_count": len(resolved_audited),
        "unresolved_count": len(unresolved_audited),
        "sha256_all_matched": len(sha_mismatches) == 0,
        "sha256_mismatch_count": len(sha_mismatches),
        "missing_file_count": len(missing_files),
        "evaluator_leakage_found": len(leakage_detections) > 0,
        "leakage_detections": leakage_detections,
        "top_level_keys_consistent": all(set(r["top_keys"]) == {"environment", "trajectory", "history", "info"} for r in all_audited),
        "step_keys_consistent": all({"action", "observation"}.issubset(set(r["step_keys"])) for r in all_audited),
        "detailed_sample_10_resolved": resolved_audited[:10],
        "detailed_sample_10_unresolved": unresolved_audited[:10],
    }

    with open(AUDIT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Audit completed: {len(all_audited)} trajectories audited.")
    print(f"  - Resolved: {len(resolved_audited)}, Unresolved: {len(unresolved_audited)}")
    print(f"  - SHA-256 match: {summary['sha256_all_matched']}")
    print(f"  - Evaluator leakage found: {summary['evaluator_leakage_found']}")
    print(f"  - Report saved to: {AUDIT_OUTPUT_PATH}")

    return summary


if __name__ == "__main__":
    run_audit()
