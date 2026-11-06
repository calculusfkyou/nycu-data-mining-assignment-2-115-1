"""Fetch official SWE-bench Verified execution reports and trajectories from public S3.

Audits all complete 500/500 mini-SWE-agent v2.0.0 frontier runs + external holdout:
- 20260217_mini-v2.0.0_claude-4-6-opus
- 20260217_mini-v2.0.0_claude-4-5-opus-high
- 20260217_mini-v2.0.0_claude-4-5-sonnet-high
- 20260217_mini-v2.0.0_gpt-5-2-high
- 20260217_mini-v2.0.0_gemini-3-flash-high
- 20260226_mini-v2.0.0_gemini-3-pro-high
- 20260217_mini-v2.0.0_deepseek-3-2-high
- External holdout: 20250522_sweagent_claude-4-sonnet-20250514
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com"
OUTPUT_DIR = Path("Dataset/verified")
RAW_DIR = OUTPUT_DIR / "raw"

TARGET_RUNS = [
    ("bash-only", "20260217_mini-v2.0.0_claude-4-6-opus", "Claude-4.6-Opus"),
    ("bash-only", "20260217_mini-v2.0.0_claude-4-5-opus-high", "Claude-4.5-Opus"),
    ("bash-only", "20260217_mini-v2.0.0_claude-4-5-sonnet-high", "Claude-4.5-Sonnet"),
    ("bash-only", "20260217_mini-v2.0.0_gpt-5-2-high", "GPT-5.2"),
    ("bash-only", "20260217_mini-v2.0.0_gemini-3-flash-high", "Gemini-3-Flash"),
    ("bash-only", "20260226_mini-v2.0.0_gemini-3-pro-high", "Gemini-3-Pro"),
    ("bash-only", "20260217_mini-v2.0.0_deepseek-3-2-high", "DeepSeek-V3.2"),
    ("verified", "20250522_sweagent_claude-4-sonnet-20250514", "SWEAgent-Claude-4-Sonnet-Holdout"),
]


def http_get(url: str, timeout: int = 15, retries: int = 3) -> Optional[bytes]:
    """Fetch URL with retries and exponential backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (2 ** attempt))
    return None


def fetch_run_instance_list(prefix: str, run_id: str) -> List[str]:
    """Fetch the 500 instance IDs evaluated for this run from GitHub or S3."""
    gh_url = f"https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified/{run_id}/per_instance_details.json"
    content = http_get(gh_url)
    if content:
        data = json.loads(content.decode("utf-8"))
        return sorted(list(data.keys()))
    
    # Fallback to standard SWE-bench verified task list if per_instance_details is not in GitHub
    split_url = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified/20260217_mini-v2.0.0_claude-4-6-opus/per_instance_details.json"
    content_fb = http_get(split_url)
    if content_fb:
        data_fb = json.loads(content_fb.decode("utf-8"))
        return sorted(list(data_fb.keys()))
    raise RuntimeError("Failed to obtain instance list.")


def process_instance_report(prefix: str, run_id: str, model_name: str, instance_id: str) -> Optional[Dict[str, Any]]:
    """Fetch and parse report.json for one instance."""
    rep_url = f"{S3_BASE}/{prefix}/{run_id}/logs/{instance_id}/report.json"
    rep_bytes = http_get(rep_url)
    if not rep_bytes:
        return None
    
    try:
        rep_raw = json.loads(rep_bytes.decode("utf-8"))
    except Exception:
        return None
    
    # Handle {instance_id: {...}} wrapper or direct {...}
    rep = rep_raw.get(instance_id, rep_raw)
    resolved = bool(rep.get("resolved", False))
    applied = bool(rep.get("patch_successfully_applied", True))
    
    ts = rep.get("tests_status", {})
    f2p = ts.get("FAIL_TO_PASS", {})
    p2p = ts.get("PASS_TO_PASS", {})
    
    f2p_succ = len(f2p.get("success", []))
    f2p_fail = len(f2p.get("failure", []))
    p2p_succ = len(p2p.get("success", []))
    p2p_fail = len(p2p.get("failure", []))
    
    # In SWE-bench:
    # Clean resolve: resolved == True AND p2p_fail == 0
    # Candidate patch: passes target reproduction tests (resolved == True OR (f2p_succ > 0 and f2p_fail == 0))
    # Deceptive regression: passes target reproduction tests BUT collaterally breaks existing regression tests (p2p_fail > 0)
    is_candidate = resolved or (f2p_succ > 0 and f2p_fail == 0)
    is_deceptive_regression = is_candidate and (p2p_fail > 0)
    is_clean_normal = is_candidate and (p2p_fail == 0) and resolved

    # Trajectory URL and patch URL
    traj_url = f"{S3_BASE}/{prefix}/{run_id}/trajs/{instance_id}/{instance_id}.traj.json"
    if prefix == "verified":
        traj_url = f"{S3_BASE}/{prefix}/{run_id}/trajs/{instance_id}.traj"
    patch_url = f"{S3_BASE}/{prefix}/{run_id}/logs/{instance_id}/patch.diff"

    return {
        "run_id": run_id,
        "prefix": prefix,
        "model_name": model_name,
        "instance_id": instance_id,
        "resolved": resolved,
        "patch_applied": applied,
        "f2p_success": f2p_succ,
        "f2p_failure": f2p_fail,
        "p2p_success": p2p_succ,
        "p2p_failure": p2p_fail,
        "f2p_failures_list": f2p.get("failure", []),
        "p2p_failures_list": p2p.get("failure", []),
        "is_candidate": is_candidate,
        "is_deceptive_regression": is_deceptive_regression,
        "is_clean_normal": is_clean_normal,
        "label": 1 if is_deceptive_regression else (0 if is_clean_normal else -1),
        "traj_url": traj_url,
        "patch_url": patch_url,
        "report_sha256": hashlib.sha256(rep_bytes).hexdigest(),
    }


def main():
    print("=" * 70)
    print("FETCHING SWE-BENCH VERIFIED RUNS & COMPUTING DECEPTIVE REGRESSIONS")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    all_runs_summary = []
    all_records = []
    
    for prefix, run_id, model_name in TARGET_RUNS:
        print(f"\nProcessing {model_name} ({run_id})...")
        t0 = time.time()
        instances = fetch_run_instance_list(prefix, run_id)
        print(f"  Found {len(instances)} tasks to audit.")
        
        run_records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            futures = [pool.submit(process_instance_report, prefix, run_id, model_name, iid) for iid in instances]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res:
                    run_records.append(res)
        
        t1 = time.time()
        n_tot = len(run_records)
        n_res = sum(r["resolved"] for r in run_records)
        n_cand = sum(r["is_candidate"] for r in run_records)
        n_deceptive = sum(r["is_deceptive_regression"] for r in run_records)
        n_clean = sum(r["is_clean_normal"] for r in run_records)
        n_any_p2p = sum(r["p2p_failure"] > 0 for r in run_records)
        
        deceptive_pct = (n_deceptive / n_cand * 100) if n_cand > 0 else 0.0
        p2p_pct = (n_any_p2p / n_tot * 100) if n_tot > 0 else 0.0
        
        print(f"  Completed in {t1-t0:.2f}s:")
        print(f"    Total evaluated: {n_tot}")
        print(f"    Resolved (Clean Pass): {n_res} ({n_res/n_tot*100:.2f}%)")
        print(f"    Candidate Patches (Passing F2P): {n_cand}")
        print(f"    Deceptive Regressions (Passing F2P yet failing P2P): {n_deceptive} ({deceptive_pct:.2f}% of candidates)")
        print(f"    Any P2P Breakage across all runs: {n_any_p2p} ({p2p_pct:.2f}% of all {n_tot})")
        
        all_runs_summary.append({
            "run_id": run_id,
            "model_name": model_name,
            "prefix": prefix,
            "total_runs": n_tot,
            "resolved": n_res,
            "candidate_runs": n_cand,
            "deceptive_regressions": n_deceptive,
            "clean_normal": n_clean,
            "any_p2p_breakage": n_any_p2p,
            "deceptive_anomaly_rate_pct": round(deceptive_pct, 2),
            "any_p2p_anomaly_rate_pct": round(p2p_pct, 2),
        })
        all_records.extend(run_records)

    # Export manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_data = {
        "benchmark": "SWE-bench Verified",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "target_runs": all_runs_summary,
        "total_records_count": len(all_records),
        "candidate_records_count": sum(r["is_candidate"] for r in all_records),
        "deceptive_regression_count": sum(r["is_deceptive_regression"] for r in all_records),
        "clean_normal_count": sum(r["is_clean_normal"] for r in all_records),
        "records": all_records,
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"MANIFEST SAVED TO {manifest_path}")
    print("=" * 70)
    print(f"Total evaluated records: {len(all_records)}")
    cand_records = [r for r in all_records if r["is_candidate"]]
    deceptive_records = [r for r in cand_records if r["is_deceptive_regression"]]
    clean_records = [r for r in cand_records if r["is_clean_normal"]]
    
    print(f"Candidate population (passing F2P): {len(cand_records)}")
    print(f"  Normal (clean resolved): {len(clean_records)} ({len(clean_records)/len(cand_records)*100:.2f}%)")
    print(f"  Anomaly (deceptive regression): {len(deceptive_records)} ({len(deceptive_records)/len(cand_records)*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
