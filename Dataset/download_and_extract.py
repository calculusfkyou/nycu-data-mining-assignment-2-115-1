"""Download real SWE-bench trajectories and extract domain-informed features.

Real data source: SWE-bench Lite (ICLR 2024)
Evaluation run: SWE-agent + GPT-4 (20240402_sweagent_gpt4)
Public S3 bucket: swe-bench-submissions.s3.amazonaws.com
Ground truth: Official SWE-bench execution logs and evaluation results.
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULTS_JSON_URL = (
    "https://raw.githubusercontent.com/SWE-bench/experiments/main/"
    "evaluation/lite/20240402_sweagent_gpt4/results/results.json"
)

S3_BASE_URL = "https://swe-bench-submissions.s3.amazonaws.com/lite/20240402_sweagent_gpt4/trajs"

RAW_DIR = Path("Dataset/raw/sweagent_gpt4")
DATASET_DIR = Path("Dataset")


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with user-agent headers if it doesn't already exist."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def fetch_evaluation_metadata() -> dict[str, Any]:
    """Fetch official SWE-bench evaluation results."""
    req = urllib.request.Request(RESULTS_JSON_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def extract_features_from_trajectory(
    data: dict[str, Any], instance_id: str, is_resolved: bool
) -> dict[str, Any]:
    """Extract 24 process and output features from a raw trajectory dict."""
    trajectory = data.get("trajectory", [])
    info = data.get("info", {})
    model_stats = info.get("model_stats", {})
    submission = info.get("submission", "") or ""
    exit_status = str(info.get("exit_status", "unknown"))

    num_steps = len(trajectory)
    tool_counter: collections.Counter = collections.Counter()
    error_steps = 0
    consecutive_errors = 0
    max_consecutive_errors = 0
    stalled_turns = 0
    reproduce_created = 0
    test_suite_runs = 0
    prev_obs = ""
    last_step_error = 0

    tool_pattern = re.compile(r"^([a-zA-Z0-9_\-\.]+)")
    error_pattern = re.compile(
        r"(Error:|Traceback|FAILED|SyntaxError|ImportError|AttributeError|"
        r"ValueError|KeyError|IndexError|command not found|exit code [1-9]|fatal:)",
        re.IGNORECASE,
    )

    for i, step in enumerate(trajectory):
        action = str(step.get("action", "")).strip()
        obs = str(step.get("observation", "")).strip()

        # Identify primary tool / command
        match = tool_pattern.match(action)
        tool_name = match.group(1).lower() if match else "unknown"
        tool_counter[tool_name] += 1

        # Check reproduction script creation
        if "reproduce" in action.lower() or (
            action.startswith("create ") and "test" in action.lower()
        ):
            reproduce_created = 1

        # Check test executions
        if any(cmd in action.lower() for cmd in ["pytest", "python ", "tox", "unittest"]):
            test_suite_runs += 1

        # Check error presence
        has_error = bool(error_pattern.search(obs))
        if has_error:
            error_steps += 1
            consecutive_errors += 1
            if consecutive_errors > max_consecutive_errors:
                max_consecutive_errors = consecutive_errors
        else:
            consecutive_errors = 0

        # Check stalled / empty / repeating observations
        if len(obs) == 0 or obs == prev_obs:
            stalled_turns += 1
        prev_obs = obs

        if i == num_steps - 1 and has_error:
            last_step_error = 1

    # Tool entropy computation
    total_tool_calls = sum(tool_counter.values())
    if total_tool_calls > 0:
        entropy = 0.0
        for count in tool_counter.values():
            p = count / total_tool_calls
            entropy -= p * math.log2(p)
        tool_entropy = float(entropy)
        max_repeated_tool = max(tool_counter.values())
    else:
        tool_entropy = 0.0
        max_repeated_tool = 0

    # Submission patch analysis
    lines_added = len(re.findall(r"^\+[^\+]", submission, re.MULTILINE))
    lines_removed = len(re.findall(r"^\-[^\-]", submission, re.MULTILINE))
    files_changed = len(re.findall(r"^diff --git a/", submission, re.MULTILINE))
    patch_char_length = len(submission)
    patch_is_empty = 1 if patch_char_length == 0 or submission.isspace() else 0
    patch_touches_tests = 1 if re.search(r"(test/|tests/|test_)", submission, re.IGNORECASE) else 0
    patch_edit_ratio = float(lines_added / (lines_added + lines_removed + 1e-5))

    # Forced termination flag (e.g. hitting cost budget)
    exit_cost_forced = 1 if "exit_cost" in exit_status else 0

    # Model and token usage stats
    total_cost_usd = float(model_stats.get("instance_cost", 0.0) or 0.0)
    tokens_sent = int(model_stats.get("tokens_sent", 0) or 0)
    tokens_received = int(model_stats.get("tokens_received", 0) or 0)
    total_tokens = tokens_sent + tokens_received
    tokens_per_step = float(total_tokens / max(1, num_steps))

    # Anomaly Definition:
    # 0 = Normal (Task resolved successfully, verified by test suite)
    # 1 = Anomaly (Ghostwork / Task failed to resolve issue)
    label = 0 if is_resolved else 1

    # Extract repository name
    repo = instance_id.split("__")[0] if "__" in instance_id else "unknown"

    return {
        "instance_id": instance_id,
        "repo": repo,
        "label": label,
        "is_resolved": int(is_resolved),
        "exit_status": exit_status,
        # Trajectory Dynamics Features (16)
        "num_steps": num_steps,
        "num_tool_calls": total_tool_calls,
        "num_unique_tools": len(tool_counter),
        "tool_entropy": tool_entropy,
        "max_repeated_tool_streak": max_repeated_tool,
        "num_error_observations": error_steps,
        "error_observation_rate": float(error_steps / max(1, num_steps)),
        "consecutive_errors_max": max_consecutive_errors,
        "stalled_turn_count": stalled_turns,
        "reproduce_script_created": reproduce_created,
        "test_suite_run_count": test_suite_runs,
        "last_step_error": last_step_error,
        "total_cost_usd": total_cost_usd,
        "tokens_sent": tokens_sent,
        "tokens_received": tokens_received,
        "tokens_per_step": tokens_per_step,
        # Static Patch Features (8)
        "patch_char_length": patch_char_length,
        "patch_lines_added": lines_added,
        "patch_lines_removed": lines_removed,
        "patch_num_files_changed": files_changed,
        "patch_is_empty": patch_is_empty,
        "patch_touches_tests": patch_touches_tests,
        "patch_edit_ratio": patch_edit_ratio,
        "exit_cost_forced": exit_cost_forced,
    }


def main() -> None:
    print("Step 1: Fetching official SWE-bench evaluation results...")
    eval_results = fetch_evaluation_metadata()
    resolved_set = set(eval_results.get("resolved", []))
    generated_set = set(eval_results.get("generated", []))
    print(f"Total generated: {len(generated_set)}, Resolved: {len(resolved_set)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 2: Downloading raw trajectory files from public AWS S3...")
    instance_list = sorted(list(generated_set))
    download_tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for instance_id in instance_list:
            url = f"{S3_BASE_URL}/{instance_id}.traj"
            dest = RAW_DIR / f"{instance_id}.traj"
            download_tasks.append((instance_id, executor.submit(download_file, url, dest)))

        downloaded_count = 0
        for instance_id, future in download_tasks:
            if future.result():
                downloaded_count += 1
    print(f"Downloaded {downloaded_count} trajectories to {RAW_DIR}.")

    print("Step 3: Extracting trajectory and patch features...")
    records = []
    for instance_id in instance_list:
        traj_path = RAW_DIR / f"{instance_id}.traj"
        if not traj_path.exists():
            continue
        try:
            with open(traj_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            is_resolved = instance_id in resolved_set
            rec = extract_features_from_trajectory(data, instance_id, is_resolved)
            records.append(rec)
        except Exception as e:
            print(f"Error parsing {instance_id}: {e}")

    df = pd.DataFrame(records)
    print(f"Successfully processed {len(df)} samples.")
    print(f"Class breakdown: Normal (Resolved) = {(df['label'] == 0).sum()}, Anomaly (Failed) = {(df['label'] == 1).sum()}")

    # Save full features dataset
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    full_path = DATASET_DIR / "sweagent_gpt4_features.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved full feature dataset to {full_path}")

    # Step 4: Create shared, leakage-free train / val / test splits
    # Following rigorous Anomaly Detection partition protocol:
    # Train set: Normal-only samples (clean baseline manifold)
    # Val set: Mixed Normal & Anomaly (strictly for threshold tuning and hyperparameter selection)
    # Test set: Mixed Normal & Anomaly (held out for identical 3-tier comparison)
    np.random.seed(42)
    normal_df = df[df["label"] == 0].sample(frac=1.0, random_state=42).reset_index(drop=True)
    anomaly_df = df[df["label"] == 1].sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 60% of normal samples go to train (normal-only)
    n_normal_train = int(len(normal_df) * 0.60)
    n_normal_val = int(len(normal_df) * 0.20)
    train_normal = normal_df.iloc[:n_normal_train]
    val_normal = normal_df.iloc[n_normal_train : n_normal_train + n_normal_val]
    test_normal = normal_df.iloc[n_normal_train + n_normal_val :]

    # Anomalies split 50% val / 50% test (0% in pure normal train)
    n_anomaly_val = int(len(anomaly_df) * 0.35)
    val_anomaly = anomaly_df.iloc[:n_anomaly_val]
    test_anomaly = anomaly_df.iloc[n_anomaly_val:]

    train_df = train_normal.copy()
    val_df = pd.concat([val_normal, val_anomaly], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = pd.concat([test_normal, test_anomaly], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_df.to_csv(DATASET_DIR / "train.csv", index=False)
    val_df.to_csv(DATASET_DIR / "val.csv", index=False)
    test_df.to_csv(DATASET_DIR / "test.csv", index=False)

    print(f"Train split: {len(train_df)} samples (Normal: {(train_df['label'] == 0).sum()}, Anomaly: {(train_df['label'] == 1).sum()})")
    print(f"Val split:   {len(val_df)} samples (Normal: {(val_df['label'] == 0).sum()}, Anomaly: {(val_df['label'] == 1).sum()})")
    print(f"Test split:  {len(test_df)} samples (Normal: {(test_df['label'] == 0).sum()}, Anomaly: {(test_df['label'] == 1).sum()})")

    # Step 5: Generate Summary Statistics Table (§3 requirement)
    feature_cols = [c for c in df.columns if c not in ["instance_id", "repo", "label", "is_resolved", "exit_status"]]
    stats = []
    for col in feature_cols:
        norm_series = df[df["label"] == 0][col]
        anom_series = df[df["label"] == 1][col]
        stats.append({
            "Feature": col,
            "Type": "Trajectory" if col in feature_cols[:16] else "Patch",
            "Normal_Mean": float(norm_series.mean()),
            "Normal_Std": float(norm_series.std()),
            "Anomaly_Mean": float(anom_series.mean()),
            "Anomaly_Std": float(anom_series.std()),
            "Overall_Median": float(df[col].median()),
            "Overall_IQR": float(df[col].quantile(0.75) - df[col].quantile(0.25)),
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(DATASET_DIR / "summary_statistics.csv", index=False)
    print(f"Saved summary statistics table with {len(stats_df)} features to {DATASET_DIR / 'summary_statistics.csv'}")


if __name__ == "__main__":
    main()
