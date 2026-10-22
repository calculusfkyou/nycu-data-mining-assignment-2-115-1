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
    previous_tool = None
    tool_streak = 0
    max_tool_streak = 0
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
        tool_streak = tool_streak + 1 if tool_name == previous_tool else 1
        max_tool_streak = max(max_tool_streak, tool_streak)
        previous_tool = tool_name

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
        max_repeated_tool = max_tool_streak
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
        "patch_hunk_count": sum(line.startswith("@@ ") for line in submission.splitlines()),
    }


def main() -> None:
    raise SystemExit(
        "Processed splits are frozen. To restore raw inputs run python tools/fetch_raw.py; "
        "then python tools/prepare_data.py. See README.md."
    )


if __name__ == "__main__":
    main()
