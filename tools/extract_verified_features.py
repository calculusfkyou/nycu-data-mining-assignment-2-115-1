"""Extract R0-struct features and dense embeddings from SWE-bench Verified candidate trajectories.

Features:
- 20 structural and Verification-Gap metrics (unverified edit ratio, cross-module dispersion, post-test edits, etc.)
- 384-D dense MiniLM log embeddings (sentence-transformers/all-MiniLM-L6-v2)
- 768-D dense CodeBERT diff embeddings (microsoft/codebert-base)
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MANIFEST_PATH = Path("Dataset/verified/manifest.json")
OUTPUT_DIR = Path("Dataset/verified")
RAW_DIR = OUTPUT_DIR / "raw"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

TEST_CMD_PATTERN = re.compile(
    r"\b(pytest|unittest|runtests|manage\.py\s+test)\b|python\s+.*tests?/", re.IGNORECASE
)
DIFF_FILE_PATTERN = re.compile(r"diff --git a/(\S+) b/\S+")
ANSI_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def clean_text(text: str) -> str:
    """Normalize text: strip ANSI codes, replace common working dir paths."""
    text = ANSI_PATTERN.sub("", text)
    text = text.replace("/testbed", "<WORKDIR>")
    return text.strip()


def parse_patch_details(patch_text: str) -> Tuple[int, int, int, List[str], int]:
    """Parse added/deleted line counts, files touched, and hunk counts."""
    lines = patch_text.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    deleted = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    files = DIFF_FILE_PATTERN.findall(patch_text)
    hunks = sum(1 for l in lines if l.startswith("@@"))
    return added, deleted, added + deleted, files, hunks


def parse_trajectory_dynamics(
    traj_data: Dict[str, Any], modified_files: List[str]
) -> Dict[str, Any]:
    """Parse sequential dynamics and verification-gap metrics from trajectory."""
    messages = traj_data.get("messages", [])
    commands: List[Tuple[int, str]] = []
    observations: List[str] = []
    returncodes: List[int] = []
    
    test_step_indices: List[int] = []
    edit_step_indices: List[int] = []
    tested_files: set[str] = set()
    
    step_idx = 0
    tracebacks = 0
    syntax_errors = 0
    total_tokens_received = 0
    
    # Check if messages is SWE-agent format or mini-SWE-agent format
    if not messages and "trajectory" in traj_data:
        # Legacy SWE-agent format
        traj_steps = traj_data.get("trajectory", [])
        for step in traj_steps:
            step_idx += 1
            act = step.get("action", "")
            obs = step.get("observation", "")
            commands.append((step_idx, act))
            observations.append(obs)
            total_tokens_received += len(obs)
            if "Traceback" in obs:
                tracebacks += 1
            if "SyntaxError" in obs:
                syntax_errors += 1
            if TEST_CMD_PATTERN.search(act):
                test_step_indices.append(step_idx)
                for f in modified_files:
                    if f in act or f in obs:
                        tested_files.add(f)
            if any(k in act.lower() for k in ["edit", "sed", "patch", "cat >", "echo >", "python /tmp/"]):
                edit_step_indices.append(step_idx)
            returncodes.append(0)
    else:
        # mini-SWE-agent format
        for m in messages:
            role = m.get("role")
            if role == "assistant":
                step_idx += 1
                for tc in m.get("tool_calls", []):
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    cmd = args.get("command", "") if isinstance(args, dict) else str(args)
                    commands.append((step_idx, cmd))
                    
                    if TEST_CMD_PATTERN.search(cmd):
                        test_step_indices.append(step_idx)
                        for f in modified_files:
                            if f in cmd:
                                tested_files.add(f)
                    if any(k in cmd.lower() for k in ["edit", "sed", "patch", "cat >", "echo >", "python /tmp/"]):
                        edit_step_indices.append(step_idx)
            elif role == "tool":
                content = str(m.get("content", ""))
                observations.append(content)
                total_tokens_received += len(content)
                if "Traceback" in content:
                    tracebacks += 1
                if "SyntaxError" in content:
                    syntax_errors += 1
                rc_match = re.search(r"<returncode>(\d+)</returncode>", content)
                rc = int(rc_match.group(1)) if rc_match else 0
                returncodes.append(rc)
                
                # Also check observation for tested files
                if test_step_indices and test_step_indices[-1] == step_idx:
                    for f in modified_files:
                        if f in content:
                            tested_files.add(f)

    # Verification Gap: files touched vs files tested
    if not modified_files:
        unverified_edit_ratio = 0.0
    else:
        unverified_count = sum(1 for f in modified_files if f not in tested_files)
        unverified_edit_ratio = unverified_count / len(modified_files)
    
    # Cross-Module Dispersion: distinct top-level directories touched
    top_dirs = set(f.split("/")[0] if "/" in f else "." for f in modified_files)
    cross_module_dispersion = len(top_dirs)
    
    # Post-test edits: modifications after the last test run
    last_test_step = max(test_step_indices) if test_step_indices else 0
    post_test_edits = sum(1 for s in edit_step_indices if s > last_test_step)
    
    # Test ratio
    total_cmds = len(commands)
    test_ratio = len(test_step_indices) / max(total_cmds, 1)
    
    # Terminal error streak
    streak = 0
    max_streak = 0
    for rc in returncodes:
        if rc != 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
            
    # Repeated command count
    repeated = 0
    for i in range(1, len(commands)):
        if commands[i][1].strip() == commands[i-1][1].strip() and len(commands[i][1].strip()) > 5:
            repeated += 1

    # Extract clean text log representation for embedding
    log_snippets = []
    for step_num, cmd in commands:
        log_snippets.append(f"[STEP {step_num}] $ {cmd}")
    for obs in observations[:15]:  # include head observations
        cleaned_obs = clean_text(obs[:500])
        if cleaned_obs:
            log_snippets.append(cleaned_obs)
    combined_log_text = "\n".join(log_snippets)

    # Extract model stats
    stats = traj_data.get("info", {}).get("model_stats", {})
    cost = float(stats.get("instance_cost", 0.0) or 0.0)
    api_calls = int(stats.get("api_calls", 0) or step_idx)

    return {
        "unverified_edit_ratio": unverified_edit_ratio,
        "cross_module_dispersion": cross_module_dispersion,
        "post_test_edit_steps": post_test_edits,
        "reproduction_test_ratio": test_ratio,
        "total_steps": step_idx,
        "total_commands": total_cmds,
        "total_tracebacks": tracebacks,
        "terminal_error_streak": max_streak,
        "repeated_command_count": repeated,
        "total_tokens_received": total_tokens_received,
        "total_cost_usd": cost,
        "api_calls": api_calls,
        "has_syntax_error": 1 if syntax_errors > 0 else 0,
        "combined_log_text": combined_log_text,
    }


def download_file_cached(url: str, dest_path: Path) -> bool:
    """Download file to dest_path if not already existing."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        return False


def process_candidate_run(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Download traj/patch and extract all features."""
    run_id = row["run_id"]
    iid = row["instance_id"]
    
    traj_path = RAW_DIR / run_id / "trajs" / f"{iid}.json"
    patch_path = RAW_DIR / run_id / "patches" / f"{iid}.diff"
    
    traj_ok = download_file_cached(row["traj_url"], traj_path)
    patch_ok = download_file_cached(row["patch_url"], patch_path)
    
    if not traj_ok or not patch_ok:
        return None
    
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
        with open(patch_path, "r", encoding="utf-8", errors="replace") as f:
            patch_text = f.read()
    except Exception:
        return None
        
    added, deleted, total_diff, files, hunks = parse_patch_details(patch_text)
    dynamics = parse_trajectory_dynamics(traj_data, files)
    
    diff_deleted_ratio = deleted / max(total_diff, 1)
    edit_to_step = total_diff / max(dynamics["total_steps"], 1)
    
    return {
        "run_id": run_id,
        "model_name": row["model_name"],
        "instance_id": iid,
        "label": row["label"],  # 0: Clean resolved, 1: Deceptive regression
        "is_deceptive_regression": row["is_deceptive_regression"],
        "is_clean_normal": row["is_clean_normal"],
        "p2p_failure": row["p2p_failure"],
        "f2p_success": row["f2p_success"],
        "f2p_failure": row["f2p_failure"],
        # 20 R0-struct features:
        "edit_lines_added": added,
        "edit_lines_deleted": deleted,
        "edit_lines_total": total_diff,
        "edit_files_count": len(files),
        "edit_hunks_count": hunks,
        "unverified_edit_ratio": dynamics["unverified_edit_ratio"],
        "cross_module_dispersion": dynamics["cross_module_dispersion"],
        "post_test_edit_steps": dynamics["post_test_edit_steps"],
        "reproduction_test_ratio": dynamics["reproduction_test_ratio"],
        "total_steps": dynamics["total_steps"],
        "total_commands": dynamics["total_commands"],
        "total_tracebacks": dynamics["total_tracebacks"],
        "terminal_error_streak": dynamics["terminal_error_streak"],
        "repeated_command_count": dynamics["repeated_command_count"],
        "total_tokens_received": dynamics["total_tokens_received"],
        "total_cost_usd": dynamics["total_cost_usd"],
        "api_calls": dynamics["api_calls"],
        "patch_size_bytes": len(patch_text),
        "has_syntax_error": dynamics["has_syntax_error"],
        "diff_deleted_ratio": diff_deleted_ratio,
        "edit_to_step_ratio": edit_to_step,
        # Raw text for encoder embedding
        "log_text": dynamics["combined_log_text"],
        "patch_text": patch_text[:10000],  # truncated for encoder
    }


def compute_embeddings(
    df: pd.DataFrame, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MiniLM log embeddings and CodeBERT diff embeddings."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    minilm_path = EMBEDDINGS_DIR / "minilm_logs.npy"
    codebert_path = EMBEDDINGS_DIR / "codebert_diffs.npy"
    
    if minilm_path.exists() and codebert_path.exists():
        print("Loading precomputed embeddings from disk...")
        return np.load(minilm_path), np.load(codebert_path)
    
    print("\nEncoding terminal logs with all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer
    minilm_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    log_texts = df["log_text"].tolist()
    minilm_embs = minilm_model.encode(
        log_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    np.save(minilm_path, minilm_embs)
    print(f"Saved MiniLM embeddings: shape {minilm_embs.shape} to {minilm_path}")
    
    print("\nEncoding patch diffs with CodeBERT...")
    cb_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    cb_model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    cb_model.eval()
    
    patch_texts = df["patch_text"].tolist()
    codebert_embs = []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(patch_texts), batch_size):
            batch = patch_texts[i : i + batch_size]
            encoded = cb_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = cb_model(**encoded)
            # Mean pool non-padding tokens
            mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            codebert_embs.append(pooled.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"  CodeBERT encoded {i + len(batch)} / {len(patch_texts)} diffs...")
                
    codebert_embs = np.vstack(codebert_embs)
    np.save(codebert_path, codebert_embs)
    print(f"Saved CodeBERT embeddings: shape {codebert_embs.shape} to {codebert_path}")
    
    return minilm_embs, codebert_embs


def main():
    print("=" * 70)
    print("EXTRACTING FEATURES & EMBEDDINGS FOR SWE-BENCH VERIFIED CANDIDATES")
    print("=" * 70)
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    candidates = [r for r in manifest["records"] if r["is_candidate"]]
    print(f"Found {len(candidates)} candidate runs to extract features for.")
    
    t0 = time.time()
    extracted_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(process_candidate_run, r) for r in candidates]
        for idx, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res:
                extracted_rows.append(res)
            if (idx + 1) % 500 == 0:
                print(f"  Extracted {idx + 1} / {len(candidates)} runs ({time.time()-t0:.1f}s)...")
                
    df = pd.DataFrame(extracted_rows)
    print(f"\nFeature extraction complete: {len(df)} candidate rows successfully extracted.")
    
    # Save CSV table
    csv_path = OUTPUT_DIR / "candidate_features.csv"
    # Drop long raw text columns from CSV for lean tabular storage
    df_tabular = df.drop(columns=["log_text", "patch_text"])
    df_tabular.to_csv(csv_path, index=False)
    print(f"Saved tabular features to {csv_path}")
    print(f"  Normal instances (y=0): {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.2f}%)")
    print(f"  Anomaly instances (y=1): {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.2f}%)")
    
    # Compute and save embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device} for embedding extraction.")
    compute_embeddings(df, device=device)
    print("\nFeature & embedding extraction fully completed!")


if __name__ == "__main__":
    main()
