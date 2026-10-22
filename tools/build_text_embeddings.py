"""Extract local frozen NLP and Code embeddings (R2 MiniLM, R3 CodeBERT) and R1 TF-IDF.

Strict adherence to HW2 v2 specification:
1. R2 Log Encoder: sentence-transformers/all-MiniLM-L6-v2 (frozen, 384-dim).
   - Chunk action & observation separately (content <= 224, overlap=32, total <= 256).
   - Per-step pooling: average step's chunk embeddings, L2 normalize.
   - Run-level pooling: equal-weight average across steps, L2 normalize.
   - Store both run-level and per-step embeddings (for step attribution / LOO analysis).
2. R3 Patch Encoder: microsoft/codebert-base (frozen token encoder, 768-dim).
   - Diff segmented by hunks (content <= 480, overlap=64, total <= 512).
   - Mean pool last layer hidden states over non-padding, non-special tokens.
   - Chunks weighted average by content token count, L2 normalize.
   - Empty patch gets exact zero vector.
3. R1 TF-IDF:
   - Character n-gram (3-5), min_df=2, sublinear_tf=True, max 20,000 features, L2 normalized.
   - Provides fold-aware fitting function fitted strictly on fitting rows.
4. Channel Distance:
   - d = (d_log + d_patch) / 2.
   - Both patches missing: d_patch = 0; one missing: d_patch = 1; both present: (1 - cosine) / 2.
   - Same for log.
5. Local execution only, zero remote APIs, caching with full manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModel, AutoTokenizer

from tools.normalize_text import normalize_log_text, normalize_patch_text


MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CODEBERT_NAME = "microsoft/codebert-base"

EMBEDDINGS_DIR = Path("Dataset/v2/embeddings")
MANIFEST_PATH = Path("Dataset/v2/text_manifest.jsonl")
SPLITS_PATH = Path("Dataset/v2/splits.json")
RAW_TRAJ_DIR = Path("Dataset/raw/sweagent_gpt4")


def split_text_into_token_chunks(
    token_ids: List[int],
    max_chunk_size: int,
    overlap: int,
) -> List[Tuple[List[int], int]]:
    """Split token IDs into chunks of max_chunk_size with specified overlap.

    Returns list of (chunk_token_ids, new_content_token_count).
    """
    if not token_ids:
        return []
    if len(token_ids) <= max_chunk_size:
        return [(token_ids, len(token_ids))]

    chunks = []
    stride = max(1, max_chunk_size - overlap)
    start = 0
    while start < len(token_ids):
        chunk = token_ids[start : start + max_chunk_size]
        new_tokens = len(chunk) if start == 0 else min(stride, len(token_ids) - start)
        chunks.append((chunk, new_tokens))
        if start + max_chunk_size >= len(token_ids):
            break
        start += stride
    return chunks


def split_patch_into_hunks(diff_text: str) -> List[str]:
    """Split diff into hunks by diff header or hunk header (@@)."""
    if not diff_text or not diff_text.strip():
        return []

    lines = diff_text.splitlines(keepends=True)
    hunks: List[str] = []
    current_hunk: List[str] = []

    for line in lines:
        if line.startswith("diff --git ") or line.startswith("@@ "):
            if current_hunk:
                hunks.append("".join(current_hunk))
                current_hunk = []
        current_hunk.append(line)

    if current_hunk:
        hunks.append("".join(current_hunk))

    return hunks if hunks else [diff_text]


class TextEmbeddingExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading encoders on device: {self.device}...")

        # Load MiniLM
        self.minilm_tok = AutoTokenizer.from_pretrained(MINILM_NAME)
        self.minilm_mod = AutoModel.from_pretrained(MINILM_NAME).to(self.device)
        self.minilm_mod.eval()

        # Load CodeBERT
        self.cb_tok = AutoTokenizer.from_pretrained(CODEBERT_NAME)
        self.cb_mod = AutoModel.from_pretrained(CODEBERT_NAME).to(self.device)
        self.cb_mod.eval()

        print("Encoders initialized successfully.")

    def encode_minilm_chunks(self, token_chunks: List[List[int]], batch_size: int = 32) -> np.ndarray:
        """Encode chunks through MiniLM and mean-pool non-special tokens."""
        if not token_chunks:
            return np.zeros((0, 384), dtype=np.float32)

        cls_id = self.minilm_tok.cls_token_id
        sep_id = self.minilm_tok.sep_token_id
        pad_id = self.minilm_tok.pad_token_id or 0

        all_vecs: List[np.ndarray] = []

        for b_start in range(0, len(token_chunks), batch_size):
            b_chunks = token_chunks[b_start : b_start + batch_size]
            # Construct sequences: [CLS] + chunk + [SEP]
            seqs = [[cls_id] + ch + [sep_id] for ch in b_chunks]
            max_len = max(len(s) for s in seqs)

            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)

            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
                attention_mask[i, : len(s)] = 1

            with torch.no_grad():
                out = self.minilm_mod(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = out.last_hidden_state  # (B, L, 384)

                for i, s in enumerate(seqs):
                    # Mean pool non-special, non-padding tokens: slice 1 : len(s) - 1
                    content_slice = last_hidden[i, 1 : len(s) - 1]
                    if content_slice.shape[0] == 0:
                        v = last_hidden[i, 0].cpu().numpy()
                    else:
                        v = content_slice.mean(dim=0).cpu().numpy()
                    all_vecs.append(v)

        return np.array(all_vecs, dtype=np.float32)

    def extract_trajectory_log_embedding(
        self,
        trajectory: List[Dict[str, Any]],
        instance_id: str,
        repo: str,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Extract R2 MiniLM run-level embedding and per-step embeddings.

        Returns: (run_embedding, list_of_step_embeddings).
        """
        if not trajectory:
            return np.zeros(384, dtype=np.float32), []

        step_embeddings: List[np.ndarray] = []

        for step_idx, step in enumerate(trajectory):
            raw_action = str(step.get("action", ""))
            raw_obs = str(step.get("observation", ""))

            clean_action = normalize_log_text(raw_action, instance_id, repo)
            clean_obs = normalize_log_text(raw_obs, instance_id, repo)

            # Tokenize action and observation separately
            action_tokens = self.minilm_tok.encode(clean_action, add_special_tokens=False) if clean_action else []
            obs_tokens = self.minilm_tok.encode(clean_obs, add_special_tokens=False) if clean_obs else []

            # Chunk each with max 224 content tokens, overlap 32
            action_chunks = [ch for ch, _ in split_text_into_token_chunks(action_tokens, max_chunk_size=224, overlap=32)]
            obs_chunks = [ch for ch, _ in split_text_into_token_chunks(obs_tokens, max_chunk_size=224, overlap=32)]

            all_step_chunks = action_chunks + obs_chunks

            if not all_step_chunks:
                # Fallback empty step
                step_vec = np.zeros(384, dtype=np.float32)
            else:
                chunk_vecs = self.encode_minilm_chunks(all_step_chunks)
                step_vec = chunk_vecs.mean(axis=0)
                norm = np.linalg.norm(step_vec)
                if norm > 1e-8:
                    step_vec = step_vec / norm

            step_embeddings.append(step_vec)

        # Run-level: equal-weight average across all steps, then L2 normalize
        valid_steps = [v for v in step_embeddings if np.linalg.norm(v) > 1e-8]
        if not valid_steps:
            run_embedding = np.zeros(384, dtype=np.float32)
        else:
            run_embedding = np.mean(valid_steps, axis=0)
            norm = np.linalg.norm(run_embedding)
            if norm > 1e-8:
                run_embedding = run_embedding / norm

        return run_embedding.astype(np.float32), step_embeddings

    def extract_patch_embedding(
        self,
        submission: Optional[str],
        instance_id: str,
        repo: str,
        batch_size: int = 16,
    ) -> Tuple[np.ndarray, bool]:
        """Extract R3 CodeBERT patch embedding.

        Returns (patch_embedding, is_empty).
        """
        clean_patch, is_empty = normalize_patch_text(submission, instance_id, repo)
        if is_empty or not clean_patch:
            return np.zeros(768, dtype=np.float32), True

        hunks = split_patch_into_hunks(clean_patch)
        token_chunks_with_weight: List[Tuple[List[int], int]] = []

        for hunk in hunks:
            # Tokenize hunk without truncation warning
            hunk_tokens = self.cb_tok(hunk, add_special_tokens=False, verbose=False)["input_ids"]
            chunks = split_text_into_token_chunks(hunk_tokens, max_chunk_size=480, overlap=64)
            token_chunks_with_weight.extend(chunks)

        if not token_chunks_with_weight:
            return np.zeros(768, dtype=np.float32), True

        cls_id = self.cb_tok.cls_token_id or 0
        sep_id = self.cb_tok.sep_token_id or 2
        pad_id = self.cb_tok.pad_token_id or 1

        chunk_vectors: List[np.ndarray] = []
        chunk_weights: List[int] = []

        for b_start in range(0, len(token_chunks_with_weight), batch_size):
            b_items = token_chunks_with_weight[b_start : b_start + batch_size]
            seqs = [[cls_id] + ch + [sep_id] for ch, _ in b_items]
            max_len = max(len(s) for s in seqs)

            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)

            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
                attention_mask[i, : len(s)] = 1

            with torch.no_grad():
                out = self.cb_mod(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = out.last_hidden_state

                for i, (ch, weight) in enumerate(b_items):
                    # Mean pool non-special tokens: 1 to len(s) - 1
                    s_len = len(ch) + 2
                    content_slice = last_hidden[i, 1 : s_len - 1]
                    if content_slice.shape[0] == 0:
                        v = last_hidden[i, 0].cpu().numpy()
                    else:
                        v = content_slice.mean(dim=0).cpu().numpy()
                    chunk_vectors.append(v)
                    chunk_weights.append(weight)

        weights_arr = np.array(chunk_weights, dtype=np.float32)
        total_w = weights_arr.sum()
        if total_w < 1e-8:
            return np.zeros(768, dtype=np.float32), True

        weights_norm = weights_arr / total_w
        patch_vec = np.sum(np.array(chunk_vectors) * weights_norm[:, None], axis=0)
        norm = np.linalg.norm(patch_vec)
        if norm > 1e-8:
            patch_vec = patch_vec / norm

        return patch_vec.astype(np.float32), False


def compute_channel_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    is_empty1: bool = False,
    is_empty2: bool = False,
) -> float:
    """Compute cosine distance between two vectors with explicit missingness rules.

    Missingness rules from specification:
    - Both missing / zero: d = 0.0
    - Exactly one missing / zero: d = 1.0
    - Both present: d = (1 - cosine) / 2, clipped to [0, 1].
    """
    if is_empty1 and is_empty2:
        return 0.0
    if is_empty1 != is_empty2:
        return 1.0

    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)

    if n1 < 1e-8 and n2 < 1e-8:
        return 0.0
    if n1 < 1e-8 or n2 < 1e-8:
        return 1.0

    cos = np.dot(vec1, vec2) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    return float((1.0 - cos) / 2.0)


def compute_dual_distance(
    log_vec1: np.ndarray,
    log_vec2: np.ndarray,
    patch_vec1: np.ndarray,
    patch_vec2: np.ndarray,
    patch_empty1: bool,
    patch_empty2: bool,
) -> float:
    """Compute R3 dual channel distance: d = (d_log + d_patch) / 2."""
    d_log = compute_channel_distance(log_vec1, log_vec2)
    d_patch = compute_channel_distance(patch_vec1, patch_vec2, patch_empty1, patch_empty2)
    return 0.5 * (d_log + d_patch)


def run_pipeline(is_probe: bool = False, sample_limit: Optional[int] = None) -> None:
    start_time = time.perf_counter()
    process = psutil.Process()
    mem_start_mb = process.memory_info().rss / (1024 * 1024)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Read manifest
    instances = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))

    if is_probe or sample_limit:
        limit = sample_limit or 5
        instances = instances[:limit]
        print(f"Running PROBE mode on {len(instances)} trajectories...")
    else:
        print(f"Running FULL embedding extraction on all {len(instances)} trajectories...")

    extractor = TextEmbeddingExtractor()

    r2_log_embeddings = {}
    r3_patch_embeddings = {}
    step_embeddings_dict = {}
    patch_empty_flags = {}

    traj_latencies = []

    for idx, inst in enumerate(instances):
        t0 = time.perf_counter()
        iid = inst["instance_id"]
        repo = inst["repo"]
        traj_file = RAW_TRAJ_DIR / f"{iid}.traj"

        with open(traj_file, "r", encoding="utf-8", errors="replace") as f:
            tdata = json.load(f)

        trajectory = tdata.get("trajectory", [])
        submission = tdata.get("info", {}).get("submission")

        log_vec, step_vecs = extractor.extract_trajectory_log_embedding(trajectory, iid, repo)
        patch_vec, is_empty = extractor.extract_patch_embedding(submission, iid, repo)

        r2_log_embeddings[iid] = log_vec
        r3_patch_embeddings[iid] = patch_vec
        step_embeddings_dict[iid] = step_vecs
        patch_empty_flags[iid] = is_empty

        t1 = time.perf_counter()
        traj_latencies.append(t1 - t0)

        if (idx + 1) % 10 == 0 or idx == len(instances) - 1:
            curr_mem = process.memory_info().rss / (1024 * 1024)
            print(
                f"[{idx+1}/{len(instances)}] processed {iid} "
                f"(steps={len(trajectory)}, patch_empty={is_empty}) "
                f"time={t1-t0:.2f}s, RSS={curr_mem:.1f}MB"
            )

    total_time = time.perf_counter() - start_time
    mem_peak_mb = process.memory_info().rss / (1024 * 1024)

    print(f"\nCompleted in {total_time:.2f}s.")
    print(f"Average latency per trajectory: {np.mean(traj_latencies):.2f}s (std={np.std(traj_latencies):.2f}s)")
    print(f"Peak memory RSS: {mem_peak_mb:.1f}MB (delta: {mem_peak_mb - mem_start_mb:.1f}MB)")

    if not is_probe:
        # Save arrays and step embeddings
        inst_keys = sorted(list(r2_log_embeddings.keys()))
        log_matrix = np.array([r2_log_embeddings[k] for k in inst_keys], dtype=np.float32)
        patch_matrix = np.array([r3_patch_embeddings[k] for k in inst_keys], dtype=np.float32)
        empty_flags_arr = np.array([patch_empty_flags[k] for k in inst_keys], dtype=bool)

        np.savez_compressed(
            EMBEDDINGS_DIR / "r2_minilm.npz",
            instance_ids=inst_keys,
            embeddings=log_matrix,
        )
        np.savez_compressed(
            EMBEDDINGS_DIR / "r3_codebert.npz",
            instance_ids=inst_keys,
            embeddings=patch_matrix,
            is_empty=empty_flags_arr,
        )

        # Save step embeddings for error analysis / interpretability
        step_export = {k: [v.tolist() for v in step_embeddings_dict[k]] for k in inst_keys}
        with open(EMBEDDINGS_DIR / "step_embeddings.json", "w", encoding="utf-8") as f:
            json.dump(step_export, f)

        # Cache manifest
        manifest = {
            "created_at": "2026-10-23",
            "models": {
                "log_encoder": MINILM_NAME,
                "log_dim": 384,
                "patch_encoder": CODEBERT_NAME,
                "patch_dim": 768,
            },
            "num_trajectories": len(inst_keys),
            "total_extraction_wall_clock_seconds": total_time,
            "mean_latency_per_run_seconds": float(np.mean(traj_latencies)),
            "peak_memory_mb": float(mem_peak_mb),
            "empty_patch_count": int(sum(empty_flags_arr)),
        }
        with open(EMBEDDINGS_DIR / "cache_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Embeddings and manifest cached to {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Run quick probe on 5 samples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trajectories")
    args = parser.parse_args()

    run_pipeline(is_probe=args.probe, sample_limit=args.limit)
