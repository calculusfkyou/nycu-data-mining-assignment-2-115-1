"""Text normalization for SWE-agent trajectory logs and patches.

Strict rules from HW2 v2 specification:
1. Strip ANSI escape sequences.
2. Replace absolute working directories with <WORKDIR>.
3. Replace repository root names and instance IDs with placeholders (<REPO_ROOT>, <INSTANCE_ID>).
4. Preserve programming symbols, syntax, error types, relative paths, line numbers, and order.
5. Do NOT remove numbers or error keywords blindly.
6. Handle empty patches with an explicit empty-patch mask; zero vector representation,
   never substituting arbitrary placeholder text as patch content.
"""

from __future__ import annotations

import re
from typing import Tuple

# Regex to match ANSI escape codes
ANSI_ESCAPE_PATTERN = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)

# Common root working directory prefixes used in SWE-bench / SWE-agent Docker containers
WORKDIR_PATTERNS = [
    re.compile(r"/testbed(?:/[a-zA-Z0-9_\-\.]+)?"),
    re.compile(r"/root(?:/[a-zA-Z0-9_\-\.]+)?"),
    re.compile(r"/home/[a-zA-Z0-9_\-\.]+"),
    re.compile(r"[A-Z]:\\[^\s:;\"'\(\)\[\]]+"),  # Windows paths if any
]


def strip_ansi(text: str) -> str:
    """Remove all ANSI color codes and terminal control sequences."""
    if not text:
        return ""
    return ANSI_ESCAPE_PATTERN.sub("", text)


def normalize_log_text(
    text: str,
    instance_id: str = "",
    repo_name: str = "",
) -> str:
    """Normalize action/observation log text while strictly preserving syntax and errors."""
    if not text:
        return ""

    # 1. Strip ANSI escape sequences
    text = strip_ansi(text)

    # 2. Mask explicit instance ID if provided
    if instance_id:
        text = text.replace(instance_id, "<INSTANCE_ID>")

    # 3. Mask repository root directory if provided (e.g. django__django or django)
    if repo_name:
        text = text.replace(repo_name, "<REPO_ROOT>")
        # Also mask repo prefix without org (e.g. 'astropy' from 'astropy__astropy')
        short_repo = repo_name.split("__")[-1] if "__" in repo_name else repo_name
        if len(short_repo) > 3:
            # Only mask when appears as directory or container path
            text = re.sub(rf"/(?:{re.escape(short_repo)})[/\b]", "/<REPO_ROOT>/", text)

    # 4. Normalize absolute container working directories
    # Matches patterns like /testbed, /workspace, /astropy__astropy, etc.
    for pat in WORKDIR_PATTERNS:
        text = pat.sub("<WORKDIR>", text)

    # Normalize generic /<repo-name>/ root container mounts
    text = re.sub(r"/(?:[a-zA-Z0-9_\-]+__[a-zA-Z0-9_\-]+)", "<WORKDIR>", text)

    return text.strip()


def normalize_patch_text(
    patch: str | None,
    instance_id: str = "",
    repo_name: str = "",
) -> Tuple[str, bool]:
    """Normalize patch diff text and return (clean_patch, is_empty).

    If patch is empty or whitespace-only: returns ("", True).
    Does NOT inject fake placeholder text.
    """
    if patch is None or not patch.strip():
        return "", True

    text = strip_ansi(patch)
    if instance_id:
        text = text.replace(instance_id, "<INSTANCE_ID>")
    if repo_name:
        text = text.replace(repo_name, "<REPO_ROOT>")

    # Normalize container paths in diff header (e.g. --- a/testbed/... +++ b/testbed/...)
    text = re.sub(r"(---|\+\+\+)\s+[ab]/(?:testbed|root|[a-zA-Z0-9_\-]+__[a-zA-Z0-9_\-]+)/", r"\1 [ab]/<WORKDIR>/", text)

    clean_patch = text.strip()
    is_empty = len(clean_patch) == 0
    return clean_patch, is_empty
