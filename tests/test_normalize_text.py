"""Unit tests for text normalization."""

import pytest
from tools.normalize_text import normalize_log_text, normalize_patch_text, strip_ansi


def test_strip_ansi():
    raw = "\x1b[31mError:\x1b[0m test failed\x1b[1m!"
    clean = strip_ansi(raw)
    assert clean == "Error: test failed!"


def test_normalize_log_text():
    raw = (
        "cd /testbed/astropy__astropy && pytest astropy/tests/test_units.py\n"
        "\x1b[31mFAILED\x1b[0m (errors=1)"
    )
    normalized = normalize_log_text(
        raw,
        instance_id="astropy__astropy-12907",
        repo_name="astropy__astropy",
    )
    assert "\x1b" not in normalized
    assert "<WORKDIR>" in normalized
    assert "FAILED" in normalized  # Preserves error keyword
    assert "errors=1" in normalized  # Preserves code/test numbers
    assert "pytest" in normalized


def test_normalize_patch_empty():
    patch_empty, is_empty = normalize_patch_text("", "instance-1", "repo-1")
    assert patch_empty == ""
    assert is_empty is True

    patch_none, is_empty_none = normalize_patch_text(None, "instance-1", "repo-1")
    assert patch_none == ""
    assert is_empty_none is True


def test_normalize_patch_real():
    diff = (
        "--- a/testbed/django/db/models/query.py\n"
        "+++ b/testbed/django/db/models/query.py\n"
        "@@ -10,6 +10,7 @@\n"
        "+    return True\n"
    )
    clean, is_empty = normalize_patch_text(diff, "django__django-11001", "django__django")
    assert is_empty is False
    assert "+    return True" in clean
    assert "<WORKDIR>" in clean
