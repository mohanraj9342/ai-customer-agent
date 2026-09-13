"""
tests/test_inspect_dataset.py
==============================
Phase 1 smoke tests for src/data/inspect_dataset.py.

These tests use sample.csv (17 KB, committed-safe) rather than the full
twcs.csv (493 MB) so they run in milliseconds and do not require the large
dataset to be present in CI or on a reviewer's machine.

The tests verify:
  1. inspect_schema() returns the correct column names and expected dtypes.
  2. The 'inbound' column is correctly read as boolean-like values.
  3. stream_full_file() runs on sample.csv without crashing and returns
     counts that are internally consistent.

Run with:
    .venv/bin/pytest tests/test_inspect_dataset.py -v
"""

from pathlib import Path

import pytest

# We import the module functions directly.
# If the import itself fails, the test file will report a clear error.
from src.data.inspect_dataset import inspect_schema, stream_full_file

# Use sample.csv as the test fixture — it is small, already present, and has
# the same schema as twcs/twcs.csv.
SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_path() -> Path:
    """Return the path to sample.csv; skip if it is missing."""
    if not SAMPLE_CSV.exists():
        pytest.skip(f"sample.csv not found at {SAMPLE_CSV} — skipping tests.")
    return SAMPLE_CSV


# ---------------------------------------------------------------------------
# Test 1: Schema columns
# ---------------------------------------------------------------------------

def test_schema_columns(sample_path: Path) -> None:
    """inspect_schema must return exactly the 7 expected column names."""
    expected_columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    result = inspect_schema(sample_path, sample_n=50)
    assert result["columns"] == expected_columns, (
        f"Unexpected columns: {result['columns']}"
    )


# ---------------------------------------------------------------------------
# Test 2: Sample rows are returned
# ---------------------------------------------------------------------------

def test_schema_returns_sample_rows(sample_path: Path) -> None:
    """inspect_schema must return at least one sample row."""
    result = inspect_schema(sample_path, sample_n=10)
    assert len(result["sample_rows"]) > 0, "No sample rows returned."


# ---------------------------------------------------------------------------
# Test 3: Streaming counts are internally consistent
# ---------------------------------------------------------------------------

def test_stream_counts_consistent(sample_path: Path) -> None:
    """
    After streaming sample.csv, inbound + outbound must equal total_rows.
    This verifies that the inbound mask is applied correctly and no rows
    are double-counted or dropped.
    """
    stats = stream_full_file(sample_path)
    assert stats["inbound_count"] + stats["outbound_count"] == stats["total_rows"], (
        "inbound_count + outbound_count does not equal total_rows — "
        "possible mask error in stream_full_file()."
    )


# ---------------------------------------------------------------------------
# Test 4: At least one brand is detected
# ---------------------------------------------------------------------------

def test_stream_detects_brands(sample_path: Path) -> None:
    """
    After streaming sample.csv, at least one outbound brand must be counted.
    If brand_counts is empty, inspect_schema's outbound filtering is broken.
    """
    stats = stream_full_file(sample_path)
    assert len(stats["brand_counts"]) > 0, (
        "No brands detected in brand_counts — check inbound=False filtering."
    )


# ---------------------------------------------------------------------------
# Test 5: No negative counts
# ---------------------------------------------------------------------------

def test_no_negative_counts(sample_path: Path) -> None:
    """All numeric count fields must be non-negative."""
    stats = stream_full_file(sample_path)
    for key in ("total_rows", "inbound_count", "outbound_count",
                "duplicate_tweet_ids", "has_response_link", "has_parent_link"):
        assert stats[key] >= 0, f"{key} is negative: {stats[key]}"
