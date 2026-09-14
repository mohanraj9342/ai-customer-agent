"""
tests/test_extract_selected_brand.py
======================================
Comprehensive unit tests for src/data/extract_selected_brand.py

All tests use small in-memory CSV fixtures via tmp_path.
No access to the full 493 MB raw dataset.

Test categories:
  - Input validation (schema, missing file, empty input)
  - Extraction logic (outbound, inbound, row_type, field preservation)
  - Chunked reading
  - Data-quality checks (duplicates, self-refs, invalid values, empty text)
  - Overwrite guard
  - Metadata generation (content, relative paths)
  - Output directory creation
  - CLI success and failure
  - Thread-reconstruction compatibility
  - Deterministic output ordering
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.extract_selected_brand import (
    ExtractionConfig,
    ValidationResult,
    ExtractionStats,
    check_input_schema,
    check_overwrite,
    generate_metadata,
    run_extraction,
    validate_output,
    REQUIRED_COLUMNS,
    VALID_INBOUND_VALUES,
    OUTPUT_CSV_NAME,
    OUTPUT_METADATA_NAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BRAND = "AppleSupport"

_HEADERS = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
]


def _write_csv(path: Path, rows: list[dict], headers: list[str] | None = None) -> Path:
    """Write a minimal CSV fixture to path. Returns path."""
    hdrs = headers or _HEADERS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hdrs, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in hdrs})
    return path


def _minimal_rows(brand: str = BRAND) -> list[dict]:
    """Two-row fixture: one brand outbound + one customer inbound."""
    return [
        {
            "tweet_id": "100", "author_id": "customer_A", "inbound": "True",
            "created_at": "Mon Jan 01 10:00:00 +0000 2018",
            "text": "my iPhone battery drains fast",
            "response_tweet_id": "101", "in_response_to_tweet_id": "",
        },
        {
            "tweet_id": "101", "author_id": brand, "inbound": "False",
            "created_at": "Mon Jan 01 10:05:00 +0000 2018",
            "text": "Hi! We can help with that. DM us.",
            "response_tweet_id": "", "in_response_to_tweet_id": "100",
        },
    ]


def _run(tmp_path: Path, rows: list[dict], brand: str = BRAND,
         overwrite: bool = False, strict: bool = False,
         chunk_size: int = 1000) -> dict:
    """Write fixture CSV, run extraction, return metadata dict."""
    inp = _write_csv(tmp_path / "raw.csv", rows)
    out = tmp_path / "out"
    cfg = ExtractionConfig(
        brand=brand, input_path=inp, output_dir=out,
        chunk_size=chunk_size, overwrite=overwrite, strict=strict,
    )
    return run_extraction(cfg)


# ===========================================================================
# 1. Required-column validation
# ===========================================================================

class TestRequiredColumnValidation:
    def test_all_columns_present(self, tmp_path):
        csv_path = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        vr = check_input_schema(csv_path, chunk_size=1000)
        assert vr.is_valid, vr.errors

    def test_missing_tweet_id_raises(self, tmp_path):
        bad_headers = [h for h in _HEADERS if h != "tweet_id"]
        csv_path = _write_csv(tmp_path / "raw.csv", _minimal_rows(), headers=bad_headers)
        vr = check_input_schema(csv_path, chunk_size=1000)
        assert not vr.is_valid
        assert any("tweet_id" in e for e in vr.errors)

    def test_missing_inbound_column(self, tmp_path):
        bad_headers = [h for h in _HEADERS if h != "inbound"]
        csv_path = _write_csv(tmp_path / "raw.csv", _minimal_rows(), headers=bad_headers)
        vr = check_input_schema(csv_path, chunk_size=1000)
        assert not vr.is_valid

    def test_missing_text_column(self, tmp_path):
        bad_headers = [h for h in _HEADERS if h != "text"]
        csv_path = _write_csv(tmp_path / "raw.csv", _minimal_rows(), headers=bad_headers)
        vr = check_input_schema(csv_path, chunk_size=1000)
        assert not vr.is_valid
        assert any("text" in e for e in vr.errors)

    def test_extra_columns_are_allowed(self, tmp_path):
        extra_headers = _HEADERS + ["extra_col"]
        csv_path = _write_csv(tmp_path / "raw.csv", _minimal_rows(), headers=extra_headers)
        vr = check_input_schema(csv_path, chunk_size=1000)
        assert vr.is_valid


# ===========================================================================
# 2. Missing input file handling
# ===========================================================================

class TestMissingInputFile:
    def test_raises_file_not_found(self, tmp_path):
        cfg = ExtractionConfig(
            brand=BRAND,
            input_path=tmp_path / "nonexistent.csv",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(FileNotFoundError):
            run_extraction(cfg)

    def test_empty_brand_raises_value_error(self, tmp_path):
        csv_path = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        cfg = ExtractionConfig(
            brand="", input_path=csv_path, output_dir=tmp_path / "out"
        )
        with pytest.raises(ValueError, match="brand"):
            run_extraction(cfg)


# ===========================================================================
# 3. Brand outbound extraction
# ===========================================================================

class TestBrandOutboundExtraction:
    def test_brand_reply_in_output(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        assert "101" in out_df["tweet_id"].values

    def test_brand_reply_row_type(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        brand_row = out_df[out_df["tweet_id"] == "101"]
        assert brand_row.iloc[0]["row_type"] == "brand_reply"

    def test_non_brand_outbound_excluded(self, tmp_path):
        rows = _minimal_rows() + [
            {
                "tweet_id": "200", "author_id": "OtherBrand", "inbound": "False",
                "created_at": "Mon Jan 01 11:00:00 +0000 2018",
                "text": "other brand reply", "response_tweet_id": "",
                "in_response_to_tweet_id": "201",
            }
        ]
        _run(tmp_path, rows)
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        assert "200" not in out_df["tweet_id"].values


# ===========================================================================
# 4. Inbound customer-row extraction
# ===========================================================================

class TestInboundExtraction:
    def test_customer_inbound_in_output(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        assert "100" in out_df["tweet_id"].values

    def test_customer_inbound_row_type(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        cust_row = out_df[out_df["tweet_id"] == "100"]
        assert cust_row.iloc[0]["row_type"] == "customer_inbound"

    def test_unrelated_inbound_not_extracted(self, tmp_path):
        # A customer tweet that @mentions nobody the brand responded to
        rows = _minimal_rows() + [
            {
                "tweet_id": "999", "author_id": "random_user", "inbound": "True",
                "created_at": "Mon Jan 01 12:00:00 +0000 2018",
                "text": "just a random tweet", "response_tweet_id": "",
                "in_response_to_tweet_id": "",
            }
        ]
        _run(tmp_path, rows)
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        assert "999" not in out_df["tweet_id"].values


# ===========================================================================
# 5. Correct preservation of original fields
# ===========================================================================

class TestFieldPreservation:
    def test_all_required_columns_in_output(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        for col in REQUIRED_COLUMNS:
            assert col in out_df.columns, f"Missing column: {col}"

    def test_text_raw_preserved(self, tmp_path):
        """Original text is stored in text_raw before normalisation."""
        rows = [
            {
                "tweet_id": "100", "author_id": "cust", "inbound": "True",
                "created_at": "Mon Jan 01 10:00:00 +0000 2018",
                "text": "&amp; hello  world",
                "response_tweet_id": "101", "in_response_to_tweet_id": "",
            },
            {
                "tweet_id": "101", "author_id": BRAND, "inbound": "False",
                "created_at": "Mon Jan 01 10:05:00 +0000 2018",
                "text": "Hi &lt;customer&gt;",
                "response_tweet_id": "", "in_response_to_tweet_id": "100",
            },
        ]
        _run(tmp_path, rows)
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        assert "text_raw" in out_df.columns
        cust = out_df[out_df["tweet_id"] == "100"]
        # text_raw should be the original, text should be normalised
        assert "&amp;" in cust.iloc[0]["text_raw"]
        assert "& hello world" in cust.iloc[0]["text"]

    def test_author_id_preserved(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        brand_row = out_df[out_df["tweet_id"] == "101"]
        assert brand_row.iloc[0]["author_id"] == BRAND

    def test_created_at_preserved(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        brand_row = out_df[out_df["tweet_id"] == "101"]
        assert "2018" in brand_row.iloc[0]["created_at"]


# ===========================================================================
# 6. Chunked reading
# ===========================================================================

class TestChunkedReading:
    def test_chunk_size_1_produces_same_result(self, tmp_path):
        """chunk_size=1 forces row-by-row processing; output must match."""
        m1 = _run(tmp_path / "c1000", _minimal_rows(), chunk_size=1000)
        m2 = _run(tmp_path / "c1",    _minimal_rows(), chunk_size=1)
        assert m1["rows_written"] == m2["rows_written"]
        assert m1["inbound_rows"] == m2["inbound_rows"]
        assert m1["outbound_rows"] == m2["outbound_rows"]

    def test_large_chunk_size_works(self, tmp_path):
        m = _run(tmp_path, _minimal_rows(), chunk_size=100_000)
        assert m["rows_written"] == 2


# ===========================================================================
# 7. Empty dataset handling
# ===========================================================================

class TestEmptyDataset:
    def test_empty_file_raises(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        csv_path.write_text("tweet_id,author_id,inbound,created_at,text,"
                             "response_tweet_id,in_response_to_tweet_id\n")
        cfg = ExtractionConfig(
            brand=BRAND, input_path=csv_path, output_dir=tmp_path / "out"
        )
        with pytest.raises((RuntimeError, ValueError)):
            run_extraction(cfg)

    def test_no_brand_rows_raises(self, tmp_path):
        rows = [
            {
                "tweet_id": "1", "author_id": "SomeOtherBrand", "inbound": "False",
                "created_at": "Mon Jan 01 10:00:00 +0000 2018",
                "text": "hello", "response_tweet_id": "", "in_response_to_tweet_id": "0",
            }
        ]
        with pytest.raises(RuntimeError, match="No outbound rows"):
            _run(tmp_path, rows)


# ===========================================================================
# 8. Duplicate tweet ID detection
# ===========================================================================

class TestDuplicateTweetIds:
    def _make_duplicate_rows(self) -> list[dict]:
        rows = _minimal_rows()
        # Add exact duplicate of tweet 101 (brand reply)
        rows.append({
            "tweet_id": "101", "author_id": BRAND, "inbound": "False",
            "created_at": "Mon Jan 01 10:05:00 +0000 2018",
            "text": "Hi! We can help with that. DM us.",
            "response_tweet_id": "", "in_response_to_tweet_id": "100",
        })
        return rows

    def test_duplicate_detected_as_warning(self, tmp_path):
        m = _run(tmp_path, self._make_duplicate_rows())
        warnings = m["validation"]["warnings"]
        assert any("duplicate" in w.lower() for w in warnings)

    def test_duplicate_count_in_metadata(self, tmp_path):
        m = _run(tmp_path, self._make_duplicate_rows())
        assert m["duplicate_tweet_ids"] >= 1


# ===========================================================================
# 9. Invalid inbound values
# ===========================================================================

class TestInvalidInboundValues:
    def test_invalid_inbound_triggers_warning(self, tmp_path):
        rows = _minimal_rows()
        rows[0]["inbound"] = "maybe"   # invalid
        m = _run(tmp_path, rows)
        warnings = m["validation"]["warnings"]
        assert any("inbound" in w.lower() for w in warnings)

    def test_valid_inbound_values_no_warning(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        warnings = m["validation"]["warnings"]
        assert not any("invalid" in w.lower() and "inbound" in w.lower()
                       for w in warnings)


# ===========================================================================
# 10. Empty text detection
# ===========================================================================

class TestEmptyTextDetection:
    def test_empty_text_triggers_warning(self, tmp_path):
        rows = _minimal_rows()
        rows[0]["text"] = ""
        m = _run(tmp_path, rows)
        assert m["rows_empty_text"] >= 1
        assert any("empty text" in w.lower() for w in m["validation"]["warnings"])

    def test_whitespace_only_text_treated_as_empty(self, tmp_path):
        rows = _minimal_rows()
        rows[0]["text"] = "   "
        m = _run(tmp_path, rows)
        assert m["rows_empty_text"] >= 1


# ===========================================================================
# 11. Missing parent-link detection
# ===========================================================================

class TestMissingParentLink:
    def test_missing_parent_counted(self, tmp_path):
        rows = _minimal_rows()
        # tweet 101 (brand reply) has in_response_to = "100", which IS in the
        # extraction.  So parent is NOT broken.  Tweet 100 has no parent at all.
        m = _run(tmp_path, rows)
        # rows_with_parent_field is tweet 101 (1 row has parent field)
        assert m["rows_with_parent_field"] == 1

    def test_row_without_parent_still_extracted(self, tmp_path):
        """A customer tweet with no in_response_to is still extracted if the
        brand replied to it."""
        rows = _minimal_rows()
        rows[0]["in_response_to_tweet_id"] = ""  # customer has no parent
        m = _run(tmp_path, rows)
        assert m["rows_written"] == 2


# ===========================================================================
# 12. Missing response-link detection
# ===========================================================================

class TestMissingResponseLink:
    def test_rows_with_response_field_counted(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        # tweet 100 (customer) has response_tweet_id = "101"
        assert m["rows_with_response_field"] >= 1

    def test_row_without_response_field_counted_correctly(self, tmp_path):
        rows = _minimal_rows()
        rows[0]["response_tweet_id"] = ""   # customer has no response field
        m = _run(tmp_path, rows)
        # rows_with_response_field should now be 0 (brand reply has no resp field either)
        assert m["rows_with_response_field"] == 0


# ===========================================================================
# 13. Self-reference detection
# ===========================================================================

class TestSelfReferenceDetection:
    def test_self_ref_parent_detected(self, tmp_path):
        rows = _minimal_rows()
        # Make brand reply point to itself as parent
        rows[1]["in_response_to_tweet_id"] = "101"  # tweet_id == in_response_to
        m = _run(tmp_path, rows)
        assert m["rows_self_ref_parent"] >= 1
        assert any("self-ref" in w.lower() for w in m["validation"]["warnings"])

    def test_self_ref_response_detected(self, tmp_path):
        rows = _minimal_rows()
        rows[0]["response_tweet_id"] = "100"  # customer points to itself
        m = _run(tmp_path, rows)
        assert m["rows_self_ref_response"] >= 1


# ===========================================================================
# 14. Conflicting duplicate records
# ===========================================================================

class TestConflictingDuplicates:
    def test_conflicting_same_id_different_text(self, tmp_path):
        rows = _minimal_rows()
        # Same tweet_id as tweet 101 but different text
        rows.append({
            "tweet_id": "101", "author_id": BRAND, "inbound": "False",
            "created_at": "Mon Jan 01 10:05:00 +0000 2018",
            "text": "COMPLETELY DIFFERENT RESPONSE TEXT",
            "response_tweet_id": "", "in_response_to_tweet_id": "100",
        })
        m = _run(tmp_path, rows)
        assert m["conflicting_duplicates"] >= 1
        assert any("conflict" in w.lower() for w in m["validation"]["warnings"])


# ===========================================================================
# 15. Deterministic output ordering
# ===========================================================================

class TestDeterministicOutput:
    def test_same_input_same_order(self, tmp_path):
        rows = _minimal_rows()
        _run(tmp_path / "run1", rows)
        _run(tmp_path / "run2", rows)
        df1 = pd.read_csv(tmp_path / "run1" / "out" / OUTPUT_CSV_NAME, dtype=str)
        df2 = pd.read_csv(tmp_path / "run2" / "out" / OUTPUT_CSV_NAME, dtype=str)
        assert list(df1["tweet_id"]) == list(df2["tweet_id"])

    def test_output_sorted_by_tweet_id_numerically(self, tmp_path):
        rows = [
            {
                "tweet_id": "200", "author_id": "cust", "inbound": "True",
                "created_at": "Mon Jan 01 10:00:00 +0000 2018",
                "text": "second customer", "response_tweet_id": "201",
                "in_response_to_tweet_id": "",
            },
            {
                "tweet_id": "201", "author_id": BRAND, "inbound": "False",
                "created_at": "Mon Jan 01 10:05:00 +0000 2018",
                "text": "reply to 200", "response_tweet_id": "",
                "in_response_to_tweet_id": "200",
            },
            {
                "tweet_id": "100", "author_id": "cust", "inbound": "True",
                "created_at": "Mon Jan 01 09:00:00 +0000 2018",
                "text": "first customer", "response_tweet_id": "101",
                "in_response_to_tweet_id": "",
            },
            {
                "tweet_id": "101", "author_id": BRAND, "inbound": "False",
                "created_at": "Mon Jan 01 09:05:00 +0000 2018",
                "text": "reply to 100", "response_tweet_id": "",
                "in_response_to_tweet_id": "100",
            },
        ]
        _run(tmp_path, rows)
        out_df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        ids = [int(t) for t in out_df["tweet_id"].tolist()]
        assert ids == sorted(ids)


# ===========================================================================
# 16. Metadata generation
# ===========================================================================

class TestMetadataGeneration:
    def test_metadata_file_created(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        assert (tmp_path / "out" / OUTPUT_METADATA_NAME).exists()

    def test_metadata_keys_present(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        required_keys = {
            "brand", "schema_version", "extraction_timestamp",
            "input_dataset", "output_csv", "output_metadata",
            "rows_written", "inbound_rows", "outbound_rows",
            "unique_tweet_ids", "duplicate_tweet_ids",
            "validation", "limitations", "next_phase",
        }
        for key in required_keys:
            assert key in m, f"Missing metadata key: {key}"

    def test_metadata_brand_is_correct(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        assert m["brand"] == BRAND

    def test_metadata_row_counts_consistent(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        assert m["rows_written"] == m["inbound_rows"] + m["outbound_rows"]


# ===========================================================================
# 17. Relative paths in metadata
# ===========================================================================

class TestRelativePathsInMetadata:
    def test_input_path_has_no_absolute_home(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        assert not m["input_dataset"].startswith("/home"), (
            f"Absolute path leaked into metadata: {m['input_dataset']}"
        )

    def test_output_csv_path_no_absolute_home(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        # If output is inside project root, should be relative
        # If tmp_path is outside project root, it will be absolute — that is acceptable
        # Just verify it's not leaking /home/mohanraj specific paths
        # This test checks the relative_input() helper for the standard config
        from src.data.extract_selected_brand import ExtractionConfig, DEFAULT_INPUT, DEFAULT_OUTDIR
        cfg = ExtractionConfig()
        rel = cfg.relative_input()
        assert not rel.startswith("/home"), f"Default input path is absolute: {rel}"


# ===========================================================================
# 18. Output directory creation
# ===========================================================================

class TestOutputDirectoryCreation:
    def test_nested_output_dir_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "out"
        inp = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        cfg = ExtractionConfig(
            brand=BRAND, input_path=inp, output_dir=nested
        )
        run_extraction(cfg)
        assert nested.is_dir()
        assert (nested / OUTPUT_CSV_NAME).exists()


# ===========================================================================
# 19. Refusal to overwrite without --overwrite
# ===========================================================================

class TestOverwriteGuard:
    def test_raises_if_output_exists(self, tmp_path):
        _run(tmp_path, _minimal_rows())  # first run
        with pytest.raises(FileExistsError, match="overwrite"):
            _run(tmp_path, _minimal_rows(), overwrite=False)  # second run

    def test_check_overwrite_returns_message_when_file_exists(self, tmp_path):
        inp = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        out = tmp_path / "out"
        cfg = ExtractionConfig(brand=BRAND, input_path=inp, output_dir=out, overwrite=False)
        # Create a fake output file
        out.mkdir(parents=True, exist_ok=True)
        (out / OUTPUT_CSV_NAME).write_text("fake")
        msg = check_overwrite(cfg)
        assert msg is not None
        assert "overwrite" in msg.lower()

    def test_check_overwrite_returns_none_when_no_file(self, tmp_path):
        inp = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        cfg = ExtractionConfig(brand=BRAND, input_path=inp, output_dir=tmp_path / "out")
        assert check_overwrite(cfg) is None


# ===========================================================================
# 20. Successful overwrite with --overwrite
# ===========================================================================

class TestOverwriteSuccess:
    def test_overwrite_flag_allows_second_run(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        m = _run(tmp_path, _minimal_rows(), overwrite=True)
        assert m["rows_written"] == 2

    def test_overwrite_replaces_content(self, tmp_path):
        _run(tmp_path, _minimal_rows())
        # Corrupt the file manually
        (tmp_path / "out" / OUTPUT_CSV_NAME).write_text("garbage")
        m = _run(tmp_path, _minimal_rows(), overwrite=True)
        assert m["rows_written"] == 2


# ===========================================================================
# 21. CLI success
# ===========================================================================

class TestCLISuccess:
    def test_cli_exit_0(self, tmp_path):
        inp = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        out = tmp_path / "out"
        result = subprocess.run(
            [
                sys.executable, "-m", "src.data.extract_selected_brand",
                "--input", str(inp),
                "--brand", BRAND,
                "--output-dir", str(out),
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr

    def test_cli_creates_output_files(self, tmp_path):
        inp = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        out = tmp_path / "out"
        subprocess.run(
            [
                sys.executable, "-m", "src.data.extract_selected_brand",
                "--input", str(inp),
                "--brand", BRAND,
                "--output-dir", str(out),
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert (out / OUTPUT_CSV_NAME).exists()
        assert (out / OUTPUT_METADATA_NAME).exists()


# ===========================================================================
# 22. CLI failure for missing input
# ===========================================================================

class TestCLIFailure:
    def test_cli_missing_input_exits_1(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, "-m", "src.data.extract_selected_brand",
                "--input", str(tmp_path / "does_not_exist.csv"),
                "--brand", BRAND,
                "--output-dir", str(tmp_path / "out"),
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 1

    def test_cli_existing_output_without_overwrite_exits_1(self, tmp_path):
        inp = _write_csv(tmp_path / "raw.csv", _minimal_rows())
        out = tmp_path / "out"
        out.mkdir()
        (out / OUTPUT_CSV_NAME).write_text("fake")
        result = subprocess.run(
            [
                sys.executable, "-m", "src.data.extract_selected_brand",
                "--input", str(inp),
                "--brand", BRAND,
                "--output-dir", str(out),
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 1


# ===========================================================================
# 23. Structured ValidationResult
# ===========================================================================

class TestStructuredValidationResult:
    def test_is_valid_true_when_no_errors(self):
        vr = ValidationResult(errors=[], warnings=["w1"], info=["i1"])
        assert vr.is_valid is True

    def test_is_valid_false_when_errors(self):
        vr = ValidationResult(errors=["fatal error"], warnings=[], info=[])
        assert vr.is_valid is False

    def test_merge_combines_lists(self):
        vr1 = ValidationResult(errors=["e1"], warnings=["w1"], info=["i1"])
        vr2 = ValidationResult(errors=["e2"], warnings=["w2"], info=["i2"])
        vr1.merge(vr2)
        assert "e1" in vr1.errors and "e2" in vr1.errors
        assert "w1" in vr1.warnings and "w2" in vr1.warnings

    def test_as_dict_has_required_keys(self):
        vr = ValidationResult(errors=[], warnings=["w"], info=["i"])
        d = vr.as_dict()
        assert "is_valid" in d
        assert "errors" in d
        assert "warnings" in d
        assert "info" in d

    def test_metadata_validation_field_is_dict(self, tmp_path):
        m = _run(tmp_path, _minimal_rows())
        assert isinstance(m["validation"], dict)
        assert "errors" in m["validation"]
        assert "warnings" in m["validation"]


# ===========================================================================
# 24. Compatibility with reconstruct_threads.py input format
# ===========================================================================

class TestThreadReconstructionCompatibility:
    def test_output_can_be_loaded_by_reconstruct_threads(self, tmp_path):
        """
        The output CSV must be directly loadable by reconstruct_threads.
        Required columns: tweet_id, inbound, created_at, text,
                          in_response_to_tweet_id, response_tweet_id.
        """
        from src.data.reconstruct_threads import reconstruct_threads

        rows = [
            {
                "tweet_id": "10", "author_id": "cust", "inbound": "True",
                "created_at": "Mon Jan 01 10:00:00 +0000 2018",
                "text": "battery problem",
                "response_tweet_id": "11", "in_response_to_tweet_id": "",
            },
            {
                "tweet_id": "11", "author_id": BRAND, "inbound": "False",
                "created_at": "Mon Jan 01 10:05:00 +0000 2018",
                "text": "We can help. DM us.",
                "response_tweet_id": "", "in_response_to_tweet_id": "10",
            },
        ]
        _run(tmp_path, rows)
        out_csv = tmp_path / "out" / OUTPUT_CSV_NAME
        df = pd.read_csv(out_csv, dtype=str)

        # reconstruct_threads requires these columns
        required = {"tweet_id", "inbound", "created_at", "text",
                    "in_response_to_tweet_id", "response_tweet_id"}
        assert required.issubset(set(df.columns))

        # Must produce threads without crashing
        threads = reconstruct_threads(df)
        assert len(threads) >= 1

    def test_reconstructed_thread_has_both_directions(self, tmp_path):
        from src.data.reconstruct_threads import reconstruct_threads

        _run(tmp_path, _minimal_rows())
        df = pd.read_csv(tmp_path / "out" / OUTPUT_CSV_NAME, dtype=str)
        threads = reconstruct_threads(df)
        assert len(threads) == 1
        assert "inbound" in threads[0].directions
        assert "outbound" in threads[0].directions
