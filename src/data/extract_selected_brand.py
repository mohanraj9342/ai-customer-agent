"""
src/data/extract_selected_brand.py
====================================
Phase 3 — Production-quality AppleSupport data extraction.

PURPOSE
-------
Extract all records associated with the selected brand (AppleSupport) from the
raw Twitter Customer Support dataset and write a clean, validated message-level
CSV artifact.  This module is intentionally scoped to *message extraction only*.
Thread reconstruction is a separate downstream step (Phase 4).

WHAT "AppleSupport EXTRACTION" MEANS
-------------------------------------
The extraction collects two categories of rows:

  brand_reply   — rows where author_id == "AppleSupport" and inbound == False.
                  These are the brand's actual support responses.

  customer_inbound — rows whose tweet_id appears in the in_response_to_tweet_id
                     field of a brand_reply row.  These are the customer
                     messages the brand directly replied to.

Rows are labelled with a ``row_type`` column so downstream stages can filter
without re-parsing author_id or inbound.

What is NOT extracted (and why):
  - Customer tweets that @mention AppleSupport but received no response are not
    in the extraction subset.  They cannot be identified without a full text scan
    that is much more expensive.  The brand_comparison_validated.py analysis
    estimates ~19,000 such tweets via an @mention heuristic; this extraction
    does not attempt to recover them.
  - Multi-level conversation ancestors (grandparent tweets) are not collected.
    Only the direct parent (the customer tweet a brand reply responded to) is
    included.  Full thread context is built in Phase 4 using
    reconstruct_threads.py.

PIPELINE
--------
  Pass 1 — stream the raw CSV, collect:
              brand_reply_ids  (tweet_ids the brand authored)
              parent_ids       (tweet_ids those replies responded to)

  Pass 2 — stream the raw CSV a second time, emit matched rows to output CSV
              Apply text normalisation (from extract_brand.normalise_text).
              Assign row_type.
              Sort entire output by tweet_id for determinism.

  Pass 3 — read output CSV back in chunks, run all validation checks,
              collect structured ValidationResult.

  Metadata — generate apple_support_metadata.json with relative paths only.

VALIDATION
----------
Validation is divided into three levels:
  ERROR   — structural problems that make the data unusable (e.g. missing
             required column).  Extraction exits with code 1 if any errors
             are present and --strict is set (or always for structural errors).
  WARNING — data quality issues that do not prevent processing but should be
             logged (e.g. duplicate tweet_ids, self-referencing links).
  INFO    — statistics and observations.

OUTPUT ARTIFACTS
----------------
  data/processed/apple_support/apple_support_messages.csv
  data/processed/apple_support/apple_support_metadata.json

  Thread reconstruction artifact (apple_support_threads.jsonl) is produced
  by the separate reconstruct_threads.py step in Phase 4.

CLI USAGE
---------
  .venv/bin/python -m src.data.extract_selected_brand \\
      --input twcs/twcs.csv \\
      --brand AppleSupport \\
      --output-dir data/processed/apple_support

  Options:
    --input       PATH   Path to the raw CSV (default: twcs/twcs.csv)
    --brand       STR    Brand author_id (default: AppleSupport)
    --output-dir  PATH   Output directory (default: data/processed/apple_support)
    --chunk-size  INT    Rows per pandas read chunk (default: 50000)
    --overwrite          Allow overwriting existing output files
    --strict             Treat warnings as errors (exit 1)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Re-use the normalise_text function and ID collector from Phase 2
from src.data.extract_brand import normalise_text, _collect_brand_tweet_ids

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("  %(levelname)s  %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "twcs" / "twcs.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "processed" / "apple_support"
DEFAULT_BRAND  = "AppleSupport"
DEFAULT_CHUNK  = 50_000
SCHEMA_VERSION = "3.0"

REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}

VALID_INBOUND_VALUES = {"True", "False"}

OUTPUT_CSV_NAME      = "apple_support_messages.csv"
OUTPUT_METADATA_NAME = "apple_support_metadata.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """All parameters that control a single extraction run."""
    brand:       str  = DEFAULT_BRAND
    input_path:  Path = field(default_factory=lambda: DEFAULT_INPUT)
    output_dir:  Path = field(default_factory=lambda: DEFAULT_OUTDIR)
    chunk_size:  int  = DEFAULT_CHUNK
    overwrite:   bool = False
    strict:      bool = False

    def output_csv(self) -> Path:
        return self.output_dir / OUTPUT_CSV_NAME

    def output_metadata(self) -> Path:
        return self.output_dir / OUTPUT_METADATA_NAME

    def relative_input(self) -> str:
        """Return input path relative to project root (for metadata)."""
        try:
            return str(self.input_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.input_path)

    def relative_output_csv(self) -> str:
        try:
            return str(self.output_csv().resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.output_csv())

    def relative_output_metadata(self) -> str:
        try:
            return str(self.output_metadata().resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.output_metadata())


@dataclass
class ValidationResult:
    """
    Structured output of one validation pass.

    Attributes
    ----------
    errors   : fatal problems — data is unusable or structurally broken.
    warnings : non-fatal quality issues — data is processable but imperfect.
    info     : statistics and observations.
    is_valid : True when errors is empty.
    """
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info:     list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: "ValidationResult") -> None:
        """Merge another ValidationResult into this one in-place."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)

    def as_dict(self) -> dict:
        return {
            "is_valid":  self.is_valid,
            "errors":    self.errors,
            "warnings":  self.warnings,
            "info":      self.info,
        }


@dataclass
class ExtractionStats:
    """Counters produced during the two streaming passes."""
    total_rows_scanned:      int = 0
    brand_reply_ids_found:   int = 0
    parent_ids_found:        int = 0
    keep_set_size:           int = 0
    rows_written:            int = 0
    inbound_rows:            int = 0
    outbound_rows:           int = 0
    unique_tweet_ids:        int = 0
    duplicate_tweet_ids:     int = 0
    rows_with_parent_field:  int = 0
    rows_with_response_field:int = 0
    rows_self_ref_parent:    int = 0
    rows_self_ref_response:  int = 0
    rows_empty_text:         int = 0
    rows_missing_timestamp:  int = 0
    rows_invalid_inbound:    int = 0
    conflicting_duplicates:  int = 0
    elapsed_pass1_s:         float = 0.0
    elapsed_pass2_s:         float = 0.0
    elapsed_pass3_s:         float = 0.0

    def total_elapsed(self) -> float:
        return self.elapsed_pass1_s + self.elapsed_pass2_s + self.elapsed_pass3_s


# ---------------------------------------------------------------------------
# Pass 1 — Schema pre-check (single chunk)
# ---------------------------------------------------------------------------

def check_input_schema(path: Path, chunk_size: int) -> ValidationResult:
    """
    Read only the first chunk of the raw CSV and verify that all required
    columns are present.  Fail fast before the expensive two-pass extraction.

    Returns a ValidationResult.  If errors is non-empty the extraction should
    not proceed.
    """
    vr = ValidationResult()
    try:
        chunk = next(
            pd.read_csv(path, chunksize=chunk_size, dtype=str,
                        encoding="utf-8", on_bad_lines="skip", low_memory=False)
        )
    except StopIteration:
        vr.errors.append("Input file is empty — no rows could be read.")
        return vr
    except Exception as exc:
        vr.errors.append(f"Could not read input file: {exc}")
        return vr

    missing = REQUIRED_COLUMNS - set(chunk.columns)
    if missing:
        vr.errors.append(
            f"Required columns missing from input: {sorted(missing)}"
        )
    else:
        vr.info.append(
            f"All {len(REQUIRED_COLUMNS)} required columns present in input."
        )

    extra = set(chunk.columns) - REQUIRED_COLUMNS
    if extra:
        vr.info.append(
            f"Input has {len(extra)} extra column(s) beyond required set: "
            f"{sorted(extra)[:5]}{'…' if len(extra) > 5 else ''}"
        )

    return vr


# ---------------------------------------------------------------------------
# Pass 2 — Emit matched rows to output CSV
# ---------------------------------------------------------------------------

def _emit_rows(
    path: Path,
    brand: str,
    brand_reply_ids: set[str],
    parent_ids: set[str],
    output_path: Path,
    chunk_size: int,
) -> ExtractionStats:
    """
    Stream the raw CSV and emit matched rows to ``output_path``.

    Collects all matched rows into memory for deterministic sort before writing.
    Memory budget: ~213k rows × ~300 bytes = ~64 MB for AppleSupport — safe.

    The output CSV gains a ``row_type`` column:
      "brand_reply"      — brand-authored outbound row
      "customer_inbound" — customer tweet the brand replied to
    """
    stats = ExtractionStats()
    stats.brand_reply_ids_found = len(brand_reply_ids)
    stats.parent_ids_found      = len(parent_ids)
    keep_set = brand_reply_ids | parent_ids
    stats.keep_set_size = len(keep_set)

    collected: list[dict] = []

    reader = pd.read_csv(
        path, chunksize=chunk_size, dtype=str,
        encoding="utf-8", on_bad_lines="skip", low_memory=False,
    )
    for chunk in reader:
        stats.total_rows_scanned += len(chunk)
        mask = chunk["tweet_id"].str.strip().isin(keep_set)
        matched = chunk[mask].copy()
        if matched.empty:
            continue

        # Normalise inbound flag
        matched["inbound"] = (
            matched["inbound"].str.strip().str.lower()
            .map({"true": "True", "false": "False"})
            .fillna("Unknown")
        )

        # Strip IDs
        for col in ("tweet_id", "author_id",
                    "response_tweet_id", "in_response_to_tweet_id"):
            if col in matched.columns:
                matched[col] = matched[col].str.strip()

        # Add original text as text_raw, then normalise text
        matched.insert(
            matched.columns.get_loc("text") + 1,
            "text_raw",
            matched["text"],
        )
        matched["text"] = matched["text"].apply(normalise_text)

        # Assign row_type
        matched["row_type"] = matched.apply(
            lambda r: (
                "brand_reply"
                if r["tweet_id"] in brand_reply_ids
                else "customer_inbound"
            ),
            axis=1,
        )

        for _, row in matched.iterrows():
            collected.append(row.to_dict())

    # Sort deterministically by tweet_id (numeric if possible)
    def sort_key(r: dict) -> tuple:
        tid = str(r.get("tweet_id", ""))
        try:
            return (0, int(tid))
        except ValueError:
            return (1, tid)

    collected.sort(key=sort_key)

    # Write in one shot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if collected:
        out_df = pd.DataFrame(collected)
        out_df.to_csv(output_path, index=False, encoding="utf-8")

        stats.rows_written  = len(out_df)
        stats.inbound_rows  = int((out_df["inbound"] == "True").sum())
        stats.outbound_rows = int((out_df["inbound"] == "False").sum())
    else:
        # Write empty file with header
        pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["text_raw", "row_type"]
                    ).to_csv(output_path, index=False, encoding="utf-8")

    return stats


# ---------------------------------------------------------------------------
# Pass 3 — Validate output CSV and collect detailed stats
# ---------------------------------------------------------------------------

def validate_output(
    output_path: Path,
    brand: str,
    brand_reply_ids: set[str],
    chunk_size: int,
) -> tuple[ValidationResult, ExtractionStats]:
    """
    Read the output CSV back in chunks and run all validation checks.

    Returns (ValidationResult, partial ExtractionStats with validation counts).
    """
    vr = ValidationResult()
    vstats = ExtractionStats()

    # --- structural check ---
    missing_cols = REQUIRED_COLUMNS - set(
        pd.read_csv(output_path, nrows=0, dtype=str).columns
    )
    if missing_cols:
        vr.errors.append(
            f"Output CSV missing required columns: {sorted(missing_cols)}"
        )
        return vr, vstats

    # We need to collect tweet_ids to detect duplicates and conflicts
    all_tids: list[str] = []
    tid_to_rows: dict[str, list[dict]] = {}

    reader = pd.read_csv(
        output_path, chunksize=chunk_size, dtype=str,
        encoding="utf-8", on_bad_lines="skip", low_memory=False,
    )
    for chunk in reader:
        for _, row in chunk.iterrows():
            tid  = str(row.get("tweet_id", "")).strip()
            text = str(row.get("text",     "")).strip()
            ts   = str(row.get("created_at","")).strip()
            inb  = str(row.get("inbound",  "")).strip()
            par  = str(row.get("in_response_to_tweet_id", "")).strip()
            resp = str(row.get("response_tweet_id", "")).strip()

            all_tids.append(tid)

            # Track for duplicate/conflict detection
            if tid not in tid_to_rows:
                tid_to_rows[tid] = []
            tid_to_rows[tid].append(row.to_dict())

            # Empty tweet_id
            if not tid or tid.lower() == "nan":
                vr.errors.append("Row with empty tweet_id found.")

            # Empty text
            if not text or text.lower() == "nan":
                vstats.rows_empty_text += 1

            # Missing timestamp
            if not ts or ts.lower() == "nan":
                vstats.rows_missing_timestamp += 1

            # Invalid inbound value
            if inb not in VALID_INBOUND_VALUES:
                vstats.rows_invalid_inbound += 1

            # Self-referencing parent link
            if par and par.lower() != "nan" and par == tid:
                vstats.rows_self_ref_parent += 1

            # Self-referencing response link
            if resp and resp.lower() != "nan":
                for r_id in resp.split(","):
                    if r_id.strip() == tid:
                        vstats.rows_self_ref_response += 1
                        break

            # Parent field present
            if par and par.lower() != "nan":
                vstats.rows_with_parent_field += 1

            # Response field present
            if resp and resp.lower() != "nan":
                vstats.rows_with_response_field += 1

    # Duplicate and conflict analysis
    all_subset_ids = set(all_tids)
    vstats.unique_tweet_ids   = len(all_subset_ids)
    vstats.duplicate_tweet_ids = len(all_tids) - len(all_subset_ids)

    conflict_count = 0
    for tid, rows in tid_to_rows.items():
        if len(rows) > 1:
            # Check if the duplicate rows have differing critical fields
            texts = set(r.get("text", "") for r in rows)
            auths = set(r.get("author_id", "") for r in rows)
            if len(texts) > 1 or len(auths) > 1:
                conflict_count += 1
    vstats.conflicting_duplicates = conflict_count

    # High-level validation checks
    if not all_tids:
        vr.errors.append("Output CSV contains no data rows.")
        return vr, vstats

    # Brand outbound rows
    brand_rows_in_output = sum(
        1 for tid, rows in tid_to_rows.items()
        if tid in brand_reply_ids
    )
    if brand_rows_in_output == 0:
        vr.errors.append(
            f"No outbound rows for brand '{brand}' found in output. "
            "Check brand name matches author_id exactly."
        )
    else:
        vr.info.append(
            f"Brand outbound rows present: {brand_rows_in_output:,}"
        )

    # Inbound rows
    inbound_in_output = sum(
        1 for rows in tid_to_rows.values()
        for r in rows
        if str(r.get("inbound", "")).strip() == "True"
    )
    if inbound_in_output == 0:
        vr.warnings.append("No inbound (customer) rows found in output.")
    else:
        vr.info.append(
            f"Customer inbound rows present: {inbound_in_output:,}"
        )

    # Duplicate tweet_ids
    if vstats.duplicate_tweet_ids > 0:
        vr.warnings.append(
            f"{vstats.duplicate_tweet_ids:,} duplicate tweet_id value(s) detected."
        )

    # Conflicting duplicates (same ID, different content)
    if vstats.conflicting_duplicates > 0:
        vr.warnings.append(
            f"{vstats.conflicting_duplicates:,} tweet_id(s) have conflicting "
            "records (same ID, different text or author)."
        )

    # Empty text
    if vstats.rows_empty_text > 0:
        vr.warnings.append(
            f"{vstats.rows_empty_text:,} rows have empty text after normalisation."
        )

    # Missing timestamps
    if vstats.rows_missing_timestamp > 0:
        vr.warnings.append(
            f"{vstats.rows_missing_timestamp:,} rows have no created_at timestamp."
        )

    # Invalid inbound values
    if vstats.rows_invalid_inbound > 0:
        vr.warnings.append(
            f"{vstats.rows_invalid_inbound:,} rows have inbound values outside "
            f"{sorted(VALID_INBOUND_VALUES)}."
        )

    # Self-referencing links
    if vstats.rows_self_ref_parent > 0:
        vr.warnings.append(
            f"{vstats.rows_self_ref_parent:,} rows where in_response_to_tweet_id "
            "equals the row's own tweet_id (self-reference)."
        )
    if vstats.rows_self_ref_response > 0:
        vr.warnings.append(
            f"{vstats.rows_self_ref_response:,} rows where response_tweet_id "
            "equals the row's own tweet_id (self-reference)."
        )

    # Reconstruction sufficiency
    if vstats.unique_tweet_ids < 100:
        vr.warnings.append(
            f"Only {vstats.unique_tweet_ids} unique tweets extracted — "
            "this may be insufficient for classification or retrieval."
        )
    else:
        vr.info.append(
            f"Unique tweet_ids: {vstats.unique_tweet_ids:,} — "
            "sufficient for downstream stages."
        )

    # Informational stats
    vr.info.append(
        f"Rows with parent link field: {vstats.rows_with_parent_field:,}"
    )
    vr.info.append(
        f"Rows with response link field: {vstats.rows_with_response_field:,}"
    )
    vr.info.append(
        "NOTE: Link field present ≠ referenced record exists. "
        "Thread reconstruction (Phase 4) will detect broken links."
    )
    vr.info.append(
        "NOTE: Customer tweets that @mentioned the brand but were not "
        "replied to are NOT in this extraction. They cannot be recovered "
        "without a full text scan of the raw dataset."
    )

    return vr, vstats


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

def generate_metadata(
    config: ExtractionConfig,
    stats: ExtractionStats,
    validation: ValidationResult,
    timestamp: str,
) -> dict:
    """
    Build the machine-readable metadata dict.
    All paths are relative to the project root.
    """
    return {
        "schema_version":        SCHEMA_VERSION,
        "extraction_timestamp":  timestamp,
        "brand":                 config.brand,

        # Paths — relative only, no absolute local paths
        "input_dataset":         config.relative_input(),
        "output_csv":            config.relative_output_csv(),
        "output_metadata":       config.relative_output_metadata(),

        # Configuration
        "chunk_size":            config.chunk_size,
        "chunked_processing":    True,

        # Volume
        "total_rows_scanned":    stats.total_rows_scanned,
        "brand_reply_ids_found": stats.brand_reply_ids_found,
        "parent_ids_found":      stats.parent_ids_found,
        "keep_set_size":         stats.keep_set_size,
        "rows_written":          stats.rows_written,
        "inbound_rows":          stats.inbound_rows,
        "outbound_rows":         stats.outbound_rows,

        # Quality counts (from validation pass)
        "unique_tweet_ids":          stats.unique_tweet_ids,
        "duplicate_tweet_ids":       stats.duplicate_tweet_ids,
        "conflicting_duplicates":    stats.conflicting_duplicates,
        "rows_with_parent_field":    stats.rows_with_parent_field,
        "rows_with_response_field":  stats.rows_with_response_field,
        "rows_empty_text":           stats.rows_empty_text,
        "rows_missing_timestamp":    stats.rows_missing_timestamp,
        "rows_invalid_inbound":      stats.rows_invalid_inbound,
        "rows_self_ref_parent":      stats.rows_self_ref_parent,
        "rows_self_ref_response":    stats.rows_self_ref_response,

        # Performance
        "elapsed_pass1_s":  round(stats.elapsed_pass1_s, 1),
        "elapsed_pass2_s":  round(stats.elapsed_pass2_s, 1),
        "elapsed_pass3_s":  round(stats.elapsed_pass3_s, 1),
        "total_elapsed_s":  round(stats.total_elapsed(), 1),

        # Validation summary
        "validation": validation.as_dict(),

        # Documented limitations
        "limitations": [
            "Customer tweets that @mentioned the brand but received no reply "
            "are not in this extraction.",
            "Only direct parents (one level up) are collected per brand reply. "
            "Full thread ancestry requires Phase 4 reconstruction.",
            "Link field presence does not guarantee the referenced record "
            "exists; broken links are detected in Phase 4.",
            "Text normalisation decodes HTML entities and collapses whitespace "
            "only; @mentions and URLs are preserved for downstream use.",
        ],

        # Next pipeline step
        "next_phase": (
            "Phase 4: run src/data/reconstruct_threads.py on "
            + config.relative_output_csv()
            + " to build apple_support_threads.jsonl"
        ),
    }


# ---------------------------------------------------------------------------
# Overwrite guard
# ---------------------------------------------------------------------------

def check_overwrite(config: ExtractionConfig) -> Optional[str]:
    """
    Return an error string if output files exist and --overwrite was not set.
    Return None if it is safe to proceed.
    """
    existing = [
        p for p in (config.output_csv(), config.output_metadata())
        if p.exists()
    ]
    if existing and not config.overwrite:
        return (
            f"Output file(s) already exist: "
            f"{[str(p.name) for p in existing]}. "
            "Use --overwrite to replace them."
        )
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_extraction(config: ExtractionConfig) -> dict:
    """
    Run the full Phase 3 extraction pipeline.

    Parameters
    ----------
    config : ExtractionConfig

    Returns
    -------
    dict
        Metadata dict (same as written to apple_support_metadata.json).

    Raises
    ------
    FileNotFoundError
        If config.input_path does not exist.
    ValueError
        If config.brand is empty, or if the input schema is invalid.
    RuntimeError
        If the extraction produces no output rows.
    """
    if not config.brand or not config.brand.strip():
        raise ValueError("config.brand must be a non-empty string.")

    if not config.input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {config.input_path}. "
            "Place the raw CSV there and retry."
        )

    # Overwrite guard
    guard_msg = check_overwrite(config)
    if guard_msg:
        raise FileExistsError(guard_msg)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    log.info("Brand extraction: %s", config.brand)
    log.info("Input:            %s", config.relative_input())
    log.info("Output dir:       %s", config.relative_output_csv())
    log.info("Chunk size:       %s rows", f"{config.chunk_size:,}")

    stats = ExtractionStats()

    # ── Schema pre-check ────────────────────────────────────────────────────
    log.info("Schema check …")
    schema_vr = check_input_schema(config.input_path, config.chunk_size)
    if not schema_vr.is_valid:
        raise ValueError(
            "Input schema errors: " + "; ".join(schema_vr.errors)
        )

    # ── Pass 1 ──────────────────────────────────────────────────────────────
    log.info("Pass 1 — collecting brand tweet IDs …")
    t0 = time.time()
    brand_reply_ids, parent_ids = _collect_brand_tweet_ids(
        config.input_path, config.brand
    )
    t1 = time.time()
    stats.elapsed_pass1_s      = t1 - t0
    stats.brand_reply_ids_found = len(brand_reply_ids)
    stats.parent_ids_found      = len(parent_ids)
    stats.keep_set_size         = len(brand_reply_ids | parent_ids)

    log.info("  Brand reply IDs:    %s", f"{len(brand_reply_ids):>8,}")
    log.info("  Customer tweet IDs: %s", f"{len(parent_ids):>8,}")
    log.info("  Keep set total:     %s", f"{stats.keep_set_size:>8,}")
    log.info("  Pass 1 elapsed:     %.1fs", stats.elapsed_pass1_s)

    if not brand_reply_ids:
        raise RuntimeError(
            f"No outbound rows found for brand '{config.brand}'. "
            "Verify the brand name matches author_id exactly in the dataset."
        )

    # ── Pass 2 ──────────────────────────────────────────────────────────────
    log.info("Pass 2 — emitting matched rows …")
    t2 = time.time()
    emit_stats = _emit_rows(
        config.input_path,
        config.brand,
        brand_reply_ids,
        parent_ids,
        config.output_csv(),
        config.chunk_size,
    )
    t3 = time.time()
    stats.elapsed_pass2_s       = t3 - t2
    stats.total_rows_scanned    = emit_stats.total_rows_scanned
    stats.rows_written          = emit_stats.rows_written
    stats.inbound_rows          = emit_stats.inbound_rows
    stats.outbound_rows         = emit_stats.outbound_rows

    log.info("  Rows scanned:  %s", f"{stats.total_rows_scanned:>8,}")
    log.info("  Rows written:  %s", f"{stats.rows_written:>8,}")
    log.info("  Inbound:       %s", f"{stats.inbound_rows:>8,}")
    log.info("  Outbound:      %s", f"{stats.outbound_rows:>8,}")
    log.info("  Pass 2 elapsed:     %.1fs", stats.elapsed_pass2_s)

    if stats.rows_written == 0:
        raise RuntimeError(
            "Extraction produced 0 rows. "
            "Check that the brand name and dataset path are correct."
        )

    # ── Pass 3 — Validation ─────────────────────────────────────────────────
    log.info("Pass 3 — validating output …")
    t4 = time.time()
    validation, vstats = validate_output(
        config.output_csv(), config.brand, brand_reply_ids, config.chunk_size
    )
    t5 = time.time()
    stats.elapsed_pass3_s        = t5 - t4
    stats.unique_tweet_ids       = vstats.unique_tweet_ids
    stats.duplicate_tweet_ids    = vstats.duplicate_tweet_ids
    stats.conflicting_duplicates = vstats.conflicting_duplicates
    stats.rows_with_parent_field = vstats.rows_with_parent_field
    stats.rows_with_response_field = vstats.rows_with_response_field
    stats.rows_empty_text        = vstats.rows_empty_text
    stats.rows_missing_timestamp = vstats.rows_missing_timestamp
    stats.rows_invalid_inbound   = vstats.rows_invalid_inbound
    stats.rows_self_ref_parent   = vstats.rows_self_ref_parent
    stats.rows_self_ref_response = vstats.rows_self_ref_response

    # Merge schema info into validation
    validation.merge(schema_vr)

    for msg in validation.errors:
        log.error("  ERROR:   %s", msg)
    for msg in validation.warnings:
        log.warning("  WARNING: %s", msg)
    for msg in validation.info[:6]:        # cap to first 6 info lines in log
        log.info("  INFO:    %s", msg)

    log.info("  Pass 3 elapsed:     %.1fs", stats.elapsed_pass3_s)

    # ── Metadata ────────────────────────────────────────────────────────────
    metadata = generate_metadata(config, stats, validation, run_ts)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(config.output_metadata(), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(
        "Metadata written: %s", config.relative_output_metadata()
    )
    log.info(
        "Total elapsed: %.1fs",
        stats.elapsed_pass1_s + stats.elapsed_pass2_s + stats.elapsed_pass3_s,
    )

    # Strict mode: raise on warnings
    if config.strict and validation.warnings:
        raise RuntimeError(
            f"--strict mode: {len(validation.warnings)} warning(s) found. "
            "See metadata for details."
        )

    # Always raise on errors (after writing metadata so it can be inspected)
    if not validation.is_valid:
        raise RuntimeError(
            f"Validation failed with {len(validation.errors)} error(s). "
            "See apple_support_metadata.json for details."
        )

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 3: Extract AppleSupport messages from the raw dataset.\n"
            "Produces apple_support_messages.csv and apple_support_metadata.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="PATH",
        help=f"Path to raw CSV (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    p.add_argument(
        "--brand",
        type=str,
        default=DEFAULT_BRAND,
        help=f"Brand author_id (default: {DEFAULT_BRAND})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        metavar="DIR",
        help=f"Output directory (default: data/processed/apple_support)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        metavar="N",
        help=f"Pandas read chunk size (default: {DEFAULT_CHUNK:,})",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings as errors (exit 1)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = ExtractionConfig(
        brand      = args.brand,
        input_path = args.input,
        output_dir = args.output_dir,
        chunk_size = args.chunk_size,
        overwrite  = args.overwrite,
        strict     = args.strict,
    )

    try:
        metadata = run_extraction(config)
    except FileNotFoundError as exc:
        log.error("Input not found: %s", exc)
        sys.exit(1)
    except FileExistsError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    print()
    print("  ── Extraction complete ────────────────────────────────")
    print(f"  Brand:          {metadata['brand']}")
    print(f"  Rows written:   {metadata['rows_written']:,}")
    print(f"  Inbound:        {metadata['inbound_rows']:,}")
    print(f"  Outbound:       {metadata['outbound_rows']:,}")
    print(f"  Unique IDs:     {metadata['unique_tweet_ids']:,}")
    print(f"  Warnings:       {len(metadata['validation']['warnings'])}")
    print(f"  Errors:         {len(metadata['validation']['errors'])}")
    print(f"  Output CSV:     {metadata['output_csv']}")
    print(f"  Metadata:       {metadata['output_metadata']}")
    print(f"  Total elapsed:  {metadata['total_elapsed_s']}s")
    print()


if __name__ == "__main__":
    main()
