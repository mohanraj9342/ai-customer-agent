"""
src/data/inspect_dataset.py
============================
Phase 1 — Safe Dataset Inspection and Brand Selection.

PURPOSE
-------
Read the Twitter Customer Support dataset (twcs/twcs.csv) in a memory-efficient
way and produce a concise statistical report.  No row of the dataset is stored
in memory beyond a single chunk at a time.

DATASET SCHEMA (confirmed from header row)
------------------------------------------
tweet_id                : int   — unique identifier for each tweet
author_id               : str   — brand handle (e.g. "AppleSupport") when
                                  inbound=False, or anonymised customer ID
                                  (numeric string, e.g. "115712") when inbound=True
inbound                 : bool  — True  → customer sent this tweet
                                  False → brand/company sent this tweet
created_at              : str   — Twitter timestamp string
text                    : str   — raw tweet text (may contain @mentions, URLs,
                                  HTML entities, emoji)
response_tweet_id       : str   — tweet_id(s) this tweet generated as a reply;
                                  may be blank, a single int, or comma-separated
                                  ints (when one tweet triggered multiple replies)
in_response_to_tweet_id : str   — tweet_id this tweet was replying to;
                                  blank for conversation-root tweets

ASSUMPTIONS
-----------
1. Brand identity: outbound tweets (inbound=False) are authored by a brand handle.
   The author_id on those rows is used as the brand name throughout this script.
2. The column order matches the header exactly; we do not rely on positional
   indexing.
3. Encoding: the file uses UTF-8 with Windows \\r\\n endings.  Pandas handles
   this transparently with the default engine.
4. Malformed rows: rows that cannot be parsed as CSV are counted and skipped via
   on_bad_lines='warn'.
5. response_tweet_id can contain comma-separated values; it is read as a string
   and is NOT cast to int in this script.

RUNNING
-------
    .venv/bin/python -m src.data.inspect_dataset

OUTPUT
------
Printed report to stdout.
Two small summary files written to docs/:
    docs/dataset_inspection.json  — machine-readable statistics
    docs/top_brands.csv           — brand-level tweet counts

Both output files are small and safe to commit.  The raw dataset is never
written to disk.
"""

import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — edit these paths if the project layout changes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "twcs" / "twcs.csv"
DOCS_DIR = PROJECT_ROOT / "docs"
SUMMARY_JSON = DOCS_DIR / "dataset_inspection.json"
TOP_BRANDS_CSV = DOCS_DIR / "top_brands.csv"

# Number of rows per chunk — chosen to keep RAM usage well under 500 MB.
# At ~165 bytes per row, 50 000 rows ≈ 8 MB per chunk.
CHUNK_SIZE = 50_000

# Number of sample rows to display in the header section of the report.
SAMPLE_ROWS = 10

# How many top brands to display in the full report.
TOP_N_BRANDS = 30

# How many example customer tweets to show per candidate brand.
EXAMPLES_PER_BRAND = 5

# Candidate brands we want to show real examples for.
# This list will be overridden at runtime with the actual measured top brands.
CANDIDATE_BRANDS: list[str] = []


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _separator(title: str = "", width: int = 70) -> None:
    """Print a visual section separator to stdout."""
    if title:
        pad = width - len(title) - 4
        left = pad // 2
        right = pad - left
        print(f"\n{'=' * left}  {title}  {'=' * right}")
    else:
        print("\n" + "=" * width)


def _print_kv(key: str, value: object, width: int = 40) -> None:
    """Print a key-value pair with aligned columns."""
    print(f"  {key:<{width}} {value}")


# ---------------------------------------------------------------------------
# Step 1 — Schema inspection (header + dtypes from a tiny sample)
# ---------------------------------------------------------------------------

def inspect_schema(path: Path, sample_n: int = 200) -> dict:
    """
    Read the column names and infer dtypes from the first `sample_n` rows.

    We deliberately read only a tiny slice so this step is fast and uses
    almost no memory, regardless of the total file size.

    Returns a dict with keys:
        columns      : list of column names in order
        dtypes       : dict mapping column name → dtype string
        sample_rows  : list of dicts (first SAMPLE_ROWS rows as records)
    """
    # Read just the first sample_n rows
    df_sample = pd.read_csv(
        path,
        nrows=sample_n,
        encoding="utf-8",
        on_bad_lines="warn",   # skip malformed rows with a warning
        dtype=str,             # read everything as string initially
    )

    # Cast inbound column to bool where possible
    if "inbound" in df_sample.columns:
        df_sample["inbound"] = df_sample["inbound"].map(
            {"True": True, "False": False}
        )

    return {
        "columns": df_sample.columns.tolist(),
        "dtypes": {col: str(df_sample[col].dtype) for col in df_sample.columns},
        "sample_rows": df_sample.head(SAMPLE_ROWS).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Step 2 — Full-file streaming pass
# ---------------------------------------------------------------------------

def stream_full_file(path: Path) -> dict:
    """
    Stream twcs.csv in fixed-size chunks and accumulate only small aggregates.

    Returns a dict with:
        total_rows          : int
        inbound_count       : int   (customer tweets)
        outbound_count      : int   (brand tweets)
        brand_counts        : Counter  {brand_handle: number_of_tweets}
        null_counts         : Counter  {column_name: number_of_nulls}
        duplicate_tweet_ids : int   (tweet_ids that appear more than once)
        has_response_link   : int   (rows where response_tweet_id is not null)
        has_parent_link     : int   (rows where in_response_to_tweet_id is not null)
        bad_rows            : int   (rows that triggered a parse warning)
        chunk_count         : int   (number of chunks processed)
        customer_ids_approx : int   (number of distinct inbound author_ids seen)
                                     [approximate — uses a Python set]
        example_tweets      : dict  {brand: [list of up to 5 customer tweet texts]}
    """
    total_rows = 0
    inbound_count = 0
    outbound_count = 0
    brand_counts: Counter = Counter()
    null_counts: Counter = Counter()
    tweet_id_set: set = set()      # used to detect duplicates
    duplicate_tweet_ids = 0
    has_response_link = 0
    has_parent_link = 0
    chunk_count = 0
    customer_id_set: set = set()   # distinct inbound author_ids

    # We collect up to EXAMPLES_PER_BRAND example customer tweets for the
    # top brands.  We build this lazily — we don't know the top brands until
    # the end, so we store examples for any brand we encounter (up to a cap)
    # and then filter afterward.
    # Cap: store at most 20 examples per brand to bound memory.
    MAX_STORED_EXAMPLES = 20
    candidate_examples: dict[str, list[str]] = {}

    print(f"  Streaming '{path.name}' in chunks of {CHUNK_SIZE:,} rows …")
    start = time.time()

    reader = pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        on_bad_lines="warn",
        dtype=str,            # parse everything as string first
        low_memory=False,
    )

    for chunk_df in reader:
        chunk_count += 1
        n = len(chunk_df)
        total_rows += n

        # --- inbound / outbound split ----------------------------------------
        # The 'inbound' column holds the strings "True" or "False".
        inbound_mask = chunk_df["inbound"].str.strip() == "True"
        inbound_count  += int(inbound_mask.sum())
        outbound_count += int((~inbound_mask).sum())

        # --- brand counts -------------------------------------------------------
        # Outbound rows are authored by the brand.  We count per author_id.
        outbound_chunk = chunk_df[~inbound_mask]
        brand_counts.update(
            outbound_chunk["author_id"].dropna().str.strip().value_counts().to_dict()
        )

        # --- nulls per column ---------------------------------------------------
        for col in chunk_df.columns:
            null_counts[col] += int(chunk_df[col].isna().sum())

        # --- duplicate tweet_id detection ---------------------------------------
        ids_in_chunk = chunk_df["tweet_id"].dropna()
        for tid in ids_in_chunk:
            if tid in tweet_id_set:
                duplicate_tweet_ids += 1
            else:
                tweet_id_set.add(tid)

        # --- conversation link statistics ---------------------------------------
        has_response_link += int(chunk_df["response_tweet_id"].notna().sum())
        has_parent_link   += int(chunk_df["in_response_to_tweet_id"].notna().sum())

        # --- unique customer IDs ------------------------------------------------
        inbound_chunk = chunk_df[inbound_mask]
        customer_id_set.update(
            inbound_chunk["author_id"].dropna().str.strip().tolist()
        )

        # --- collect example customer tweets ------------------------------------
        # For each brand's outbound row, we need the customer's tweet that came
        # before it.  That requires joining on in_response_to_tweet_id, which is
        # expensive.  Instead, we collect raw inbound tweet texts per brand by
        # looking at which brand names appear in inbound tweet @mentions.
        # This is approximate but sufficient for examples.
        for _, row in inbound_chunk.iterrows():
            text = str(row.get("text", "") or "")
            # Extract @mentioned brand from the tweet text
            for word in text.split():
                if word.startswith("@") and len(word) > 1:
                    brand_handle = word[1:].rstrip(".,!?;:")
                    if brand_handle and not brand_handle.isdigit():
                        if brand_handle not in candidate_examples:
                            candidate_examples[brand_handle] = []
                        if len(candidate_examples[brand_handle]) < MAX_STORED_EXAMPLES:
                            candidate_examples[brand_handle].append(text)
                        break   # only use the first @mention per tweet

        # Progress indicator every 10 chunks
        if chunk_count % 10 == 0:
            elapsed = time.time() - start
            print(f"    … chunk {chunk_count:>4}  |  rows so far: {total_rows:>10,}  |  {elapsed:.1f}s elapsed")

    elapsed_total = time.time() - start
    print(f"  Finished streaming.  {chunk_count} chunks, {total_rows:,} rows in {elapsed_total:.1f}s.\n")

    return {
        "total_rows": total_rows,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "brand_counts": brand_counts,
        "null_counts": dict(null_counts),
        "duplicate_tweet_ids": duplicate_tweet_ids,
        "has_response_link": has_response_link,
        "has_parent_link": has_parent_link,
        "chunk_count": chunk_count,
        "customer_ids_approx": len(customer_id_set),
        "candidate_examples": candidate_examples,
        "elapsed_seconds": round(elapsed_total, 1),
    }


# ---------------------------------------------------------------------------
# Step 3 — Collect actual customer-tweet examples for the top brands
# ---------------------------------------------------------------------------

def build_brand_examples(candidate_examples: dict, top_brands: list[str]) -> dict:
    """
    Filter the collected example pool down to the top brands.
    Returns {brand: [up to EXAMPLES_PER_BRAND tweet texts]}.
    """
    result = {}
    for brand in top_brands:
        tweets = candidate_examples.get(brand, [])
        result[brand] = tweets[:EXAMPLES_PER_BRAND]
    return result


# ---------------------------------------------------------------------------
# Step 4 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(schema: dict, stats: dict, top_brands_df: pd.DataFrame) -> None:
    """
    Write the JSON summary and top-brands CSV to docs/.
    Both files are small and safe to commit to Git.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Build a JSON-serialisable stats dict (Counter is not directly serialisable)
    summary = {
        "dataset_path": str(DATASET_PATH),
        "columns": schema["columns"],
        "dtypes": schema["dtypes"],
        "total_rows": stats["total_rows"],
        "inbound_rows": stats["inbound_count"],
        "outbound_rows": stats["outbound_count"],
        "unique_customer_ids_approx": stats["customer_ids_approx"],
        "duplicate_tweet_ids": stats["duplicate_tweet_ids"],
        "rows_with_response_link": stats["has_response_link"],
        "rows_with_parent_link": stats["has_parent_link"],
        "null_counts_per_column": stats["null_counts"],
        "chunks_processed": stats["chunk_count"],
        "streaming_elapsed_seconds": stats["elapsed_seconds"],
        "top_30_brands": {
            brand: count
            for brand, count in stats["brand_counts"].most_common(TOP_N_BRANDS)
        },
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")

    top_brands_df.to_csv(TOP_BRANDS_CSV, index=False)
    print(f"  Saved: {TOP_BRANDS_CSV.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step 5 — Print the full report
# ---------------------------------------------------------------------------

def print_report(schema: dict, stats: dict, brand_examples: dict) -> None:
    """Print the complete inspection report to stdout."""

    _separator("PHASE 1 — DATASET INSPECTION REPORT")

    # --- Basic file info ---
    _separator("1. DATASET FILE")
    file_size_mb = DATASET_PATH.stat().st_size / (1024 ** 2)
    _print_kv("File", str(DATASET_PATH.relative_to(PROJECT_ROOT)))
    _print_kv("Size on disk", f"{file_size_mb:.1f} MB")
    _print_kv("Chunks processed", f"{stats['chunk_count']:,}  (chunk size = {CHUNK_SIZE:,} rows)")
    _print_kv("Elapsed (streaming)", f"{stats['elapsed_seconds']}s")

    # --- Schema ---
    _separator("2. SCHEMA (columns and dtypes)")
    print(f"  {'Column':<35} {'Dtype (from sample)'}")
    print(f"  {'-'*35} {'-'*20}")
    for col in schema["columns"]:
        print(f"  {col:<35} {schema['dtypes'][col]}")

    # --- Row counts ---
    _separator("3. ROW COUNTS")
    total = stats["total_rows"]
    inbound = stats["inbound_count"]
    outbound = stats["outbound_count"]
    _print_kv("Total rows", f"{total:>12,}")
    _print_kv("Inbound  (inbound=True,  customer tweets)", f"{inbound:>12,}  ({inbound/total*100:.1f}%)")
    _print_kv("Outbound (inbound=False, brand replies)",   f"{outbound:>12,}  ({outbound/total*100:.1f}%)")
    _print_kv("Approx. unique customer author_ids", f"{stats['customer_ids_approx']:>12,}")
    _print_kv("Duplicate tweet_ids", f"{stats['duplicate_tweet_ids']:>12,}")

    # --- Nulls ---
    _separator("4. MISSING VALUES (null count per column)")
    for col, cnt in stats["null_counts"].items():
        pct = cnt / total * 100
        _print_kv(col, f"{cnt:>12,}  ({pct:.1f}%)")

    # --- Conversation links ---
    _separator("5. CONVERSATION LINK STATISTICS")
    _print_kv("Rows with response_tweet_id set",       f"{stats['has_response_link']:>12,}  ({stats['has_response_link']/total*100:.1f}%)")
    _print_kv("Rows with in_response_to_tweet_id set", f"{stats['has_parent_link']:>12,}  ({stats['has_parent_link']/total*100:.1f}%)")
    root_count = total - stats["has_parent_link"]
    _print_kv("Estimated conversation-root tweets",    f"{root_count:>12,}  ({root_count/total*100:.1f}%)")

    # --- Brand distribution ---
    _separator(f"6. TOP {TOP_N_BRANDS} BRANDS (by outbound tweet count)")
    print(f"  {'Rank':<6} {'Brand':<30} {'Outbound Tweets':>16}")
    print(f"  {'-'*6} {'-'*30} {'-'*16}")
    for rank, (brand, count) in enumerate(stats["brand_counts"].most_common(TOP_N_BRANDS), 1):
        print(f"  {rank:<6} {brand:<30} {count:>16,}")

    # --- Sample rows ---
    _separator("7. SAMPLE ROWS (first 10 rows of dataset)")
    for i, row in enumerate(schema["sample_rows"], 1):
        print(f"\n  Row {i}:")
        for k, v in row.items():
            val_str = str(v)[:90] + ("…" if len(str(v)) > 90 else "")
            print(f"    {k:<35} {val_str}")

    # --- Brand examples ---
    _separator("8. REAL CUSTOMER-TWEET EXAMPLES — TOP 5 BRANDS")
    print("  NOTE: tweets are raw and may contain @mentions, URLs, and emoji.")
    print("  These are actual messages from the dataset, not fabricated.\n")
    for brand, tweets in brand_examples.items():
        print(f"  ── {brand} ──────────────────────────────────────────")
        if tweets:
            for j, t in enumerate(tweets, 1):
                # Truncate long tweets for display
                display = t[:130] + ("…" if len(t) > 130 else "")
                print(f"    {j}. {display}")
        else:
            print("    (no examples collected — @mention pattern not found in sample)")
        print()

    # --- Recommendation guidance ---
    _separator("9. BRAND SELECTION GUIDANCE")
    print("""
  Based solely on the measured data above, a good candidate brand should have:
    ✓ A large number of outbound tweets (more training data)
    ✓ Enough variety in customer issues to define 6–8 distinct intents
    ✓ Consistent, recognisable brand voice in its replies
    ✓ A manageable data volume (not so large it overwhelms the laptop)

  The brand will be officially selected in Phase 2 after reviewing these
  results.  No brand is selected automatically by this script.
    """)

    _separator("END OF REPORT")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("  Dataset Inspection")
    print("  AI Customer Support Agent")
    print("=" * 70)

    # Guard: ensure the dataset exists before doing anything
    if not DATASET_PATH.exists():
        print(f"\n  ERROR: Dataset not found at:\n    {DATASET_PATH}")
        print("  Place twcs.csv at that path and re-run.")
        sys.exit(1)

    print(f"\n  Dataset found: {DATASET_PATH}")
    print(f"  Project root:  {PROJECT_ROOT}\n")

    # Step 1: Schema (fast — reads only first 200 rows)
    _separator("Reading schema from small sample …")
    schema = inspect_schema(DATASET_PATH)
    print(f"  Columns found: {schema['columns']}")

    # Step 2: Full streaming pass (slow — processes all 3M rows in chunks)
    _separator("Streaming full dataset …")
    print("  This may take several minutes on a CPU-only machine.")
    print("  Progress is printed every 10 chunks.\n")
    stats = stream_full_file(DATASET_PATH)

    # Step 3: Build top-brands DataFrame and example tweets
    top_brands_records = [
        {"rank": i + 1, "brand": brand, "outbound_tweets": count}
        for i, (brand, count) in enumerate(stats["brand_counts"].most_common(TOP_N_BRANDS))
    ]
    top_brands_df = pd.DataFrame(top_brands_records)

    top5_brands = [r["brand"] for r in top_brands_records[:5]]
    brand_examples = build_brand_examples(stats["candidate_examples"], top5_brands)

    # Step 4: Save outputs
    _separator("Saving summary files …")
    save_outputs(schema, stats, top_brands_df)

    # Step 5: Print the full report
    print_report(schema, stats, brand_examples)


if __name__ == "__main__":
    main()
