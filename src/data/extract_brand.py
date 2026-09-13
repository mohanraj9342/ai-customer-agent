"""
src/data/extract_brand.py
==========================
Phase 2 — Brand-specific data extraction pipeline.

PURPOSE
-------
Filter the full Twitter Customer Support dataset down to rows belonging to a
single selected brand, then save a compact brand-specific CSV file to
data/processed/.  The extraction runs in memory-efficient chunks so the full
~493 MB raw CSV is never held in RAM at once.

The brand is identified by the ``author_id`` column on outbound rows
(inbound=False).  Crucially, we also need the *inbound* customer tweets that
the brand was responding to — these are identified through the conversation
link columns.

STRATEGY
--------
Two-pass approach:
  Pass 1 — collect all tweet_ids that the chosen brand authored (outbound) and
            the tweet_ids those replies were responding to (parent tweets).
  Pass 2 — emit any row whose tweet_id appears in either set, preserving the
            full conversation context.

This avoids hard-coding the assumption that every customer tweet contains an
``@BrandName`` mention, which is not always true (customers sometimes reply
without re-mentioning the brand).

OUTPUT
------
  data/processed/{brand_lower}_tweets.csv   — brand-specific subset
  docs/{brand_lower}_extraction_report.json — small machine-readable summary

USAGE
-----
    # Programmatic
    from src.data.extract_brand import run_extraction
    result = run_extraction(brand="AppleSupport")

    # CLI
    .venv/bin/python -m src.data.extract_brand --brand AppleSupport
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "twcs" / "twcs.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

CHUNK_SIZE = 50_000

# HTML entities commonly found in tweet text
_HTML_ENTITY_MAP: dict[str, str] = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
}

_HTML_ENTITY_RE = re.compile("|".join(re.escape(k) for k in _HTML_ENTITY_MAP))


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """
    Lightly normalise a raw tweet string.

    Steps applied (in order):
      1. Decode common HTML entities (e.g. ``&amp;`` → ``&``).
      2. Collapse internal runs of whitespace to a single space.
      3. Strip leading/trailing whitespace.

    What is deliberately NOT done:
      - @mentions are kept — they are used for conversation-link heuristics later.
      - URLs are kept — the retrieval system may use them.
      - Emoji are kept — they carry sentiment signal.
      - Case is not changed — classifiers can handle mixed case.
      - Punctuation is not removed.

    The original text column is preserved in a separate ``text_raw`` column so
    that downstream components can choose their own normalisation.
    """
    if not isinstance(text, str):
        return ""
    text = _HTML_ENTITY_RE.sub(lambda m: _HTML_ENTITY_MAP[m.group(0)], text)
    text = " ".join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# Pass 1 — collect the tweet IDs that belong to the brand conversation
# ---------------------------------------------------------------------------

def _collect_brand_tweet_ids(
    path: Path,
    brand: str,
) -> tuple[set[str], set[str]]:
    """
    Stream the CSV once and collect two sets:

      brand_reply_ids  — tweet_ids authored by the brand (outbound)
      parent_tweet_ids — tweet_ids that brand replies were responding to
                         (i.e. the customer inbound messages one level up)

    Returns (brand_reply_ids, parent_tweet_ids).
    """
    brand_reply_ids: set[str] = set()
    parent_tweet_ids: set[str] = set()

    reader = pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        encoding="utf-8",
        on_bad_lines="skip",   # skip — don't clutter stderr during extraction
        low_memory=False,
    )

    for chunk in reader:
        # Normalise the inbound flag to a boolean mask
        outbound_mask = chunk["inbound"].str.strip().str.lower() == "false"
        brand_mask = chunk["author_id"].str.strip() == brand

        # Rows this brand authored
        brand_rows = chunk[outbound_mask & brand_mask]
        brand_reply_ids.update(brand_rows["tweet_id"].dropna().str.strip().tolist())

        # The customer tweets those replies were responding to
        parents = brand_rows["in_response_to_tweet_id"].dropna().str.strip()
        parent_tweet_ids.update(parents.tolist())

    return brand_reply_ids, parent_tweet_ids


# ---------------------------------------------------------------------------
# Pass 2 — emit matching rows and apply normalisation
# ---------------------------------------------------------------------------

def _emit_brand_rows(
    path: Path,
    brand: str,
    keep_ids: set[str],
    output_path: Path,
) -> dict:
    """
    Stream the CSV a second time, emit only rows whose tweet_id is in
    ``keep_ids``, apply text normalisation, and write to ``output_path``.

    Returns a statistics dict.
    """
    total_rows_written = 0
    inbound_written = 0
    outbound_written = 0
    first_chunk = True

    reader = pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for chunk in reader:
        # Filter to rows in our keep set
        mask = chunk["tweet_id"].str.strip().isin(keep_ids)
        matched = chunk[mask].copy()

        if matched.empty:
            continue

        # Normalise inbound to a proper boolean string
        matched["inbound"] = matched["inbound"].str.strip().str.lower().map(
            {"true": "True", "false": "False"}
        ).fillna("Unknown")

        # Add a normalised text column (keep original as text_raw)
        matched.insert(
            matched.columns.get_loc("text") + 1,
            "text_raw",
            matched["text"]
        )
        matched["text"] = matched["text"].apply(normalise_text)

        # Strip whitespace from ID columns
        for col in ("tweet_id", "author_id", "response_tweet_id",
                    "in_response_to_tweet_id"):
            if col in matched.columns:
                matched[col] = matched[col].str.strip()

        # Write (header only on the first chunk)
        matched.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8",
        )
        first_chunk = False

        n = len(matched)
        total_rows_written += n
        inbound_written += int(
            (matched["inbound"] == "True").sum()
        )
        outbound_written += int(
            (matched["inbound"] == "False").sum()
        )

    return {
        "total_rows_written": total_rows_written,
        "inbound_rows": inbound_written,
        "outbound_rows": outbound_written,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_extraction(
    brand: str,
    dataset_path: Path | None = None,
    processed_dir: Path | None = None,
    docs_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """
    Run the full two-pass brand extraction pipeline.

    Parameters
    ----------
    brand : str
        The brand's author_id exactly as it appears in the dataset
        (e.g. "AppleSupport").
    dataset_path : Path, optional
        Override the default path to twcs.csv.
    processed_dir : Path, optional
        Override the default data/processed/ output directory.
    docs_dir : Path, optional
        Override the default docs/ output directory.
    chunk_size : int, optional
        Rows per pandas chunk (default 50 000).

    Returns
    -------
    dict
        Extraction statistics and output paths.

    Raises
    ------
    FileNotFoundError
        If the raw dataset CSV does not exist.
    ValueError
        If the brand string is empty.
    """
    if not brand or not brand.strip():
        raise ValueError("brand must be a non-empty string.")

    brand = brand.strip()
    path = dataset_path or DATASET_PATH
    out_dir = processed_dir or PROCESSED_DIR
    doc_dir = docs_dir or DOCS_DIR
    brand_lower = brand.lower()

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download twcs.csv from Kaggle and place it there."
        )

    output_csv = out_dir / f"{brand_lower}_tweets.csv"
    report_json = doc_dir / f"{brand_lower}_extraction_report.json"

    print(f"\n  Brand extraction: {brand}")
    print(f"  Source:          {path}")
    print(f"  Output CSV:      {output_csv}")
    print(f"  Chunk size:      {chunk_size:,} rows\n")

    # ---- Pass 1 ----
    t0 = time.time()
    print("  Pass 1 — identifying brand tweet IDs …")
    brand_reply_ids, parent_tweet_ids = _collect_brand_tweet_ids(path, brand)
    keep_ids = brand_reply_ids | parent_tweet_ids
    t1 = time.time()
    print(f"    Brand reply IDs:   {len(brand_reply_ids):>8,}")
    print(f"    Customer tweet IDs:{len(parent_tweet_ids):>8,}")
    print(f"    Total keep set:    {len(keep_ids):>8,}")
    print(f"    Pass 1 elapsed:    {t1 - t0:.1f}s\n")

    if not brand_reply_ids:
        print(f"  WARNING: No outbound rows found for brand '{brand}'.")
        print("  Check the brand name exactly matches the author_id column.")

    # ---- Pass 2 ----
    print("  Pass 2 — emitting matched rows …")
    stats = _emit_brand_rows(path, brand, keep_ids, output_csv)
    t2 = time.time()
    print(f"    Total rows written: {stats['total_rows_written']:>8,}")
    print(f"    Inbound rows:       {stats['inbound_rows']:>8,}")
    print(f"    Outbound rows:      {stats['outbound_rows']:>8,}")
    print(f"    Pass 2 elapsed:    {t2 - t1:.1f}s")
    print(f"    Total elapsed:     {t2 - t0:.1f}s\n")

    # ---- Report ----
    report = {
        "brand": brand,
        "dataset_path": str(path.relative_to(PROJECT_ROOT) if path == DATASET_PATH else path),
        "output_csv": str(output_csv.relative_to(PROJECT_ROOT)),
        "brand_reply_ids_found": len(brand_reply_ids),
        "customer_tweet_ids_found": len(parent_tweet_ids),
        "total_keep_set": len(keep_ids),
        "total_rows_written": stats["total_rows_written"],
        "inbound_rows_written": stats["inbound_rows"],
        "outbound_rows_written": stats["outbound_rows"],
        "elapsed_seconds": round(t2 - t0, 1),
    }

    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Extraction report saved: {report_json.relative_to(PROJECT_ROOT)}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract brand-specific tweets from the full dataset."
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="AppleSupport",
        help="Brand author_id as it appears in the dataset (default: AppleSupport)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        report = run_extraction(brand=args.brand)
        print("\n  Extraction complete.")
        print(f"  Rows written: {report['total_rows_written']:,}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
