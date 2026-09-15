"""
src/retrieval/build_retrieval_corpus.py
======================================
Phase 11 — Historical Customer-Brand Interaction Corpus Builder & Isolation Engine.

Extracts the grounded pair of (initial customer inquiry, first brand reply) from
each reconstructed AppleSupport conversation thread, strictly quarantines and
excludes all Golden Evaluation Set records, performs duplicate diagnostics, and
exports the verified historical response retrieval corpus.

Design Constraints:
1. Boundary Fidelity: Uses the chronologically earliest non-empty customer inquiry
   (inbound == True) and pairs it with the first following AppleSupport brand reply.
   Never incorporates subsequent customer turns or future conversation exchanges.
2. Strict Golden Set Isolation: Mathematically guarantees S_retrieval ∩ S_golden = ∅.
3. Traceability: Retains original thread_id, customer_tweet_id, brand_tweet_id,
   and timestamps across all records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_THREADS_JSONL = "data/processed/apple_support/apple_support_threads.jsonl"
DEFAULT_GOLDEN_CSV = "data/processed/apple_support/apple_support_intent_golden_set.csv"
DEFAULT_CANDIDATES_CSV = "data/processed/apple_support/apple_support_intent_training_candidates.csv"
DEFAULT_OUTPUT_CSV = "data/processed/apple_support/apple_support_retrieval_corpus.csv"
DEFAULT_OUTPUT_METADATA = "data/processed/apple_support/apple_support_retrieval_metadata.json"


def extract_customer_brand_pair(thread_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract the initial customer message and its immediate brand reply from a reconstructed thread.

    Returns None if:
    - No customer message exists or text is empty.
    - No brand reply exists chronologically after the customer message.
    """
    messages = thread_dict.get("messages", [])
    if not messages:
        return None

    # 1. Identify earliest non-empty customer message
    cust_idx: Optional[int] = None
    cust_msg: Optional[Dict[str, Any]] = None

    for i, m in enumerate(messages):
        is_inbound = str(m.get("inbound", "")).strip().lower() in ("true", "1")
        if is_inbound:
            text = str(m.get("text", "") or "").strip()
            if text and text.lower() != "nan":
                cust_idx = i
                cust_msg = m
                break

    if cust_idx is None or cust_msg is None:
        return None

    # 2. Identify first brand reply following customer inquiry
    brand_msg: Optional[Dict[str, Any]] = None
    for j in range(cust_idx + 1, len(messages)):
        m = messages[j]
        is_outbound = (
            str(m.get("inbound", "")).strip().lower() in ("false", "0")
            or m.get("author_id") == "AppleSupport"
        )
        if is_outbound:
            text = str(m.get("text", "") or "").strip()
            if text and text.lower() != "nan":
                brand_msg = m
                break

    if brand_msg is None:
        return None

    return {
        "thread_id": str(thread_dict.get("thread_id", "")),
        "customer_tweet_id": int(cust_msg["tweet_id"]),
        "customer_text": str(cust_msg["text"]).strip(),
        "customer_timestamp": str(cust_msg.get("created_at", "")),
        "customer_author_id": str(cust_msg.get("author_id", "")),
        "brand_tweet_id": int(brand_msg["tweet_id"]),
        "brand_text": str(brand_msg["text"]).strip(),
        "brand_timestamp": str(brand_msg.get("created_at", "")),
        "brand_author_id": str(brand_msg.get("author_id", "AppleSupport")),
    }


def verify_retrieval_isolation(
    retrieval_customer_ids: Set[int],
    golden_tweet_ids: Set[int],
) -> bool:
    """
    Verify that zero golden evaluation records exist in the retrieval corpus.
    Raises ValueError on violation.
    """
    overlap = retrieval_customer_ids.intersection(golden_tweet_ids)
    if overlap:
        example_leaks = sorted(list(overlap))[:5]
        raise ValueError(
            f"CRITICAL ISOLATION VIOLATION: {len(overlap)} Golden Evaluation Set records "
            f"detected in retrieval corpus! Example leaking IDs: {example_leaks}"
        )
    return True


def audit_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze exact and near duplicates across customer queries and brand replies."""
    total_records = len(df)
    unique_cust_ids = int(df["customer_tweet_id"].nunique())
    unique_cust_texts = int(df["customer_text"].nunique())
    unique_brand_ids = int(df["brand_tweet_id"].nunique())
    unique_brand_texts = int(df["brand_text"].nunique())
    unique_pairs = int(df.drop_duplicates(subset=["customer_text", "brand_text"]).shape[0])

    duplicate_cust_texts = total_records - unique_cust_texts
    identical_pairs_duplicates = total_records - unique_pairs

    # Top repeated customer queries
    top_repeated_queries = (
        df["customer_text"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    return {
        "total_records": total_records,
        "unique_customer_tweet_ids": unique_cust_ids,
        "duplicate_customer_tweet_ids": total_records - unique_cust_ids,
        "unique_customer_texts": unique_cust_texts,
        "duplicate_customer_text_count": duplicate_cust_texts,
        "duplicate_customer_text_pct": round(duplicate_cust_texts / max(1, total_records) * 100, 2),
        "unique_brand_tweet_ids": unique_brand_ids,
        "unique_brand_texts": unique_brand_texts,
        "unique_pair_count": unique_pairs,
        "identical_pair_duplicates": identical_pairs_duplicates,
        "top_repeated_customer_queries": top_repeated_queries,
    }


def build_retrieval_corpus(
    threads_path: str | Path = DEFAULT_THREADS_JSONL,
    golden_csv_path: str | Path = DEFAULT_GOLDEN_CSV,
    candidates_csv_path: Optional[str | Path] = DEFAULT_CANDIDATES_CSV,
    output_csv_path: str | Path = DEFAULT_OUTPUT_CSV,
    output_meta_path: str | Path = DEFAULT_OUTPUT_METADATA,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build, isolate, and save the historical customer-brand response retrieval corpus."""
    t_path = Path(threads_path)
    g_path = Path(golden_csv_path)

    if not t_path.exists():
        raise FileNotFoundError(f"Reconstructed threads JSONL not found at: {t_path}")
    if not g_path.exists():
        raise FileNotFoundError(f"Golden evaluation CSV not found at: {g_path}")

    logger.info("Loading Golden Evaluation Set from: %s", g_path)
    golden_df = pd.read_csv(g_path)
    golden_ids: Set[int] = set(golden_df["tweet_id"].astype(int))
    logger.info("Quarantine registry loaded: %d golden records.", len(golden_ids))

    # Optional: Load candidate intent mapping for enriched metadata
    intent_map: Dict[int, str] = {}
    if candidates_csv_path and Path(candidates_csv_path).exists():
        logger.info("Loading intent metadata from candidates CSV: %s", candidates_csv_path)
        cand_df = pd.read_csv(candidates_csv_path)
        intent_col = "candidate_intent" if "candidate_intent" in cand_df.columns else "inferred_intent"
        if intent_col in cand_df.columns:
            intent_map = dict(zip(cand_df["tweet_id"].astype(int), cand_df[intent_col]))

    logger.info("Scanning threads and extracting grounded pairs from: %s", t_path)
    extracted_pairs: List[Dict[str, Any]] = []
    quarantined_golden_records: List[int] = []
    threads_processed = 0
    threads_excluded_no_pair = 0

    with open(t_path, "r", encoding="utf-8") as f:
        for line in f:
            threads_processed += 1
            thread_dict = json.loads(line)
            pair = extract_customer_brand_pair(thread_dict)

            if pair is None:
                threads_excluded_no_pair += 1
                continue

            c_tid = pair["customer_tweet_id"]
            if c_tid in golden_ids:
                quarantined_golden_records.append(c_tid)
                continue

            pair["inferred_intent"] = intent_map.get(c_tid, "unknown_other")
            extracted_pairs.append(pair)

    df_corpus = pd.DataFrame(extracted_pairs)
    # Assign sequential 0-indexed corpus_id for array alignment
    df_corpus.insert(0, "corpus_id", range(len(df_corpus)))

    logger.info(
        "Extraction complete. Total threads: %d | Formed pairs: %d | Quarantined Golden: %d | Excluded (no pair): %d",
        threads_processed,
        len(df_corpus),
        len(quarantined_golden_records),
        threads_excluded_no_pair,
    )

    # Verify Golden Set Isolation
    retrieval_ids = set(df_corpus["customer_tweet_id"].astype(int))
    verify_retrieval_isolation(retrieval_ids, golden_ids)
    logger.info("SUCCESS: Golden Set isolation verified: S_retrieval ∩ S_golden = ∅.")

    # Duplicate audit
    dup_stats = audit_duplicates(df_corpus)

    # Save corpus CSV
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_corpus.to_csv(out_csv, index=False)
    logger.info("Saved retrieval corpus (%d records) to: %s", len(df_corpus), out_csv)

    # Metadata & Checksum
    with open(out_csv, "rb") as f:
        csv_sha256 = hashlib.sha256(f.read()).hexdigest()

    metadata: Dict[str, Any] = {
        "pipeline_phase": "Phase 11: Historical Response Retrieval & Grounding",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_threads_file": str(t_path),
        "golden_set_file": str(g_path),
        "total_threads_processed": threads_processed,
        "threads_with_grounded_pair": len(df_corpus) + len(quarantined_golden_records),
        "golden_records_quarantined_count": len(quarantined_golden_records),
        "golden_records_quarantined_ids": sorted(quarantined_golden_records),
        "retrieval_corpus_size": len(df_corpus),
        "corpus_file": str(out_csv),
        "corpus_sha256": csv_sha256,
        "isolation_verified": True,
        "duplicate_audit": dup_stats,
        "schema_columns": list(df_corpus.columns),
    }

    out_meta = Path(output_meta_path)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved retrieval metadata to: %s", out_meta)

    return df_corpus, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 11 Retrieval Corpus from Reconstructed Threads.")
    parser.add_argument("--threads", default=DEFAULT_THREADS_JSONL)
    parser.add_argument("--golden-csv", default=DEFAULT_GOLDEN_CSV)
    parser.add_argument("--candidates-csv", default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--out-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--out-meta", default=DEFAULT_OUTPUT_METADATA)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    build_retrieval_corpus(
        threads_path=args.threads,
        golden_csv_path=args.golden_csv,
        candidates_csv_path=args.candidates_csv,
        output_csv_path=args.out_csv,
        output_meta_path=args.out_meta,
    )


if __name__ == "__main__":
    main()
