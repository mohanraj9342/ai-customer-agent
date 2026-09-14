"""
src/data/reconstruct_threads.py
================================
Phase 4 — Conversation thread reconstruction.

PURPOSE
-------
Given a brand-specific tweet DataFrame (produced by extract_selected_brand.py),
rebuild multi-turn customer-support conversations in chronological order and
produce a validated JSONL artifact for downstream use.

THREAD DEFINITION
-----------------
A "thread" is a connected component of tweets linked through the
``in_response_to_tweet_id`` edge (child → parent direction).  The
reconstruction algorithm:

  1. Detects and reports duplicate tweet_ids before building the graph.
  2. Builds O(1) lookup tables (parent→children, child→parent, id→row).
  3. Identifies root tweets (those whose parent is absent from the dataset).
  4. Walks each root's subtree using BFS, collecting messages in order.
  5. Sorts each thread chronologically by ``created_at``, with ``tweet_id``
     as a stable tie-breaker.
  6. Enriches each thread with detailed quality metrics.
  7. Writes one JSON object per thread to a JSONL file.

QUALITY METRICS
---------------
Every Thread carries precise quality flags and counters:

  has_both_directions  — thread contains ≥1 inbound AND ≥1 outbound message.
  has_3plus_messages   — thread has ≥3 messages (genuine context).
  has_multiple_exchanges — thread has ≥2 customer messages AND ≥2 brand
                           messages (at least 4 total; strict back-and-forth).
  has_broken_links     — ≥1 parent link whose target is absent from the dataset.
  cycle_detected       — BFS encountered an already-visited node (covers both
                         true cycles and diamond patterns; conservative flag).
  is_complete          — last message in chronological order is a brand reply.
                         NOTE: this is a necessary but NOT sufficient condition
                         for a "resolved" conversation.

MULTIPLE-EXCHANGE DEFINITION
-----------------------------
A thread has ``has_multiple_exchanges = True`` if and only if:
  customer_message_count ≥ 2  AND  brand_message_count ≥ 2

This requires at least 4 messages and represents genuine dialogue rather than
a single customer→brand pair.

LINK METRICS (per thread)
-------------------------
Parent links:
  parent_links_present  — messages with a non-empty in_response_to_tweet_id
  parent_links_valid    — those where the referenced ID exists in the dataset
  parent_links_missing  — those where the referenced ID is absent

Response links:
  response_links_present  — messages with a non-empty response_tweet_id
  response_links_valid    — individual response IDs that exist in the dataset
  response_links_missing  — individual response IDs that are absent

Percentages use explicit denominators (documented in metadata).

OUTPUT
------
  data/processed/apple_support/apple_support_threads.jsonl
  data/processed/apple_support/apple_support_thread_metadata.json

CLI USAGE
---------
  .venv/bin/python -m src.data.reconstruct_threads \\
      --input data/processed/apple_support/apple_support_messages.csv \\
      --output-dir data/processed/apple_support

  Options:
    --input       PATH   Input messages CSV (Phase 3 output)
    --output-dir  DIR    Output directory
    --brand       STR    Brand name for metadata (default: AppleSupport)
    --overwrite          Allow replacing existing output files
    --strict             Treat quality warnings as errors (exit 1)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

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
SCHEMA_VERSION = "4.0"
OUTPUT_JSONL_NAME    = "apple_support_threads.jsonl"
OUTPUT_META_NAME     = "apple_support_thread_metadata.json"
DEFAULT_BRAND        = "AppleSupport"
DEFAULT_INPUT        = PROJECT_ROOT / "data" / "processed" / "apple_support" / "apple_support_messages.csv"
DEFAULT_OUTDIR       = PROJECT_ROOT / "data" / "processed" / "apple_support"

REQUIRED_COLUMNS = {
    "tweet_id", "author_id", "inbound", "created_at",
    "text", "response_tweet_id", "in_response_to_tweet_id",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Thread:
    """
    A reconstructed customer-support conversation thread.

    Phase 3 fields (unchanged for backward compatibility)
    -----------------------------------------------------
    root_tweet_id  : tweet_id of the earliest reachable ancestor in this thread.
    tweet_ids      : ordered tweet_ids (chronological).
    texts          : tweet text for each message, aligned with tweet_ids.
    directions     : "inbound" or "outbound" per message.
    timestamps     : raw created_at string per message.
    turn_count     : number of messages.
    is_complete    : True if last message is outbound (brand reply).
    has_broken_links : True if any parent link target is absent from the dataset.

    Phase 4 fields (new, all have safe defaults)
    --------------------------------------------
    thread_id           : deterministic identifier "thread_{root_tweet_id}".
    author_ids          : author_id per message, aligned with tweet_ids.
    row_types           : row_type per message ("brand_reply"/"customer_inbound"/…).
    response_ids        : raw response_tweet_id string per message.
    in_response_ids     : raw in_response_to_tweet_id string per message.
    customer_message_count : count of inbound messages.
    brand_message_count    : count of outbound messages.
    start_timestamp     : created_at of the chronologically first message.
    end_timestamp       : created_at of the chronologically last message.
    has_both_directions : True if both customer and brand messages present.
    has_3plus_messages  : True if turn_count ≥ 3.
    has_multiple_exchanges : True if customer ≥ 2 AND brand ≥ 2.
    cycle_detected      : True if BFS found an already-visited node.
    quality_flags       : list of descriptive warning strings.

    Link metrics (per thread)
    --------------------------
    parent_links_present  : messages with non-empty in_response_to_tweet_id.
    parent_links_valid    : those whose target ID exists in the dataset.
    parent_links_missing  : those whose target ID is absent.
    response_links_present: messages with non-empty response_tweet_id.
    response_links_valid  : individual response IDs existing in dataset.
    response_links_missing: individual response IDs absent from dataset.

    Other quality
    -------------
    has_duplicate_ids         : any tweet_id appears more than once in thread.
    has_conflicting_duplicates: same tweet_id, different text or author_id.
    has_empty_text            : any message has empty text.
    malformed_timestamps      : count of timestamps that could not be parsed.
    """

    # ---- Phase 3 fields (preserved exactly) ----
    root_tweet_id:    str = ""
    tweet_ids:        list[str] = field(default_factory=list)
    texts:            list[str] = field(default_factory=list)
    directions:       list[str] = field(default_factory=list)
    timestamps:       list[str] = field(default_factory=list)
    turn_count:       int  = 0
    is_complete:           bool = False
    ends_with_brand_reply: bool = False
    has_broken_links:      bool = False

    # ---- Phase 4 fields ----
    thread_id:              str       = ""
    author_ids:             list[str] = field(default_factory=list)
    row_types:              list[str] = field(default_factory=list)
    response_ids:           list[str] = field(default_factory=list)
    in_response_ids:        list[str] = field(default_factory=list)
    customer_message_count: int  = 0
    brand_message_count:    int  = 0
    start_timestamp:        str  = ""
    end_timestamp:          str  = ""
    has_both_directions:    bool = False
    has_3plus_messages:     bool = False
    has_multiple_exchanges: bool = False
    cycle_detected:         bool = False
    quality_flags:          list[str] = field(default_factory=list)

    # Link metrics
    parent_links_present:   int = 0
    parent_links_valid:     int = 0
    parent_links_missing:   int = 0
    response_links_present: int = 0
    response_links_valid:   int = 0
    response_links_missing: int = 0

    # Other quality flags
    has_duplicate_ids:          bool = False
    has_conflicting_duplicates: bool = False
    has_empty_text:             bool = False
    malformed_timestamps:       int  = 0

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "incomplete"
        broken = " [broken links]" if self.has_broken_links else ""
        cycle  = " [cycle]" if self.cycle_detected else ""
        return (
            f"Thread(root={self.root_tweet_id!r}, turns={self.turn_count}, "
            f"{status}{broken}{cycle})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _parse_ts(ts: str | None) -> float:
    """Parse created_at string to float timestamp safely and quickly."""
    if not ts or not isinstance(ts, str) or str(ts).strip().lower() in ("", "nan"):
        return float("inf")
    ts_str = str(ts).strip()
    try:
        return datetime.strptime(ts_str, _TS_FORMAT).timestamp()
    except Exception:
        try:
            return pd.to_datetime(ts_str, utc=True).timestamp()
        except Exception:
            return float("inf")


def _parse_response_ids(raw: str | None) -> list[str]:
    """
    Parse the response_tweet_id field, which can be:
      - empty / NaN → []
      - a single integer string → [id]
      - comma-separated integer strings → [id1, id2, ...]
    """
    if not raw or pd.isna(raw) or str(raw).strip() in ("", "nan"):
        return []
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _detect_duplicates(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """
    Detect duplicate tweet_ids in the DataFrame.

    Returns
    -------
    (duplicate_ids, conflict_ids)
        duplicate_ids : tweet_ids appearing more than once.
        conflict_ids  : tweet_ids where different rows have different text or author_id.
    """
    seen: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    conflict_ids: set[str] = set()

    records = df.to_dict("records")
    for row in records:
        tid = str(row.get("tweet_id", "")).strip()
        if not tid or tid.lower() == "nan":
            continue
        if tid in seen:
            duplicate_ids.add(tid)
            prev = seen[tid]
            if (str(row.get("text", "")) != str(prev.get("text", "")) or
                    str(row.get("author_id", "")) != str(prev.get("author_id", ""))):
                conflict_ids.add(tid)
        else:
            seen[tid] = row
    return duplicate_ids, conflict_ids


def _build_lookup_tables(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Build three lookup tables from the DataFrame for O(1) access.

    Duplicate tweet_ids: the first occurrence is kept (deterministic after
    upstream sort by tweet_id from Phase 3).

      id_to_row          : tweet_id → row dict
      child_to_parent    : tweet_id → in_response_to_tweet_id (or None)
      parent_to_children : in_response_to_tweet_id → [child tweet_ids]
    """
    id_to_row: dict[str, dict] = {}
    child_to_parent: dict[str, Optional[str]] = {}
    parent_to_children: dict[str, list[str]] = {}

    records = df.to_dict("records")
    for row in records:
        tid = str(row["tweet_id"]).strip()
        parent = row.get("in_response_to_tweet_id", None)
        parent_str = (
            str(parent).strip()
            if parent and str(parent).strip() not in ("", "nan")
            else None
        )

        if tid in id_to_row:
            continue   # keep first occurrence; duplicate already logged

        id_to_row[tid] = row
        child_to_parent[tid] = parent_str

        if parent_str:
            parent_to_children.setdefault(parent_str, []).append(tid)

    return id_to_row, child_to_parent, parent_to_children


def _find_roots(
    id_to_row: dict,
    child_to_parent: dict,
    all_ids: set[str],
) -> list[tuple[str, bool]]:
    """
    Identify root tweets: messages whose parent is absent from the dataset.

    Returns a list of (tweet_id, has_broken_links) tuples.
    """
    roots = []
    for tid in id_to_row:
        parent = child_to_parent.get(tid)
        if parent is None or not parent:
            roots.append((tid, False))
        elif parent not in all_ids:
            roots.append((tid, True))
    return roots


def _walk_thread(
    root_id: str,
    id_to_row: dict,
    parent_to_children: dict,
    initial_broken: bool = False,
) -> Thread:
    """
    Walk forward from a root tweet, building an ordered thread using BFS.

    Cycle / diamond detection: if BFS pops an already-visited ID, the
    ``cycle_detected`` flag is set on the Thread.  This is a conservative flag
    that covers both true parent-link cycles and diamond-shaped graphs.

    Returns a Thread with Phase 3 fields populated.
    Phase 4 enrichment (quality metrics) is applied separately by
    ``_enrich_thread()``.
    """
    visited: set[str] = set()
    collected: list[dict] = []
    has_broken_links = initial_broken
    cycle_detected   = False

    queue = [root_id]
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            cycle_detected = True
            continue
        if current_id not in id_to_row:
            has_broken_links = True
            continue
        visited.add(current_id)
        collected.append(id_to_row[current_id])

        children = parent_to_children.get(current_id, [])
        queue.extend(children)

    if not collected:
        return Thread(
            root_tweet_id=root_id,
            has_broken_links=True,
            cycle_detected=cycle_detected,
            ends_with_brand_reply=False,
        )

    # Sort chronologically; tweet_id is the tie-breaker for determinism
    def _sort_key(r: dict) -> tuple:
        ts = r.get("created_at", "")
        return (_parse_ts(ts), str(r.get("tweet_id", "")))

    collected.sort(key=_sort_key)

    tweet_ids  = [str(r["tweet_id"]) for r in collected]
    texts      = [str(r.get("text",      "")) for r in collected]
    directions = [
        "inbound"
        if str(r.get("inbound", "False")).strip().lower() in ("true",)
        else "outbound"
        for r in collected
    ]
    timestamps = [str(r.get("created_at", "")) for r in collected]

    last_direction = directions[-1] if directions else "outbound"
    ends_with_brand = last_direction == "outbound"
    is_complete    = ends_with_brand

    return Thread(
        root_tweet_id=root_id,
        tweet_ids=tweet_ids,
        texts=texts,
        directions=directions,
        timestamps=timestamps,
        turn_count=len(collected),
        is_complete=is_complete,
        ends_with_brand_reply=ends_with_brand,
        has_broken_links=has_broken_links,
        cycle_detected=cycle_detected,
    )


def _enrich_thread(
    thread: Thread,
    id_to_row: dict,
    all_ids: set[str],
    duplicate_ids: set[str],
    conflict_ids: set[str],
) -> Thread:
    """
    Populate all Phase 4 quality fields on a Thread in-place.

    Parameters
    ----------
    thread       : Thread produced by _walk_thread (Phase 3 fields set).
    id_to_row    : full id→row lookup (first occurrence).
    all_ids      : set of all tweet_ids present in the dataset.
    duplicate_ids: tweet_ids appearing more than once in the input.
    conflict_ids : tweet_ids with conflicting content across duplicate rows.
    """
    # Thread identifier
    thread.thread_id = f"thread_{thread.root_tweet_id}"

    # Per-message extra fields
    author_ids    = []
    row_types     = []
    resp_ids_list = []
    in_resp_list  = []

    p_present = p_valid = p_missing = 0
    r_present = r_valid = r_missing = 0
    mal_ts = 0
    has_empty_text = False
    has_dup_ids    = False
    has_conflict   = False
    flags: list[str] = []

    for tid in thread.tweet_ids:
        row = id_to_row.get(tid, {})

        # Author and row_type
        author_ids.append(str(row.get("author_id", "")))
        row_types.append(str(row.get("row_type", "")))

        # Raw link fields
        in_resp = str(row.get("in_response_to_tweet_id", "") or "").strip()
        resp    = str(row.get("response_tweet_id",        "") or "").strip()
        if in_resp.lower() == "nan":
            in_resp = ""
        if resp.lower() == "nan":
            resp = ""
        in_resp_list.append(in_resp)
        resp_ids_list.append(resp)

        # Parent link metrics
        if in_resp:
            p_present += 1
            if in_resp in all_ids:
                p_valid += 1
            else:
                p_missing += 1

        # Response link metrics (may be comma-separated)
        parsed_resp = _parse_response_ids(resp)
        if parsed_resp:
            r_present += 1
            for rid in parsed_resp:
                if rid in all_ids:
                    r_valid += 1
                else:
                    r_missing += 1

        # Empty text
        text = str(row.get("text", "") or "").strip()
        if not text or text.lower() == "nan":
            has_empty_text = True

        # Malformed timestamp
        ts = str(row.get("created_at", "") or "").strip()
        if ts and ts.lower() != "nan":
            if _parse_ts(ts) == float("inf"):
                mal_ts += 1
        else:
            mal_ts += 1   # missing timestamp is also flagged

        # Duplicate / conflict
        if tid in duplicate_ids:
            has_dup_ids = True
        if tid in conflict_ids:
            has_conflict = True

    thread.author_ids        = author_ids
    thread.row_types         = row_types
    thread.response_ids      = resp_ids_list
    thread.in_response_ids   = in_resp_list
    thread.parent_links_present   = p_present
    thread.parent_links_valid     = p_valid
    thread.parent_links_missing   = p_missing
    thread.response_links_present = r_present
    thread.response_links_valid   = r_valid
    thread.response_links_missing = r_missing
    thread.has_empty_text         = has_empty_text
    thread.malformed_timestamps   = mal_ts
    thread.has_duplicate_ids      = has_dup_ids
    thread.has_conflicting_duplicates = has_conflict

    # Directional counts
    thread.customer_message_count = thread.directions.count("inbound")
    thread.brand_message_count    = thread.directions.count("outbound")
    thread.has_both_directions    = (
        thread.customer_message_count > 0 and thread.brand_message_count > 0
    )
    thread.has_3plus_messages     = thread.turn_count >= 3
    thread.has_multiple_exchanges = (
        thread.customer_message_count >= 2 and thread.brand_message_count >= 2
    )

    # Timestamps
    if thread.timestamps:
        thread.start_timestamp = thread.timestamps[0]
        thread.end_timestamp   = thread.timestamps[-1]

    # Quality flags
    if thread.has_broken_links:
        flags.append("missing_parent_link")
    if r_missing > 0:
        flags.append("missing_response_target")
    if thread.cycle_detected:
        flags.append("cycle_detected")
    if mal_ts > 0:
        flags.append(f"malformed_or_missing_timestamp ({mal_ts})")
    if has_empty_text:
        flags.append("empty_text")
    if thread.turn_count == 1:
        flags.append("single_message")
    if thread.customer_message_count == 0:
        flags.append("no_customer_message")
    if thread.brand_message_count == 0:
        flags.append("no_brand_message")
    if has_dup_ids:
        flags.append("duplicate_tweet_ids_in_input")
    if has_conflict:
        flags.append("conflicting_duplicate_records_in_input")
    thread.quality_flags = flags

    return thread


# ---------------------------------------------------------------------------
# Public interface — core reconstruction
# ---------------------------------------------------------------------------

def reconstruct_threads(df: pd.DataFrame) -> list[Thread]:
    """
    Reconstruct all conversation threads from a brand-specific tweet DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum: tweet_id, inbound, created_at, text,
        in_response_to_tweet_id, response_tweet_id.
        All columns should be string dtype.

    Returns
    -------
    list[Thread]
        One Thread per connected component, ordered by root_tweet_id numerically.
        Each thread is fully enriched with Phase 4 quality metrics.

    Notes
    -----
    - Duplicate tweet_ids: first occurrence wins; both are flagged.
    - Cycles / diamond patterns: BFS stops revisiting; ``cycle_detected`` is set.
    - Missing parent links: thread is still reconstructed from the orphan node.
    - All threads in the list are enriched (Phase 4 fields populated).
    """
    if df.empty:
        return []

    df = df.copy()
    for col in ("tweet_id", "in_response_to_tweet_id", "response_tweet_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", "")

    # Detect duplicates before collapsing
    duplicate_ids, conflict_ids = _detect_duplicates(df)

    id_to_row, child_to_parent, parent_to_children = _build_lookup_tables(df)
    all_ids = set(id_to_row.keys())

    roots = _find_roots(id_to_row, child_to_parent, all_ids)

    threads: list[Thread] = []
    visited_in_threads: set[str] = set()

    for root_id, initial_broken in roots:
        thread = _walk_thread(
            root_id, id_to_row, parent_to_children, initial_broken
        )
        _enrich_thread(thread, id_to_row, all_ids, duplicate_ids, conflict_ids)
        threads.append(thread)
        visited_in_threads.update(thread.tweet_ids)

    # Cycle fallback: nodes trapped in mutual-parent cycles
    unvisited = sorted(set(id_to_row.keys()) - visited_in_threads)
    for tid in unvisited:
        if tid not in visited_in_threads:
            thread = _walk_thread(
                tid, id_to_row, parent_to_children, initial_broken=True
            )
            _enrich_thread(
                thread, id_to_row, all_ids, duplicate_ids, conflict_ids
            )
            threads.append(thread)
            visited_in_threads.update(thread.tweet_ids)

    # Sort threads deterministically by root_tweet_id (numeric then string)
    def _thread_sort_key(t: Thread) -> tuple:
        try:
            return (0, int(t.root_tweet_id))
        except ValueError:
            return (1, t.root_tweet_id)

    threads.sort(key=_thread_sort_key)
    return threads


def summarise_threads(threads: list[Thread]) -> dict:
    """
    Compute summary statistics over a list of reconstructed threads.

    Backward-compatible with Phase 3 callers.  Phase 4 fields are added
    alongside the existing keys.

    Returns a dict with:
      total_threads, single_turn, multi_turn, complete, incomplete,
      broken_links, avg_turns, max_turns, min_turns  (Phase 3)
      +
      has_both_directions, has_3plus_messages, has_multiple_exchanges,
      cycle_detected, total_messages, avg_messages_per_thread,
      median_messages_per_thread, customer_messages, brand_messages (Phase 4)
    """
    if not threads:
        return {
            "total_threads": 0, "single_turn": 0, "multi_turn": 0,
            "complete": 0, "incomplete": 0, "broken_links": 0,
            "avg_turns": 0.0, "max_turns": 0, "min_turns": 0,
            # Phase 4
            "has_both_directions": 0, "has_3plus_messages": 0,
            "has_multiple_exchanges": 0, "cycle_detected": 0,
            "total_messages": 0, "avg_messages_per_thread": 0.0,
            "median_messages_per_thread": 0.0,
            "customer_messages": 0, "brand_messages": 0,
        }

    turn_counts = [t.turn_count for t in threads]
    sorted_turns = sorted(turn_counts)
    n = len(sorted_turns)
    median = (
        sorted_turns[n // 2]
        if n % 2
        else (sorted_turns[n // 2 - 1] + sorted_turns[n // 2]) / 2
    )

    return {
        # Phase 3 keys (unchanged)
        "total_threads":  len(threads),
        "single_turn":    sum(1 for t in threads if t.turn_count == 1),
        "multi_turn":     sum(1 for t in threads if t.turn_count > 1),
        "complete":       sum(1 for t in threads if t.is_complete),
        "incomplete":     sum(1 for t in threads if not t.is_complete),
        "ends_with_brand_reply": sum(1 for t in threads if t.ends_with_brand_reply),
        "broken_links":   sum(1 for t in threads if t.has_broken_links),
        "avg_turns":      round(sum(turn_counts) / len(turn_counts), 2),
        "max_turns":      max(turn_counts),
        "min_turns":      min(turn_counts),
        # Phase 4 keys
        "has_both_directions":    sum(1 for t in threads if t.has_both_directions),
        "has_3plus_messages":     sum(1 for t in threads if t.has_3plus_messages),
        "has_multiple_exchanges": sum(1 for t in threads if t.has_multiple_exchanges),
        "cycle_detected":         sum(1 for t in threads if t.cycle_detected),
        "total_messages":         sum(turn_counts),
        "avg_messages_per_thread": round(sum(turn_counts) / len(turn_counts), 2),
        "median_messages_per_thread": float(median),
        "customer_messages": sum(t.customer_message_count for t in threads),
        "brand_messages":    sum(t.brand_message_count    for t in threads),
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def thread_to_dict(thread: Thread) -> dict:
    """
    Serialise a Thread to a dict suitable for JSON/JSONL output.

    The messages list contains one dict per turn with all preserved fields.
    """
    messages = []
    for i, tid in enumerate(thread.tweet_ids):
        messages.append({
            "tweet_id":               tid,
            "author_id":              thread.author_ids[i]     if i < len(thread.author_ids)     else "",
            "inbound":                "True" if thread.directions[i] == "inbound" else "False",
            "row_type":               thread.row_types[i]      if i < len(thread.row_types)      else "",
            "created_at":             thread.timestamps[i]     if i < len(thread.timestamps)     else "",
            "text":                   thread.texts[i]          if i < len(thread.texts)          else "",
            "response_tweet_id":      thread.response_ids[i]   if i < len(thread.response_ids)   else "",
            "in_response_to_tweet_id":thread.in_response_ids[i] if i < len(thread.in_response_ids) else "",
        })

    return {
        "thread_id":              thread.thread_id,
        "root_tweet_id":          thread.root_tweet_id,
        "message_count":          thread.turn_count,
        "customer_message_count": thread.customer_message_count,
        "brand_message_count":    thread.brand_message_count,
        "start_timestamp":        thread.start_timestamp,
        "end_timestamp":          thread.end_timestamp,
        # Quality flags
        "is_complete":            thread.is_complete,
        "ends_with_brand_reply":  thread.ends_with_brand_reply,
        "has_both_directions":    thread.has_both_directions,
        "has_3plus_messages":     thread.has_3plus_messages,
        "has_multiple_exchanges": thread.has_multiple_exchanges,
        "has_broken_links":       thread.has_broken_links,
        "cycle_detected":         thread.cycle_detected,
        "has_duplicate_ids":      thread.has_duplicate_ids,
        "has_conflicting_duplicates": thread.has_conflicting_duplicates,
        "has_empty_text":         thread.has_empty_text,
        "malformed_timestamps":   thread.malformed_timestamps,
        "quality_flags":          thread.quality_flags,
        # Link metrics
        "link_metrics": {
            "parent_links_present":    thread.parent_links_present,
            "parent_links_valid":      thread.parent_links_valid,
            "parent_links_missing":    thread.parent_links_missing,
            "response_links_present":  thread.response_links_present,
            "response_links_valid":    thread.response_links_valid,
            "response_links_missing":  thread.response_links_missing,
        },
        "messages": messages,
    }


def write_jsonl(threads: list[Thread], output_path: Path) -> int:
    """
    Write one JSON object per thread to output_path in JSONL format.
    Returns the number of threads written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(json.dumps(thread_to_dict(thread), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Corpus-level metrics and metadata
# ---------------------------------------------------------------------------

def _corpus_link_metrics(threads: list[Thread]) -> dict:
    """Aggregate link metrics across all threads with explicit denominators."""
    p_present = sum(t.parent_links_present   for t in threads)
    p_valid   = sum(t.parent_links_valid     for t in threads)
    p_missing = sum(t.parent_links_missing   for t in threads)
    r_present = sum(t.response_links_present for t in threads)
    r_valid   = sum(t.response_links_valid   for t in threads)
    r_missing = sum(t.response_links_missing for t in threads)
    return {
        # Parent links
        "messages_with_parent_link":          p_present,
        "parent_links_valid":                 p_valid,
        "parent_links_missing":               p_missing,
        "pct_valid_parent_links":             (
            round(100 * p_valid / p_present, 1) if p_present else None
        ),
        "denominator_valid_parent_pct":       "messages_with_parent_link (parent_links_valid + parent_links_missing)",
        # Response links
        "messages_with_response_link":        r_present,
        "total_individual_response_ids":      r_valid + r_missing,
        "response_link_ids_valid":            r_valid,
        "response_link_ids_missing":          r_missing,
        "pct_valid_response_ids":             (
            round(100 * r_valid / (r_valid + r_missing), 1)
            if (r_valid + r_missing) > 0 else None
        ),
        "denominator_valid_response_pct":     "total_individual_response_ids (response_link_ids_valid + response_link_ids_missing)",
    }


def generate_reconstruction_metadata(
    brand: str,
    input_path: Path,
    output_jsonl: Path,
    output_meta: Path,
    n_input_messages: int,
    threads: list[Thread],
    duplicate_ids: set[str],
    conflict_ids: set[str],
    elapsed_s: float,
    timestamp: str,
) -> dict:
    """Build the machine-readable metadata dict. All paths are project-relative."""
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    summary = summarise_threads(threads)
    link_m  = _corpus_link_metrics(threads)

    # Quality thread counts
    missing_parent_threads   = summary["broken_links"]
    missing_response_threads = sum(1 for t in threads if t.response_links_missing > 0)
    cycle_threads            = summary["cycle_detected"]
    malformed_ts_threads     = sum(1 for t in threads if t.malformed_timestamps > 0)
    dup_id_threads           = sum(1 for t in threads if t.has_duplicate_ids)
    conflict_threads         = sum(1 for t in threads if t.has_conflicting_duplicates)
    empty_text_threads       = sum(1 for t in threads if t.has_empty_text)
    single_msg_threads       = summary["single_turn"]
    no_customer_threads      = sum(1 for t in threads if t.customer_message_count == 0)
    no_brand_threads         = sum(1 for t in threads if t.brand_message_count == 0)

    return {
        "schema_version":       SCHEMA_VERSION,
        "reconstruction_timestamp": timestamp,
        "brand":                brand,

        # Paths
        "input_messages_csv":   _rel(input_path),
        "output_threads_jsonl": _rel(output_jsonl),
        "output_metadata_json": _rel(output_meta),

        # Volume
        "input_message_count":  n_input_messages,
        "total_threads":        summary["total_threads"],
        "total_messages_in_threads": summary["total_messages"],
        "duplicate_tweet_ids_in_input":  len(duplicate_ids),
        "conflicting_duplicates_in_input": len(conflict_ids),

        # Thread summary
        "thread_summary": {
            "single_message_threads":   single_msg_threads,
            "multi_message_threads":    summary["multi_turn"],
            "complete_threads":         summary["complete"],
            "incomplete_threads":       summary["incomplete"],
            "ends_with_brand_threads":  summary["ends_with_brand_reply"],
            "extraction_dependent_final_brand": True,
            "avg_messages_per_thread":  summary["avg_messages_per_thread"],
            "median_messages_per_thread": summary["median_messages_per_thread"],
            "max_messages_per_thread":  summary["max_turns"],
            "min_messages_per_thread":  summary["min_turns"],
            "threads_with_both_directions":    summary["has_both_directions"],
            "threads_with_3plus_messages":     summary["has_3plus_messages"],
            "threads_with_multiple_exchanges": summary["has_multiple_exchanges"],
            "customer_messages_total": summary["customer_messages"],
            "brand_messages_total":    summary["brand_messages"],
        },

        # Link metrics with explicit denominators
        "link_metrics": link_m,

        # Quality counts
        "quality_metrics": {
            "threads_with_missing_parent":     missing_parent_threads,
            "threads_with_missing_response_target": missing_response_threads,
            "threads_with_cycle_detected":     cycle_threads,
            "threads_with_malformed_timestamps": malformed_ts_threads,
            "threads_with_duplicate_ids":      dup_id_threads,
            "threads_with_conflicting_duplicates": conflict_threads,
            "threads_with_empty_text":         empty_text_threads,
            "single_message_threads":          single_msg_threads,
            "threads_with_no_customer_message": no_customer_threads,
            "threads_with_no_brand_message":   no_brand_threads,
        },

        # Definitions
        "metric_definitions": {
            "ends_with_brand_reply":  "Chronologically last message in thread was authored by brand (outbound).",
            "is_complete":            "EXTRACTION-DEPENDENT: Equivalent to ends_with_brand_reply. Because Phase 3 only extracted customer tweets that were replied to by the brand, every leaf in the extracted conversation graph terminates at a brand message by construction. This does NOT indicate true customer resolution or satisfaction.",
            "has_both_directions":    "Thread contains ≥1 inbound AND ≥1 outbound message.",
            "has_3plus_messages":     "Thread contains ≥3 messages.",
            "has_multiple_exchanges": "Thread contains ≥2 customer messages AND ≥2 brand messages (≥4 total).",
            "has_broken_links":       "≥1 in_response_to_tweet_id target absent from the dataset.",
            "cycle_detected":         "BFS encountered an already-visited node (includes diamond patterns).",
            "pct_valid_parent_links": "parent_links_valid / messages_with_parent_link × 100",
            "pct_valid_response_ids": "response_link_ids_valid / (response_link_ids_valid + response_link_ids_missing) × 100",
            "threads_with_missing_response_target": "Count of threads containing ≥1 message whose response_tweet_id references an ID not found in the dataset (measured at thread level).",
        },

        # Performance
        "elapsed_seconds": round(elapsed_s, 1),

        # Limitations
        "limitations": [
            "Threads with broken links are reconstructed from the orphan node; "
            "missing context from earlier turns cannot be recovered.",
            "cycle_detected covers both true parent-link cycles and diamond-shaped "
            "graphs; it is a conservative flag.",
            "ends_with_brand_threads / is_complete is 100% in this dataset as a "
            "direct consequence of the Phase 3 extraction filter, which collected "
            "customer tweets only when replied to by AppleSupport. It is an "
            "extraction-dependent structural property, NOT proof of genuine conversation "
            "resolution or customer satisfaction. Indeed, 8,848 final brand messages "
            "(10.8%) have outgoing response_tweet_ids pointing to subsequent customer "
            "tweets on Twitter that were not extracted.",
            "has_multiple_exchanges requires ≥2 customer AND ≥2 brand messages (≥4 total); "
            "a simple customer→brand pair is NOT counted as multi-exchange.",
            "Customer tweets that @mentioned the brand but were not replied to "
            "are absent from the input (unanswered tweet exclusion from Phase 3).",
        ],

        "next_phase": (
            "Phase 5: intent classification using "
            + _rel(output_jsonl)
        ),
    }


# ---------------------------------------------------------------------------
# Overwrite guard
# ---------------------------------------------------------------------------

def check_output_exists(output_dir: Path, overwrite: bool) -> Optional[str]:
    """Return an error string if outputs exist and overwrite is False."""
    existing = [
        p for p in (
            output_dir / OUTPUT_JSONL_NAME,
            output_dir / OUTPUT_META_NAME,
        )
        if p.exists()
    ]
    if existing and not overwrite:
        return (
            f"Output file(s) already exist: {[p.name for p in existing]}. "
            "Use --overwrite to replace them."
        )
    return None


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def check_input_schema(path: Path) -> list[str]:
    """
    Read only the header row and return a list of error strings.
    Empty list means schema is valid.
    """
    try:
        header_df = pd.read_csv(path, nrows=0, dtype=str)
    except Exception as exc:
        return [f"Cannot read input file: {exc}"]
    missing = REQUIRED_COLUMNS - set(header_df.columns)
    if missing:
        return [f"Required columns missing: {sorted(missing)}"]
    return []


# ---------------------------------------------------------------------------
# High-level pipeline entry point
# ---------------------------------------------------------------------------

def run_reconstruction(
    input_path: Path,
    output_dir: Path,
    brand: str       = DEFAULT_BRAND,
    overwrite: bool  = False,
    strict: bool     = False,
) -> dict:
    """
    Run the full Phase 4 reconstruction pipeline.

    Raises
    ------
    FileNotFoundError  : input CSV does not exist.
    ValueError         : schema errors.
    FileExistsError    : output exists and overwrite is False.
    RuntimeError       : no threads reconstructed, or strict+warnings.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    guard = check_output_exists(output_dir, overwrite)
    if guard:
        raise FileExistsError(guard)

    schema_errors = check_input_schema(input_path)
    if schema_errors:
        raise ValueError("Schema errors: " + "; ".join(schema_errors))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    log.info("Loading messages from %s …", _rel(input_path))
    t0 = time.time()
    df = pd.read_csv(input_path, dtype=str, encoding="utf-8", low_memory=False)
    n_input = len(df)
    log.info("  Loaded %s rows", f"{n_input:,}")

    # Detect duplicates before BFS
    duplicate_ids, conflict_ids = _detect_duplicates(df)
    if duplicate_ids:
        log.warning(
            "  %s duplicate tweet_id(s) detected (%s with conflicts)",
            len(duplicate_ids), len(conflict_ids)
        )

    log.info("Reconstructing threads …")
    threads = reconstruct_threads(df)
    t1 = time.time()

    if not threads:
        raise RuntimeError("Reconstruction produced 0 threads.")

    log.info(
        "  Threads: %s  |  Messages: %s  |  Elapsed: %.1fs",
        f"{len(threads):,}", f"{sum(t.turn_count for t in threads):,}", t1 - t0
    )

    # Write JSONL
    output_jsonl = output_dir / OUTPUT_JSONL_NAME
    output_meta  = output_dir / OUTPUT_META_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Writing JSONL to %s …", _rel(output_jsonl))
    written = write_jsonl(threads, output_jsonl)
    t2 = time.time()
    log.info("  Written %s thread records  |  %.1fs", f"{written:,}", t2 - t1)

    # Generate metadata
    metadata = generate_reconstruction_metadata(
        brand=brand,
        input_path=input_path,
        output_jsonl=output_jsonl,
        output_meta=output_meta,
        n_input_messages=n_input,
        threads=threads,
        duplicate_ids=duplicate_ids,
        conflict_ids=conflict_ids,
        elapsed_s=t2 - t0,
        timestamp=ts,
    )
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Metadata written: %s", _rel(output_meta))
    log.info("Total elapsed: %.1fs", t2 - t0)

    # Strict mode
    qm = metadata["quality_metrics"]
    warnings = [k for k, v in qm.items() if v and v > 0]
    if strict and warnings:
        raise RuntimeError(
            f"--strict mode: quality warnings present: {warnings}"
        )

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 4: Reconstruct AppleSupport conversation threads from "
            "the Phase 3 message CSV.\n"
            "Produces apple_support_threads.jsonl and "
            "apple_support_thread_metadata.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, metavar="PATH",
        help="Path to the Phase 3 messages CSV",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR, metavar="DIR",
        help="Output directory (default: data/processed/apple_support)",
    )
    p.add_argument(
        "--brand", type=str, default=DEFAULT_BRAND,
        help=f"Brand name for metadata (default: {DEFAULT_BRAND})",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing existing output files",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Treat quality warnings as errors (exit 1)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        meta = run_reconstruction(
            input_path=args.input,
            output_dir=args.output_dir,
            brand=args.brand,
            overwrite=args.overwrite,
            strict=args.strict,
        )
    except FileNotFoundError as exc:
        log.error("Input not found: %s", exc)
        sys.exit(1)
    except FileExistsError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    ts = meta["thread_summary"]
    print()
    print("  ── Thread Reconstruction Complete ─────────────────────")
    print(f"  Brand:                 {meta['brand']}")
    print(f"  Input messages:        {meta['input_message_count']:,}")
    print(f"  Threads:               {meta['total_threads']:,}")
    print(f"  3+ message threads:    {ts['threads_with_3plus_messages']:,}")
    print(f"  Multi-exchange:        {ts['threads_with_multiple_exchanges']:,}")
    print(f"  Complete (ends brand): {ts['complete_threads']:,}")
    print(f"  Broken-link threads:   {meta['quality_metrics']['threads_with_missing_parent']:,}")
    print(f"  Avg messages/thread:   {ts['avg_messages_per_thread']}")
    print(f"  Output JSONL:          {meta['output_threads_jsonl']}")
    print(f"  Metadata:              {meta['output_metadata_json']}")
    print(f"  Elapsed:               {meta['elapsed_seconds']}s")
    print()


if __name__ == "__main__":
    main()
