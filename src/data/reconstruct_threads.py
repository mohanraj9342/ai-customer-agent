"""
src/data/reconstruct_threads.py
================================
Phase 2 — Conversation thread reconstruction.

PURPOSE
-------
Given a brand-specific tweet DataFrame (produced by extract_brand.py), rebuild
multi-turn customer-support conversations in chronological order.

A "thread" is a sequence of tweets connected by parent–child links:
  tweet A ──(in_response_to)──▶ tweet B ──(in_response_to)──▶ tweet C …

The reconstruction algorithm walks from every root tweet (one with no parent
present in the dataset) forward through the reply chain using
``in_response_to_tweet_id`` links.

DESIGN DECISIONS
----------------
1. We use ``in_response_to_tweet_id`` as the primary edge (child → parent).
   ``response_tweet_id`` is used as a fallback to detect reply IDs when the
   child row is missing.
2. We detect and break cycles (malformed data) by tracking visited tweet IDs
   per thread walk.
3. Threads with only one message are valid; they represent unanswered tweets or
   isolated messages.
4. Missing parent tweets (where the parent tweet_id is not in our filtered
   subset) result in the thread starting at the earliest available node.
5. Chronological ordering uses ``created_at`` after reconstructing the
   structural order, so threads are sorted by time even if the CSV is not.

OUTPUT
------
Returns a list of Thread dataclass instances.  Callers decide how to serialise
them (the module itself does not write to disk, keeping it testable).

USAGE
-----
    import pandas as pd
    from src.data.reconstruct_threads import reconstruct_threads

    df = pd.read_csv("data/processed/applesupport_tweets.csv", dtype=str)
    threads = reconstruct_threads(df)
    for t in threads[:5]:
        print(t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Thread:
    """
    A reconstructed customer-support conversation thread.

    Attributes
    ----------
    root_tweet_id : str
        The tweet_id of the first (oldest) message in this thread.
    tweet_ids : list[str]
        Ordered list of tweet_ids in chronological order.
    texts : list[str]
        Text of each tweet, aligned with tweet_ids.
    directions : list[str]
        "inbound" or "outbound" for each message, aligned with tweet_ids.
    timestamps : list[str]
        Raw created_at string for each message, aligned with tweet_ids.
    turn_count : int
        Number of messages in the thread.
    is_complete : bool
        True if the thread ends with an outbound (brand) reply.
        False if the last message is inbound (unanswered) or the thread
        is a single isolated message.
    has_broken_links : bool
        True if any expected parent or child tweet was missing from the
        filtered dataset.
    """

    root_tweet_id: str
    tweet_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    turn_count: int = 0
    is_complete: bool = False
    has_broken_links: bool = False

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "incomplete"
        broken = " [broken links]" if self.has_broken_links else ""
        return (
            f"Thread(root={self.root_tweet_id!r}, turns={self.turn_count}, "
            f"{status}{broken})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _build_lookup_tables(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Build three lookup tables from the DataFrame for O(1) access:

      id_to_row       : tweet_id → row dict
      child_to_parent : tweet_id → in_response_to_tweet_id (or None)
      parent_to_children : in_response_to_tweet_id → [child tweet_ids]
    """
    id_to_row: dict[str, dict] = {}
    child_to_parent: dict[str, Optional[str]] = {}
    parent_to_children: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        tid = str(row["tweet_id"]).strip()
        parent = row.get("in_response_to_tweet_id", None)
        parent_str = (
            str(parent).strip()
            if parent and str(parent).strip() not in ("", "nan")
            else None
        )

        id_to_row[tid] = row.to_dict()
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

    Returns a list of (tweet_id, has_broken_links) tuples where
    has_broken_links is True when the tweet has a parent ID that is simply
    absent from the filtered dataset (i.e. a broken link, not a true root).
    """
    roots = []
    for tid in id_to_row:
        parent = child_to_parent.get(tid)
        if parent is None or not parent:
            # True root: no parent recorded at all
            roots.append((tid, False))
        elif parent not in all_ids:
            # Apparent root: parent exists in raw data but not in our subset
            roots.append((tid, True))
    return roots


def _walk_thread(
    root_id: str,
    id_to_row: dict,
    parent_to_children: dict,
    initial_broken: bool = False,
) -> Thread:
    """
    Walk forward from a root tweet, building an ordered thread.

    Uses BFS (breadth-first) to handle branching replies.  When multiple
    children exist, all are included — the thread may have a tree-like shape
    but we flatten it in timestamp order.

    Cycle detection: we track visited IDs and stop if we revisit one.
    """
    visited: set[str] = set()
    collected: list[dict] = []
    # Start with True if the root itself had a broken link (missing parent)
    has_broken_links = initial_broken

    queue = [root_id]
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        if current_id not in id_to_row:
            has_broken_links = True
            continue
        visited.add(current_id)
        collected.append(id_to_row[current_id])

        # Enqueue children (tweets that replied to this one)
        children = parent_to_children.get(current_id, [])
        queue.extend(children)

    if not collected:
        # Degenerate case: root was not in the dataset
        return Thread(
            root_tweet_id=root_id,
            has_broken_links=True,
        )

    # Sort collected messages chronologically by created_at
    # The Twitter timestamp format is: "Tue Oct 31 22:10:47 +0000 2017"
    # We sort lexicographically on this string after converting to a sortable form.
    # For robustness, fall back to original order if parsing fails.
    try:
        collected.sort(key=lambda r: pd.to_datetime(r.get("created_at", ""), utc=True))
    except Exception:
        pass  # keep original BFS order

    tweet_ids = [str(r["tweet_id"]) for r in collected]
    texts = [str(r.get("text", "")) for r in collected]
    directions = [
        "inbound" if str(r.get("inbound", "False")).strip().lower() in ("true",) else "outbound"
        for r in collected
    ]
    timestamps = [str(r.get("created_at", "")) for r in collected]

    last_direction = directions[-1] if directions else "outbound"
    is_complete = last_direction == "outbound"

    return Thread(
        root_tweet_id=root_id,
        tweet_ids=tweet_ids,
        texts=texts,
        directions=directions,
        timestamps=timestamps,
        turn_count=len(collected),
        is_complete=is_complete,
        has_broken_links=has_broken_links,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def reconstruct_threads(df: pd.DataFrame) -> list[Thread]:
    """
    Reconstruct all conversation threads from a brand-specific tweet DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: tweet_id, inbound, created_at, text,
        in_response_to_tweet_id, response_tweet_id.
        All columns should be string dtype.

    Returns
    -------
    list[Thread]
        One Thread per reconstructed conversation, ordered by root tweet_id.
        May include single-message threads for isolated tweets.

    Notes
    -----
    - Threads with broken links (missing parents/children in the filtered
      subset) are included but flagged with ``has_broken_links=True``.
    - Multi-branch threads (where one tweet received multiple replies) are
      flattened into a single time-sorted sequence.
    - The algorithm is O(n) in the number of rows.
    """
    if df.empty:
        return []

    # Ensure tweet_id is a string column without leading/trailing whitespace
    df = df.copy()
    for col in ("tweet_id", "in_response_to_tweet_id", "response_tweet_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", "")

    id_to_row, child_to_parent, parent_to_children = _build_lookup_tables(df)
    all_ids = set(id_to_row.keys())

    roots = _find_roots(id_to_row, child_to_parent, all_ids)

    # Cycle fallback: if some nodes form a mutual-parent cycle with no external
    # root, none of them appear in the roots list.  Find any unvisited nodes
    # and treat the lowest tweet_id as an arbitrary root (broken link).
    rooted_ids = {rid for rid, _ in roots}

    threads = []
    visited_in_threads: set[str] = set()

    for root_id, initial_broken in roots:
        thread = _walk_thread(root_id, id_to_row, parent_to_children,
                              initial_broken=initial_broken)
        threads.append(thread)
        visited_in_threads.update(thread.tweet_ids)

    # Cycle fallback: walk any nodes not yet assigned to a thread.
    # These are nodes trapped in mutual-parent cycles with no external root.
    unvisited = sorted(set(id_to_row.keys()) - visited_in_threads)
    for tid in unvisited:
        if tid not in visited_in_threads:
            thread = _walk_thread(tid, id_to_row, parent_to_children,
                                  initial_broken=True)
            threads.append(thread)
            visited_in_threads.update(thread.tweet_ids)

    return threads



def summarise_threads(threads: list[Thread]) -> dict:
    """
    Compute summary statistics over a list of reconstructed threads.

    Returns a dict with:
      total_threads, single_turn, multi_turn, complete, incomplete,
      broken_links, avg_turns, max_turns, min_turns
    """
    if not threads:
        return {
            "total_threads": 0,
            "single_turn": 0,
            "multi_turn": 0,
            "complete": 0,
            "incomplete": 0,
            "broken_links": 0,
            "avg_turns": 0.0,
            "max_turns": 0,
            "min_turns": 0,
        }

    turn_counts = [t.turn_count for t in threads]
    return {
        "total_threads": len(threads),
        "single_turn": sum(1 for t in threads if t.turn_count == 1),
        "multi_turn": sum(1 for t in threads if t.turn_count > 1),
        "complete": sum(1 for t in threads if t.is_complete),
        "incomplete": sum(1 for t in threads if not t.is_complete),
        "broken_links": sum(1 for t in threads if t.has_broken_links),
        "avg_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "max_turns": max(turn_counts),
        "min_turns": min(turn_counts),
    }
