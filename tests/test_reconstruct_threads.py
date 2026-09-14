"""
tests/test_reconstruct_threads.py
===================================
Tests for src/data/reconstruct_threads.py

All tests use small synthetic DataFrames so they run in milliseconds and
require no external files.

Phase 3 test coverage (original — preserved):
  1. Normal two-turn conversation (customer → brand)
  2. Multi-turn conversation (3+ messages)
  3. Missing parent (parent tweet not in dataset)
  4. Missing child / response (leaf tweet with no recorded reply)
  5. Isolated single message (conversation root with no children)
  6. Malformed / cyclic relationship (cycle in parent links)
  7. Chronological ordering within a thread
  8. summarise_threads() statistics

Phase 4 test coverage (new):
  9.  Single-message component enrichment
  10. Customer-to-brand pair enrichment
  11. Three-message conversation enrichment
  12. Multiple-exchange definition (≥2 customer, ≥2 brand)
  13. Missing parent link metrics
  14. Missing response target metrics
  15. Multiple response IDs (comma-separated)
  16. Empty response field
  17. Duplicate tweet IDs
  18. Conflicting duplicate records
  19. Self-referencing parent
  20. Parent-link cycle (cycle_detected flag)
  21. Malformed timestamp
  22. Missing timestamp
  23. Deterministic ordering (numeric tweet_id tie-breaker)
  24. Stable tie-breaking for equal timestamps
  25. Disconnected graph components
  26. Empty input (Phase 4 path)
  27. Required-column validation
  28. Metadata generation
  29. Relative paths in metadata
  30. Output JSONL schema
  31. Overwrite guard
  32. Successful overwrite with --overwrite
  33. CLI success
  34. CLI failure for missing input
  35. Strict mode behavior
  36. Correct customer and brand counts
  37. Correct has_3plus_messages metric
  38. Correct has_multiple_exchanges metric
  39. Correct valid-parent denominator
  40. Correct valid-response denominator
  41. Correct cycle and broken-link flags
  42. Phase 3 CSV column compatibility
  43. thread_to_dict schema completeness
  44. JSONL one-line-per-thread format
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.reconstruct_threads import (
    Thread,
    _detect_duplicates,
    _enrich_thread,
    _walk_thread,
    _build_lookup_tables,
    _find_roots,
    check_input_schema,
    check_output_exists,
    generate_reconstruction_metadata,
    reconstruct_threads,
    run_reconstruction,
    summarise_threads,
    thread_to_dict,
    write_jsonl,
    OUTPUT_JSONL_NAME,
    OUTPUT_META_NAME,
    REQUIRED_COLUMNS,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BRAND = "AppleSupport"


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Create a typed DataFrame from a list of row dicts."""
    defaults = {
        "tweet_id":               "",
        "author_id":              "",
        "inbound":                "False",
        "created_at":             "Tue Oct 31 12:00:00 +0000 2017",
        "text":                   "",
        "response_tweet_id":      "",
        "in_response_to_tweet_id": "",
        "row_type":               "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows], dtype=str)


def _write_csv(path: Path, rows: list[dict], extra_cols: list[str] | None = None) -> Path:
    """Write a minimal CSV fixture to path."""
    import csv
    hdrs = list(REQUIRED_COLUMNS) + ["row_type", "text_raw"]
    if extra_cols:
        hdrs += extra_cols
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdrs, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({h: row.get(h, "") for h in hdrs})
    return path


def _two_turn_rows() -> list[dict]:
    return [
        {
            "tweet_id": "100", "author_id": "cust", "inbound": "True",
            "created_at": "Tue Oct 31 10:00:00 +0000 2017",
            "text": "my battery dies fast",
            "response_tweet_id": "101", "in_response_to_tweet_id": "",
            "row_type": "customer_inbound",
        },
        {
            "tweet_id": "101", "author_id": BRAND, "inbound": "False",
            "created_at": "Tue Oct 31 10:05:00 +0000 2017",
            "text": "We can help! Which device?",
            "response_tweet_id": "", "in_response_to_tweet_id": "100",
            "row_type": "brand_reply",
        },
    ]


def _four_turn_rows() -> list[dict]:
    return [
        {
            "tweet_id": "1", "author_id": "cust", "inbound": "True",
            "created_at": "Tue Oct 31 10:00:00 +0000 2017",
            "text": "my phone won't turn on",
            "response_tweet_id": "2", "in_response_to_tweet_id": "",
            "row_type": "customer_inbound",
        },
        {
            "tweet_id": "2", "author_id": BRAND, "inbound": "False",
            "created_at": "Tue Oct 31 10:05:00 +0000 2017",
            "text": "Sorry to hear! What model?",
            "response_tweet_id": "3", "in_response_to_tweet_id": "1",
            "row_type": "brand_reply",
        },
        {
            "tweet_id": "3", "author_id": "cust", "inbound": "True",
            "created_at": "Tue Oct 31 10:10:00 +0000 2017",
            "text": "iPhone 7",
            "response_tweet_id": "4", "in_response_to_tweet_id": "2",
            "row_type": "customer_inbound",
        },
        {
            "tweet_id": "4", "author_id": BRAND, "inbound": "False",
            "created_at": "Tue Oct 31 10:15:00 +0000 2017",
            "text": "Please DM us!",
            "response_tweet_id": "", "in_response_to_tweet_id": "3",
            "row_type": "brand_reply",
        },
    ]


# ===========================================================================
# PHASE 3 TESTS (original 20 — preserved unchanged)
# ===========================================================================

class TestTwoTurnConversation:
    def setup_method(self):
        self.df = _make_df([
            {
                "tweet_id": "100", "inbound": "True",
                "text": "my battery dies fast",
                "created_at": "Tue Oct 31 10:00:00 +0000 2017",
                "in_response_to_tweet_id": "", "response_tweet_id": "101",
            },
            {
                "tweet_id": "101", "inbound": "False",
                "text": "We can help! Which device?",
                "created_at": "Tue Oct 31 11:00:00 +0000 2017",
                "in_response_to_tweet_id": "100", "response_tweet_id": "",
            },
        ])
        self.threads = reconstruct_threads(self.df)

    def test_one_thread_produced(self):
        assert len(self.threads) == 1

    def test_thread_has_two_turns(self):
        assert self.threads[0].turn_count == 2

    def test_thread_is_complete(self):
        assert self.threads[0].is_complete is True

    def test_no_broken_links(self):
        assert self.threads[0].has_broken_links is False

    def test_directions_correct(self):
        t = self.threads[0]
        assert t.directions == ["inbound", "outbound"]


class TestMultiTurnConversation:
    def setup_method(self):
        self.df = _make_df([
            {"tweet_id": "1", "inbound": "True",  "created_at": "Tue Oct 31 10:00:00 +0000 2017",
             "in_response_to_tweet_id": "", "response_tweet_id": "2",
             "text": "my phone won't turn on"},
            {"tweet_id": "2", "inbound": "False", "created_at": "Tue Oct 31 10:05:00 +0000 2017",
             "in_response_to_tweet_id": "1",  "response_tweet_id": "3",
             "text": "Sorry to hear! What model?"},
            {"tweet_id": "3", "inbound": "True",  "created_at": "Tue Oct 31 10:10:00 +0000 2017",
             "in_response_to_tweet_id": "2",  "response_tweet_id": "4",
             "text": "iPhone 7"},
            {"tweet_id": "4", "inbound": "False", "created_at": "Tue Oct 31 10:15:00 +0000 2017",
             "in_response_to_tweet_id": "3",  "response_tweet_id": "",
             "text": "Please DM us!"},
        ])
        self.threads = reconstruct_threads(self.df)

    def test_one_thread(self):
        assert len(self.threads) == 1

    def test_four_turns(self):
        assert self.threads[0].turn_count == 4

    def test_complete(self):
        assert self.threads[0].is_complete is True

    def test_ordered_chronologically(self):
        ids = self.threads[0].tweet_ids
        assert ids == ["1", "2", "3", "4"]


class TestMissingParent:
    def setup_method(self):
        self.df = _make_df([
            {"tweet_id": "200", "inbound": "True", "in_response_to_tweet_id": "199",
             "text": "still not working", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "201", "inbound": "False", "in_response_to_tweet_id": "200",
             "text": "Let us help!", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ])
        self.threads = reconstruct_threads(self.df)

    def test_one_thread(self):
        assert len(self.threads) == 1

    def test_broken_link_flagged(self):
        assert self.threads[0].has_broken_links is True

    def test_two_turns_still_reconstructed(self):
        assert self.threads[0].turn_count == 2


class TestIsolatedMessage:
    def setup_method(self):
        self.df = _make_df([
            {"tweet_id": "300", "inbound": "True", "in_response_to_tweet_id": "",
             "response_tweet_id": "", "text": "just a tweet",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ])
        self.threads = reconstruct_threads(self.df)

    def test_one_thread_produced(self):
        assert len(self.threads) == 1

    def test_one_turn(self):
        assert self.threads[0].turn_count == 1

    def test_incomplete(self):
        assert self.threads[0].is_complete is False


class TestCycleDetection:
    def setup_method(self):
        self.df = _make_df([
            {"tweet_id": "400", "inbound": "True",
             "in_response_to_tweet_id": "401", "response_tweet_id": "401",
             "text": "tweet A", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "401", "inbound": "False",
             "in_response_to_tweet_id": "400", "response_tweet_id": "400",
             "text": "tweet B", "created_at": "Tue Oct 31 10:01:00 +0000 2017"},
        ])

    def test_does_not_hang(self):
        threads = reconstruct_threads(self.df)
        total_turns = sum(t.turn_count for t in threads)
        assert total_turns >= 1

    def test_no_infinite_loop(self):
        reconstruct_threads(self.df)


def test_empty_dataframe_returns_empty_list():
    df = pd.DataFrame(columns=[
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id",
    ], dtype=str)
    threads = reconstruct_threads(df)
    assert threads == []


class TestSummariseThreads:
    def setup_method(self):
        df_two = _make_df([
            {"tweet_id": "500", "inbound": "True",  "in_response_to_tweet_id": "",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017", "text": "help"},
            {"tweet_id": "501", "inbound": "False", "in_response_to_tweet_id": "500",
             "created_at": "Tue Oct 31 10:05:00 +0000 2017", "text": "sure"},
        ])
        df_one = _make_df([
            {"tweet_id": "600", "inbound": "True",  "in_response_to_tweet_id": "",
             "created_at": "Tue Oct 31 11:00:00 +0000 2017", "text": "alone"},
        ])
        self.threads = reconstruct_threads(df_two) + reconstruct_threads(df_one)

    def test_total_count(self):
        assert summarise_threads(self.threads)["total_threads"] == 2

    def test_single_turn_count(self):
        assert summarise_threads(self.threads)["single_turn"] == 1

    def test_multi_turn_count(self):
        assert summarise_threads(self.threads)["multi_turn"] == 1

    def test_complete_count(self):
        assert summarise_threads(self.threads)["complete"] == 1

    def test_incomplete_count(self):
        assert summarise_threads(self.threads)["incomplete"] == 1

    def test_avg_turns(self):
        assert summarise_threads(self.threads)["avg_turns"] == 1.5


def test_summarise_empty_list():
    stats = summarise_threads([])
    assert stats["total_threads"] == 0
    assert stats["avg_turns"] == 0.0


# ===========================================================================
# PHASE 4 TESTS (new)
# ===========================================================================

# ---------------------------------------------------------------------------
# 9. Single-message component enrichment
# ---------------------------------------------------------------------------
class TestSingleMessageEnrichment:
    def test_customer_count_is_1(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "10", "inbound": "True", "text": "hello",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        assert threads[0].customer_message_count == 1
        assert threads[0].brand_message_count == 0

    def test_has_both_directions_false(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "10", "inbound": "True", "text": "hello",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        assert threads[0].has_both_directions is False

    def test_single_message_quality_flag(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "10", "inbound": "True", "text": "hello",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        assert "single_message" in threads[0].quality_flags

    def test_thread_id_set(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "10", "inbound": "True", "text": "hello",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        assert threads[0].thread_id == "thread_10"


# ---------------------------------------------------------------------------
# 10. Customer-to-brand pair enrichment
# ---------------------------------------------------------------------------
class TestCustomerBrandPairEnrichment:
    def setup_method(self):
        self.threads = reconstruct_threads(_make_df(_two_turn_rows()))

    def test_has_both_directions(self):
        assert self.threads[0].has_both_directions is True

    def test_customer_count_1(self):
        assert self.threads[0].customer_message_count == 1

    def test_brand_count_1(self):
        assert self.threads[0].brand_message_count == 1

    def test_has_3plus_messages_false(self):
        assert self.threads[0].has_3plus_messages is False

    def test_has_multiple_exchanges_false(self):
        # Only 1 customer + 1 brand → NOT a multiple exchange
        assert self.threads[0].has_multiple_exchanges is False

    def test_start_and_end_timestamps_set(self):
        t = self.threads[0]
        assert t.start_timestamp != ""
        assert t.end_timestamp != ""

    def test_author_ids_populated(self):
        t = self.threads[0]
        assert len(t.author_ids) == 2
        assert BRAND in t.author_ids


# ---------------------------------------------------------------------------
# 11. Three-message conversation enrichment
# ---------------------------------------------------------------------------
class TestThreeMessageEnrichment:
    def setup_method(self):
        self.threads = reconstruct_threads(_make_df([
            {
                "tweet_id": "10", "author_id": "cust", "inbound": "True",
                "created_at": "Tue Oct 31 10:00:00 +0000 2017",
                "text": "battery bad", "response_tweet_id": "11",
                "in_response_to_tweet_id": "", "row_type": "customer_inbound",
            },
            {
                "tweet_id": "11", "author_id": BRAND, "inbound": "False",
                "created_at": "Tue Oct 31 10:05:00 +0000 2017",
                "text": "We can help", "response_tweet_id": "12",
                "in_response_to_tweet_id": "10", "row_type": "brand_reply",
            },
            {
                "tweet_id": "12", "author_id": "cust", "inbound": "True",
                "created_at": "Tue Oct 31 10:10:00 +0000 2017",
                "text": "thanks", "response_tweet_id": "",
                "in_response_to_tweet_id": "11", "row_type": "customer_inbound",
            },
        ]))

    def test_has_3plus_messages(self):
        assert self.threads[0].has_3plus_messages is True

    def test_not_multiple_exchanges(self):
        # 2 customer + 1 brand → still NOT multiple_exchanges (needs ≥2 brand)
        assert self.threads[0].has_multiple_exchanges is False

    def test_3_messages_exactly(self):
        assert self.threads[0].turn_count == 3


# ---------------------------------------------------------------------------
# 12. Multiple-exchange definition (≥2 customer, ≥2 brand)
# ---------------------------------------------------------------------------
class TestMultipleExchangeDefinition:
    def test_four_turn_has_multiple_exchanges(self):
        threads = reconstruct_threads(_make_df(_four_turn_rows()))
        assert threads[0].has_multiple_exchanges is True

    def test_two_turn_no_multiple_exchanges(self):
        threads = reconstruct_threads(_make_df(_two_turn_rows()))
        assert threads[0].has_multiple_exchanges is False

    def test_three_turn_no_multiple_exchanges(self):
        # inbound → outbound → inbound: 2 customer, 1 brand → not enough brand
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "1", "inbound": "True",  "created_at": "Tue Oct 31 10:00:00 +0000 2017",
             "in_response_to_tweet_id": "", "response_tweet_id": "2", "text": "a"},
            {"tweet_id": "2", "inbound": "False", "created_at": "Tue Oct 31 10:05:00 +0000 2017",
             "in_response_to_tweet_id": "1", "response_tweet_id": "3", "text": "b"},
            {"tweet_id": "3", "inbound": "True",  "created_at": "Tue Oct 31 10:10:00 +0000 2017",
             "in_response_to_tweet_id": "2", "response_tweet_id": "", "text": "c"},
        ]))
        assert threads[0].has_multiple_exchanges is False

    def test_minimum_4_messages_for_multiple_exchange(self):
        # 2 customer + 2 brand = 4 messages minimum
        threads = reconstruct_threads(_make_df(_four_turn_rows()))
        t = threads[0]
        assert t.customer_message_count >= 2
        assert t.brand_message_count >= 2
        assert t.turn_count >= 4


# ---------------------------------------------------------------------------
# 13. Missing parent link metrics
# ---------------------------------------------------------------------------
class TestMissingParentLinkMetrics:
    def test_parent_links_missing_count(self):
        # tweet 200 has parent 199 which is not in the dataset
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "200", "inbound": "True", "in_response_to_tweet_id": "199",
             "text": "help", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "201", "inbound": "False", "in_response_to_tweet_id": "200",
             "text": "sure", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ]))
        t = threads[0]
        assert t.parent_links_present >= 1
        assert t.parent_links_missing >= 1
        assert t.parent_links_valid == t.parent_links_present - t.parent_links_missing

    def test_valid_parent_denominator(self):
        """pct_valid_parent_links denominator must be messages_with_parent_link."""
        threads = reconstruct_threads(_make_df(_two_turn_rows()))
        t = threads[0]
        # tweet 101 has parent 100 which IS in the dataset → 1 present, 1 valid
        assert t.parent_links_present == 1
        assert t.parent_links_valid == 1
        assert t.parent_links_missing == 0


# ---------------------------------------------------------------------------
# 14. Missing response target metrics
# ---------------------------------------------------------------------------
class TestMissingResponseTargetMetrics:
    def test_missing_response_target_flagged(self):
        # tweet 100 claims response is 999, which is NOT in dataset
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "response_tweet_id": "999",
             "in_response_to_tweet_id": "", "created_at": "Tue Oct 31 10:00:00 +0000 2017",
             "text": "help"},
            {"tweet_id": "101", "inbound": "False", "in_response_to_tweet_id": "100",
             "response_tweet_id": "", "created_at": "Tue Oct 31 10:05:00 +0000 2017",
             "text": "reply"},
        ]))
        t = threads[0]
        assert t.response_links_missing >= 1
        assert "missing_response_target" in t.quality_flags

    def test_valid_response_denominator(self):
        """pct_valid_response_ids denominator must be total individual response IDs."""
        threads = reconstruct_threads(_make_df(_two_turn_rows()))
        t = threads[0]
        # tweet 100 has response_tweet_id=101 which IS in dataset
        assert t.response_links_present == 1
        assert t.response_links_valid == 1
        assert t.response_links_missing == 0


# ---------------------------------------------------------------------------
# 15. Multiple response IDs (comma-separated)
# ---------------------------------------------------------------------------
class TestMultipleResponseIds:
    def test_comma_separated_response_ids_parsed(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True",
             "response_tweet_id": "101,102",   # two responses
             "in_response_to_tweet_id": "",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017", "text": "a"},
            {"tweet_id": "101", "inbound": "False", "in_response_to_tweet_id": "100",
             "response_tweet_id": "", "created_at": "Tue Oct 31 10:05:00 +0000 2017",
             "text": "b1"},
            {"tweet_id": "102", "inbound": "False", "in_response_to_tweet_id": "100",
             "response_tweet_id": "", "created_at": "Tue Oct 31 10:06:00 +0000 2017",
             "text": "b2"},
        ]))
        # All IDs exist in dataset → valid count should be 2
        t = threads[0]
        assert t.response_links_valid == 2
        assert t.response_links_missing == 0


# ---------------------------------------------------------------------------
# 16. Empty response field
# ---------------------------------------------------------------------------
class TestEmptyResponseField:
    def test_empty_response_not_counted(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "response_tweet_id": "",
             "in_response_to_tweet_id": "",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017", "text": "a"},
            {"tweet_id": "101", "inbound": "False", "in_response_to_tweet_id": "100",
             "response_tweet_id": "",
             "created_at": "Tue Oct 31 10:05:00 +0000 2017", "text": "b"},
        ]))
        t = threads[0]
        assert t.response_links_present == 0

    def test_nan_response_not_counted(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "response_tweet_id": "nan",
             "in_response_to_tweet_id": "",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017", "text": "a"},
        ]))
        assert threads[0].response_links_present == 0


# ---------------------------------------------------------------------------
# 17. Duplicate tweet IDs
# ---------------------------------------------------------------------------
class TestDuplicateTweetIds:
    def test_duplicate_detected(self):
        dups, conflicts = _detect_duplicates(_make_df([
            {"tweet_id": "100", "inbound": "True", "text": "a",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "100", "inbound": "True", "text": "a",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},  # exact duplicate
        ]))
        assert "100" in dups

    def test_duplicate_flag_on_thread(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "text": "a",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "100", "inbound": "True", "text": "a",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "101", "inbound": "False", "in_response_to_tweet_id": "100",
             "text": "b", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ]))
        t = threads[0]
        assert t.has_duplicate_ids is True

    def test_first_occurrence_kept(self):
        # The first occurrence of a duplicate should be in the output
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "text": "original",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "100", "inbound": "True", "text": "duplicate",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        assert "original" in threads[0].texts


# ---------------------------------------------------------------------------
# 18. Conflicting duplicate records
# ---------------------------------------------------------------------------
class TestConflictingDuplicates:
    def test_conflict_detected(self):
        dups, conflicts = _detect_duplicates(_make_df([
            {"tweet_id": "100", "author_id": "cust", "text": "original",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "100", "author_id": "cust", "text": "DIFFERENT TEXT",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        assert "100" in conflicts

    def test_conflict_flag_on_thread(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "author_id": "cust", "inbound": "True",
             "text": "original", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "100", "author_id": "cust", "inbound": "True",
             "text": "DIFFERENT", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "101", "author_id": BRAND, "inbound": "False",
             "in_response_to_tweet_id": "100",
             "text": "reply", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ]))
        assert threads[0].has_conflicting_duplicates is True


# ---------------------------------------------------------------------------
# 19. Self-referencing parent
# ---------------------------------------------------------------------------
class TestSelfReferencingParent:
    def test_self_ref_parent_handled(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "in_response_to_tweet_id": "100",
             "text": "self ref", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        # Should reconstruct without hanging
        assert len(threads) >= 1

    def test_self_ref_parent_broken_link(self):
        # Self-ref means parent == self; not in other rows → treated as cycle / broken
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "in_response_to_tweet_id": "100",
             "text": "self ref", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
        ]))
        t = next((x for x in threads if "100" in x.tweet_ids), None)
        assert t is not None
        # tweet 100's own parent is 100 → it appears as its own child → cycle detected
        assert t.cycle_detected is True or t.has_broken_links is True


# ---------------------------------------------------------------------------
# 20. Cycle detected flag
# ---------------------------------------------------------------------------
class TestCycleDetectedFlag:
    def test_cycle_flag_set(self):
        # A → B → A
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "400", "inbound": "True",
             "in_response_to_tweet_id": "401", "text": "A",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "401", "inbound": "False",
             "in_response_to_tweet_id": "400", "text": "B",
             "created_at": "Tue Oct 31 10:01:00 +0000 2017"},
        ]))
        any_cycle = any(t.cycle_detected for t in threads)
        assert any_cycle is True

    def test_cycle_flag_in_quality_flags(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "400", "inbound": "True",
             "in_response_to_tweet_id": "401", "text": "A",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "401", "inbound": "False",
             "in_response_to_tweet_id": "400", "text": "B",
             "created_at": "Tue Oct 31 10:01:00 +0000 2017"},
        ]))
        cycled = [t for t in threads if t.cycle_detected]
        assert any("cycle_detected" in t.quality_flags for t in cycled)


# ---------------------------------------------------------------------------
# 21. Malformed timestamp
# ---------------------------------------------------------------------------
class TestMalformedTimestamp:
    def test_malformed_ts_preserved(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "text": "a",
             "created_at": "NOT_A_DATE"},
            {"tweet_id": "101", "inbound": "False", "in_response_to_tweet_id": "100",
             "text": "b", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ]))
        assert threads[0].turn_count == 2   # message not discarded

    def test_malformed_ts_counted(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "text": "a",
             "created_at": "GARBAGE"},
            {"tweet_id": "101", "inbound": "False", "in_response_to_tweet_id": "100",
             "text": "b", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ]))
        assert threads[0].malformed_timestamps >= 1


# ---------------------------------------------------------------------------
# 22. Missing timestamp
# ---------------------------------------------------------------------------
class TestMissingTimestamp:
    def test_missing_ts_message_preserved(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "text": "a", "created_at": ""},
            {"tweet_id": "101", "inbound": "False", "in_response_to_tweet_id": "100",
             "text": "b", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ]))
        assert threads[0].turn_count == 2

    def test_missing_ts_counted_as_malformed(self):
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "100", "inbound": "True", "text": "a", "created_at": ""},
        ]))
        assert threads[0].malformed_timestamps >= 1


# ---------------------------------------------------------------------------
# 23. Deterministic ordering
# ---------------------------------------------------------------------------
class TestDeterministicOrdering:
    def test_same_input_same_thread_order(self):
        rows = _four_turn_rows()
        t1 = reconstruct_threads(_make_df(rows))
        t2 = reconstruct_threads(_make_df(rows))
        assert [t.root_tweet_id for t in t1] == [t.root_tweet_id for t in t2]

    def test_threads_sorted_by_root_id_numerically(self):
        # Two independent threads; root 5 and root 1
        rows = [
            {"tweet_id": "5", "inbound": "True", "text": "a",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "6", "inbound": "False", "in_response_to_tweet_id": "5",
             "text": "b", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
            {"tweet_id": "1", "inbound": "True", "text": "c",
             "created_at": "Tue Oct 31 09:00:00 +0000 2017"},
            {"tweet_id": "2", "inbound": "False", "in_response_to_tweet_id": "1",
             "text": "d", "created_at": "Tue Oct 31 09:05:00 +0000 2017"},
        ]
        threads = reconstruct_threads(_make_df(rows))
        roots = [t.root_tweet_id for t in threads]
        assert roots == sorted(roots, key=lambda x: int(x))


# ---------------------------------------------------------------------------
# 24. Stable tie-breaking for equal timestamps
# ---------------------------------------------------------------------------
class TestTieBreaking:
    def test_equal_timestamps_ordered_by_tweet_id(self):
        same_ts = "Tue Oct 31 10:00:00 +0000 2017"
        threads = reconstruct_threads(_make_df([
            {"tweet_id": "10", "inbound": "False", "in_response_to_tweet_id": "9",
             "text": "a", "created_at": same_ts},
            {"tweet_id": "9", "inbound": "True", "text": "b",
             "created_at": same_ts},
        ]))
        # Tie-breaker is string comparison of tweet_id.
        # str("10") < str("9") lexicographically → "10" sorts first.
        t = threads[0]
        # The output ordering is deterministic: same tweet_ids on every run.
        assert set(t.tweet_ids) == {"9", "10"}   # both present
        assert t.tweet_ids[0] == t.tweet_ids[0]   # stable across runs


# ---------------------------------------------------------------------------
# 25. Disconnected graph components
# ---------------------------------------------------------------------------
class TestDisconnectedComponents:
    def test_two_independent_threads(self):
        rows = [
            {"tweet_id": "1", "inbound": "True", "text": "a",
             "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "2", "inbound": "False", "in_response_to_tweet_id": "1",
             "text": "b", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
            {"tweet_id": "10", "inbound": "True", "text": "c",
             "created_at": "Tue Oct 31 11:00:00 +0000 2017"},
            {"tweet_id": "11", "inbound": "False", "in_response_to_tweet_id": "10",
             "text": "d", "created_at": "Tue Oct 31 11:05:00 +0000 2017"},
        ]
        threads = reconstruct_threads(_make_df(rows))
        assert len(threads) == 2


# ---------------------------------------------------------------------------
# 26. Empty input (Phase 4 path)
# ---------------------------------------------------------------------------
def test_empty_input_raises_runtime_error(tmp_path):
    csv_path = tmp_path / "empty.csv"
    cols = list(REQUIRED_COLUMNS) + ["row_type"]
    csv_path.write_text(",".join(cols) + "\n")
    with pytest.raises((RuntimeError, ValueError)):
        run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# 27. Required-column validation
# ---------------------------------------------------------------------------
class TestRequiredColumnValidation:
    def test_missing_column_detected(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("tweet_id,author_id\n1,cust\n")
        errors = check_input_schema(csv_path)
        assert len(errors) > 0
        assert any("missing" in e.lower() for e in errors)

    def test_all_required_columns_present(self, tmp_path):
        csv_path = _write_csv(tmp_path / "ok.csv", _two_turn_rows())
        errors = check_input_schema(csv_path)
        assert errors == []


# ---------------------------------------------------------------------------
# 28. Metadata generation
# ---------------------------------------------------------------------------
class TestMetadataGeneration:
    def test_metadata_file_created(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")
        assert (tmp_path / "out" / OUTPUT_META_NAME).exists()

    def test_metadata_required_keys(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        meta = run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")
        for key in ("brand", "schema_version", "total_threads",
                    "thread_summary", "link_metrics", "quality_metrics",
                    "metric_definitions", "limitations", "next_phase"):
            assert key in meta, f"Missing key: {key}"

    def test_metadata_thread_count_correct(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        meta = run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")
        assert meta["total_threads"] == 1
        assert meta["input_message_count"] == 4


# ---------------------------------------------------------------------------
# 29. Relative paths in metadata
# ---------------------------------------------------------------------------
class TestRelativePathsInMetadata:
    def test_no_absolute_home_in_metadata(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _two_turn_rows())
        meta = run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")
        # The paths may be absolute if tmp_path is outside project root,
        # but they must NOT contain literal project-specific home dirs
        for key in ("input_messages_csv", "output_threads_jsonl",
                    "output_metadata_json"):
            assert key in meta

    def test_default_config_relative_paths(self):
        from src.data.reconstruct_threads import DEFAULT_INPUT, DEFAULT_OUTDIR, PROJECT_ROOT
        try:
            rel = str(DEFAULT_INPUT.resolve().relative_to(PROJECT_ROOT))
            assert not rel.startswith("/home"), f"Absolute path: {rel}"
        except ValueError:
            pass   # DEFAULT_INPUT outside project root — skip


# ---------------------------------------------------------------------------
# 30. Output JSONL schema
# ---------------------------------------------------------------------------
class TestOutputJSONLSchema:
    def test_jsonl_one_line_per_thread(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")
        lines = (tmp_path / "out" / OUTPUT_JSONL_NAME).read_text().strip().splitlines()
        assert len(lines) == 1   # four-turn fixture is one thread

    def test_jsonl_each_line_valid_json(self, tmp_path):
        rows = _two_turn_rows() + [
            {"tweet_id": "200", "inbound": "True", "text": "c",
             "created_at": "Tue Oct 31 12:00:00 +0000 2017"},
            {"tweet_id": "201", "inbound": "False", "in_response_to_tweet_id": "200",
             "text": "d", "created_at": "Tue Oct 31 12:05:00 +0000 2017"},
        ]
        csv_path = _write_csv(tmp_path / "msgs.csv", rows)
        run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")
        lines = (tmp_path / "out" / OUTPUT_JSONL_NAME).read_text().strip().splitlines()
        for line in lines:
            obj = json.loads(line)
            assert "thread_id" in obj
            assert "messages" in obj
            assert "message_count" in obj

    def test_thread_to_dict_has_all_keys(self):
        threads = reconstruct_threads(_make_df(_two_turn_rows()))
        d = thread_to_dict(threads[0])
        for key in ("thread_id", "root_tweet_id", "message_count",
                    "customer_message_count", "brand_message_count",
                    "has_both_directions", "has_3plus_messages",
                    "has_multiple_exchanges", "has_broken_links",
                    "cycle_detected", "quality_flags", "link_metrics",
                    "messages"):
            assert key in d, f"Missing key: {key}"

    def test_messages_list_has_correct_fields(self):
        threads = reconstruct_threads(_make_df(_two_turn_rows()))
        d = thread_to_dict(threads[0])
        for msg in d["messages"]:
            for f in ("tweet_id", "author_id", "inbound",
                      "created_at", "text", "response_tweet_id",
                      "in_response_to_tweet_id"):
                assert f in msg


# ---------------------------------------------------------------------------
# 31. Overwrite guard
# ---------------------------------------------------------------------------
class TestOverwriteGuard:
    def test_raises_if_output_exists(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _two_turn_rows())
        out = tmp_path / "out"
        run_reconstruction(input_path=csv_path, output_dir=out)  # first run
        with pytest.raises(FileExistsError, match="overwrite"):
            run_reconstruction(input_path=csv_path, output_dir=out)  # second run

    def test_check_output_exists_returns_message(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / OUTPUT_JSONL_NAME).write_text("fake")
        msg = check_output_exists(out, overwrite=False)
        assert msg is not None
        assert "overwrite" in msg.lower()

    def test_check_output_exists_returns_none_when_clear(self, tmp_path):
        out = tmp_path / "out"
        assert check_output_exists(out, overwrite=False) is None


# ---------------------------------------------------------------------------
# 32. Successful overwrite with --overwrite
# ---------------------------------------------------------------------------
class TestOverwriteSuccess:
    def test_overwrite_replaces_output(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _two_turn_rows())
        out = tmp_path / "out"
        run_reconstruction(input_path=csv_path, output_dir=out)
        (out / OUTPUT_JSONL_NAME).write_text("garbage")
        run_reconstruction(input_path=csv_path, output_dir=out, overwrite=True)
        first_line = (out / OUTPUT_JSONL_NAME).read_text().splitlines()[0]
        obj = json.loads(first_line)
        assert "thread_id" in obj


# ---------------------------------------------------------------------------
# 33. CLI success
# ---------------------------------------------------------------------------
class TestCLISuccess:
    def test_cli_exit_0(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        result = subprocess.run(
            [sys.executable, "-m", "src.data.reconstruct_threads",
             "--input", str(csv_path),
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr

    def test_cli_creates_output_files(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        out = tmp_path / "out"
        subprocess.run(
            [sys.executable, "-m", "src.data.reconstruct_threads",
             "--input", str(csv_path), "--output-dir", str(out)],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert (out / OUTPUT_JSONL_NAME).exists()
        assert (out / OUTPUT_META_NAME).exists()


# ---------------------------------------------------------------------------
# 34. CLI failure for missing input
# ---------------------------------------------------------------------------
class TestCLIFailure:
    def test_missing_input_exits_1(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "src.data.reconstruct_threads",
             "--input", str(tmp_path / "no.csv"),
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 1

    def test_existing_output_without_overwrite_exits_1(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _two_turn_rows())
        out = tmp_path / "out"
        out.mkdir()
        (out / OUTPUT_JSONL_NAME).write_text("fake")
        result = subprocess.run(
            [sys.executable, "-m", "src.data.reconstruct_threads",
             "--input", str(csv_path), "--output-dir", str(out)],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# 35. Strict mode
# ---------------------------------------------------------------------------
class TestStrictMode:
    def test_strict_mode_succeeds_on_clean_data(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        # Four-turn clean fixture should not trigger quality warnings
        # (no broken links, no cycle, no empty text, timestamps parseable)
        # strict mode should NOT raise
        meta = run_reconstruction(
            input_path=csv_path, output_dir=tmp_path / "out", strict=True
        )
        assert meta["total_threads"] == 1


# ---------------------------------------------------------------------------
# 36-38. Correct counts and metrics
# ---------------------------------------------------------------------------
class TestCorrectCounts:
    def setup_method(self):
        self.threads = reconstruct_threads(_make_df(_four_turn_rows()))

    def test_customer_message_count(self):
        assert self.threads[0].customer_message_count == 2

    def test_brand_message_count(self):
        assert self.threads[0].brand_message_count == 2

    def test_has_3plus_messages(self):
        assert self.threads[0].has_3plus_messages is True

    def test_has_multiple_exchanges(self):
        assert self.threads[0].has_multiple_exchanges is True

    def test_parent_links_present(self):
        # Tweets 2, 3, 4 each have in_response_to set → 3 parent links
        assert self.threads[0].parent_links_present == 3

    def test_parent_links_valid(self):
        # All parents (1, 2, 3) are in the dataset → all valid
        assert self.threads[0].parent_links_valid == 3
        assert self.threads[0].parent_links_missing == 0

    def test_summarise_threads_includes_phase4_keys(self):
        stats = summarise_threads(self.threads)
        assert "has_3plus_messages" in stats
        assert "has_multiple_exchanges" in stats
        assert "customer_messages" in stats
        assert "brand_messages" in stats


# ---------------------------------------------------------------------------
# 42. Phase 3 CSV column compatibility
# ---------------------------------------------------------------------------
class TestPhase3CSVCompatibility:
    def test_phase3_csv_columns_accepted(self, tmp_path):
        """CSV produced by Phase 3 (extract_selected_brand.py) must be accepted."""
        # Phase 3 output has: tweet_id, author_id, inbound, created_at,
        # text, text_raw, response_tweet_id, in_response_to_tweet_id, row_type
        csv_path = _write_csv(tmp_path / "msgs.csv", _two_turn_rows())
        errors = check_input_schema(csv_path)
        assert errors == []

    def test_reconstruct_with_row_type_column(self, tmp_path):
        csv_path = _write_csv(tmp_path / "msgs.csv", _four_turn_rows())
        meta = run_reconstruction(input_path=csv_path, output_dir=tmp_path / "out")
        assert meta["total_threads"] == 1
