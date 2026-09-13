"""
tests/test_reconstruct_threads.py
===================================
Tests for src/data/reconstruct_threads.py

All tests use small synthetic DataFrames so they run in milliseconds and
require no external files.

Test coverage:
  1. Normal two-turn conversation (customer → brand)
  2. Multi-turn conversation (3+ messages)
  3. Missing parent (parent tweet not in dataset)
  4. Missing child / response (leaf tweet with no recorded reply)
  5. Isolated single message (conversation root with no children)
  6. Malformed / cyclic relationship (cycle in parent links)
  7. Chronological ordering within a thread
  8. summarise_threads() statistics
"""

import pandas as pd
import pytest

from src.data.reconstruct_threads import (
    Thread,
    reconstruct_threads,
    summarise_threads,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Create a typed DataFrame from a list of row dicts."""
    defaults = {
        "tweet_id": "",
        "author_id": "",
        "inbound": "False",
        "created_at": "Tue Oct 31 12:00:00 +0000 2017",
        "text": "",
        "response_tweet_id": "",
        "in_response_to_tweet_id": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows], dtype=str)


# ---------------------------------------------------------------------------
# Test 1: Normal two-turn conversation
# ---------------------------------------------------------------------------

class TestTwoTurnConversation:
    def setup_method(self):
        self.df = _make_df([
            {
                "tweet_id": "100",
                "inbound": "True",
                "text": "my battery dies fast",
                "created_at": "Tue Oct 31 10:00:00 +0000 2017",
                "in_response_to_tweet_id": "",
                "response_tweet_id": "101",
            },
            {
                "tweet_id": "101",
                "inbound": "False",
                "text": "We can help! Which device?",
                "created_at": "Tue Oct 31 11:00:00 +0000 2017",
                "in_response_to_tweet_id": "100",
                "response_tweet_id": "",
            },
        ])
        self.threads = reconstruct_threads(self.df)

    def test_one_thread_produced(self):
        assert len(self.threads) == 1

    def test_thread_has_two_turns(self):
        assert self.threads[0].turn_count == 2

    def test_thread_is_complete(self):
        # Last message is outbound → complete
        assert self.threads[0].is_complete is True

    def test_no_broken_links(self):
        assert self.threads[0].has_broken_links is False

    def test_directions_correct(self):
        t = self.threads[0]
        assert t.directions == ["inbound", "outbound"]


# ---------------------------------------------------------------------------
# Test 2: Multi-turn conversation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Test 3: Missing parent (parent tweet not in dataset)
# ---------------------------------------------------------------------------

class TestMissingParent:
    def setup_method(self):
        # Tweet 200 claims its parent is tweet 199, but 199 is NOT in our subset
        self.df = _make_df([
            {"tweet_id": "200", "inbound": "True", "in_response_to_tweet_id": "199",
             "text": "still not working", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "201", "inbound": "False", "in_response_to_tweet_id": "200",
             "text": "Let us help!", "created_at": "Tue Oct 31 10:05:00 +0000 2017"},
        ])
        self.threads = reconstruct_threads(self.df)

    def test_one_thread(self):
        # Tweet 200 becomes the root because its parent (199) is absent
        assert len(self.threads) == 1

    def test_broken_link_flagged(self):
        # The missing parent should be detected
        assert self.threads[0].has_broken_links is True

    def test_two_turns_still_reconstructed(self):
        assert self.threads[0].turn_count == 2


# ---------------------------------------------------------------------------
# Test 4: Isolated single message
# ---------------------------------------------------------------------------

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
        # Single inbound message, no brand reply → incomplete
        assert self.threads[0].is_complete is False


# ---------------------------------------------------------------------------
# Test 5: Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def setup_method(self):
        # Create a cycle: A → B → A (malformed data)
        self.df = _make_df([
            {"tweet_id": "400", "inbound": "True",
             "in_response_to_tweet_id": "401", "response_tweet_id": "401",
             "text": "tweet A", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"tweet_id": "401", "inbound": "False",
             "in_response_to_tweet_id": "400", "response_tweet_id": "400",
             "text": "tweet B", "created_at": "Tue Oct 31 10:01:00 +0000 2017"},
        ])

    def test_does_not_hang(self):
        # Most importantly: the function must not loop forever
        threads = reconstruct_threads(self.df)
        # Both tweets should be found, one per thread starting at each root
        total_turns = sum(t.turn_count for t in threads)
        assert total_turns >= 1

    def test_no_infinite_loop(self):
        # If we get here the test passed (no hang)
        reconstruct_threads(self.df)


# ---------------------------------------------------------------------------
# Test 6: Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_list():
    df = pd.DataFrame(columns=[
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id",
    ], dtype=str)
    threads = reconstruct_threads(df)
    assert threads == []


# ---------------------------------------------------------------------------
# Test 7: summarise_threads statistics
# ---------------------------------------------------------------------------

class TestSummariseThreads:
    def setup_method(self):
        # Build a mix: one 2-turn complete, one 1-turn incomplete
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
        threads_two = reconstruct_threads(df_two)
        threads_one = reconstruct_threads(df_one)
        self.threads = threads_two + threads_one

    def test_total_count(self):
        stats = summarise_threads(self.threads)
        assert stats["total_threads"] == 2

    def test_single_turn_count(self):
        stats = summarise_threads(self.threads)
        assert stats["single_turn"] == 1

    def test_multi_turn_count(self):
        stats = summarise_threads(self.threads)
        assert stats["multi_turn"] == 1

    def test_complete_count(self):
        stats = summarise_threads(self.threads)
        assert stats["complete"] == 1

    def test_incomplete_count(self):
        stats = summarise_threads(self.threads)
        assert stats["incomplete"] == 1

    def test_avg_turns(self):
        stats = summarise_threads(self.threads)
        # (2 + 1) / 2 = 1.5
        assert stats["avg_turns"] == 1.5


def test_summarise_empty_list():
    stats = summarise_threads([])
    assert stats["total_threads"] == 0
    assert stats["avg_turns"] == 0.0
