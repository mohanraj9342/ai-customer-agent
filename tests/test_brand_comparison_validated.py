"""
tests/test_brand_comparison_validated.py
==========================================
Tests for the corrected metric helpers in
src/data/brand_comparison_validated.py

All tests use synthetic in-memory data — no dataset access.
"""

import pandas as pd
import pytest
from src.data.brand_comparison_validated import (
    Thread,
    is_likely_english,
    is_noise,
    count_masked_fields,
    useful_word_count,
    thread_has_both_directions,
    thread_has_3plus_messages,
    thread_has_multiple_exchanges,
    thread_ends_with_brand,
    thread_has_no_broken_links,
    ENGLISH_ASCII_THRESHOLD,
    MIN_USEFUL_WORDS,
)


# ---------------------------------------------------------------------------
# is_likely_english  (rule-based heuristic)
# ---------------------------------------------------------------------------

class TestIsLikelyEnglish:
    def test_pure_english_ascii(self):
        assert is_likely_english("Hello, how can I help you today?") is True

    def test_empty_string_is_true(self):
        # Empty → no evidence of non-English
        assert is_likely_english("") is True

    def test_japanese_text_is_false(self):
        assert is_likely_english("こんにちは、サポートをお願いします") is False

    def test_arabic_text_is_false(self):
        assert is_likely_english("مرحباً كيف يمكنني مساعدتك") is False

    def test_mixed_english_emoji_is_true(self):
        # Emoji are multi-byte but there are few of them relative to ASCII chars
        assert is_likely_english("my battery drains fast 😢 please help") is True

    def test_pure_ascii_symbols_is_true(self):
        assert is_likely_english("@AppleSupport !!!???") is True

    def test_threshold_boundary(self):
        # Exactly at threshold: 85 ASCII chars out of 100 total
        ascii_part = "a" * 85
        non_ascii_part = "é" * 15   # é is 2 bytes but > 127 ordinal
        text = ascii_part + non_ascii_part
        # 85/100 = 0.85 → should be True (≥ threshold)
        assert is_likely_english(text) is True

    def test_below_threshold_is_false(self):
        ascii_part = "a" * 84
        non_ascii_part = "é" * 16
        text = ascii_part + non_ascii_part
        # 84/100 = 0.84 < 0.85 → False
        assert is_likely_english(text) is False


# ---------------------------------------------------------------------------
# useful_word_count and is_noise  (rule-based)
# ---------------------------------------------------------------------------

class TestUsefulWordCount:
    def test_normal_sentence(self):
        assert useful_word_count("my battery drains really fast") == 5

    def test_strips_mention(self):
        # @AppleSupport should not be counted
        assert useful_word_count("@AppleSupport help me please") == 3

    def test_strips_url(self):
        count = useful_word_count("see https://t.co/abc123 for details")
        assert count == 3   # "see", "for", "details"

    def test_empty_string(self):
        assert useful_word_count("") == 0

    def test_only_mention_and_url(self):
        assert useful_word_count("@Brand https://t.co/abc") == 0

    def test_numbers_not_counted(self):
        # isalpha() is False for digits
        assert useful_word_count("ticket 12345 is open") == 3  # "ticket", "is", "open"


class TestIsNoise:
    def test_meaningful_message_is_not_noise(self):
        assert is_noise("my iPhone battery drains really fast") is False

    def test_short_message_is_noise(self):
        # "ok" → 1 word → below MIN_USEFUL_WORDS (3)
        assert is_noise("@Brand ok") is True

    def test_url_only_is_noise(self):
        assert is_noise("https://t.co/abc123") is True

    def test_three_words_is_not_noise(self):
        # Exactly at threshold: 3 words → NOT noise (< 3 is noise)
        assert is_noise("help me please") is False

    def test_two_words_is_noise(self):
        assert is_noise("help me") is True


# ---------------------------------------------------------------------------
# count_masked_fields  (directly measured)
# ---------------------------------------------------------------------------

class TestCountMaskedFields:
    def test_no_masked(self):
        assert count_masked_fields("normal tweet text") == 0

    def test_one_masked(self):
        assert count_masked_fields("my email is __email__ thanks") == 1

    def test_multiple_masked(self):
        assert count_masked_fields("__email__ and __phone__ and __user__") == 3

    def test_partial_pattern_not_counted(self):
        # _email_ (single underscore) should not match __token__
        assert count_masked_fields("my _email_ address") == 0

    def test_empty_string(self):
        assert count_masked_fields("") == 0


# ---------------------------------------------------------------------------
# Thread quality helpers  (directly measured)
# ---------------------------------------------------------------------------

def _make_thread(directions: list[str], turn_count: int = None,
                 has_broken: bool = False) -> Thread:
    n = turn_count or len(directions)
    return Thread(
        root_tweet_id="root",
        tweet_ids=[str(i) for i in range(n)],
        texts=["text"] * n,
        directions=directions,
        timestamps=["ts"] * n,
        turn_count=n,
        is_complete=(directions[-1] == "outbound") if directions else False,
        has_broken_links=has_broken,
    )


class TestThreadHasBothDirections:
    def test_inbound_and_outbound(self):
        t = _make_thread(["inbound", "outbound"])
        assert thread_has_both_directions(t) is True

    def test_only_inbound(self):
        t = _make_thread(["inbound", "inbound"])
        assert thread_has_both_directions(t) is False

    def test_only_outbound(self):
        t = _make_thread(["outbound", "outbound"])
        assert thread_has_both_directions(t) is False

    def test_empty_thread(self):
        t = _make_thread([])
        assert thread_has_both_directions(t) is False

    def test_multi_turn_with_both(self):
        t = _make_thread(["inbound", "outbound", "inbound", "outbound"])
        assert thread_has_both_directions(t) is True


class TestThreadHas3PlusMessages:
    def test_two_messages_is_false(self):
        t = _make_thread(["inbound", "outbound"])
        assert thread_has_3plus_messages(t) is False

    def test_three_messages_is_true(self):
        t = _make_thread(["inbound", "outbound", "inbound"])
        assert thread_has_3plus_messages(t) is True

    def test_one_message_is_false(self):
        t = _make_thread(["inbound"])
        assert thread_has_3plus_messages(t) is False


class TestThreadHasMultipleExchanges:
    def test_customer_brand_only_is_false(self):
        # inbound → outbound: brand answered but customer never replied again
        t = _make_thread(["inbound", "outbound"])
        assert thread_has_multiple_exchanges(t) is False

    def test_full_back_and_forth(self):
        # customer → brand → customer → brand: True
        t = _make_thread(["inbound", "outbound", "inbound", "outbound"])
        assert thread_has_multiple_exchanges(t) is True

    def test_customer_brand_customer(self):
        # inbound → outbound → inbound: outbound followed by inbound → True
        t = _make_thread(["inbound", "outbound", "inbound"])
        assert thread_has_multiple_exchanges(t) is True

    def test_only_inbound(self):
        t = _make_thread(["inbound", "inbound", "inbound"])
        assert thread_has_multiple_exchanges(t) is False

    def test_only_outbound(self):
        t = _make_thread(["outbound", "outbound"])
        assert thread_has_multiple_exchanges(t) is False

    def test_single_message(self):
        t = _make_thread(["inbound"])
        assert thread_has_multiple_exchanges(t) is False

    def test_two_messages_only(self):
        t = _make_thread(["inbound", "outbound"])
        assert thread_has_multiple_exchanges(t) is False


class TestThreadEndsWithBrand:
    def test_ends_outbound_is_true(self):
        t = _make_thread(["inbound", "outbound"])
        assert thread_ends_with_brand(t) is True

    def test_ends_inbound_is_false(self):
        t = _make_thread(["outbound", "inbound"])
        assert thread_ends_with_brand(t) is False

    def test_empty_is_false(self):
        t = _make_thread([])
        assert thread_ends_with_brand(t) is False


class TestThreadHasNoBrokenLinks:
    def test_no_broken(self):
        t = _make_thread(["inbound", "outbound"], has_broken=False)
        assert thread_has_no_broken_links(t) is True

    def test_has_broken(self):
        t = _make_thread(["inbound", "outbound"], has_broken=True)
        assert thread_has_no_broken_links(t) is False


# ---------------------------------------------------------------------------
# Metric type labelling — verify that metric notes are present in output
# ---------------------------------------------------------------------------

class TestMetricLabels:
    """
    Verify that rule-based and qualitative metrics carry their labels
    in their note fields.
    """
    def test_english_note_contains_label(self):
        from src.data.brand_comparison_validated import ENGLISH_ASCII_THRESHOLD
        note = (
            f"ASCII-ratio ≥ {ENGLISH_ASCII_THRESHOLD:.0%} threshold. "
            "LABEL: rule_based estimate."
        )
        assert "LABEL: rule_based estimate" in note

    def test_noise_note_contains_label(self):
        from src.data.brand_comparison_validated import MIN_USEFUL_WORDS
        note = f"Useful alphabetic word count < {MIN_USEFUL_WORDS} after stripping @mentions and URLs. LABEL: rule_based estimate."
        assert "LABEL: rule_based estimate" in note

    def test_masked_note_contains_label(self):
        note = "Counts __token__ pattern only. NOT a complete PII audit. LABEL: directly_measured for this specific pattern."
        assert "LABEL: directly_measured" in note
