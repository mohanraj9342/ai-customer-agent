"""
tests/test_extract_brand.py
============================
Tests for src/data/extract_brand.py — text normalisation.

These tests use only synthetic in-memory data and do NOT touch the raw dataset.
They are fast (< 1 second) and run without any external files.
"""

import pytest
from src.data.extract_brand import normalise_text


class TestNormaliseText:
    """Tests for the normalise_text() helper."""

    def test_html_amp_decoded(self):
        assert normalise_text("cats &amp; dogs") == "cats & dogs"

    def test_html_lt_gt_decoded(self):
        assert normalise_text("a &lt; b &gt; c") == "a < b > c"

    def test_html_quot_decoded(self):
        assert normalise_text("she said &quot;hello&quot;") == 'she said "hello"'

    def test_html_apos_decoded(self):
        assert normalise_text("it&#39;s fine") == "it's fine"

    def test_whitespace_collapsed(self):
        result = normalise_text("  too   many    spaces  ")
        assert result == "too many spaces"

    def test_embedded_newline_collapsed(self):
        # Tweets may contain literal newlines inside quoted CSV fields
        result = normalise_text("line one\nline two")
        assert result == "line one line two"

    def test_empty_string_returns_empty(self):
        assert normalise_text("") == ""

    def test_none_returns_empty(self):
        # normalise_text receives non-string values from NaN cells
        assert normalise_text(None) == ""  # type: ignore[arg-type]

    def test_mentions_preserved(self):
        # @mentions must NOT be removed — used for conversation heuristics
        result = normalise_text("@AppleSupport my battery drains fast")
        assert "@AppleSupport" in result

    def test_emoji_preserved(self):
        result = normalise_text("it works now 😊")
        assert "😊" in result

    def test_url_preserved(self):
        result = normalise_text("see https://t.co/abc123 for info")
        assert "https://t.co/abc123" in result

    def test_multiple_entities_in_one_tweet(self):
        result = normalise_text("Q &amp; A: &lt;5 items&gt; costs &quot;free&quot;")
        assert result == 'Q & A: <5 items> costs "free"'
