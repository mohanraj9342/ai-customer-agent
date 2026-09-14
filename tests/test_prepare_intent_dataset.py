"""
tests/test_prepare_intent_dataset.py
====================================
Tests for src/classification/prepare_intent_dataset.py

All tests use small synthetic JSONL/dict fixtures so they execute in
milliseconds without depending on external large datasets.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from src.classification.prepare_intent_dataset import (
    OUTPUT_COLUMNS,
    OUTPUT_CSV_NAME,
    OUTPUT_META_NAME,
    TAXONOMY_VERSION,
    VALID_CONFIDENCES,
    VALID_INTENTS,
    VALID_LABEL_SOURCES,
    classify_customer_message,
    normalize_for_matching,
    process_thread_record,
    run_intent_preparation,
    select_first_customer_message,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _make_thread(
    thread_id: str = "thread_1",
    messages: list[dict] | None = None,
    **overrides,
) -> dict:
    if messages is None:
        messages = [
            {
                "tweet_id": "100",
                "author_id": "cust_1",
                "inbound": "True",
                "created_at": "Tue Oct 31 10:00:00 +0000 2017",
                "text": "my battery dies fast on iOS 11 update",
            },
            {
                "tweet_id": "101",
                "author_id": "AppleSupport",
                "inbound": "False",
                "created_at": "Tue Oct 31 10:05:00 +0000 2017",
                "text": "We can help! Try restarting your device.",
            },
        ]
    base = {
        "thread_id": thread_id,
        "root_tweet_id": messages[0]["tweet_id"] if messages else "",
        "message_count": len(messages),
        "customer_message_count": sum(1 for m in messages if m.get("inbound") == "True"),
        "brand_message_count": sum(1 for m in messages if m.get("inbound") == "False"),
        "start_timestamp": messages[0].get("created_at", "") if messages else "",
        "end_timestamp": messages[-1].get("created_at", "") if messages else "",
        "is_complete": True,
        "ends_with_brand_reply": True,
        "has_broken_links": False,
        "cycle_detected": False,
        "link_metrics": {"response_links_missing": 0},
        "messages": messages,
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# 1. Candidate Selection Tests
# ---------------------------------------------------------------------------

class TestCandidateSelection:
    def test_selects_earliest_customer_message(self):
        msgs = [
            {"tweet_id": "10", "inbound": "True", "text": "First problem"},
            {"tweet_id": "11", "inbound": "False", "text": "Brand reply"},
            {"tweet_id": "12", "inbound": "True", "text": "Second message"},
        ]
        sel = select_first_customer_message(msgs)
        assert sel.excluded is False
        assert sel.message["tweet_id"] == "10"
        assert sel.reason == "first_customer_message_chronological"

    def test_brand_message_never_selected(self):
        msgs = [
            {"tweet_id": "100", "author_id": "AppleSupport", "inbound": "False", "text": "Hello!"},
            {"tweet_id": "101", "author_id": "cust", "inbound": "True", "text": "My phone is broken"},
        ]
        sel = select_first_customer_message(msgs)
        assert sel.excluded is False
        assert sel.message["tweet_id"] == "101"
        assert sel.message["inbound"] == "True"

    def test_thread_without_customer_message_excluded(self):
        msgs = [
            {"tweet_id": "100", "inbound": "False", "text": "Proactive brand tweet"},
            {"tweet_id": "101", "inbound": "False", "text": "Second brand tweet"},
        ]
        sel = select_first_customer_message(msgs)
        assert sel.excluded is True
        assert sel.exclusion_reason == "no_customer_message"
        assert sel.message is None

    def test_empty_customer_message_fallback_to_next(self):
        msgs = [
            {"tweet_id": "10", "inbound": "True", "text": "   "},
            {"tweet_id": "11", "inbound": "True", "text": "Actual problem text"},
        ]
        sel = select_first_customer_message(msgs)
        assert sel.excluded is False
        assert sel.message["tweet_id"] == "11"
        assert "fallback" in sel.reason

    def test_all_customer_messages_empty_excluded(self):
        msgs = [
            {"tweet_id": "10", "inbound": "True", "text": ""},
            {"tweet_id": "11", "inbound": "True", "text": "   "},
        ]
        sel = select_first_customer_message(msgs)
        assert sel.excluded is True
        assert sel.exclusion_reason == "all_customer_messages_empty"


# ---------------------------------------------------------------------------
# 2. Text Normalization & Preservation
# ---------------------------------------------------------------------------

class TestTextNormalization:
    def test_original_text_preserved_in_candidate(self):
        raw = "  @AppleSupport My &amp; battery drops fast! https://t.co/xyz  "
        thread = _make_thread(
            messages=[
                {"tweet_id": "1", "inbound": "True", "text": raw, "created_at": "ts"},
            ]
        )
        cand, exc = process_thread_record(thread)
        assert exc is None
        # Original text is trimmed but not mutated
        assert cand["text"] == raw.strip()
        assert "&amp;" in cand["text"]
        assert "@AppleSupport" in cand["text"]

    def test_normalize_for_matching_strips_entities_and_mentions(self):
        raw = "@AppleSupport my iPhone &amp; screen https://t.co/link"
        norm = normalize_for_matching(raw)
        assert "@applesupport" not in norm
        assert "https://" not in norm
        assert "&amp;" not in norm
        assert "iphone & screen" in norm or "iphone screen" in norm


# ---------------------------------------------------------------------------
# 3. Rule Classification & Intent Determinism
# ---------------------------------------------------------------------------

class TestClassificationRules:
    def test_software_update_detection(self):
        res = classify_customer_message("my phone started freezing constantly since the iOS 11 update")
        assert res.intent == "software_update"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_battery_power_detection(self):
        res = classify_customer_message("my iPhone battery percentage drains from 90 to 20 in thirty minutes")
        assert res.intent == "battery_power"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_device_hardware_detection(self):
        res = classify_customer_message("dropped my iPhone 8 and the cracked screen touch is unresponsive")
        assert res.intent == "device_hardware"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_account_access_detection(self):
        res = classify_customer_message("my Apple ID account has been locked and I forgot password")
        assert res.intent == "account_access"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_connectivity_network_detection(self):
        res = classify_customer_message("my phone keeps disconnecting from wifi and Bluetooth won't pair")
        assert res.intent == "connectivity_network"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_billing_payment_detection(self):
        res = classify_customer_message("I was charged twice for an accidental subscription purchase refund please")
        assert res.intent == "billing_payment"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_order_shipping_detection(self):
        res = classify_customer_message("my online order tracking number says package delivered but nothing arrived")
        assert res.intent == "order_shipping"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_feature_how_to_detection(self):
        res = classify_customer_message("how do I turn off automatic app updates in the settings")
        assert res.intent == "feature_how_to"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_complaint_feedback_detection(self):
        res = classify_customer_message("worst phone ever and terrible customer service fix your shit")
        assert res.intent == "complaint_feedback"
        assert res.source == "rule"
        assert res.confidence == "high"

    def test_ambiguous_conflict_becomes_needs_review(self):
        # Mentions battery drain phrase AND cracked screen damage
        res = classify_customer_message(
            "battery drain problem and also my cracked screen display touch is broken"
        )
        assert res.intent == "needs_review"
        assert res.source == "needs_review"
        assert "conflicting_rules_matched" in res.reason

    def test_vague_short_message_becomes_unknown_other(self):
        res = classify_customer_message("@AppleSupport help me")
        assert res.intent == "unknown_other"
        assert res.source == "fallback"
        assert res.confidence == "low"

    def test_unmatched_text_becomes_unknown_other(self):
        res = classify_customer_message("The quick brown fox jumps over the lazy dog near Cupertino")
        assert res.intent == "unknown_other"
        assert res.source == "fallback"
        assert res.confidence == "low"


# ---------------------------------------------------------------------------
# 4. Leakage Protection Tests
# ---------------------------------------------------------------------------

class TestLeakageProtection:
    def test_brand_reply_text_never_influences_intent(self):
        # Customer message is generic, but brand reply talks about billing
        thread = _make_thread(
            messages=[
                {"tweet_id": "1", "inbound": "True", "text": "Hello I have a question about my account"},
                {"tweet_id": "2", "inbound": "False", "text": "We can help with your subscription refund and billing charge!"},
            ]
        )
        cand, exc = process_thread_record(thread)
        assert exc is None
        # Customer says 'account' → account_access, NOT billing_payment
        assert cand["candidate_intent"] == "account_access"

    def test_ends_with_brand_reply_is_not_treated_as_resolution(self):
        thread = _make_thread(
            ends_with_brand_reply=True,
            is_complete=True,
            messages=[
                {"tweet_id": "1", "inbound": "True", "text": "my battery drain is terrible"},
            ]
        )
        cand, exc = process_thread_record(thread)
        assert exc is None
        assert cand["candidate_intent"] == "battery_power"
        # ends_with_brand_reply is preserved as boolean context, not intent
        assert cand["ends_with_brand_reply"] is True


# ---------------------------------------------------------------------------
# 5. Full Pipeline & Metadata Reconciliation Tests
# ---------------------------------------------------------------------------

class TestPipelineExecution:
    def test_run_intent_preparation_end_to_end(self, tmp_path):
        threads = [
            _make_thread("t1", [{"tweet_id": "1", "inbound": "True", "text": "iOS 11 update issue"}]),
            _make_thread("t2", [{"tweet_id": "2", "inbound": "True", "text": "battery dies fast"}]),
            _make_thread("t3", [{"tweet_id": "3", "inbound": "False", "text": "brand only tweet"}]), # excluded
            _make_thread("t4", [{"tweet_id": "4", "inbound": "True", "text": "tracking number delivery delay"}]),
        ]
        in_file = _write_jsonl(tmp_path / "threads.jsonl", threads)
        out_dir = tmp_path / "out"

        meta = run_intent_preparation(input_path=in_file, output_dir=out_dir)

        assert meta["total_threads_scanned"] == 4
        assert meta["eligible_threads_count"] == 3
        assert meta["excluded_threads_count"] == 1
        assert meta["candidate_messages_count"] == 3
        assert meta["taxonomy_version"] == TAXONOMY_VERSION

        # Verify CSV output
        csv_path = out_dir / OUTPUT_CSV_NAME
        assert csv_path.exists()

        rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        assert len(rows) == 3
        # Check all required columns
        for col in OUTPUT_COLUMNS:
            assert col in rows[0]

        # Verify reconciliation
        sum_intents = sum(meta["counts_by_candidate_intent"].values())
        sum_sources = sum(meta["counts_by_label_source"].values())
        sum_confs = sum(meta["counts_by_confidence"].values())

        assert sum_intents == len(rows)
        assert sum_sources == len(rows)
        assert sum_confs == len(rows)

    def test_overwrite_guard(self, tmp_path):
        threads = [_make_thread("t1")]
        in_file = _write_jsonl(tmp_path / "threads.jsonl", threads)
        out_dir = tmp_path / "out"

        run_intent_preparation(input_path=in_file, output_dir=out_dir)

        # Running again without overwrite should fail
        with pytest.raises(FileExistsError):
            run_intent_preparation(input_path=in_file, output_dir=out_dir, overwrite=False)

        # Running with overwrite should succeed
        meta2 = run_intent_preparation(input_path=in_file, output_dir=out_dir, overwrite=True)
        assert meta2["candidate_messages_count"] == 1

    def test_missing_input_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_intent_preparation(
                input_path=tmp_path / "nonexistent.jsonl",
                output_dir=tmp_path / "out",
            )
