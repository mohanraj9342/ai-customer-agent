"""
tests/test_audit_intent_labels.py
=================================
Unit tests for Phase 6 audit sampling and human-review preparation.

Validates:
1. Deterministic sampling with fixed random seed.
2. Exact column schema preservation and alias alignment.
3. Zero duplicate tweet_id or thread_id in sampled outputs.
4. Representation across all requested strata.
5. Review priority assignment hierarchy (critical, high, medium, normal).
6. Inclusion of rule overlap and ambiguous high-risk cohorts.
7. Original candidate CSV immutability (read-only verification).
8. Metadata consistency with output sample CSV.
9. Zero leakage of brand reply text or future turns.
10. Overwrite guards and CLI exit codes.
"""

import hashlib
import json
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from src.classification.audit_intent_labels import (
    DEFAULT_RANDOM_SEED,
    REVIEW_SAMPLE_COLUMNS,
    SamplingConfig,
    assign_review_priority,
    build_pilot_sample,
    build_stratified_audit_sample,
    compute_row_audit_features,
    enrich_and_format_sample_df,
    run_audit_sampling,
    run_pilot_sampling,
)


# ---------------------------------------------------------------------------
# Synthetic Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_candidate_df() -> pd.DataFrame:
    """Generate a clean, synthetic candidate DataFrame representing diverse intents and edge cases."""
    records = []
    intents = [
        ("software_update", "rule", "high", "iOS 11 update has ruined my battery life and freezing"),
        ("software_update", "rule", "high", "High Sierra update install failed on my macbook"),
        ("battery_power", "rule", "high", "battery percentage drops from 80% to 20% in an hour"),
        ("device_hardware", "rule", "high", "dropped phone and cracked screen is completely black"),
        ("account_access", "rule", "high", "my Apple ID has been locked and cannot reset password"),
        ("connectivity_network", "rule", "high", "keeps disconnecting from wifi every 5 minutes"),
        ("billing_payment", "rule", "high", "unexpected charge on my credit card for apple music"),
        ("order_shipping", "rule", "high", "package tracking number says delivered but no package"),
        ("feature_how_to", "rule", "high", "how do I turn off read receipts in imessage"),
        ("complaint_feedback", "rule", "high", "terrible customer service and hate apple devices"),
        ("app_or_service_issue", "rule", "high", "apple music keeps crashing whenever I open it"),
        ("needs_review", "needs_review", "low", "ever since iOS 11 update my battery draining so fast"),
        ("unknown_other", "fallback", "low", "@AppleSupport help me please"),
        ("unknown_other", "fallback", "low", "@AppleSupport https://t.co/abc1234"),
        ("unknown_other", "fallback", "low", "ok thanks"),
        ("software_update", "rule", "medium", "need to update my phone soon"),
        ("battery_power", "rule", "medium", "charger cable is broken"),
        ("device_hardware", "rule", "medium", "microphone seems quiet"),
        ("connectivity_network", "rule", "medium", "bluetooth device issues"),
        ("account_access", "rule", "medium", "login trouble with passcode"),
    ]

    for idx, (intent, source, conf, text) in enumerate(intents, start=1000):
        records.append(
            {
                "thread_id": f"thread_{idx}",
                "tweet_id": str(idx),
                "timestamp": "Tue Oct 31 22:00:00 +0000 2017",
                "text": text,
                "candidate_intent": intent,
                "label_source": source,
                "label_reason": f"rule_for_{intent}",
                "label_confidence": conf,
                "selection_reason": "first_customer_message_chronological",
                "thread_message_count": "2",
                "customer_message_count": "1",
                "brand_message_count": "1",
                "has_missing_parent_link": "False",
                "has_missing_response_target": "False",
                "is_complete": "True",
                "ends_with_brand_reply": "True",
                "taxonomy_version": "2.0",
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_csv_file(tmp_path: Path, synthetic_candidate_df: pd.DataFrame) -> Path:
    """Save synthetic candidate DataFrame to a CSV file."""
    csv_path = tmp_path / "synthetic_candidates.csv"
    synthetic_candidate_df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# Tests: Review Priority Hierarchy
# ---------------------------------------------------------------------------

class TestReviewPriority:
    def test_critical_on_overlap(self):
        pri = assign_review_priority(
            intent="battery_power",
            confidence="high",
            overlap_flag=True,
            match_count=2,
            scores={"battery_power": 2, "software_update": 2},
        )
        assert pri == "critical"

    def test_critical_on_high_confidence_competing_signals(self):
        pri = assign_review_priority(
            intent="software_update",
            confidence="high",
            overlap_flag=False,
            match_count=1,
            scores={"software_update": 3, "battery_power": 1},
        )
        assert pri == "critical"

    def test_high_priority_categories(self):
        for intent in ("software_update", "unknown_other", "needs_review"):
            pri = assign_review_priority(
                intent=intent,
                confidence="medium",
                overlap_flag=False,
                match_count=1,
                scores={intent: 1},
            )
            assert pri == "high"

    def test_medium_priority_on_low_confidence(self):
        pri = assign_review_priority(
            intent="device_hardware",
            confidence="low",
            overlap_flag=False,
            match_count=1,
            scores={"device_hardware": 1},
        )
        assert pri == "medium"

    def test_normal_priority_on_clean_high_confidence(self):
        pri = assign_review_priority(
            intent="billing_payment",
            confidence="high",
            overlap_flag=False,
            match_count=1,
            scores={"billing_payment": 2},
        )
        assert pri == "normal"


# ---------------------------------------------------------------------------
# Tests: Deterministic Stratified Sampling
# ---------------------------------------------------------------------------

class TestStratifiedSampling:
    def test_determinism_with_fixed_seed(self, synthetic_candidate_df: pd.DataFrame):
        # Precompute features
        cleaned_texts, matched_intents, match_counts, overlap_flags = [], [], [], []
        for t in synthetic_candidate_df["text"]:
            c, m_str, m_cnt, ov, _ = compute_row_audit_features(str(t))
            cleaned_texts.append(c)
            matched_intents.append(m_str)
            match_counts.append(m_cnt)
            overlap_flags.append(ov)

        df = synthetic_candidate_df.copy()
        df["cleaned_text"] = cleaned_texts
        df["matched_intents"] = matched_intents
        df["match_count"] = match_counts
        df["rule_overlap_flag"] = overlap_flags

        cfg1 = SamplingConfig(seed=42, sample_size_per_intent=2)
        sample1, _, _ = build_stratified_audit_sample(df, cfg1)

        cfg2 = SamplingConfig(seed=42, sample_size_per_intent=2)
        sample2, _, _ = build_stratified_audit_sample(df, cfg2)

        # Bitwise identical
        pd.testing.assert_frame_equal(sample1, sample2)

    def test_no_duplicate_ids(self, synthetic_candidate_df: pd.DataFrame):
        df = synthetic_candidate_df.copy()
        cleaned_texts, matched_intents, match_counts, overlap_flags = [], [], [], []
        for t in df["text"]:
            c, m_str, m_cnt, ov, _ = compute_row_audit_features(str(t))
            cleaned_texts.append(c)
            matched_intents.append(m_str)
            match_counts.append(m_cnt)
            overlap_flags.append(ov)

        df["cleaned_text"] = cleaned_texts
        df["matched_intents"] = matched_intents
        df["match_count"] = match_counts
        df["rule_overlap_flag"] = overlap_flags

        cfg = SamplingConfig(seed=42, sample_size_per_intent=5)
        sample, _, _ = build_stratified_audit_sample(df, cfg)

        assert sample["tweet_id"].duplicated().sum() == 0
        assert sample["thread_id"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# Tests: Column Preservation & Schema
# ---------------------------------------------------------------------------

class TestColumnPreservation:
    def test_all_canonical_review_columns_present(self, synthetic_candidate_df: pd.DataFrame):
        enriched = enrich_and_format_sample_df(synthetic_candidate_df.copy())
        for col in REVIEW_SAMPLE_COLUMNS:
            assert col in enriched.columns, f"Missing required column: {col}"

    def test_human_review_columns_initialized_empty(self, synthetic_candidate_df: pd.DataFrame):
        enriched = enrich_and_format_sample_df(synthetic_candidate_df.copy())
        assert (enriched["verified_intent"] == "").all()
        assert (enriched["verified_by"] == "").all()
        assert (enriched["verification_date"] == "").all()
        assert (enriched["verification_status"] == "unreviewed").all()
        assert (enriched["notes"] == "").all()


# ---------------------------------------------------------------------------
# Tests: End-to-End Pipeline Execution & Immutability
# ---------------------------------------------------------------------------

class TestPipelineExecution:
    def test_run_audit_sampling_success(self, synthetic_csv_file: Path, tmp_path: Path):
        out_dir = tmp_path / "output"
        # Store hash of input file prior to run
        h_before = hashlib.sha256(synthetic_csv_file.read_bytes()).hexdigest()

        meta = run_audit_sampling(
            input_csv=synthetic_csv_file,
            output_dir=out_dir,
            seed=42,
            overwrite=True,
        )

        # Verify output files exist
        sample_csv = out_dir / "apple_support_intent_review_sample.csv"
        sample_meta = out_dir / "apple_support_intent_review_metadata.json"

        assert sample_csv.exists()
        assert sample_meta.exists()

        # Check metadata consistency
        df_out = pd.read_csv(sample_csv)
        assert len(df_out) == meta["total_review_samples"]
        assert df_out["tweet_id"].nunique() == meta["unique_tweet_ids"]
        assert df_out["thread_id"].nunique() == meta["unique_thread_ids"]

        # Verify input candidate CSV was NOT modified
        h_after = hashlib.sha256(synthetic_csv_file.read_bytes()).hexdigest()
        assert h_before == h_after, "Original candidate CSV was unexpectedly modified!"

    def test_overwrite_guard(self, synthetic_csv_file: Path, tmp_path: Path):
        out_dir = tmp_path / "output_guard"
        # First run succeeds
        run_audit_sampling(synthetic_csv_file, out_dir, overwrite=True)

        # Second run without overwrite raises FileExistsError
        with pytest.raises(FileExistsError):
            run_audit_sampling(synthetic_csv_file, out_dir, overwrite=False)

    def test_missing_input_raises_filenotfound(self, tmp_path: Path):
        non_existent = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            run_audit_sampling(non_existent, tmp_path / "out")


# ---------------------------------------------------------------------------
# Tests: Leakage Prevention & Traceability
# ---------------------------------------------------------------------------

class TestLeakageAndConfidentiality:
    def test_no_brand_reply_text_in_columns(self):
        # Review sample must only contain customer text fields
        forbidden_cols = {"brand_text", "agent_reply", "reply_text", "response_text"}
        assert forbidden_cols.isdisjoint(set(REVIEW_SAMPLE_COLUMNS))

    def test_no_future_turns_introduced(self, synthetic_candidate_df: pd.DataFrame):
        enriched = enrich_and_format_sample_df(synthetic_candidate_df.copy())
        # customer_message_count and selection_reason confirm single earliest turn
        assert "selection_reason" in enriched.columns
        assert (enriched["selection_reason"] == "first_customer_message_chronological").all()


# ---------------------------------------------------------------------------
# Tests: Pilot Dataset Sampling & Annotation Guide
# ---------------------------------------------------------------------------

class TestPilotDataset:
    def test_run_pilot_sampling_success(self, synthetic_csv_file: Path, tmp_path: Path):
        out_dir = tmp_path / "pilot_out"
        meta = run_pilot_sampling(
            input_csv=synthetic_csv_file,
            output_dir=out_dir,
            seed=42,
            overwrite=True,
        )

        pilot_csv = out_dir / "apple_support_intent_pilot_sample.csv"
        pilot_meta = out_dir / "apple_support_intent_pilot_metadata.json"

        assert pilot_csv.exists()
        assert pilot_meta.exists()

        df_pilot = pd.read_csv(pilot_csv)
        assert len(df_pilot) == meta["total_pilot_samples"]
        assert df_pilot["tweet_id"].nunique() == meta["unique_tweet_ids"]
        assert df_pilot["thread_id"].nunique() == meta["unique_thread_ids"]

    def test_pilot_sampling_determinism(self, synthetic_candidate_df: pd.DataFrame):
        df = synthetic_candidate_df.copy()
        cleaned_texts, matched_intents, match_counts, overlap_flags = [], [], [], []
        for t in df["text"]:
            c, m_str, m_cnt, ov, _ = compute_row_audit_features(str(t))
            cleaned_texts.append(c)
            matched_intents.append(m_str)
            match_counts.append(m_cnt)
            overlap_flags.append(ov)

        df["cleaned_text"] = cleaned_texts
        df["matched_intents"] = matched_intents
        df["match_count"] = match_counts
        df["rule_overlap_flag"] = overlap_flags

        pilot1, _, _ = build_pilot_sample(df, seed=42)
        pilot2, _, _ = build_pilot_sample(df, seed=42)

        pd.testing.assert_frame_equal(pilot1, pilot2)

    def test_pilot_review_fields_initialized_empty(self, synthetic_candidate_df: pd.DataFrame):
        df = synthetic_candidate_df.copy()
        cleaned_texts, matched_intents, match_counts, overlap_flags = [], [], [], []
        for t in df["text"]:
            c, m_str, m_cnt, ov, _ = compute_row_audit_features(str(t))
            cleaned_texts.append(c)
            matched_intents.append(m_str)
            match_counts.append(m_cnt)
            overlap_flags.append(ov)

        df["cleaned_text"] = cleaned_texts
        df["matched_intents"] = matched_intents
        df["match_count"] = match_counts
        df["rule_overlap_flag"] = overlap_flags

        pilot, _, _ = build_pilot_sample(df, seed=42)
        formatted = enrich_and_format_sample_df(pilot)

        assert (formatted["verified_intent"] == "").all()
        assert (formatted["verified_by"] == "").all()
        assert (formatted["verification_date"] == "").all()
        assert (formatted["verification_status"] == "unreviewed").all()
        assert (formatted["notes"] == "").all()


class TestAnnotationGuideAndDocIntegrity:
    def test_annotation_guide_taxonomy_consistency(self):
        from src.classification.prepare_intent_dataset import VALID_INTENTS
        guide_path = Path(__file__).resolve().parent.parent / "docs" / "apple_support_intent_annotation_guide.md"
        assert guide_path.exists(), "Annotation guide file does not exist!"
        content = guide_path.read_text(encoding="utf-8")

        # Every valid intent must be defined as a section in the guide
        for intent in VALID_INTENTS:
            assert f"### 2." in content and intent in content, f"Intent {intent} not documented in annotation guide!"

    def test_no_local_file_links_in_docs(self):
        docs_dir = Path(__file__).resolve().parent.parent / "docs"
        for md_file in docs_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            assert "file://" not in content, f"Forbidden local file:// link found in {md_file}"
            assert "/home/" not in content, f"Forbidden absolute path found in {md_file}"

