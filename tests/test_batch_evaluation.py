"""
tests/test_batch_evaluation.py
==============================
Phase 14 — End-to-End Batch Evaluation & Benchmarking Test Suite.

Verifies:
1. Benchmark dataset structure, schema columns, and 100-case count.
2. Strict mathematical isolation: zero overlap with Golden Evaluation Set.
3. Deterministic offline replay generator behavior and strict schema output.
4. EndToEndBatchEvaluator execution across pipeline stages.
5. Correctness of mathematical metrics computation (intent, retrieval, escalation, quality).
6. Strictly positive empirical latencies measured via time.perf_counter().
7. Zero secret leakage across all records, payloads, and JSON serializations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.batch_evaluator import (
    BatchEvaluationMetrics,
    DeterministicOfflineGenerationClient,
    EndToEndBatchEvaluator,
    PipelineEvaluationRecord,
)
from src.evaluation.benchmark_dataset import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_GOLDEN_CSV,
    _verify_golden_isolation,
    load_or_build_benchmark,
)

GOLDEN_CSV = DEFAULT_GOLDEN_CSV


# ---------------------------------------------------------------------------
# 1. Dataset Integrity & Golden Set Quarantine
# ---------------------------------------------------------------------------
class TestBenchmarkDatasetIntegrity:
    def test_benchmark_dataset_structure_and_counts(self):
        """Benchmark dataset must contain 100 records and all required columns."""
        df = load_or_build_benchmark()
        assert len(df) == 100

        required_cols = [
            "benchmark_id",
            "tweet_id",
            "thread_id",
            "customer_message",
            "cohort",
            "ground_truth_intent",
            "intent_label_source",
            "expected_escalate",
            "expected_escalation_rule",
            "escalation_label_source",
            "notes",
        ]
        for col in required_cols:
            assert col in df.columns

        # Verify cohorts
        cohort_counts = df["cohort"].value_counts().to_dict()
        assert cohort_counts["routine"] == 55
        assert cohort_counts["ambiguous_low_confidence"] == 15
        assert cohort_counts["escalation_sensitive"] == 15
        assert cohort_counts["vague_short"] == 8
        assert cohort_counts["weak_retrieval"] == 7

    def test_golden_set_isolation_guarantee(self):
        """Zero intersection between benchmark dataset and Phase 8 Golden Set."""
        df = load_or_build_benchmark()
        assert GOLDEN_CSV.exists()
        golden_df = pd.read_csv(GOLDEN_CSV)

        golden_ids = set(golden_df["tweet_id"].astype(int))
        benchmark_ids = set(df["tweet_id"].astype(int))

        overlap = golden_ids.intersection(benchmark_ids)
        assert len(overlap) == 0, f"CONTAMINATION DETECTED! Overlap: {overlap}"

        # Test that _verify_golden_isolation raises RuntimeError on leak
        leaked_df = pd.concat([df.head(1), golden_df.head(1)], ignore_index=True)
        with pytest.raises(RuntimeError, match="CRITICAL CONTAMINATION"):
            _verify_golden_isolation(leaked_df, golden_csv=GOLDEN_CSV)


# ---------------------------------------------------------------------------
# 2. Deterministic Offline Generator
# ---------------------------------------------------------------------------
class TestDeterministicOfflineGenerator:
    def test_offline_replay_generator_synthesizes_grounded_response(self):
        """Offline replay generator must output strict JSON with cited evidence."""
        gen = DeterministicOfflineGenerationClient()
        assert gen.config.model == "offline-deterministic-replay"

        prompt_messages = [
            {"role": "system", "content": "System instructions"},
            {
                "role": "user",
                "content": (
                    "Inquiry: battery dying\n"
                    "Customer Tweet ID: 999111\n"
                    "Brand Reply: @user please inspect Settings > Battery on your device\n"
                ),
            },
        ]

        result = gen.generate_json_response(messages=prompt_messages)
        assert "draft_response" in result
        assert "Settings > Battery" in result["draft_response"]
        assert result["cited_evidence_ids"] == [999111]
        assert result["requires_clarification"] is False


# ---------------------------------------------------------------------------
# 3. End-to-End Pipeline Execution
# ---------------------------------------------------------------------------
class TestEndToEndBatchEvaluator:
    def test_evaluator_runs_deterministic_mode(self):
        """Evaluator runs without error in deterministic mode on benchmark sample."""
        df = load_or_build_benchmark()
        sample_df = df.head(3)

        evaluator = EndToEndBatchEvaluator(mode="deterministic")
        records, metrics = evaluator.run_benchmark(benchmark_df=sample_df)

        assert len(records) == 3
        for r in records:
            assert isinstance(r, PipelineEvaluationRecord)
            assert r.benchmark_id.startswith("BM_")
            assert r.agent_model_used == "offline-deterministic-replay"
            assert r.reviewer_decision in ["PASS", "NEEDS_HUMAN_REVIEW", "FAIL"]
            assert r.reviewer_mode == "deterministic"

            # Strictly positive latencies measured via time.perf_counter()
            assert r.classifier_latency_ms > 0
            assert r.retriever_latency_ms > 0
            assert r.orchestration_latency_ms > 0
            assert r.reviewer_latency_ms > 0
            assert r.total_latency_ms > 0

        # Metrics structure verification
        assert "intent_classification" in metrics
        assert "historical_retrieval" in metrics
        assert "escalation_safety" in metrics
        assert "response_quality_reviewer" in metrics
        assert "empirical_latencies" in metrics


# ---------------------------------------------------------------------------
# 4. Metrics Mathematical Correctness
# ---------------------------------------------------------------------------
class TestBatchEvaluationMetricsMath:
    def test_synthetic_metrics_calculation(self):
        """Verify mathematical formulas for accuracy, F1, precision, recall, and confusion."""
        records = [
            PipelineEvaluationRecord(
                benchmark_id="BM_001",
                tweet_id=1,
                thread_id="t1",
                customer_message="battery drains",
                cohort="routine",
                ground_truth_intent="battery_power",
                intent_label_source="programmatically_derived_heuristic",
                expected_escalate=False,
                expected_escalation_rule=None,
                escalation_label_source="deterministic_policy_expectation",
                predicted_intent="battery_power",  # Correct intent, Correct non-escalate (TN)
                intent_confidence=0.95,
                intent_margin=0.90,
                intent_correct=True,
                top_1_similarity=0.85,
                top_k_similarities=[0.85],
                evidence_count=1,
                retrieval_below_threshold=False,
                agent_should_escalate=False,
                agent_escalation_reason=None,
                agent_response_text="Check battery settings",
                agent_grounding_score=0.85,
                agent_model_used="offline-deterministic-replay",
                reviewer_decision="PASS",
                reviewer_passed=True,
                reviewer_overall_score=0.90,
                reviewer_dimension_scores={"grounding_support": 0.9},
                reviewer_detected_issues=[],
                reviewer_rationale="Good",
                reviewer_mode="deterministic",
                classifier_latency_ms=1.5,
                retriever_latency_ms=3.0,
                orchestration_latency_ms=1.0,
                reviewer_latency_ms=2.0,
                total_latency_ms=7.5,
            ),
            PipelineEvaluationRecord(
                benchmark_id="BM_002",
                tweet_id=2,
                thread_id="t2",
                customer_message="fire exploded",
                cohort="escalation_sensitive",
                ground_truth_intent=None,
                intent_label_source="unlabeled_diagnostic",
                expected_escalate=True,
                expected_escalation_rule="safety_hazard_alert",
                escalation_label_source="deterministic_policy_expectation",
                predicted_intent="unknown_other",
                intent_confidence=0.50,
                intent_margin=0.10,
                intent_correct=None,  # Unlabeled diagnostic
                top_1_similarity=0.45,
                top_k_similarities=[0.45],
                evidence_count=1,
                retrieval_below_threshold=True,
                agent_should_escalate=True,  # Correct escalate (TP)
                agent_escalation_reason="Hazard",
                agent_response_text="Emergency contact Apple",
                agent_grounding_score=1.0,
                agent_model_used="offline-deterministic-replay",
                reviewer_decision="PASS",
                reviewer_passed=True,
                reviewer_overall_score=0.95,
                reviewer_dimension_scores={"grounding_support": 1.0},
                reviewer_detected_issues=[],
                reviewer_rationale="Passed emergency",
                reviewer_mode="deterministic",
                classifier_latency_ms=1.0,
                retriever_latency_ms=2.5,
                orchestration_latency_ms=0.5,
                reviewer_latency_ms=1.5,
                total_latency_ms=5.5,
            ),
        ]

        metrics = BatchEvaluationMetrics.compute(records, execution_mode="deterministic")

        # Intent: 1 labeled record, 1 correct -> 100% accuracy
        im = metrics["intent_classification"]
        assert im["evaluated_sample_size"] == 1
        assert im["accuracy"] == 1.0
        assert im["macro_f1"] == 1.0

        # Escalation: 1 TP, 1 TN, 0 FP, 0 FN -> Precision=1.0, Recall=1.0, F1=1.0
        em = metrics["escalation_safety"]
        assert em["confusion_counts"]["true_positives"] == 1
        assert em["confusion_counts"]["true_negatives"] == 1
        assert em["confusion_counts"]["false_positives"] == 0
        assert em["confusion_counts"]["false_negatives"] == 0
        assert em["precision"] == 1.0
        assert em["recall"] == 1.0
        assert em["f1_score"] == 1.0

        # Retrieval: 2 queries, 1 below threshold (0.45 < 0.50) -> 50%
        rm = metrics["historical_retrieval"]
        assert rm["total_queries_evaluated"] == 2
        assert rm["below_similarity_threshold_rate"] == 0.5
        assert rm["retrieval_coverage_rate"] == 1.0


# ---------------------------------------------------------------------------
# 5. Security & Secret Safety
# ---------------------------------------------------------------------------
class TestBatchEvaluationSecurity:
    def test_zero_secret_leakage_in_records(self):
        """Evaluation records and JSON payloads must contain zero API keys."""
        df = load_or_build_benchmark()
        evaluator = EndToEndBatchEvaluator(mode="deterministic")
        records, metrics = evaluator.run_benchmark(benchmark_df=df.head(2))

        payload = {"metrics": metrics, "records": [r.to_dict() for r in records]}
        payload_str = json.dumps(payload)

        assert "gsk_" not in payload_str
        assert "GROQ_API_KEY" not in payload_str
