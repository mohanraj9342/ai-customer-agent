#!/usr/bin/env python3
"""
scripts/run_batch_evaluation.py
===============================
Phase 14 — End-to-End Batch Evaluation & Quality Benchmarking CLI.

Executes the complete pipeline (Phase 10 -> 11 -> 12 -> 13) across the 100-case
controlled benchmark dataset.

Usage:
  # 1. Deterministic offline structural evaluation (Recommended for CI / regression):
  .venv/bin/python scripts/run_batch_evaluation.py --mode deterministic --limit 100

  # 2. Live Groq evaluation (Manual invocation only):
  .venv/bin/python scripts/run_batch_evaluation.py --mode live --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.batch_evaluator import EndToEndBatchEvaluator
from src.evaluation.benchmark_dataset import DEFAULT_BENCHMARK_CSV, load_or_build_benchmark
from src.generation.config import load_groq_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_evaluator")

DEFAULT_RESULTS_JSON = PROJECT_ROOT / "data/processed/apple_support/apple_support_e2e_benchmark_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 14 End-to-End Batch Evaluation & Quality Benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "live"],
        default="deterministic",
        help="Execution mode: 'deterministic' (offline mock replay, zero API costs) or 'live' (real Groq API).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of benchmark cases to evaluate.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=DEFAULT_BENCHMARK_CSV,
        help="Path to the controlled benchmark CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help="Path to save the evaluation results JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case execution diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 75)
    print("PHASE 14: END-TO-END BATCH EVALUATION & QUALITY BENCHMARKING")
    print(f"Mode: {args.mode.upper()}")
    print("=" * 75)

    if args.mode == "live":
        try:
            cfg = load_groq_config()
            print(f"WARNING: Live mode uses the actual Groq API (Model: {cfg.model}).")
            print("Token costs and rate limits apply. Never run inside pytest.")
        except Exception as exc:
            logger.error("Failed to load Groq configuration for live mode: %s", exc)
            return 1

    # Load or build benchmark dataset
    benchmark_df = load_or_build_benchmark(benchmark_csv=args.benchmark_csv)
    total_to_run = min(len(benchmark_df), args.limit) if args.limit else len(benchmark_df)
    print(f"Loaded benchmark dataset: {len(benchmark_df)} records (evaluating {total_to_run})")

    # Initialize evaluator
    evaluator = EndToEndBatchEvaluator(mode=args.mode)

    # Run evaluation
    records, metrics = evaluator.run_benchmark(benchmark_df=benchmark_df, limit=args.limit)

    if args.verbose:
        print("\n--- SAMPLE CASE RESULTS (First 5) ---")
        for r in records[:5]:
            print(f"[{r.benchmark_id}] ({r.cohort}) Query: {r.customer_message[:50]!r}")
            print(f"   Intent: {r.predicted_intent} (Conf: {r.intent_confidence:.2f}) | Escalate: {r.agent_should_escalate}")
            print(f"   Review: {r.reviewer_decision} (Score: {r.reviewer_overall_score:.2f}) | Latency: {r.total_latency_ms:.1f}ms")

    # Save output JSON
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "records": [r.to_dict() for r in records],
    }

    # Security check: verify zero API keys in output payload
    payload_str = json.dumps(payload)
    if "gsk_" in payload_str:
        raise RuntimeError("SECURITY VIOLATION: Groq API key detected in benchmark output payload!")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults successfully saved to: {output_path}")

    # Print summary report
    print("\n" + "=" * 75)
    print("EVALUATION BENCHMARK SUMMARY REPORT")
    print("=" * 75)

    im = metrics.get("intent_classification", {})
    if "accuracy" in im:
        print(f"\n1. INTENT CLASSIFICATION (Labeled Routine Cases, N={im['evaluated_sample_size']}):")
        print(f"   Accuracy:         {im['accuracy'] * 100:.2f}%")
        print(f"   Macro-F1:         {im['macro_f1']:.4f}")
        print(f"   Weighted-F1:      {im['weighted_f1']:.4f}")
        comp = im.get("historical_baselines_reference", {})
        print(f"   Ref Baselines:    Maj: {comp.get('phase9_majority_baseline_f1')}, SVM: {comp.get('phase9_linear_svm_f1')}, LR: {comp.get('phase9_logistic_regression_f1')}, RoBERTa: {comp.get('phase10_distilroberta_f1')}")

    rm = metrics.get("historical_retrieval", {})
    print(f"\n2. HISTORICAL RETRIEVAL (N={rm.get('total_queries_evaluated', 0)}):")
    print(f"   Coverage Rate:    {rm.get('retrieval_coverage_rate', 0) * 100:.1f}%")
    print(f"   Below Threshold:  {rm.get('below_similarity_threshold_rate', 0) * 100:.1f}% (Threshold: {rm.get('threshold_configured', 0.50)})")
    dist = rm.get("top1_similarity_distribution", {})
    print(f"   Top-1 Sim Stats:  Mean: {dist.get('mean')}, Median: {dist.get('p50_median')}, p95: {dist.get('p95')}, Min: {dist.get('min')}, Max: {dist.get('max')}")

    em = metrics.get("escalation_safety", {})
    print(f"\n3. ESCALATION SAFETY (Deterministic Policy Audit, N={em.get('evaluated_sample_size', 0)}):")
    print(f"   Precision:        {em.get('precision', 0) * 100:.1f}%")
    print(f"   Recall:           {em.get('recall', 0) * 100:.1f}%")
    print(f"   F1-Score:         {em.get('f1_score', 0):.4f}")
    conf_c = em.get("confusion_counts", {})
    print(f"   Confusion Matrix: TP: {conf_c.get('true_positives')}, FP: {conf_c.get('false_positives')}, TN: {conf_c.get('true_negatives')}, FN: {conf_c.get('false_negatives')}")

    qm = metrics.get("response_quality_reviewer", {})
    print(f"\n4. RESPONSE QUALITY & REVIEWER (Phase 13 Reviewer, N={qm.get('total_evaluated', 0)}):")
    print(f"   PASS Rate:        {qm.get('pass_rate', 0) * 100:.1f}% ({qm.get('pass_count', 0)} cases)")
    print(f"   NEEDS REVIEW:     {qm.get('needs_human_review_rate', 0) * 100:.1f}% ({qm.get('needs_human_review_count', 0)} cases)")
    print(f"   FAIL Rate:        {qm.get('fail_rate', 0) * 100:.1f}% ({qm.get('fail_count', 0)} cases)")
    print(f"   Mean Overall:     {qm.get('mean_overall_score', 0):.4f}")
    dms = qm.get("dimension_mean_scores", {})
    print(f"   Dimension Scores: Grounding: {dms.get('grounding_support')}, Hallucination: {dms.get('hallucination_risk')}, Escalation: {dms.get('escalation_correctness')}")
    print(f"                     Intent: {dms.get('intent_consistency')}, Relevance: {dms.get('response_relevance')}, Completeness: {dms.get('response_completeness')}, Tone: {dms.get('professional_quality')}")

    lm = metrics.get("empirical_latencies", {})
    print(f"\n5. EMPIRICALLY MEASURED LATENCIES (time.perf_counter):")
    print(f"   Classifier:       Mean: {lm.get('classifier_latency', {}).get('mean_ms')} ms | p95: {lm.get('classifier_latency', {}).get('p95_ms')} ms")
    print(f"   Retriever:        Mean: {lm.get('retriever_latency', {}).get('mean_ms')} ms | p95: {lm.get('retriever_latency', {}).get('p95_ms')} ms")
    print(f"   Orchestrator:     Mean: {lm.get('orchestration_latency', {}).get('mean_ms')} ms | p95: {lm.get('orchestration_latency', {}).get('p95_ms')} ms")
    print(f"   Reviewer:         Mean: {lm.get('reviewer_latency', {}).get('mean_ms')} ms | p95: {lm.get('reviewer_latency', {}).get('p95_ms')} ms")
    print(f"   Total Pipeline:   Mean: {lm.get('total_pipeline_latency', {}).get('mean_ms')} ms | p95: {lm.get('total_pipeline_latency', {}).get('p95_ms')} ms")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
