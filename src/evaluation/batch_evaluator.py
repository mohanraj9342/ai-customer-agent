"""
src/evaluation/batch_evaluator.py
=================================
Phase 14 — End-to-End Batch Evaluation & Benchmarking Engine.

Orchestrates execution across the complete 4-phase architecture:
  Phase 10: Intent Classification
  Phase 11: Historical Response Retrieval
  Phase 12: Grounded Support Agent Response Generation
  Phase 13: Independent Agent Reviewer

Supports:
- --mode deterministic: Offline structural evaluation using an explicit deterministic
  replay generator (zero token/network overhead, fully reproducible).
- --mode live: Real Groq API execution using qwen/qwen3.8-27b.

All latencies are strictly measured using time.perf_counter().
Zero Golden Set access is enforced.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.evaluation.agent_reviewer import AgentReviewer, EvaluationResult
from src.evaluation.benchmark_dataset import TAXONOMY_CLASSES
from src.generation.agent_orchestrator import (
    AgentResponse,
    GroundedSupportAgent,
    IntentPredictor,
)
from src.generation.config import GroqConfig
from src.generation.groq_client import GroqGenerationClient
from src.retrieval.historical_response_retriever import (
    HistoricalResponseRetriever,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explicit Deterministic Offline Replay Generator
# ---------------------------------------------------------------------------
class DeterministicOfflineGenerationClient:
    """
    Explicit offline mock/replay generation client for deterministic mode.
    Synthesizes a response strictly from top-1 retrieved historical evidence.

    Clearly labeled as offline replay — NEVER pretends to be a live Groq generation.
    """

    def __init__(self, model_name: str = "offline-deterministic-replay") -> None:
        self.config = GroqConfig(api_key="mock_deterministic_offline_key", model=model_name)

    def generate_json_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extracts evidence from prompt and formats an offline replay response.
        """
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")

        # Look for historical brand replies in the user prompt
        brand_replies: List[str] = []
        evidence_ids: List[int] = []

        for line in user_content.split("\n"):
            if "Brand Reply:" in line:
                rep = line.split("Brand Reply:", 1)[1].strip()
                if rep:
                    # Clean handle
                    rep_clean = " ".join(w for w in rep.split() if not w.startswith("@"))
                    brand_replies.append(rep_clean)
            if "Customer Tweet ID:" in line:
                try:
                    cid = int(line.split("Customer Tweet ID:", 1)[1].strip())
                    evidence_ids.append(cid)
                except ValueError:
                    pass

        if brand_replies:
            top_reply = brand_replies[0]
            draft = (
                f"Thanks for reaching out! Based on verified historical guidance: {top_reply} "
                f"If you still need help, please DM us your details."
            )
            cited = evidence_ids[:1] if evidence_ids else []
            summary = "Grounded response synthesized deterministically from top-1 historical evidence."
        else:
            draft = "Thanks for contacting Apple Support! Please DM us with your device model and iOS version so we can assist."
            cited = []
            summary = "Fallback generic response (no evidence available in offline replay)."

        return {
            "draft_response": draft,
            "cited_evidence_ids": cited,
            "grounding_summary": summary,
            "requires_clarification": False,
        }


# ---------------------------------------------------------------------------
# Pipeline Evaluation Record Schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineEvaluationRecord:
    """Single end-to-end evaluation result for one benchmark inquiry."""

    benchmark_id: str
    tweet_id: int
    thread_id: str
    customer_message: str
    cohort: str
    ground_truth_intent: Optional[str]
    intent_label_source: str
    expected_escalate: bool
    expected_escalation_rule: Optional[str]
    escalation_label_source: str

    # Phase 10 outputs
    predicted_intent: str
    intent_confidence: float
    intent_margin: float
    intent_correct: Optional[bool]

    # Phase 11 outputs
    top_1_similarity: float
    top_k_similarities: List[float]
    evidence_count: int
    retrieval_below_threshold: bool

    # Phase 12 outputs
    agent_should_escalate: bool
    agent_escalation_reason: Optional[str]
    agent_response_text: str
    agent_grounding_score: float
    agent_model_used: str

    # Phase 13 outputs
    reviewer_decision: str  # "PASS", "NEEDS_HUMAN_REVIEW", "FAIL"
    reviewer_passed: bool
    reviewer_overall_score: float
    reviewer_dimension_scores: Dict[str, float]
    reviewer_detected_issues: List[Dict[str, Any]]
    reviewer_rationale: str
    reviewer_mode: str

    # Measured empirical latencies (in milliseconds)
    classifier_latency_ms: float
    retriever_latency_ms: float
    orchestration_latency_ms: float
    reviewer_latency_ms: float
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics Computation Engine
# ---------------------------------------------------------------------------
class BatchEvaluationMetrics:
    """
    Computes rigorous metrics across Intent Classification, Historical Retrieval,
    Escalation Correctness, Response Quality, and Measured Component Latencies.
    """

    @classmethod
    def compute(
        cls,
        records: List[PipelineEvaluationRecord],
        execution_mode: str = "deterministic",
    ) -> Dict[str, Any]:
        if not records:
            return {"error": "No evaluation records provided"}

        intent_metrics = cls._compute_intent_metrics(records)
        retrieval_metrics = cls._compute_retrieval_metrics(records)
        escalation_metrics = cls._compute_escalation_metrics(records)
        quality_metrics = cls._compute_quality_metrics(records)
        latency_metrics = cls._compute_latency_metrics(records)

        return {
            "evaluation_metadata": {
                "execution_mode": execution_mode,
                "benchmark_size": len(records),
                "is_live_groq": (execution_mode == "live"),
                "methodological_notice": (
                    "This 100-case benchmark is an engineered stress-test and coverage cohort. "
                    "Metrics must not be interpreted as statistically representative of the overall macro distribution."
                ),
            },
            "intent_classification": intent_metrics,
            "historical_retrieval": retrieval_metrics,
            "escalation_safety": escalation_metrics,
            "response_quality_reviewer": quality_metrics,
            "empirical_latencies": latency_metrics,
        }

    # -----------------------------------------------------------------------
    # 1. Intent Classification Metrics
    # -----------------------------------------------------------------------
    @classmethod
    def _compute_intent_metrics(cls, records: List[PipelineEvaluationRecord]) -> Dict[str, Any]:
        # Filter to cases where ground truth exists and is not diagnostic
        labeled_records = [
            r for r in records
            if r.ground_truth_intent and r.intent_label_source != "unlabeled_diagnostic"
        ]

        if not labeled_records:
            return {"status": "no_ground_truth_records_evaluated"}

        y_true = [r.ground_truth_intent for r in labeled_records]
        y_pred = [r.predicted_intent for r in labeled_records]

        # Overall accuracy
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = round(correct / len(labeled_records), 4)

        # Standard 11 taxonomy classes
        labels = TAXONOMY_CLASSES

        # Precision, recall, F1
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-intent metrics
        p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_intent: Dict[str, Dict[str, float]] = {}
        for lbl, p, r, f1, sup in zip(labels, p_per, r_per, f1_per, sup_per):
            per_intent[lbl] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1_score": round(float(f1), 4),
                "support": int(sup),
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Historical comparison references (from Phase 9 & 10 Golden Set evaluations)
        baselines_comparison = {
            "phase9_majority_baseline_f1": 0.0151,
            "phase9_linear_svm_f1": 0.5961,
            "phase9_logistic_regression_f1": 0.6319,
            "phase10_distilroberta_f1": 0.5929,
            "current_benchmark_macro_f1": round(float(f1_macro), 4),
            "note": (
                "Benchmark dataset is a distinct non-golden stress-test; differences from "
                "Phase 9/10 reflect the distinct composition of this 100-case cohort."
            ),
        }

        return {
            "evaluated_sample_size": len(labeled_records),
            "accuracy": accuracy,
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(p_weighted), 4),
            "weighted_recall": round(float(r_weighted), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_intent": per_intent,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
            "historical_baselines_reference": baselines_comparison,
        }

    # -----------------------------------------------------------------------
    # 2. Historical Retrieval Metrics
    # -----------------------------------------------------------------------
    @classmethod
    def _compute_retrieval_metrics(cls, records: List[PipelineEvaluationRecord]) -> Dict[str, Any]:
        total = len(records)
        covered = sum(1 for r in records if r.evidence_count > 0)
        coverage_rate = round(covered / total, 4)

        top1_sims = [r.top_1_similarity for r in records]
        below_thresh_count = sum(1 for r in records if r.retrieval_below_threshold)
        below_thresh_rate = round(below_thresh_count / total, 4)

        # Failure rate: 0 results returned
        failure_rate = round(sum(1 for r in records if r.evidence_count == 0) / total, 4)

        # Distribution statistics
        top1_arr = np.array(top1_sims)
        dist = {
            "mean": round(float(np.mean(top1_arr)), 4),
            "std": round(float(np.std(top1_arr)), 4),
            "min": round(float(np.min(top1_arr)), 4),
            "p25": round(float(np.percentile(top1_arr, 25)), 4),
            "p50_median": round(float(np.percentile(top1_arr, 50)), 4),
            "p75": round(float(np.percentile(top1_arr, 75)), 4),
            "p95": round(float(np.percentile(top1_arr, 95)), 4),
            "max": round(float(np.max(top1_arr)), 4),
        }

        # Intent-conditioned retrieval similarity
        intent_groups: Dict[str, List[float]] = {}
        for r in records:
            intent_groups.setdefault(r.predicted_intent, []).append(r.top_1_similarity)

        intent_conditioned = {
            intent: round(float(np.mean(sims)), 4)
            for intent, sims in sorted(intent_groups.items())
        }

        return {
            "total_queries_evaluated": total,
            "retrieval_coverage_rate": coverage_rate,
            "retrieval_failure_rate": failure_rate,
            "below_similarity_threshold_rate": below_thresh_rate,
            "threshold_configured": 0.50,
            "leave_one_out_exclusion_active": True,
            "top1_similarity_distribution": dist,
            "intent_conditioned_mean_similarity": intent_conditioned,
        }

    # -----------------------------------------------------------------------
    # 3. Escalation Metrics
    # -----------------------------------------------------------------------
    @classmethod
    def _compute_escalation_metrics(cls, records: List[PipelineEvaluationRecord]) -> Dict[str, Any]:
        # Filter to cases with expected escalation policy labels
        eval_records = [
            r for r in records
            if r.escalation_label_source != "none"
        ]

        total = len(eval_records)
        tp = sum(1 for r in eval_records if r.expected_escalate and r.agent_should_escalate)
        fn = sum(1 for r in eval_records if r.expected_escalate and not r.agent_should_escalate)
        fp = sum(1 for r in eval_records if not r.expected_escalate and r.agent_should_escalate)
        tn = sum(1 for r in eval_records if not r.expected_escalate and not r.agent_should_escalate)

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
        fnr = round(fn / (tp + fn), 4) if (tp + fn) > 0 else 0.0

        # Cohort breakdown
        cohort_breakdown: Dict[str, Dict[str, int]] = {}
        for r in eval_records:
            c_dict = cohort_breakdown.setdefault(r.cohort, {"total": 0, "escalated": 0, "expected_escalate": 0})
            c_dict["total"] += 1
            if r.agent_should_escalate:
                c_dict["escalated"] += 1
            if r.expected_escalate:
                c_dict["expected_escalate"] += 1

        return {
            "evaluated_sample_size": total,
            "confusion_counts": {
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "true_negatives": tn,
            },
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "cohort_breakdown": cohort_breakdown,
            "methodology_note": (
                "Expected escalation labels are deterministic policy/rule expectations "
                "(e.g. physical hazard, legal threat, account access, vague query, weak retrieval)."
            ),
        }

    # -----------------------------------------------------------------------
    # 4. Response Quality & Reviewer Metrics
    # -----------------------------------------------------------------------
    @classmethod
    def _compute_quality_metrics(cls, records: List[PipelineEvaluationRecord]) -> Dict[str, Any]:
        total = len(records)
        pass_count = sum(1 for r in records if r.reviewer_decision == "PASS")
        review_count = sum(1 for r in records if r.reviewer_decision == "NEEDS_HUMAN_REVIEW")
        fail_count = sum(1 for r in records if r.reviewer_decision == "FAIL")

        overall_scores = [r.reviewer_overall_score for r in records]

        # Dimension averages
        dim_names = [
            "grounding_support",
            "hallucination_risk",
            "escalation_correctness",
            "intent_consistency",
            "response_relevance",
            "response_completeness",
            "professional_quality",
        ]
        dim_means: Dict[str, float] = {}
        for d in dim_names:
            scores = [r.reviewer_dimension_scores.get(d, 0.0) for r in records]
            dim_means[d] = round(float(np.mean(scores)), 4) if scores else 0.0

        # Issue rates
        hallucination_issues = sum(
            1 for r in records
            if any(iss.get("dimension") == "hallucination_risk" for iss in r.reviewer_detected_issues)
        )
        grounding_issues = sum(
            1 for r in records
            if any(iss.get("dimension") == "grounding_support" for iss in r.reviewer_detected_issues)
        )
        intent_drift_issues = sum(
            1 for r in records
            if any(iss.get("dimension") == "intent_consistency" for iss in r.reviewer_detected_issues)
        )
        escalation_issues = sum(
            1 for r in records
            if any(iss.get("dimension") == "escalation_correctness" for iss in r.reviewer_detected_issues)
        )

        return {
            "total_evaluated": total,
            "pass_count": pass_count,
            "pass_rate": round(pass_count / total, 4),
            "needs_human_review_count": review_count,
            "needs_human_review_rate": round(review_count / total, 4),
            "fail_count": fail_count,
            "fail_rate": round(fail_count / total, 4),
            "mean_overall_score": round(float(np.mean(overall_scores)), 4),
            "dimension_mean_scores": dim_means,
            "issue_rates": {
                "hallucination_issue_rate": round(hallucination_issues / total, 4),
                "grounding_issue_rate": round(grounding_issues / total, 4),
                "intent_drift_issue_rate": round(intent_drift_issues / total, 4),
                "escalation_mismatch_rate": round(escalation_issues / total, 4),
            },
            "disclaimer": "Metrics represent reviewer evaluations and are not absolute human ground truth.",
        }

    # -----------------------------------------------------------------------
    # 5. Measured Latencies
    # -----------------------------------------------------------------------
    @classmethod
    def _compute_latency_metrics(cls, records: List[PipelineEvaluationRecord]) -> Dict[str, Any]:
        def _calc_stats(vals: List[float]) -> Dict[str, float]:
            arr = np.array(vals)
            return {
                "mean_ms": round(float(np.mean(arr)), 2),
                "std_ms": round(float(np.std(arr)), 2),
                "min_ms": round(float(np.min(arr)), 2),
                "p50_median_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "max_ms": round(float(np.max(arr)), 2),
            }

        return {
            "measurement_method": "time.perf_counter() around each component invocation",
            "classifier_latency": _calc_stats([r.classifier_latency_ms for r in records]),
            "retriever_latency": _calc_stats([r.retriever_latency_ms for r in records]),
            "orchestration_latency": _calc_stats([r.orchestration_latency_ms for r in records]),
            "reviewer_latency": _calc_stats([r.reviewer_latency_ms for r in records]),
            "total_pipeline_latency": _calc_stats([r.total_latency_ms for r in records]),
        }


# ---------------------------------------------------------------------------
# End-to-End Batch Evaluator Runner
# ---------------------------------------------------------------------------
class EndToEndBatchEvaluator:
    """
    Executes the complete Phase 10 -> Phase 11 -> Phase 12 -> Phase 13 pipeline
    across benchmark records with rigorous per-component latency profiling.
    """

    def __init__(
        self,
        mode: str = "deterministic",
        groq_client: Optional[Any] = None,
        retriever: Optional[HistoricalResponseRetriever] = None,
        intent_predictor: Optional[IntentPredictor] = None,
        reviewer: Optional[AgentReviewer] = None,
    ) -> None:
        self.mode = mode

        # Phase 10 Intent Predictor
        self.intent_predictor = intent_predictor or IntentPredictor()

        # Phase 11 Historical Retriever
        self.retriever = retriever or HistoricalResponseRetriever()

        # Phase 12 Groq / Offline Generation Client
        if mode == "deterministic":
            self.gen_client = DeterministicOfflineGenerationClient()
        else:
            self.gen_client = groq_client or GroqGenerationClient()

        # Phase 12 Orchestrator
        self.agent = GroundedSupportAgent(
            retriever=self.retriever,
            groq_client=self.gen_client,
            intent_predictor=self.intent_predictor,
        )

        # Phase 13 Reviewer
        self.reviewer = reviewer or AgentReviewer()

    def run_benchmark(
        self,
        benchmark_df: pd.DataFrame,
        limit: Optional[int] = None,
    ) -> Tuple[List[PipelineEvaluationRecord], Dict[str, Any]]:
        """
        Runs the full 4-stage pipeline sequentially on benchmark records.
        """
        df = benchmark_df.head(limit) if limit else benchmark_df
        records: List[PipelineEvaluationRecord] = []

        logger.info(
            "Starting End-to-End Batch Evaluation (mode: %s, cases: %d)...",
            self.mode,
            len(df),
        )

        for idx, row in df.iterrows():
            customer_msg = str(row["customer_message"]).strip()
            b_id = str(row["benchmark_id"])
            cohort = str(row["cohort"])
            gt_intent = row["ground_truth_intent"] if pd.notna(row["ground_truth_intent"]) else None
            intent_source = str(row["intent_label_source"])
            exp_esc = bool(row["expected_escalate"])
            exp_rule = row["expected_escalation_rule"] if pd.notna(row["expected_escalation_rule"]) else None
            esc_source = str(row["escalation_label_source"])

            # ---------------------------------------------------------------
            # Stage 1: Phase 10 Intent Classification
            # ---------------------------------------------------------------
            t0 = time.perf_counter()
            pred_intent, conf, margin = self.intent_predictor.predict(customer_msg)
            t1 = time.perf_counter()
            t_clf_ms = (t1 - t0) * 1000.0

            # Per-query exclusions (leave-one-out self-match prevention)
            q_tweet_id = int(row["tweet_id"]) if pd.notna(row.get("tweet_id")) else None
            q_thread_id = str(row["thread_id"]).strip() if pd.notna(row.get("thread_id")) else None
            q_text = customer_msg.strip().lower() if customer_msg else None

            exclude_tids = {q_tweet_id} if q_tweet_id is not None else None
            exclude_thids = {q_thread_id} if q_thread_id else None
            exclude_txts = {q_text} if q_text else None

            # ---------------------------------------------------------------
            # Stage 2: Phase 11 Historical Retrieval (Leave-One-Out)
            # ---------------------------------------------------------------
            t2 = time.perf_counter()
            retrieved_cases = self.retriever.retrieve(
                customer_msg,
                top_k=3,
                deduplicate_customer_text=True,
                exclude_customer_tweet_ids=exclude_tids,
                exclude_thread_ids=exclude_thids,
                exclude_exact_customer_texts=exclude_txts,
            )
            t3 = time.perf_counter()
            t_ret_ms = (t3 - t2) * 1000.0

            top1_sim = retrieved_cases[0].similarity_score if retrieved_cases else 0.0
            top_k_sims = [c.similarity_score for c in retrieved_cases]
            below_thresh = bool(top1_sim < 0.50)

            # ---------------------------------------------------------------
            # Stage 3: Phase 12 Agent Orchestration & Response Generation
            # ---------------------------------------------------------------
            t4 = time.perf_counter()
            agent_resp: AgentResponse = self.agent.process_message(
                customer_msg,
                top_k_evidence=3,
                exclude_customer_tweet_ids=exclude_tids,
                exclude_thread_ids=exclude_thids,
                exclude_exact_customer_texts=exclude_txts,
            )
            t5 = time.perf_counter()
            t_gen_ms = (t5 - t4) * 1000.0

            # ---------------------------------------------------------------
            # Stage 4: Phase 13 Agent Reviewer (Deterministic Audit)
            # ---------------------------------------------------------------
            t6 = time.perf_counter()
            review_res: EvaluationResult = self.reviewer.evaluate(
                customer_message=customer_msg,
                agent_response=agent_resp,
                use_llm=(self.mode == "live"),
            )
            t7 = time.perf_counter()
            t_rev_ms = (t7 - t6) * 1000.0

            t_total_ms = t_clf_ms + t_ret_ms + t_gen_ms + t_rev_ms

            # Ground truth intent comparison
            intent_correct: Optional[bool] = None
            if gt_intent and intent_source != "unlabeled_diagnostic":
                intent_correct = (pred_intent == gt_intent)

            record = PipelineEvaluationRecord(
                benchmark_id=b_id,
                tweet_id=int(row["tweet_id"]),
                thread_id=str(row["thread_id"]),
                customer_message=customer_msg,
                cohort=cohort,
                ground_truth_intent=gt_intent,
                intent_label_source=intent_source,
                expected_escalate=exp_esc,
                expected_escalation_rule=exp_rule,
                escalation_label_source=esc_source,
                predicted_intent=pred_intent,
                intent_confidence=round(conf, 4),
                intent_margin=round(margin, 4),
                intent_correct=intent_correct,
                top_1_similarity=round(top1_sim, 4),
                top_k_similarities=[round(s, 4) for s in top_k_sims],
                evidence_count=len(retrieved_cases),
                retrieval_below_threshold=below_thresh,
                agent_should_escalate=agent_resp.should_escalate,
                agent_escalation_reason=agent_resp.escalation_reason,
                agent_response_text=agent_resp.response,
                agent_grounding_score=round(agent_resp.grounding_score, 4),
                agent_model_used=str(agent_resp.model_metadata.get("model_used", "unknown")),
                reviewer_decision=review_res.decision,
                reviewer_passed=review_res.passed,
                reviewer_overall_score=review_res.overall_score,
                reviewer_dimension_scores=review_res.dimension_scores,
                reviewer_detected_issues=review_res.detected_issues,
                reviewer_rationale=review_res.reviewer_rationale,
                reviewer_mode=review_res.reviewer_mode,
                classifier_latency_ms=round(t_clf_ms, 2),
                retriever_latency_ms=round(t_ret_ms, 2),
                orchestration_latency_ms=round(t_gen_ms, 2),
                reviewer_latency_ms=round(t_rev_ms, 2),
                total_latency_ms=round(t_total_ms, 2),
            )
            records.append(record)

        summary_metrics = BatchEvaluationMetrics.compute(records, execution_mode=self.mode)
        return records, summary_metrics
