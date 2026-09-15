"""
src/evaluation package.
Phase 13 — Agent Evaluation and Reviewer Layer.
Phase 14 — End-to-End Batch Evaluation & Quality Benchmarking.
"""

from src.evaluation.agent_reviewer import (
    AgentReviewer,
    EvaluationDimension,
    EvaluationIssue,
    EvaluationResult,
    ReviewDecision,
)
from src.evaluation.batch_evaluator import (
    BatchEvaluationMetrics,
    DeterministicOfflineGenerationClient,
    EndToEndBatchEvaluator,
    PipelineEvaluationRecord,
)
from src.evaluation.benchmark_dataset import (
    load_or_build_benchmark,
)

__all__ = [
    "AgentReviewer",
    "EvaluationDimension",
    "EvaluationIssue",
    "EvaluationResult",
    "ReviewDecision",
    "BatchEvaluationMetrics",
    "DeterministicOfflineGenerationClient",
    "EndToEndBatchEvaluator",
    "PipelineEvaluationRecord",
    "load_or_build_benchmark",
]
