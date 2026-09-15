"""
src/api/schemas.py
==================
Phase 15 — Pydantic Request & Response Schemas for Production API.

Provides strict request validation, input sanitization, and structured serializable
response contracts for consumption by external clients and a future Vercel frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Inbound customer inquiry request payload."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The customer's original inquiry text.",
        examples=["My iPhone 13 battery drains very quickly when playing music."],
    )
    top_k_evidence: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of historical interactions to retrieve for grounding.",
    )
    include_review: bool = Field(
        default=True,
        description="Whether to run the Phase 13 independent reviewer layer.",
    )
    review_mode: Optional[Literal["deterministic", "hybrid"]] = Field(
        default=None,
        description="Reviewer evaluation mode: 'deterministic' (fast, policy-audited) or 'hybrid' (Groq LLM).",
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or solely whitespace.")
        return cleaned


# ---------------------------------------------------------------------------
# Response Component Models
# ---------------------------------------------------------------------------
class RetrievedEvidenceItem(BaseModel):
    """A single grounded historical interaction retrieved from the dense corpus."""

    rank: int
    similarity_score: float
    thread_id: str
    customer_tweet_id: int
    customer_text: str
    brand_tweet_id: int
    brand_text: str
    inferred_intent: str


class AgentResponsePayload(BaseModel):
    """Structured response from GroundedSupportAgent (Phase 12)."""

    customer_message: str
    predicted_intent: str
    intent_confidence: float
    intent_uncertainty: float
    should_escalate: bool
    escalation_reason: Optional[str] = None
    escalation_rule_triggered: Optional[str] = None
    escalation_severity: str = "normal"
    draft_response: str
    grounding_summary: str
    cited_evidence_ids: List[int]
    retrieved_evidence: List[RetrievedEvidenceItem]
    model_used: str
    requires_clarification: bool = False


class ReviewerDetectedIssue(BaseModel):
    """A detected quality or policy violation from AgentReviewer (Phase 13)."""

    dimension: str
    severity: str
    description: str
    context: Optional[str] = None


class ReviewPayload(BaseModel):
    """Structured evaluation verdict from AgentReviewer (Phase 13)."""

    passed: bool
    decision: Literal["PASS", "NEEDS_HUMAN_REVIEW", "FAIL"]
    overall_score: float
    dimension_scores: Dict[str, float]
    detected_issues: List[ReviewerDetectedIssue]
    escalation_consistency: Dict[str, Any]
    grounding_findings: Dict[str, Any]
    reviewer_rationale: str
    reviewer_mode: str
    model_used: Optional[str] = None


class ChatResponseData(BaseModel):
    """Inner data container for chat response."""

    response: AgentResponsePayload
    review: Optional[ReviewPayload] = None
    execution_time_ms: float


class ChatResponse(BaseModel):
    """Top-level success response envelope for chat requests."""

    status: str = "success"
    data: ChatResponseData


# ---------------------------------------------------------------------------
# Health & Diagnostic Models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = "ok"
    service: str = "apple-support-ai-agent"
    version: str = "1.0.0"
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness probe response verifying all pipeline assets are loaded."""

    status: Literal["ready", "degraded", "not_ready"]
    intent_classifier_ready: bool
    retriever_ready: bool
    retriever_corpus_size: int
    groq_configured: bool
    groq_model: str
    timestamp: str


class ErrorResponse(BaseModel):
    """RFC-compliant error response envelope preventing secret or stack trace leakage."""

    status: str = "error"
    code: str
    message: str
    details: Optional[Any] = None
