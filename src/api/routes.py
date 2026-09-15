"""
src/api/routes.py
=================
Phase 15 — Production API Endpoints & Request Routing.

Exposes:
- GET  /health              -> Service liveness probe.
- GET  /ready               -> Pipeline asset and configuration readiness probe.
- POST /api/chat            -> Primary customer support message processing endpoint.
- POST /api/support/message -> Backward-compatible endpoint alias.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.config import ApiConfig
from src.api.schemas import (
    AgentResponsePayload,
    ChatRequest,
    ChatResponse,
    ChatResponseData,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    RetrievedEvidenceItem,
    ReviewerDetectedIssue,
    ReviewPayload,
)
from src.evaluation.agent_reviewer import AgentReviewer, EvaluationResult
from src.generation.agent_orchestrator import AgentResponse, GroundedSupportAgent
from src.generation.groq_client import GroqGenerationError

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency Injectors
# ---------------------------------------------------------------------------
def get_agent(request: Request) -> GroundedSupportAgent:
    """Retrieve pre-warmed GroundedSupportAgent from application state."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Support agent pipeline is not yet initialized.",
        )
    return agent


def get_reviewer(request: Request) -> AgentReviewer:
    """Retrieve pre-warmed AgentReviewer from application state."""
    reviewer = getattr(request.app.state, "reviewer", None)
    if reviewer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent reviewer layer is not yet initialized.",
        )
    return reviewer


def get_api_config(request: Request) -> ApiConfig:
    """Retrieve API configuration from application state."""
    config = getattr(request.app.state, "api_config", None)
    if config is None:
        return ApiConfig()
    return config


# ---------------------------------------------------------------------------
# Health & Readiness Endpoints
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Liveness probe for load balancers and orchestrators",
)
async def health_check() -> HealthResponse:
    """Return 200 OK if the web server process is alive and responsive."""
    return HealthResponse(
        status="ok",
        service="apple-support-ai-agent",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Diagnostics"],
    summary="Readiness probe verifying all pipeline assets are loaded in memory",
)
async def readiness_check(request: Request) -> ReadinessResponse:
    """
    Verifies that the intent classifier, historical retrieval index, and Groq
    configuration are loaded and ready to serve customer traffic.
    """
    agent: Optional[GroundedSupportAgent] = getattr(request.app.state, "agent", None)
    timestamp = datetime.now(timezone.utc).isoformat()

    if agent is None:
        return ReadinessResponse(
            status="not_ready",
            intent_classifier_ready=False,
            retriever_ready=False,
            retriever_corpus_size=0,
            groq_configured=False,
            groq_model="none",
            timestamp=timestamp,
        )

    # Check Intent Classifier
    intent_ready = (
        agent.intent_predictor is not None
        and hasattr(agent.intent_predictor, "classifier")
        and agent.intent_predictor.classifier is not None
    )

    # Check Retriever
    retriever_ready = False
    corpus_size = 0
    try:
        retriever = agent._ensure_retriever()
        if retriever is not None:
            corpus_size = retriever.corpus_size
            retriever_ready = corpus_size > 0
    except Exception:
        retriever_ready = False

    # Check Groq configuration safely (without exposing keys)
    cfg = agent._get_config()
    meta = cfg.safe_metadata()
    groq_ready = bool(meta.get("api_key_configured", False))
    groq_model = str(meta.get("model", "unknown"))

    overall_status = "ready" if (intent_ready and retriever_ready and groq_ready) else "degraded"

    return ReadinessResponse(
        status=overall_status,
        intent_classifier_ready=intent_ready,
        retriever_ready=retriever_ready,
        retriever_corpus_size=corpus_size,
        groq_configured=groq_ready,
        groq_model=groq_model,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Core Chat / Message Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid customer message"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal pipeline failure"},
        503: {"model": ErrorResponse, "description": "Upstream LLM or service unavailable"},
    },
    tags=["Agent Interaction"],
    summary="Process customer inquiry through the Grounded Support Agent",
)
@router.post(
    "/api/support/message",
    response_model=ChatResponse,
    include_in_schema=False,
    tags=["Agent Interaction"],
)
async def process_chat_message(
    payload: ChatRequest,
    agent: GroundedSupportAgent = Depends(get_agent),
    reviewer: AgentReviewer = Depends(get_reviewer),
    api_config: ApiConfig = Depends(get_api_config),
) -> ChatResponse:
    """
    Primary API entry point for customer message triage and response generation:
    1. Validates message context and bounds.
    2. Runs Phase 10 Intent Classification.
    3. Runs Phase 11 Historical Semantic Retrieval.
    4. Evaluates Phase 12 Deterministic Safety & Escalation Rules.
    5. Generates grounded response via Groq (or returns safe escalation).
    6. Audits output via Phase 13 independent reviewer.
    """
    t_start = time.perf_counter()

    # Determine reviewer mode (request explicit override > server default)
    review_mode = payload.review_mode or api_config.default_review_mode
    use_llm_reviewer = (review_mode == "hybrid")

    try:
        # Phase 10 -> Phase 11 -> Phase 12 Execution
        agent_resp: AgentResponse = agent.process_message(
            customer_message=payload.message,
            top_k_evidence=payload.top_k_evidence,
        )

        # Phase 13 Reviewer Execution (if requested)
        review_payload: Optional[ReviewPayload] = None
        if payload.include_review:
            eval_result: EvaluationResult = reviewer.evaluate(
                customer_message=payload.message,
                agent_response=agent_resp,
                use_llm=use_llm_reviewer,
            )

            detected_issues = [
                ReviewerDetectedIssue(
                    dimension=str(iss.get("dimension", "unknown")),
                    severity=str(iss.get("severity", "medium")),
                    description=str(iss.get("description", "")),
                    context=iss.get("context"),
                )
                for iss in eval_result.detected_issues
            ]

            review_payload = ReviewPayload(
                passed=eval_result.passed,
                decision=eval_result.decision,
                overall_score=eval_result.overall_score,
                dimension_scores=eval_result.dimension_scores,
                detected_issues=detected_issues,
                escalation_consistency=eval_result.escalation_consistency,
                grounding_findings=eval_result.grounding_findings,
                reviewer_rationale=eval_result.reviewer_rationale,
                reviewer_mode=eval_result.reviewer_mode,
                model_used=eval_result.model_used,
            )

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Transform retrieved evidence into Pydantic schema
        evidence_items: List[RetrievedEvidenceItem] = []
        for ev in agent_resp.evidence:
            evidence_items.append(
                RetrievedEvidenceItem(
                    rank=int(ev.get("rank", 0)),
                    similarity_score=float(ev.get("similarity_score", 0.0)),
                    thread_id=str(ev.get("thread_id", "")),
                    customer_tweet_id=int(ev.get("customer_tweet_id", 0)),
                    customer_text=str(ev.get("customer_text", "")),
                    brand_tweet_id=int(ev.get("brand_tweet_id", 0)),
                    brand_text=str(ev.get("brand_text", "")),
                    inferred_intent=str(ev.get("inferred_intent", "unknown_other")),
                )
            )

        meta = agent_resp.model_metadata or {}
        model_used = str(meta.get("model_used", "unknown"))
        rule_triggered = meta.get("escalation_rule_triggered")
        severity = str(meta.get("escalation_severity", "normal"))
        grounding_summary = str(meta.get("grounding_summary", ""))

        agent_payload = AgentResponsePayload(
            customer_message=agent_resp.customer_message,
            predicted_intent=agent_resp.predicted_intent,
            intent_confidence=agent_resp.intent_confidence,
            intent_uncertainty=agent_resp.intent_uncertainty,
            should_escalate=agent_resp.should_escalate,
            escalation_reason=agent_resp.escalation_reason,
            escalation_rule_triggered=rule_triggered,
            escalation_severity=severity,
            draft_response=agent_resp.response,
            grounding_summary=grounding_summary,
            cited_evidence_ids=agent_resp.evidence_source_ids,
            retrieved_evidence=evidence_items,
            model_used=model_used,
            requires_clarification=False,
        )

        return ChatResponse(
            status="success",
            data=ChatResponseData(
                response=agent_payload,
                review=review_payload,
                execution_time_ms=round(t_elapsed, 2),
            ),
        )

    except GroqGenerationError as err:
        logger.error("Groq Generation Error: %s", str(err))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upstream AI generation service is temporarily unavailable.",
        )
    except Exception as err:
        logger.exception("Unexpected error processing message: %s", str(err))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the inquiry.",
        )
