"""
tests/test_agent_reviewer.py
============================
Phase 13 — Agent Evaluation & Reviewer Layer Test Suite.

Verifies:
1. High-quality grounded responses pass with PASS verdict and high scores.
2. Hallucinated refund promises, pricing claims, and fabricated actions are detected and failed.
3. Invented non-evidence URLs are flagged.
4. Weak grounding / missing evidence is flagged.
5. Intent drift / cross-domain contradiction is detected.
6. Inappropriate non-escalation (critical safety / legal hazard) triggers immediate FAIL.
7. Unnecessary escalation of routine inquiries is flagged.
8. Truncated or empty responses fail completeness.
9. Unprofessional / rude language is penalized.
10. Response immutability: AgentResponse is never mutated by reviewer.
11. Mocked hybrid LLM reviewer invocation, score blending, and graceful error fallback.
12. Zero secret leakage in EvaluationResult.
13. Golden Set isolation preservation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.evaluation.agent_reviewer import (
    AgentReviewer,
    EvaluationDimension,
    EvaluationIssue,
    EvaluationResult,
    ReviewDecision,
)
from src.generation.agent_orchestrator import AgentResponse
from src.generation.config import GroqConfig
from src.generation.groq_client import GroqGenerationError

GOLDEN_CSV = Path("data/processed/apple_support/apple_support_intent_golden_set.csv")
FAKE_API_KEY = "gsk_test_fake_api_key_00000000000000000000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_battery_response() -> AgentResponse:
    """A high-quality, grounded, non-escalated AgentResponse."""
    return AgentResponse(
        customer_message="My battery drains fast on iOS 11 after the update",
        predicted_intent="battery_power",
        intent_confidence=0.95,
        intent_uncertainty=0.05,
        response="We want your battery to last! Take a look at your battery usage in Settings > Battery, and let us know what you see.",
        evidence=[
            {
                "customer_tweet_id": 555001,
                "customer_text": "My battery drains in 30 minutes after updating to iOS 11",
                "brand_tweet_id": 555002,
                "brand_text": "@user Take a look at your battery usage in Settings > Battery.",
                "similarity_score": 0.88,
                "inferred_intent": "battery_power",
            }
        ],
        evidence_source_ids=[555001],
        should_escalate=False,
        escalation_reason=None,
        grounding_score=0.88,
        model_metadata={
            "orchestrator_version": "1.0",
            "model_used": "qwen/qwen3.8-27b",
            "top_similarity_score": 0.88,
        },
    )


@pytest.fixture
def mock_groq_reviewer_client():
    """Mock Groq client returning structured JSON evaluation."""
    mock_client = MagicMock()
    mock_client.config = GroqConfig(api_key=FAKE_API_KEY, model="qwen/qwen3.8-27b")
    mock_client.generate_json_response.return_value = {
        "intent_consistency_score": 0.95,
        "response_relevance_score": 0.92,
        "grounding_support_score": 0.90,
        "hallucination_risk_score": 1.00,
        "response_completeness_score": 0.95,
        "professional_quality_score": 0.98,
        "detected_issues": [],
        "rationale": "Response is grounded, courteous, and accurately addresses battery drain.",
    }
    return mock_client


# ---------------------------------------------------------------------------
# 1. Quality & Passing Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerPassingCases:
    def test_high_quality_grounded_response_passes(self, clean_battery_response):
        reviewer = AgentReviewer()
        result: EvaluationResult = reviewer.evaluate(
            customer_message=clean_battery_response.customer_message,
            agent_response=clean_battery_response,
            use_llm=False,
        )

        assert result.passed is True
        assert result.decision == ReviewDecision.PASS.value
        assert result.overall_score >= 0.75
        assert result.dimension_scores[EvaluationDimension.ESCALATION_CORRECTNESS.value] == 1.0
        assert result.dimension_scores[EvaluationDimension.HALLUCINATION_RISK.value] == 1.0
        assert result.dimension_scores[EvaluationDimension.GROUNDING_SUPPORT.value] >= 0.70
        assert result.dimension_scores[EvaluationDimension.INTENT_CONSISTENCY.value] >= 0.90
        assert result.reviewer_mode == "deterministic"

    def test_properly_escalated_response_passes(self):
        """Legitimate safety emergency properly escalated by agent should pass."""
        safety_response = AgentResponse(
            customer_message="My phone charger caught fire and burned my desk!",
            predicted_intent="unknown_other",
            intent_confidence=0.85,
            intent_uncertainty=0.15,
            response="This is an urgent matter. Please disconnect the device and contact Apple Support immediately at support.apple.com.",
            evidence=[],
            evidence_source_ids=[],
            should_escalate=True,
            escalation_reason="Physical hazard or safety emergency reported in customer message",
            grounding_score=1.0,
            model_metadata={"top_similarity_score": 0.40, "model_used": "qwen/qwen3.8-27b"},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=safety_response.customer_message,
            agent_response=safety_response,
        )

        assert result.passed is True
        assert result.decision == ReviewDecision.PASS.value
        assert result.dimension_scores[EvaluationDimension.ESCALATION_CORRECTNESS.value] == 1.0
        assert result.escalation_consistency["is_consistent"] is True


# ---------------------------------------------------------------------------
# 2. Hallucination & Policy Violation Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerHallucinationDetection:
    def test_hallucinated_refund_guarantee_fails(self, clean_battery_response):
        """Unauthorized refund promise must trigger critical issue and fail."""
        bad_response = AgentResponse(
            customer_message=clean_battery_response.customer_message,
            predicted_intent="battery_power",
            intent_confidence=0.95,
            intent_uncertainty=0.05,
            response="We are sorry! We guarantee a full refund will be processed to your card.",
            evidence=clean_battery_response.evidence,
            evidence_source_ids=[555001],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.50,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=bad_response.customer_message,
            agent_response=bad_response,
        )

        assert result.passed is False
        assert result.decision == ReviewDecision.FAIL.value
        assert result.dimension_scores[EvaluationDimension.HALLUCINATION_RISK.value] <= 0.20
        # Critical severity issue recorded
        severities = [iss["severity"] for iss in result.detected_issues]
        assert "critical" in severities

    def test_fabricated_transaction_action_penalized(self, clean_battery_response):
        """Claiming the bot performed an account action ('I have reset your password') is penalized."""
        bad_response = AgentResponse(
            customer_message="Forgot my password",
            predicted_intent="account_access",
            intent_confidence=0.95,
            intent_uncertainty=0.05,
            response="I have reset your password and unlocked your account! Check your email.",
            evidence=[],
            evidence_source_ids=[],
            should_escalate=True,  # Account access requires escalation
            escalation_reason="Account security and authentication credentials require verified human triage",
            grounding_score=0.50,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=bad_response.customer_message,
            agent_response=bad_response,
        )

        assert result.dimension_scores[EvaluationDimension.HALLUCINATION_RISK.value] <= 0.30
        severities = [iss["severity"] for iss in result.detected_issues]
        assert "high" in severities

    def test_unauthorized_price_claim_penalized(self, clean_battery_response):
        """Fabricated repair price not present in evidence is penalized."""
        bad_response = AgentResponse(
            customer_message=clean_battery_response.customer_message,
            predicted_intent="battery_power",
            intent_confidence=0.95,
            intent_uncertainty=0.05,
            response="A new battery replacement will cost $29 at your local Apple Store.",
            evidence=clean_battery_response.evidence,  # Evidence has no '$29'
            evidence_source_ids=[555001],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.60,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=bad_response.customer_message,
            agent_response=bad_response,
        )

        assert result.dimension_scores[EvaluationDimension.HALLUCINATION_RISK.value] <= 0.30
        assert any("pricing" in iss["description"].lower() for iss in result.detected_issues)

    def test_invented_url_penalized(self, clean_battery_response):
        """URL not present in historical evidence is flagged."""
        bad_response = AgentResponse(
            customer_message=clean_battery_response.customer_message,
            predicted_intent="battery_power",
            intent_confidence=0.95,
            intent_uncertainty=0.05,
            response="Check this guide: https://unverified-third-party-fixes.com/battery",
            evidence=clean_battery_response.evidence,
            evidence_source_ids=[555001],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.50,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=bad_response.customer_message,
            agent_response=bad_response,
        )

        assert result.dimension_scores[EvaluationDimension.HALLUCINATION_RISK.value] <= 0.50
        assert any("url" in iss["description"].lower() for iss in result.detected_issues)


# ---------------------------------------------------------------------------
# 3. Grounding & Evidence Support Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerGrounding:
    def test_zero_evidence_on_non_escalated_response_penalized(self):
        """Non-escalated response with no evidence attached is penalized."""
        resp = AgentResponse(
            customer_message="My screen is black",
            predicted_intent="device_hardware",
            intent_confidence=0.90,
            intent_uncertainty=0.10,
            response="Try holding the power button for 10 seconds to force restart.",
            evidence=[],
            evidence_source_ids=[],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.0,
            model_metadata={"top_similarity_score": 0.70},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=resp.customer_message,
            agent_response=resp,
        )

        assert result.dimension_scores[EvaluationDimension.GROUNDING_SUPPORT.value] <= 0.30
        assert any("evidence" in iss["description"].lower() for iss in result.detected_issues)

    def test_weak_evidence_overlap_penalized(self, clean_battery_response):
        """Response whose tokens completely diverge from evidence receives low grounding score."""
        resp = AgentResponse(
            customer_message=clean_battery_response.customer_message,
            predicted_intent="battery_power",
            intent_confidence=0.90,
            intent_uncertainty=0.10,
            # Evidence talks about 'Settings > Battery', but response talks about unrelated nonsense
            response="Make sure your refrigerator door is closed properly so the bluetooth sync works.",
            evidence=clean_battery_response.evidence,
            evidence_source_ids=[555001],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.20,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=resp.customer_message,
            agent_response=resp,
        )

        assert result.dimension_scores[EvaluationDimension.GROUNDING_SUPPORT.value] <= 0.50


# ---------------------------------------------------------------------------
# 4. Intent Consistency & Topic Drift Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerIntentConsistency:
    def test_cross_domain_intent_drift_detected(self):
        """Customer has battery issue, but response gives subscription billing advice."""
        resp = AgentResponse(
            customer_message="My iPhone battery drains in 2 hours",
            predicted_intent="battery_power",
            intent_confidence=0.92,
            intent_uncertainty=0.08,
            # Response drifts completely to billing/subscription
            response="To cancel your subscription and view your receipt, go to billing and request a refund.",
            evidence=[{"brand_text": "@user check settings battery", "similarity_score": 0.8}],
            evidence_source_ids=[100],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.40,
            model_metadata={"top_similarity_score": 0.80},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=resp.customer_message,
            agent_response=resp,
        )

        assert result.dimension_scores[EvaluationDimension.INTENT_CONSISTENCY.value] <= 0.40
        assert any("intent drift" in iss["description"].lower() for iss in result.detected_issues)


# ---------------------------------------------------------------------------
# 5. Escalation Correctness Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerEscalationAudit:
    def test_missed_safety_hazard_escalation_causes_critical_fail(self):
        """Customer reported fire/smoke; agent erroneously did NOT escalate -> Critical FAIL."""
        bad_response = AgentResponse(
            customer_message="My phone battery is smoking and smelling like burnt plastic!",
            predicted_intent="unknown_other",
            intent_confidence=0.50,
            intent_uncertainty=0.50,
            response="Try charging it with a different cable and see if that helps.",
            evidence=[],
            evidence_source_ids=[],
            should_escalate=False,  # ERRONEOUS: Must escalate!
            escalation_reason=None,
            grounding_score=0.40,
            model_metadata={"top_similarity_score": 0.30},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=bad_response.customer_message,
            agent_response=bad_response,
        )

        assert result.passed is False
        assert result.decision == ReviewDecision.FAIL.value
        assert result.dimension_scores[EvaluationDimension.ESCALATION_CORRECTNESS.value] == 0.0
        assert any(iss["severity"] == "critical" for iss in result.detected_issues)

    def test_missed_legal_threat_escalation_causes_critical_fail(self):
        """Customer threatened legal action; agent failed to escalate -> Critical FAIL."""
        bad_response = AgentResponse(
            customer_message="I have hired a lawyer and will be suing Apple for this fraud",
            predicted_intent="complaint_feedback",
            intent_confidence=0.85,
            intent_uncertainty=0.15,
            response="We are sorry to hear that. What model iPhone do you have?",
            evidence=[],
            evidence_source_ids=[],
            should_escalate=False,  # ERRONEOUS: Must escalate!
            escalation_reason=None,
            grounding_score=0.40,
            model_metadata={"top_similarity_score": 0.40},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=bad_response.customer_message,
            agent_response=bad_response,
        )

        assert result.passed is False
        assert result.decision == ReviewDecision.FAIL.value
        assert result.dimension_scores[EvaluationDimension.ESCALATION_CORRECTNESS.value] == 0.0

    def test_unnecessary_escalation_of_routine_query_flagged(self, clean_battery_response):
        """Routine inquiry unnecessarily marked as escalated."""
        unnecessary_esc_response = AgentResponse(
            customer_message=clean_battery_response.customer_message,
            predicted_intent="battery_power",
            intent_confidence=0.95,
            intent_uncertainty=0.05,
            response="Please contact Apple Support directly.",
            evidence=[],
            evidence_source_ids=[],
            should_escalate=True,  # Unnecessary escalation
            escalation_reason="Agent forced escalation",
            grounding_score=0.50,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=unnecessary_esc_response.customer_message,
            agent_response=unnecessary_esc_response,
        )

        assert result.dimension_scores[EvaluationDimension.ESCALATION_CORRECTNESS.value] == 0.60
        assert any("unnecessarily escalated" in iss["description"].lower() for iss in result.detected_issues)


# ---------------------------------------------------------------------------
# 6. Completeness & Professional Tone Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerToneAndCompleteness:
    def test_truncated_response_penalized(self, clean_battery_response):
        """Response with under 5 words is penalized for completeness."""
        truncated_response = AgentResponse(
            customer_message=clean_battery_response.customer_message,
            predicted_intent="battery_power",
            intent_confidence=0.95,
            intent_uncertainty=0.05,
            response="Restart phone.",
            evidence=clean_battery_response.evidence,
            evidence_source_ids=[555001],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.50,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=truncated_response.customer_message,
            agent_response=truncated_response,
        )

        assert result.dimension_scores[EvaluationDimension.RESPONSE_COMPLETENESS.value] <= 0.30

    def test_unprofessional_rude_language_penalized(self, clean_battery_response):
        """Rude phrase ('deal with it') is penalized in professional quality."""
        rude_response = AgentResponse(
            customer_message=clean_battery_response.customer_message,
            predicted_intent="battery_power",
            intent_confidence=0.95,
            intent_uncertainty=0.05,
            response="Whatever, deal with it yourself. Check your settings.",
            evidence=clean_battery_response.evidence,
            evidence_source_ids=[555001],
            should_escalate=False,
            escalation_reason=None,
            grounding_score=0.50,
            model_metadata={"top_similarity_score": 0.88},
        )

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=rude_response.customer_message,
            agent_response=rude_response,
        )

        assert result.dimension_scores[EvaluationDimension.PROFESSIONAL_QUALITY.value] <= 0.25
        assert any("unprofessional" in iss["description"].lower() for iss in result.detected_issues)


# ---------------------------------------------------------------------------
# 7. Non-Mutation & Immutability Guarantee
# ---------------------------------------------------------------------------
class TestAgentReviewerImmutability:
    def test_original_agent_response_remains_completely_unaltered(self, clean_battery_response):
        """Reviewer must NEVER alter the AgentResponse input."""
        before_dict = clean_battery_response.to_dict()
        before_resp_str = clean_battery_response.response

        reviewer = AgentReviewer()
        result = reviewer.evaluate(
            customer_message=clean_battery_response.customer_message,
            agent_response=clean_battery_response,
        )

        assert clean_battery_response.response == before_resp_str
        assert clean_battery_response.to_dict() == before_dict
        assert isinstance(result, EvaluationResult)


# ---------------------------------------------------------------------------
# 8. Mocked Hybrid LLM Reviewer Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerHybridLLM:
    def test_mocked_llm_reviewer_successful_evaluation(
        self, clean_battery_response, mock_groq_reviewer_client
    ):
        reviewer = AgentReviewer(groq_client=mock_groq_reviewer_client)
        result = reviewer.evaluate(
            customer_message=clean_battery_response.customer_message,
            agent_response=clean_battery_response,
            use_llm=True,
        )

        assert result.reviewer_mode == "hybrid_llm"
        assert result.model_used == "qwen/qwen3.8-27b"
        assert result.passed is True
        assert "LLM reviewer" in result.reviewer_rationale
        # Verify LLM was actually invoked
        mock_groq_reviewer_client.generate_json_response.assert_called_once()

    def test_mocked_llm_reviewer_graceful_error_fallback(
        self, clean_battery_response, mock_groq_reviewer_client
    ):
        """When Groq client raises an error, reviewer falls back to deterministic without crashing."""
        mock_groq_reviewer_client.generate_json_response.side_effect = GroqGenerationError(
            "Rate limit reached", error_type="rate_limit"
        )

        reviewer = AgentReviewer(groq_client=mock_groq_reviewer_client)
        result = reviewer.evaluate(
            customer_message=clean_battery_response.customer_message,
            agent_response=clean_battery_response,
            use_llm=True,
        )

        # Fallback to deterministic mode cleanly
        assert result.reviewer_mode == "deterministic"
        assert result.passed is True
        assert result.overall_score >= 0.75


# ---------------------------------------------------------------------------
# 9. Security & Secret Safety Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerSecurity:
    def test_no_api_key_leakage_in_evaluation_result(self, clean_battery_response):
        reviewer = AgentReviewer(config=GroqConfig(api_key=FAKE_API_KEY, model="qwen/qwen3.8-27b"))
        result = reviewer.evaluate(
            customer_message=clean_battery_response.customer_message,
            agent_response=clean_battery_response,
        )

        result_json = json.dumps(result.to_dict())
        assert FAKE_API_KEY not in result_json
        assert "gsk_" not in result_json


# ---------------------------------------------------------------------------
# 10. Golden Set Isolation Tests
# ---------------------------------------------------------------------------
class TestAgentReviewerGoldenSetIsolation:
    def test_golden_set_not_modified_or_accessed(self, clean_battery_response):
        """Reviewer operations do not mutate or contaminate Golden Evaluation Set."""
        assert GOLDEN_CSV.exists()
        golden_df = pd.read_csv(GOLDEN_CSV)
        initial_count = len(golden_df)
        assert initial_count == 158

        reviewer = AgentReviewer()
        for idx, row in golden_df.head(5).iterrows():
            mock_resp = AgentResponse(
                customer_message=str(row["text"]),
                predicted_intent=str(row["golden_intent"]),
                intent_confidence=0.90,
                intent_uncertainty=0.10,
                response="Please reach out to Apple Support directly via DM.",
                evidence=[],
                evidence_source_ids=[],
                should_escalate=False,
                escalation_reason=None,
                grounding_score=0.80,
                model_metadata={"top_similarity_score": 0.85},
            )
            res = reviewer.evaluate(customer_message=str(row["text"]), agent_response=mock_resp)
            assert isinstance(res, EvaluationResult)

        # Confirm golden set unchanged
        golden_df_after = pd.read_csv(GOLDEN_CSV)
        assert len(golden_df_after) == initial_count
