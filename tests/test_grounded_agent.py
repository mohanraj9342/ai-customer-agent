"""
tests/test_grounded_agent.py
============================
Unit and mock tests for Phase 12 Grounded Agent Response Generation & Orchestration.

Tests:
1. Environment configuration loading and key masking.
2. Missing key and missing model error handling.
3. Isolated Groq client mocking and API error handling (timeout, auth, rate-limit, malformed JSON).
4. Deterministic escalation and safety rule evaluation.
5. Strict response schema validation.
6. Golden Set non-contamination across retrieved evidence.
7. Zero API key leakage in logs, representations, and response outputs.
8. End-to-end orchestration with mocked Groq completions and fallback behavior.

NOTE: Real external Groq API calls are NEVER made in unit tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from groq import APITimeoutError, AuthenticationError, RateLimitError

from src.generation.agent_orchestrator import (
    AgentResponse,
    GroundedSupportAgent,
    IntentPredictor,
)
from src.generation.config import GroqConfig, load_groq_config
from src.generation.escalation_rules import EscalationDecision, EscalationEngine
from src.generation.groq_client import GroqGenerationClient, GroqGenerationError
from src.generation.prompt_builder import GroundedPromptBuilder
from src.retrieval.historical_response_retriever import (
    HistoricalResponseRetriever,
    RetrievalResult,
)

FAKE_API_KEY = "gsk_test_fake_api_key_1234567890abcdef"
GOLDEN_CSV = "data/processed/apple_support/apple_support_intent_golden_set.csv"


# ---------------------------------------------------------------------------
# 1. Configuration and Key Masking Tests
# ---------------------------------------------------------------------------
class TestGroqConfig:
    def test_load_groq_config_valid(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", FAKE_API_KEY)
        monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        cfg = load_groq_config()
        assert cfg.api_key == FAKE_API_KEY
        assert cfg.model == "llama-3.3-70b-versatile"
        assert cfg.timeout_seconds == 30.0

    def test_load_groq_config_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
            load_groq_config(api_key="")

    def test_load_groq_config_missing_model_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", FAKE_API_KEY)
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        with pytest.raises(ValueError, match="GROQ_MODEL is not configured"):
            load_groq_config(api_key=FAKE_API_KEY, model="")

    def test_api_key_masked_in_repr_and_metadata(self):
        cfg = GroqConfig(api_key=FAKE_API_KEY, model="test-model")
        # Repr must not reveal the key
        assert FAKE_API_KEY not in repr(cfg)
        assert "***configured***" in repr(cfg)

        # safe_metadata must not contain the key string
        meta = cfg.safe_metadata()
        assert "api_key" not in meta
        assert meta["api_key_configured"] is True
        assert FAKE_API_KEY not in str(meta)


# ---------------------------------------------------------------------------
# 2. Mocked Groq Client Error Handling Tests
# ---------------------------------------------------------------------------
class TestGroqGenerationClient:
    def test_successful_json_generation(self):
        mock_raw_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "draft_response": "We can help you inspect your battery health in Settings > Battery.",
            "cited_evidence_ids": [2079208],
            "grounding_summary": "Guided by historical battery diagnostic steps.",
            "requires_clarification": False,
        })
        mock_raw_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        cfg = GroqConfig(api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile")
        client = GroqGenerationClient(config=cfg, client=mock_raw_client)

        resp = client.generate_json_response([{"role": "user", "content": "test"}])
        assert "draft_response" in resp
        assert "Settings > Battery" in resp["draft_response"]
        assert resp["cited_evidence_ids"] == [2079208]

    def test_timeout_error_handling(self):
        mock_raw_client = MagicMock()
        mock_raw_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

        cfg = GroqConfig(api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile", timeout_seconds=5.0)
        client = GroqGenerationClient(config=cfg, client=mock_raw_client)

        with pytest.raises(GroqGenerationError) as exc_info:
            client.generate_json_response([{"role": "user", "content": "test"}])
        assert exc_info.value.error_type == "timeout_error"

    def test_authentication_error_handling(self):
        mock_raw_client = MagicMock()
        mock_raw_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API key", response=MagicMock(), body=None
        )

        cfg = GroqConfig(api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile")
        client = GroqGenerationClient(config=cfg, client=mock_raw_client)

        with pytest.raises(GroqGenerationError) as exc_info:
            client.generate_json_response([{"role": "user", "content": "test"}])
        assert exc_info.value.error_type == "authentication_error"
        assert exc_info.value.status_code == 401

    def test_rate_limit_error_handling(self):
        mock_raw_client = MagicMock()
        mock_raw_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit reached", response=MagicMock(), body=None
        )

        cfg = GroqConfig(api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile")
        client = GroqGenerationClient(config=cfg, client=mock_raw_client)

        with pytest.raises(GroqGenerationError) as exc_info:
            client.generate_json_response([{"role": "user", "content": "test"}])
        assert exc_info.value.error_type == "rate_limit_error"
        assert exc_info.value.status_code == 429

    def test_malformed_json_error_handling(self):
        mock_raw_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is not valid JSON at all!"
        mock_raw_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        cfg = GroqConfig(api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile")
        client = GroqGenerationClient(config=cfg, client=mock_raw_client)

        with pytest.raises(GroqGenerationError) as exc_info:
            client.generate_json_response([{"role": "user", "content": "test"}])
        assert exc_info.value.error_type == "malformed_json"


# ---------------------------------------------------------------------------
# 3. Deterministic Safety & Escalation Rules Tests
# ---------------------------------------------------------------------------
class TestEscalationRules:
    def test_safety_hazard_triggers_critical_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="My phone battery started smoking and exploded!",
            predicted_intent="battery_power",
        )
        assert res.should_escalate is True
        assert res.severity == "critical"
        assert res.rule_triggered == "safety_hazard_alert"

    def test_legal_threat_triggers_critical_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="I have contacted my lawyer and we are taking Apple to court!",
            predicted_intent="complaint_feedback",
        )
        assert res.should_escalate is True
        assert res.severity == "critical"
        assert res.rule_triggered == "legal_fraud_alert"

    def test_account_access_intent_triggers_high_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="Locked out of my Apple ID and cannot reset password",
            predicted_intent="account_access",
        )
        assert res.should_escalate is True
        assert res.severity == "high"
        assert res.rule_triggered == "account_security_escalation"

    def test_high_value_refund_triggers_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="I demand a full refund of £3,250 for my broken MacBook!",
            predicted_intent="billing_payment",
        )
        assert res.should_escalate is True
        assert res.severity == "high"
        assert res.rule_triggered == "high_value_billing_dispute"

    def test_low_intent_confidence_triggers_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="Something is wrong with my device",
            predicted_intent="unknown_other",
            intent_confidence=0.45,  # Below 0.60
        )
        assert res.should_escalate is True
        assert res.rule_triggered == "low_intent_confidence"

    def test_weak_retrieval_grounding_triggers_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="What is the weather in Paris today?",
            predicted_intent="unknown_other",
            intent_confidence=0.90,
            top_retrieval_similarity=0.32,  # Below 0.50
        )
        assert res.should_escalate is True
        assert res.rule_triggered == "weak_retrieval_grounding"

    def test_vague_short_query_triggers_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="@AppleSupport help",
            predicted_intent="unknown_other",
            intent_confidence=0.85,
            top_retrieval_similarity=0.75,
        )
        assert res.should_escalate is True
        assert res.rule_triggered == "vague_short_query"

    def test_standard_clear_query_passes_without_escalation(self):
        res = EscalationEngine.evaluate(
            customer_text="My iPhone 7 battery is draining fast after updating to iOS 11",
            predicted_intent="battery_power",
            intent_confidence=0.95,
            intent_margin=0.90,
            top_retrieval_similarity=0.78,
        )
        assert res.should_escalate is False
        assert res.escalation_reason is None
        assert res.severity == "normal"


# ---------------------------------------------------------------------------
# 4. Prompt Builder Tests
# ---------------------------------------------------------------------------
class TestPromptBuilder:
    def test_build_prompt_structure(self):
        evidence = [
            RetrievalResult(
                rank=1,
                similarity_score=0.82,
                thread_id="t_1",
                customer_tweet_id=111,
                customer_text="battery drain query",
                brand_tweet_id=222,
                brand_text="check settings battery",
                inferred_intent="battery_power",
            )
        ]
        messages = GroundedPromptBuilder.build_prompt(
            customer_message="My battery dies quickly",
            predicted_intent="battery_power",
            intent_confidence=0.95,
            evidence=evidence,
            should_escalate=False,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "battery drain query" in messages[1]["content"]
        assert "check settings battery" in messages[1]["content"]
        assert "111" in messages[1]["content"]


# ---------------------------------------------------------------------------
# 5. End-to-End Orchestration & Safety Tests (Mocked Groq)
# ---------------------------------------------------------------------------
class TestGroundedSupportAgent:
    @pytest.fixture
    def mock_agent(self):
        # Mock retriever returning 1 verified piece of evidence
        mock_retriever = MagicMock()
        mock_retriever.corpus_size = 81943
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                rank=1,
                similarity_score=0.85,
                thread_id="t_bat",
                customer_tweet_id=555001,
                customer_text="My battery dies in 30 minutes after updating to iOS 11",
                brand_tweet_id=555002,
                brand_text="@user Take a look at your battery usage in Settings > Battery.",
                inferred_intent="battery_power",
            )
        ]

        # Mock intent predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ("battery_power", 0.96, 0.92)

        # Mock Groq client
        mock_groq = MagicMock()
        mock_groq.config = GroqConfig(api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile")
        mock_groq.generate_json_response.return_value = {
            "draft_response": "We want your battery to last as expected! Take a look in Settings > Battery.",
            "cited_evidence_ids": [555001],
            "grounding_summary": "Advised customer to inspect Settings > Battery based on verified case.",
            "requires_clarification": False,
        }

        return GroundedSupportAgent(
            retriever=mock_retriever,
            groq_client=mock_groq,
            intent_predictor=mock_predictor,
        )

    def test_process_message_success_schema(self, mock_agent):
        query = "My battery is draining very fast on my iPhone after the new update"
        agent_response: AgentResponse = mock_agent.process_message(query)

        assert agent_response.customer_message == query
        assert agent_response.predicted_intent == "battery_power"
        assert agent_response.intent_confidence == 0.96
        assert agent_response.should_escalate is False
        assert "Settings > Battery" in agent_response.response
        assert agent_response.evidence_source_ids == [555001]
        assert agent_response.grounding_score == 0.85
        assert agent_response.model_metadata["orchestrator_version"] == "1.0"

        # Verify to_dict serialization
        data = agent_response.to_dict()
        assert "customer_message" in data
        assert "predicted_intent" in data
        assert "response" in data

    def test_no_api_key_leakage_in_response(self, mock_agent):
        agent_response = mock_agent.process_message("My battery dies in 30 minutes")
        data = agent_response.to_dict()
        data_str = json.dumps(data)

        assert FAKE_API_KEY not in data_str
        assert "gsk_" not in data_str

    def test_fallback_on_groq_failure(self, mock_agent):
        # Simulate Groq client raising a timeout error
        mock_agent.groq_client.generate_json_response.side_effect = GroqGenerationError(
            "API timed out", error_type="timeout_error"
        )

        agent_response = mock_agent.process_message("My battery dies in 30 minutes")
        assert agent_response.response is not None
        assert len(agent_response.response) > 10
        # Fallback uses historical brand reply text cleanly
        assert "Settings > Battery" in agent_response.response
        assert "Fallback generated directly" in agent_response.model_metadata["grounding_summary"]

    def test_golden_set_isolation_in_evidence(self):
        """Verify that real retriever never returns Golden Set records."""
        golden_df = pd.read_csv(GOLDEN_CSV)
        golden_ids = set(golden_df["tweet_id"].astype(int))

        retriever = HistoricalResponseRetriever()
        results = retriever.retrieve("My battery drains in 20 minutes", top_k=10)

        for res in results:
            assert res.customer_tweet_id not in golden_ids
            assert res.brand_tweet_id not in golden_ids
