"""
tests/test_api.py
=================
Phase 15 — Production API Test Suite for Grounded AI Customer Support Agent.

Verifies:
1. Health check liveness probe (/health).
2. Readiness check probe (/ready) with model and index statuses.
3. CORS headers and preflight requests for Vercel production and preview domains.
4. End-to-end chat endpoint (/api/chat) on routine inquiries with mocked Groq responses.
5. Critical safety escalation handling via API.
6. Request validation (empty text, whitespace, oversized payload).
7. Zero secret leakage across all responses and headers.
8. Strict Golden Set isolation via API retrieval evidence.
9. Upstream Groq error handling and RFC-compliant error envelopes.
10. Backward-compatible endpoint alias (/api/support/message).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import ApiConfig
from src.evaluation.agent_reviewer import AgentReviewer
from src.generation.agent_orchestrator import GroundedSupportAgent, IntentPredictor
from src.generation.config import GroqConfig
from src.generation.groq_client import GroqGenerationClient, GroqGenerationError
from src.retrieval.historical_response_retriever import HistoricalResponseRetriever

GOLDEN_CSV = Path("data/processed/apple_support/apple_support_intent_golden_set.csv")
FAKE_API_KEY = "gsk_test_fake_api_key_1234567890abcdef"


@pytest.fixture(scope="module")
def mock_groq_client():
    """Isolated mock GroqGenerationClient returning strict JSON."""
    raw_mock = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "draft_response": "We can help you inspect your battery health in Settings > Battery.",
        "cited_evidence_ids": [2079208],
        "grounding_summary": "Guided by historical battery diagnostic steps in evidence.",
        "requires_clarification": False,
    })
    raw_mock.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    cfg = GroqConfig(api_key=FAKE_API_KEY, model="qwen/qwen3.8-27b")
    client = GroqGenerationClient(config=cfg, client=raw_mock)
    return client


@pytest.fixture(scope="module")
def test_app(mock_groq_client):
    """FastAPI test app configured with shared test instances."""
    retriever = HistoricalResponseRetriever()
    intent_predictor = IntentPredictor()
    agent = GroundedSupportAgent(
        retriever=retriever,
        groq_client=mock_groq_client,
        intent_predictor=intent_predictor,
    )
    reviewer = AgentReviewer()
    api_config = ApiConfig(
        cors_origins=["http://localhost:3000", "https://my-app.vercel.app"],
        cors_origin_regex=r"^https://.*\.vercel\.app$",
    )
    return create_app(agent=agent, reviewer=reviewer, api_config=api_config)


@pytest.fixture(scope="module")
def client(test_app):
    """TestClient instance."""
    with TestClient(test_app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Health & Readiness Tests
# ---------------------------------------------------------------------------
class TestHealthAndReadiness:
    def test_health_endpoint(self, client):
        """GET /health returns 200 OK and valid service metadata."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "apple-support-ai-agent"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_readiness_endpoint(self, client):
        """GET /ready verifies models, retriever, and Groq configuration are loaded."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "degraded")
        assert data["intent_classifier_ready"] is True
        assert data["retriever_ready"] is True
        assert data["retriever_corpus_size"] > 80000
        assert data["groq_configured"] is True
        assert data["groq_model"] == "qwen/qwen3.8-27b"
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# 2. CORS & Vercel Integration Tests
# ---------------------------------------------------------------------------
class TestCorsConfiguration:
    def test_cors_preflight_production_vercel_domain(self, client):
        """OPTIONS /api/chat handles preflight for production Vercel frontend."""
        headers = {
            "Origin": "https://my-app.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        resp = client.options("/api/chat", headers=headers)
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://my-app.vercel.app"
        assert "POST" in resp.headers.get("access-control-allow-methods", "")

    def test_cors_preflight_preview_vercel_domain_regex(self, client):
        """OPTIONS /api/chat handles preview branch deployments via wildcard regex."""
        headers = {
            "Origin": "https://my-app-git-feat-phase15-team.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        resp = client.options("/api/chat", headers=headers)
        assert resp.status_code == 200
        assert (
            resp.headers.get("access-control-allow-origin")
            == "https://my-app-git-feat-phase15-team.vercel.app"
        )

    def test_cors_preflight_local_development(self, client):
        """OPTIONS /api/chat handles local frontend development."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
        resp = client.options("/api/chat", headers=headers)
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ---------------------------------------------------------------------------
# 3. Chat Processing & Grounded Response Generation Tests
# ---------------------------------------------------------------------------
class TestChatEndpoints:
    def test_chat_routine_inquiry(self, client):
        """POST /api/chat processes routine inquiry, returns agent response & review."""
        payload = {
            "message": "My iPhone 13 battery dies after only 2 hours of use.",
            "top_k_evidence": 3,
            "include_review": True,
            "review_mode": "deterministic",
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200
        res_json = resp.json()
        assert res_json["status"] == "success"

        data = res_json["data"]
        # Agent response verification
        agent_resp = data["response"]
        assert agent_resp["predicted_intent"] == "battery_power"
        assert agent_resp["intent_confidence"] > 0.5
        assert agent_resp["should_escalate"] is False
        assert "draft_response" in agent_resp
        assert len(agent_resp["draft_response"]) > 10
        assert len(agent_resp["retrieved_evidence"]) <= 3
        assert agent_resp["model_used"] == "qwen/qwen3.8-27b"

        # Reviewer verification
        review = data["review"]
        assert review is not None
        assert review["decision"] in ("PASS", "NEEDS_HUMAN_REVIEW", "FAIL")
        assert 0.0 <= review["overall_score"] <= 1.0
        assert "grounding_support" in review["dimension_scores"]

        # Latency verification
        assert data["execution_time_ms"] > 0

    def test_chat_critical_safety_escalation(self, client):
        """POST /api/chat triggers immediate escalation on physical safety hazards."""
        payload = {
            "message": "My iPhone battery swollen and exploded in smoke and fire!",
            "top_k_evidence": 3,
            "include_review": True,
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()["data"]["response"]

        assert data["should_escalate"] is True
        assert data["escalation_severity"] == "critical"
        assert data["escalation_rule_triggered"] == "safety_hazard_alert"
        assert "immediate specialized assistance" in data["draft_response"]

    def test_chat_backward_compatible_endpoint_alias(self, client):
        """POST /api/support/message alias behaves identically to /api/chat."""
        payload = {"message": "iOS software update keeps failing with error code"}
        resp = client.post("/api/support/message", json=payload)
        assert resp.status_code == 200
        data = resp.json()["data"]["response"]
        assert data["predicted_intent"] == "software_update"


# ---------------------------------------------------------------------------
# 4. Request Validation & Error Handling Tests
# ---------------------------------------------------------------------------
class TestValidationAndErrorHandling:
    def test_empty_message_validation(self, client):
        """Empty string returns 422 Unprocessable Entity."""
        resp = client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 422
        err = resp.json()
        assert err["status"] == "error"
        assert err["code"] == "VALIDATION_ERROR"

    def test_whitespace_only_message_validation(self, client):
        """Whitespace-only string returns 422."""
        resp = client.post("/api/chat", json={"message": "   \n\t  "})
        assert resp.status_code == 422
        err = resp.json()
        assert err["status"] == "error"
        assert err["code"] == "VALIDATION_ERROR"

    def test_oversized_message_validation(self, client):
        """Message exceeding 1000 characters returns 422."""
        resp = client.post("/api/chat", json={"message": "A" * 1001})
        assert resp.status_code == 422
        err = resp.json()
        assert err["status"] == "error"
        assert err["code"] == "VALIDATION_ERROR"

    def test_upstream_groq_error_returns_503(self, test_app):
        """Upstream GroqGenerationError gracefully maps to 503 Service Unavailable."""
        failing_agent = MagicMock()
        failing_agent.process_message.side_effect = GroqGenerationError("Upstream timeout")

        app = create_app(agent=failing_agent, reviewer=test_app.state.reviewer)
        with TestClient(app) as c:
            resp = c.post("/api/chat", json={"message": "My screen is cracked"})
            assert resp.status_code == 503
            err = resp.json()
            assert err["status"] == "error"
            assert "temporarily unavailable" in err["message"]


# ---------------------------------------------------------------------------
# 5. Security & Isolation Verification Tests
# ---------------------------------------------------------------------------
class TestSecurityAndGoldenSetIsolation:
    def test_zero_secret_leakage_in_api_responses(self, client):
        """Responses must NEVER contain raw Groq API keys."""
        resp = client.post(
            "/api/chat",
            json={"message": "Need help with iCloud login"},
        )
        assert resp.status_code == 200
        payload_str = resp.text
        assert "gsk_" not in payload_str
        assert FAKE_API_KEY not in payload_str

        # Check /ready endpoint
        ready_str = client.get("/ready").text
        assert "gsk_" not in ready_str
        assert FAKE_API_KEY not in ready_str

    def test_golden_set_isolation_via_api(self, client):
        """Retrieved evidence served through the API must NEVER intersect with Golden Set."""
        assert GOLDEN_CSV.exists()
        golden_df = pd.read_csv(GOLDEN_CSV)
        golden_ids = set(golden_df["tweet_id"].astype(int))

        resp = client.post(
            "/api/chat",
            json={"message": "Can I cancel my subscription order?"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]["response"]

        evidence_ids = {ev["customer_tweet_id"] for ev in data["retrieved_evidence"]}
        cited_ids = set(data["cited_evidence_ids"])
        all_ids = evidence_ids.union(cited_ids)

        overlap = golden_ids.intersection(all_ids)
        assert len(overlap) == 0, f"CONTAMINATION DETECTED: Golden Set leaked via API: {overlap}"
