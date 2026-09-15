"""
src/generation/agent_orchestrator.py
====================================
Phase 12 — Grounded Customer Support Agent Orchestrator.

Combines:
1. Intent Classification (Predicts domain, confidence, and margin).
2. Historical Semantic Retrieval (Extracts verified customer-brand pairs).
3. Deterministic Safety & Escalation Rules (Hard guardrails for safety, legal, and low confidence).
4. Groq LLM Response Generation (Grounds reply in verified brand actions).
5. Strict Structured Schema Validation (Outputs clean, validated, auditable response objects).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.generation.config import GroqConfig, load_groq_config
from src.generation.escalation_rules import EscalationDecision, EscalationEngine
from src.generation.groq_client import GroqGenerationClient, GroqGenerationError
from src.generation.prompt_builder import GroundedPromptBuilder
from src.retrieval.historical_response_retriever import (
    HistoricalResponseRetriever,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models/intent_classifier")


@dataclass(frozen=True)
class AgentResponse:
    """Strict, standardized output schema for grounded agent responses."""

    customer_message: str
    predicted_intent: str
    intent_confidence: float
    intent_uncertainty: float
    response: str
    evidence: List[Dict[str, Any]]
    evidence_source_ids: List[int]
    should_escalate: bool
    escalation_reason: Optional[str]
    grounding_score: float
    model_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IntentPredictor:
    """Lightweight inference wrapper over the trained intent classifier."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.vec_path = model_dir / "tfidf_vectorizer.joblib"
        self.clf_path = model_dir / "logistic_regression_model.joblib"

        if not self.vec_path.exists() or not self.clf_path.exists():
            logger.warning("Classifier models not found at: %s. Using default fallback.", model_dir)
            self.vectorizer = None
            self.classifier = None
        else:
            self.vectorizer = joblib.load(self.vec_path)
            self.classifier = joblib.load(self.clf_path)

    def predict(self, text: str) -> Tuple[str, float, float]:
        """Predict (intent_name, confidence, margin)."""
        if not text.strip() or self.vectorizer is None or self.classifier is None:
            return "unknown_other", 0.0, 0.0

        x_feat = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(x_feat)[0]
        sorted_indices = probs.argsort()[::-1]

        top_class = str(self.classifier.classes_[sorted_indices[0]])
        top_prob = float(probs[sorted_indices[0]])
        second_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = top_prob - second_prob

        return top_class, round(top_prob, 4), round(margin, 4)


class GroundedSupportAgent:
    """
    End-to-end grounded customer support agent orchestrator.
    """

    def __init__(
        self,
        retriever: Optional[HistoricalResponseRetriever] = None,
        groq_client: Optional[GroqGenerationClient] = None,
        intent_predictor: Optional[IntentPredictor] = None,
        config: Optional[GroqConfig] = None,
    ) -> None:
        self.intent_predictor = intent_predictor or IntentPredictor()
        self.retriever = retriever
        self.groq_client = groq_client
        self._config = config

    def _ensure_retriever(self) -> HistoricalResponseRetriever:
        if self.retriever is None:
            self.retriever = HistoricalResponseRetriever()
        return self.retriever

    def _get_config(self) -> GroqConfig:
        """Return the GroqConfig without constructing a Groq client.

        Used to read safe metadata (e.g. model name) for all code paths,
        including escalated paths that never invoke the LLM.
        """
        if self._config is not None:
            return self._config
        if self.groq_client is not None and hasattr(self.groq_client, "config"):
            return self.groq_client.config
        # Load from environment — reads GROQ_API_KEY and GROQ_MODEL from .env.
        # This does NOT make any network call.
        self._config = load_groq_config()
        return self._config

    def _ensure_groq_client(self) -> GroqGenerationClient:
        if self.groq_client is None:
            cfg = self._get_config()
            self.groq_client = GroqGenerationClient(config=cfg)
        return self.groq_client

    def process_message(
        self,
        customer_message: str,
        top_k_evidence: int = 3,
    ) -> AgentResponse:
        """
        Process an inbound customer message through the grounded RAG workflow.
        """
        cleaned_text = (customer_message or "").strip()

        # Resolve config early — needed for metadata on ALL code paths (including escalated).
        # _get_config() reads from env without making any network call.
        cfg = self._get_config()

        # 1. Intent Classification
        pred_intent, confidence, margin = self.intent_predictor.predict(cleaned_text)
        uncertainty = round(1.0 - confidence, 4)

        # 2. Historical Response Retrieval
        retriever = self._ensure_retriever()
        evidence_results = retriever.retrieve(
            query=cleaned_text,
            top_k=top_k_evidence,
            deduplicate_customer_text=True,
        )

        top_similarity = (
            evidence_results[0].similarity_score if evidence_results else 0.0
        )
        evidence_dicts = [res.to_dict() for res in evidence_results]
        source_ids = [res.customer_tweet_id for res in evidence_results]

        # 3. Deterministic Escalation Evaluation
        escalation: EscalationDecision = EscalationEngine.evaluate(
            customer_text=cleaned_text,
            predicted_intent=pred_intent,
            intent_confidence=confidence,
            intent_margin=margin,
            top_retrieval_similarity=top_similarity,
        )

        # 4. LLM Response Generation or Safe Fallback
        generated_response_text = ""
        grounding_summary = ""

        # If escalation is critical (e.g., safety hazard, legal threat, fraud),
        # return safe immediate escalation notice without calling LLM
        if escalation.severity == "critical":
            generated_response_text = (
                "Thank you for contacting AppleSupport. Because your inquiry requires immediate specialized "
                "assistance, we have escalated this to a supervisor. Please send us a Direct Message with "
                "your details so we can assist you safely."
            )
            grounding_summary = f"Automated escalation triggered by critical rule: {escalation.rule_triggered}"
        elif escalation.rule_triggered == "account_security_escalation":
            generated_response_text = (
                "For your security, account access and password inquiries cannot be handled over public tweets. "
                "Please visit https://iforgot.apple.com or send us a Direct Message so our security team can assist."
            )
            grounding_summary = "Account access security rule: directed to official iForgot portal and private DM."
        elif escalation.rule_triggered == "vague_short_query":
            generated_response_text = (
                "We're here to help! Could you please let us know what device and software version you're using, "
                "and share a few more details about what's happening?"
            )
            grounding_summary = "Vague customer message: generated standard clarifying inquiry."
        else:
            # Build grounded prompt and dispatch to Groq
            prompt_messages = GroundedPromptBuilder.build_prompt(
                customer_message=cleaned_text,
                predicted_intent=pred_intent,
                intent_confidence=confidence,
                evidence=evidence_results,
                should_escalate=escalation.should_escalate,
                escalation_reason=escalation.escalation_reason,
            )

            try:
                client = self._ensure_groq_client()
                llm_output = client.generate_json_response(prompt_messages)

                generated_response_text = str(llm_output.get("draft_response", "")).strip()
                grounding_summary = str(llm_output.get("grounding_summary", "")).strip()

                if not generated_response_text:
                    raise GroqGenerationError("Groq output missing 'draft_response' key")

            except (GroqGenerationError, Exception) as gen_err:
                logger.warning("Groq generation failed or raised error: %s. Using grounded fallback.", gen_err)
                # Fallback to the top historical response text with polite adjustment
                if evidence_results:
                    fallback_reply = evidence_results[0].brand_text
                    # Clean handle prefix like @115854
                    cleaned_brand_reply = " ".join(
                        w for w in fallback_reply.split() if not w.startswith("@")
                    )
                    generated_response_text = (
                        f"We'd like to help you with this. {cleaned_brand_reply}"
                    )
                    grounding_summary = f"Fallback generated directly from historical tweet {evidence_results[0].brand_tweet_id}."
                else:
                    generated_response_text = (
                        "We'd love to look into this with you. Please reach out to us via Direct Message "
                        "with your device details so we can assist."
                    )
                    grounding_summary = "Fallback generic response due to absent evidence and generation error."

        # Metadata (safe to expose, zero credentials)
        # model_used is always set from config — it must never be 'unknown',
        # even on escalated paths that skip the Groq API call entirely.
        safe_meta = {
            "orchestrator_version": "1.0",
            "phase": "Phase 12: Grounded Agent Response Generation",
            "model_used": cfg.model,
            "retrieval_corpus_size": retriever.corpus_size,
            "top_similarity_score": round(top_similarity, 4),
            "escalation_rule_triggered": escalation.rule_triggered,
            "escalation_severity": escalation.severity,
            "grounding_summary": grounding_summary,
            "groq": cfg.safe_metadata(),
        }

        return AgentResponse(
            customer_message=cleaned_text,
            predicted_intent=pred_intent,
            intent_confidence=confidence,
            intent_uncertainty=uncertainty,
            response=generated_response_text,
            evidence=evidence_dicts,
            evidence_source_ids=source_ids,
            should_escalate=escalation.should_escalate,
            escalation_reason=escalation.escalation_reason,
            grounding_score=round(top_similarity, 4),
            model_metadata=safe_meta,
        )
