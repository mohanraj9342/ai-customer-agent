"""
src/generation/escalation_rules.py
==================================
Phase 12 — Deterministic Escalation and Safety Rule Layer.

Applies deterministic rule-based safety evaluation to decide whether an incoming
customer inquiry requires immediate human agent escalation before or alongside
automated response drafting. Ensures safety decisions do not rely solely on LLM judgment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EscalationDecision:
    """The result of deterministic safety and routing evaluation."""

    should_escalate: bool
    escalation_reason: Optional[str] = None
    rule_triggered: Optional[str] = None
    severity: str = "normal"  # "normal", "medium", "high", "critical"

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "should_escalate": self.should_escalate,
            "escalation_reason": self.escalation_reason,
            "rule_triggered": self.rule_triggered,
            "severity": self.severity,
        }


# High-risk legal and severe dispute keyword triggers
_LEGAL_KEYWORDS = re.compile(
    r"\b(lawyer|attorney|sue|suing|lawsuit|police|fraud|stolen|unauthorized\s+charge|report\s+you|court)\b",
    re.IGNORECASE,
)

# High-value monetary refund claim triggers (e.g. £100+, $100+)
_HIGH_VALUE_REFUND = re.compile(
    r"(\$|£|€)\s*[1-9]\d{2,}|\b(full\s+refund|demand\s+refund|refund\s+my\s+money)\b",
    re.IGNORECASE,
)

# Harassment / emergency triggers
_SAFETY_KEYWORDS = re.compile(
    r"\b(emergency|threat|kill|harm|die|fire|smoke|exploded|burnt)\b",
    re.IGNORECASE,
)


class EscalationEngine:
    """
    Deterministic rule engine for front-line customer inquiry triage and escalation.
    """

    MIN_WORDS_THRESHOLD = 3
    MIN_SIMILARITY_THRESHOLD = 0.50
    MIN_CONFIDENCE_THRESHOLD = 0.60
    MIN_CONFIDENCE_MARGIN = 0.20

    @classmethod
    def evaluate(
        cls,
        customer_text: str,
        predicted_intent: str,
        intent_confidence: float = 1.0,
        intent_margin: float = 1.0,
        top_retrieval_similarity: float = 1.0,
    ) -> EscalationDecision:
        """
        Evaluate deterministic rules on customer input and model confidence.
        """
        text = (customer_text or "").strip()
        words = text.split()

        # 1. Critical Physical Safety / Hazard Trigger
        if _SAFETY_KEYWORDS.search(text):
            return EscalationDecision(
                should_escalate=True,
                escalation_reason="Physical hazard or safety emergency reported in customer message",
                rule_triggered="safety_hazard_alert",
                severity="critical",
            )

        # 2. Legal / Threat Trigger
        if _LEGAL_KEYWORDS.search(text):
            return EscalationDecision(
                should_escalate=True,
                escalation_reason="Legal dispute, lawsuit threat, or fraud allegation detected",
                rule_triggered="legal_fraud_alert",
                severity="critical",
            )

        # 3. Account Security / Identity Lockout Trigger
        if predicted_intent == "account_access":
            return EscalationDecision(
                should_escalate=True,
                escalation_reason="Account security and authentication credentials require verified human triage",
                rule_triggered="account_security_escalation",
                severity="high",
            )

        # 4. High-Value Financial / Refund Dispute
        if predicted_intent == "billing_payment" and _HIGH_VALUE_REFUND.search(text):
            return EscalationDecision(
                should_escalate=True,
                escalation_reason="Substantial commercial refund or billing dispute requires supervisor authorization",
                rule_triggered="high_value_billing_dispute",
                severity="high",
            )

        # 5. Intent Classifier Ambiguity / Multi-Intent Collision
        if predicted_intent == "needs_review" or intent_confidence < cls.MIN_CONFIDENCE_THRESHOLD:
            return EscalationDecision(
                should_escalate=True,
                escalation_reason=(
                    f"Ambiguous customer intent (confidence {intent_confidence:.2f} < "
                    f"{cls.MIN_CONFIDENCE_THRESHOLD:.2f} or flagged multi-intent collision)"
                ),
                rule_triggered="low_intent_confidence",
                severity="medium",
            )

        if intent_margin < cls.MIN_CONFIDENCE_MARGIN:
            return EscalationDecision(
                should_escalate=True,
                escalation_reason=(
                    f"Narrow classification margin ({intent_margin:.2f} < {cls.MIN_CONFIDENCE_MARGIN:.2f}) "
                    f"between competing taxonomy categories"
                ),
                rule_triggered="narrow_confidence_margin",
                severity="medium",
            )

        # 6. Ultra-Short / Vague Customer Query
        # If the customer provides very little context (e.g., "@AppleSupport", "help")
        if len(words) < cls.MIN_WORDS_THRESHOLD:
            return EscalationDecision(
                should_escalate=True,
                escalation_reason=f"Insufficient context in customer inquiry ({len(words)} words < {cls.MIN_WORDS_THRESHOLD})",
                rule_triggered="vague_short_query",
                severity="low",
            )

        # 7. Out-of-Domain / Weak Retrieval Grounding
        if top_retrieval_similarity < cls.MIN_SIMILARITY_THRESHOLD:
            return EscalationDecision(
                should_escalate=True,
                escalation_reason=(
                    f"Insufficient historical response grounding (top similarity "
                    f"{top_retrieval_similarity:.2f} < {cls.MIN_SIMILARITY_THRESHOLD:.2f})"
                ),
                rule_triggered="weak_retrieval_grounding",
                severity="medium",
            )

        # Standard automated handling permitted
        return EscalationDecision(
            should_escalate=False,
            escalation_reason=None,
            rule_triggered=None,
            severity="normal",
        )
