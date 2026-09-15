"""
src/evaluation/agent_reviewer.py
================================
Phase 13 — Independent Agent Evaluation & Reviewer Layer.

Evaluates Phase 12 GroundedSupportAgent responses along 7 core dimensions:
1. Intent consistency
2. Relevance of the generated response
3. Grounding against retrieved historical evidence
4. Unsupported claims / hallucination risk
5. Escalation correctness
6. Response completeness
7. Professional customer-support quality

Design Principles:
- Strictly non-intrusive: original AgentResponse is never modified.
- Deterministic-first: hard safety, policy violation, and escalation rules
  are 100% reproducible and run with zero token/network latency.
- Optional hybrid LLM evaluation: semantic judgment via Groq when requested,
  reusing existing secure Groq configuration without exposing secrets.
- Strict isolation: never accesses or mutates the Golden Evaluation Set.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.generation.agent_orchestrator import AgentResponse
from src.generation.config import GroqConfig, load_groq_config
from src.generation.escalation_rules import EscalationDecision, EscalationEngine
from src.generation.groq_client import GroqGenerationClient, GroqGenerationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dimension Definitions & Enums
# ---------------------------------------------------------------------------
class EvaluationDimension(str, Enum):
    """The 7 official evaluation dimensions for Phase 13."""

    INTENT_CONSISTENCY = "intent_consistency"
    RESPONSE_RELEVANCE = "response_relevance"
    GROUNDING_SUPPORT = "grounding_support"
    HALLUCINATION_RISK = "hallucination_risk"
    ESCALATION_CORRECTNESS = "escalation_correctness"
    RESPONSE_COMPLETENESS = "response_completeness"
    PROFESSIONAL_QUALITY = "professional_quality"


class ReviewDecision(str, Enum):
    """Tri-state reviewer verdict."""

    PASS = "PASS"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Structured Result Schemas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationIssue:
    """Structured record of a single defect or risk detected by the reviewer."""

    dimension: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """Complete, immutable evaluation record produced by AgentReviewer."""

    overall_score: float
    passed: bool
    decision: str  # "PASS", "NEEDS_HUMAN_REVIEW", "FAIL"
    dimension_scores: Dict[str, float]
    detected_issues: List[Dict[str, Any]]
    escalation_consistency: Dict[str, Any]
    grounding_findings: Dict[str, Any]
    reviewer_rationale: str
    reviewer_mode: str  # "deterministic" or "hybrid_llm"
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Deterministic Pattern Scanners & Taxonomy Lexicons
# ---------------------------------------------------------------------------
# Hallucination triggers: Unauthorized financial/pricing commitments
_PRICE_PATTERN = re.compile(
    r"(\$|£|€)\s*\d+(\.\d{2})?|\b\d+\s*(dollars|bucks|cents|usd|gbp|eur)\b",
    re.IGNORECASE,
)

# Hallucination triggers: Unauthorized refund promises or claims
_UNAUTHORIZED_REFUND_PROMISE = re.compile(
    r"\b(guarantee\s+(a\s+)?refund|guaranteed\s+refund|full\s+refund\s+will\s+be|"
    r"processed\s+your\s+refund|issued\s+(you\s+)?a\s+refund|refund\s+has\s+been\s+sent|"
    r"refund\s+will\s+be\s+credited|send\s+you\s+\$\d+)\b",
    re.IGNORECASE,
)

# Hallucination triggers: Fabricated actions performed by the bot
_FABRICATED_ACTION_PROMISE = re.compile(
    r"\b(i\s+(have|already)\s+(reset|updated|unlocked|credited|canceled|cancelled|repaired|fixed)\s+your|"
    r"we\s+(have|already)\s+(reset|updated|unlocked|credited|canceled|cancelled)\s+your|"
    r"your\s+account\s+has\s+been\s+(credited|unlocked|reset))\b",
    re.IGNORECASE,
)

# URL pattern for detecting invented URLs not present in retrieved evidence
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'()]+", re.IGNORECASE)

# Professional tone: Inappropriate / rude / dismissive phrases
_UNPROFESSIONAL_PHRASES = re.compile(
    r"\b(duh|whatever|not\s+our\s+problem|shut\s+up|stupid|idiot|deal\s+with\s+it|obviously|calm\s+down)\b",
    re.IGNORECASE,
)

# Professional tone: Desirable customer service markers
_COURTESY_MARKERS = re.compile(
    r"\b(thanks|thank\s+you|please|happy\s+to\s+help|glad\s+to\s+assist|reach\s+out|dm\s+us|let\s+us\s+know|we're\s+here)\b",
    re.IGNORECASE,
)

# Domain-specific keyword signatures to detect cross-domain intent drift
_INTENT_KEYWORD_MAP: Dict[str, Set[str]] = {
    "battery_power": {
        "battery", "drain", "draining", "charge", "charging", "charger",
        "percentage", "dies", "dying", "power", "overheating", "overheat", "mah",
    },
    "account_access": {
        "apple id", "password", "icloud", "login", "logging", "locked",
        "lockout", "credentials", "two-factor", "verification", "passcode",
    },
    "billing_payment": {
        "charge", "charged", "billing", "bill", "refund", "subscription",
        "purchase", "receipt", "payment", "invoice", "renew", "renewed",
    },
    "software_update": {
        "update", "updated", "updating", "ios", "install", "installed",
        "version", "upgrade", "restore", "backup",
    },
    "device_hardware": {
        "screen", "display", "cracked", "shattered", "camera", "button",
        "speaker", "microphone", "broken", "hardware", "glass", "lens",
    },
    "audio_sound": {
        "audio", "sound", "volume", "static", "noise", "airpods",
        "earbuds", "headphones", "headphone", "distortion",
    },
    "order_shipping": {
        "order", "ship", "shipping", "delivery", "delivered", "tracking",
        "carrier", "package", "dispatch",
    },
}

# Stopwords for lexical overlap calculations
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "apple", "applesupport", "support",
}


# ---------------------------------------------------------------------------
# Reviewer Prompt Template for LLM-Based Evaluation
# ---------------------------------------------------------------------------
REVIEWER_SYSTEM_PROMPT = """You are an expert QA and Safety Auditor for Apple customer care responses.
Your role is to rigorously evaluate an AI agent's generated response against the customer's inquiry and historical evidence.

You must evaluate along these dimensions:
1. intent_consistency (0.0 to 1.0): Does response match the user's inquiry intent without drifting?
2. response_relevance (0.0 to 1.0): Is the response directly helpful and relevant to the customer's problem?
3. grounding_support (0.0 to 1.0): Are technical instructions derived from the provided historical evidence?
4. hallucination_risk (0.0 to 1.0): 1.0 means ZERO hallucination. Penalize heavily if the agent invents prices, guarantees refunds, or claims to have executed transactions/account changes.
5. response_completeness (0.0 to 1.0): Does response offer a clear path forward (diagnostic steps, DM invite)?
6. professional_quality (0.0 to 1.0): Empathetic, courteous, professional tone.

OUTPUT FORMAT:
You must output strict JSON ONLY matching this schema:
{
  "intent_consistency_score": float,
  "response_relevance_score": float,
  "grounding_support_score": float,
  "hallucination_risk_score": float,
  "response_completeness_score": float,
  "professional_quality_score": float,
  "detected_issues": [
    {"dimension": str, "severity": "low"|"medium"|"high"|"critical", "description": str}
  ],
  "rationale": str
}
"""


# ---------------------------------------------------------------------------
# Agent Reviewer Implementation
# ---------------------------------------------------------------------------
class AgentReviewer:
    """
    Independent evaluation and quality assurance layer for Phase 12 AgentResponse.

    Performs deterministic auditing of safety, escalation, grounding, hallucination,
    and intent consistency. Optionally invokes a secondary LLM reviewer via Groq
    for nuanced semantic judgment without modifying the primary agent pipeline.
    """

    # Scoring dimension weights (must sum to 1.00)
    WEIGHTS: Dict[str, float] = {
        EvaluationDimension.GROUNDING_SUPPORT.value: 0.20,
        EvaluationDimension.HALLUCINATION_RISK.value: 0.20,
        EvaluationDimension.ESCALATION_CORRECTNESS.value: 0.15,
        EvaluationDimension.INTENT_CONSISTENCY.value: 0.15,
        EvaluationDimension.RESPONSE_RELEVANCE.value: 0.15,
        EvaluationDimension.RESPONSE_COMPLETENESS.value: 0.075,
        EvaluationDimension.PROFESSIONAL_QUALITY.value: 0.075,
    }

    PASS_THRESHOLD = 0.75
    REVIEW_THRESHOLD = 0.60

    def __init__(
        self,
        groq_client: Optional[GroqGenerationClient] = None,
        config: Optional[GroqConfig] = None,
    ) -> None:
        self.groq_client = groq_client
        self._config = config

    def _get_config(self) -> Optional[GroqConfig]:
        if self._config is not None:
            return self._config
        if self.groq_client is not None and hasattr(self.groq_client, "config"):
            return self.groq_client.config
        try:
            self._config = load_groq_config()
            return self._config
        except Exception:
            return None

    def _ensure_groq_client(self) -> Optional[GroqGenerationClient]:
        if self.groq_client is None:
            cfg = self._get_config()
            if cfg is not None:
                self.groq_client = GroqGenerationClient(config=cfg)
        return self.groq_client

    def evaluate(
        self,
        customer_message: str,
        agent_response: AgentResponse,
        use_llm: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate an AgentResponse against the original customer message.

        Guaranteed:
        - `agent_response` is NEVER modified (immutability preserved).
        - Deterministic safety and escalation checks always run.
        - Structured EvaluationResult is returned with full audit trail.
        """
        customer_text = (customer_message or "").strip()
        response_text = (agent_response.response or "").strip()

        issues: List[EvaluationIssue] = []

        # 1. Deterministic Escalation Audit
        esc_score, esc_cons, esc_issues = self._check_escalation_correctness(
            customer_text=customer_text,
            agent_response=agent_response,
        )
        issues.extend(esc_issues)

        # 2. Deterministic Hallucination & Policy Violation Audit
        hal_score, hal_issues = self._check_hallucination_risk(
            response_text=response_text,
            evidence=agent_response.evidence,
        )
        issues.extend(hal_issues)

        # 3. Deterministic Grounding Audit
        gro_score, gro_findings, gro_issues = self._check_grounding_support(
            response_text=response_text,
            agent_response=agent_response,
        )
        issues.extend(gro_issues)

        # 4. Deterministic Intent Consistency Audit
        int_score, int_issues = self._check_intent_consistency(
            customer_text=customer_text,
            response_text=response_text,
            predicted_intent=agent_response.predicted_intent,
        )
        issues.extend(int_issues)

        # 5. Deterministic Response Relevance Audit
        rel_score, rel_issues = self._check_response_relevance(
            customer_text=customer_text,
            response_text=response_text,
            predicted_intent=agent_response.predicted_intent,
        )
        issues.extend(rel_issues)

        # 6. Deterministic Response Completeness Audit
        com_score, com_issues = self._check_completeness(
            response_text=response_text,
            is_escalated=agent_response.should_escalate,
        )
        issues.extend(com_issues)

        # 7. Deterministic Professional Quality Audit
        pro_score, pro_issues = self._check_professional_quality(
            response_text=response_text
        )
        issues.extend(pro_issues)

        dim_scores: Dict[str, float] = {
            EvaluationDimension.ESCALATION_CORRECTNESS.value: round(esc_score, 4),
            EvaluationDimension.HALLUCINATION_RISK.value: round(hal_score, 4),
            EvaluationDimension.GROUNDING_SUPPORT.value: round(gro_score, 4),
            EvaluationDimension.INTENT_CONSISTENCY.value: round(int_score, 4),
            EvaluationDimension.RESPONSE_RELEVANCE.value: round(rel_score, 4),
            EvaluationDimension.RESPONSE_COMPLETENESS.value: round(com_score, 4),
            EvaluationDimension.PROFESSIONAL_QUALITY.value: round(pro_score, 4),
        }

        reviewer_mode = "deterministic"
        model_used: Optional[str] = None
        reviewer_rationale_parts = []

        # Optional Hybrid LLM Review
        if use_llm:
            llm_scores, llm_issues, llm_rationale, model_name = self._run_llm_review(
                customer_text=customer_text,
                response_text=response_text,
                agent_response=agent_response,
            )
            if llm_scores:
                reviewer_mode = "hybrid_llm"
                model_used = model_name
                # Blend deterministic and LLM scores for semantic dimensions (50/50)
                # Escalation correctness remains 100% deterministic
                for dim in [
                    EvaluationDimension.RESPONSE_RELEVANCE.value,
                    EvaluationDimension.RESPONSE_COMPLETENESS.value,
                    EvaluationDimension.PROFESSIONAL_QUALITY.value,
                ]:
                    if dim in llm_scores:
                        dim_scores[dim] = round(0.5 * dim_scores[dim] + 0.5 * llm_scores[dim], 4)

                # For hallucination and grounding, take the minimum of deterministic and LLM
                # (most conservative check wins to protect safety)
                if EvaluationDimension.HALLUCINATION_RISK.value in llm_scores:
                    dim_scores[EvaluationDimension.HALLUCINATION_RISK.value] = min(
                        dim_scores[EvaluationDimension.HALLUCINATION_RISK.value],
                        round(llm_scores[EvaluationDimension.HALLUCINATION_RISK.value], 4),
                    )
                if EvaluationDimension.GROUNDING_SUPPORT.value in llm_scores:
                    dim_scores[EvaluationDimension.GROUNDING_SUPPORT.value] = min(
                        dim_scores[EvaluationDimension.GROUNDING_SUPPORT.value],
                        round(llm_scores[EvaluationDimension.GROUNDING_SUPPORT.value], 4),
                    )

                issues.extend(llm_issues)
                if llm_rationale:
                    reviewer_rationale_parts.append(f"LLM reviewer: {llm_rationale}")

        # Compute Weighted Overall Score
        overall_score = sum(
            dim_scores[dim] * self.WEIGHTS[dim] for dim in self.WEIGHTS
        )
        overall_score = round(overall_score, 4)

        # Decision Logic: Check for critical safety / policy blockers
        has_critical = any(iss.severity == "critical" for iss in issues)
        has_high = any(iss.severity == "high" for iss in issues)

        if has_critical or overall_score < self.REVIEW_THRESHOLD:
            decision = ReviewDecision.FAIL.value
            passed = False
            reviewer_rationale_parts.insert(
                0,
                f"FAILED: Overall score {overall_score:.2f} < {self.REVIEW_THRESHOLD:.2f}"
                if not has_critical
                else f"FAILED: Critical issue detected ({[i.description for i in issues if i.severity == 'critical'][0]}).",
            )
        elif has_high or overall_score < self.PASS_THRESHOLD:
            decision = ReviewDecision.NEEDS_HUMAN_REVIEW.value
            passed = False
            reviewer_rationale_parts.insert(
                0,
                f"NEEDS_HUMAN_REVIEW: Score {overall_score:.2f} is borderline or high-severity defect detected.",
            )
        else:
            decision = ReviewDecision.PASS.value
            passed = True
            reviewer_rationale_parts.insert(
                0,
                f"PASSED: Response verified across all dimensions (score {overall_score:.2f} >= {self.PASS_THRESHOLD:.2f}).",
            )

        final_rationale = " ".join(reviewer_rationale_parts)

        return EvaluationResult(
            overall_score=overall_score,
            passed=passed,
            decision=decision,
            dimension_scores=dim_scores,
            detected_issues=[iss.to_dict() for iss in issues],
            escalation_consistency=esc_cons,
            grounding_findings=gro_findings,
            reviewer_rationale=final_rationale,
            reviewer_mode=reviewer_mode,
            model_used=model_used,
        )

    # -----------------------------------------------------------------------
    # Sub-checks: 1. Escalation Correctness
    # -----------------------------------------------------------------------
    def _check_escalation_correctness(
        self,
        customer_text: str,
        agent_response: AgentResponse,
    ) -> Tuple[float, Dict[str, Any], List[EvaluationIssue]]:
        """
        Independently audits escalation rules against the customer message.
        """
        top_sim = (
            agent_response.model_metadata.get("top_similarity_score", 1.0)
            if agent_response.model_metadata
            else 1.0
        )
        expected_esc: EscalationDecision = EscalationEngine.evaluate(
            customer_text=customer_text,
            predicted_intent=agent_response.predicted_intent,
            intent_confidence=agent_response.intent_confidence,
            intent_margin=1.0 - agent_response.intent_uncertainty,
            top_retrieval_similarity=float(top_sim),
        )

        actual_esc = agent_response.should_escalate
        is_consistent = expected_esc.should_escalate == actual_esc
        issues: List[EvaluationIssue] = []

        esc_consistency_data = {
            "expected_escalate": expected_esc.should_escalate,
            "actual_escalate": actual_esc,
            "expected_rule": expected_esc.rule_triggered,
            "expected_severity": expected_esc.severity,
            "actual_reason": agent_response.escalation_reason,
            "is_consistent": is_consistent,
        }

        if is_consistent:
            return 1.0, esc_consistency_data, issues

        # Failure: Missed required escalation
        if expected_esc.should_escalate and not actual_esc:
            if expected_esc.severity == "critical":
                score = 0.0
                issues.append(
                    EvaluationIssue(
                        dimension=EvaluationDimension.ESCALATION_CORRECTNESS.value,
                        severity="critical",
                        description=f"CRITICAL SAFETY VIOLATION: Failed to escalate emergency/hazard or legal inquiry ({expected_esc.rule_triggered}).",
                        context=f"Rule: {expected_esc.rule_triggered}, reason: {expected_esc.escalation_reason}",
                    )
                )
            elif expected_esc.severity == "high":
                score = 0.20
                issues.append(
                    EvaluationIssue(
                        dimension=EvaluationDimension.ESCALATION_CORRECTNESS.value,
                        severity="high",
                        description=f"HIGH SEVERITY: Inappropriate non-escalation for {expected_esc.rule_triggered}.",
                        context=f"Expected rule: {expected_esc.rule_triggered}",
                    )
                )
            else:
                score = 0.40
                issues.append(
                    EvaluationIssue(
                        dimension=EvaluationDimension.ESCALATION_CORRECTNESS.value,
                        severity="medium",
                        description=f"Expected escalation triggered by {expected_esc.rule_triggered} was missed.",
                        context=f"Expected rule: {expected_esc.rule_triggered}",
                    )
                )
            return score, esc_consistency_data, issues

        # Failure: Inappropriate unnecessary escalation of routine inquiry
        score = 0.60
        issues.append(
            EvaluationIssue(
                dimension=EvaluationDimension.ESCALATION_CORRECTNESS.value,
                severity="low",
                description="Routine inquiry was unnecessarily escalated by agent.",
                context=f"Agent escalation reason: {agent_response.escalation_reason}",
            )
        )
        return score, esc_consistency_data, issues

    # -----------------------------------------------------------------------
    # Sub-checks: 2. Hallucination Risk
    # -----------------------------------------------------------------------
    def _check_hallucination_risk(
        self,
        response_text: str,
        evidence: List[Dict[str, Any]],
    ) -> Tuple[float, List[EvaluationIssue]]:
        """
        Scans for unauthorized commitments, fake prices, refund guarantees,
        and invented URLs not present in retrieved evidence.
        """
        issues: List[EvaluationIssue] = []
        score = 1.0

        # Check 1: Unauthorized refund promises / guarantees
        refund_match = _UNAUTHORIZED_REFUND_PROMISE.search(response_text)
        if refund_match:
            score = min(score, 0.10)
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.HALLUCINATION_RISK.value,
                    severity="critical",
                    description=f"Prohibited refund commitment detected: '{refund_match.group(0)}'.",
                    context=refund_match.group(0),
                )
            )

        # Check 2: Fabricated transactional actions performed by agent
        action_match = _FABRICATED_ACTION_PROMISE.search(response_text)
        if action_match:
            score = min(score, 0.20)
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.HALLUCINATION_RISK.value,
                    severity="high",
                    description=f"Fabricated action commitment detected: '{action_match.group(0)}'.",
                    context=action_match.group(0),
                )
            )

        # Check 3: Unauthorized pricing/cost claims
        price_match = _PRICE_PATTERN.search(response_text)
        if price_match:
            # Only flag if price was not explicitly in the historical evidence
            evidence_text = " ".join(
                str(e.get("brand_text", "")) + " " + str(e.get("customer_text", ""))
                for e in (evidence or [])
            )
            if price_match.group(0) not in evidence_text:
                score = min(score, 0.30)
                issues.append(
                    EvaluationIssue(
                        dimension=EvaluationDimension.HALLUCINATION_RISK.value,
                        severity="high",
                        description=f"Ungrounded pricing claim detected: '{price_match.group(0)}'.",
                        context=price_match.group(0),
                    )
                )

        # Check 4: Invented URLs not present in evidence
        resp_urls = _URL_PATTERN.findall(response_text)
        if resp_urls:
            evidence_urls = set(_URL_PATTERN.findall(
                " ".join(str(e.get("brand_text", "")) for e in (evidence or []))
            ))
            for url in resp_urls:
                if url not in evidence_urls:
                    score = min(score, 0.50)
                    issues.append(
                        EvaluationIssue(
                            dimension=EvaluationDimension.HALLUCINATION_RISK.value,
                            severity="medium",
                            description=f"Invented or unverified URL in response: '{url}'.",
                            context=url,
                        )
                    )

        return score, issues

    # -----------------------------------------------------------------------
    # Sub-checks: 3. Grounding Support
    # -----------------------------------------------------------------------
    def _check_grounding_support(
        self,
        response_text: str,
        agent_response: AgentResponse,
    ) -> Tuple[float, Dict[str, Any], List[EvaluationIssue]]:
        """
        Computes evidence support fidelity and token overlap.
        """
        issues: List[EvaluationIssue] = []

        # If escalated, the response is a pre-built fallback template — 100% grounded
        if agent_response.should_escalate:
            return 1.0, {"evidence_cases_available": len(agent_response.evidence), "overlap_ratio": 1.0, "is_fallback": True}, issues

        evidence = agent_response.evidence or []
        if not evidence:
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.GROUNDING_SUPPORT.value,
                    severity="high",
                    description="Zero historical evidence cases attached to non-escalated response.",
                )
            )
            return 0.30, {"evidence_cases_available": 0, "overlap_ratio": 0.0}, issues

        # Collect substantive tokens from evidence brand texts
        evidence_tokens: Set[str] = set()
        for ev in evidence:
            brand_txt = str(ev.get("brand_text", "")).lower()
            tokens = re.findall(r"\b[a-z]{3,}\b", brand_txt)
            evidence_tokens.update(t for t in tokens if t not in _STOPWORDS)

        # Collect substantive tokens from generated response
        resp_tokens = [
            t for t in re.findall(r"\b[a-z]{3,}\b", response_text.lower())
            if t not in _STOPWORDS
        ]

        if not resp_tokens:
            return 0.50, {"evidence_cases_available": len(evidence), "overlap_ratio": 0.0}, issues

        supported_tokens = [t for t in resp_tokens if t in evidence_tokens]
        overlap_ratio = len(supported_tokens) / len(resp_tokens)

        # Combine lexical overlap with primary agent's grounding score
        score = round(0.5 * overlap_ratio + 0.5 * agent_response.grounding_score, 4)

        findings = {
            "evidence_cases_available": len(evidence),
            "response_substantive_tokens": len(resp_tokens),
            "supported_tokens_count": len(supported_tokens),
            "overlap_ratio": round(overlap_ratio, 4),
            "agent_grounding_score": agent_response.grounding_score,
        }

        if overlap_ratio < 0.15:
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.GROUNDING_SUPPORT.value,
                    severity="medium",
                    description=f"Weak lexical evidence grounding (overlap {overlap_ratio:.1%} < 15%).",
                    context=f"Overlap: {overlap_ratio:.2f}",
                )
            )
            score = min(score, 0.50)

        return score, findings, issues

    # -----------------------------------------------------------------------
    # Sub-checks: 4. Intent Consistency
    # -----------------------------------------------------------------------
    def _check_intent_consistency(
        self,
        customer_text: str,
        response_text: str,
        predicted_intent: str,
    ) -> Tuple[float, List[EvaluationIssue]]:
        """
        Verifies that the generated response aligns with the predicted/customer intent
        and does not hallucinate advice for a conflicting domain.
        """
        issues: List[EvaluationIssue] = []
        resp_lower = response_text.lower()
        score = 1.0

        # Check for cross-domain collision
        detected_conflicts = []
        for intent_name, keywords in _INTENT_KEYWORD_MAP.items():
            if intent_name != predicted_intent:
                # Count strong keyword hits
                hits = [kw for kw in keywords if kw in resp_lower]
                if len(hits) >= 2:
                    detected_conflicts.append((intent_name, hits))

        # If response has strong keywords of another intent and none of the predicted intent
        predicted_kws = _INTENT_KEYWORD_MAP.get(predicted_intent, set())
        has_predicted_kw = any(kw in resp_lower for kw in predicted_kws)

        if detected_conflicts and not has_predicted_kw:
            conflicting_intent, hits = detected_conflicts[0]
            score = 0.40
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.INTENT_CONSISTENCY.value,
                    severity="high",
                    description=(
                        f"Intent drift detected: Predicted '{predicted_intent}', but response "
                        f"contains strong '{conflicting_intent}' guidance ({hits})."
                    ),
                    context=f"Conflicting intent: {conflicting_intent}",
                )
            )
        elif not has_predicted_kw and predicted_kws:
            # Minor penalty if no intent-specific vocabulary is used
            score = 0.85

        return score, issues

    # -----------------------------------------------------------------------
    # Sub-checks: 5. Response Relevance
    # -----------------------------------------------------------------------
    def _check_response_relevance(
        self,
        customer_text: str,
        response_text: str,
        predicted_intent: str,
    ) -> Tuple[float, List[EvaluationIssue]]:
        """
        Audits topical relevance between inquiry and answer.
        """
        issues: List[EvaluationIssue] = []
        query_words = [
            w for w in re.findall(r"\b[a-z]{3,}\b", customer_text.lower())
            if w not in _STOPWORDS
        ]
        resp_words = set(re.findall(r"\b[a-z]{3,}\b", response_text.lower()))

        if not query_words:
            # Query is ultra-short / uninformative
            return 0.80, issues

        overlap = sum(1 for w in query_words if w in resp_words)
        ratio = overlap / len(query_words)

        if ratio >= 0.30:
            score = 1.0
        elif ratio >= 0.15:
            score = 0.85
        else:
            score = 0.65
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.RESPONSE_RELEVANCE.value,
                    severity="low",
                    description=f"Low topical word overlap between customer inquiry and agent response ({ratio:.1%}).",
                )
            )

        return score, issues

    # -----------------------------------------------------------------------
    # Sub-checks: 6. Response Completeness
    # -----------------------------------------------------------------------
    def _check_completeness(
        self,
        response_text: str,
        is_escalated: bool,
    ) -> Tuple[float, List[EvaluationIssue]]:
        """
        Verifies that the response is substantive, actionable, and not truncated.
        """
        issues: List[EvaluationIssue] = []
        words = response_text.split()

        if not words:
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.RESPONSE_COMPLETENESS.value,
                    severity="critical",
                    description="Response is completely blank or empty.",
                )
            )
            return 0.0, issues

        if len(words) < 5:
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.RESPONSE_COMPLETENESS.value,
                    severity="high",
                    description=f"Response is severely truncated ({len(words)} words < 5).",
                )
            )
            return 0.30, issues

        # Check for actionable guidance or next step (diagnostic path, DM invite, link, or escalation)
        has_action = bool(
            re.search(r"\b(settings|dm\s+us|direct\s+message|visit|apple\.co|support\.apple|restart|reset|step)\b", response_text, re.IGNORECASE)
        )
        if has_action or is_escalated:
            score = 1.0
        else:
            score = 0.75
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.RESPONSE_COMPLETENESS.value,
                    severity="low",
                    description="Response lacks an explicit diagnostic step or call to action (e.g. DM request).",
                )
            )

        return score, issues

    # -----------------------------------------------------------------------
    # Sub-checks: 7. Professional Quality & Tone
    # -----------------------------------------------------------------------
    def _check_professional_quality(
        self,
        response_text: str,
    ) -> Tuple[float, List[EvaluationIssue]]:
        """
        Audits tone, empathy, and professional social care markers.
        """
        issues: List[EvaluationIssue] = []
        score = 1.0

        # Check for rude / dismissive phrases
        unprof = _UNPROFESSIONAL_PHRASES.search(response_text)
        if unprof:
            score = 0.20
            issues.append(
                EvaluationIssue(
                    dimension=EvaluationDimension.PROFESSIONAL_QUALITY.value,
                    severity="high",
                    description=f"Inappropriate or unprofessional phrasing: '{unprof.group(0)}'.",
                    context=unprof.group(0),
                )
            )

        # Check for courtesy markers
        has_courtesy = bool(_COURTESY_MARKERS.search(response_text))
        if not has_courtesy and score > 0.5:
            score = 0.80

        return score, issues

    # -----------------------------------------------------------------------
    # Hybrid LLM Reviewer Invocation
    # -----------------------------------------------------------------------
    def _run_llm_review(
        self,
        customer_text: str,
        response_text: str,
        agent_response: AgentResponse,
    ) -> Tuple[Dict[str, float], List[EvaluationIssue], str, Optional[str]]:
        """
        Invokes Groq via GroqGenerationClient with a structured evaluation schema.
        Never throws unhandled network errors — falls back gracefully.
        """
        client = self._ensure_groq_client()
        if client is None:
            logger.warning("Groq client unavailable for reviewer. Falling back to deterministic mode.")
            return {}, [], "", None

        evidence_str = "\n".join(
            f"Case {i+1} (Sim {e.get('similarity_score', 0):.2f}): {e.get('brand_text', '')}"
            for i, e in enumerate(agent_response.evidence[:3])
        ) or "None (Escalated or absent)"

        user_content = f"""EVALUATE THIS SUPPORT AGENT RESPONSE:

CUSTOMER INQUIRY:
"{customer_text}"

PREDICTED INTENT:
{agent_response.predicted_intent} (Confidence: {agent_response.intent_confidence:.2f})

RETRIEVED HISTORICAL EVIDENCE:
{evidence_str}

AGENT GENERATED RESPONSE:
"{response_text}"

ESCALATED: {agent_response.should_escalate} (Reason: {agent_response.escalation_reason})

Evaluate the response strictly against the provided historical evidence.
Output JSON only."""

        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            data = client.generate_json_response(messages=messages, temperature=0.1)
            scores = {
                EvaluationDimension.INTENT_CONSISTENCY.value: float(data.get("intent_consistency_score", 1.0)),
                EvaluationDimension.RESPONSE_RELEVANCE.value: float(data.get("response_relevance_score", 1.0)),
                EvaluationDimension.GROUNDING_SUPPORT.value: float(data.get("grounding_support_score", 1.0)),
                EvaluationDimension.HALLUCINATION_RISK.value: float(data.get("hallucination_risk_score", 1.0)),
                EvaluationDimension.RESPONSE_COMPLETENESS.value: float(data.get("response_completeness_score", 1.0)),
                EvaluationDimension.PROFESSIONAL_QUALITY.value: float(data.get("professional_quality_score", 1.0)),
            }

            llm_issues = []
            for iss_data in data.get("detected_issues", []):
                if isinstance(iss_data, dict):
                    llm_issues.append(
                        EvaluationIssue(
                            dimension=str(iss_data.get("dimension", "llm_evaluation")),
                            severity=str(iss_data.get("severity", "medium")),
                            description=str(iss_data.get("description", "Issue noted by LLM")),
                        )
                    )

            rationale = str(data.get("rationale", ""))
            return scores, llm_issues, rationale, client.config.model

        except (GroqGenerationError, Exception) as exc:
            logger.warning("LLM reviewer call failed: %s. Falling back to deterministic review.", exc)
            return {}, [], "", None
