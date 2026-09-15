# Phase 13 — Agent Evaluation & Reviewer Layer

## Overview

Phase 13 implements an independent, non-intrusive evaluation and reviewer layer for the Phase 12 `GroundedSupportAgent`. The reviewer audits primary agent responses along 7 quality, safety, and operational dimensions without altering the primary agent's output.

The reviewer enforces a **deterministic-first** architecture: critical safety hazards, unauthorized financial promises, policy violations, and escalation inconsistencies are caught deterministically with zero token overhead and zero network latency. An optional **hybrid LLM reviewer mode** leverages the existing Groq configuration to provide semantic nuance and grading on tone, relevance, and completeness.

---

## Architectural Principles

1. **Non-Intrusive Immutability**: The primary `AgentResponse` is strictly read-only (`frozen=True`). The reviewer never alters, filters, or silently mutates the original response. Downstream systems can inspect both the original response and the independent `EvaluationResult`.
2. **Deterministic-First Safety**: Safety emergency detection, legal threat triage, pricing claim checks, refund guarantee detection, and escalation audits are governed by deterministic rules. Hard safety violations immediately trigger a `FAIL` verdict regardless of numerical averages.
3. **Hybrid Semantic Grading**: When configured with `use_llm=True`, the reviewer uses Groq (`qwen/qwen3.8-27b`) to assess nuanced semantic relevance, grounding fidelity, and conversational tone, blending LLM scores with deterministic checks.
4. **Golden Set Isolation**: The reviewer operates strictly at the message-evaluation level. The Phase 8 Golden Set is never loaded, indexed, or contaminated.
5. **Zero Credential Exposure**: `GROQ_API_KEY` is never logged, printed, or included in `EvaluationResult` objects or serialized output.

---

## 7 Core Evaluation Dimensions & Scoring Rubric

The reviewer scores 7 dimensions from `0.0` (critical failure) to `1.0` (flawless):

| Dimension | Weight | Primary Audit Mechanism | Description |
|---|:---:|---|---|
| `grounding_support` | **20%** | Token overlap with historical evidence & agent grounding score | Verifies that technical advice and diagnostic paths exist in the retrieved historical brand replies. Escalated responses (fallback templates) receive 1.0. |
| `hallucination_risk` | **20%** | Regex prohibited pattern scanner & non-evidence URL detection | 1.0 = clean. Penalizes unauthorized refund promises (score $\le 0.10$), fabricated bot actions (score $\le 0.20$), ungrounded pricing claims (score $\le 0.30$), and unverified URLs (score $\le 0.50$). |
| `escalation_correctness` | **15%** | Independent `EscalationEngine` audit | Verifies whether the primary agent correctly escalated emergencies (physical safety, legal threats, account security, large disputes). Missed critical escalation yields 0.0 (`FAIL`). |
| `intent_consistency` | **15%** | Cross-domain vocabulary collision detector | Detects intent drift (e.g. customer asked about battery life, but agent drafted instructions for subscription cancellation and refunds). |
| `response_relevance` | **15%** | Inquiry-to-response topical overlap | Assesses whether the response directly addresses the customer's specific inquiry keywords and problem statement. |
| `response_completeness` | **7.5%** | Length check and actionable guidance detector | Verifies that response is substantive (not truncated or empty) and offers a clear next step (Settings path, DM request, or support link). |
| `professional_quality` | **7.5%** | Courtesy marker scanner & unprofessional language filter | Checks for empathetic, polite, customer-centric tone and absence of dismissive or rude phrasing. |

---

## Reviewer Decision Logic

The overall score is the weighted linear combination:
$$\text{Score} = \sum_{d \in D} w_d \cdot s_d$$

### Decision Verdicts:
* **`PASS`**: Overall score $\ge 0.75$, zero high- or critical-severity issues, and escalation is consistent.
* **`NEEDS_HUMAN_REVIEW`**: Overall score between $0.60$ and $0.74$, or at least one high-severity defect detected (e.g. ungrounded pricing, intent drift).
* **`FAIL`**: Overall score $< 0.60$, or any **critical-severity issue** detected (e.g. missed fire/smoke/legal escalation, or unauthorized refund guarantee).

---

## Structured Output Schema (`EvaluationResult`)

```python
@dataclass(frozen=True)
class EvaluationResult:
    overall_score: float                  # e.g. 0.8850
    passed: bool                          # True if PASS, False otherwise
    decision: str                         # "PASS", "NEEDS_HUMAN_REVIEW", "FAIL"
    dimension_scores: Dict[str, float]    # Scores for all 7 dimensions
    detected_issues: List[Dict[str, Any]] # Structured list of issues (severity, description, context)
    escalation_consistency: Dict[str, Any]# Expected vs actual escalation audit
    grounding_findings: Dict[str, Any]    # Evidence token overlap and support metrics
    reviewer_rationale: str               # Human-readable summary rationale
    reviewer_mode: str                    # "deterministic" or "hybrid_llm"
    model_used: Optional[str]             # e.g. "qwen/qwen3.8-27b" if LLM reviewer ran
```

---

## Example Usage

### Deterministic Review
```python
from src.evaluation.agent_reviewer import AgentReviewer

reviewer = AgentReviewer()
result = reviewer.evaluate(
    customer_message=customer_text,
    agent_response=agent_response,
    use_llm=False,
)

print(f"Decision: {result.decision} (Score: {result.overall_score:.2f})")
for issue in result.detected_issues:
    print(f" - [{issue['severity'].upper()}] {issue['description']}")
```

### Hybrid LLM Review
```python
reviewer = AgentReviewer()
result = reviewer.evaluate(
    customer_message=customer_text,
    agent_response=agent_response,
    use_llm=True,
)

print(f"Mode: {result.reviewer_mode} (Model: {result.model_used})")
print(f"Rationale: {result.reviewer_rationale}")
```
