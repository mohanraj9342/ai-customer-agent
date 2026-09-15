"""
src/generation/prompt_builder.py
================================
Phase 12 — Grounded Support Prompt Construction and Brand Voice Guardrails.

Constructs structured, grounding-enforced prompts for LLM response generation.
Supplies verified historical interactions as grounded evidence while strictly
prohibiting the invention of ungrounded policies, pricing, repair claims, or
hallucinated action completions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.retrieval.historical_response_retriever import RetrievalResult

SYSTEM_PROMPT = """You are an official AppleSupport customer care agent on social channels.
Your role is to draft concise, empathetic, professional, and brand-aligned customer responses.

STRICT GROUNDING & OPERATIONAL CONSTRAINTS:
1. USE RETRIEVED HISTORICAL EVIDENCE AS GROUNDING:
   - Base your suggested troubleshooting and guidance strictly on the provided verified historical AppleSupport replies.
   - You may reference verified diagnostic pathways (e.g. "Settings > Battery", "Settings > General > Reset") only if supported by the evidence.
   - If historical replies provide official Apple URLs (e.g. support.apple.com links or apple.co links), you may include them.
2. PROHIBITED HALLUCINATIONS:
   - DO NOT invent repair costs, warranty eligibility, refund guarantees, diagnostic outcomes, or technical facts.
   - DO NOT claim that any action was performed (e.g., "I have reset your password" or "We processed your refund").
   - DO NOT invent non-existent settings, iOS versions, or policies.
3. CONVERSATIONAL BRAND TONE:
   - Be concise, helpful, and polite (under 280 characters if possible, similar to Twitter/X support responses).
   - If troubleshooting requires direct identity or diagnostic details, advise the customer to reach out via DM (Direct Message).
4. SENSITIVE OR VAGUE INQUIRIES:
   - If the issue involves password lockouts or refund demands, acknowledge the concern politely and state that a specialist will review it.
   - If the customer inquiry is vague, ask clarifying questions politely.

OUTPUT FORMAT:
You MUST respond strictly with a valid JSON object matching this schema:
{
  "draft_response": "Concise grounded customer reply",
  "cited_evidence_ids": [12345],
  "grounding_summary": "Brief 1-sentence explanation of how evidence supported this draft",
  "requires_clarification": false
}
"""


class GroundedPromptBuilder:
    """
    Assembles contextual, grounded prompts containing classifier metadata and retrieved evidence.
    """

    @classmethod
    def build_prompt(
        cls,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence: List[RetrievalResult],
        should_escalate: bool = False,
        escalation_reason: str | None = None,
    ) -> List[Dict[str, str]]:
        """
        Build the chat message payload (system and user messages) for Groq.
        """
        evidence_blocks: List[Dict[str, Any]] = []
        for item in evidence:
            evidence_blocks.append({
                "source_tweet_id": item.customer_tweet_id,
                "similarity_score": item.similarity_score,
                "historical_customer_query": item.customer_text,
                "paired_applesupport_reply": item.brand_text,
                "historical_intent": item.inferred_intent,
            })

        user_content = {
            "incoming_customer_message": customer_message.strip(),
            "classifier_metadata": {
                "predicted_intent": predicted_intent,
                "confidence": round(intent_confidence, 4),
            },
            "deterministic_routing": {
                "escalation_flagged": should_escalate,
                "escalation_reason": escalation_reason,
            },
            "retrieved_historical_evidence": evidence_blocks,
        }

        user_message_text = (
            "Draft a grounded customer support response for the following inquiry based on the evidence provided:\n\n"
            f"```json\n{json.dumps(user_content, indent=2)}\n```\n\n"
            "Remember: Return ONLY the JSON object with keys 'draft_response', 'cited_evidence_ids', "
            "'grounding_summary', and 'requires_clarification'."
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message_text},
        ]
