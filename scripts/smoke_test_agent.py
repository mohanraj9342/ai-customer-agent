#!/usr/bin/env python3
"""
scripts/smoke_test_agent.py
===========================
Phase 12 — Live Smoke Test (Manual Invocation Only).

Sends a small set of representative customer messages through the full
GroundedSupportAgent pipeline using the real Groq API and the real Phase 11
retrieval index. This script is NEVER run automatically by unit tests.

Usage:
    .venv/bin/python scripts/smoke_test_agent.py

Requirements:
    - .env must be populated with GROQ_API_KEY and GROQ_MODEL
    - Phase 11 retrieval index must be built (data/processed/apple_support/)
    - Phase 10 classifier artifacts must exist (models/intent_classifier/)

IMPORTANT: This script NEVER prints the API key. It reports only safe metadata.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ── Safety guard: do not print the API key ──────────────────────────────────
_raw_key = os.getenv("GROQ_API_KEY", "")
if not _raw_key:
    print("ERROR: GROQ_API_KEY is not set in .env. Cannot run live smoke test.")
    sys.exit(1)
print(f"API key configured: {'YES — ' + '*' * 8 + _raw_key[-4:] if _raw_key else 'NO'}")
del _raw_key  # Do not hold in scope longer than needed
# ────────────────────────────────────────────────────────────────────────────

SMOKE_TEST_QUERIES = [
    # Standard cases
    "My iPhone 14 battery drains in 3 hours after updating to iOS 17",
    "Wi-Fi keeps disconnecting from my MacBook every 10 minutes",
    "How do I transfer contacts from my old iPhone to my new one?",
    # Edge cases
    "@AppleSupport help",                             # Vague — should escalate
    "Locked out of my Apple ID and can't reset it",  # Account access — should escalate
    "What time does the Apple Store close today?",    # Out-of-domain
]


def run_smoke_test() -> None:
    """Run live smoke test through the grounded agent pipeline."""
    from src.generation.agent_orchestrator import GroundedSupportAgent

    print("\n" + "=" * 70)
    print("Phase 12 Live Smoke Test — GroundedSupportAgent")
    print("NOTE: This test uses the real Groq API. DO NOT run inside pytest.")
    print("=" * 70)

    agent = GroundedSupportAgent()

    for i, query in enumerate(SMOKE_TEST_QUERIES, 1):
        print(f"\n[{i}/{len(SMOKE_TEST_QUERIES)}] Query: {query!r}")
        print("-" * 60)
        try:
            response = agent.process_message(query)
            print(f"  Intent:        {response.predicted_intent} ({response.intent_confidence:.2f})")
            print(f"  Escalate:      {response.should_escalate}")
            if response.should_escalate:
                print(f"  Escalation:    {response.escalation_reason}")
            print(f"  Grounding:     {response.grounding_score:.3f} ({response.intent_uncertainty:.3f} uncertainty)")
            print(f"  Evidence used: {len(response.evidence)} cases")
            print(f"  Response:      {response.response[:200]!r}{'...' if len(response.response) > 200 else ''}")

            # Safe metadata only — no API key
            safe_meta = {
                k: v for k, v in response.model_metadata.items()
                if k != "groq" or "api_key" not in str(v)
            }
            model_info = safe_meta.get("model_used", "unknown")
            print(f"  Model used:    {model_info}")

            # Verify key is not in response dict
            response_dict_str = json.dumps(response.to_dict())
            assert "gsk_" not in response_dict_str, "SECURITY FAILURE: API key leaked in response!"
            print(f"  Key leak check: PASS")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 70)
    print("Smoke test complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
