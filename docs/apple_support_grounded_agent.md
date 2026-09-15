# Phase 12: Grounded Agent Response Generation & Orchestration

## Overview

Phase 12 implements a fully locally runnable, grounded support agent that combines three
previously built components — the Phase 10 intent classifier, the Phase 11 historical
semantic retriever, and a Groq-hosted LLM — into a unified orchestration pipeline.

The agent never generates responses from raw parametric knowledge. Every response is
anchored to verified historical AppleSupport agent replies retrieved from the Phase 11 index.

---

## Architecture

```
Customer Query
      │
      ▼
┌─────────────────────┐
│  IntentPredictor    │  Phase 9/10 DistilRoBERTa intent classifier
│  (Phase 10 model)   │  → intent label + confidence score
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ HistoricalRetriever │  Phase 11 all-MiniLM-L6-v2 dense index (81,943 pairs)
│ (Phase 11 index)    │  → top-k grounded (customer_msg, brand_reply, tweet_id, score)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  EscalationEngine   │  Deterministic rule engine (pre-LLM safety check)
│  (8 rule classes)   │  → escalation flags and severity levels
└──────────┬──────────┘
           │ No critical escalation?
           ▼
┌─────────────────────┐
│  PromptBuilder      │  Constructs system + user prompt with grounding constraints
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  GroqGenerationClient│ qwen/qwen3.8-27b via Groq API
│  (Groq API)          │ → validated JSON-schema response
└──────────┬──────────┘
           │
           ▼
     AgentResponse
 (structured dataclass)
```

---

## Components

### `src/generation/config.py`

Loads `GROQ_API_KEY` and `GROQ_MODEL` from the local `.env` file.

- API key is **never** printed, logged, or embedded in any response output.
- `__repr__` masks the key as `***configured***`.
- `safe_metadata()` exposes only non-sensitive configuration for reporting.
- Raises a clear `ValueError` if either variable is missing or blank.

### `src/generation/groq_client.py`

Wraps the Groq Python SDK with robust error handling:

| Error Type | Behaviour |
|---|---|
| `APITimeoutError` | Raises `GroqGenerationError` with timeout message |
| `RateLimitError` | Raises `GroqGenerationError` with retry guidance |
| `AuthenticationError` | Raises `GroqGenerationError` (key invalid/expired) |
| Malformed JSON | Raises `GroqGenerationError` with raw content attached |
| Schema violation | Raises `GroqGenerationError` with field-level diff |

### `src/generation/escalation_rules.py`

Deterministic safety engine executed **before** any LLM call, conserving API tokens
and preventing the model from reasoning about high-risk cases.

| Rule | Trigger | Severity |
|---|---|---|
| `safety_hazard_alert` | Keywords: fire, smoke, burn, explosion, shock | `critical` |
| `legal_fraud_alert` | Keywords: lawyer, sue, lawsuit, fraud, scam | `critical` |
| `account_security_escalation` | Intent: `account_access` with confidence ≥ 0.6 | `high` |
| `high_value_billing_dispute` | Intent: `billing` + monetary value > $200 | `high` |
| `low_intent_confidence` | Classifier confidence < 0.45 | `medium` |
| `narrow_confidence_margin` | Top-2 confidence gap < 0.10 | `medium` |
| `vague_short_query` | Token count < 4 | `low` |
| `weak_retrieval_grounding` | Top-1 similarity score < 0.55 | `low` |

Critical-severity escalations trigger a **safe fallback response** that routes the
customer to direct Apple Support channels, without invoking the Groq API.

### `src/generation/prompt_builder.py`

Builds a two-part prompt:

- **System prompt**: Grounding constraints — agent must only use evidence provided,
  must never fabricate Apple product specs, pricing, or policy commitments, and must
  output strict JSON matching the defined schema.
- **User prompt**: Intent classification metadata (label, confidence) followed by
  the top-k retrieved evidence cases (customer message, brand reply, similarity score).

### `src/generation/agent_orchestrator.py`

`GroundedSupportAgent` — the public entry point. Accepts a plain string query and
returns a structured `AgentResponse` dataclass containing:

- `response_text` — the draft response text for the customer
- `intent` — classified intent label
- `intent_confidence` — classifier confidence score
- `evidence_count` — number of retrieved evidence cases used
- `escalation_flags` — list of triggered escalation rule names
- `escalation_severity` — highest severity level across all flags
- `model_used` — Groq model name (from config)
- `is_fallback` — boolean; `True` if safe fallback was used instead of Groq

---

## Model Selection: qwen/qwen3.8-27b

The model was selected by **programmatic live probing** of the Groq `/v1/models` API
endpoint to determine which models are actually available for the configured API key.

`llama-3.3-70b-versatile` (the originally planned model) was **not returned** by the
models endpoint and would have caused `404` errors at inference time. After smoke-testing
all available chat-completion models with a simple instruction-following prompt,
`qwen/qwen3.8-27b` was selected because it:

1. Is available and responds without error.
2. Follows JSON-schema output instructions precisely without unsolicited reasoning preambles.
3. Provides a 131,072-token context window suitable for multi-case evidence embedding.

See **Decision 21** in `docs/decision_log.md` for the full rationale and alternatives considered.

---

## Golden Evaluation Set Isolation

The Phase 11 retrieval index was constructed with strict quarantine of all 158 Phase 8
Golden Evaluation Set records. No golden record tweet ID appears in the retrieval index.

The orchestrator validates at agent initialisation time that no evidence returned by
the retriever belongs to the golden set. This check is enforced in `agent_orchestrator.py`
and covered by `tests/test_grounded_agent.py::TestGroundedSupportAgent::test_golden_set_isolation_in_evidence`.

---

## Security

| Requirement | Implementation |
|---|---|
| No API key leakage | `GroqConfig.__repr__` masks key; `safe_metadata()` returns only boolean `api_key_configured` |
| `.env` not tracked | `.gitignore` line `*.env` confirmed present (verified via `git check-ignore`) |
| No hardcoded keys | Full repository scan confirms 0 hardcoded secrets |
| Test isolation | All 22 unit tests mock the Groq API; no real network calls during testing |

---

## Testing

```bash
# Phase 12 unit tests only
.venv/bin/pytest tests/test_grounded_agent.py -v

# Full test suite (378 tests across all phases)
.venv/bin/pytest tests/ -v
```

### Test Coverage Summary

| Test Class | Tests | Covers |
|---|---|---|
| `TestGroqConfig` | 4 | Key loading, validation, masking |
| `TestGroqGenerationClient` | 5 | Timeout, auth, rate-limit, JSON errors |
| `TestEscalationRules` | 8 | All 8 rule triggers + clean-pass |
| `TestPromptBuilder` | 1 | Prompt structure validation |
| `TestGroundedSupportAgent` | 4 | E2E schema, key leak, fallback, golden isolation |

All tests run without calling the real Groq API.

---

## Environment Setup

Copy `.env.example` and fill in your credentials:

```bash
cp .env.example .env
# Edit .env and add your actual values
```

Required variables:

```env
GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=qwen/qwen3.8-27b
```

To verify which models your API key can access:

```bash
curl -s "https://api.groq.com/openai/v1/models" \
  -H "Authorization: Bearer $GROQ_API_KEY" | python3 -m json.tool
```

---

## Limitations

- The Groq model availability depends on the API key's plan tier. Run the models
  endpoint check above to confirm availability before changing `GROQ_MODEL`.
- Phase 12 does not retrain the intent classifier or rebuild the retrieval index;
  it orchestrates the existing Phase 10 and Phase 11 artefacts.
- Multi-turn conversation context is not yet modelled; the agent processes single-turn
  initial customer messages only.
- Non-English queries will produce lower retrieval similarity scores due to the
  English-dominant `all-MiniLM-L6-v2` encoder.
