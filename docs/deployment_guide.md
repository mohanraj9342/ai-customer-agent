# Phase 15 — Production API & Deployment Guide

## Overview & Architecture

The AI Customer Support Agent utilizes a decoupled, modern cloud architecture:

```
+------------------------------------+
|  GitHub Repository                 |
|  - Source code & ML artifacts      |
|  - Continuous Deployment trigger   |
+-----------------+------------------+
                  |
         Auto-deploy via git
                  |
                  v
+------------------------------------+          Cross-Origin Requests (CORS)          +------------------------------------+
|  Render Web Service                | <-------------------------------------------- |  Vercel Frontend (Future Phase)    |
|  - Python FastAPI + Uvicorn        |                                               |  - Next.js / React Web UI          |
|  - Phase 10 Intent Classification  | --------------------------------------------> |  - Production: your-app.vercel.app |
|  - Phase 11 Historical Retrieval   |          Structured JSON Responses            |  - Dev: localhost:3000 / :5173     |
|  - Phase 12 Grounded Agent (Groq)  |                                               +------------------------------------+
|  - Phase 13 Agent Reviewer Layer   |
+------------------------------------+
```

* **GitHub**: Central version control repository.
* **Render**: Hosts the Python AI Backend API, running `uvicorn src.api.app:app` over HTTPS.
* **Vercel**: Consumes the Render API from a modern frontend without housing server-side secrets or heavy ML dependencies.

---

## 1. Render Deployment Instructions

### Method A: Blueprint Deployment (Recommended)
1. Push your latest commits to your GitHub repository on branch `main`.
2. In the [Render Dashboard](https://dashboard.render.com/), select **New +** > **Blueprint**.
3. Connect your GitHub repository. Render will automatically detect and parse `render.yaml`.
4. Enter the required sensitive environment variable:
   - `GROQ_API_KEY`: Your secret Groq API key (`gsk_...`).
5. Click **Apply**. Render will automatically build the service, install dependencies via `pip install -r requirements.txt`, and boot the Uvicorn ASGI server.

### Method B: Manual Web Service Creation
1. Go to **New +** > **Web Service**.
2. Connect your GitHub repository.
3. Configure the service settings:
   - **Name**: `apple-support-ai-agent`
   - **Environment**: `Python 3`
   - **Region**: `Oregon (US West)` or preferred region.
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Standard` (Recommended: 2 GB RAM / 1 CPU to comfortably load dense embeddings and the bi-encoder model).
   - **Health Check Path**: `/health`

---

## 2. Environment Variables Specification

The following environment variables must be configured in Render (under **Environment** tab):

| Variable Name | Required | Default Value | Description |
|---|:---:|:---:|---|
| `GROQ_API_KEY` | **Yes** | *(None)* | Secret authentication key for Groq API (`gsk_...`). Server-side only; never exposed to clients. |
| `GROQ_MODEL` | **Yes** | `qwen/qwen3.8-27b` | Model identifier verified available on GroqCloud. |
| `TARGET_BRAND` | No | `AppleSupport` | Primary brand name filter for historical context. |
| `PORT` | Auto | `8000` | Injected dynamically by Render during container startup. |
| `API_HOST` | No | `0.0.0.0` | Network binding interface. |
| `CORS_ORIGINS` | No | `http://localhost:3000,http://localhost:5173` | Comma-separated list of explicit allowed origins (e.g. `http://localhost:3000,http://localhost:5173,https://apple-support-ai-agent.vercel.app`). Wildcard Vercel regexes are prohibited in production to prevent unauthorized cross-origin requests from arbitrary Vercel tenants. |
| `API_DEFAULT_REVIEW_MODE` | No | `deterministic` | Reviewer execution mode (`deterministic` or `hybrid`). |
| `API_REQUEST_TIMEOUT_SECONDS` | No | `30.0` | Request processing timeout limit. |

> [!CAUTION]
> **Secret Hygiene Notice**  
> `GROQ_API_KEY` must **NEVER** be committed to GitHub or added to client-side code. Render encrypts environment variables at rest. The API routes and schemas strictly omit raw API keys from all diagnostic, health, and error responses.

---

## 3. API Endpoints Specification

### 3.1 `GET /health` (Liveness Probe)
Used by Render, Kubernetes, and uptime checkers to verify container responsiveness.
* **HTTP Status**: `200 OK`
* **Response**:
```json
{
  "status": "ok",
  "service": "apple-support-ai-agent",
  "version": "1.0.0",
  "timestamp": "2026-09-15T13:50:00Z"
}
```

### 3.2 `GET /ready` (Readiness Probe)
Used by deployment gateways to confirm all ML assets are pre-warmed in memory before routing user traffic.
* **HTTP Status**: `200 OK` (if ready), `503 Service Unavailable` (if still loading).
* **Response**:
```json
{
  "status": "ready",
  "intent_classifier_ready": true,
  "retriever_ready": true,
  "retriever_corpus_size": 81943,
  "groq_configured": true,
  "groq_model": "qwen/qwen3.8-27b",
  "timestamp": "2026-09-15T13:50:00Z"
}
```

### 3.3 `POST /api/chat` (Main Customer Inquiry Endpoint)
Accepts customer inquiries, performs intent classification, retrieves historical grounded cases, enforces safety escalation rules, generates draft replies via Groq, and returns an independent reviewer audit.

* **Request Headers**:
  - `Content-Type: application/json`
* **Request Body**:
```json
{
  "message": "My iPhone battery dies after only 2 hours of use.",
  "top_k_evidence": 3,
  "include_review": true,
  "review_mode": "deterministic"
}
```

* **Success Response (`200 OK`)**:
```json
{
  "status": "success",
  "data": {
    "response": {
      "customer_message": "My iPhone battery dies after only 2 hours of use.",
      "predicted_intent": "battery_power",
      "intent_confidence": 0.9654,
      "intent_uncertainty": 0.0346,
      "should_escalate": false,
      "escalation_reason": null,
      "escalation_rule_triggered": null,
      "escalation_severity": "normal",
      "draft_response": "We understand how concerning battery drain can be. Please check Settings > Battery > Battery Health to inspect your maximum capacity. If you need further help, send us a DM.",
      "grounding_summary": "Guided by historical battery diagnostic steps in evidence #2079208.",
      "cited_evidence_ids": [2079208],
      "retrieved_evidence": [
        {
          "rank": 1,
          "similarity_score": 0.8842,
          "thread_id": "116492",
          "customer_tweet_id": 2079208,
          "customer_text": "@AppleSupport battery dying very quickly on iOS 11",
          "brand_tweet_id": 2079209,
          "brand_text": "@user Let's look into this with you. What does your Battery Health show in Settings?",
          "inferred_intent": "battery_power"
        }
      ],
      "model_used": "qwen/qwen3.8-27b",
      "requires_clarification": false
    },
    "review": {
      "passed": true,
      "decision": "PASS",
      "overall_score": 0.9420,
      "dimension_scores": {
        "grounding_support": 0.92,
        "hallucination_risk": 1.0,
        "escalation_correctness": 1.0,
        "intent_consistency": 0.95,
        "response_relevance": 0.90,
        "response_completeness": 1.0,
        "professional_quality": 1.0
      },
      "detected_issues": [],
      "escalation_consistency": {
        "consistent": true
      },
      "grounding_findings": {
        "cited_evidence_count": 1
      },
      "reviewer_rationale": "High quality response, grounded in battery diagnostic evidence.",
      "reviewer_mode": "deterministic",
      "model_used": null
    },
    "execution_time_ms": 118.45
  }
}
```

* **Escalation Example (`200 OK`)**:
For inquiries containing safety hazards (e.g. *"my phone battery exploded in fire"*):
```json
{
  "status": "success",
  "data": {
    "response": {
      "customer_message": "My phone battery exploded in fire",
      "predicted_intent": "battery_power",
      "should_escalate": true,
      "escalation_reason": "Physical safety hazard or battery expansion detected",
      "escalation_rule_triggered": "safety_hazard_alert",
      "escalation_severity": "critical",
      "draft_response": "SAFETY NOTICE: Immediate human support is required. Please discontinue device use immediately and contact Apple Support directly at 1-800-MY-APPLE or visit an authorized service center.",
      "model_used": "qwen/qwen3.8-27b"
    }
  }
}
```

---

## 4. Vercel Frontend Integration Guide

When constructing the frontend in a subsequent phase, integrate as follows:

### 4.1 Frontend Environment Variable
In your Vercel project settings:
```env
NEXT_PUBLIC_API_URL=https://apple-support-ai-agent.onrender.com
# or for Vite / React:
VITE_API_URL=https://apple-support-ai-agent.onrender.com
```

### 4.2 Consuming the API via Fetch / Axios
```typescript
interface ChatResponse {
  status: string;
  data: {
    response: {
      draft_response: string;
      predicted_intent: string;
      should_escalate: boolean;
      escalation_reason?: string;
      cited_evidence_ids: number[];
      retrieved_evidence: Array<{
        customer_text: string;
        brand_text: string;
        similarity_score: number;
      }>;
    };
    review?: {
      decision: "PASS" | "NEEDS_HUMAN_REVIEW" | "FAIL";
      overall_score: number;
    };
    execution_time_ms: number;
  };
}

export async function sendCustomerInquiry(message: string): Promise<ChatResponse> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const res = await fetch(`${apiUrl}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      top_k_evidence: 3,
      include_review: true,
      review_mode: "deterministic",
    }),
  });

  if (!res.ok) {
    const errorData = await res.json();
    throw new Error(errorData.message || `API error: ${res.status}`);
  }

  return res.json();
}
```

### 4.3 Secure CORS Origin Configuration on Render
* To protect against unauthorized cross-origin requests from untrusted external domains or arbitrary Vercel projects, the backend does not use a permissive wildcard regex (`*.vercel.app`).
* Instead, specify the exact production Vercel frontend URL in the Render environment variable `CORS_ORIGINS`:
  ```env
  CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://apple-support-ai-agent.vercel.app
  ```
* Localhost development ports (`3000` and `5173`) are supported by default.
* Disallowed origins will not receive `Access-Control-Allow-Origin` response headers, and preflight `OPTIONS` requests from unapproved domains are rejected.

---

## 5. Local Execution & Smoke Testing

1. Start the local server:
```bash
.venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

2. Test liveness probe:
```bash
curl -s http://127.0.0.1:8000/health
```

3. Test readiness probe:
```bash
curl -s http://127.0.0.1:8000/ready
```

4. Send customer inquiry:
```bash
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I check battery health on my iPhone?"}'
```
