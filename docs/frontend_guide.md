# Phase 16 — Vercel Frontend Deployment & Integration Guide

## 1. Architecture Overview

The AI Customer Support Agent utilizes a decoupled, modern cloud deployment architecture:

```
+------------------------------------+
|  GitHub Repository                 |
|  - Source code                     |
|  - Continuous Deployment trigger   |
+-----------------+------------------+
                  |
         Auto-deploy via git
                  |
                  +-----------------------------------+
                  |                                   |
                  v                                   v
+------------------------------------+    CORS    +------------------------------------+
|  Render Web Service (AI Backend)   | <========= |  Vercel Web App (Frontend UI)      |
|  - Python FastAPI + Uvicorn        |            |  - React + Vite SPA                |
|  - Phase 10 Intent Classification  | =========> |  - Root Directory: frontend/       |
|  - Phase 11 Historical Retrieval   |   JSON     |  - Env: VITE_API_URL               |
|  - Phase 12 Grounded Agent (Groq)  | Responses  |  - Zero server-side secrets        |
|  - Phase 13 Agent Reviewer Layer   |            |  - Modern Apple-inspired dark mode |
+------------------------------------+            +------------------------------------+
```

* **Frontend**: React + Vite Single Page Application (SPA) located in the `frontend/` directory.
* **Backend**: FastAPI ASGI service hosted on Render, exposing `/health`, `/ready`, and `/api/chat`.
* **Security & Secret Hygiene**: All model weights, embeddings, Groq API keys, and reviewer configurations reside exclusively on the Render server. The browser client receives only sanitized structured JSON responses.

---

## 2. Environment Variables

### Frontend (`frontend/.env`)

| Variable Name | Required | Default Value (Local) | Production Example | Description |
|---|:---:|---|---|---|
| `VITE_API_URL` | **Yes** | `http://127.0.0.1:8000` | `https://apple-support-ai-agent.onrender.com` | Base URL of the deployed Render AI backend. |

> [!CAUTION]
> **Zero Secret Rule**  
> Do **NOT** place `GROQ_API_KEY`, database credentials, or internal tokens in the frontend environment. Vite packages any `VITE_*` variable into public client JavaScript bundles.

---

## 3. Local Development Instructions

### Step 1: Start the Backend (Terminal 1)
```bash
# From workspace root
.venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Verify the backend readiness probe:
```bash
curl -s http://127.0.0.1:8000/ready
```

### Step 2: Start the Frontend (Terminal 2)
```bash
# Navigate to frontend directory or use npm --prefix
npm --prefix frontend install
npm --prefix frontend run dev
```
Open [http://127.0.0.1:5173](http://127.0.0.1:5173) in your browser. The status badge in the top navigation bar will show **Backend Ready**.

### Step 3: Run Frontend Tests
```bash
npm --prefix frontend test
```

### Step 4: Validate Production Build
```bash
npm --prefix frontend run build
```

---

## 4. Vercel Deployment Steps

### Method A: Vercel Dashboard (Recommended)

1. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New...** > **Project**.
2. Import your GitHub repository (`Hiver` / `ai-customer-agent`).
3. In the project configuration screen, configure the following:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click *Edit* and select `frontend`
   - **Build Command**: `vite build` (or leave default)
   - **Output Directory**: `dist` (default)
4. Under **Environment Variables**, add:
   - **Key**: `VITE_API_URL`
   - **Value**: Your Render service URL, e.g., `https://apple-support-ai-agent.onrender.com`
5. Click **Deploy**. Vercel will build the frontend and provision an immutable production URL (e.g. `https://apple-support-ai-frontend.vercel.app`).

### Method B: Vercel CLI

```bash
cd frontend
vercel
# When prompted for Root Directory, use '.' since you are inside frontend/
# Add environment variable:
vercel env add VITE_API_URL production
```

---

## 5. Render ↔ Vercel CORS Integration

Once Vercel assigns your production domain (e.g. `https://apple-support-ai-frontend.vercel.app`):

1. Open your [Render Dashboard](https://dashboard.render.com/) for `apple-support-ai-agent`.
2. Navigate to **Environment**.
3. Update `CORS_ORIGINS` to include the Vercel domain:
   ```env
   CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://apple-support-ai-frontend.vercel.app
   ```
4. Save and deploy. The backend will now accept preflight `OPTIONS` and cross-origin `POST /api/chat` requests from your Vercel frontend.

---

## 6. Features & Capabilities

* **Customer Message Input**: Multiline textarea with 1000 character limit, counter, and keyboard shortcut (`Enter` to send, `Shift+Enter` for newline).
* **Quick Scenario Chips**: One-click test scenarios covering routine inquiries, update loops, and critical battery safety alerts.
* **Intelligent Grounded Responses**: Formatted draft responses with model provenance and quick copy-to-clipboard.
* **Real-time Intent & Confidence Badges**: Visual pills displaying detected intent and confidence percentage from Phase 10 classifier.
* **Prominent Escalation Alerts**: High-severity red warning banners triggered on safety hazards or policy violations.
* **Independent Quality Audit Card**: Real-time Phase 13 reviewer decision (`PASS`, `NEEDS_HUMAN_REVIEW`, `FAIL`), overall score gauge, and dimension breakdown.
* **Grounded Context Accordion**: Collapsible cards inspecting historical customer inquiries, AppleSupport replies, cosine similarity match percentage, and citation markers.
* **Resilient Error Handling**: RFC-compliant error mapping for network downtime, timeouts, and validation errors with instant retry.
