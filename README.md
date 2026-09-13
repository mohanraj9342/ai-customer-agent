# AI Customer Support Agent

[![Status](https://img.shields.io/badge/status-Phase%202%20%E2%80%94%20In%20Progress-blue)]()
[![Python](https://img.shields.io/badge/python-3.14-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Project Overview

An end-to-end AI customer-support agent built on the publicly available
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (`thoughtvector/customer-support-on-twitter` on Kaggle).

The agent is designed to:

1. **Classify** each incoming customer tweet into a defined set of support intents.
2. **Retrieve** historically similar resolved conversations from the same brand.
3. **Draft a reply** grounded in how the brand historically resolved similar issues.
4. **Decide** whether the case should be handled automatically or escalated to a human agent,
   with a clear reason for the decision.

---

## Current Status

> **Phase 2 — Brand extraction, conversation reconstruction, and intent definition**
>
> Phase 1 (dataset inspection) is complete. Brand selection is recorded in
> `docs/brand_selection.md`. Phase 2 implements the brand-specific extraction
> pipeline, conversation thread reconstruction, and initial intent taxonomy.

---

## Components

| Component | Description | Status |
|---|---|---|
| Dataset inspection | Measure schema, brand distribution, row counts | ✅ Done |
| Brand selection | Evidence-based brand choice | ✅ Done |
| Data extraction | Filter brand-specific conversations from the dataset | 🔄 Phase 2 |
| Conversation reconstruction | Rebuild multi-turn threads from tweet links | 🔄 Phase 2 |
| Intent taxonomy | Define 6–8 named intent classes | 🔄 Phase 2 |
| Intent labelling | Hand-label a training set | Not started |
| Intent classifier | TF-IDF baseline and embedding-based main classifier | Not started |
| Retrieval index | Similarity index over historical brand responses | Not started |
| Reply generation | LLM-based grounded reply drafting | Not started |
| Escalation logic | Auto-handle vs. human escalation with reasons | Not started |
| Evaluation harness | Automated metrics + LLM-as-judge + human agreement | Not started |
| Golden evaluation set | 150–250 hand-labelled held-out examples | Not started |
| Report and decision log | Technical report + engineering decision log | In progress |

---

## Development Phases

| Phase | Objective |
|---|---|
| **0** | Project scaffolding ✅ |
| **1** | Dataset inspection and brand selection ✅ |
| **2** | Brand extraction, conversation reconstruction, intent definition 🔄 |
| **3** | Intent labelling and training set creation |
| **4** | Intent classifier — baseline and main system |
| **5** | Retrieval index construction |
| **6** | LLM-based reply generation |
| **7** | Escalation decision logic |
| **8** | Golden evaluation set creation |
| **9** | Evaluation harness |
| **10** | Report polish and public demo preparation |

---

## Dataset

The dataset (`twcs/twcs.csv`, ~493 MB) is stored **locally only** and is not committed
to this repository.

To reproduce this project:

```bash
# Download from Kaggle
kaggle datasets download thoughtvector/customer-support-on-twitter
# Unzip and place the file at:
#   twcs/twcs.csv
```

The `.gitignore` excludes all raw data, ZIP archives, and CSV files.

---

## Development Principles

- **Reproducibility** — Every result is reproducible from committed code plus a fresh dataset download.
- **Explainability** — Every engineering decision is recorded in `docs/decision_log.md`.
- **No fabricated metrics** — No accuracy, F1, or evaluation result is reported until actually measured.
- **Incremental implementation** — Components are built and verified one phase at a time.
- **Secret management** — API keys are stored in a local `.env` file; never committed.
- **Interface independence** — Core pipeline logic is separate from any future web interface.
- **Minimal dependencies** — Only packages needed for the current phase are included.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/mohanraj9342/ai-customer-agent.git
cd ai-customer-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Download twcs.csv from Kaggle and save it to:
#      twcs/twcs.csv

# 5. Configure API keys (required in later phases only)
cp .env.example .env
# Edit .env and add your key(s)

# 6. Run tests
pytest -q

# 7. Run dataset inspection
python -m src.data.inspect_dataset
```

---

## Repository Structure

```
ai-customer-agent/
├── data/
│   ├── raw/            # Brand-filtered source data (gitignored)
│   └── processed/      # Cleaned data, splits, index artifacts (gitignored)
├── notebooks/          # Exploratory analysis notebooks
├── src/
│   ├── data/           # Ingestion, filtering, reconstruction scripts
│   ├── classification/ # Intent classifier and taxonomy
│   ├── retrieval/      # Similarity index and retrieval engine
│   ├── generation/     # LLM-based reply generation
│   └── evaluation/     # Metrics, LLM judge, human agreement
├── tests/              # Unit and integration tests (pytest)
├── docs/               # Decision log, brand selection, architecture notes
├── .env.example        # API key template (safe to commit)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Author

**Mohanraj** — [github.com/mohanraj9342](https://github.com/mohanraj9342)
