# AI Customer Support Agent — Hiver SDE Intern Take-Home Assignment

[![Status](https://img.shields.io/badge/status-Phase%200%20%E2%80%94%20Scaffolding-yellow)]()
[![Python](https://img.shields.io/badge/python-3.14-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Project Overview

This project builds a Twitter-based AI customer support agent for a selected brand, using the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (Kaggle: `thoughtvector/customer-support-on-twitter`).

The agent must:

1. **Classify** each incoming customer tweet into a small, well-defined set of support intents.
2. **Retrieve** historically similar resolved conversations from the same brand.
3. **Draft a reply** grounded in how the brand previously resolved similar issues.
4. **Decide** whether the case should be handled automatically or escalated to a human agent,
   and provide a clear reason for the decision.

---

## Current Status

> **Phase 0 — Project Scaffolding**
>
> The directory structure, `.gitignore`, `README.md`, and `requirements.txt` have been created.
> No model has been trained. No evaluation has been run. No metrics have been reported.
> Implementation begins in Phase 1.

---

## Planned Components

| Component | Description | Status |
|---|---|---|
| Data ingestion | Extract and filter brand-specific conversations from the dataset | Not started |
| Intent discovery | Surface natural topic clusters from real customer messages | Not started |
| Intent labelling | Define 6–8 named intent classes; hand-label a training set | Not started |
| Intent classifier | Train a baseline (TF-IDF + Logistic Regression) and a main classifier | Not started |
| Retrieval index | Build a similarity index over historical brand responses | Not started |
| Reply generation | Use an LLM API to draft grounded replies from retrieved evidence | Not started |
| Escalation logic | Decide auto-handle vs. human escalation; output a reason | Not started |
| Evaluation harness | Automated metrics + LLM-as-judge + human agreement measurement | Not started |
| Golden evaluation set | 150–250 hand-labelled examples sampled from held-out data | Not started |
| Report and decision log | 6-page report + 10–15-item engineering decision log | Not started |

---

## Planned Development Phases

| Phase | Objective |
|---|---|
| **0** | Project scaffolding (current) |
| **1** | Safe dataset inspection and brand selection |
| **2** | Data extraction, cleaning, and conversation reconstruction |
| **3** | Intent discovery and labelling |
| **4** | Intent classifier — baseline and main system |
| **5** | Retrieval index construction |
| **6** | LLM-based reply generation (pending API approval) |
| **7** | Escalation decision logic |
| **8** | Golden evaluation set creation |
| **9** | Evaluation harness: automated metrics + LLM judge |
| **10** | Report, decision log, and README polish |

---

## Dataset Note

The dataset (`twcs/twcs.csv`, ~493 MB; `archive.zip`, ~169 MB) is stored **locally only**.

- **Do not** commit the dataset to this repository.
- **Do not** download another copy of the dataset.
- The `.gitignore` excludes all raw data, ZIP archives, and CSV files.
- If you are reproducing this project, download the dataset from Kaggle:
  `kaggle datasets download thoughtvector/customer-support-on-twitter`
  and place the extracted `twcs/twcs.csv` at the project root under `twcs/`.

---

## Development Principles

This project is developed according to the following engineering principles:

- **Reproducibility**: Every result must be reproducible from the committed code and a fresh dataset download.
- **Explainability**: Every implementation decision is explained in `docs/decision_log.md`.
- **No fabricated metrics**: No accuracy score, F1-score, retrieval quality, or evaluation result is reported until it is actually measured.
- **Incremental implementation**: Components are built and verified one phase at a time.
- **Privacy and secret management**: API keys and credentials are stored in a local `.env` file, never committed.
- **Minimal dependencies**: Only packages genuinely needed for each phase are added.

---

## Setup Instructions

> ⚠️ **This section will be completed after the development environment is verified in Phase 1.**

The following steps are placeholders and will be updated with exact, tested commands.

```bash
# 1. Clone the repository
git clone https://github.com/mohanraj9342/ai-customer-agent.git
cd ai-customer-agent

# 2. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Download twcs.csv from Kaggle and place it at:
#   twcs/twcs.csv

# 5. Configure API keys (when needed in later phases)
cp .env.example .env
# Edit .env and fill in your API key(s)
```

---

## Repository Structure

```
hiver-ai-support-agent/
├── data/
│   ├── raw/            # Brand-filtered, cleaned source data (gitignored)
│   └── processed/      # Labelled intents, train/dev/test splits (gitignored)
├── notebooks/          # Exploratory analysis notebooks
├── src/
│   ├── data/           # Data ingestion, filtering, cleaning scripts
│   ├── classification/ # Intent classifier (baseline + main)
│   ├── retrieval/      # Similarity index and retrieval engine
│   ├── generation/     # LLM-based reply generation
│   └── evaluation/     # Metrics, LLM judge, human agreement
├── tests/              # Unit and integration tests (pytest)
├── docs/               # Report, decision log, architecture diagrams
├── .env.example        # Template for API keys (safe to commit)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Contact

**Candidate:** Mohanraj
**Repository:** https://github.com/mohanraj9342/ai-customer-agent
