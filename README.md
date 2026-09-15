# AI Customer Support Agent

[![Status](https://img.shields.io/badge/status-Phase%209%20Complete%20%7C%20Phase%2010%20In%20Progress-blue)]()
[![Tests](https://img.shields.io/badge/tests-330%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.14-blue)]()
[![Brand](https://img.shields.io/badge/brand-AppleSupport-lightgrey)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

An end-to-end AI customer-support agent built on the publicly available [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (`thoughtvector/customer-support-on-twitter` on Kaggle), specifically optimized and grounded on **AppleSupport** (the largest enterprise support brand in the dataset).

The agent architecture performs:
1. **Intent Classification**: Classifies incoming customer support tweets into an empirically grounded 11-category customer-care domain taxonomy.
2. **Context-Aware Retrieval**: Retrieves historically similar, verified customer-support resolutions from the same brand.
3. **Grounded Reply Generation**: Drafts accurate, helpful replies adhering strictly to brand voice and historical resolutions.
4. **Escalation Routing**: Triages cases between automated resolution and human escalation with clear, auditable reasoning and confidence margins.

---

## Architecture Overview

```
[ Incoming Customer Tweet ]
             │
             ▼
┌─────────────────────────────┐
│ 1. Intent Classification     │ ◄── TF-IDF + Logistic Regression (L2, Balanced)
│    (11-Domain Taxonomy)     │     Benchmarked on Golden Set (Macro-F1: 63.2%)
└────────────┬────────────────┘
             │
             ├── Low Confidence / Multi-Intent Margin (< 0.20) ──► [ Human Agent Escalation ]
             │
             ▼ High Confidence Intent
┌─────────────────────────────┐
│ 2. Similar Issue Retrieval  │ ◄── Historical Resolved Thread Index (82k+ Threads)
│    (Context Extraction)     │     Dense Semantic + Lexical Matching
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ 3. Grounded Reply Drafting  │ ◄── LLM Response Engine Grounded on Retrieved Cases
│    (Brand Voice Alignment)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ 4. Triage & Quality Gate    │ ─── Risk Check: Account Lock / Billing Dispute / Ambiguity
│    (Automate vs. Escalate)  │
└─────────────────────────────┘
```

---

## Project Status & Completed Phases

| Phase | Description | Status | Deliverables / Metrics |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Dataset Inspection | ✅ Complete | Full schema profile, brand volume breakdown (`docs/dataset_inspection.json`). |
| **Phase 2** | Brand Selection | ✅ Complete | Selected `AppleSupport` across 108 brands based on 213k+ volume and thread depth (`docs/brand_selection.md`). |
| **Phase 3** | Brand Extraction | ✅ Complete | Filtered 213,483 AppleSupport messages with 100% integrity validation (`src/data/extract_selected_brand.py`). |
| **Phase 4** | Thread Reconstruction | ✅ Complete | Reconstructed 82,105 conversation trees; validated topological ordering and link integrity (`docs/apple_support_thread_reconstruction.md`). |
| **Phase 5** | Intent Taxonomy & Dataset | ✅ Complete | Defined 11-domain taxonomy (v2.0) and prepared 82,101 initial classification candidates (`docs/apple_support_intent_taxonomy.md`). |
| **Phase 6** | Quality Audit & Annotation Guide | ✅ Complete | Audited heuristic edge cases; drafted comprehensive 7-section annotation guidelines (`docs/apple_support_intent_annotation_guide.md`). |
| **Phase 7** | Pilot Human Annotation | ✅ Complete | Audited 158 pilot customer messages, establishing baseline agreement and disambiguation rules (`docs/decision_log.md`). |
| **Phase 8** | Golden Evaluation Set | ✅ Complete | Created 158-record human-verified Golden Evaluation Set across 8 difficulty slices with strict training isolation ($S_{\text{train}} \cap S_{\text{golden}} = \emptyset$). |
| **Phase 9** | Intent Classifier Baseline & Main Model | ✅ Complete | Implemented Majority Baseline, Linear SVM, and Logistic Regression; benchmarked on Golden Set (`docs/apple_support_intent_classifier_evaluation.md`). |
| **Phase 10** | Retrieval Index Construction | 🔄 In Progress | Vector embedding and lexical index over historical resolved conversations. |
| **Phase 11** | Grounded Reply Generation | ⏳ Upcoming | LLM-based reply drafting grounded in retrieved historical resolutions. |
| **Phase 12** | Escalation Decision Logic | ⏳ Upcoming | Automated triage rules and confidence-based routing. |
| **Phase 13** | Evaluation Harness & Public Interface | ⏳ Upcoming | End-to-end evaluation harness, metrics reporting, and interactive interface. |

---

## Intent Classification Benchmark Results (Phase 9)

All models are trained strictly on isolated training candidates ($N = 76,071$) and evaluated against the human-verified **Golden Evaluation Set** (155 closed-world resolved records + 3 ambiguous cases):

### Comparative Model Performance

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 9.03% | 0.0151 | 0.0150 | 0.0082 | 0.0909 |
| **TF-IDF + Linear SVM** | 54.84% | 0.5961 | 0.5527 | 0.6600 | 0.6214 |
| **TF-IDF + Logistic Regression (Main)** | **59.35%** | **0.6319** | **0.5964** | **0.6765** | **0.6622** |

### Per-Intent Breakdown (Best Model: Logistic Regression)

| Intent Domain | Precision | Recall | F1-Score | Support | Diagnostic Capabilities |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `connectivity_network` | 85.7% | 85.7% | **85.7%** | 7 | Exceptional detection of Wi-Fi, Bluetooth, cellular, and signal drop issues. |
| `order_shipping` | 87.5% | 77.8% | **82.4%** | 9 | High recall despite rare class frequency (195 training samples), validating balanced weighting. |
| `billing_payment` | 90.0% | 69.2% | **78.3%** | 13 | High precision; accurately classifies subscription charges, refunds, and store purchases. |
| `account_access` | 62.5% | 100.0% | **76.9%** | 5 | Perfect recall (100%); reliably routes locked Apple IDs and two-factor verification issues. |
| `device_hardware` | 66.7% | 75.0% | **70.6%** | 8 | Captures physical defects (cracked screens, speaker distortion, physical buttons). |
| `battery_power` | 88.2% | 57.7% | **69.8%** | 26 | High precision (88.2%); distinguishes battery drain from general OS inquiries. |
| `feature_how_to` | 46.7% | 70.0% | **56.0%** | 10 | Captures user guidance and configuration questions. |
| `software_update` | 51.6% | 50.0% | **50.8%** | 32 | Separates update bugs and patch inquiries from underlying symptom categories. |
| `unknown_other` | 35.3% | 85.7% | **50.0%** | 14 | Conservative fallback capturing out-of-domain or ambiguous noise. |
| `app_or_service_issue` | 100.0% | 30.0% | **46.2%** | 20 | 100% precision on core first-party apps (Apple Music, App Store, Podcasts). |
| `complaint_feedback` | 30.0% | 27.3% | **28.6%** | 11 | Detects general customer dissatisfaction and store experience complaints. |

### Slice-Level Performance Across Case Difficulty Tiers
- **Rule Conflict Resolution**: **79.17%** accuracy (19/24) on complex multi-rule boundary cases.
- **Representative Production Cases**: **75.71%** accuracy (53/70) on standard customer tweets.
- **Short / Noisy Tweets**: **71.43%** accuracy (5/7) on microblogging fragments with typos or slang.
- **Symptom vs. Attribution**: **54.17%** accuracy (13/24) on causal post-update complaints.
- **Ambiguous Multi-Intent Diagnostic**: On genuine multi-intent inquiries (e.g. 20-min battery drain vs cellular internet failure), the model splits probability mass ($P \approx 0.53$, margin $< 0.14$), providing a reliable trigger for human agent escalation.

---

## Repository Structure

```
ai-customer-agent/
├── data/
│   ├── raw/                           # Raw downloaded source dataset (gitignored)
│   └── processed/apple_support/       # Cleaned, reconstructed, and evaluated data
│       ├── apple_support_messages.csv # Extracted AppleSupport messages (gitignored)
│       ├── apple_support_threads.jsonl# Reconstructed conversation trees (gitignored)
│       ├── apple_support_intent_golden_set.csv           # Human-verified Golden Set (N=158)
│       ├── apple_support_intent_golden_metadata.json      # Golden set SHA-256 & schema metadata
│       ├── apple_support_intent_training_candidates.csv  # Isolated training candidate pool (N=81,943)
│       └── apple_support_intent_evaluation_results.json  # Comprehensive benchmark results
├── models/
│   └── intent_classifier/             # Serialized production model artifacts (gitignored)
│       ├── tfidf_vectorizer.joblib    # Fitted sublinear TF-IDF vectorizer (10,000 features)
│       ├── logistic_regression_model.joblib # Trained Logistic Regression model
│       ├── linear_svm_model.joblib    # Trained LinearSVC model
│       ├── majority_baseline_model.joblib   # Majority class baseline
│       └── model_metadata.json        # Training configuration & metrics metadata
├── src/
│   ├── data/                          # Data extraction and conversation reconstruction
│   │   ├── extract_selected_brand.py  # Phase 3 extraction pipeline
│   │   └── reconstruct_threads.py     # Phase 4 topological thread builder
│   └── classification/                # Intent classification subsystem
│       ├── intents.yaml               # Canonical 11-domain taxonomy specification (v2.0)
│       ├── intent_loader.py           # Taxonomy parser and validator
│       ├── baseline_classifier.py     # Majority class baseline estimator
│       ├── build_golden_evaluation_set.py # Golden set builder & isolation verification
│       └── train_eval_intent_classifier.py# End-to-end training & evaluation pipeline
├── tests/                             # Full automated test suite (330 tests)
│   ├── test_extract_selected_brand.py # Extraction validation tests
│   ├── test_reconstruct_threads.py    # Thread reconstruction & graph cycle tests
│   ├── test_intent_loader.py          # Taxonomy parser tests
│   ├── test_golden_evaluation_set.py  # Golden set schema, slices, & isolation tests
│   └── test_train_eval_intent_classifier.py # Model training, reproducibility, & eval tests
├── docs/                              # Architecture and design documentation
│   ├── decision_log.md                # 18 formal Architectural Decision Records (ADRs)
│   ├── brand_selection.md             # Empirical brand selection analysis
│   ├── apple_support_intent_taxonomy.md # Taxonomy specification and criteria
│   ├── apple_support_intent_annotation_guide.md # Human annotation guide and boundary rules
│   ├── apple_support_intent_golden_evaluation_set.md # Golden benchmark documentation
│   └── apple_support_intent_classifier_evaluation.md # Benchmark evaluation report
├── requirements.txt                   # Production and testing dependencies
└── README.md
```

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/mohanraj9342/ai-customer-agent.git
cd ai-customer-agent

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup (Local Storage Only)

Download the dataset (`twcs.csv`, ~493 MB) from Kaggle:
```bash
kaggle datasets download thoughtvector/customer-support-on-twitter
unzip customer-support-on-twitter.zip -d twcs/
# Confirms file exists at twcs/twcs.csv
```

### 3. Run Pipeline Stages

```bash
# Extract AppleSupport brand messages
python -m src.data.extract_selected_brand

# Reconstruct conversation threads
python -m src.data.reconstruct_threads

# Build human-verified Golden Evaluation Set & isolated training candidates
python -m src.classification.build_golden_evaluation_set

# Train intent classifiers, evaluate against Golden Set, and save artifacts
python -m src.classification.train_eval_intent_classifier --train --evaluate --save-models
```

### 4. Run Production API & Frontend

```bash
# Terminal 1: Run Production FastAPI backend (Render environment)
.venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Run React + Vite frontend (Vercel environment)
npm --prefix frontend install
npm --prefix frontend run dev
```

Visit [http://127.0.0.1:5173](http://127.0.0.1:5173) to interact with the web interface.

### 5. Run Automated Test Suite

```bash
# Python API & ML pipeline regression suite (434 tests)
pytest -q

# React + Vite frontend unit & integration test suite (13 tests)
npm --prefix frontend test
```

---

## Engineering Decision Log

All major architectural choices, trade-offs, and design rationales are formally recorded in `docs/decision_log.md`:
* **Decision 1–12**: Project setup, metric reporting, brand selection (`AppleSupport`), thread graph reconstruction, topological cycle detection, and extraction bias mitigation.
* **Decision 13–15**: Empirical taxonomy design (v2.0), customer triage unit definition, heuristic conflict labeling, and boundary disambiguation.
* **Decision 16**: Human annotation pilot audit and multi-intent edge-case methodology.
* **Decision 17**: Golden Evaluation Set construction (158 records), 8-tier case difficulty taxonomy, and mathematical training data isolation ($S_{\text{train}} \cap S_{\text{golden}} = \emptyset$).
* **Decision 18–22**: Intent classifier model architecture selection, grounded agent response generation, and independent quality review.
* **Decision 23–24**: Phase 14 end-to-end benchmark evaluation and self-retrieval candidate exclusion.
* **Decision 25–26**: Phase 15 FastAPI ASGI backend, Render deployment, and CORS origin restriction.
* **Decision 27**: Phase 16 Vercel frontend architecture, React + Vite SPA, and client-side secret safety.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

