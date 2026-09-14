# AppleSupport Customer Intent Dataset Documentation

**Dataset Name:** `apple_support_intent_candidates.csv`  
**Dataset Metadata:** `apple_support_intent_metadata.json`  
**Taxonomy Version:** `2.0`  
**Selected Brand:** `AppleSupport`  
**Source Corpus:** Reconstructed Conversation Threads (`data/processed/apple_support/apple_support_threads.jsonl`)  
**Pipeline Implementation:** `src/classification/prepare_intent_dataset.py`  
**Taxonomy Specification:** `docs/apple_support_intent_taxonomy.md`  

---

## 1. Executive Summary

Phase 5 establishes a reproducible, production-grade intent classification preparation pipeline for AppleSupport customer support interactions. Drawing from the 82,105 conversation threads reconstructed in Phase 4, the pipeline isolates the initial customer contact, normalizes textual content, applies transparent rule-based heuristic labeling, detects multi-intent ambiguities, and structures a standardized candidate dataset for subsequent human annotation and supervised model training.

### Key Dataset Metrics
* **Threads Scanned:** 82,105
* **Eligible Customer Threads:** 82,101 (99.995%)
* **Excluded Threads:** 4 (0.005%) — brand-only announcements with zero customer messages
* **Candidate Messages Extracted:** 82,101
* **Duplicate Candidate Tweet IDs:** 0
* **Duplicate Thread IDs:** 0
* **Deterministic Execution:** 100% bitwise identical output across repeated runs (SHA-256 verified)

---

## 2. Classification Unit & Candidate Selection Process

### 2.1 The Classification Unit
In customer support operations, intent classification serves two distinct operational functions:
1. **Front-line Triage / Initial Routing:** Classifying the customer's initial problem statement at the start of an interaction to direct them to self-service resolution, automated diagnostic flows, or the appropriate specialist tier.
2. **Turn-Level Dialogue Act Classification:** Classifying every utterance across a multi-turn conversation (greetings, confirmations, troubleshooting responses).

For the purpose of benchmark dataset preparation, Phase 5 isolates the **initial customer contact** (the earliest customer message in the thread).

### 2.2 Selection Criteria & Tie-Breaking
For each reconstructed thread in `apple_support_threads.jsonl`:
1. Filter all messages belonging to the thread where `inbound == 'True'`.
2. Sort eligible customer messages chronologically by `created_at` (parsed ISO-8601 UTC timestamp).
3. In the rare event of identical timestamps within a thread, break ties deterministically using ascending integer order of `tweet_id`.
4. Select the earliest message as the canonical candidate.
5. If a thread contains zero messages with `inbound == 'True'`, exclude the thread and record the exclusion reason (`no_customer_message`). Exactly 4 threads out of 82,105 met this exclusion condition.

### 2.3 Preserved Dialogue Linkage
Each candidate row retains its parent `thread_id`. While the candidate text represents the initial contact utterance, downstream retrieval and dialogue modeling systems can traverse back to `apple_support_threads.jsonl` using `thread_id` to access the full multi-turn context, agent responses, and temporal timeline.

---

## 3. Rule-Based Preliminary Labeling Methodology

### 3.1 Text Normalization
Before applying pattern matching, the raw customer message text is normalized to produce `cleaned_text`:
* Twitter handle mentions (`@AppleSupport`, `@username`) are stripped.
* URLs (`https://t.co/...`) are removed.
* Extra whitespace, leading/trailing punctuation, and line breaks are collapsed.
* Text is lowercased for case-insensitive matching while preserving word boundaries.

### 3.2 Pattern Matching & Heuristic Scoring
The labeling engine evaluates regular expressions representing canonical technical support vocabulary grounded in actual Apple customer inquiries across 10 operational domains:
1. `software_update`: OS updates (iOS, macOS, High Sierra), bugs, glitches, install failures, autocorrect bugs.
2. `battery_power`: Rapid battery drain, percentage drops, overheating, charger/cable defects.
3. `device_hardware`: Cracked screens, unresponsive touch displays, black camera screens, broken buttons/speakers.
4. `app_or_service_issue`: Apple native services (Music, iMessage, FaceTime, Safari, App Store, iCloud sync).
5. `account_access`: Apple ID locked, password reset, 2FA verification code failures.
6. `connectivity_network`: Wi-Fi disconnects, Bluetooth pairing failure, cellular "No Service", dropped calls.
7. `billing_payment`: Unrecognized credit card charges, subscription renewals, refund requests, Apple Pay declines.
8. `order_shipping`: Online store order tracking, shipment delays, delivery status, courier tracking.
9. `feature_how_to`: How-to questions regarding feature configuration and settings.
10. `complaint_feedback`: Negative sentiment vents, dissatisfaction with service or design decisions.

Each category calculates a match score based on keyword and phrase density.

### 3.3 Conflict & Ambiguity Detection (`needs_review`)
Forcing an arbitrary single label on a multi-intent inquiry creates label noise that harms downstream classifier performance. The engine flags ambiguous candidates as `needs_review` under two conditions:
* **Multi-Category Matches:** If two or more distinct categories match with equal positive scores (e.g., a customer complaining that an iOS update caused battery drain).
* **Competing High-Confidence Signals:** If multiple categories have strong keyword matches without a decisive winner.

When flagged, the candidate receives:
* `preliminary_intent = "needs_review"`
* `preliminary_confidence = "low"`
* `preliminary_label_source = "needs_review"`
* `preliminary_rule_id = "conflict: [category_1, category_2]"`
* `rule_overlap_flag = True`

### 3.4 Fallback Category (`unknown_other`)
If a message matches zero domain rules, it defaults to:
* `preliminary_intent = "unknown_other"`
* `preliminary_confidence = "low"`
* `preliminary_label_source = "fallback"`
* `preliminary_rule_id = "fallback:no_rules_matched"`

In the empirical corpus, this category comprises 45.88% of initial messages. Many of these are conversational greetings (`@AppleSupport help please`), invitations to direct message (`@AppleSupport DM me`), or image-only tweets with minimal accompanying text.

---

## 4. Difference Between Preliminary Labels and Ground Truth

> [!IMPORTANT]
> Preliminary labels generated in Phase 5 are **heuristic rule suggestions**. They are designed for data exploration, cohort stratification, active learning prioritization, and human annotation bootstrapping. They **must never be evaluated as verified ground truth** or reported as test-set benchmarks.

### Structural Comparison
| Dimension | Preliminary Labels (Phase 5) | Verified Ground Truth (Phase 6+) |
| :--- | :--- | :--- |
| **Generation Mechanism** | Deterministic regex heuristics | Trained human annotators / adjudicated consensus |
| **Ambiguity Handling** | Automatically routed to `needs_review` | Adjudicated via annotation guidelines |
| **Coverage** | 100% of candidate messages | Gold standard curated evaluation subsets |
| **Confidence Scoring** | Heuristic tiers (`high`, `medium`, `low`) | Inter-annotator agreement metrics (Cohen's / Fleiss' kappa) |
| **Downstream Role** | Candidate stratification, weak supervision | Benchmark validation, test set evaluation |

### Human Review Fields
To support structured human-in-the-loop review, every candidate row includes empty audit columns:
* `verified_intent`: Final adjudicated intent label.
* `verified_by`: Annotator ID or review pool identifier.
* `verification_date`: ISO-8601 timestamp of annotation.
* `verification_status`: Status enum (`unreviewed`, `verified`, `corrected`, `flagged`).
* `notes`: Free-text field for edge cases, multi-intent notes, or dialectical nuances.

---

## 5. Strict Data Leakage Protections

Data leakage is a severe risk in conversational NLP, where model performance is artificially inflated by using future dialogue information not available at runtime. Phase 5 enforces three strict protections:

1. **Zero Access to Brand Response Text:**
   The candidate selection and labeling engine inspects **only customer messages** (`inbound == 'True'`). AppleSupport agent replies are never accessed, parsed, or used as feature inputs for intent classification.
2. **Zero Use of Terminal Thread Metadata:**
   Metadata flags such as `ends_with_brand_reply`, `is_complete`, `turn_count`, or `duration_seconds` are preserved strictly as conversation context in metadata. They are never used as features or labels for intent categorization.
3. **No Target Leakage from Future Customer Turns:**
   The candidate isolates the *initial* customer utterance. Subsequent customer clarifications are excluded from the candidate text to ensure the classification task reflects real-world front-line triage.

---

## 6. Dataset Schema

The generated candidate dataset `data/processed/apple_support/apple_support_intent_candidates.csv` consists of 18 columns:

| Column Name | Data Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `thread_id` | string | No | Foreign key linking to reconstructed thread in `apple_support_threads.jsonl`. |
| `tweet_id` | integer | No | Unique Twitter message identifier of the customer inquiry. |
| `author_id` | string | No | Anonymized customer user identifier. |
| `created_at` | string | No | Timestamp of message creation in UTC (ISO-8601). |
| `text` | string | No | Original raw text of the customer message. |
| `cleaned_text` | string | No | Preprocessed text (handles and URLs stripped, lowercased, trimmed). |
| `preliminary_intent` | string | No | Rule-based preliminary intent category or `needs_review`. |
| `preliminary_confidence` | string | No | Confidence tier: `high`, `medium`, or `low`. |
| `preliminary_rule_id` | string | No | Identifier of the rule that produced the label. |
| `preliminary_label_source` | string | No | Source category: `rule`, `fallback`, or `needs_review`. |
| `matched_intents` | string | No | Pipe-delimited list of all intent categories matched by rules. |
| `match_count` | integer | No | Total number of distinct categories matched. |
| `rule_overlap_flag` | boolean | No | `True` if multiple categories matched; `False` otherwise. |
| `verified_intent` | string | Yes | Human-verified ground truth intent (empty pending review). |
| `verified_by` | string | Yes | Identifier of the annotator (empty pending review). |
| `verification_date` | string | Yes | Timestamp of verification (empty pending review). |
| `verification_status` | string | No | Status indicator, initialized to `"unreviewed"`. |
| `notes` | string | Yes | Free-text notes for annotators (empty pending review). |

---

## 7. Empirical Statistics & Distributions

### 7.1 Distribution of Preliminary Intents (N = 82,101)
| Category | Candidate Count | Percentage | Source Breakdown | Confidence Breakdown |
| :--- | :--- | :--- | :--- | :--- |
| `unknown_other` | 37,669 | 45.88% | 37,669 fallback | 37,669 low |
| `software_update` | 22,942 | 27.94% | 22,942 rule | 19,451 high, 3,491 med |
| `needs_review` | 5,895 | 7.18% | 5,895 needs_review | 5,895 low |
| `battery_power` | 3,406 | 4.15% | 3,406 rule | 1,932 high, 1,474 med |
| `app_or_service_issue` | 2,365 | 2.88% | 2,365 rule | 1,180 high, 1,185 med |
| `account_access` | 2,166 | 2.64% | 2,166 rule | 1,324 high, 842 med |
| `device_hardware` | 2,138 | 2.60% | 2,138 rule | 722 high, 1,416 med |
| `feature_how_to` | 1,889 | 2.30% | 1,889 rule | 592 high, 1,297 med |
| `connectivity_network` | 1,717 | 2.09% | 1,717 rule | 450 high, 1,267 med |
| `complaint_feedback` | 1,105 | 1.35% | 1,105 rule | 260 high, 845 med |
| `billing_payment` | 607 | 0.74% | 607 rule | 34 high, 573 med |
| `order_shipping` | 202 | 0.25% | 202 rule | 6 high, 196 med |
| **Total** | **82,101** | **100.00%** | — | — |

### 7.2 Breakdown by Label Source & Confidence
* **Label Source:**
  * `rule`: 38,537 (46.94%)
  * `fallback`: 37,669 (45.88%)
  * `needs_review`: 5,895 (7.18%)
* **Confidence Tiers:**
  * `high`: 25,951 (31.61%)
  * `medium`: 12,586 (15.33%)
  * `low`: 43,564 (53.06%) — includes all fallback and needs_review items

### 7.3 Candidate Utterance Text Statistics
* **Character Length:**
  * Min: 7 characters
  * Median: 111 characters
  * Mean: 112.8 characters
  * Max: 362 characters (extended Twitter card format)
* **Word Count:**
  * Min: 1 word
  * Median: 19 words
  * Mean: 19.5 words
  * Max: 69 words

---

## 8. Dataset Limitations & Boundary Conditions

1. **Extraction Boundary Constraint:**
   Customer messages were extracted in Phase 3 conditionally based on AppleSupport replies. Unanswered customer inquiries (inquiries where AppleSupport never responded) are structurally absent from the Kaggle corpus. Therefore, this dataset reflects the distribution of *serviced* customer inquiries.
2. **Twitter Microblogging Format:**
   Inquiries reflect informal Twitter constraints (historical 140-character limits, abbreviations, non-standard spelling, colloquial slang, emojis).
3. **Multi-Turn Evolution:**
   A customer's initial tweet may be brief or ambiguous (`@AppleSupport my phone is broken`), with technical details revealed only in turn 3 or 4. The candidate dataset isolates the initial contact intent; subsequent turns are available in `apple_support_threads.jsonl`.
4. **Temporal Context:**
   The source dataset spans late 2017. As a result, software update inquiries prominently feature `iOS 11` issues (e.g., the infamous autocorrect bug). The taxonomy categories remain generalizable, but topic models must account for this temporal focus.

---

## 9. Verification & Determinism Results

* **Reproducibility:** Executing `python src/classification/prepare_intent_dataset.py` twice consecutively produced identical outputs with SHA-256 hash:
  `1d574aa007bedb2e36b7b3f5c3cb32c92f2bd1c264e63ee96218b9dff34d9bb7`.
* **Automated Unit Tests:** 24 unit tests in `tests/test_prepare_intent_dataset.py` validate selection determinism, brand non-selection, tie-breaking, conflict routing, leakage prevention, overwrite guards, and metadata reconciliations.
* **Full Suite Integration:** All 272 automated tests in the repository pass without warnings or errors.
