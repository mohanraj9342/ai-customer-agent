# AppleSupport Intent Classification Golden Evaluation Set Specification

**Document Version:** 1.0  
**Date:** 2026-09-15  
**Taxonomy Version:** `2.0`  
**Dataset Artifact:** `data/processed/apple_support/apple_support_intent_golden_set.csv` (158 records)  
**Metadata Artifact:** `data/processed/apple_support/apple_support_intent_golden_metadata.json`  
**Associated Module:** `src/classification/build_golden_evaluation_set.py`  
**Annotation Guide:** `docs/apple_support_intent_annotation_guide.md`  
**Audit & Annotation Report:** `docs/apple_support_intent_pilot_annotation_report.md`  

---

## 1. Overview & Purpose

The **AppleSupport Intent Classification Golden Evaluation Set** is the canonical, human-verified benchmark for evaluating intent classification systems developed in this project. 

In automated customer service pipelines, inbound intent classification serves as front-line triage: routing incoming customer messages to self-service knowledge articles, automated diagnostic flows, or specialized human support queues. To reliably benchmark classifier performance, the evaluation set must satisfy four strict engineering properties:
1. **Human-Verified Ground Truth:** Every record has been individually audited, verified, and justified by human reviewers according to standardized taxonomy guidelines.
2. **Strict Zero Data Leakage:** Ground truth is established solely from the customer's initial problem formulation (`text`). No brand replies, subsequent conversation turns, or dialogue resolution outcomes were inspected.
3. **Guaranteed Training Isolation:** Golden set records are permanently quarantined from model training. Any candidate pool used for training must strictly satisfy $S_{\text{train}} \cap S_{\text{golden}} = \emptyset$.
4. **Fine-Grained Difficulty Slicing:** Records are annotated with operational difficulty tiers to allow slice-based error analysis (measuring performance on representative versus boundary, conflict, and ambiguous cases).

---

## 2. Sampling Methodology & Dataset Construction

The golden set is constructed deterministically from the 82,101 first-inbound customer messages extracted in Phase 5 (`data/processed/apple_support/apple_support_intent_candidates.csv`).

### 2.1 Multi-Slice Stratification (Seed 42)
Sampling was conducted across 18 targeted strata to ensure coverage of high-frequency domains, rare intents, multi-rule collisions, and known linguistic edge cases:

| Stratum Type | Sub-strata Included | Target Quota | Sampled Records |
| :--- | :--- | :---: | :---: |
| **Taxonomy Baseline Intents** | 12 classes (`account_access`, `app_or_service_issue`, `battery_power`, `billing_payment`, `complaint_feedback`, `connectivity_network`, `device_hardware`, `feature_how_to`, `needs_review`, `order_shipping`, `software_update`, `unknown_other`) | 7 each | 84 |
| **Rule Overlap Cohort** | Inquiries matching 2 or more conflicting regex rules | 20 | 20 |
| **Software Update Edge Cases** | Boundary cases: App Store app updates, payment updates, shipping updates | 12 | 12 |
| **Unknown / Other Recovery** | Keyword fallback candidates targeted for intent recovery | 12 | 12 |
| **Needs Review Conflicts** | Equal-confidence multi-rule collisions | 12 | 12 |
| **Short / Noisy Text** | Microblogging fragments (< 25 characters) | 10 | 10 |
| **URL / Media Only** | Inquiries containing standalone images/links | 8 | 8 |
| **Total Golden Set** | | **158** | **158** |

### 2.2 Preserved Integrity & Determinism
- **Total Records:** 158 (satisfying the 150–250 range requirement).
- **Duplicate Checks:** 0 duplicate `tweet_id`s, 0 duplicate `thread_id`s.
- **Deterministic Content SHA-256:** `df8846e6dcb7ed559c766d2ab30f7e8f54df466f49ea30722cdf19aee481cbd9`.

---

## 3. Intent Distribution & Adjudication Summary

Every record was manually reviewed and verified under Taxonomy Version 2.0 (11 domain classes + `unknown_other` fallback + `needs_review` ambiguity flag).

### 3.1 Class Distribution

| Golden Intent Category | Count | Percentage | Operational Support Role |
| :--- | :---: | :---: | :--- |
| `software_update` | 32 | 20.25% | OS installation issues, iOS system bugs, autocorrect glitch |
| `battery_power` | 26 | 16.46% | Battery drain, overheating, charging hardware failures |
| `app_or_service_issue` | 20 | 12.66% | Native app crashes (Music, Mail, Maps, Photos) & App Store |
| `unknown_other` | 14 | 8.86% | Conversational greetings, uninterpretable fragments |
| `billing_payment` | 13 | 8.23% | Subscriptions, disputed charges, refund claims, Apple Pay |
| `complaint_feedback` | 11 | 6.96% | Retail store dissatisfaction, Genius Bar policies, venting |
| `feature_how_to` | 10 | 6.33% | Feature configuration, settings location, button shortcuts |
| `order_shipping` | 9 | 5.70% | Online store order tracking, reservation status, store pickup |
| `device_hardware` | 8 | 5.06% | Screen unresponsiveness, camera autofocus, drive failure |
| `connectivity_network` | 7 | 4.43% | Wi-Fi disconnects, Bluetooth pairing, cellular signal |
| `account_access` | 5 | 3.16% | Apple ID lock, password recovery, 2FA codes |
| `needs_review` | 3 | 1.90% | Genuinely irreconcilable multi-intent cases |
| **Total** | **158** | **100.00%** | |

### 3.2 Human Review Verification Status
- **`verified` (76 records, 48.10%):** Preliminary heuristic prediction confirmed as ground truth.
- **`corrected` (79 records, 50.00%):** Preliminary heuristic rule reassigned to correct operational class.
- **`flagged_ambiguous` (3 records, 1.90%):** Preserved as `needs_review`.

### 3.3 Preservation of `needs_review` Cases
Rather than forcing an arbitrary classification, 3 genuinely irreconcilable multi-intent cases are explicitly flagged:
1. **Tweet 410510 (Spanish):** Simultaneous, co-equal complaints of Bluetooth disconnect, screen freezing, and battery drain post-update.
2. **Tweet 965710:** In-store hardware repair damage (`device_hardware`) combined with a £3,250 full purchase price refund dispute (`billing_payment`).
3. **Tweet 1066754:** Catastrophic 20-minute battery life (`battery_power`) alongside total internet connectivity failure (`connectivity_network`).

---

## 4. Operational Difficulty Slices

To support fine-grained diagnostic evaluation in Phase 9, every golden set record is classified into an operational difficulty slice:

| Difficulty Slice | Count | Percentage | Description & Evaluation Purpose |
| :--- | :---: | :---: | :--- |
| `representative` | 70 | 44.30% | Clean, unambiguous, single-intent inquiries. Evaluates baseline recall. |
| `attribution_vs_symptom` | 24 | 15.19% | Inquiries citing an OS update as the cause of a specific symptom (battery, network). Evaluates adherence to Rule 3. |
| `rule_conflict` | 24 | 15.19% | Inquiries triggering multiple competing heuristic rules. Evaluates classifier disambiguation against conflicting keywords. |
| `fallback_recovery` | 18 | 11.39% | Actionable inquiries that preliminary rules failed to match (`unknown_other`). Evaluates model generalization beyond static lexicons. |
| `boundary_disambiguation` | 7 | 4.43% | Boundary collisions (App Store downloads vs OS update, payment method updates, UI rendering vs power). |
| `short_noisy` | 7 | 4.43% | Microblogging fragments (<= 25 chars). Evaluates handling of extreme brevity. |
| `multilingual` | 5 | 3.16% | Non-English inquiries (Spanish, German, Japanese, Portuguese). Evaluates multilingual robustness. |
| `ambiguous_multi_intent` | 3 | 1.90% | Genuinely irreconcilable cases. Evaluates classifier confidence calibration and escalation logic. |
| **Total** | **158** | **100.00%** | |

---

## 5. Training Data Isolation & Anti-Leakage Protocol

To guarantee rigorous holdout evaluation, golden set records must never contaminate training datasets.

### 5.1 Mathematical Isolation Guarantee
Let $S_{\text{candidates}}$ be the 82,101 first-inbound customer tweets, $S_{\text{golden}}$ be the 158 golden evaluation tweets, and $S_{\text{train}}$ be any candidate pool used for training:
$$S_{\text{train}} = S_{\text{candidates}} \setminus S_{\text{golden}}$$
$$S_{\text{train}} \cap S_{\text{golden}} = \emptyset$$

### 5.2 Implementation Safeguards
1. **Isolated Candidates Export:**
   The module `src/classification/build_golden_evaluation_set.py` exports `apple_support_intent_training_candidates.csv` containing exactly **81,943 rows** ($82,101 - 158$).
2. **Programmatic Verification:**
   The utility function `verify_training_isolation(train_df, golden_df)` programmatically asserts that the intersection of tweet IDs is empty. Any overlap immediately raises a `ValueError`.
3. **Golden Tweet ID Registry:**
   All 158 isolated tweet IDs are permanently recorded in `data/processed/apple_support/apple_support_intent_golden_metadata.json` for automated verification.

---

## 6. Schema & Data Dictionary

The golden evaluation set CSV (`data/processed/apple_support/apple_support_intent_golden_set.csv`) contains 35 columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `tweet_id` | integer | Unique identifier of the customer tweet. |
| `thread_id` | string | Unique conversation thread identifier (`thread_{root_id}`). |
| `created_at` | string | Raw Twitter timestamp format. |
| `timestamp` | string | Parsed timestamp string. |
| `text` | string | **Mandatory.** Raw customer tweet text. |
| `cleaned_text` | string | Lightly normalized text (handles stripped, lowercased). |
| `golden_intent` | string | **Primary Ground Truth.** One of 11 taxonomy classes, `unknown_other`, or `needs_review`. |
| `verified_intent` | string | Backward-compatible alias for `golden_intent`. |
| `verification_status` | string | `verified`, `corrected`, or `flagged_ambiguous`. |
| `verified_by` | string | Identifier of the human reviewer (`pilot_annotator_human`). |
| `verification_date` | string | Date annotation was finalized (`2026-09-14`). |
| `notes` | string | **Mandatory.** Rationale explaining adjudication or ambiguity. |
| `evaluation_split` | string | Fixed to `"golden_test"` across all rows. |
| `case_difficulty` | string | One of 8 difficulty tiers for error slicing. |
| `preliminary_intent` | string | Heuristic rule prediction for benchmark comparison. |
| `candidate_intent` | string | Alias of `preliminary_intent`. |
| `preliminary_confidence` | string | Heuristic confidence tier (`high`, `medium`, `low`). |
| `label_confidence` | string | Alias of `preliminary_confidence`. |
| `preliminary_rule_id` | string | Heuristic rule ID triggered. |
| `label_reason` | string | Description of heuristic rule match. |
| `preliminary_label_source` | string | `rule`, `fallback`, or `needs_review`. |
| `label_source` | string | Alias of `preliminary_label_source`. |
| `matched_intents` | string | Pipe-separated list of all heuristic rules that matched. |
| `match_count` | integer | Total number of heuristic rules matched. |
| `rule_overlap_flag` | boolean | True if 2 or more distinct rules matched. |
| `review_priority` | string | Heuristic review priority (`critical`, `high`, `medium`, `normal`). |
| `selection_reason` | string | Fixed to `"first_customer_message_chronological"`. |
| `thread_message_count` | integer | Total messages in the reconstructed conversation thread. |
| `customer_message_count` | integer | Total customer messages in thread. |
| `brand_message_count` | integer | Total brand messages in thread. |
| `has_missing_parent_link` | boolean | Quality flag from thread reconstruction. |
| `has_missing_response_target` | boolean | Quality flag from thread reconstruction. |
| `is_complete` | boolean | Legacy completion flag. |
| `ends_with_brand_reply` | boolean | Audit-verified final brand message indicator. |
| `taxonomy_version` | string | Fixed to `"2.0"`. |

---

## 7. Downstream Usage Guidelines

When evaluating intent classification models (e.g. TF-IDF baselines, embedding classifiers, or LLM-based classifiers):
1. **Load via Official Loader:**
   ```python
   from src.classification.build_golden_evaluation_set import load_golden_evaluation_set
   df_golden = load_golden_evaluation_set()
   ```
2. **Standard Evaluation Metric:**
   Evaluate overall accuracy, macro-F1, and weighted-F1 against `df_golden["golden_intent"]`.
3. **Slice-Level Reporting:**
   Compute metrics broken down by `case_difficulty` to isolate weaknesses (e.g. evaluating accuracy on `attribution_vs_symptom` separately from `representative`).
4. **Handling `needs_review`:**
   In standard 12-class closed-world evaluation, the 3 `needs_review` cases can either be:
   - Excluded from single-class confusion matrices, or
   - Used to evaluate escalation logic (models predicting escalation / low-confidence on ambiguous cases).
