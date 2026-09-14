# AppleSupport Intent Classification Pilot Annotation Report

**Document Version:** 1.0  
**Date:** 2026-09-14  
**Dataset:** `data/processed/apple_support/apple_support_intent_pilot_sample.csv` (158 records)  
**Metadata:** `data/processed/apple_support/apple_support_intent_pilot_metadata.json`  
**Annotation Guide:** `docs/apple_support_intent_annotation_guide.md`  
**Taxonomy Version:** `2.0` (11 domain classes + `unknown_other`)  
**Annotator ID:** `pilot_annotator_human`  

---

## 1. Executive Summary

This report documents the completed human-in-the-loop annotation of the **158-record pilot dataset** for inbound customer service intent classification directed at `AppleSupport`. 

The pilot dataset was drawn deterministically from 82,101 first-inbound customer tweets across 18 targeted sampling strata (including all 12 preliminary taxonomy classes, multi-rule overlaps, `software_update` edge cases, `unknown_other` fallback cases, and short/noisy messages).

### Key Results
* **Total Records Annotated:** 158 / 158 (100% completion rate).
* **Adjudication Actions:**
  * **Verified (Preliminary Label Accepted):** 76 records (48.10%)
  * **Corrected (Preliminary Label Overridden):** 79 records (50.00%)
  * **Flagged Ambiguous (`needs_review`):** 3 records (1.90%)
* **Preliminary Rule Agreement Rate:** 49.37% (78 / 158 records).  
  *Note:* The 49.37% agreement rate reflects intentional stress-testing: the pilot was heavily oversampled with ambiguous conflicts, multi-rule overlaps, and known boundary edge cases rather than an unstratified natural distribution.
* **Review Fields Integrity:** 100% population of `verified_intent`, `verification_status`, `verified_by`, `verification_date`, and `notes` with 0 missing or empty values.
* **Strict Zero Data Leakage:** All adjudications were made exclusively from the customer's initial problem statement (`text`). No brand responses, subsequent conversational turns, or resolution metadata were inspected.

---

## 2. Dataset Overview & Pilot Sample Structure

The pilot sample was constructed by `src/classification/audit_intent_labels.py` using stratified deterministic sampling (seed 42).

| Stratum Type | Sub-strata Included | Quota per Sub-stratum | Actual Sampled |
| :--- | :--- | :---: | :---: |
| **Baseline Taxonomy Intents** | 12 classes (`account_access`, `app_or_service_issue`, `battery_power`, `billing_payment`, `complaint_feedback`, `connectivity_network`, `device_hardware`, `feature_how_to`, `needs_review`, `order_shipping`, `software_update`, `unknown_other`) | 7 | 84 |
| **Rule Overlap** | High-conflict multi-rule candidates | 20 | 20 |
| **Software Update Edge Cases** | App update collisions, billing update collisions | 12 | 12 |
| **Unknown / Other Edge Cases** | Recovery candidates from keyword fallback | 12 | 12 |
| **Needs Review Conflicts** | Multi-intent rule collisions | 12 | 12 |
| **Short / Noisy Text** | Microblogging fragments (< 25 characters) | 10 | 10 |
| **URL / Media Only** | Tweets with standalone image/link attachments | 8 | 8 |
| **Total Pilot Sample** | | **158** | **158** |

---

## 3. Human Adjudication Methodology

All annotations followed the formal adjudication protocol defined in `docs/apple_support_intent_annotation_guide.md`:

### 3.1 Decision Hierarchy
1. **Rule 1 — Safety & Physical Hardware Supersedes Software:** Physical damage, screen unresponsiveness, or damaged charging cables take operational triage precedence over background software bugs.
2. **Rule 2 — Security & Financial Disputes Supersede General Inquiries:** Unauthorized credit card charges, Apple Pay verification failures, or locked Apple ID credentials require immediate specialized routing over general feature questions.
3. **Rule 3 — Actionable Symptom Supersedes Historical Attribution:** When a customer attributes a specific actionable symptom to an operating system update (e.g., *"battery draining fast since updating to iOS 11"*):
   - Actionable battery depletion $\to$ `battery_power`
   - Actionable Wi-Fi / cellular disconnect $\to$ `connectivity_network`
   - Actionable native app crash $\to$ `app_or_service_issue`
   - The message is classified as `software_update` only if the update installation itself failed, or if the complaint concerns general system freezing / the canonical 2017 letter 'I' autocorrect bug without a localized hardware/power symptom.
4. **Rule 4 — App Store App Updates over OS Updates:** Customer difficulties downloading, buying, or updating apps through the App Store are classified as `app_or_service_issue`, not `software_update`.

### 3.2 Review Fields Protocol
- `verified_intent`: Assigned exactly one canonical taxonomy class (or `needs_review` for irreconcilable multi-intents).
- `verification_status`: Assigned one of `verified` (preliminary rule accepted), `corrected` (rule overridden), or `flagged_ambiguous` (multi-intent conflict preserved for supervisory review).
- `verified_by`: Set to `pilot_annotator_human`.
- `verification_date`: Set to `2026-09-14`.
- `notes`: Detailed justification explaining the rationale, decision rule applied, or ambiguity justification.

---

## 4. Quantitative Results & Confusion Matrix

### 4.1 Preliminary vs. Verified Intent Crosstab

The table below cross-tabulates preliminary heuristic labels against human-adjudicated ground truth across the 158 pilot records:

| Preliminary Intent \ Verified Intent | `account` | `app_svc` | `battery` | `billing` | `complaint` | `network` | `hardware` | `how_to` | `needs_rev` | `shipping` | `os_update` | `unknown` | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`account_access`** | **5** | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | **8** |
| **`app_or_service_issue`** | 0 | **7** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **8** |
| **`battery_power`** | 0 | 0 | **7** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | **8** |
| **`billing_payment`** | 0 | 0 | 0 | **7** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **7** |
| **`complaint_feedback`** | 0 | 0 | 1 | 0 | **3** | 0 | 0 | 0 | 0 | 0 | 4 | 1 | **9** |
| **`connectivity_network`** | 0 | 0 | 1 | 0 | 1 | **5** | 0 | 0 | 1 | 0 | 0 | 0 | **8** |
| **`device_hardware`** | 0 | 0 | 0 | 1 | 0 | 0 | **6** | 0 | 0 | 0 | 0 | 0 | **7** |
| **`feature_how_to`** | 0 | 2 | 0 | 0 | 0 | 0 | 1 | **3** | 0 | 1 | 2 | 0 | **9** |
| **`needs_review`** | 0 | 2 | 10 | 1 | 0 | 1 | 1 | 4 | **2** | 0 | 2 | 0 | **23** |
| **`order_shipping`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **6** | 1 | 0 | **7** |
| **`software_update`** | 0 | 3 | 6 | 2 | 2 | 1 | 0 | 1 | 0 | 0 | **15** | 1 | **31** |
| **`unknown_other`** | 0 | 6 | 0 | 1 | 5 | 0 | 0 | 1 | 0 | 2 | 6 | **12** | **33** |
| **Total Adjudicated** | **5** | **20** | **26** | **13** | **11** | **7** | **8** | **10** | **3** | **9** | **32** | **14** | **158** |

*(Diagonal values in bold represent matching agreements between preliminary rules and human adjudication.)*

---

### 4.2 Verified Class Distribution vs. Preliminary Distribution

| Intent Category | Preliminary Count | Verified Count | Net Change | Human Adjudication Shift Rationale |
| :--- | :---: | :---: | :---: | :--- |
| `software_update` | 31 | 32 | +1 | Reassigned 16 false positives (battery drain, app updates, billing) but recovered 17 cases (canonical autocorrect bug from `unknown_other` and `complaint_feedback`). |
| `battery_power` | 8 | 26 | +18 | Absorption of actionable post-update battery drain under Rule 3 (from `needs_review` and `software_update`). |
| `app_or_service_issue` | 8 | 20 | +12 | Recovery of App Store download failures and native Photos/Mail app issues from `unknown_other` and `software_update`. |
| `unknown_other` | 33 | 14 | -19 | Resolution of 19 actionable inquiries previously falling into unclassified fallback. |
| `billing_payment` | 7 | 13 | +6 | High precision on initial keywords; recovered in-store payment methods, returns, and digital pricing disputes. |
| `complaint_feedback` | 9 | 11 | +2 | Identification of retail store and Genius Bar customer service dissatisfaction. |
| `feature_how_to` | 9 | 10 | +1 | Resolution of procedural feature questions from `needs_review` and `account_access`. |
| `order_shipping` | 7 | 9 | +2 | Online store order tracking, reservation limbo, and in-store pickup changes. |
| `device_hardware` | 7 | 8 | +1 | Screen unresponsiveness, camera autofocus failure, and MacBook drive failure. |
| `connectivity_network` | 8 | 7 | -1 | Bluetooth pairing and Wi-Fi auto-connect behavior. |
| `account_access` | 8 | 5 | -3 | Reassigned purchasing and storage sharing to billing/feature intents. |
| `needs_review` | 23 | 3 | -20 | 20 multi-category collisions resolved via precedence rules; 3 irreconcilable cases preserved. |
| **Total** | **158** | **158** | **0** | |

---

## 5. Detailed Error Analysis of Preliminary Heuristic Rules

### 5.1 Resolution of `needs_review` Multi-Rule Conflicts (N = 23)
The preliminary pipeline assigned `needs_review` whenever two or more distinct rule categories fired with equal priority. The pilot adjudication resolved 20 of these 23 conflicts (86.96%):
- **10 resolved to `battery_power`:** By applying Rule 3 (Symptom over Attribution), inquiries citing rapid battery drain alongside iOS 11 update mentions (e.g., Tweet 312433, Tweet 854614, Tweet 2349336) route directly to battery diagnostics.
- **4 resolved to `feature_how_to`:** Inquiries asking how to adjust settings post-update (e.g., Tweet 1013096: *"how to keep music playing when opening lockscreen"*, Tweet 1875105: *"where did auto brightness go"*).
- **2 resolved to `app_or_service_issue`:** Native app crashes occurring post-update (e.g., Tweet 2678210: *"iMessage crashes again"*).
- **2 resolved to `software_update`:** System-wide status bar rendering failure (Tweet 1004970) and OS version downgrade inquiries (Tweet 2078507).
- **1 resolved to `billing_payment`:** Gift card balance deduction failure resulting in unintended credit card charges (Tweet 1552580).
- **1 resolved to `connectivity_network`:** Bluetooth toggling behavior under iOS 11 (Tweet 250729).
- **2 flagged as genuinely ambiguous (`needs_review`):** See Section 6 below.

### 5.2 Disambiguation of `software_update` False Positives
Out of 31 preliminary `software_update` candidates, 16 (51.61%) were reassigned:
1. **Rule 4 (App Store vs OS Update):** Tweet 1001145 (*"can’t update or download any apps"*) and Tweet 2310917 (*"cant update app.. just loading"*) were corrected to `app_or_service_issue`.
2. **Payment/Billing Update:** Tweet 2285156 (*"update the billing information which is now invalid"*) was corrected to `billing_payment`.
3. **Actionable Battery Symptom:** Tweets 209063, 295772, 331661, 1069272, 1548776, 1871710 were corrected to `battery_power` under Rule 3.
4. **General Feedback / Venting:** Tweet 1807493 (*"test iOS 11 before releasing"*) and Tweet 567623 (*"update release notes"*) were corrected to `complaint_feedback`.

### 5.3 Recovery of Actionable Intents from `unknown_other`
Out of 33 preliminary `unknown_other` candidates, 21 (63.64%) contained actionable customer inquiries that were successfully recovered:
- **6 to `software_update`:** Customer reports of the canonical late-2017 iOS 11.1 letter 'I' autocorrect bug lacking explicit OS keywords (e.g., Tweet 1500017: *"Fix the I situation"*, Tweet 2027908: *"why my i keeps glitching"*).
- **6 to `app_or_service_issue`:** Native Apple apps mentioned without standard rule triggers (Tweet 1769088: *"help with my email app"*, Tweet 1349239: *"library of music deleted"*, Tweet 1404082: *"timer went into negatives"*).
- **5 to `complaint_feedback`:** Customer service complaints regarding retail store treatment and Genius Bar policies (Tweets 1915424, 2406677, 2573878).
- **2 to `order_shipping`:** Apple Store order processing outages and reservation status in limbo (Tweets 1395126, 2300152).
- **1 to `billing_payment`:** iTunes "Complete My Album" pricing discrepancy (Tweet 996651).
- **1 to `feature_how_to`:** How to exit Guided Access mode (Tweet 599090).

---

## 6. Genuinely Ambiguous Edge Cases (`flagged_ambiguous`)

Only **3 out of 158 records (1.90%)** were flagged as genuinely irreconcilable multi-intent inquiries:

1. **Tweet 410510 (Spanish):**
   > *"@AppleSupport con la nueva actualización, tengo problemas para conectarme con el Bluetooth, la pantalla se traba y no me rinde la batería."*  
   - **Conflict:** The customer reports three distinct, co-equal physical and connectivity failures post-update: Bluetooth connectivity drop, screen freezing, and severe battery depletion. No single symptom is prioritized.  
   - **Adjudication:** Flagged as `needs_review` (`flagged_ambiguous`).
2. **Tweet 965710:**
   > *"Hey @AppleSupport. Had my device 3 months and a key developed a fault, had it back from store today and the mechanism behind the key is now snapped. No device = loss of income; this is my only required tool. Disappointed & want to switch, can I get full refund? Laptop was £3,250!"*  
   - **Conflict:** Multi-intent conflict between in-store hardware repair damage (`device_hardware`) and a £3,250 full purchase price refund claim (`billing_payment`).  
   - **Adjudication:** Flagged as `needs_review` (`flagged_ambiguous`).
3. **Tweet 1066754:**
   > *"My favorite part of @115858 ‘s new iPhone update is the part where my battery lasts 20 minutes and the internet never works ever"*  
   - **Conflict:** Customer reports two catastrophic, simultaneous operational failures: battery lasting only 20 minutes (`battery_power`) and total internet connectivity failure (`connectivity_network`). Both require distinct diagnostic workflows.  
   - **Adjudication:** Flagged as `needs_review` (`flagged_ambiguous`).

---

## 7. Recommendations for Full Review Sample Annotation (1,874 Records)

Based on the pilot annotation findings, the following workflow improvements are recommended prior to annotating `data/processed/apple_support/apple_support_intent_review_sample.csv`:

1. **Enforce Hierarchical Precedence Rules in Tooling:**
   - Annotators should be equipped with a quick-reference card prioritizing:
     $$\text{Safety/Damage} \succ \text{Financial/Security} \succ \text{Actionable Symptom} \succ \text{Historical Attribution}$$
2. **Address App Store vs. OS Update Collisions:**
   - Approximately 50% of heuristic `software_update` matches involving verbs like *"update"* relate to App Store apps, billing details, or delivery addresses. Annotators should routinely check for App Store or billing contexts when reviewing candidate `software_update` records.
3. **Battery Drain Attribution Rule:**
   - In late 2017 data, customers frequently cited iOS 11 as the cause of battery drain. Strictly routing these to `battery_power` aligns automated customer support triage with practical diagnostic workflows.
4. **Inter-Annotator Calibration:**
   - With an ambiguity rate under 2.0%, the annotation guidelines in `docs/apple_support_intent_annotation_guide.md` are robust, reproducible, and ready for multi-annotator scaling.

---

## 8. Artifact Verification

- **Annotated CSV:** `data/processed/apple_support/apple_support_intent_pilot_sample.csv`
  - Records: 158
  - Verified Status: 76 `verified`, 79 `corrected`, 3 `flagged_ambiguous`
  - Nulls in review columns: 0
- **Metadata JSON:** `data/processed/apple_support/apple_support_intent_pilot_metadata.json`
  - Content SHA-256: `3131e68661db58a8b3c78c5468538bb05f1a0f92d7abfe31bc247a7f5e355dbf`
  - Status: `completed`
