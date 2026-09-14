# AppleSupport Intent Label Quality Audit & Human-Review Preparation Report

**Audit Target:** Phase 5 Preliminary Intent Labels (`data/processed/apple_support/apple_support_intent_candidates.csv`)  
**Audit Output Sample:** `data/processed/apple_support/apple_support_intent_review_sample.csv` (1,874 records)  
**Annotation Pilot Sample:** `data/processed/apple_support/apple_support_intent_pilot_sample.csv` (158 records)  
**Audit Metadata:** `data/processed/apple_support/apple_support_intent_review_metadata.json`  
**Pilot Metadata:** `data/processed/apple_support/apple_support_intent_pilot_metadata.json`  
**Taxonomy Version:** `2.0`  
**Brand:** `AppleSupport`  
**Audit Tooling:** `src/classification/audit_intent_labels.py`  
**Annotation Guide:** `docs/apple_support_intent_annotation_guide.md`  
**Deterministic Content SHA-256 (Review Sample):** `c4fed9095befad0737f15401879cd667a9a6ae9735712829af570a9be312c7ae`  
**Deterministic Content SHA-256 (Pilot Sample):** `8e24d16a14814811df1a1e31fbec1786e4691313dc02274810d84a516dc77873`  
**Random Seed:** `42`  

---

## 1. Executive Summary & Audit Scope

Phase 6 conducts a rigorous, reproducible quality audit of the 82,101 preliminary intent labels generated in Phase 5. In customer support machine learning pipelines, preliminary heuristic labels provide an efficient bootstrapping mechanism for active learning and dataset stratification. However, **preliminary rule labels are not ground truth and must never be treated as verified benchmark evaluation data or fed directly into model training without human review**.

### 1.1 Methodological Distinction of Evidence
To maintain strict scientific integrity, this audit explicitly distinguishes four classes of findings:
1. **Full-Corpus Deterministic Metrics (N = 82,101):** Exact counts and distributions computed across all candidates extracted from Phase 4 reconstructed threads.
2. **Review-Sample Statistics (N = 1,874):** Stratified sample metrics across 26 distinct strata. Sample percentages reflect intentional quota sampling (oversampling rare classes and ambiguous edge cases), **not** population prevalences.
3. **Heuristic Observations:** Exploratory pattern-matching and keyword searches on candidate text. These represent qualitative investigative signals and hypotheses, not verified classifications.
4. **Findings Requiring Human Annotation:** Ambiguous multi-category overlaps, borderline update mentions, and unclassifiable fallback cases that require human adjudication in `apple_support_intent_review_sample.csv` and `apple_support_intent_pilot_sample.csv`.

---

## 2. In-Depth Quality Investigation

### 2.1 Quality of `unknown_other` (Full Corpus N = 37,669; Sample N = 471)

In Phase 5, `unknown_other` accounted for 37,669 candidates (45.88%). A critical question is whether this high share reflects an incomplete taxonomy, overly narrow rules, or natural conversational characteristics of public Twitter support.

#### Heuristic Subgroup Observations (Review Sample N = 471)
Inspection of the 471 sampled `unknown_other` records revealed the following heuristic subgroups:

1. **Conversational Pleasantries & Standalone Mentions (52 of 471 sample rows, 11.0%):**
   - Customer messages containing only greetings, requests to open a direct message (DM), or brief mentions without describing a problem:
     - `[709]`: *"@AppleSupport I️ need answers because it’s annoying 🙃"*
     - `[1010]`: *"@AppleSupport DM me please"*
     - `[1245]`: *"@AppleSupport Hello can you help me with an issue"*
   - *Heuristic Assessment:* Appropriately assigned to `unknown_other`. An automated agent cannot triage these without conversational elicitation.
2. **Media-Only & URL-Only Tweets (18 of 471 sample rows, 3.8%):**
   - Tweets consisting only of an `@AppleSupport` handle and a `https://t.co/...` URL pointing to an attached screenshot or photo:
     - `[698]`: *"@AppleSupport https://t.co/NV0yucs0lB"*
   - *Heuristic Assessment:* Appropriately assigned to `unknown_other`. Text classifiers cannot infer intent from raw image links without multimodal vision inputs.
3. **Vague / General Defect Statements (53 of 471 sample rows, 11.3%):**
   - Tweets reporting a device failure without specifying hardware, software, battery, or network:
     - `[719]`: *"Tf is wrong with my keyboard @115858"*
     - `[41662]`: *"My wonderful MacBook Pro won't come on 😯😯😢 @applesupport"*
   - *Heuristic Assessment:* In a live triage flow, an agent must elicit details ("Does the screen illuminate?"). Heuristic rules rightly avoid forcing an ungrounded domain label.
4. **Lexical Rule Gaps & Potential False Negatives (20 of 471 sample rows, 4.2%):**
   - Inquiries concerning specific domain topics that were missed due to phrasing variations, smart punctuation, or unicode characters:
     - `[31931]`: *"@AppleSupport Can I pay for  Care + online with an Apple Store gift card or do I have go into the store?"* (missed `billing_payment` due to unicode `` and missing "gift card" pattern).
     - `[61071]`: *"What dafugg is wrong with my phone, Everytime i️ type the letter “I️” 😡.. @115858"* (missed `software_update` autocorrect bug due to unicode smart quotes `“I️”`).
     - `[61083]`: *"until @115858 put a hold on my order and nobody knows why...."* (missed `order_shipping` because "put a hold on" was omitted from shipping rules).
5. **Niche Apple Ecosystem Topics (12 of 471 sample rows, 2.5%):**
   - Inquiries regarding Apple Watch hardware, Apple TV setup, or retail store Genius Bar appointments:
     - 372 bigram occurrences of `apple watch` across all 37,669 `unknown_other` candidates.
     - 250 bigram occurrences of `apple store` across all 37,669 `unknown_other` candidates.

> [!NOTE]
> **Key Finding on `unknown_other`:** These subgroup figures represent **heuristic observations from unverified preliminary labels**, not confirmed ground-truth classifications. The vast majority of fallback cases represent conversational brevity, image links, or unspecific problem formulations that fundamentally require conversational elicitation rather than static single-turn classification.

---

### 2.2 Quality of `software_update` (Full Corpus N = 22,942; Sample N = 436)

`software_update` is the largest non-fallback category, representing 27.94% of all candidates. While the category captures prominent 2017 operating system releases, our empirical recheck identified substantial false-positive risks resulting from generic uses of the word "update".

#### Full-Corpus Collision Analysis (N = 22,942)
An empirical audit across all 22,942 `software_update` candidates revealed significant cross-domain lexical collisions:

| Lexical Collision Pattern | Corpus Count | Share of `software_update` | Likely Intended Domain |
| :--- | :--- | :--- | :--- |
| **App Store / Third-Party Apps** (`app`, `apps`, `download`, `updating apps`) | 2,142 | 9.34% | `app_or_service_issue` |
| **Payment / Cards / Billing** (`card`, `payment`, `billing`, `credit card`, `bank`) | 484 | 2.11% | `billing_payment` |
| **Account / Address / Details** (`address`, `details`, `status`, `info`, `information`) | 141 | 0.61% | `billing_payment` / `account_access` |
| **Shipping / Pre-orders** (`order`, `shipping`, `delivery`, `tracking`, `package`) | 61 | 0.27% | `order_shipping` |
| **Retail / Repairs** (`repair`, `store`, `appointment`, `genius bar`) | 126 | 0.55% | `device_hardware` / `unknown_other` |

#### Confidence Breakdown & False Positive Vulnerability
The vulnerability is strongly correlated with confidence tier:
* **High-Confidence Candidates (N = 16,810):** Triggered by specific signatures (`os_update_version_mention`, `ios11_autocorrect_bug`, `post_update_malfunction`). High-confidence predictions show strong qualitative fidelity to genuine OS update issues:
  - `[10689]`: *"updated to iOS 11.1 @AppleSupport and there's so many glitches and bugs I found inside the update"*
  - `[118117]`: *"@115858 iOS 11 is messed up, now my phone is freezing so much"*
  - `[90651]`: *"I️ updated my phone recently. The I️ thing never happened to me when it apparently happened to literally everyone else. And NOW it happens? Hurry with the next update @115858"*
* **Medium-Confidence Candidates (N = 6,132):** Triggered solely by the single keyword pattern `\b(update|updated|upgrade|upgraded|patch)\b` (score 1). In this subset:
  - **633 candidates (10.32%)** involve third-party apps or App Store downloading issues.
  - **151 candidates (2.46%)** involve updating credit cards or payment methods.
  - **74 candidates (1.21%)** involve updating address or customer details.

#### Authentic False Positive Examples from Corpus
1. **App Store App Download Collision:**
   - *"why every time I go to update of download an app I have to turn my phone on and off for it to work!?"* $\to$ Intended domain: `app_or_service_issue`.
   - *"why won’t my apps fuccin download or update @AppleSupport"* $\to$ Intended domain: `app_or_service_issue`.
2. **Payment Method Update Collision:**
   - *"Help!! I need to update the payment method for my iPhone X pre-order. It won’t let me do it online"* $\to$ Intended domain: `billing_payment` / `order_shipping`.
   - *"So i got a email to update my payment method, i keep trying & reset my iPhone to do this but it keeps saying try again later it cannot update."* $\to$ Intended domain: `billing_payment`.
   - *"Not able to update do to Billling Issue error? Billing server does not seem to be responding Tried to update billing info"* $\to$ Intended domain: `billing_payment`.
3. **Shipping Address Update Collision:**
   - *"there huge bug ios 11 wallet shipping addres i chose country and it crashed"* $\to$ Dual domain: `billing_payment` and `software_update`.

> [!WARNING]
> **Audit Conclusion on `software_update`:** Standalone occurrences of "update" are semantically overloaded in English customer support. Relying on "update" without requiring co-occurring OS terms (`ios`, `macos`, `system`, `phone`) introduces a ~12% false-positive rate within medium-confidence predictions. Human verification must carefully review medium-confidence `software_update` records.

---

### 2.3 Quality of `needs_review` (Full Corpus N = 5,895; Sample N = 336)

In Phase 5, 5,895 candidates were routed to `needs_review` due to tied scores or competing high-confidence patterns across distinct categories.

#### Distribution of Conflict Pairs (Full Corpus Top 10)
| Conflicting Category Pair | Corpus Count | Share of Conflicts | Primary Multi-Intent Formulation |
| :--- | :--- | :--- | :--- |
| `['software_update', 'battery_power']` | 1,866 | 31.65% | Customer complaining that an iOS update caused rapid battery drain |
| `['software_update', 'connectivity_network']` | 867 | 14.71% | Wi-Fi / Bluetooth disconnecting after updating to iOS 11 |
| `['software_update', 'app_or_service_issue']` | 593 | 10.06% | Native app (iMessage, FaceTime, Music) crashing after OS update |
| `['software_update', 'feature_how_to']` | 441 | 7.48% | Inquiring how to use or configure a new OS update feature |
| `['software_update', 'device_hardware']` | 361 | 6.12% | Touch display or speaker malfunction noticed after updating |
| `['software_update', 'complaint_feedback']` | 218 | 3.70% | Frustration venting regarding an update breaking phone functionality |
| `['account_access', 'software_update']` | 165 | 2.80% | iCloud login / restore failure during OS setup |
| `['account_access', 'feature_how_to']` | 160 | 2.71% | Inquiring how to reset Apple ID password or configure 2FA |
| `['battery_power', 'device_hardware']` | 134 | 2.27% | Device overheating accompanied by physical enclosure symptoms |
| `['billing_payment', 'app_or_service_issue']` | 112 | 1.90% | Apple Music subscription charges and renewal questions |

#### Representative Multi-Intent Utterances
* `[11992]`: *"@115858 @AppleSupport your latest iOS update had made my phone slower and apps like iMessage are slow to open and load. What’s up with this?"*
* `[31474]`: *"Ios 11.1 & I can’t even turn off the wifi & bluetooth from the control center!! @115858 @AppleSupport"*
* `[103174]`: *"@AppleSupport @115858 why does my battery drain so fast now on iOS 11.1.2 I lose 10% in an hour or less and I don’t even use the phone. Im at 26% now"*

> [!IMPORTANT]
> **Audit Finding on `needs_review`:** Over 31% of conflicts involve **OS update attribution combined with a specific actionable symptom** (battery, Wi-Fi, app crash). These utterances are genuinely multi-intent. Forcing an arbitrary single label would introduce label noise into downstream supervised models. Routing them to `critical` human review is structurally sound.

---

### 2.4 Rule Overlap & Multi-Category Matches (Full Corpus N = 11,410)

Across the entire 82,101 candidates:
* **0 categories matched:** 37,666 candidates (45.88%)
* **Exactly 1 category matched:** 33,025 candidates (40.22%)
* **Exactly 2 categories matched:** 10,277 candidates (12.52%)
* **Exactly 3 categories matched:** 1,069 candidates (1.30%)
* **4 or more categories matched:** 64 candidates (0.08%)
* **Total Multi-Category Matches:** 11,410 candidates (13.90%)

In the stratified review sample, **550 records** feature `rule_overlap_flag == True` and are assigned `critical` review priority. In the pilot sample, **46 records** feature `rule_overlap_flag == True`.

---

## 3. Review Sample & Pilot Dataset Statistics

### Table 1: Stratified Distributions Across Full Corpus, Review Sample, and Pilot Sample
| Dimension | Full Candidate Corpus (N = 82,101) | Stratified Review Sample (N = 1,874) | Annotation Pilot Sample (N = 158) |
| :--- | :--- | :--- | :--- |
| **`unknown_other`** | 37,669 (45.88%) | 471 (25.13%) | 33 (20.89%) |
| **`software_update`** | 22,942 (27.94%) | 436 (23.27%) | 31 (19.62%) |
| **`needs_review`** | 5,895 (7.18%) | 336 (17.93%) | 23 (14.56%) |
| **`battery_power`** | 3,406 (4.15%) | 99 (5.28%) | 8 (5.06%) |
| **`app_or_service_issue`** | 2,365 (2.88%) | 77 (4.11%) | 8 (5.06%) |
| **`account_access`** | 2,166 (2.64%) | 75 (4.00%) | 8 (5.06%) |
| **`feature_how_to`** | 1,889 (2.30%) | 74 (3.95%) | 9 (5.70%) |
| **`device_hardware`** | 2,138 (2.60%) | 71 (3.79%) | 7 (4.43%) |
| **`connectivity_network`** | 1,717 (2.09%) | 66 (3.52%) | 8 (5.06%) |
| **`complaint_feedback`** | 1,105 (1.35%) | 60 | 9 (5.70%) |
| **`billing_payment`** | 607 (0.74%) | 56 (2.99%) | 7 (4.43%) |
| **`order_shipping`** | 202 (0.25%) | 53 (2.83%) | 7 (4.43%) |
| **High Confidence** | 25,951 (31.61%) | 667 (35.59%) | 63 (39.87%) |
| **Medium Confidence** | 12,586 (15.33%) | 400 (21.34%) | 39 (24.68%) |
| **Low Confidence** | 43,564 (53.06%) | 807 (43.06%) | 56 (35.44%) |
| **Rule Overlap (`flag == True`)** | 11,410 (13.90%) | 550 (29.35%) | 46 (29.11%) |
| **`critical` Review Priority** | — | 550 (29.35%) | 46 (29.11%) |
| **`high` Review Priority** | — | 793 (42.32%) | 51 (32.28%) |
| **`medium` Review Priority** | — | 286 (15.26%) | 32 (20.25%) |
| **`normal` Review Priority** | — | 245 (13.07%) | 29 (18.35%) |

---

## 4. Formal Recommendations & Human Annotation Readiness

### 4.1 Taxonomy Recommendation: **`retain taxonomy unchanged`** with **`revise rule matching only`**

1. **Taxonomy Retention:**
   Taxonomy Version 2.0 (11 domain classes + `unknown_other`) accurately mirrors real-world Apple customer support operations. Small clusters like retail store appointments and Apple Watch hardware represent $<1.0\%$ of customer inquiries and do not justify adding imbalanced micro-classes.
2. **Targeted Rule Refinement (Post-Annotation):**
   - Disambiguate `update`: Require `update` to co-occur with OS/device tokens (`ios`, `macos`, `system`, `phone`) to prevent App Store app download collisions.
   - Expand `billing_payment`: Add `gift card`, `apple care`, and `payment method`.
   - Normalize smart quotes (`“`, `”`, `’`) before regex matching.
3. **No Direct Model Training on Preliminary Labels:**
   Preliminary labels must **never** be used as ground truth. Supervised intent models must be trained only after annotators complete the pilot dataset and prioritized review cohorts.

### 4.2 Annotation Pilot Workflow
Before launching full-scale annotation across all 1,874 review records, human annotators must complete the **158-record pilot dataset** (`data/processed/apple_support/apple_support_intent_pilot_sample.csv`) following the guidelines in `docs/apple_support_intent_annotation_guide.md` to measure inter-annotator agreement and resolve boundary ambiguities.
