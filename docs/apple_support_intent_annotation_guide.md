# AppleSupport Customer Intent Annotation Guide

**Document:** Practical Human Annotation Guide  
**Taxonomy Version:** `2.0`  
**Application Domain:** Customer Support Conversations  
**Selected Brand:** `AppleSupport`  
**Target Datasets:**  
- Pilot Sample: `data/processed/apple_support/apple_support_intent_pilot_sample.csv` (158 records)  
- Full Review Sample: `data/processed/apple_support/apple_support_intent_review_sample.csv` (1,874 records)  
**Associated Configuration:** `src/classification/intents.yaml`  
**Quality Audit Report:** `docs/apple_support_intent_label_quality_audit.md`  

---

## 1. Objective & Annotation Purpose

This guide establishes standardized annotation procedures for labeling customer inquiries directed to AppleSupport on Twitter. In automated customer support systems, intent classification serves **front-line triage**: routing an inbound message at initial contact to self-service knowledge base articles, automated diagnostic flows, or specialized human support queues.

### Fundamental Principles
1. **Initial Problem Formulation:** Annotators classify the customer's *initial problem statement* from their first utterance (`text`). Do not attempt to predict subsequent turns or conversation outcomes.
2. **Ground Truth Independence:** Preliminary labels (`preliminary_intent`, `label_source`, `preliminary_confidence`) are deterministic rule suggestions provided solely to assist auditing and active learning. **Never assume preliminary labels are correct.**
3. **No Fabrication of Resolution:** Inbound customer messages often terminate without explicit resolution in public tweets. Do not confuse conversation termination with issue resolution.
4. **Consistency & Traceability:** Every correction or ambiguous judgment must be documented in the `notes` column.

---

## 2. Taxonomy Definitions & Decision Boundaries (Version 2.0)

Every record must be assigned exactly one canonical intent from the 12 mutually exclusive classes below.

---

### 2.1 `software_update` (Software / OS Update Issue)
* **Display Name:** Software / OS Update Issue
* **Definition:** The customer reports a malfunction, performance bug, or glitch that started after, or is directly associated with, an operating system update, iOS/macOS version, or system patch.
* **Positive Inclusion:**
  - Explicitly cites OS releases (e.g., `iOS 11`, `iOS 11.1`, `High Sierra`, `macOS`, `watchOS`).
  - Mentions update installation failure, boot loop during update, or bugs introduced by an update.
  - Mentions known 2017 OS bugs (e.g., the letter 'I' autocorrecting to 'A [?]').
* **Exclusion & Boundary Disambiguation:**
  - *App Store app downloads failing:* Classify as `app_or_service_issue`.
  - *Updating payment methods, cards, or billing info:* Classify as `billing_payment`.
  - *Updating shipping address or delivery status:* Classify as `order_shipping`.
  - *Battery drain where no update is cited:* Classify as `battery_power`.
* **Benchmark Examples:**
  - *"updated to iOS 11.1 and now my phone freezes constantly"* $\to$ `software_update`
  - *"why does the letter i turn into a question mark on my keyboard"* $\to$ `software_update`
  - *"High Sierra install failed and now my MacBook won't boot"* $\to$ `software_update`

---

### 2.2 `battery_power` (Battery, Power & Charging)
* **Display Name:** Battery, Power & Charging
* **Definition:** The customer reports abnormal battery depletion, sudden shutdowns, overheating, charging failure, or damaged charging accessories.
* **Positive Inclusion:**
  - Mentions battery life, rapid drain, percentage drops (e.g., jumping from 50% to 10%).
  - Device overheating or getting hot during use or charging.
  - Device refusing to charge, charging slowly, or broken charging cables/adapters.
* **Exclusion & Boundary Disambiguation:**
  - *Overheating accompanied by a bulging screen or swollen battery:* Classify as `device_hardware` (safety hazard).
  - *System-wide freeze where battery is not the primary complaint:* Classify as `software_update`.
* **Benchmark Examples:**
  - *"my iPhone 7 battery drops from 80% to 20% in 30 minutes"* $\to$ `battery_power`
  - *"my phone gets boiling hot whenever I plug it into the wall charger"* $\to$ `battery_power`
  - *"charger cable stopped working and won't charge my phone"* $\to$ `battery_power`

---

### 2.3 `device_hardware` (Device Hardware & Physical Damage)
* **Display Name:** Device Hardware & Physical Damage
* **Definition:** The customer reports physical damage, screen defects, touch unresponsiveness, camera blackouts, audio speaker/mic failures, physical button sticking, or liquid exposure.
* **Positive Inclusion:**
  - Cracked, shattered, or physically damaged displays.
  - Touchscreen failing to register taps or ghost touch.
  - Rear or front camera displaying a black screen or lens artifacts.
  - Earpiece, microphone, or speaker muffled/crackling during voice calls.
  - Home button or power button stuck or unresponsive.
* **Exclusion & Boundary Disambiguation:**
  - *Display freezing caused by a specific native app crash:* Classify as `app_or_service_issue`.
  - *Dirty charging port preventing charge:* Classify as `battery_power`.
* **Benchmark Examples:**
  - *"dropped my phone on concrete and the screen is cracked and unresponsive"* $\to$ `device_hardware`
  - *"my front camera shows completely black when opening the camera app"* $\to$ `device_hardware`
  - *"the speaker is crackling and callers cannot hear me"* $\to$ `device_hardware`

---

### 2.4 `app_or_service_issue` (Native App or Apple Service Issue)
* **Display Name:** Native App or Apple Service Issue
* **Definition:** The customer reports an error, crash, or syncing failure with an Apple-native application or core cloud service.
* **Positive Inclusion:**
  - Native applications: `Apple Music`, `iMessage`, `FaceTime`, `Safari`, `Apple Maps`, `App Store`, `Photos`, `Notes`, `Siri`, `CarPlay`, `AirDrop`.
  - App Store failures: unable to download, buy, or update third-party apps through the App Store.
  - iMessage delivery failures (messages sending as green SMS instead of blue iMessage).
* **Exclusion & Boundary Disambiguation:**
  - *Third-party app backend failures unrelated to Apple ecosystem (e.g., Instagram server down):* Classify as `unknown_other`.
  - *Apple Music unauthorized credit card charges:* Classify as `billing_payment`.
* **Benchmark Examples:**
  - *"why won't my apps download or update in the App Store"* $\to$ `app_or_service_issue`
  - *"Apple Music keeps pausing and skipping songs on its own"* $\to$ `app_or_service_issue`
  - *"iMessage is not delivering texts to my contacts"* $\to$ `app_or_service_issue`

---

### 2.5 `account_access` (Account, iCloud & Security)
* **Display Name:** Account, iCloud & Security
* **Definition:** The customer cannot log in to their Apple ID, iCloud, or iTunes account, has forgotten credentials, failed two-factor authentication (2FA), or has an account locked for security.
* **Positive Inclusion:**
  - Forgotten Apple ID password, passcode, or security questions.
  - Account disabled or locked for security reasons.
  - Two-factor authentication verification codes not arriving.
  - iCloud backup restoration errors during device setup.
* **Exclusion & Boundary Disambiguation:**
  - *Disputing an in-app purchase made while signed in:* Classify as `billing_payment`.
* **Benchmark Examples:**
  - *"my Apple ID has been locked for security reasons and I can't reset password"* $\to$ `account_access`
  - *"not receiving the two-factor verification code on my trusted number"* $\to$ `account_access`
  - *"forgot my iCloud password and recovery email is no longer active"* $\to$ `account_access`

---

### 2.6 `connectivity_network` (Connectivity & Network)
* **Display Name:** Connectivity & Network
* **Definition:** The customer reports difficulty connecting to Wi-Fi networks, Bluetooth devices (AirPods, car audio), cellular mobile data / LTE, dropped voice calls, or SIM card errors.
* **Positive Inclusion:**
  - Wi-Fi disconnecting frequently or failing to connect to known networks.
  - Bluetooth failing to pair or disconnecting from AirPods/headphones.
  - Cellular display showing 'No Service', 'Searching...', or dropped carrier calls.
* **Exclusion & Boundary Disambiguation:**
  - *Carrier billing disputes:* Classify as `unknown_other` (not Apple billing).
* **Benchmark Examples:**
  - *"iPhone keeps disconnecting from home Wi-Fi every 5 minutes"* $\to$ `connectivity_network`
  - *"my phone says No Service even though my carrier network is fine"* $\to$ `connectivity_network`
  - *"Bluetooth won't pair with my AirPods after restarting"* $\to$ `connectivity_network`

---

### 2.7 `billing_payment` (Billing, Subscriptions & Payments)
* **Display Name:** Billing, Subscriptions & Payments
* **Definition:** The customer inquires about unexpected credit card charges, recurring subscription fees, App Store purchase disputes, refund requests, Apple Pay errors, or updating payment methods.
* **Positive Inclusion:**
  - Disputed credit/debit card charges from iTunes, Apple, or App Store.
  - Subscription cancellation or refund requests for Apple Music, iCloud storage, or in-app purchases.
  - Apple Pay transaction declines or errors.
  - Inquiries about updating payment methods, billing details, or Apple Store gift cards.
* **Exclusion & Boundary Disambiguation:**
  - *Physical product pre-order shipping delays:* Classify as `order_shipping`.
* **Benchmark Examples:**
  - *"I need to update the payment method for my iPhone X pre-order"* $\to$ `billing_payment`
  - *"charged $9.99 for Apple Music even though I cancelled my trial, want a refund"* $\to$ `billing_payment`
  - *"Apple Pay declined my card at checkout but the charge shows pending"* $\to$ `billing_payment`

---

### 2.8 `order_shipping` (Online Orders, Shipping & Delivery)
* **Display Name:** Online Orders, Shipping & Delivery
* **Definition:** The customer inquires about the status, tracking number, shipment delay, delivery, or in-store pickup of a physical order placed through the Apple Online Store.
* **Positive Inclusion:**
  - Order status tracking (e.g., "Preparing for Shipment", "Dispatched").
  - Courier tracking inquiries (UPS, FedEx) or delivery delays.
  - In-store pickup scheduling for hardware orders (iPhone, Mac).
  - Changing shipping addresses for an active physical order.
* **Exclusion & Boundary Disambiguation:**
  - *Digital software refunds or App Store subscriptions:* Classify as `billing_payment`.
* **Benchmark Examples:**
  - *"my iPhone X order has been preparing for shipment for 5 days — when does it ship"* $\to$ `order_shipping`
  - *"tracking number says delivered but package is not on my porch"* $\to$ `order_shipping`
  - *"can I change the shipping address for order W12345678"* $\to$ `order_shipping`

---

### 2.9 `feature_how_to` (Feature Question & How-To)
* **Display Name:** Feature Question & How-To
* **Definition:** The customer asks how to configure, enable/disable, or find an existing feature or setting in an Apple product without reporting a technical defect.
* **Positive Inclusion:**
  - Questions beginning with *"how do I"*, *"how can I"*, *"is there a way to"*, *"where is the setting"*.
  - Procedural instructions (e.g., screen recording, transferring photos, setting up family sharing).
* **Exclusion & Boundary Disambiguation:**
  - *Feature is malfunctioning or broken:* Classify into the relevant defect category (`device_hardware`, `app_or_service_issue`).
* **Benchmark Examples:**
  - *"how do I turn off read receipts in iMessage for one specific contact"* $\to$ `feature_how_to`
  - *"how can I transfer photos from my iPhone to a Windows PC"* $\to$ `feature_how_to`
  - *"is there a way to record the screen on iOS without third party apps"* $\to$ `feature_how_to`

---

### 2.10 `complaint_feedback` (Complaint & Product Feedback)
* **Display Name:** Complaint & Product Feedback
* **Definition:** The customer expresses general dissatisfaction, anger, or venting regarding Apple devices, policies, or customer service without asking a resolvable technical support question.
* **Positive Inclusion:**
  - Strong negative sentiment expressions (*"worst phone ever"*, *"terrible customer service"*, *"hate apple"*).
  - Venting regarding product design decisions (e.g., headphone jack removal, notch design).
* **Exclusion & Boundary Disambiguation:**
  - *Complaint accompanied by a specific actionable malfunction:* Classify into the relevant technical intent (`battery_power`, `software_update`, `device_hardware`).
* **Benchmark Examples:**
  - *"@AppleSupport your in-store customer service was absolutely terrible today"* $\to$ `complaint_feedback`
  - *"removing the headphone jack was the dumbest idea ever Apple fix this"* $\to$ `complaint_feedback`
  - *"worst company in history, I am switching to Android tomorrow"* $\to$ `complaint_feedback`

---

### 2.11 `unknown_other` (Unknown / Fallback / Other)
* **Display Name:** Unknown / Fallback / Other
* **Definition:** The customer message does not contain sufficient semantic information to assign any specific domain category, is a brief greeting, is an uninterpretable fragment, or falls outside Apple technical support.
* **Positive Inclusion:**
  - Conversational greetings or standalone handles (*"@AppleSupport hello"*, *"@AppleSupport please DM me"*).
  - Standalone media links or image URLs without descriptive problem text.
  - Vague complaints lacking domain context (*"my phone is broken"*, *"fix this glitch"*).
  - Third-party app server outages unrelated to Apple (e.g., *"is Netflix down?"*).
* **Exclusion & Boundary Disambiguation:**
  - *Any message where a specific domain issue can be inferred with reasonable confidence.*
* **Benchmark Examples:**
  - *"@AppleSupport can you DM me"* $\to$ `unknown_other`
  - *"@AppleSupport https://t.co/abc1234"* $\to$ `unknown_other`
  - *"Tf is wrong with my phone @115858"* $\to$ `unknown_other`

---

## 3. Multi-Intent Handling & Adjudication Rules

When a customer reports multiple issues within a single tweet (e.g., in rows flagged `needs_review` or `rule_overlap_flag == True`), annotators must select the single most operational intent using the following **hierarchical precedence rules**:

### Rule 1: Safety & Physical Damage Supersedes Software
* If physical damage, screen cracking, or battery swelling is reported alongside a software bug, classify as **`device_hardware`**.
* *Rationale:* Physical hardware repair takes operational precedence over software troubleshooting.

### Rule 2: Security & Financial Disputes Supersede General Inquiries
* If unauthorized credit card charges or locked Apple IDs are reported, classify as **`billing_payment`** or **`account_access`**.
* *Rationale:* Security and financial transactions require immediate escalation.

### Rule 3: Actionable Symptom Supersedes Historical Attribution
* When a customer reports that an OS update caused a specific symptom (e.g., *"battery draining fast since updating to iOS 11"* or *"Wi-Fi stopped connecting after update"*):
  - If the primary actionable complaint is battery drain $\to$ **`battery_power`**.
  - If the primary actionable complaint is Wi-Fi failure $\to$ **`connectivity_network`**.
  - If the primary actionable complaint is an app crashing $\to$ **`app_or_service_issue`**.
  - Classify as **`software_update`** only if the update itself failed, or if the complaint is general system freezing / autocorrect bugs without a specific hardware/network/battery symptom.
* *Rationale:* In automated support, diagnosing the active symptom (e.g., running battery diagnostics) provides the actionable resolution path.

### Rule 4: Native App Store over OS Update
* When a customer reports inability to download or update apps through the App Store, classify as **`app_or_service_issue`**, not `software_update`.

---

## 4. Resolving `needs_review` Records

Candidates with `preliminary_intent == "needs_review"` represent multi-category rule conflicts. **Annotators must resolve every `needs_review` row into one of the 11 valid domain intents or `unknown_other`.**

### Step-by-Step Resolution Process:
1. Inspect the raw text (`text`) and the matched categories (`matched_intents`).
2. Identify the core actionable problem the customer wants resolved.
3. Apply the hierarchical precedence rules (Safety > Security/Billing > Actionable Symptom > Attribution).
4. Enter the chosen category into `verified_intent`.
5. Enter a brief justification in `notes` (e.g., *"Resolved conflict: symptom is battery drain; iOS 11 is attribution"*).
6. Set `verification_status` to `"corrected"`.

---

## 5. Human Review Spreadsheet Fields

Annotators must record their judgments in the review CSV (`apple_support_intent_pilot_sample.csv` or `apple_support_intent_review_sample.csv`).

| Column Name | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `verified_intent` | string | One of the 12 taxonomy categories: `software_update`, `battery_power`, `device_hardware`, `app_or_service_issue`, `account_access`, `connectivity_network`, `billing_payment`, `order_shipping`, `feature_how_to`, `complaint_feedback`, `unknown_other` | **Mandatory.** The adjudicated ground-truth intent. Never enter `"needs_review"` as verified intent. |
| `verified_by` | string | e.g., `annotator_1`, `reviewer_js` | **Mandatory.** Unique identifier of the human annotator. |
| `verification_date` | string | `YYYY-MM-DD` (e.g., `2026-09-15`) | **Mandatory.** Date annotation was completed. |
| `verification_status` | string | `verified`, `corrected`, `escalated`, `flagged_ambiguous` | **Mandatory.** Indicator of annotation action taken: <br>• `verified`: preliminary intent was correct. <br>• `corrected`: preliminary intent was changed. <br>• `escalated`: requires senior supervisor adjudication. <br>• `flagged_ambiguous`: mutually irreconcilable statement. |
| `notes` | string | Free-text string | **Required** when status is `corrected`, `escalated`, or `flagged_ambiguous`. Explain reasoning or edge-case rationale. |

---

## 6. Pilot Annotation Protocol (158 Records)

Before annotating the full 1,874 review records:
1. Two independent annotators must annotate `data/processed/apple_support/apple_support_intent_pilot_sample.csv` (158 records) without consulting each other.
2. Compute inter-annotator agreement (Cohen's Kappa).
3. Any discrepancies must be discussed with the lead researcher and reconciled.
4. Target inter-annotator agreement threshold: **$\kappa \ge 0.85$** before proceeding to the full review sample.
