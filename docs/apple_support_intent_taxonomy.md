# AppleSupport Customer Intent Taxonomy

**Taxonomy Version:** `2.0`  
**Application Domain:** Customer Support Conversations  
**Selected Brand:** `AppleSupport`  
**Configuration File:** `src/classification/intents.yaml`  
**Module Loader:** `src/classification/intent_loader.py`  

---

## 1. Executive Summary & Design Principles

The Phase 5 intent taxonomy defines the structured categorization schema for incoming customer inquiries directed to AppleSupport. Grounded in empirical analysis of 82,101 candidate customer messages extracted from the Kaggle Customer Support on Twitter dataset, the taxonomy balances fine-grained operational utility with strict category separation.

### Core Design Principles
1. **Mutually Exclusive at Primary Level:** Each category addresses a distinct technical or operational concern.
2. **Empirical Grounding:** Rather than imposing arbitrary categories, classes correspond to genuine conversation clusters observed in the AppleSupport corpus (e.g., separating OS update glitches from physical battery degradation).
3. **Conservative Labeling:** The taxonomy explicitly avoids forced labeling. Where multiple categories match with similar strength, instances are designated as `needs_review`.
4. **Transparent Boundaries:** Every class includes positive inclusion criteria, explicit exclusion rules, and clear conflict-resolution guidelines.
5. **Mandatory Fallback:** Ambiguous, uninterpretable, or out-of-scope inquiries default to `unknown_other`.

---

## 2. Intent Categories (Version 2.0)

### 2.1 `software_update`
* **Display Name:** Software / OS Update Issue
* **Auto-Handle Candidate:** Yes (self-service / knowledge-base retrieval)
* **Escalation Risk:** Low
* **Definition:** The customer reports a problem, bug, or performance degradation that started after, or is directly associated with, a software update, iOS release, macOS version, or system patch.
* **Inclusion Criteria:**
  - Mentions specific OS versions (e.g., `iOS 11`, `High Sierra`, `macOS`, `watchOS`).
  - Mentions `update`, `upgrade`, `patch`, `bug`, `glitch`, `after updating`, `since updating`.
  - Mentions known widespread update bugs (e.g., the iOS 11 letter 'I' autocorrect glitch).
* **Exclusion Criteria:**
  - Physical screen/hardware damage not introduced by software ($\to$ `device_hardware`).
  - Battery issues where no update is cited ($\to$ `battery_power`).
  - Inability to sign in after update ($\to$ `account_access`).
* **Example Patterns & Utterances:**
  - *"ever since the new iOS 11 update my phone has been freezing constantly"*
  - *"why is the letter i turning into an A with a question mark on my iPhone"*
  - *"High Sierra install failed and now my MacBook Pro won't boot"*

### 2.2 `battery_power`
* **Display Name:** Battery, Power & Charging
* **Auto-Handle Candidate:** Yes (diagnostic guide retrieval)
* **Escalation Risk:** Low
* **Definition:** The customer reports abnormal battery performance, rapid battery drain, device overheating, sudden shutdowns, or charging equipment failure.
* **Inclusion Criteria:**
  - Mentions `battery`, `drain`, `draining`, `battery life`, `percentage`, `overheating`, `hot`, `charger`, `charging cable`.
  - Battery percentage dropping rapidly under normal usage.
  - Device shutting down unexpectedly at moderate battery percentages (e.g., 20–30%).
* **Exclusion Criteria:**
  - System-wide software freezes where battery is not the primary complaint ($\to$ `software_update`).
  - Swollen battery or physical enclosure bulging ($\to$ `device_hardware`).
* **Example Patterns & Utterances:**
  - *"my iPhone 7 battery drops from 80% to 20% in an hour without use"*
  - *"my phone gets extremely hot whenever I plug it into the charger"*
  - *"battery percentage jumps all over the place and shuts down at 30%"*

### 2.3 `device_hardware`
* **Display Name:** Device Hardware & Physical Damage
* **Auto-Handle Candidate:** No (requires physical repair scheduling / AppleCare)
* **Escalation Risk:** High
* **Definition:** The customer reports physical damage, screen defects, touch display unresponsiveness, camera failure, speaker/microphone issues, or broken physical buttons.
* **Inclusion Criteria:**
  - Mentions `screen`, `cracked`, `display`, `touch screen`, `unresponsive`, `camera`, `speaker`, `microphone`, `mic`, `button`, `water damage`.
  - Touchscreen fails to register finger taps.
  - Camera sensor outputs a black screen.
* **Exclusion Criteria:**
  - Display freezing caused by an app crash ($\to$ `app_or_service_issue`).
  - Charging port dirty or cable frayed ($\to$ `battery_power`).
* **Example Patterns & Utterances:**
  - *"dropped my phone and the screen is cracked and touch isn't working"*
  - *"the front camera on my iPhone shows a completely black screen"*
  - *"my speaker is crackling and people can't hear me on calls"*

### 2.4 `app_or_service_issue`
* **Display Name:** Native App or Apple Service Issue
* **Auto-Handle Candidate:** Yes (troubleshooting steps)
* **Escalation Risk:** Low
* **Definition:** The customer reports an issue or crash with a specific Apple native application or cloud service.
* **Inclusion Criteria:**
  - Names a specific Apple app or service: `Apple Music`, `iMessage`, `FaceTime`, `Safari`, `Apple Maps`, `App Store`, `Siri`, `CarPlay`, `Photos`, `Notes`, `AirDrop`.
  - App crashing repeatedly upon launch, freezing, or failing to sync content.
* **Exclusion Criteria:**
  - Third-party app issues outside Apple's ecosystem (e.g., Instagram, Spotify) ($\to$ `unknown_other`).
  - System-wide connectivity failure preventing all apps from connecting ($\to$ `connectivity_network`).
* **Example Patterns & Utterances:**
  - *"Apple Music keeps pausing and skipping songs on its own"*
  - *"iMessage is not delivering text messages to my contacts"*
  - *"FaceTime hangs on connecting whenever I call my family"*

### 2.5 `account_access`
* **Display Name:** Account, iCloud & Security
* **Auto-Handle Candidate:** No (sensitive credential verification)
* **Escalation Risk:** High
* **Definition:** The customer cannot sign in to their Apple ID, iCloud, or iTunes account, has forgotten passwords, failed 2FA verification, or has an account locked for security.
* **Inclusion Criteria:**
  - Mentions `Apple ID`, `iCloud`, `password`, `passcode`, `login`, `sign in`, `locked`, `disabled`, `verification code`, `2FA`, `two-factor`.
  - Account locked due to repeated incorrect password entries.
  - Not receiving SMS verification codes on trusted devices.
* **Exclusion Criteria:**
  - Third-party website passwords stored in Safari ($\to$ `app_or_service_issue`).
* **Example Patterns & Utterances:**
  - *"my Apple ID has been locked for security reasons and I can't reset my password"*
  - *"I'm not getting the verification code sent to my trusted number"*
  - *"forgot my iCloud password and the recovery email is old"*

### 2.6 `connectivity_network`
* **Display Name:** Connectivity & Network
* **Auto-Handle Candidate:** Yes (network reset guidance)
* **Escalation Risk:** Low
* **Definition:** The customer reports problems connecting to Wi-Fi networks, Bluetooth devices (e.g. AirPods, car audio), cellular mobile data / LTE, dropped calls, or SIM errors.
* **Inclusion Criteria:**
  - Mentions `wifi`, `wi-fi`, `bluetooth`, `cellular`, `mobile data`, `lte`, `no service`, `sim card`, `carrier`, `dropped calls`, `airpods disconnect`.
  - Device displays 'Searching...' or 'No Service' unexpectedly.
* **Exclusion Criteria:**
  - Audio glitch in music app while connected ($\to$ `app_or_service_issue`).
* **Example Patterns & Utterances:**
  - *"my iPhone keeps disconnecting from home Wi-Fi every 5 minutes"*
  - *"my phone says No Service even though my carrier network is fine"*
  - *"Bluetooth won't pair with my AirPods after restarting"*

### 2.7 `billing_payment`
* **Display Name:** Billing, Subscriptions & Payments
* **Auto-Handle Candidate:** No (financial transactions)
* **Escalation Risk:** High
* **Definition:** The customer inquires about unexpected credit card charges, recurring subscription fees, App Store purchase disputes, refund requests, or Apple Pay errors.
* **Inclusion Criteria:**
  - Mentions `billing`, `charge`, `charged`, `refund`, `subscription`, `receipt`, `purchase`, `apple pay`, `credit card`, `bank`.
  - Customer disputes an in-app purchase made accidentally.
* **Exclusion Criteria:**
  - Inability to download free apps due to account lock ($\to$ `account_access`).
* **Example Patterns & Utterances:**
  - *"I was charged $9.99 for Apple Music even though I cancelled my trial"*
  - *"how do I request a refund for an accidental purchase in the App Store"*
  - *"Apple Pay declined my card at checkout but the charge shows pending"*

### 2.8 `order_shipping`
* **Display Name:** Online Orders, Shipping & Delivery
* **Auto-Handle Candidate:** Yes (order tracking lookup)
* **Escalation Risk:** Low
* **Definition:** The customer inquires about the status, tracking, shipment delay, delivery, or in-store pickup of a physical order from the Apple Online Store.
* **Inclusion Criteria:**
  - Mentions `order`, `shipping`, `delivery`, `delivered`, `tracking number`, `ups`, `fedex`, `store pickup`, `preparing for shipment`.
  - Inquiring about shipping dispatch dates.
* **Exclusion Criteria:**
  - Digital software order disputes ($\to$ `billing_payment`).
* **Example Patterns & Utterances:**
  - *"my iPhone X order has been preparing for shipment for 5 days — when does it ship"*
  - *"tracking number says delivered but my package was not on my porch"*
  - *"can I change my delivery address for order W12345678"*

### 2.9 `feature_how_to`
* **Display Name:** Feature Question & How-To
* **Auto-Handle Candidate:** Yes (knowledge-base walkthrough)
* **Escalation Risk:** Low
* **Definition:** The customer asks how to configure, enable/disable, or find an existing feature or setting in an Apple product without reporting a technical defect.
* **Inclusion Criteria:**
  - Starts with `how do I`, `how can I`, `is there a way to`, `can you tell me how`, `where is the setting`.
  - Inquiring how to perform a task (e.g. transfer photos, screen recording).
* **Exclusion Criteria:**
  - Feature is broken or failing to execute ($\to$ `app_or_service_issue`).
* **Example Patterns & Utterances:**
  - *"how do I turn off read receipts in iMessage for one specific person"*
  - *"is there a way to record the screen on iOS without third party apps"*
  - *"how can I transfer photos from my iPhone to a Windows PC"*

### 2.10 `complaint_feedback`
* **Display Name:** Complaint & Product Feedback
* **Auto-Handle Candidate:** No (requires human de-escalation)
* **Escalation Risk:** High
* **Definition:** The customer expresses general frustration, dissatisfaction, or venting regarding Apple devices, customer service, or policies, without a specific technical support question.
* **Inclusion Criteria:**
  - Strong negative sentiment expressions (`fix your shit`, `worst phone`, `hate apple`, `terrible customer service`, `ridiculous`).
  - Venting about design decisions (e.g. headphone jack removal).
* **Exclusion Criteria:**
  - Negative sentiment accompanied by a concrete technical malfunction ($\to$ map to the specific technical intent, e.g. `battery_power` or `software_update`).
* **Example Patterns & Utterances:**
  - *"@AppleSupport your customer service in the store was absolutely terrible today"*
  - *"removing the headphone jack was the dumbest idea ever Apple fix this"*
  - *"worst update in history, so tired of Apple breaking things"*

### 2.11 `unknown_other`
* **Display Name:** Unknown / Fallback / Other
* **Auto-Handle Candidate:** No (requires clarification)
* **Escalation Risk:** Low
* **Definition:** The message does not contain sufficient semantic signal to classify into any domain intent, is overly brief/conversational, is an uninterpretable fragment, or concerns topics outside Apple product support.
* **Inclusion Criteria:**
  - Short greetings or standalone mentions (`@AppleSupport help`, `@AppleSupport`).
  - Incomplete continuation messages (`ok thanks for the help`).
  - Links without descriptive text (`@AppleSupport https://t.co/xyz123`).
* **Exclusion Criteria:**
  - Any message matching a domain intent above with confidence.
* **Example Patterns & Utterances:**
  - *"@AppleSupport hello can you DM me"*
  - *"@AppleSupport https://t.co/xyz123"*
  - *"ok thanks for the help"*

---

## 3. Ambiguity & Conflict Handling (`needs_review`)

In production customer support pipelines, forcing an arbitrary single label on a multi-intent message degrades training data quality. When a message triggers strong signatures from two or more distinct categories with comparable strength, the preliminary labeling pipeline outputs:
* `candidate_intent`: `"needs_review"`
* `label_source`: `"needs_review"`
* `label_confidence`: `"low"`
* `label_reason`: `"conflicting_rules_matched: ['category_a', 'category_b']"`

### Examples of Conflict Scenarios
1. **Battery + Software Update:**
   - *"my battery drain is terrible ever since installing the iOS 11 update"*
   - Triggers: `software_update` (score 2) and `battery_power` (score 2).
   - Flagged as `needs_review` so human reviewers can decide whether to treat this primarily as an OS update defect or a battery health inquiry.
2. **Hardware + App Service:**
   - *"screen is unresponsive when FaceTime app opens"*
   - Triggers: `device_hardware` and `app_or_service_issue`.
   - Flagged as `needs_review`.

---

## 4. Summary of Taxonomy Mapping

| Intent Name | Auto-Handle | Escalation Risk | Empirical Corpus Share |
| :--- | :--- | :--- | :--- |
| **`unknown_other`** | No | Low | 45.88% (37,669) |
| **`software_update`** | Yes | Low | 27.94% (22,942) |
| **`needs_review`** | No | Low | 7.18% (5,895) |
| **`battery_power`** | Yes | Low | 4.15% (3,406) |
| **`app_or_service_issue`** | Yes | Low | 2.88% (2,365) |
| **`account_access`** | No | High | 2.64% (2,166) |
| **`device_hardware`** | No | High | 2.60% (2,138) |
| **`feature_how_to`** | Yes | Low | 2.30% (1,889) |
| **`connectivity_network`** | Yes | Low | 2.09% (1,717) |
| **`complaint_feedback`** | No | High | 1.35% (1,105) |
| **`billing_payment`** | No | High | 0.74% (607) |
| **`order_shipping`** | Yes | Low | 0.25% (202) |
| **Total** | — | — | **100.0% (82,101)** |
