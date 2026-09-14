# Brand Comparison — Validated Analysis

> **This report supersedes `brand_comparison.md`.**
> All metrics now have precise definitions and measurement labels.
> Misleading 100% figures from the original report are explained and corrected.

**Brands compared:** 5  |  **Analysis time:** 497s

---

## Why the Original Report Showed 100% for Many Metrics

The original analysis used a **biased extraction subset**: only rows that
participated in a brand conversation (brand outbound replies ∪ customer tweets
that received a reply) were collected. This excluded all unanswered customer
tweets by construction, producing the following artifacts:

| Original metric | Reported value | Why it was inflated |
|---|---|---|
| Parent-link coverage | 100% | Every row was collected because it had a link |
| Response-link coverage | 100% | Same extraction bias |
| Customer msgs responded to | 100% | Unanswered tweets never entered subset |
| Multi-turn threads | ~100% | Every thread had ≥ 2 rows (by construction) |
| 'Complete' threads | 100% | Last message always outbound (by construction) |

These artifacts do NOT mean the original code was wrong — they mean the
metrics were measured on the wrong denominator. The corrected analysis uses
precise definitions and documents each measurement type.

---

## Metric Measurement Labels

| Label | Meaning |
|---|---|
| **DM** | Directly measured from data — no heuristic or threshold |
| **RB** | Rule-based estimate — uses a threshold or pattern; reproducible but approximate |
| **QL** | Qualitative — manually assessed; justified in text but not auto-computed |

---

## Corrected Metric Definitions

### Link Coverage
- **Link field present** (DM): the CSV column is non-null and non-empty.
- **Valid reference** (DM): the referenced tweet_id exists in the collected subset.
- **Broken reference** (DM): field present but referenced ID is not in subset.

### Response Coverage (inbound rows only)
- **Level 1 — field present** (DM): `response_tweet_id` is non-null.
- **Level 2 — exists in subset** (DM): referenced ID is in the brand subset.
- **Level 3 — is brand tweet** (DM): referenced ID is authored by the brand.
- **Unanswered estimate** (RB): inbound tweets @mentioning the brand but not in
  the subset. Under-counts customers who replied without an @mention.

### Thread Quality
- **≥ 2 messages** (DM): thread has at least two rows.
- **Both directions** (DM): thread has ≥ 1 inbound AND ≥ 1 outbound message.
- **≥ 3 messages** (DM): requires genuine back-and-forth beyond Q+A.
- **Multiple exchanges** (DM): at least one outbound→inbound transition exists
  (customer replied back after brand response). Strictest quality criterion.
- **Ends with brand** (DM): last chronological message is outbound.
  (This was called 'complete' in the original — renamed to avoid ambiguity.)

---

## Summary Rankings

### Volume-Oriented Ranking
_(Weights: vol 20%, conv_both_dirs 20%, intent QL 20%, eng 15%, retrieval 15%, distinct QL 10%)_
_(Note: volume and retrieval use the same base metric — documented overlap.)_

| Rank | Brand | Score | Vol | Both-dirs | Intent QL | Eng | Retrieval | Distinct QL |
|---|---|---|---|---|---|---|---|---|
| **1** | AmazonHelp | **0.883** | 1.00 | 1.00 | 0.75 | 0.85 | 1.00 | 0.55 |
| **2** | AppleSupport | **0.790** | 0.63 | 1.00 | 0.85 | 0.97 | 0.63 | 0.55 |
| **3** | Uber_Support | **0.693** | 0.33 | 1.00 | 0.75 | 0.98 | 0.33 | 0.80 |
| **4** | TMobileHelp | **0.650** | 0.20 | 1.00 | 0.80 | 0.96 | 0.20 | 0.75 |
| **5** | SpotifyCares | **0.623** | 0.26 | 1.00 | 0.60 | 0.96 | 0.26 | 0.70 |

### Quality-Oriented Ranking
_(Weights: conv_composite 30%, intent QL 25%, eng_quality 25%, distinct QL 20%)_
_(No volume overlap. Favours clean, structured data over raw size.)_

| Rank | Brand | Score | Conv composite | Intent QL | Eng quality | Distinct QL |
|---|---|---|---|---|---|---|
| **1** | Uber_Support | **0.749** | 0.52 | 0.75 | 0.98 | 0.80 |
| **2** | TMobileHelp | **0.748** | 0.52 | 0.80 | 0.96 | 0.75 |
| **3** | AppleSupport | **0.722** | 0.53 | 0.85 | 0.97 | 0.55 |
| **4** | AmazonHelp | **0.712** | 0.67 | 0.75 | 0.85 | 0.55 |
| **5** | SpotifyCares | **0.697** | 0.56 | 0.60 | 0.96 | 0.70 |

---

## 1. AmazonHelp
_(Volume rank: 1 | Quality rank: 4)_

### Volume (DM)
| Metric | Value |
|---|---|
| Total subset rows | 324,808 |
| Inbound (customer) | 154,968 |
| Outbound (brand) | 169,840 |

### Link Coverage (DM)
| Metric | Count | % of total |
|---|---|---|
| Parent field present | 247,462 | 76.2% |
| Parent valid in subset | 238,133 | 96.2% of field-present |
| Parent broken refs | 9,329 | — |
| Response field present | 240,242 | 74.0% |
| Response valid in subset | 206,625 | 86.0% of field-present |
| Response is brand tweet | 154,977 | 64.5% of field-present |
| Response broken refs | 33,617 | — |

### Response Coverage — Inbound Rows Only (DM)
| Level | Count | % of inbound |
|---|---|---|
| Level 1: response field present | 154,968 | 100.0% |
| Level 2: response exists in subset | 149,925 | 96.7% |
| Level 3: response is brand tweet | 154,968 | 100.0% |

### Unanswered Customer Estimate (RB — @mention heuristic)
| Metric | Value |
|---|---|
| Inbound in subset (answered) | 154,968 |
| Est. unanswered (not in subset) | 42,277 |
| Est. total customer reach | 197,245 |
| Est. true response rate | 78.6% |
| Note | Estimated via @mention heuristic on inbound tweets not in brand subset. Under-counts customers who replied without re-mentioning the brand. LABEL: rule_based estimate. |

### Thread Quality (DM)
| Metric | Count | % of threads |
|---|---|---|
| Total reconstructed threads | 86,675 | 100% |
| Single-message threads | 36 | 0.0% |
| **≥ 2 messages** | 86,639 | **100.0%** |
| **Both directions (customer + brand)** | 86,639 | **100.0%** |
| **≥ 3 messages** | 41,429 | **47.8%** |
| **Multiple exchanges (cust→brand→cust)** | 36,810 | **42.5%** |
| Ends with brand reply | 86,675 | 100.0% |
| No broken links | 77,346 | 89.2% |
| Has broken links | 9,329 | 10.8% |
| Avg thread length | 3.75 | — |
| Median thread length | 2 | — |
| Max thread length | 116 | — |

### Text Quality (RB)
| Metric | Value | Note |
|---|---|---|
| Likely English | 305,219 (94.0%) | ASCII-ratio ≥ 85% threshold. LABEL: rule_based estimate. Transliterated text may be mis-classified. |
| Noise messages | 29,762 (9.2%) | Useful alphabetic word count < 3 after stripping @mentions and URLs. LABEL: rule_based estimate. |
| Rows with masked fields | 773 | Counts __token__ pattern only. NOT a complete PII audit. Other PII forms (bare phone numbers, emails) are not counted here. LABEL: directly_measured for this specific pattern. |

### Assessment (QL — manually assessed, not auto-computed)
- **Intent diversity** (0.75/1.0): QL: E-commerce topics are broad. Non-English contamination (~6%) requires language filtering before any classifier training.
- **Distinctiveness** (0.55/1.0): QL: Well-known brand; commonly used. Multilingual contamination adds preprocessing complexity that other brands don't require.
- **Main risks**: 6% non-English rows (~19k messages) require language detection step. Volume advantage is partially offset by this quality cost.

### Scores
| Ranking | Score | Bar |
|---|---|---|
| Volume-oriented (rank 1) | 0.8830 | `████████████████░░` |
| Quality-oriented (rank 4) | 0.7121 | `█████████████░░░░░` |

### Sample Customer Messages (DM — non-identifying, truncated to 120 chars)

1. _@AmazonHelp Shipped by amazon order # 113-7771598-5646659_
2. _@AmazonHelp called the number. your tele guy keeping on hold for more than 10 mins. is this really what i paid prime mem_
3. _Bene mi è arrivata la #FireTVStick e... non funziona. Iniziamo con la sostituzione va. #Amazon_
4. _@AmazonHelp Ok thanx i dleleted the pic_
5. _@AmazonHelp Hey, wäre es möglich, dass ich morgen von 09:50 - 10:50 Uhr angerufen werde? Das Paket ist nach über einer W_

---

## 2. AppleSupport
_(Volume rank: 2 | Quality rank: 3)_

### Volume (DM)
| Metric | Value |
|---|---|
| Total subset rows | 213,483 |
| Inbound (customer) | 106,623 |
| Outbound (brand) | 106,860 |

### Link Coverage (DM)
| Metric | Count | % of total |
|---|---|---|
| Parent field present | 138,729 | 65.0% |
| Parent valid in subset | 131,378 | 94.7% of field-present |
| Parent broken refs | 7,351 | — |
| Response field present | 138,187 | 64.7% |
| Response valid in subset | 119,445 | 86.4% of field-present |
| Response is brand tweet | 106,625 | 77.2% of field-present |
| Response broken refs | 18,742 | — |

### Response Coverage — Inbound Rows Only (DM)
| Level | Count | % of inbound |
|---|---|---|
| Level 1: response field present | 106,623 | 100.0% |
| Level 2: response exists in subset | 99,508 | 93.3% |
| Level 3: response is brand tweet | 106,623 | 100.0% |

### Unanswered Customer Estimate (RB — @mention heuristic)
| Metric | Value |
|---|---|
| Inbound in subset (answered) | 106,623 |
| Est. unanswered (not in subset) | 19,285 |
| Est. total customer reach | 125,908 |
| Est. true response rate | 84.7% |
| Note | Estimated via @mention heuristic on inbound tweets not in brand subset. Under-counts customers who replied without re-mentioning the brand. LABEL: rule_based estimate. |

### Thread Quality (DM)
| Metric | Count | % of threads |
|---|---|---|
| Total reconstructed threads | 82,105 | 100% |
| Single-message threads | 4 | 0.0% |
| **≥ 2 messages** | 82,101 | **100.0%** |
| **Both directions (customer + brand)** | 82,101 | **100.0%** |
| **≥ 3 messages** | 17,359 | **21.1%** |
| **Multiple exchanges (cust→brand→cust)** | 17,065 | **20.8%** |
| Ends with brand reply | 82,105 | 100.0% |
| No broken links | 74,754 | 91.0% |
| Has broken links | 7,351 | 9.0% |
| Avg thread length | 2.6 | — |
| Median thread length | 2 | — |
| Max thread length | 252 | — |

### Text Quality (RB)
| Metric | Value | Note |
|---|---|---|
| Likely English | 213,088 (99.8%) | ASCII-ratio ≥ 85% threshold. LABEL: rule_based estimate. Transliterated text may be mis-classified. |
| Noise messages | 6,694 (3.1%) | Useful alphabetic word count < 3 after stripping @mentions and URLs. LABEL: rule_based estimate. |
| Rows with masked fields | 150 | Counts __token__ pattern only. NOT a complete PII audit. Other PII forms (bare phone numbers, emails) are not counted here. LABEL: directly_measured for this specific pattern. |

### Assessment (QL — manually assessed, not auto-computed)
- **Intent diversity** (0.85/1.0): QL: 6 well-separated categories confirmed in actual data: update issues, battery, account access, hardware, app/service, how-to.
- **Distinctiveness** (0.55/1.0): QL: Commonly used in public AI-support demos. Well-bounded, high-quality English data. Scores lower on novelty.
- **Main risks**: Lower distinctiveness; intent taxonomy may overlap with public examples.

### Scores
| Ranking | Score | Bar |
|---|---|---|
| Volume-oriented (rank 2) | 0.7903 | `██████████████░░░░` |
| Quality-oriented (rank 3) | 0.7220 | `█████████████░░░░░` |

### Sample Customer Messages (DM — non-identifying, truncated to 120 chars)

1. _My iPhone screen is not working anymore after updating iOS 11, I did restore the phone, but still the problem remained!!_
2. _@AppleSupport please help. My home kit app will not get past “loading...”. Tried everything the forums say. I think my i_
3. _@AppleSupport please help. I’ve lost 25% of battery life in about 30 mins and wasn’t even using the phone. Is this bc of_
4. _@AppleSupport is anybody gonna keep helping or did you guys hope I’d forget?_
5. _@510066 @AppleSupport I️ love the newest iPhone feature! I️ like to call it, I️ am switching to google pixel because iPh_

---

## 3. Uber_Support
_(Volume rank: 3 | Quality rank: 1)_

### Volume (DM)
| Metric | Value |
|---|---|
| Total subset rows | 111,452 |
| Inbound (customer) | 55,182 |
| Outbound (brand) | 56,270 |

### Link Coverage (DM)
| Metric | Count | % of total |
|---|---|---|
| Parent field present | 72,088 | 64.7% |
| Parent valid in subset | 68,314 | 94.8% of field-present |
| Parent broken refs | 3,774 | — |
| Response field present | 73,218 | 65.7% |
| Response valid in subset | 62,321 | 85.1% of field-present |
| Response is brand tweet | 55,215 | 75.4% of field-present |
| Response broken refs | 10,897 | — |

### Response Coverage — Inbound Rows Only (DM)
| Level | Count | % of inbound |
|---|---|---|
| Level 1: response field present | 55,182 | 100.0% |
| Level 2: response exists in subset | 53,039 | 96.1% |
| Level 3: response is brand tweet | 55,182 | 100.0% |

### Unanswered Customer Estimate (RB — @mention heuristic)
| Metric | Value |
|---|---|
| Inbound in subset (answered) | 55,182 |
| Est. unanswered (not in subset) | 14,052 |
| Est. total customer reach | 69,234 |
| Est. true response rate | 79.7% |
| Note | Estimated via @mention heuristic on inbound tweets not in brand subset. Under-counts customers who replied without re-mentioning the brand. LABEL: rule_based estimate. |

### Thread Quality (DM)
| Metric | Count | % of threads |
|---|---|---|
| Total reconstructed threads | 43,138 | 100% |
| Single-message threads | 2 | 0.0% |
| **≥ 2 messages** | 43,136 | **100.0%** |
| **Both directions (customer + brand)** | 43,136 | **100.0%** |
| **≥ 3 messages** | 9,200 | **21.3%** |
| **Multiple exchanges (cust→brand→cust)** | 8,629 | **20.0%** |
| Ends with brand reply | 43,138 | 100.0% |
| No broken links | 39,364 | 91.3% |
| Has broken links | 3,774 | 8.7% |
| Avg thread length | 2.58 | — |
| Median thread length | 2.0 | — |
| Max thread length | 323 | — |

### Text Quality (RB)
| Metric | Value | Note |
|---|---|---|
| Likely English | 111,424 (100.0%) | ASCII-ratio ≥ 85% threshold. LABEL: rule_based estimate. Transliterated text may be mis-classified. |
| Noise messages | 2,462 (2.2%) | Useful alphabetic word count < 3 after stripping @mentions and URLs. LABEL: rule_based estimate. |
| Rows with masked fields | 947 | Counts __token__ pattern only. NOT a complete PII audit. Other PII forms (bare phone numbers, emails) are not counted here. LABEL: directly_measured for this specific pattern. |

### Assessment (QL — manually assessed, not auto-computed)
- **Intent diversity** (0.75/1.0): QL: Clear categories: ride issues, driver disputes, payment, safety, account. Dispute-heavy cases add escalation variety.
- **Distinctiveness** (0.80/1.0): QL: Underrepresented in public tutorials. Clear auto-handle vs. escalation split in ride-sharing domain.
- **Main risks**: Smaller training pool (~56k outbound). Some dispute topics are sensitive and may be hard to auto-resolve.

### Scores
| Ranking | Score | Bar |
|---|---|---|
| Volume-oriented (rank 3) | 0.6927 | `████████████░░░░░░` |
| Quality-oriented (rank 1) | 0.7492 | `█████████████░░░░░` |

### Sample Customer Messages (DM — non-identifying, truncated to 120 chars)

1. _@Uber_Support I have sent feedback to the the support team on app...you have the driver info and a complaint registered _
2. _@Uber_Support Sent you the info please check!_
3. _@Uber_Support Already sent DM.. have you received it?_
4. _@Uber_Support I've recently shifted from New Delhi to Mumbai. I'm still getting promotion tailored made promotions for D_
5. _@115873 The driver was so rude and unprofessional. We called from my wife's account. __email___

---

## 4. TMobileHelp
_(Volume rank: 4 | Quality rank: 2)_

### Volume (DM)
| Metric | Value |
|---|---|
| Total subset rows | 68,139 |
| Inbound (customer) | 33,822 |
| Outbound (brand) | 34,317 |

### Link Coverage (DM)
| Metric | Count | % of total |
|---|---|---|
| Parent field present | 48,257 | 70.8% |
| Parent valid in subset | 41,754 | 86.5% of field-present |
| Parent broken refs | 6,503 | — |
| Response field present | 43,581 | 64.0% |
| Response valid in subset | 38,402 | 88.1% of field-present |
| Response is brand tweet | 33,836 | 77.6% of field-present |
| Response broken refs | 5,179 | — |

### Response Coverage — Inbound Rows Only (DM)
| Level | Count | % of inbound |
|---|---|---|
| Level 1: response field present | 33,822 | 100.0% |
| Level 2: response exists in subset | 32,765 | 96.9% |
| Level 3: response is brand tweet | 33,822 | 100.0% |

### Unanswered Customer Estimate (RB — @mention heuristic)
| Metric | Value |
|---|---|
| Inbound in subset (answered) | 33,822 |
| Est. unanswered (not in subset) | 6,343 |
| Est. total customer reach | 40,165 |
| Est. true response rate | 84.2% |
| Note | Estimated via @mention heuristic on inbound tweets not in brand subset. Under-counts customers who replied without re-mentioning the brand. LABEL: rule_based estimate. |

### Thread Quality (DM)
| Metric | Count | % of threads |
|---|---|---|
| Total reconstructed threads | 26,385 | 100% |
| Single-message threads | 18 | 0.1% |
| **≥ 2 messages** | 26,367 | **99.9%** |
| **Both directions (customer + brand)** | 26,367 | **99.9%** |
| **≥ 3 messages** | 5,530 | **21.0%** |
| **Multiple exchanges (cust→brand→cust)** | 5,343 | **20.3%** |
| Ends with brand reply | 26,385 | 100.0% |
| No broken links | 19,882 | 75.4% |
| Has broken links | 6,503 | 24.6% |
| Avg thread length | 2.58 | — |
| Median thread length | 2 | — |
| Max thread length | 94 | — |

### Text Quality (RB)
| Metric | Value | Note |
|---|---|---|
| Likely English | 68,090 (99.9%) | ASCII-ratio ≥ 85% threshold. LABEL: rule_based estimate. Transliterated text may be mis-classified. |
| Noise messages | 2,378 (3.5%) | Useful alphabetic word count < 3 after stripping @mentions and URLs. LABEL: rule_based estimate. |
| Rows with masked fields | 33 | Counts __token__ pattern only. NOT a complete PII audit. Other PII forms (bare phone numbers, emails) are not counted here. LABEL: directly_measured for this specific pattern. |

### Assessment (QL — manually assessed, not auto-computed)
- **Intent diversity** (0.80/1.0): QL: Telecom topics: coverage, billing, SIM, activation, outages, roaming. Strong escalation variety (billing disputes, service outages).
- **Distinctiveness** (0.75/1.0): QL: Telecom support underrepresented in public AI demos. But broken-link rate is concerning (24.6% of threads).
- **Main risks**: 24.6% broken-link thread rate is highest of all five brands — indicates significant missing conversation context in dataset.

### Scores
| Ranking | Score | Bar |
|---|---|---|
| Volume-oriented (rank 4) | 0.6501 | `████████████░░░░░░` |
| Quality-oriented (rank 2) | 0.7481 | `█████████████░░░░░` |

### Sample Customer Messages (DM — non-identifying, truncated to 120 chars)

1. _@TMobileHelp The week's gone by and I never got my Panda Express bowl coupon from last week to work._
2. _@TMobileHelp I know the status. Just wondering why updates are being held up? Four months behind on monthly security upd_
3. _Every time I try to order I’m forced to do a trade-in and then it takes me to a listing of iPhones WITHOUT The X @115913_
4. _@118272 I'm am upset that I had to pay $175 for a replacement phone through you guys and you sent me a busted phone and _
5. _@115911 pro move switching my number back to the temporary one over night. #greatstart_

---

## 5. SpotifyCares
_(Volume rank: 5 | Quality rank: 5)_

### Volume (DM)
| Metric | Value |
|---|---|
| Total subset rows | 84,840 |
| Inbound (customer) | 41,575 |
| Outbound (brand) | 43,265 |

### Link Coverage (DM)
| Metric | Count | % of total |
|---|---|---|
| Parent field present | 58,759 | 69.3% |
| Parent valid in subset | 54,950 | 93.5% of field-present |
| Parent broken refs | 3,809 | — |
| Response field present | 55,361 | 65.3% |
| Response valid in subset | 51,366 | 92.8% of field-present |
| Response is brand tweet | 41,687 | 75.3% of field-present |
| Response broken refs | 3,995 | — |

### Response Coverage — Inbound Rows Only (DM)
| Level | Count | % of inbound |
|---|---|---|
| Level 1: response field present | 41,575 | 100.0% |
| Level 2: response exists in subset | 40,838 | 98.2% |
| Level 3: response is brand tweet | 41,575 | 100.0% |

### Unanswered Customer Estimate (RB — @mention heuristic)
| Metric | Value |
|---|---|
| Inbound in subset (answered) | 41,575 |
| Est. unanswered (not in subset) | 5,179 |
| Est. total customer reach | 46,754 |
| Est. true response rate | 88.9% |
| Note | Estimated via @mention heuristic on inbound tweets not in brand subset. Under-counts customers who replied without re-mentioning the brand. LABEL: rule_based estimate. |

### Thread Quality (DM)
| Metric | Count | % of threads |
|---|---|---|
| Total reconstructed threads | 29,890 | 100% |
| Single-message threads | 7 | 0.0% |
| **≥ 2 messages** | 29,883 | **100.0%** |
| **Both directions (customer + brand)** | 29,880 | **100.0%** |
| **≥ 3 messages** | 8,239 | **27.6%** |
| **Multiple exchanges (cust→brand→cust)** | 7,496 | **25.1%** |
| Ends with brand reply | 29,890 | 100.0% |
| No broken links | 26,081 | 87.3% |
| Has broken links | 3,809 | 12.7% |
| Avg thread length | 2.84 | — |
| Median thread length | 2.0 | — |
| Max thread length | 72 | — |

### Text Quality (RB)
| Metric | Value | Note |
|---|---|---|
| Likely English | 84,763 (99.9%) | ASCII-ratio ≥ 85% threshold. LABEL: rule_based estimate. Transliterated text may be mis-classified. |
| Noise messages | 3,320 (3.9%) | Useful alphabetic word count < 3 after stripping @mentions and URLs. LABEL: rule_based estimate. |
| Rows with masked fields | 235 | Counts __token__ pattern only. NOT a complete PII audit. Other PII forms (bare phone numbers, emails) are not counted here. LABEL: directly_measured for this specific pattern. |

### Assessment (QL — manually assessed, not auto-computed)
- **Intent diversity** (0.60/1.0): QL: Narrower scope — playback, premium billing, account, device compat. Fewer than 5 strongly distinct categories visible in data.
- **Distinctiveness** (0.70/1.0): QL: Moderate novelty. Cleanest English data but limited topic depth.
- **Main risks**: Narrow topic set limits intent taxonomy depth; fewer escalation examples.

### Scores
| Ranking | Score | Bar |
|---|---|---|
| Volume-oriented (rank 5) | 0.6232 | `███████████░░░░░░░` |
| Quality-oriented (rank 5) | 0.6974 | `█████████████░░░░░` |

### Sample Customer Messages (DM — non-identifying, truncated to 120 chars)

1. _@SpotifyCares I have a fb created account and want to change my email. I have already changed it on fb (over a month ago_
2. _Help me fam. I already paid for premium acct. For @115888 but its still in free trial?_
3. _@706591 @115888 Oh god, my true self is out or all to see!_
4. _@SpotifyCares Many thanks. Your support staff were able to get the issue resolved. Great customer service!_
5. _Hey @115888 I don’t wanna use @135616 stop asking_

---

## Recommendation

### Volume-oriented best: **AmazonHelp** | Quality-oriented best: **Uber_Support**

### AppleSupport vs Uber_Support — Practical Comparison

| Dimension | AppleSupport | Uber_Support |
|---|---|---|
| Outbound training pool | 106,860 | 56,270 |
| Threads with both directions | 100.0% | 100.0% |
| Threads ≥ 3 messages | 21.1% | 21.3% |
| Multiple-exchange threads | 20.8% | 20.0% |
| Likely English | 99.8% | 100.0% |
| Noise rate | 3.1% | 2.2% |
| Broken-link threads | 9.0% | 8.7% |
| Est. true response rate | 84.7% | 79.7% |
| Intent diversity (QL) | 0.85 | 0.75 |
| Distinctiveness (QL) | 0.55 | 0.80 |
| Volume rank | 2 | 3 |
| Quality rank | 3 | 1 |

### Conclusions

1. **Best overall (volume-oriented):** **AmazonHelp** — largest outbound pool and highest composite score when data volume is weighted heavily.

2. **Best overall (quality-oriented):** **Uber_Support** — best combination of conversation structure, text cleanliness, intent diversity, and domain distinctiveness.

3. **Should AppleSupport remain selected?**
   AppleSupport ranks **2** (volume) and **3** (quality).
   It has the highest intent-diversity score of all candidates (0.85 QL) and
   the best English-quality score among high-volume brands (99.8%, 3.1% noise).
   Its main weakness is lower distinctiveness (0.55 QL) and smaller outbound
   pool than AmazonHelp.

4. **Is Uber_Support a better practical choice?**
   Uber_Support ranks higher on quality and distinctiveness, with the cleanest
   English corpus (100%, 2.2% noise) and highest distinctiveness score (0.80 QL).
   However, its outbound pool (~56k) is roughly half of AppleSupport's (~107k),
   which matters for retrieval-based reply generation.
   Uber_Support's estimated true response rate (79.7%) vs AppleSupport (84.7%).

5. **Main trade-offs:**
   - **AppleSupport**: larger retrieval pool, highest intent diversity, well-known domain — lower distinctiveness.
   - **Uber_Support**: cleanest data, most distinctive domain, clear escalation scenarios — smaller training pool.

6. **Recommendation:** The selected brand should remain **AppleSupport**
   unless you specifically want a more distinctive domain and can accept
   the smaller training pool. If distinctiveness matters more than retrieval
   depth, **Uber_Support** is the correct alternative.

   Both are technically sound choices. The final decision is yours.

---

_No brand configuration has been changed. `src/classification/intents.yaml` is unchanged._
_This report was generated by `src/data/brand_comparison_validated.py`._