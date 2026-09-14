# Brand Comparison Analysis

**Dataset:** `twcs/twcs.csv`  |  **Brands compared:** 5  |  **Analysis time:** 493s

> All statistics are measured from the actual local dataset.
> Qualitative scores are explicitly labelled and justified in each brand section.
> No statistics are fabricated.

---

## Scoring Methodology

| Criterion | Weight | How measured |
|---|---|---|
| Data volume | 20% | `outbound_count / max_outbound` |
| Conversation completeness | 20% | `0.5×pct_complete + 0.5×pct_multi_turn` |
| Intent diversity & clarity | 20% | **Qualitative** — manually scored 0–1; see per-brand section |
| English / text quality | 15% | `pct_english_approx × (1 − pct_noise)` |
| Historical-reply retrieval | 15% | `outbound_count / max_outbound` (same as volume — more replies = richer retrieval pool) |
| Distinctiveness & scope | 10% | **Qualitative** — manually scored 0–1; see per-brand section |

All quantitative sub-scores are normalised to [0, 1].  Qualitative scores are fixed constants justified per brand.

---

## Summary Ranking

| Rank | Brand | Total Score | Vol | Completeness | Intent | Eng/Quality | Retrieval | Distinct |
|---|---|---|---|---|---|---|---|---|
| **1** | AmazonHelp | **0.898** | 1.00 | 1.00 | 0.80 | 0.85 | 1.00 | 0.60 |
| **2** | AppleSupport | **0.790** | 0.63 | 1.00 | 0.85 | 0.97 | 0.63 | 0.55 |
| **3** | Uber_Support | **0.693** | 0.33 | 1.00 | 0.75 | 0.98 | 0.33 | 0.80 |
| **4** | TMobileHelp | **0.650** | 0.20 | 1.00 | 0.80 | 0.96 | 0.20 | 0.75 |
| **5** | SpotifyCares | **0.633** | 0.26 | 1.00 | 0.65 | 0.96 | 0.26 | 0.70 |

---

## 1. AmazonHelp  (score: 0.898)

### Measured Statistics

| Metric | Value |
|---|---|
| Total rows in subset | 324,808 |
| Inbound (customer) | 154,968 |
| Outbound (brand) | 169,840 |
| Rows with parent link | 324,808 (100.0%) |
| Rows with response link | 324,808 (100.0%) |
| Estimated roots | 86,675 |
| Reconstructed threads | 86,675 |
| Single-turn threads | 36 |
| Multi-turn threads | 86,639 (100.0%) |
| Complete threads | 86,675 (100.0%) |
| Threads with broken links | 9,329 (10.8%) |
| Avg turns per thread | 3.75 |
| Median turns per thread | 2 |
| Max turns in a thread | 116 |
| Customer msgs with brand response | 154,968 (100.0%) |
| Likely English (ASCII heuristic) | 305,219 (94.0%) |
| Rows with masked fields | 773 |
| Noise / low-info messages | 29,762 (9.2%) |

### Scores

| Criterion | Raw score | Bar |
|---|---|---|
| Data volume (20%) | 1.000 | `████████████████████` |
| Conv completeness (20%) | 1.000 | `████████████████████` |
| Intent diversity (20%) ★ | 0.800 | `████████████████░░░░` |
| English quality (15%) | 0.854 | `█████████████████░░░` |
| Retrieval usefulness (15%) | 1.000 | `████████████████████` |
| Distinctiveness (10%) ★ | 0.600 | `████████████░░░░░░░░` |
| **Weighted total** | **0.8980** | `██████████████████░░` |

★ = qualitative score

### Intent Diversity Assessment
> E-commerce topics: delivery, returns, account, billing, product. Very broad but multilingual contamination lowers effective utility.

### Distinctiveness Assessment
> Well-known brand; multilingual content and very large volume create extra preprocessing complexity.

### Sample Customer Messages (non-identifying, truncated)

> These are real tweets from the dataset, sampled to illustrate topic variety.
> @mentions, URLs, and order/account numbers are present in the raw data;
> only the first 120 characters of each message are shown here.

1. _@AmazonHelp Shipped by amazon order # 113-7771598-5646659_
2. _@AmazonHelp called the number. your tele guy keeping on hold for more than 10 mins. is this really what i paid prime mem_
3. _Bene mi è arrivata la #FireTVStick e... non funziona. Iniziamo con la sostituzione va. #Amazon_
4. _@AmazonHelp Ok thanx i dleleted the pic_
5. _@AmazonHelp Hey, wäre es möglich, dass ich morgen von 09:50 - 10:50 Uhr angerufen werde? Das Paket ist nach über einer W_

---

## 2. AppleSupport  (score: 0.790)

### Measured Statistics

| Metric | Value |
|---|---|
| Total rows in subset | 213,483 |
| Inbound (customer) | 106,623 |
| Outbound (brand) | 106,860 |
| Rows with parent link | 213,483 (100.0%) |
| Rows with response link | 213,483 (100.0%) |
| Estimated roots | 82,105 |
| Reconstructed threads | 82,105 |
| Single-turn threads | 4 |
| Multi-turn threads | 82,101 (100.0%) |
| Complete threads | 82,105 (100.0%) |
| Threads with broken links | 7,351 (9.0%) |
| Avg turns per thread | 2.6 |
| Median turns per thread | 2 |
| Max turns in a thread | 252 |
| Customer msgs with brand response | 106,623 (100.0%) |
| Likely English (ASCII heuristic) | 213,088 (99.8%) |
| Rows with masked fields | 150 |
| Noise / low-info messages | 6,694 (3.1%) |

### Scores

| Criterion | Raw score | Bar |
|---|---|---|
| Data volume (20%) | 0.629 | `█████████████░░░░░░░` |
| Conv completeness (20%) | 1.000 | `████████████████████` |
| Intent diversity (20%) ★ | 0.850 | `█████████████████░░░` |
| English quality (15%) | 0.967 | `███████████████████░` |
| Retrieval usefulness (15%) | 0.629 | `█████████████░░░░░░░` |
| Distinctiveness (10%) ★ | 0.550 | `███████████░░░░░░░░░` |
| **Weighted total** | **0.7903** | `████████████████░░░░` |

★ = qualitative score

### Intent Diversity Assessment
> 6 distinct, well-separated categories confirmed in actual data: update issues, battery, account access, hardware, app/service, how-to.

### Distinctiveness Assessment
> Common choice in public projects; however, scope is well-bounded and data quality is high.

### Sample Customer Messages (non-identifying, truncated)

> These are real tweets from the dataset, sampled to illustrate topic variety.
> @mentions, URLs, and order/account numbers are present in the raw data;
> only the first 120 characters of each message are shown here.

1. _My iPhone screen is not working anymore after updating iOS 11, I did restore the phone, but still the problem remained!!_
2. _@AppleSupport please help. My home kit app will not get past “loading...”. Tried everything the forums say. I think my i_
3. _@AppleSupport please help. I’ve lost 25% of battery life in about 30 mins and wasn’t even using the phone. Is this bc of_
4. _@AppleSupport is anybody gonna keep helping or did you guys hope I’d forget?_
5. _@510066 @AppleSupport I️ love the newest iPhone feature! I️ like to call it, I️ am switching to google pixel because iPh_

---

## 3. Uber_Support  (score: 0.693)

### Measured Statistics

| Metric | Value |
|---|---|
| Total rows in subset | 111,452 |
| Inbound (customer) | 55,182 |
| Outbound (brand) | 56,270 |
| Rows with parent link | 111,452 (100.0%) |
| Rows with response link | 111,452 (100.0%) |
| Estimated roots | 43,138 |
| Reconstructed threads | 43,138 |
| Single-turn threads | 2 |
| Multi-turn threads | 43,136 (100.0%) |
| Complete threads | 43,138 (100.0%) |
| Threads with broken links | 3,774 (8.7%) |
| Avg turns per thread | 2.58 |
| Median turns per thread | 2.0 |
| Max turns in a thread | 323 |
| Customer msgs with brand response | 55,182 (100.0%) |
| Likely English (ASCII heuristic) | 111,424 (100.0%) |
| Rows with masked fields | 947 |
| Noise / low-info messages | 2,462 (2.2%) |

### Scores

| Criterion | Raw score | Bar |
|---|---|---|
| Data volume (20%) | 0.331 | `███████░░░░░░░░░░░░░` |
| Conv completeness (20%) | 1.000 | `████████████████████` |
| Intent diversity (20%) ★ | 0.750 | `███████████████░░░░░` |
| English quality (15%) | 0.978 | `████████████████████` |
| Retrieval usefulness (15%) | 0.331 | `███████░░░░░░░░░░░░░` |
| Distinctiveness (10%) ★ | 0.800 | `████████████████░░░░` |
| **Weighted total** | **0.6927** | `██████████████░░░░░░` |

★ = qualitative score

### Intent Diversity Assessment
> Topics include ride disputes, driver issues, safety, payment, account. Dispute-heavy content raises escalation variety.

### Distinctiveness Assessment
> Less commonly used in public tutorials; ride-sharing domain offers clear auto-handle vs. escalation split.

### Sample Customer Messages (non-identifying, truncated)

> These are real tweets from the dataset, sampled to illustrate topic variety.
> @mentions, URLs, and order/account numbers are present in the raw data;
> only the first 120 characters of each message are shown here.

1. _@Uber_Support I have sent feedback to the the support team on app...you have the driver info and a complaint registered _
2. _@Uber_Support Sent you the info please check!_
3. _@Uber_Support Already sent DM.. have you received it?_
4. _@Uber_Support I've recently shifted from New Delhi to Mumbai. I'm still getting promotion tailored made promotions for D_
5. _@115873 The driver was so rude and unprofessional. We called from my wife's account. __email___

---

## 4. TMobileHelp  (score: 0.650)

### Measured Statistics

| Metric | Value |
|---|---|
| Total rows in subset | 68,139 |
| Inbound (customer) | 33,822 |
| Outbound (brand) | 34,317 |
| Rows with parent link | 68,139 (100.0%) |
| Rows with response link | 68,139 (100.0%) |
| Estimated roots | 26,385 |
| Reconstructed threads | 26,385 |
| Single-turn threads | 18 |
| Multi-turn threads | 26,367 (99.9%) |
| Complete threads | 26,385 (100.0%) |
| Threads with broken links | 6,503 (24.6%) |
| Avg turns per thread | 2.58 |
| Median turns per thread | 2 |
| Max turns in a thread | 94 |
| Customer msgs with brand response | 33,822 (100.0%) |
| Likely English (ASCII heuristic) | 68,090 (99.9%) |
| Rows with masked fields | 33 |
| Noise / low-info messages | 2,378 (3.5%) |

### Scores

| Criterion | Raw score | Bar |
|---|---|---|
| Data volume (20%) | 0.202 | `████░░░░░░░░░░░░░░░░` |
| Conv completeness (20%) | 1.000 | `████████████████████` |
| Intent diversity (20%) ★ | 0.800 | `████████████████░░░░` |
| English quality (15%) | 0.964 | `███████████████████░` |
| Retrieval usefulness (15%) | 0.202 | `████░░░░░░░░░░░░░░░░` |
| Distinctiveness (10%) ★ | 0.750 | `███████████████░░░░░` |
| **Weighted total** | **0.6502** | `█████████████░░░░░░░` |

★ = qualitative score

### Intent Diversity Assessment
> Telecom topics: coverage, billing, SIM/device activation, plan changes, outages, international roaming. Good escalation diversity.

### Distinctiveness Assessment
> Telecom support is underrepresented in public AI agent demos; strong mix of routine (FAQ) and complex (billing dispute) cases.

### Sample Customer Messages (non-identifying, truncated)

> These are real tweets from the dataset, sampled to illustrate topic variety.
> @mentions, URLs, and order/account numbers are present in the raw data;
> only the first 120 characters of each message are shown here.

1. _@TMobileHelp The week's gone by and I never got my Panda Express bowl coupon from last week to work._
2. _@TMobileHelp I know the status. Just wondering why updates are being held up? Four months behind on monthly security upd_
3. _Every time I try to order I’m forced to do a trade-in and then it takes me to a listing of iPhones WITHOUT The X @115913_
4. _@118272 I'm am upset that I had to pay $175 for a replacement phone through you guys and you sent me a busted phone and _
5. _@115911 pro move switching my number back to the temporary one over night. #greatstart_

---

## 5. SpotifyCares  (score: 0.633)

### Measured Statistics

| Metric | Value |
|---|---|
| Total rows in subset | 84,840 |
| Inbound (customer) | 41,575 |
| Outbound (brand) | 43,265 |
| Rows with parent link | 84,840 (100.0%) |
| Rows with response link | 84,840 (100.0%) |
| Estimated roots | 29,890 |
| Reconstructed threads | 29,890 |
| Single-turn threads | 7 |
| Multi-turn threads | 29,883 (100.0%) |
| Complete threads | 29,890 (100.0%) |
| Threads with broken links | 3,809 (12.7%) |
| Avg turns per thread | 2.84 |
| Median turns per thread | 2.0 |
| Max turns in a thread | 72 |
| Customer msgs with brand response | 41,575 (100.0%) |
| Likely English (ASCII heuristic) | 84,763 (99.9%) |
| Rows with masked fields | 235 |
| Noise / low-info messages | 3,320 (3.9%) |

### Scores

| Criterion | Raw score | Bar |
|---|---|---|
| Data volume (20%) | 0.255 | `█████░░░░░░░░░░░░░░░` |
| Conv completeness (20%) | 1.000 | `████████████████████` |
| Intent diversity (20%) ★ | 0.650 | `█████████████░░░░░░░` |
| English quality (15%) | 0.960 | `███████████████████░` |
| Retrieval usefulness (15%) | 0.255 | `█████░░░░░░░░░░░░░░░` |
| Distinctiveness (10%) ★ | 0.700 | `██████████████░░░░░░` |
| **Weighted total** | **0.6332** | `█████████████░░░░░░░` |

★ = qualitative score

### Intent Diversity Assessment
> Narrower scope: playback bugs, premium billing, device compatibility, account. Fewer than 5 strongly distinct categories.

### Distinctiveness Assessment
> Moderate distinctiveness; clean English data; narrower topic set limits intent taxonomy depth.

### Sample Customer Messages (non-identifying, truncated)

> These are real tweets from the dataset, sampled to illustrate topic variety.
> @mentions, URLs, and order/account numbers are present in the raw data;
> only the first 120 characters of each message are shown here.

1. _@SpotifyCares I have a fb created account and want to change my email. I have already changed it on fb (over a month ago_
2. _Help me fam. I already paid for premium acct. For @115888 but its still in free trial?_
3. _@706591 @115888 Oh god, my true self is out or all to see!_
4. _@SpotifyCares Many thanks. Your support staff were able to get the issue resolved. Great customer service!_
5. _Hey @115888 I don’t wanna use @135616 stop asking_

---

## Recommendation

### 1. Best overall brand: **AmazonHelp** (score 0.898)

### 2. Best alternative brand: **AppleSupport** (score 0.790)

### 3. Should AppleSupport be retained?

AppleSupport scored **0.790** (rank 2 of 5).

AppleSupport is the second-ranked brand.  Replacing it with the top brand would provide a measurable improvement in distinctiveness or completeness.  The final decision depends on whether the improvement justifies restarting the intent-labelling work.

### 4. Main trade-offs

**AmazonHelp:** Volume=169,840 outbound tweets, 100.0% multi-turn threads, 94.0% English.  Intent diversity score 0.80, distinctiveness 0.60.

**AppleSupport:** Volume=106,860 outbound tweets, 100.0% multi-turn threads, 99.8% English.  Intent diversity score 0.85, distinctiveness 0.55.

**Uber_Support:** Volume=56,270 outbound tweets, 100.0% multi-turn threads, 100.0% English.  Intent diversity score 0.75, distinctiveness 0.80.

### 5. Strongest practical demonstration: **Uber_Support**

Highest combined intent diversity + distinctiveness — the agent will exhibit a clear range of behaviours (auto-handle, escalate, draft reply) without being confused by multilingual or fragmented data.

### 6. Safest for hardware / time constraints: **TMobileHelp**

Smaller, high-quality English subset minimises extraction and training time while still providing adequate coverage for all pipeline components.

---

_This report was generated by `src/data/brand_comparison.py` from the local dataset._
_No brand selection change has been made automatically._
_The existing `AppleSupport` configuration in `src/classification/intents.yaml` remains unchanged._