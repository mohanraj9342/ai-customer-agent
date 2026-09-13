# Brand Selection — Decision Record

## Summary

**Selected brand: `AppleSupport`**

This document records the evidence-based reasoning for choosing AppleSupport
as the target brand for this project.  No brand was selected by assumption.

---

## Candidate Assessment

The following measurements were taken from the full dataset (2,811,774 logical
rows; see `docs/dataset_inspection.json` for full statistics).

| Rank | Brand | Outbound Tweets | Assessment |
|------|-------|----------------|------------|
| 1 | AmazonHelp | 169,840 | Largest volume but contains multilingual content (Japanese, German, etc.) — complicates single-model intent classification |
| **2** | **AppleSupport** | **106,860** | **Selected — see rationale below** |
| 3 | Uber_Support | 56,270 | Good volume; topics include driver/rider disputes that are hard to auto-handle |
| 4 | SpotifyCares | 43,265 | Strong runner-up; narrower topic set |
| 5 | Delta | 42,253 | Airline topics have high regulatory sensitivity and compensation complexity |

---

## Selection Criteria Applied

Each criterion was assessed by inspecting actual customer tweet examples from
the dataset — not by assumption.

| Criterion | AppleSupport | SpotifyCares | AmazonHelp |
|---|---|---|---|
| Outbound tweet volume | 106,860 (high) | 43,265 (medium) | 169,840 (very high) |
| Language consistency | English-only in sample | English-only | Mixed languages detected |
| Distinct intent coverage | 6–7 clear topics | 4–5 topics | Very broad, hard to bound |
| Brand voice consistency | Highly consistent | Consistent | Variable across regions |
| Escalation variety | Yes (account lockouts, hardware) | Limited | Yes (fraud, delivery) |
| Auto-handle candidates | Yes (how-to, software questions) | Yes | Yes |
| Data volume for this laptop | Manageable | Manageable | Borderline |

---

## Rationale for AppleSupport

### 1. Volume with manageability
106,860 outbound tweets provides a large and statistically representative
training corpus while remaining manageable on a CPU-only development machine.
The extracted brand subset is expected to be approximately 200,000–250,000 rows
(including inbound customer messages).

### 2. Language consistency
Inspection of 50 sampled customer tweets showed English-only content.
AmazonHelp, by contrast, showed Japanese messages in the first 20 examples
reviewed.  A single-language corpus avoids the need for language detection or
multilingual embeddings in the prototype.

### 3. Well-separated intent categories
The following distinct topic categories were visible in the actual data:
- iOS / macOS software update problems
- Battery drain and performance issues
- Account, Apple ID, and iCloud access failures
- Hardware and device damage
- Specific app or service failures (iMessage, Siri, CarPlay)
- How-to and feature questions

These categories have clear inclusion and exclusion criteria, making them
suitable for a hand-labelled training set.

### 4. Consistent brand voice
AppleSupport replies follow a recognisable pattern:
- Ask for the iOS version and device model
- Direct the customer to a DM
- Provide a specific troubleshooting step or knowledge base link

This consistency makes grounded reply generation more tractable: the LLM has
a stable style to imitate.

### 5. Escalation diversity
Account lockouts, hardware damage, and data loss cases provide genuine
escalation-required examples, which are necessary to evaluate the escalation
decision component.

### 6. Recognition and assessability
The brand is internationally recognisable, making it straightforward for any
reviewer to judge whether a generated reply sounds plausible.

---

## Risks and Limitations

- The dataset spans October 2017 only.  Current Apple product vocabulary,
  iOS versions, and support policies differ.  Models trained on this data will
  reflect 2017 content — this is a prototype limitation, not a project flaw.
- Some multi-lingual AppleSupport tweets may exist in markets not sampled.
  The extraction pipeline uses exact author_id matching, so this is bounded by
  what appears in the dataset.
- 106,860 outbound tweets are brand *replies* — the inbound set is larger.
  Exact counts are reported by `extract_brand.py` after extraction.

---

## Next Steps After Brand Selection

1. Run `src/data/extract_brand.py --brand AppleSupport` to produce
   `data/processed/applesupport_tweets.csv`.
2. Run `src/data/reconstruct_threads.py` to rebuild conversation threads.
3. Sample 500–600 customer tweets for intent labelling.
4. Validate the intent taxonomy against the labelled set.
