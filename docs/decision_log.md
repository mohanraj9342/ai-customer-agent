# Engineering Decision Log

This log records non-obvious technical decisions made during the project.
Each entry explains what was decided, what alternatives were considered, and
why the chosen approach was preferred.

---

## Decision 1: Process the dataset in fixed-size chunks, not fully in memory

**Decision:** Read `twcs.csv` using `pd.read_csv(..., chunksize=50_000)` so
that only one chunk (~8 MB) is in RAM at a time.

**Alternatives considered:**
- Load the full 493 MB CSV into memory with a single `pd.read_csv()` call.
- Use a database (SQLite, DuckDB) as an intermediate store.

**Reason:** The development machine has approximately 11 GB RAM but also runs
an OS, browser, and editor.  Loading 493 MB of raw CSV expands to ~1–2 GB as
a DataFrame.  Chunked reading keeps peak usage under 100 MB for inspection
tasks.  A database would add a dependency and a setup step without providing
meaningful benefit at this prototype scale.

**Trade-offs accepted:** Slightly more complex code (accumulators instead of
whole-frame operations); aggregate statistics must be computed incrementally.

---

## Decision 2: The actual row count is 2,811,774 (not ~3,002,523)

**Decision:** Accept 2,811,774 as the authoritative logical row count and
document the source of the discrepancy rather than treating it as data loss.

**Investigation result:** `wc -l` reported 3,002,524 lines (3,002,523 data
lines + 1 header).  The Python `csv` module counted 2,811,774 logical rows
with zero malformed rows.  The difference (190,749) is caused by tweets whose
`text` field contains embedded literal newlines inside quoted CSV fields.  The
`csv` module and pandas both handle these correctly.  `wc -l` counts newline
characters, not logical records, so it over-counts multi-line fields.

**Trade-offs accepted:** The inspection script uses `on_bad_lines='skip'` to
prevent encoding edge cases from aborting a long streaming pass.  A separate
diagnostic confirmed zero structurally malformed rows.

---

## Decision 3: Use two-pass extraction rather than a single filtering pass

**Decision:** Extract brand data using two passes over the CSV: Pass 1 builds
a set of tweet IDs to retain; Pass 2 emits those rows.

**Alternatives considered:**
- Single pass: emit brand outbound rows only, ignore customer inbound context.
- Load the full CSV and filter with pandas in memory.
- Extract by @mention pattern in inbound tweet text.

**Reason:** A single outbound-only pass would discard the customer messages
that triggered the brand's replies, losing the conversational context needed
for intent classification and retrieval.  @mention matching is a heuristic
that fails when customers reply without re-mentioning the brand.  The two-pass
approach uses the actual conversation link columns (`in_response_to_tweet_id`)
for exact matching.  Memory-loading the full CSV would exceed safe RAM limits.

**Trade-offs accepted:** Two sequential passes over the 493 MB file add
~3 minutes of I/O time but produce a correct, complete brand subset.

---

## Decision 4: Select AppleSupport as the target brand

**Decision:** Work with the `AppleSupport` brand (106,860 outbound tweets).

**Alternatives considered:** AmazonHelp (169,840), SpotifyCares (43,265),
Uber_Support (56,270).

**Reason:** AppleSupport provides a large English-only corpus with clearly
separable support topics, consistent brand voice, and a mix of auto-handleable
and escalation-required cases.  AmazonHelp is larger but contains multilingual
content that would require language detection.  SpotifyCares is a strong
runner-up but has narrower topic coverage (fewer natural intent classes).
Full reasoning is in `docs/brand_selection.md`.

**Trade-offs accepted:** The 2017 dataset reflects iOS 11-era content; current
Apple product vocabulary differs.  This is a prototype constraint, not a flaw.

---

## Decision 5: Use `in_response_to_tweet_id` as the primary conversation edge

**Decision:** Reconstruct threads by following `in_response_to_tweet_id` links
(child → parent) rather than `response_tweet_id` links (parent → children).

**Alternatives considered:**
- Use `response_tweet_id` to build parent-to-children maps.
- Use both columns and merge results.

**Reason:** `in_response_to_tweet_id` is set on 71.7% of all rows and
unambiguously identifies the direct parent of each tweet.
`response_tweet_id` contains comma-separated values (one parent can generate
multiple replies), making it harder to use as a primary edge.  The BFS walk
from root tweets uses `response_tweet_id` only as a complementary signal for
detecting branches.

**Trade-offs accepted:** Some multi-branch conversations are flattened into a
single time-ordered sequence, which is a simplification.

---

## Decision 6: Keep the intent taxonomy small (6 specific intents + catch-all)

**Decision:** Define 7 intent labels for the initial labelling phase.

**Alternatives considered:** Finer-grained taxonomy (10–15 labels); coarser
(3–4 labels).

**Reason:** A taxonomy with more than 10 labels at this stage creates three
problems: (a) hard to hand-label consistently without a trained team,
(b) some classes will have too few examples for a reliable classifier,
(c) the prototype evaluation becomes unwieldy.  A taxonomy with fewer than
5 labels risks being too coarse to be useful for routing decisions.  The 6+1
taxonomy covers the observed topic distribution without forcing sparse classes.

**Trade-offs accepted:** Edge cases between adjacent intents (e.g. a battery
issue caused by an update) will need labelling guidelines; the "other" catch-all
absorbs the ambiguous remainder.

---

## Decision 7: Keep the core pipeline interface-independent

**Decision:** All pipeline modules (`extract_brand.py`, `reconstruct_threads.py`,
`intent_loader.py`) expose pure Python functions with no web framework
dependencies.

**Alternatives considered:** Build the pipeline as Flask routes from the start;
use a Jupyter notebook as the primary interface.

**Reason:** Tying business logic to a web framework early makes unit testing
harder and reduces portability.  By keeping functions pure (input → output
with no HTTP context), the same code is testable by pytest, callable from
CLI scripts, and composable into a future Flask route or async handler without
modification.

**Trade-offs accepted:** An extra integration layer will be needed when the
web interface is eventually added, but this is a one-time cost.

---

## Decision 8: Store only small, generated summary files in the repository

**Decision:** Commit only `docs/dataset_inspection.json` and `docs/top_brands.csv`
(total: ~2.4 KB).  All raw data, processed data, and model artifacts are
gitignored.

**Alternatives considered:** Store a small sample CSV; store the full processed
brand subset; use Git LFS.

**Reason:** Raw data must not be committed for privacy, size, and licence
reasons.  Processed data can always be regenerated from the committed code and
the dataset.  Git LFS adds a dependency and cost.  The JSON/CSV summaries are
purely informational and contain no personal data.

**Trade-offs accepted:** A reviewer must download the dataset themselves before
running the extraction pipeline.  The README and setup instructions document
exactly where to obtain it.

---

## Decision 9: Normalise text lightly — preserve @mentions, URLs, and emoji

**Decision:** Apply only HTML entity decoding and whitespace collapsing.  Do
not remove @mentions, strip URLs, lowercase text, or remove punctuation at the
data-layer normalisation step.

**Alternatives considered:** Aggressive cleaning (remove all special tokens);
no cleaning at all.

**Reason:** @mentions are used as conversation-link heuristics.  URLs may
contain domain signals (e.g. support.apple.com).  Emoji carry sentiment
information useful for escalation detection.  Lowercase conversion and
stopword removal are classifier-specific choices that should be made in the
classification component, not at the data layer.  This way the same extracted
CSV can serve multiple downstream uses without re-extraction.

**Trade-offs accepted:** The extracted CSV is slightly larger than a
fully-cleaned version.

---

## Decision 10: Defer web hosting and LLM API integration

**Decision:** No deployment configuration, no API key wiring, and no LLM
calls are added until those phases are explicitly approved.

**Alternatives considered:** Set up Render now; wire the Gemini API early to
test end-to-end.

**Reason:** Premature hosting adds deployment risk (accidental exposure of
API keys), cost risk (unintended API calls during development), and obscures
whether the core pipeline works independently.  Each phase should be fully
tested locally before adding external dependencies.

**Trade-offs accepted:** The project cannot be demoed via URL until the hosting
phase is complete, but local CLI execution provides equivalent verification.

---

## Decision 11: Graph-based conversation reconstruction and explicit link denominators

**Decision:** Conversation threads are reconstructed using directed graph
relationships (`in_response_to_tweet_id` and `response_tweet_id`) with
connected-component analysis, deterministic timestamp + tweet_id ordering,
cycle detection, and explicit metric denominators. Multiple-exchange threads
are strictly defined as >= 4 messages with >= 2 customer and >= 2 brand turns.

**Alternatives considered:** Flat grouping by author or arbitrary time
windows; treating every 2-message customer-brand pair as multi-exchange;
reporting link coverage percentages without explicit denominators.

**Reason:** In Twitter customer service datasets, parent tweets may be missing
(e.g., deleted or outside extract window) or branch into multiple responses.
Earlier naive metrics reported 100% parent link coverage by dividing against
the entire corpus rather than isolating messages with non-null parent IDs.
Similarly, labeling single question-and-answer pairs as "multi-exchange"
masks true conversation depth. Explicit denominators and strict thresholds
ensure reliable quality evaluation for downstream retrieval and generation.

**Trade-offs accepted:** Unconnected single messages and broken parent chains
are preserved as flagged components rather than silently discarded or
stitched heuristically.

---

## Decision 12: Demarcate extraction-dependent final-brand metric and audit link reconciliations

**Decision:** The metric previously labeled as "100% complete threads" is
aliased to `ends_with_brand_reply` and explicitly flagged in metadata as
`extraction_dependent_final_brand: true`. It must not be interpreted as
customer resolution or dialogue closure. Response-link metrics explicitly
reconcile `total_individual_response_ids = valid + missing`.

**Alternatives considered:** Maintaining the unqualified label "complete";
omitting threads that have unextracted reply IDs; attempting heuristic
resolution classification based on text.

**Reason:** In Phase 3, customer tweets were collected only if AppleSupport
replied to them. By construction, every leaf in the extracted conversation
graph terminates at a brand message. An audit revealed that 8,848 final brand
messages (10.8% of threads) have outgoing `response_tweet_id` links pointing
to 9,717 subsequent customer tweets on Twitter that were never extracted
because AppleSupport did not reply again. Conflating "terminates at brand reply"
with "conversation is complete" is structurally misleading. Explicitly
flagging extraction bias ensures downstream intent classification and
retrieval models do not assume dialogue termination.

**Trade-offs accepted:** Retaining the legacy `is_complete` key alongside
`ends_with_brand_reply` maintains backward compatibility with existing
callers while clarifying semantic limitations.

---

## Decision 13: Customer-intent classification unit, empirical taxonomy (v2.0), and conservative heuristic labeling rules

**Decision:** Select the earliest customer message (`inbound == 'True'`) per
reconstructed thread as the primary classification unit for front-line intent
routing. Adopt an empirically grounded 11-category domain taxonomy (v2.0)
defined in `src/classification/intents.yaml`. Implement deterministic lexical
rules with multi-category conflict detection routing overlapping matches to
`needs_review` and non-matches to `unknown_other`. Prohibit inspecting agent
response text or thread-terminal metadata flags (`ends_with_brand_reply`,
`is_complete`) during intent classification to eliminate data leakage.

**Alternatives considered:**
1. *Classifying every message across multi-turn threads:* Deferred to turn-level
   dialogue act modeling; front-line customer agent triage operates on initial
   inbound contact.
2. *Forcing single labels via arbitrary rule precedence:* Rejected because
   forcing a category when multiple domains match (e.g., iOS update causing
   battery drain) introduces label noise into training pipelines.
3. *Using brand response text or thread-completion status to infer intent:*
   Strictly rejected as target leakage, which would artificially inflate model
   accuracy on historical data while failing on live incoming inquiries.

**Reason:** In front-line customer support, automated agents must determine the
customer's issue at the point of initial intake. The empirical AppleSupport
corpus demonstrates that customer inquiries cluster into distinct technical
concerns (e.g., software updates represent 27.9% of inquiries, distinct from
physical battery degradation at 4.15%). Preliminary heuristic labels bootstrap
active learning and annotation workflows but are not ground truth. Flagging
competing matches as `needs_review` and brief greetings as `unknown_other`
preserves data integrity and prevents misleading downstream benchmark evaluations.

**Trade-offs accepted:** A large share of candidate inquiries (45.88%) fall
into `unknown_other` due to Twitter conversational brevity (e.g., "@AppleSupport
DM me", "help please"). These instances require conversational elicitation
rather than speculative forced classification.

---

## Decision 14: Deterministic stratified audit sampling, 4-tier review prioritization, and taxonomy retention strategy

**Decision:** Construct a deterministic multi-slice stratified review sample
(1,874 records) from the 82,101 Phase 5 candidates using a fixed random seed
(`seed=42`) and canonical pre-sorting by `tweet_id`. Implement a 4-tier human
review prioritization schema (`critical`: 550 rows, `high`: 793 rows,
`medium`: 286 rows, `normal`: 245 rows) to concentrate annotation resources on
rule overlaps and ambiguous high-volume classes. Retain Taxonomy Version 2.0
unchanged while recommending targeted lexical rule refinements and human
verification prior to supervised classifier training.

**Alternatives considered:**
1. *Adding niche taxonomy categories (e.g., retail store appointments, Apple Watch):*
   Rejected because each cluster accounts for $<1.0\%$ of customer inquiries;
   adding them prematurely would fragment training data into severely imbalanced
   micro-classes.
2. *Forcing resolution on `needs_review` multi-category conflicts via heuristic precedence:*
   Rejected because over 31% of conflicts represent genuine multi-intent
   inquiries (e.g., OS update attribution with battery drain symptoms) where
   ground truth requires human adjudication.
3. *Simple unstratified random sampling:* Rejected because rare classes such as
   `order_shipping` (0.25% corpus share) would yield fewer than 5 instances in a
   random 1,000-sample pool, preventing statistically sound auditing.

**Reason:** Quantitative inspection of the 82,101 candidates confirmed that
`unknown_other` (45.88%) reflects natural conversational intake characteristics
(greetings, DM requests, image-only tweets, vague complaints) rather than
systematic taxonomy omission. The core 11 domains in Taxonomy v2.0 correctly
map to support routing workflows. Deterministic multi-slice stratification
ensures reproducible review datasets with verified representation across all
intents, confidence tiers, and linguistic edge cases.

**Trade-offs accepted:** The review sample (`apple_support_intent_review_sample.csv`)
is intentionally enriched for edge cases, multi-category conflicts, and rare
domains. Consequently, it represents an active learning / quality audit cohort
rather than an unstratified natural distribution test benchmark.

---

## Decision 15: Human-annotation pilot dataset (158 records), empirical keyword collision audit, and standardized annotation guide

**Decision:** Construct a deterministic 158-record pilot annotation dataset
(`data/processed/apple_support/apple_support_intent_pilot_sample.csv`) using a
fixed random seed (`seed=42`) and pre-sorting by `tweet_id`. Establish
standardized operational procedures in `docs/apple_support_intent_annotation_guide.md`
including hierarchical multi-intent precedence rules (Safety > Financial/Security >
Actionable Symptom > Attribution). Document empirical collision findings for
`software_update` (2,142 app download collisions, 484 billing method collisions)
and mandate pilot calibration (target $\kappa \ge 0.85$) before full-scale human review.

**Alternatives considered:**
1. *Proceeding directly to full 1,874-sample annotation without a pilot:*
   Rejected because calibrating annotators on edge cases, smart quote nuances,
   and multi-intent resolutions is essential to avoid systematic label noise.
2. *Automated heuristic reassignment of "update" collisions:* Rejected because
   manually overriding rules with further regex heuristics risks secondary
   classification noise; human adjudication on prioritized samples provides
   authentic ground truth.

**Reason:** The quality audit established that standalone uses of "update"
collide with App Store app downloads (9.34% of `software_update`) and payment
method updates (2.11%). A calibrated pilot dataset spanning all 11 domains,
conflict pairs, and edge cases enables annotator alignment and verifies
guideline clarity prior to annotating the full 1,874-record review sample.

**Trade-offs accepted:** The pilot dataset is intentionally enriched with
high-risk edge cases and ambiguous collisions; it is designed for annotator
calibration and inter-rater agreement measurement rather than natural test
distribution evaluation.

---

## Decision 16: Human pilot annotation completion (158 records), adjudication error analysis, and operational precedence validation

**Decision:** Complete manual adjudication of all 158 records in the pilot dataset
(`data/processed/apple_support/apple_support_intent_pilot_sample.csv`) following
Taxonomy Version 2.0 and the standardized guidelines in
`docs/apple_support_intent_annotation_guide.md`. Verify 76 records (48.10%),
correct 79 records (50.00%), and preserve 3 records (1.90%) as `flagged_ambiguous`
(`needs_review`) for supervisory escalation. Populate 100% of review metadata fields
(`verified_intent`, `verification_status`, `verified_by`, `verification_date`, `notes`)
with zero missing values and zero data leakage.

**Alternatives considered:**
1. *Forcing resolution on all 158 records into 11 concrete classes without allowing `flagged_ambiguous`:*
   Rejected because genuine microblogging multi-intent cases (e.g., physical hardware repair
   damage combined with a £3,250 refund dispute, or co-equal simultaneous battery death and
   complete cellular/network failure) cannot be resolved without supervisor adjudication
   or conversational clarification; forcing a single class would introduce arbitrary label noise.
2. *Relying on brand replies or conversation turns for intent resolution:*
   Strictly rejected because evaluating conversational outcomes or brand agent actions introduces
   data leakage; front-line triage models must operate exclusively on the customer's initial
   inbound formulation.
3. *Automated post-hoc heuristic correction of preliminary labels:* Rejected because human
   verification provides rigorous, audited ground-truth labels needed to benchmark both
   heuristic rules and subsequent supervised models.

**Reason:** The pilot adjudication confirmed that hierarchical precedence rules
(Safety > Security/Financial > Actionable Symptom > Historical Attribution) effectively
resolve 86.96% (20/23) of multi-category rule conflicts. The empirical agreement rate of
49.37% demonstrates that the preliminary heuristic rules struggle on collision boundaries
(e.g., App Store app downloads and billing method updates labeled as `software_update`,
actionable post-update battery drain assigned to `software_update`, and actionable native app
failures falling into `unknown_other`). Completing this audited pilot proves guideline
soundness, measures baseline error modes, and establishes a validated protocol for
scaling human annotation across the full 1,874-record review dataset.

**Trade-offs accepted:** Due to heavy oversampling of edge cases and conflict strata in the
pilot design, the observed preliminary rule agreement rate (49.37%) is substantially lower
than natural distribution accuracy. This is intentional and necessary for stress-testing
guidelines against worst-case boundary ambiguities.

---

## Decision 17: Golden evaluation set construction, slice difficulty taxonomy, and training isolation protocol

**Decision:** Establish the 158-record human-verified dataset
(`data/processed/apple_support/apple_support_intent_golden_set.csv`) as the
canonical Golden Evaluation Set for the AppleSupport intent classifier.
Annotate each instance with an operational difficulty slice (`representative`,
`attribution_vs_symptom`, `rule_conflict`, `fallback_recovery`,
`boundary_disambiguation`, `short_noisy`, `multilingual`, `ambiguous_multi_intent`).
Explicitly preserve the 3 genuinely ambiguous multi-intent cases as `needs_review`
with `verification_status: flagged_ambiguous`. Mandate strict training isolation
guaranteeing $S_{\text{train}} \cap S_{\text{golden}} = \emptyset$ across all future model
training candidate pools via programmatic verification.

**Alternatives considered:**
1. *Expanding the golden set beyond 158 records using automated rule filtering:*
   Rejected because evaluation benchmarks must consist strictly of human-verified
   ground truth; synthetic or heuristic expansion introduces label noise and degrades
   benchmark reliability.
2. *Forcing a single dominant class on the 3 `needs_review` cases in the golden set:*
   Rejected because artificial forced single-label assignment misrepresents true
   conversational ambiguity; keeping them explicitly flagged provides ground truth for
   evaluating classifier confidence and escalation logic.
3. *Ad-hoc train/test splitting without a permanent golden registry:* Rejected because
   dynamic random splits risk evaluation set leakage into training pools and prevent
   reproducible longitudinal benchmarking across model iterations.

**Reason:** A high-quality evaluation set must be independent, human-verified,
representative of operational challenges, and strictly quarantined from training data.
The 158-record golden set provides audited representation across all 11 taxonomy domains
and fallback classes, incorporates stress-tested boundary and conflict cases, and provides
fine-grained difficulty slices for diagnostic error analysis. Programmatic isolation
guarantees zero data leakage for subsequent model training.

**Trade-offs accepted:** At 158 records, statistical confidence intervals for rare
classes (e.g., `account_access` with 5 instances) are wider than on high-frequency classes.
However, 100% human verification and zero data leakage provide far higher evaluation
validity than a larger, noisy heuristic set.

---

## Decision 18: Intent classifier benchmarking architecture, class-weight balancing, and diagnostic evaluation on the golden set

**Decision:** Train and benchmark three intent classification models: a Majority-Class Baseline
(`DummyClassifier`), TF-IDF + Logistic Regression (multinomial L2, balanced class weighting),
and TF-IDF + Linear SVM (`LinearSVC`, balanced class weighting) using sublinear TF-IDF
(10,000 max features, unigrams and bigrams). Enforce strict training isolation
($S_{\text{train}} \cap S_{\text{golden}} = \emptyset$) on 76,071 clean training candidates
(excluding 5,872 unverified heuristic conflicts). Benchmark primary closed-world performance
on the 155 resolved golden evaluation records and conduct diagnostic confidence/margin
analysis on the 3 ambiguous `needs_review` cases. Select TF-IDF + Logistic Regression as
the primary model artifact.

**Alternatives considered:**
1. *Unweighted empirical loss optimization:* Rejected because extreme candidate class
   imbalance (49.5% `unknown_other` vs 0.26% `order_shipping`) causes unweighted models
   to collapse recall on critical low-frequency operational intents like billing and shipping.
2. *Deep neural / generative architectures at baseline phase:* Deferred in favor of
   establishing an interpretable, deterministic linear baseline first, establishing clear
   performance floors before adding model complexity.
3. *Ad-hoc random test splitting from candidates:* Rejected because evaluating against
   noisy heuristic labels corrupts metric integrity; benchmarking strictly against the
   human-verified golden set ensures true ground-truth validity.

**Reason:** TF-IDF + Logistic Regression with balanced class weighting achieves 59.35% accuracy,
63.19% Macro-F1, and 59.64% Weighted-F1 on the golden set, decisively beating the Majority
Baseline (9.03% accuracy, 1.51% Macro-F1) and outperforming Linear SVM (54.84% accuracy,
59.61% Macro-F1). The smooth probabilistic outputs provide calibrated posteriors, enabling
principled confidence thresholding ($P < 0.60$ or margin $< 0.20$) to trigger human escalation
for genuinely ambiguous inquiries (e.g. multi-symptom inquiries).

**Trade-offs accepted:** Surface n-gram models struggle with semantic attribution
(distinguishing root symptoms from casual temporal mentions like "after update" in the
`attribution_vs_symptom` slice, 54.17% accuracy) and subtle boundary disambiguation
(0.0% accuracy on 7 boundary cases). However, robust performance on representative
production cases (75.71% accuracy) and core customer care categories (`connectivity_network`
85.71% F1, `order_shipping` 82.35% F1, `billing_payment` 78.26% F1, `account_access` 76.92% F1)
validates this model as an effective, auditable baseline for front-line intent triage.

---

## Decision 19: DistilRoBERTa fine-tuning on remote Colab GPU, class-weighted cross-entropy loss, golden benchmark comparison, and hybrid cascade strategy

**Decision:** Fine-tune `distilroberta-base` (~82M parameters) strictly on remote GPU hardware
(NVIDIA Tesla T4 in Google Colab) using class-weighted cross-entropy loss, 128 max sequence length,
batch size 32, AdamW with initial learning rate 3e-5 and fp16 mixed precision across 3 epochs.
Prohibit local CPU training to preserve laptop responsiveness and avoid 5+ hours of thermal
throttling. Enforce strict mathematical training isolation ($S_{\text{train}} \cap S_{\text{golden}} = \emptyset$)
over 76,071 candidate records, benchmark against the 155 closed-world golden records, and adopt a
two-tier hybrid cascade deployment recommendation.

**Alternatives considered:**
1. *Local CPU transformer fine-tuning:* Prohibited by operational guidelines. The local laptop's
   AMD Ryzen 3 5300U processor lacks CUDA/Tensor Cores and would have required 4–6+ hours of 100% CPU
   saturation with high OOM risks, whereas the remote Tesla T4 completed all 3 epochs in 11m 21s.
2. *Full replacement of Phase 9 baseline with DistilRoBERTa:* Rejected based on empirical golden
   set benchmarking. While DistilRoBERTa achieves superior F1 on semantic domains (`device_hardware`
   82.35% vs 70.59%, `app_or_service_issue` 55.17% vs 46.15%), Logistic Regression achieves higher
   overall Macro-F1 (63.19% vs 59.29%) and 125x lower inference latency (<0.2 ms vs 25 ms).
3. *Unweighted transformer training:* Rejected due to severe class imbalance; unweighted loss
   penalizes low-support business categories (`order_shipping`, `billing_payment`).

**Reason:** DistilRoBERTa demonstrates genuine semantic understanding where n-gram models fail,
gaining +11.76% F1 on descriptive hardware damage phrasing and +9.02% F1 on service/app issues.
However, because training candidates were heuristically labeled, the deep neural model partially
memorized heuristic nuances. The optimal production strategy is a hybrid cascade: TF-IDF + Logistic
Regression performs sub-millisecond front-line triage for high-confidence structured queries, while
low-confidence or hardware/app queries are escalated to DistilRoBERTa.

**Trade-offs accepted:** Maintaining two model architectures adds deployment footprint, but delivers
the best of both worlds: ultra-low latency (<0.2 ms) and high overall F1 for common categories,
combined with deep contextual representation (82.35% hardware F1) for complex customer complaints.

---

## Decision 20: Historical response retrieval & grounding architecture, dense bi-encoder indexing (all-MiniLM-L6-v2), and strict Golden Set quarantine

**Decision:** Build a semantic response retrieval and grounding engine using dense vector embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalized) indexing 81,943 grounded customer-inquiry $\leftrightarrow$ brand-reply interaction pairs extracted from reconstructed AppleSupport threads. Enforce strict mathematical quarantine of all 158 Phase 8 Golden Evaluation Set records ($S_{\text{retrieval}} \cap S_{\text{golden}} = \emptyset$). Implement sub-10ms exact cosine similarity retrieval via normalized BLAS inner products ($S = X \cdot q^T$) on local CPU hardware without deploying generative LLMs or external vector database infrastructure.

**Alternatives considered:**
1. *Generative LLM response synthesis:* Strictly rejected for Phase 11. Generative language models present unacceptable risks of hallucination, inaccurate troubleshooting advice, and ungrounded policy commitments in regulated customer support contexts. Grounded retrieval guarantees that all proposed responses represent verbatim historical actions taken by verified AppleSupport agents.
2. *TF-IDF / BM25 lexical sparse retrieval:* Rejected as the primary retrieval mechanism because keyword-based matching suffers from vocabulary mismatch (e.g. "battery dies in 30 mins" vs "battery drains rapidly", or "screen shattered" vs "cracked glass"). Dense bi-encoders capture contextual semantic equivalence across varied colloquial customer phrasings.
3. *Cloud vector database deployment (e.g. Pinecone, Milvus, Qdrant):* Deferred. At 81,943 vectors of dimension 384, the entire index occupies only ~126 MB in RAM. A vectorized in-memory NumPy matrix product executes in < 4 milliseconds per query, eliminating external network latency, cloud costs, and operational dependency burden.

**Reason:** In high-stakes enterprise customer support, grounding and safety are paramount. Historical response retrieval allows the automated assistant to suggest proven, agent-validated troubleshooting workflows and official Apple Knowledge Base URLs. Quarantining the 158 Golden Set records preserves the pristine evaluation integrity of future end-to-end response generation and triage benchmarks.

**Trade-offs accepted:** 
1. The dense index encodes single-turn initial inquiries; conversational context beyond the first exchange is not indexed.
2. `all-MiniLM-L6-v2` is an English-dominant bi-encoder; non-English microblog queries (e.g., Spanish inquiries) experience lower similarity scores and require language detection or multilingual bi-encoders in subsequent iterations.

---

## Decision 21: Groq API LLM selection — qwen/qwen3.8-27b as primary grounded response model

**Decision:** Select `qwen/qwen3.8-27b` (Alibaba Cloud, 27B parameters, 131k-token context window) as the primary Groq-hosted LLM for Phase 12 grounded agent response generation. Groq model availability was validated programmatically against the live Groq `/v1/models` endpoint before selection.

**Alternatives considered:**
1. *`llama-3.3-70b-versatile` (Meta):* Originally planned primary model. Confirmed **unavailable** on the project's Groq API key at time of Phase 12 implementation. Would have caused `404 model not found` errors at runtime.
2. *`groq/compound` (Groq native):* Available. However, emits verbose chain-of-thought reasoning preambles even for simple instruction-following tasks (observed in live probing), making strict JSON schema output less reliable without complex post-processing.
3. *`openai/gpt-oss-20b` (OpenAI OSS):* Available. However, returned empty content on the instruction-following smoke test, suggesting inconsistent tool-call mode vs. chat-completion mode on this API key configuration.
4. *`openai/gpt-oss-120b` (OpenAI OSS):* Available but 120B parameters exceeds what is necessary for structured JSON generation in a grounded-retrieval context with a pre-built system prompt and explicit schema. Higher cost / token rate not justified.

**Reason:** `qwen/qwen3.8-27b` was the only model that (a) passed the live Groq endpoint availability check, (b) responded correctly and precisely to instruction-following prompts without unsolicited reasoning preambles, and (c) supports the 131k-token context window required to embed multi-case retrieved evidence in the prompt. Its instruction-following quality is well-established on JSON-structured outputs.

**Trade-offs accepted:**
1. Qwen models are optimized for multilingual use; occasional tokenization idiosyncrasies may arise on English-only Apple support jargon (e.g., model IDs, iOS version strings). Mitigated by explicit JSON schema enforcement and output validation in `groq_client.py`.
2. If Groq rotates model availability again, `GROQ_MODEL` must be re-validated. The `load_groq_config()` function raises a clear `ValueError` if the model env var is unset, and the `groq_client.py` raises `GroqGenerationError` if the API rejects the model name, so failures are surfaced immediately rather than silently producing wrong outputs.

---

## Decision 22: Phase 13 Agent Evaluation & Reviewer Layer architecture — deterministic-first auditing, hybrid LLM grading, and strict response immutability

**Decision:** Implement an independent, post-generation evaluation layer (`AgentReviewer` in `src/evaluation/agent_reviewer.py`) that evaluates Phase 12 `AgentResponse` outputs along 7 core dimensions (`grounding_support`, `hallucination_risk`, `escalation_correctness`, `intent_consistency`, `response_relevance`, `response_completeness`, and `professional_quality`). Adopt a deterministic-first architecture: critical physical safety hazards, legal threats, unauthorized refund guarantees, and fabricated account actions are audited deterministically with hard-gated `FAIL` decisions. Support an optional hybrid LLM mode utilizing the existing Groq client configuration (`qwen/qwen3.8-27b`) for semantic nuance and tone assessment. Guarantee strict non-mutation of primary agent outputs and absolute quarantine of the Phase 8 Golden Evaluation Set.

**Alternatives considered:**
1. *In-pipeline response alteration / self-correction loop:* Rejected. Allowing the reviewer to silently edit or regenerate responses complicates auditability, obscures primary agent failure modes, and creates non-deterministic feedback loops. Downstream consumer systems must receive the unadulterated `AgentResponse` alongside the independent `EvaluationResult`.
2. *Pure LLM-as-a-Judge evaluation:* Rejected. LLM judges are prone to non-determinism, instruction-following drift, and latency/cost penalties on high-throughput microblog pipelines. Hard safety violations (such as missed physical hazards or unauthorized refund commitments) must never rely solely on probabilistic LLM judgment.
3. *Pure deterministic rule evaluation:* Insufficient on its own for complex semantic assessment. While deterministic rules excel at detecting hard policy violations, regex cannot reliably grade conversational tone, subtle relevance, or implicit technical advice. The hybrid architecture provides the speed and certainty of deterministic safety with the semantic flexibility of LLM judging.

**Reason:** In enterprise customer care, safety, grounding, and auditability must be verifiable independently of the generative model that produced the draft. Decoupling the reviewer ensures that model changes or prompt iterations in Phase 12 can be objectively benchmarked against a consistent, unchanging evaluation standard.

**Trade-offs accepted:**
1. Lexical grounding checks rely on token overlap heuristics and may underestimate grounding when the LLM rephrases historical evidence into novel synonyms. Mitigated in hybrid mode where the LLM reviewer evaluates semantic equivalence.
2. The reviewer introduces an evaluation step; running in pure deterministic mode adds < 2ms latency, while hybrid LLM mode requires an additional Groq completion call (~500ms). Deterministic mode is therefore recommended for inline routing, reserving hybrid mode for offline quality audits or asynchronous human-in-the-loop review.

---

## Decision 23: Phase 14 End-to-End Batch Evaluation methodology, benchmark design, dual-mode execution, and strict Golden Set quarantine

**Decision:** Construct a controlled 100-case evaluation benchmark dataset (`apple_support_e2e_benchmark.csv`) sampled across 5 operational cohorts (`routine`, `ambiguous_low_confidence`, `escalation_sensitive`, `vague_short`, and `weak_retrieval`). Explicitly disclose that this benchmark is an engineered coverage stress-test rather than a statistically representative sample of the full AppleSupport population. Enforce strict mathematical quarantine of all 158 Phase 8 Golden Set records ($S_{\text{benchmark}} \cap S_{\text{golden}} = \emptyset$). Support dual evaluation modes: `--mode deterministic` utilizing an explicit offline replay generator (`offline-deterministic-replay`) for fast, zero-cost, reproducible regression benchmarking, and `--mode live` for real Groq API execution. Measure all latencies empirically via `time.perf_counter()`. Strictly prohibit tuning classifier thresholds, prompts, or reviewer rules against this benchmark.

**Alternatives considered:**
1. *Benchmarking on the Phase 8 Golden Evaluation Set:* Strictly rejected. Reusing the Golden Evaluation Set for end-to-end multi-phase orchestration risks data leakage, compromises its role as a pristine held-out intent benchmark, and conflates classification evaluation with response generation.
2. *Full live Groq benchmarking for all 100 cases on every run:* Rejected due to GroqCloud API rate limits (observed HTTP 429 backoff) and unnecessary token costs. Providing an explicit deterministic replay mode enables sub-second regression testing in CI environments while reserving live API runs for targeted validation.
3. *Simulating live generation in deterministic mode:* Rejected. Deterministic mode must explicitly attribute draft responses to offline replay from historical evidence rather than masquerading as live LLM output.

**Reason:** Rigorous machine learning engineering requires end-to-end validation of integrated pipelines. Measuring real per-component latencies (Classifier: 1.19ms, Retriever: 88.55ms, Orchestration: 89.52ms, Reviewer: 0.27ms) demonstrates that the entire pipeline operates well within interactive SLAs (< 200ms total deterministic latency), while capturing policy precision (94.1%) and zero critical safety failures across edge-case cohorts.

**Trade-offs accepted:**
1. The 100-case benchmark over-indexes on difficult edge cases (45% non-routine cohorts), which depresses overall escalation recall (64.0%) compared to what would be observed on routine macro traffic. This trade-off is accepted to ensure edge-case policy robustness.
2. The benchmark routine cohort uses programmatically derived heuristic labels rather than double-blind human annotations, which is explicitly disclosed in the label typology.

---

## Decision 24: Prevention of Benchmark Self-Retrieval via Reusable Per-Query Leave-One-Out Candidate Exclusion

**Decision:** Implement a generic, reusable candidate exclusion mechanism in `HistoricalResponseRetriever.retrieve()` (`exclude_customer_tweet_ids`, `exclude_thread_ids`, `exclude_exact_customer_texts`) and forward these parameters through `GroundedSupportAgent.process_message()`. Apply this mechanism per-query in `EndToEndBatchEvaluator` so that during Phase 14 benchmarking, each query excludes its own customer tweet ID, thread ID, and normalized customer text from retrieval candidates. This eliminates artificial cosine similarity 1.0000 self-matches without globally mutating or rebuilding the 81,943-record retrieval corpus and preserves pristine quarantine of the Phase 8 Golden Evaluation Set ($S_{\text{retrieval}} \cap S_{\text{golden}} = \emptyset$).

**Alternatives considered:**
1. *Permanently removing all 100 benchmark cases from the Phase 11 retrieval index:* Rejected. Globally deleting benchmark records from the corpus would mutate the baseline retrieval index, reduce index coverage for queries whose true historical context is unrelated, and introduce index-level configuration drift between evaluation and production.
2. *Hard-coding exclusion logic inside the benchmark script only:* Rejected. Hard-coding filtering outside the retriever obscures retrieval boundaries, prevents reuse across other evaluation suites, and fails to protect the downstream `GroundedSupportAgent` and `AgentReviewer` pipeline stages.

**Reason:** In leave-one-out evaluation of retrieval-augmented generation systems, evaluating an in-corpus query against an index containing that exact historical query record results in trivial self-retrieval (measured top-1 median similarity 1.0000). By dynamically excluding only the active query's own records at retrieval time, the retriever is forced to locate distinct, semantically similar historical cases (corrected top-1 median similarity 0.7976, mean 0.7841), providing a realistic assessment of semantic retrieval and grounding quality.

**Trade-offs accepted:**
1. Dynamically filtering candidates in the ranking loop incurs negligible CPU overhead (< 0.1ms per query) during candidate iteration, but completely avoids re-encoding or maintaining separate retrieval indices.
2. Two additional benchmark cases in the weak retrieval cohort correctly triggered `weak_retrieval_grounding` escalation (similarity $< 0.50$), shifting the escalation confusion matrix from 30 TP / 20 FN to 32 TP / 18 FN and improving escalation F1 from 0.7317 to 0.7619 without any policy threshold modification.

---

## Decision 25: Production API Architecture — FastAPI with Lifespan Pre-Warming, Render Blueprint, and Vercel CORS Integration

**Decision:** Encapsulate the Phase 10–13 agent pipeline inside a modern ASGI web application using **FastAPI** and **Uvicorn**. Implement asynchronous `lifespan` pre-warming to load the intent classifier and 81,943 dense retrieval embeddings into memory upon server startup, guaranteeing sub-200ms user request latencies. Configure declarative Infrastructure-as-Code deployment for **Render** via `render.yaml` and `Procfile`. Support cross-origin consumption from a future **Vercel** frontend via wildcard regex CORS configuration (`^https://.*\.vercel\.app$`). Enforce strict Pydantic v2 input validation, RFC-compliant error envelopes, and zero server-side secret leakage in API outputs.

**Alternatives considered:**
1. *Flask / WSGI:* Rejected. Lacks native async request handling, modern OpenAPI schema generation, and requires separate serialization/validation packages (marshmallow/pydantic) that increase architectural complexity.
2. *Full-stack Next.js monolith with in-process Python binding:* Rejected. Python ML dependencies (PyTorch, sentence-transformers, scikit-learn, joblib) cannot run natively on Vercel's serverless edge environment due to bundle size limits (> 250 MB). Decoupling Render (Python AI compute) from Vercel (frontend UI) maintains clean separation of concerns and cost efficiency.
3. *On-demand pipeline initialization per request:* Rejected. Cold-loading the 81,943 embeddings on the first HTTP request adds 2–3 seconds of latency; pre-warming during FastAPI `lifespan` eliminates cold-start penalties for incoming users.

**Reason:** FastAPI provides high throughput, automatic interactive documentation (`/docs`), and robust Pydantic validation while maintaining compatibility with Python 3.14. Wildcard regex CORS handling ensures that Vercel preview environments generated across Git branches function immediately without manual DNS or CORS re-configuration.

**Trade-offs accepted:**
1. Pre-warming the retrieval corpus in memory consumes ~800MB to 1.2GB RAM during container execution. Render's Standard plan comfortably provides 2GB RAM to satisfy this requirement.
2. The initial server boot takes ~5 seconds to load models into memory before `lifespan` yields, but subsequent HTTP requests process in real-time.
