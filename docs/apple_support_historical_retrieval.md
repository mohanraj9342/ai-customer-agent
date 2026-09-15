# Phase 11: Historical Response Retrieval & Grounding Engineering Report

**Pipeline Phase:** Phase 11  
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters, 384 dimensions)  
**Corpus Target:** Reconstructed AppleSupport historical customer-brand interactions  
**Dataset Isolation:** Guaranteed $S_{\text{retrieval}} \cap S_{\text{golden}} = \emptyset$ (158 Golden Set records strictly quarantined)  
**Retrieval Corpus Size:** 81,943 grounded pairs  

---

## 1. Executive Summary

Phase 11 implements a high-performance, deterministic semantic response retrieval and grounding system for AppleSupport customer service inquiries. Rather than generating ungrounded responses via large language models—which carry substantial risks of hallucination, factual drift, and policy violations in customer care—this system retrieves verified historical responses provided by human AppleSupport agents to semantically similar customer inquiries.

The retrieval engine maps incoming inquiries into a 384-dimensional dense semantic vector space using `all-MiniLM-L6-v2`, executing exact cosine similarity ranking via normalized inner products across **81,943 historical customer-brand interaction pairs**. Sub-10ms retrieval latency is achieved on standard multi-core CPU hardware without requiring external vector database infrastructure.

---

## 2. Reconstructed Corpus Extraction & Boundary Fidelity

### 2.1 Extraction Protocol
The retrieval corpus was constructed from the 82,105 conversation threads reconstructed in Phase 4 (`data/processed/apple_support/apple_support_threads.jsonl`):

1. **Initial Customer Boundary:** The chronologically earliest non-empty customer inquiry (`inbound == True`) is extracted. Subsequent customer turns and conversational evolutions are strictly excluded to preserve initial triage boundary fidelity.
2. **First Grounded Brand Reply:** The chronologically first brand reply (`author_id == "AppleSupport"` or `inbound == False`) directly addressing the customer inquiry is paired with the message.
3. **Traceability:** Every record retains its source `thread_id`, `customer_tweet_id`, `customer_timestamp`, `customer_author_id`, `brand_tweet_id`, `brand_timestamp`, `brand_author_id`, and `inferred_intent`.

### 2.2 Golden Set Quarantine & Mathematical Isolation
To ensure longitudinal benchmark integrity, all 158 human-verified records comprising the Phase 8 Golden Evaluation Set (`data/processed/apple_support/apple_support_intent_golden_set.csv`) were explicitly quarantined and excluded during extraction:

$$\mathcal{S}_{\text{retrieval}} \cap \mathcal{S}_{\text{golden}} = \emptyset$$

* Total threads scanned: **82,105**
* Threads with valid customer-brand pairs: **82,101** (4 single-message threads excluded)
* Quarantined Golden records: **158** (including canonical ambiguous records 410510, 965710, 1066754)
* Final indexed retrieval corpus: **81,943** interaction pairs

Programmatic verification via `verify_retrieval_isolation()` confirms exactly **0 overlapping tweet IDs**.

---

## 3. Duplicate Analysis & Corpus Diagnostics

A comprehensive duplicate audit was performed across the 81,943 pairs:

| Diagnostic Metric | Count | Percentage | Operational Implication |
|---|:---:|:---:|---|
| **Total Corpus Records** | 81,943 | 100.0% | Indexed search pool. |
| **Unique Customer Tweet IDs** | 81,943 | 100.0% | Zero duplicate tweets in input. |
| **Unique Customer Texts** | 81,596 | 99.58% | High lexical diversity across customer inquiries. |
| **Duplicate Customer Queries** | 347 | 0.42% | Low-information phrases repeated by different users. |
| **Unique Brand Tweet IDs** | 81,943 | 100.0% | 1-to-1 tweet identifier mapping. |
| **Unique Brand Texts** | 81,872 | 99.91% | Tailored brand responses (71 standardized greeting repeats). |
| **Unique (Customer, Brand) Pairs** | 81,943 | 100.0% | Zero identical interaction collisions. |

### Top Repeated Customer Inquiries
1. `@AppleSupport` (109 occurrences)
2. `@115858 @AppleSupport` (50 occurrences)
3. `@AppleSupport help` (21 occurrences)
4. `@AppleSupport @115858` (20 occurrences)
5. `@115858 fix this I️` (12 occurrences)

*Design Decision:* Repeated customer queries are preserved in the underlying corpus because different brand replies demonstrate varied routing behaviors (e.g. asking for device model vs. directing immediately to DM). However, the retriever provides a runtime parameter `deduplicate_customer_text=True` to prevent near-duplicate customer inquiries from crowding top-$k$ search results.

---

## 4. Dense Vector Indexing Architecture

### 4.1 Embedding Configuration
* **Model Checkpoint:** `sentence-transformers/all-MiniLM-L6-v2`
* **Base Architecture:** MiniLM (6 layers, 384 hidden dimensions, 12 attention heads)
* **Pretrained Parameters:** ~22.7M parameters
* **Vector Normalization:** $L_2$ unit normalization ($\|v\|_2 = 1.0$)
* **Storage Format:** 32-bit floating point NumPy matrix (`.npy`)
* **Index Size on Disk:** **125.86 MB** ($81,943 \times 384 \times 4 \text{ bytes}$)
* **In-Memory Footprint:** ~126 MB RAM

### 4.2 Vectorized Cosine Search
Because both the index embeddings matrix $\mathbf{X} \in \mathbb{R}^{N \times 384}$ and the encoded query vector $\mathbf{q} \in \mathbb{R}^{384}$ are $L_2$-normalized:

$$\text{CosineSimilarity}(\mathbf{x}_i, \mathbf{q}) = \frac{\mathbf{x}_i \cdot \mathbf{q}}{\|\mathbf{x}_i\|_2 \|\mathbf{q}\|_2} = \mathbf{x}_i \cdot \mathbf{q}$$

The full top-$k$ retrieval across all 81,943 records is executed via a single BLAS matrix-vector product:

$$\mathbf{s} = \mathbf{X} \mathbf{q}^T$$

followed by `np.argpartition` or `np.argsort`. On standard CPU hardware, this operation takes **under 4 milliseconds per query**.

---

## 5. Qualitative Retrieval Analysis & Canonical Traces

Below are representative retrieval traces across primary customer service scenarios:

### Case 1: Battery Drain / Power Inquiries
* **Incoming Query:** `"My iPhone 7 battery is draining super fast after the new update and dies in 30 minutes"`
* **Rank 1 Match (Similarity: 0.8412):**
  * *Historical Customer Query:* `My iPhone 6s battery dies in 30 minutes after updating to iOS 11 @AppleSupport`
  * *Paired Brand Reply:* `@user We want to make sure your battery lasts as expected. Take a look at your battery usage in Settings > Battery and let us know what's using the most power.`
  * *Grounding Quality:* High. Directly provides Apple's official diagnostic step (Settings > Battery) without agent hallucination.

### Case 2: Hardware Damage / Physical Defect
* **Incoming Query:** `"Dropped my iPhone X on concrete and the front glass is completely shattered"`
* **Rank 1 Match (Similarity: 0.8245):**
  * *Historical Customer Query:* `I dropped my phone and the screen completely shattered @AppleSupport how much to fix it`
  * *Paired Brand Reply:* `@user We can definitely provide information on screen repair options and pricing. Check out this link to see options and schedule an appointment: http://apple.co/iPhoneRepair`
  * *Grounding Quality:* High. Directly provides the official Apple repair portal URL and appointment booking path.

### Case 3: iOS 11 Predictive Text / Letter Glitch
* **Incoming Query:** `"Why does my keyboard type an exclamation mark and a box when I type the letter i"`
* **Rank 1 Match (Similarity: 0.8872):**
  * *Historical Customer Query:* `@AppleSupport every time I type the letter i it autocorrects to an A and a question mark box`
  * *Paired Brand Reply:* `@user We're aware of this issue and have a workaround available until the software update is released. Check out these steps: https://support.apple.com/HT208240`
  * *Grounding Quality:* Exceptional. Accurately retrieves Apple's dedicated Knowledge Base article for the known iOS 11.1 autocorrect bug.

### Case 4: Connectivity / WiFi Failure
* **Incoming Query:** `"WiFi keeps disconnecting and grayed out in settings cannot turn on"`
* **Rank 1 Match (Similarity: 0.8019):**
  * *Historical Customer Query:* `My wifi button is greyed out and won't turn on on my iPhone 6 @AppleSupport`
  * *Paired Brand Reply:* `@user Let's work together on this. Have you tried restarting your device or resetting network settings in Settings > General > Reset > Reset Network Settings?`
  * *Grounding Quality:* High. Provides standard network triage steps without hallucinating non-existent settings.

---

## 6. Failure Modes & Edge Case Diagnostics

1. **Ultra-Short / Vague Inquiries (e.g., `"help"`, `"@AppleSupport"`):**
   * *Symptom:* High similarity to generic historical greetings (similarity ~0.72), but the paired responses are non-specific: *"We're here to help! What's happening?"*
   * *Remediation:* A query character/token length guard ($\text{words} \ge 3$) should trigger an automated clarifying prompt before querying the retrieval index.
2. **Multilingual / Code-Switching Inquiries (e.g. Spanish, French):**
   * *Symptom:* `all-MiniLM-L6-v2` is primarily trained on English text. While multilingual queries retrieve matching non-English queries where available, similarity scores are compressed (0.45–0.58).
   * *Remediation:* In future phases, a multilingual dense bi-encoder (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) or language detection front-end should be introduced.
3. **Media / Screenshot Dependency:**
   * *Symptom:* Customers posting tweets like *"See what's happening here [screenshot link]"* lack textual descriptions in Twitter microblog data. Retrieved brand replies consistently say *"DM us with details"*.
   * *Remediation:* OCR on customer-attached images or conversational fallback to human triage.
4. **Out-of-Domain Inquiries (e.g., Android, PC, unrelated complaints):**
   * *Symptom:* Top similarity score remains below 0.35–0.40.
   * *Remediation:* Enforce a strict minimum cosine threshold (`min_score = 0.50`). Queries scoring below 0.50 are flagged as ungrounded and routed to human agent triage.

---

## 7. Artifact Registry

All Phase 11 artifacts are persisted in `data/processed/apple_support/`:

| Artifact Name | Path | Size | Description |
|---|---|:---:|---|
| **Retrieval Corpus CSV** | `data/processed/apple_support/apple_support_retrieval_corpus.csv` | ~19.8 MB | 81,943 customer-brand grounded interaction pairs. |
| **Retrieval Metadata JSON** | `data/processed/apple_support/apple_support_retrieval_metadata.json` | ~3.6 KB | Extraction parameters, Golden Set quarantine registry, duplicate audit. |
| **Retrieval Embeddings NPY** | `data/processed/apple_support/apple_support_retrieval_embeddings.npy` | 120.03 MB | Precomputed $81,943 \times 384$ float32 normalized dense vector matrix. |
| **Embedding Metadata JSON** | `data/processed/apple_support/apple_support_embedding_metadata.json` | ~1.2 KB | Embedding model parameters, batching config, checksums, and timing. |

---

## 8. Test Suite Verification

The Phase 11 retrieval engine is validated by a dedicated test suite in `tests/test_historical_response_retriever.py`:

* **20 Passed Tests (100% Pass Rate):**
  * `TestPairExtraction`: 5 tests verifying boundary fidelity, non-empty text, and single-turn isolation.
  * `TestIsolationValidation`: 3 tests verifying programmatic rejection of golden leaks and asserting $S_{\text{retrieval}} \cap S_{\text{golden}} = \emptyset$.
  * `TestCorpusIntegrity`: 5 tests verifying 81,943 row count, schema columns, non-empty texts, unique IDs, and duplicate stats.
  * `TestHistoricalResponseRetriever`: 6 tests verifying empty query handling, exact self-match ($1.0$), semantic ranking, top-$k$ slicing, intent filtering, and batch retrieval.
  * `TestFullRetrievalIndex`: 1 test verifying end-to-end integration over the actual 81,943-record embeddings matrix on disk.
