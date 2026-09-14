# AppleSupport Conversation Thread Reconstruction & Quality Validation

**Phase 4 Documentation**  
**Selected Brand:** `AppleSupport`  
**Pipeline Module:** `src/data/reconstruct_threads.py`  
**Test Suite:** `tests/test_reconstruct_threads.py`  

---

## 1. Executive Summary & Purpose

Phase 4 transforms message-level Twitter customer support interactions from Phase 3 into structured, multi-turn conversation threads. In production customer service applications, models for intent classification, retrieval-augmented response generation, and escalation prediction cannot operate on isolated tweets. They require complete conversational context, accurate turn ordering, directional attribution (customer vs. agent), and transparent data quality diagnostics.

Phase 4 enforces a deterministic, graph-based reconstruction contract with rigorous link validation, explicit denominators, cycle detection, and honest quality metrics. It specifically rejects superficial heuristics (such as treating any single customer-agent pair as "multi-exchange" or reporting 100% link validity by dividing over all messages).

---

## 2. Input & Output Artifacts

### 2.1 Input Artifact
* **File Path:** `data/processed/apple_support/apple_support_messages.csv`
* **Source:** Phase 3 extraction (`src/data/extract_selected_brand.py`)
* **Total Records:** 213,483 rows
* **Required Schema:**
  - `tweet_id` (string): Unique identifier for each tweet.
  - `author_id` (string): Anonymized user or brand handle (`AppleSupport`).
  - `inbound` (bool): `True` for incoming customer messages; `False` for brand responses.
  - `created_at` (string): UTC timestamp formatted e.g., `Tue Oct 31 10:00:00 +0000 2017`.
  - `text` (string): Original message text.
  - `response_tweet_id` (string): Comma-separated tweet IDs responding to this tweet.
  - `in_response_to_tweet_id` (string): Parent tweet ID to which this tweet responds.
  - `row_type` (string): Provenance category (`brand_authored`, `inbound_to_brand`, `brand_referenced`).

### 2.2 Output Artifacts
All outputs are written deterministically to `data/processed/apple_support/`:
* **`apple_support_threads.jsonl`**: UTF-8 encoded JSON Lines file containing one complete conversation thread object per line.
* **`apple_support_thread_metadata.json`**: Comprehensive metadata and quality audit containing dataset statistics, relationship metrics with explicit denominators, quality flags, and operational parameters.

---

## 3. Thread Data Model & JSONL Schema

Each line in `apple_support_threads.jsonl` conforms to the following schema:

```json
{
  "thread_id": "thread_123456",
  "root_tweet_id": "123456",
  "message_count": 4,
  "customer_message_count": 2,
  "brand_message_count": 2,
  "start_timestamp": "Tue Oct 31 09:00:00 +0000 2017",
  "end_timestamp": "Tue Oct 31 09:30:00 +0000 2017",
  "is_complete": true,
  "has_both_directions": true,
  "has_3plus_messages": true,
  "has_multiple_exchanges": true,
  "has_broken_links": false,
  "cycle_detected": false,
  "has_duplicate_ids": false,
  "has_conflicting_duplicates": false,
  "has_empty_text": false,
  "malformed_timestamps": 0,
  "quality_flags": [],
  "link_metrics": {
    "parent_links_present": 3,
    "parent_links_valid": 3,
    "parent_links_missing": 0,
    "response_links_present": 3,
    "response_links_valid": 3,
    "response_links_missing": 0
  },
  "messages": [
    {
      "tweet_id": "123456",
      "author_id": "cust_101",
      "inbound": "True",
      "row_type": "inbound_to_brand",
      "created_at": "Tue Oct 31 09:00:00 +0000 2017",
      "text": "My phone won't charge after update.",
      "response_tweet_id": "123457",
      "in_response_to_tweet_id": ""
    },
    {
      "tweet_id": "123457",
      "author_id": "AppleSupport",
      "inbound": "False",
      "row_type": "brand_authored",
      "created_at": "Tue Oct 31 09:10:00 +0000 2017",
      "text": "@cust_101 Have you tried a hard restart?",
      "response_tweet_id": "123458",
      "in_response_to_tweet_id": "123456"
    }
  ]
}
```

---

## 4. Graph Reconstruction Methodology

### 4.1 Lookup Table Construction
Three hash indexes are constructed in $O(N)$ time:
1. `id_to_row`: Map from `tweet_id` to message row dictionary. For duplicate tweet IDs, the first occurrence is retained deterministically, and duplicates are logged.
2. `child_to_parent`: Map from `tweet_id` to parent ID (`in_response_to_tweet_id`).
3. `parent_to_children`: Map from `parent_tweet_id` to list of downstream reply IDs.

### 4.2 Root Identification
A tweet is identified as a conversation root if:
* It has no parent (`in_response_to_tweet_id` is empty/null).
* Its parent ID is referenced, but the parent tweet does NOT exist in the extracted dataset. In this case, the message is preserved as an orphan root and marked with `has_broken_links = True`.

### 4.3 Breadth-First Graph Traversal (BFS)
Starting from each root node, BFS traverses downstream children via `parent_to_children`:
* **Visited Tracking:** Nodes are tracked in a set. If BFS encounters a node already in `visited`, a cycle or diamond graph is detected, setting `cycle_detected = True` without infinite recursion.
* **Missing References:** If a child ID is missing from `id_to_row`, `has_broken_links` is set to `True`.
* **Cycle Fallback:** After root traversal, any remaining unvisited nodes in `id_to_row` (e.g., closed mutual-parent loops without external roots) are traversed in a fallback loop so that zero messages are lost.

### 4.4 Deterministic Ordering & Tie-Breaking
Messages within a thread are sorted using:
1. **Primary Key:** Chronological timestamp (`created_at` parsed to UTC timestamp). Malformed or missing timestamps evaluate to `+inf` (placed at the end) and increment `malformed_timestamps`.
2. **Secondary Key (Tie-breaker):** `tweet_id` as a deterministic string tie-breaker.

The list of reconstructed threads is deterministically sorted by `root_tweet_id` (numerically when possible, falling back to lexicographic string).

---

## 5. Precise Quality Metrics & Denominators

To prevent misleading evaluation results, every metric explicitly defines its numerator and denominator:

| Metric Name | Numerator | Denominator | Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`pct_valid_parent_links`** | Parent links whose target exists in dataset (131,378) | Messages with non-empty `in_response_to_tweet_id` (138,729) | **94.7%** | Measures how often a declared parent actually exists in the corpus (`131,378 / (131,378 + 7,351)`). |
| **`pct_valid_response_ids`** | Response targets existing in dataset (131,378) | Total individual IDs listed in `response_tweet_id` (157,589) | **83.4%** | Measures how often referenced reply IDs actually exist (`131,378 / (131,378 + 26,211)`). |
| **`pct_three_plus_messages`** | Threads with $\ge 3$ messages (17,359) | Total reconstructed threads (82,105) | **21.1%** | Captures conversations beyond brief single interactions. |
| **`pct_multiple_exchanges`** | Threads with $\ge 2$ customer AND $\ge 2$ brand messages (17,232) | Total reconstructed threads (82,105) | **21.0%** | Strict measure of genuine back-and-forth dialog ($\ge 4$ turns). |
| **`pct_ends_with_brand_reply`** | Threads ending with an outbound (brand) response (82,105) | Total reconstructed threads (82,105) | **100.0%** | **EXTRACTION-DEPENDENT STRUCTURAL ARTIFACT:** 100% of threads end with brand because Phase 3 only extracted customer tweets that were replied to by AppleSupport. It is NOT proof of true customer issue resolution. |
| **`pct_broken_parent_threads`** | Threads with $\ge 1$ missing parent link (7,351) | Total reconstructed threads (82,105) | **8.95%** | Proportion of threads where the conversation starts mid-dialogue due to unextracted parent tweets. |
| **`pct_missing_response_threads`**| Threads with $\ge 1$ missing response target (16,406) | Total reconstructed threads (82,105) | **19.98%** | Proportion of threads where referenced reply tweets exist on Twitter but are absent from the dataset. |

### 5.1 Multiple-Exchange Definition
A thread is categorized as `has_multiple_exchanges` if and only if:
* `customer_message_count >= 2`
* `brand_message_count >= 2`
* `message_count >= 4`

A two-turn exchange (1 customer question + 1 brand answer) is a single exchange. Labeling 2-turn exchanges as "multi-turn" or "multiple-exchange" artificially inflates conversational complexity.

---

## 6. CLI Usage

The reconstruction pipeline is executed via `src/data/reconstruct_threads.py`:

```bash
# Standard execution
.venv/bin/python -m src.data.reconstruct_threads \
  --input data/processed/apple_support/apple_support_messages.csv \
  --output-dir data/processed/apple_support

# Overwrite existing artifacts
.venv/bin/python -m src.data.reconstruct_threads \
  --input data/processed/apple_support/apple_support_messages.csv \
  --output-dir data/processed/apple_support \
  --overwrite

# Strict mode: fail with exit code 1 if data errors or conflicts are found
.venv/bin/python -m src.data.reconstruct_threads \
  --input data/processed/apple_support/apple_support_messages.csv \
  --output-dir data/processed/apple_support \
  --strict
```

### Options & Safety Guardrails
* `--input PATH`: Input CSV path (required).
* `--output-dir DIR`: Output directory (default: `data/processed/apple_support`).
* `--brand BRAND`: Brand identifier for metadata (default: `AppleSupport`).
* `--overwrite`: Explicit permission to overwrite existing JSONL and metadata files. If omitted and target files exist, execution halts with exit code 1.
* `--strict`: Fails on quality anomalies or validation errors.

---

## 7. Known Limitations & Audit Findings

1. **Extraction-Dependent Final-Brand Bias:**
   `100% of threads end with a brand message` is a direct consequence of the Phase 3 extraction filter. Phase 3 collected messages authored by AppleSupport and customer messages that AppleSupport replied to. Any customer tweet that AppleSupport did NOT reply to was excluded. Consequently, every leaf in the extracted conversation graph terminates at a brand reply by construction. Furthermore, **8,848 final brand messages (10.8% of threads)** declare outgoing `response_tweet_id` links referencing 9,717 subsequent customer tweets that were never captured. This metric must never be interpreted as complete problem resolution.
2. **Message Conservation:**
   Verification confirmed exact conservation: all 213,483 rows in `apple_support_messages.csv` are accounted for across the 82,105 threads. Exactly 0 messages were dropped, 0 messages were duplicated, and 0 messages appear in multiple threads.
3. **Twitter API Extract Window Truncation:**
   7,351 parent links point to tweets created outside the Kaggle dataset's time window or since deleted. These conversations start mid-dialogue at the first captured turn.
4. **Branching Conversations:**
   When a brand responds to a customer with multiple alternative options, or when customers reply multiple times, trees branch into multi-path graphs. BFS linearizes these branches chronologically.
5. **Private Direct Messages (DM):**
   Many public support interactions conclude with "Please DM us your serial number." In the public dataset, the private resolution is absent. Downstream escalation and resolution models must treat DM redirection as a specific conversation terminal state.

---

## 8. Downstream Handoff

Reconstructed threads in `apple_support_threads.jsonl` are structured to directly feed:
* **Phase 5 (Intent Classification):** Initial customer messages (`messages[0]` or customer turns) map to defined intent taxonomies.
* **Phase 6 (Retrieval & Knowledge Base):** Customer queries paired with brand resolutions form the grounding index for RAG.
* **Phase 7 (Response Generation & Escalation):** Conversation context history evaluated for sentiment decline, repeated failed solutions, and human escalation triggers.
