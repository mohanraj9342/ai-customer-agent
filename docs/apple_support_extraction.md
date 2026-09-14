# AppleSupport Data Extraction

This document describes Phase 3 of the pipeline: extracting AppleSupport
customer-support records from the raw dataset and producing a clean,
validated message-level artifact for downstream use.

---

## Why AppleSupport Was Selected

A quantitative comparison of five candidate brands was performed against the
raw dataset using `src/data/brand_comparison_validated.py`.  AppleSupport was
selected for the following reasons:

| Criterion | AppleSupport | Rationale |
|---|---|---|
| English purity (RB) | 99.8% | Virtually no multilingual noise |
| Noise rate (RB) | 3.1% | Lowest among high-volume brands |
| Intent diversity (QL) | 0.85 / 1.0 | Highest of all five candidates; 6 well-separated categories |
| Outbound training pool | 106,860 tweets | Adequate for retrieval and classification |
| Broken-link threads | 9.0% | Lowest of all high-volume brands |
| Est. true response rate | 84.7% | Highest among high-volume brands |

The full scoring methodology and per-brand corrected statistics are in
`docs/brand_comparison_validated.md`.

---

## Extraction Definition

"AppleSupport extraction" means collecting exactly two categories of rows:

### `brand_reply`
Rows where `author_id == "AppleSupport"` and `inbound == "False"`.
These are the brand's actual support responses — the ground truth for
reply generation and retrieval.

### `customer_inbound`
Rows whose `tweet_id` appears in the `in_response_to_tweet_id` field of
a `brand_reply` row.  These are the customer messages the brand
directly replied to.

### What is NOT extracted

| Excluded content | Reason |
|---|---|
| Customer tweets that @mentioned AppleSupport but received no reply | Cannot be identified without a full text scan; estimated ~19,000 via @mention heuristic |
| Conversation ancestors beyond the direct parent | Full thread context is rebuilt in Phase 4 |
| Other brand outbound rows | Only AppleSupport is extracted |
| Retweets / quoted tweets not linked to AppleSupport | No link to brand's conversation graph |

> **Sampling bias note:** Because unanswered customer tweets are excluded,
> the extracted subset overstates response coverage when measured against
> only the extracted rows.  The Phase 2 validated analysis estimated a true
> response rate of ~84.7% (not 100%).

---

## Input Schema

The raw dataset must be a CSV file with at minimum these columns:

| Column | Type | Description |
|---|---|---|
| `tweet_id` | string | Unique tweet identifier |
| `author_id` | string | Twitter username of the author |
| `inbound` | string | `"True"` if inbound (customer), `"False"` if outbound (brand) |
| `created_at` | string | Twitter timestamp (e.g. `Tue Oct 31 22:10:47 +0000 2017`) |
| `text` | string | Full tweet text |
| `response_tweet_id` | string | tweet_id(s) of the brand reply to this tweet (comma-sep) |
| `in_response_to_tweet_id` | string | tweet_id of the tweet this row is replying to |

Extra columns in the raw CSV are passed through to the output.

---

## Output Schema

### `apple_support_messages.csv`

Contains all extracted rows, sorted deterministically by `tweet_id` (numeric).
Columns include all input columns plus:

| Extra column | Values | Description |
|---|---|---|
| `text_raw` | string | Original tweet text before normalisation |
| `text` | string | Normalised text (HTML entities decoded, whitespace collapsed) |
| `row_type` | `"brand_reply"` / `"customer_inbound"` | Identifies which extraction category the row belongs to |

**Normalisation rules applied to `text`:**

- HTML entities decoded: `&amp;` → `&`, `&lt;` → `<`, `&gt;` → `>`, `&quot;` → `"`, `&#39;` → `'`, `&nbsp;` → ` `
- Consecutive whitespace collapsed to a single space
- Leading/trailing whitespace stripped

What is NOT changed: @mentions, URLs, emoji, punctuation, casing.

### `apple_support_metadata.json`

Machine-readable summary of the extraction run.  Contains:

- Selected brand and schema version
- Input and output paths (repository-relative only — no absolute local paths)
- Extraction timestamp (UTC)
- Row counts: total scanned, written, inbound, outbound
- Unique and duplicate tweet IDs
- Link field coverage: rows with parent field, rows with response field
- Data quality counts: empty text, missing timestamps, invalid inbound values,
  self-referencing links, conflicting duplicates
- Validation summary (errors, warnings, info messages)
- Documented limitations
- Next pipeline step reference

---

## CLI Usage

```bash
# Default (AppleSupport, twcs/twcs.csv → data/processed/apple_support/)
.venv/bin/python -m src.data.extract_selected_brand

# Explicit paths
.venv/bin/python -m src.data.extract_selected_brand \
  --input twcs/twcs.csv \
  --brand AppleSupport \
  --output-dir data/processed/apple_support

# Re-run (overwrite existing output)
.venv/bin/python -m src.data.extract_selected_brand --overwrite

# Fail on any data-quality warning (strict mode)
.venv/bin/python -m src.data.extract_selected_brand --strict
```

### All options

| Option | Default | Description |
|---|---|---|
| `--input PATH` | `twcs/twcs.csv` | Path to raw CSV |
| `--brand STR` | `AppleSupport` | Brand `author_id` as in the dataset |
| `--output-dir DIR` | `data/processed/apple_support` | Output directory |
| `--chunk-size N` | `50000` | Rows per pandas read chunk |
| `--overwrite` | off | Allow replacing existing output files |
| `--strict` | off | Treat warnings as errors (exit 1) |

---

## Output Artifacts

```
data/processed/apple_support/
  apple_support_messages.csv      ← extracted messages (gitignored)
  apple_support_metadata.json     ← metadata summary (gitignored)
```

> `apple_support_threads.jsonl` is produced in Phase 4 by
> `src/data/reconstruct_threads.py`.

---

## Validation Rules

The extraction validates the output in three levels:

### ERROR (extraction fails)
| Check | Consequence |
|---|---|
| Required column missing from input | Abort before any processing |
| Input file not readable | Abort before any processing |
| Zero rows extracted | Abort with RuntimeError |
| Zero brand outbound rows found | Abort with RuntimeError |
| Output CSV missing required columns | Fail after write |

### WARNING (logged, recorded in metadata)
| Check |
|---|
| Duplicate `tweet_id` values |
| Conflicting records (same `tweet_id`, different text or author) |
| Empty or whitespace-only `text` after normalisation |
| Missing `created_at` timestamp |
| `inbound` value not in `{"True", "False"}` |
| `in_response_to_tweet_id` equals own `tweet_id` (self-reference) |
| `response_tweet_id` equals own `tweet_id` (self-reference) |
| Fewer than 100 unique tweet IDs (insufficient for downstream) |

### INFO (always present)
| Check |
|---|
| All required columns present |
| Brand outbound row count |
| Customer inbound row count |
| Unique tweet count with sufficiency judgement |
| Rows with parent and response link fields |
| Notes on link field vs reference validity |
| Note on unanswered-tweet exclusion |

---

## Difference Between Message Extraction and Thread Reconstruction

| Phase 3 (this phase) | Phase 4 |
|---|---|
| Produces a flat CSV of individual messages | Produces structured conversation threads |
| Rows are sorted by `tweet_id` | Threads are sorted chronologically |
| Broken links are counted but not resolved | BFS walks all reachable nodes per thread |
| Does not assume conversations are complete | Explicitly flags incomplete threads |
| Output: `apple_support_messages.csv` | Output: `apple_support_threads.jsonl` |

---

## Reproducibility

The extraction is fully deterministic:

1. The output CSV is always sorted by `tweet_id` (numeric).
2. The same raw CSV produces the same output CSV, byte-for-byte.
3. All configuration parameters are written to `apple_support_metadata.json`.
4. The `schema_version` field in metadata records which extraction logic was used.

To reproduce from scratch:

```bash
# 1. Ensure the raw dataset is at twcs/twcs.csv
# 2. Run extraction
.venv/bin/python -m src.data.extract_selected_brand --overwrite

# 3. Verify row counts against the metadata
cat data/processed/apple_support/apple_support_metadata.json | python -m json.tool
```

---

## Known Limitations

1. **Unanswered tweets excluded.** Customer tweets that @mentioned
   `AppleSupport` but received no reply are not in the extracted subset.
   The Phase 2 analysis estimated ~19,000 such tweets via @mention heuristic.

2. **One level of context only.** Only the direct parent (the customer tweet
   a brand reply responded to) is collected. Multi-turn conversation ancestors
   are reconstructed in Phase 4.

3. **Link field ≠ reference exists.** `rows_with_parent_field` counts
   non-null field values, not whether the referenced tweet_id exists in the
   subset. Broken links are detected during Phase 4 thread reconstruction.

4. **Text truncated in brand_comparison intermediate analysis.** The
   analysis scripts truncated text to 200 characters for memory efficiency.
   This extraction preserves the full text.

5. **No deduplication by default.** Duplicate tweet_ids are reported as
   warnings in metadata but both rows are retained for auditability.
   Downstream stages should deduplicate if needed.

---

## How This Phase Feeds the Next Phase

The output of this phase (`apple_support_messages.csv`) is the direct input
to Phase 4 thread reconstruction:

```bash
# Phase 4 (not yet implemented — this is the handoff contract)
.venv/bin/python -m src.data.reconstruct_threads \
  data/processed/apple_support/apple_support_messages.csv
```

The `reconstruct_threads.py` module expects a DataFrame with these columns
(all present in the Phase 3 output):

```
tweet_id, inbound, created_at, text,
in_response_to_tweet_id, response_tweet_id
```

The `row_type` and `text_raw` columns are additional and are ignored by the
reconstruction logic.
