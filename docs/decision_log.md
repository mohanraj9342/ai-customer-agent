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
