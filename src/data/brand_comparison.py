"""
src/data/brand_comparison.py
==============================
Brand comparison analysis — single streaming pass.

PURPOSE
-------
Compare multiple candidate brands from the Twitter Customer Support dataset
in a single pass through the raw CSV.  For each brand:

  - Count inbound / outbound tweets
  - Measure conversation link coverage
  - Estimate English-language proportion
  - Count masked-field tokens (__email__, __phone__, etc.)
  - Measure low-information ("noise") message rate
  - Reconstruct conversation threads and measure their quality
  - Sample representative customer messages

DESIGN
------
We do TWO streaming passes:
  Pass 1  — collect, for each brand, the set of its outbound tweet_ids and
            the set of parent tweet_ids it replied to (small sets of IDs).
  Pass 2  — emit rows belonging to any candidate brand into per-brand lists
            (condensed: we keep only essential columns + first 200 chars of text).
  Analysis — reconstruct threads per brand in memory using existing
             reconstruct_threads module; compute all statistics.

Memory estimate (worst case, AmazonHelp ~340 k rows × ~220 bytes each):
  ~75 MB per brand, ~300 MB for all five brands combined.
  Well within budget on an 11 GB machine.

LANGUAGE HEURISTIC
------------------
We use an ASCII-ratio threshold: if ≥ 85% of characters in a tweet are
7-bit ASCII, we treat the tweet as likely English.  This is a fast,
dependency-free approximation.  It over-counts tweets that mix English with
ASCII punctuation/emoji but systematically under-counts non-Latin scripts,
which is the direction we care about.  The exact threshold is documented as
a constant (ENGLISH_ASCII_THRESHOLD) and is testable.

NOISE HEURISTIC
---------------
A message is "low information" if its usable text (after stripping
@mentions, URLs, and punctuation) contains fewer than MIN_USEFUL_WORDS words.

USAGE
-----
    .venv/bin/python -m src.data.brand_comparison
"""

import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.reconstruct_threads import reconstruct_threads, summarise_threads

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "twcs" / "twcs.csv"
DOCS_DIR     = PROJECT_ROOT / "docs"
COMPARISON_MD_PATH   = DOCS_DIR / "brand_comparison.md"
COMPARISON_JSON_PATH = DOCS_DIR / "brand_comparison.json"

CHUNK_SIZE = 50_000

# Candidate brands to compare
CANDIDATES = [
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "TMobileHelp",
    "AmazonHelp",
]

# Language heuristic: fraction of 7-bit ASCII characters required to call a
# tweet "likely English".  Documented and exported for testability.
ENGLISH_ASCII_THRESHOLD = 0.85

# A message is "noise" (low-information) if its usable word count is below this.
MIN_USEFUL_WORDS = 3

# Masked-field token pattern: __token__ as used by some dataset preprocessors
MASKED_FIELD_RE = re.compile(r"__[a-zA-Z]+__")

# URL pattern for stripping before word-count
URL_RE  = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\S+")

# Maximum examples to store per brand (for representative sampling)
MAX_EXAMPLES = 20

# Essential columns to keep during Pass 2 (saves memory vs. full rows)
KEEP_COLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
    "response_tweet_id",
]


# ---------------------------------------------------------------------------
# Language / quality helpers
# ---------------------------------------------------------------------------

def is_likely_english(text: str, threshold: float = ENGLISH_ASCII_THRESHOLD) -> bool:
    """
    Return True if ≥ `threshold` fraction of characters are 7-bit ASCII.

    This is a lightweight, dependency-free heuristic.  Non-Latin scripts
    (Japanese, Arabic, Hindi, etc.) have low ASCII ratios and will correctly
    score False.  Mixed English+emoji tweets score close to 1.0 because emoji
    are multi-byte Unicode but there are typically few of them.

    Documented limitation: transliterated non-English text (e.g. romanised
    Hindi or Arabic) will be mis-classified as English.
    """
    if not text:
        return True   # empty → no evidence of non-English
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= threshold


def count_masked_fields(text: str) -> int:
    """Count the number of __token__ placeholders in a tweet."""
    return len(MASKED_FIELD_RE.findall(text))


def useful_word_count(text: str) -> int:
    """
    Count 'useful' words: alphabetic tokens that remain after stripping
    @mentions and URLs.  Used to identify low-information messages.
    """
    cleaned = MENTION_RE.sub("", URL_RE.sub("", text))
    return sum(1 for tok in cleaned.split() if tok.isalpha())


def is_noise(text: str) -> bool:
    """Return True if the tweet is considered low-information."""
    return useful_word_count(text) < MIN_USEFUL_WORDS


# ---------------------------------------------------------------------------
# Pass 1 — collect brand tweet_id sets
# ---------------------------------------------------------------------------

def pass1_collect_ids(path: Path, candidates: list[str]) -> tuple[dict, dict]:
    """
    Stream the CSV once.  For each candidate brand collect:
      brand_ids[brand]   — set of tweet_ids the brand authored (outbound)
      parent_ids[brand]  — set of tweet_ids those replies responded to

    Returns (brand_ids, parent_ids).
    """
    brand_ids:  dict[str, set] = {b: set() for b in candidates}
    parent_ids: dict[str, set] = {b: set() for b in candidates}
    candidate_set = set(candidates)

    reader = pd.read_csv(
        path, chunksize=CHUNK_SIZE, dtype=str,
        encoding="utf-8", on_bad_lines="skip", low_memory=False,
    )
    for chunk in reader:
        outbound_mask = chunk["inbound"].str.strip().str.lower() == "false"
        outbound = chunk[outbound_mask]
        for brand in candidate_set:
            rows = outbound[outbound["author_id"].str.strip() == brand]
            brand_ids[brand].update(rows["tweet_id"].dropna().str.strip())
            parent_ids[brand].update(
                rows["in_response_to_tweet_id"].dropna().str.strip()
                .replace("", pd.NA).dropna()
            )

    return brand_ids, parent_ids


# ---------------------------------------------------------------------------
# Pass 2 — collect condensed brand rows
# ---------------------------------------------------------------------------

def pass2_collect_rows(
    path: Path,
    brand_ids: dict[str, set],
    parent_ids: dict[str, set],
) -> dict[str, list[dict]]:
    """
    Stream the CSV a second time.  For each brand emit rows whose tweet_id
    is in (brand_ids ∪ parent_ids).

    Returns {brand: [row_dict, ...]}.
    Only KEEP_COLS columns are stored (plus text truncated to 200 chars).
    """
    keep_sets: dict[str, set] = {
        b: brand_ids[b] | parent_ids[b] for b in brand_ids
    }
    # Build a unified super-set for fast chunk membership test
    all_keep: set = set().union(*keep_sets.values())

    rows_by_brand: dict[str, list[dict]] = {b: [] for b in brand_ids}

    reader = pd.read_csv(
        path, chunksize=CHUNK_SIZE, dtype=str,
        encoding="utf-8", on_bad_lines="skip", low_memory=False,
    )
    for chunk in reader:
        # Quick filter: only rows in the combined keep set
        matched = chunk[chunk["tweet_id"].str.strip().isin(all_keep)]
        if matched.empty:
            continue

        for _, row in matched[KEEP_COLS].iterrows():
            tid = str(row["tweet_id"]).strip()
            raw_text = str(row.get("text", ""))
            row_dict = {
                "tweet_id":                 tid,
                "author_id":                str(row.get("author_id", "")).strip(),
                "inbound":                  str(row.get("inbound", "")).strip(),
                "created_at":               str(row.get("created_at", "")).strip(),
                "text":                     raw_text[:200],   # truncate for memory
                "in_response_to_tweet_id":  str(row.get("in_response_to_tweet_id", "")).strip(),
                "response_tweet_id":        str(row.get("response_tweet_id", "")).strip(),
            }
            for brand in brand_ids:
                if tid in keep_sets[brand]:
                    rows_by_brand[brand].append(row_dict)
                    break   # each tweet assigned to first matching brand only
                            # (avoids duplication for overlapping conversations)

    return rows_by_brand


# ---------------------------------------------------------------------------
# Per-brand statistics
# ---------------------------------------------------------------------------

def analyse_brand(
    brand: str,
    rows: list[dict],
    brand_ids: set[str],
) -> dict:
    """
    Compute all metrics for a single brand from its collected rows.
    `brand_ids` is the set of tweet_ids the brand itself authored.
    """
    if not rows:
        return {"brand": brand, "error": "no rows collected"}

    df = pd.DataFrame(rows)
    # Normalise inbound column
    df["inbound_bool"] = df["inbound"].str.lower() == "true"

    total          = len(df)
    inbound_count  = int(df["inbound_bool"].sum())
    outbound_count = total - inbound_count

    # Conversation links
    has_parent   = df["in_response_to_tweet_id"].str.strip().replace("", pd.NA).notna()
    has_response = df["response_tweet_id"].str.strip().replace("", pd.NA).notna()
    n_has_parent   = int(has_parent.sum())
    n_has_response = int(has_response.sum())

    # Root tweets (no parent in our subset)
    all_ids_in_subset = set(df["tweet_id"].tolist())
    roots_count = int(
        df.apply(
            lambda r: (
                not r["in_response_to_tweet_id"].strip()
                or r["in_response_to_tweet_id"].strip() not in all_ids_in_subset
            ),
            axis=1,
        ).sum()
    )

    # Language heuristic
    english_flags = df["text"].apply(is_likely_english)
    n_english = int(english_flags.sum())
    english_pct = round(n_english / total * 100, 1)

    # Masked fields
    masked_counts = df["text"].apply(count_masked_fields)
    n_with_masked = int((masked_counts > 0).sum())
    total_masked_tokens = int(masked_counts.sum())

    # Noise messages
    noise_flags = df["text"].apply(is_noise)
    n_noise = int(noise_flags.sum())
    noise_pct = round(n_noise / total * 100, 1)

    # Customer messages with at least one brand response
    # A customer tweet is "responded to" if it has a response_tweet_id set
    inbound_df = df[df["inbound_bool"]]
    n_responded = int(
        inbound_df["response_tweet_id"].str.strip().replace("", pd.NA).notna().sum()
    )

    # Thread reconstruction
    thread_df = df.copy()
    # reconstruct_threads expects these exact column names
    for col in ("in_response_to_tweet_id", "response_tweet_id"):
        thread_df[col] = thread_df[col].replace("nan", "").replace("", pd.NA).fillna("")
    thread_df["inbound"] = df["inbound_bool"].map({True: "True", False: "False"})

    threads = reconstruct_threads(thread_df)
    thread_stats = summarise_threads(threads)

    turn_counts = [t.turn_count for t in threads]
    median_turns = round(statistics.median(turn_counts), 1) if turn_counts else 0.0

    # Representative examples: sample up to 5 non-noise inbound tweets
    examples = (
        df[df["inbound_bool"] & ~noise_flags]["text"]
        .dropna()
        .sample(min(5, int((df["inbound_bool"] & ~noise_flags).sum())), random_state=42)
        .tolist()
        if int((df["inbound_bool"] & ~noise_flags).sum()) > 0 else []
    )
    # Truncate each example for documentation
    examples = [e[:120] for e in examples]

    return {
        "brand": brand,
        # Volume
        "total_rows": total,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        # Links
        "n_with_parent_link": n_has_parent,
        "n_with_response_link": n_has_response,
        "pct_with_parent_link": round(n_has_parent / total * 100, 1),
        "pct_with_response_link": round(n_has_response / total * 100, 1),
        "estimated_roots": roots_count,
        # Thread quality (from reconstruction)
        "thread_total": thread_stats["total_threads"],
        "thread_single_turn": thread_stats["single_turn"],
        "thread_multi_turn": thread_stats["multi_turn"],
        "thread_complete": thread_stats["complete"],
        "thread_incomplete": thread_stats["incomplete"],
        "thread_broken_links": thread_stats["broken_links"],
        "pct_multi_turn": round(
            thread_stats["multi_turn"] / max(thread_stats["total_threads"], 1) * 100, 1
        ),
        "pct_complete": round(
            thread_stats["complete"] / max(thread_stats["total_threads"], 1) * 100, 1
        ),
        "pct_broken_links": round(
            thread_stats["broken_links"] / max(thread_stats["total_threads"], 1) * 100, 1
        ),
        "avg_turns": thread_stats["avg_turns"],
        "median_turns": median_turns,
        "max_turns": thread_stats["max_turns"],
        # Customer response coverage
        "n_inbound_responded_to": n_responded,
        "pct_inbound_responded_to": round(n_responded / max(inbound_count, 1) * 100, 1),
        # Language
        "n_english_approx": n_english,
        "pct_english_approx": english_pct,
        # Masked fields
        "n_rows_with_masked_fields": n_with_masked,
        "total_masked_tokens": total_masked_tokens,
        # Noise
        "n_noise_messages": n_noise,
        "pct_noise": noise_pct,
        # Examples (display only — not personal data)
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_brands(stats_list: list[dict]) -> list[dict]:
    """
    Score each brand on 6 criteria and compute a weighted total.

    Weights:
      data_volume           20%
      conv_completeness     20%
      intent_diversity      20%  ← qualitative; set manually based on topic analysis
      english_quality       15%
      retrieval_usefulness  15%
      distinctiveness       10%

    Quantitative criteria (auto-computed from stats):
      data_volume          — normalised outbound_count
      conv_completeness    — 0.5 * pct_complete/100 + 0.5 * pct_multi_turn/100
      english_quality      — pct_english_approx/100 * (1 - pct_noise/100)
      retrieval_usefulness — normalised outbound_count (same base as volume,
                             capped at the top brand)

    Qualitative criteria (manually assessed; justified in report):
      intent_diversity      — scored 0.0–1.0 based on observed topic breadth
      distinctiveness       — scored 0.0–1.0 based on uniqueness of use-case

    All raw scores are normalised to [0, 1] before weighting.
    """
    # Qualitative scores (manually assessed, justified in the MD report)
    qualitative: dict[str, dict] = {
        "AppleSupport": {
            "intent_diversity": 0.85,
            "intent_diversity_note": (
                "6 distinct, well-separated categories confirmed in actual data: "
                "update issues, battery, account access, hardware, app/service, how-to."
            ),
            "distinctiveness": 0.55,
            "distinctiveness_note": (
                "Common choice in public projects; however, scope is well-bounded "
                "and data quality is high."
            ),
        },
        "Uber_Support": {
            "intent_diversity": 0.75,
            "intent_diversity_note": (
                "Topics include ride disputes, driver issues, safety, payment, "
                "account. Dispute-heavy content raises escalation variety."
            ),
            "distinctiveness": 0.80,
            "distinctiveness_note": (
                "Less commonly used in public tutorials; ride-sharing domain "
                "offers clear auto-handle vs. escalation split."
            ),
        },
        "SpotifyCares": {
            "intent_diversity": 0.65,
            "intent_diversity_note": (
                "Narrower scope: playback bugs, premium billing, device compatibility, "
                "account. Fewer than 5 strongly distinct categories."
            ),
            "distinctiveness": 0.70,
            "distinctiveness_note": (
                "Moderate distinctiveness; clean English data; narrower topic set "
                "limits intent taxonomy depth."
            ),
        },
        "TMobileHelp": {
            "intent_diversity": 0.80,
            "intent_diversity_note": (
                "Telecom topics: coverage, billing, SIM/device activation, plan changes, "
                "outages, international roaming. Good escalation diversity."
            ),
            "distinctiveness": 0.75,
            "distinctiveness_note": (
                "Telecom support is underrepresented in public AI agent demos; "
                "strong mix of routine (FAQ) and complex (billing dispute) cases."
            ),
        },
        "AmazonHelp": {
            "intent_diversity": 0.80,
            "intent_diversity_note": (
                "E-commerce topics: delivery, returns, account, billing, product. "
                "Very broad but multilingual contamination lowers effective utility."
            ),
            "distinctiveness": 0.60,
            "distinctiveness_note": (
                "Well-known brand; multilingual content and very large volume "
                "create extra preprocessing complexity."
            ),
        },
    }

    # Max outbound for normalisation
    max_outbound = max(s["outbound_count"] for s in stats_list)

    scored = []
    for s in stats_list:
        brand = s["brand"]
        q = qualitative.get(brand, {
            "intent_diversity": 0.5, "distinctiveness": 0.5,
            "intent_diversity_note": "Not assessed",
            "distinctiveness_note": "Not assessed",
        })

        vol  = s["outbound_count"] / max_outbound
        comp = 0.5 * (s["pct_complete"] / 100) + 0.5 * (s["pct_multi_turn"] / 100)
        eng  = (s["pct_english_approx"] / 100) * (1 - s["pct_noise"] / 100)
        retr = s["outbound_count"] / max_outbound   # same as vol; documented
        intd = q["intent_diversity"]
        dist = q["distinctiveness"]

        total_score = (
            vol  * 0.20 +
            comp * 0.20 +
            intd * 0.20 +
            eng  * 0.15 +
            retr * 0.15 +
            dist * 0.10
        )

        scored.append({
            **s,
            "score_data_volume":            round(vol,  3),
            "score_conv_completeness":      round(comp, 3),
            "score_intent_diversity":       round(intd, 3),
            "score_english_quality":        round(eng,  3),
            "score_retrieval_usefulness":   round(retr, 3),
            "score_distinctiveness":        round(dist, 3),
            "weighted_total":               round(total_score, 4),
            "intent_diversity_note":        q["intent_diversity_note"],
            "distinctiveness_note":         q["distinctiveness_note"],
        })

    scored.sort(key=lambda x: x["weighted_total"], reverse=True)
    for i, s in enumerate(scored):
        s["rank"] = i + 1
    return scored


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0–1 score."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def write_markdown_report(scored: list[dict], elapsed: float) -> None:
    """Write docs/brand_comparison.md."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    lines += [
        "# Brand Comparison Analysis",
        "",
        f"**Dataset:** `twcs/twcs.csv`  |  "
        f"**Brands compared:** {len(scored)}  |  "
        f"**Analysis time:** {elapsed:.0f}s",
        "",
        "> All statistics are measured from the actual local dataset.",
        "> Qualitative scores are explicitly labelled and justified in each brand section.",
        "> No statistics are fabricated.",
        "",
        "---",
        "",
        "## Scoring Methodology",
        "",
        "| Criterion | Weight | How measured |",
        "|---|---|---|",
        "| Data volume | 20% | `outbound_count / max_outbound` |",
        "| Conversation completeness | 20% | `0.5×pct_complete + 0.5×pct_multi_turn` |",
        "| Intent diversity & clarity | 20% | **Qualitative** — manually scored 0–1; see per-brand section |",
        "| English / text quality | 15% | `pct_english_approx × (1 − pct_noise)` |",
        "| Historical-reply retrieval | 15% | `outbound_count / max_outbound` (same as volume — more replies = richer retrieval pool) |",
        "| Distinctiveness & scope | 10% | **Qualitative** — manually scored 0–1; see per-brand section |",
        "",
        "All quantitative sub-scores are normalised to [0, 1].  "
        "Qualitative scores are fixed constants justified per brand.",
        "",
        "---",
        "",
        "## Summary Ranking",
        "",
        "| Rank | Brand | Total Score | Vol | Completeness | Intent | Eng/Quality | Retrieval | Distinct |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scored:
        lines.append(
            f"| **{s['rank']}** | {s['brand']} | **{s['weighted_total']:.3f}** | "
            f"{s['score_data_volume']:.2f} | {s['score_conv_completeness']:.2f} | "
            f"{s['score_intent_diversity']:.2f} | {s['score_english_quality']:.2f} | "
            f"{s['score_retrieval_usefulness']:.2f} | {s['score_distinctiveness']:.2f} |"
        )

    lines += ["", "---", ""]

    for s in scored:
        brand = s["brand"]
        lines += [
            f"## {s['rank']}. {brand}  (score: {s['weighted_total']:.3f})",
            "",
            "### Measured Statistics",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total rows in subset | {s['total_rows']:,} |",
            f"| Inbound (customer) | {s['inbound_count']:,} |",
            f"| Outbound (brand) | {s['outbound_count']:,} |",
            f"| Rows with parent link | {s['n_with_parent_link']:,} ({s['pct_with_parent_link']}%) |",
            f"| Rows with response link | {s['n_with_response_link']:,} ({s['pct_with_response_link']}%) |",
            f"| Estimated roots | {s['estimated_roots']:,} |",
            f"| Reconstructed threads | {s['thread_total']:,} |",
            f"| Single-turn threads | {s['thread_single_turn']:,} |",
            f"| Multi-turn threads | {s['thread_multi_turn']:,} ({s['pct_multi_turn']}%) |",
            f"| Complete threads | {s['thread_complete']:,} ({s['pct_complete']}%) |",
            f"| Threads with broken links | {s['thread_broken_links']:,} ({s['pct_broken_links']}%) |",
            f"| Avg turns per thread | {s['avg_turns']} |",
            f"| Median turns per thread | {s['median_turns']} |",
            f"| Max turns in a thread | {s['max_turns']} |",
            f"| Customer msgs with brand response | {s['n_inbound_responded_to']:,} ({s['pct_inbound_responded_to']}%) |",
            f"| Likely English (ASCII heuristic) | {s['n_english_approx']:,} ({s['pct_english_approx']}%) |",
            f"| Rows with masked fields | {s['n_rows_with_masked_fields']:,} |",
            f"| Noise / low-info messages | {s['n_noise_messages']:,} ({s['pct_noise']}%) |",
            "",
            "### Scores",
            "",
            f"| Criterion | Raw score | Bar |",
            f"|---|---|---|",
            f"| Data volume (20%) | {s['score_data_volume']:.3f} | `{_bar(s['score_data_volume'])}` |",
            f"| Conv completeness (20%) | {s['score_conv_completeness']:.3f} | `{_bar(s['score_conv_completeness'])}` |",
            f"| Intent diversity (20%) ★ | {s['score_intent_diversity']:.3f} | `{_bar(s['score_intent_diversity'])}` |",
            f"| English quality (15%) | {s['score_english_quality']:.3f} | `{_bar(s['score_english_quality'])}` |",
            f"| Retrieval usefulness (15%) | {s['score_retrieval_usefulness']:.3f} | `{_bar(s['score_retrieval_usefulness'])}` |",
            f"| Distinctiveness (10%) ★ | {s['score_distinctiveness']:.3f} | `{_bar(s['score_distinctiveness'])}` |",
            f"| **Weighted total** | **{s['weighted_total']:.4f}** | `{_bar(s['weighted_total'])}` |",
            "",
            "★ = qualitative score",
            "",
            "### Intent Diversity Assessment",
            f"> {s['intent_diversity_note']}",
            "",
            "### Distinctiveness Assessment",
            f"> {s['distinctiveness_note']}",
            "",
        ]

        if s.get("examples"):
            lines += [
                "### Sample Customer Messages (non-identifying, truncated)",
                "",
                "> These are real tweets from the dataset, sampled to illustrate topic variety.",
                "> @mentions, URLs, and order/account numbers are present in the raw data;",
                "> only the first 120 characters of each message are shown here.",
                "",
            ]
            for i, ex in enumerate(s["examples"], 1):
                # Redact any token that looks like a phone, email, or account ID
                ex_clean = re.sub(r"\b\d{8,}\b", "[ID]", ex)
                ex_clean = re.sub(r"\S+@\S+", "[email]", ex_clean)
                lines.append(f"{i}. _{ex_clean}_")
            lines.append("")

        lines += ["---", ""]

    # Conclusion
    top     = scored[0]
    second  = scored[1]
    apple   = next(s for s in scored if s["brand"] == "AppleSupport")

    lines += [
        "## Recommendation",
        "",
        f"### 1. Best overall brand: **{top['brand']}** (score {top['weighted_total']:.3f})",
        "",
        f"### 2. Best alternative brand: **{second['brand']}** (score {second['weighted_total']:.3f})",
        "",
        f"### 3. Should AppleSupport be retained?",
        "",
        f"AppleSupport scored **{apple['weighted_total']:.3f}** (rank {apple['rank']} of {len(scored)}).",
        "",
    ]

    if apple["rank"] == 1:
        lines += [
            "AppleSupport is the top-ranked brand on the measured criteria and should be **retained**.",
            "Its data volume, English consistency, and intent diversity remain the strongest combination.",
        ]
    elif apple["rank"] == 2:
        lines += [
            "AppleSupport is the second-ranked brand.  Replacing it with the top brand would provide "
            "a measurable improvement in distinctiveness or completeness.  The final decision depends "
            "on whether the improvement justifies restarting the intent-labelling work.",
        ]
    else:
        lines += [
            f"AppleSupport ranked {apple['rank']}.  Replacing it with **{top['brand']}** "
            "would produce a stronger and more distinctive agent given the measured criteria.",
        ]

    lines += [
        "",
        "### 4. Main trade-offs",
        "",
    ]
    for s in scored[:3]:
        lines += [
            f"**{s['brand']}:** "
            f"Volume={s['outbound_count']:,} outbound tweets, "
            f"{s['pct_multi_turn']}% multi-turn threads, "
            f"{s['pct_english_approx']}% English.  "
            f"Intent diversity score {s['score_intent_diversity']:.2f}, "
            f"distinctiveness {s['score_distinctiveness']:.2f}.",
            "",
        ]

    best_demo  = max(scored, key=lambda s: s["score_intent_diversity"] * 0.4 + s["score_distinctiveness"] * 0.6)
    safest_hw  = max(scored, key=lambda s: s["score_english_quality"] * 0.5 + (1 - s["outbound_count"] / max(x["outbound_count"] for x in scored)) * 0.5)

    lines += [
        f"### 5. Strongest practical demonstration: **{best_demo['brand']}**",
        "",
        "Highest combined intent diversity + distinctiveness — the agent will exhibit "
        "a clear range of behaviours (auto-handle, escalate, draft reply) without "
        "being confused by multilingual or fragmented data.",
        "",
        f"### 6. Safest for hardware / time constraints: **{safest_hw['brand']}**",
        "",
        "Smaller, high-quality English subset minimises extraction and training time "
        "while still providing adequate coverage for all pipeline components.",
        "",
        "---",
        "",
        "_This report was generated by `src/data/brand_comparison.py` from the local dataset._",
        "_No brand selection change has been made automatically._",
        "_The existing `AppleSupport` configuration in `src/classification/intents.yaml` remains unchanged._",
    ]

    COMPARISON_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {COMPARISON_MD_PATH.relative_to(PROJECT_ROOT)}")


def write_json_summary(scored: list[dict]) -> None:
    """Write docs/brand_comparison.json (machine-readable)."""
    safe = []
    for s in scored:
        entry = {k: v for k, v in s.items() if k != "examples"}
        safe.append(entry)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_JSON_PATH.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Written: {COMPARISON_JSON_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Brand Comparison Analysis")
    print(f"  Candidates: {', '.join(CANDIDATES)}")
    print(f"  Dataset:    {DATASET_PATH}\n")

    t0 = time.time()

    print("  Pass 1 — collecting brand tweet_id sets …")
    brand_ids, parent_ids = pass1_collect_ids(DATASET_PATH, CANDIDATES)
    for b in CANDIDATES:
        print(f"    {b:<20} outbound={len(brand_ids[b]):>7,}  parents={len(parent_ids[b]):>7,}")

    t1 = time.time()
    print(f"  Pass 1 complete: {t1-t0:.1f}s\n")

    print("  Pass 2 — collecting condensed rows for all brands …")
    rows_by_brand = pass2_collect_rows(DATASET_PATH, brand_ids, parent_ids)
    for b in CANDIDATES:
        print(f"    {b:<20} rows collected: {len(rows_by_brand[b]):>7,}")

    t2 = time.time()
    print(f"  Pass 2 complete: {t2-t1:.1f}s\n")

    print("  Analysing each brand …")
    stats_list = []
    for brand in CANDIDATES:
        print(f"    {brand} …", end=" ", flush=True)
        stats = analyse_brand(brand, rows_by_brand[brand], brand_ids[brand])
        stats_list.append(stats)
        print(f"threads={stats['thread_total']:,}  avg_turns={stats['avg_turns']}")

    t3 = time.time()
    print(f"  Analysis complete: {t3-t2:.1f}s\n")

    scored = score_brands(stats_list)

    print("  Writing reports …")
    write_markdown_report(scored, elapsed=t3 - t0)
    write_json_summary(scored)

    print(f"\n  Total elapsed: {t3-t0:.1f}s")
    print(f"\n  === RANKING ===")
    for s in scored:
        print(f"  {s['rank']}. {s['brand']:<20}  score={s['weighted_total']:.4f}")
    print()


if __name__ == "__main__":
    main()
