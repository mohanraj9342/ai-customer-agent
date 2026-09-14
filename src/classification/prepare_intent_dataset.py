"""
src/classification/prepare_intent_dataset.py
============================================
Phase 5 — Intent classification candidate dataset preparation.

PURPOSE
-------
Extracts the primary customer inquiry from each reconstructed conversation
thread in Phase 4, proposes preliminary intent labels using transparent,
deterministic pattern-matching rules, detects ambiguous multi-category
conflicts, and exports a standardized dataset for human review and downstream
model training.

DESIGN PRINCIPLES
-----------------
1. Unit of Classification:
   The chronologically earliest customer-authored message (inbound == 'True')
   with non-empty text in each thread.
2. Leakage Protection:
   Brand replies and future turns are NEVER inspected for intent assignment.
   ends_with_brand_reply and is_complete are preserved strictly as thread context,
   never as intent or resolution signals.
3. Transparent & Conservative Labeling:
   Deterministic keyword and regex pattern matching. If multiple categories
   match with competing strength, the row is flagged as 'needs_review' rather
   than forcing an arbitrary label.
4. Preserved Traceability:
   Original tweet_id, thread_id, raw text, timestamps, and graph flags are
   retained across all records.

OUTPUT ARTIFACTS
----------------
- data/processed/apple_support/apple_support_intent_candidates.csv
- data/processed/apple_support/apple_support_intent_metadata.json

CLI USAGE
---------
.venv/bin/python -m src.classification.prepare_intent_dataset \\
    --input data/processed/apple_support/apple_support_threads.jsonl \\
    --output-dir data/processed/apple_support
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("  %(levelname)s  %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TAXONOMY_VERSION = "2.0"
OUTPUT_CSV_NAME = "apple_support_intent_candidates.csv"
OUTPUT_META_NAME = "apple_support_intent_metadata.json"

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_support"
    / "apple_support_threads.jsonl"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "processed" / "apple_support"

# 11 Grounded Intent Categories in Taxonomy v2.0
VALID_INTENTS = {
    "software_update",
    "battery_power",
    "device_hardware",
    "app_or_service_issue",
    "account_access",
    "connectivity_network",
    "billing_payment",
    "order_shipping",
    "feature_how_to",
    "complaint_feedback",
    "unknown_other",
}

# Label source categories
VALID_LABEL_SOURCES = {"rule", "fallback", "needs_review"}

# Confidence levels
VALID_CONFIDENCES = {"high", "medium", "low", "unassigned"}

OUTPUT_COLUMNS = [
    "thread_id",
    "tweet_id",
    "timestamp",
    "text",
    "candidate_intent",
    "label_source",
    "label_reason",
    "label_confidence",
    "selection_reason",
    "thread_message_count",
    "customer_message_count",
    "brand_message_count",
    "has_missing_parent_link",
    "has_missing_response_target",
    "is_complete",
    "ends_with_brand_reply",
    "taxonomy_version",
]


# ---------------------------------------------------------------------------
# Text Normalization for Pattern Matching (Internal Only)
# ---------------------------------------------------------------------------
_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|#39|apos|nbsp);")
_ENTITY_MAP = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "#39": "'",
    "apos": "'",
    "nbsp": " ",
}
_MENTION_URL_RE = re.compile(r"(@\w+|https?://\S+)")


def normalize_for_matching(text: str) -> str:
    """
    Lightly normalize customer text for rule evaluation only.
    Preserves original text for the CSV output.
    """
    if not isinstance(text, str):
        return ""
    # Decode entities
    t = _ENTITY_RE.sub(lambda m: _ENTITY_MAP.get(m.group(1), " "), text)
    # Strip @mentions and URLs for clean lexical matching
    t = _MENTION_URL_RE.sub(" ", t)
    # Lowercase & collapse whitespace
    t = " ".join(t.lower().split())
    return t


# ---------------------------------------------------------------------------
# Lexical & Pattern Rules
# ---------------------------------------------------------------------------

# Patterns mapped to (regex, match_strength, reason)
# Strength: 2 = high-specificity signature, 1 = moderate keyword
INTENT_PATTERNS: dict[str, list[tuple[re.Pattern, int, str]]] = {
    "software_update": [
        (
            re.compile(
                r"\b(ios\s*11|high\s*sierra|macos|ios\s*update|new\s*update|latest\s*update|software\s*update|system\s*update)\b"
            ),
            2,
            "os_update_version_mention",
        ),
        (
            re.compile(r"\b(question\s*mark|question\s*marks|letter\s*i\b)"),
            2,
            "ios11_autocorrect_bug",
        ),
        (
            re.compile(
                r"\b(after\s*the\s*update|since\s*the\s*update|updated\s*my\s*phone|update\s*glitch|update\s*bug)\b"
            ),
            2,
            "post_update_malfunction",
        ),
        (
            re.compile(r"\b(update|updated|upgrade|upgraded|patch)\b"),
            1,
            "update_keyword",
        ),
    ],
    "battery_power": [
        (
            re.compile(
                r"\b(battery\s*drain|battery\s*draining|battery\s*life|battery\s*percentage|battery\s*dies|battery\s*dropping)\b"
            ),
            2,
            "battery_drain_phrase",
        ),
        (
            re.compile(
                r"\b(overheating|phone\s*gets\s*hot|phone\s*hot|charging\s*cable|charger\s*cable|won't\s*charge|not\s*charging)\b"
            ),
            2,
            "power_charging_overheat",
        ),
        (
            re.compile(r"\b(battery|charger|charging|recharge|power\s*bank)\b"),
            1,
            "battery_keyword",
        ),
    ],
    "device_hardware": [
        (
            re.compile(
                r"\b(cracked\s*screen|broken\s*screen|shattered\s*screen|screen\s*cracked|display\s*cracked)\b"
            ),
            2,
            "screen_damage_phrase",
        ),
        (
            re.compile(
                r"\b(touch\s*screen\s*not\s*working|screen\s*unresponsive|touch\s*unresponsive|unresponsive\s*screen)\b"
            ),
            2,
            "touch_screen_failure",
        ),
        (
            re.compile(
                r"\b(camera\s*black|camera\s*not\s*working|front\s*camera|speaker\s*crackling|microphone\s*not\s*working|home\s*button\s*broken|power\s*button\s*stuck|water\s*damage)\b"
            ),
            2,
            "hardware_component_defect",
        ),
        (
            re.compile(r"\b(hardware|screen|display|speaker|camera|mic|microphone)\b"),
            1,
            "hardware_keyword",
        ),
    ],
    "account_access": [
        (
            re.compile(
                r"\b(apple\s*id|icloud|itunes\s*account|verification\s*code|two\s*factor|2fa|security\s*questions?)\b"
            ),
            2,
            "account_security_phrase",
        ),
        (
            re.compile(
                r"\b(account\s*(locked|disabled|suspended)|forgot\s*password|reset\s*password|can't\s*log\s*in|can't\s*sign\s*in)\b"
            ),
            2,
            "account_access_failure",
        ),
        (
            re.compile(r"\b(account|login|log\s*in|sign\s*in|passcode|password)\b"),
            1,
            "login_keyword",
        ),
    ],
    "connectivity_network": [
        (
            re.compile(
                r"\b(wi-?fi|bluetooth|cellular\s*data|mobile\s*data|no\s*service|sim\s*card|airpods\s*disconnect|dropped\s*calls?)\b"
            ),
            2,
            "network_connectivity_phrase",
        ),
        (
            re.compile(
                r"\b(disconnecting\s*from\s*wifi|won't\s*connect\s*to\s*wifi|can't\s*connect\s*to\s*wifi|bluetooth\s*won't\s*pair)\b"
            ),
            2,
            "connection_pairing_issue",
        ),
        (
            re.compile(r"\b(wifi|bluetooth|lte|4g|sim|carrier|signal|hotspot)\b"),
            1,
            "connectivity_keyword",
        ),
    ],
    "billing_payment": [
        (
            re.compile(
                r"\b(unexpected\s*charge|charged\s*twice|apple\s*pay|refund\s*request|request\s*a\s*refund|unauthorized\s*charge|subscription\s*renewal)\b"
            ),
            2,
            "billing_dispute_phrase",
        ),
        (
            re.compile(
                r"\b(billed|charged|receipt|subscription|refund|in-?app\s*purchase|credit\s*card|debit\s*card)\b"
            ),
            1,
            "billing_keyword",
        ),
    ],
    "order_shipping": [
        (
            re.compile(
                r"\b(tracking\s*number|package\s*delivered|shipment\s*status|preparing\s*for\s*shipment|store\s*pickup|online\s*order)\b"
            ),
            2,
            "order_tracking_phrase",
        ),
        (
            re.compile(r"\b(fedex|ups|shipped|shipping|delivery|dispatch|dispatched)\b"),
            1,
            "shipping_keyword",
        ),
    ],
    "app_or_service_issue": [
        (
            re.compile(
                r"\b(apple\s*music|imessage|facetime|app\s*store|safari|apple\s*maps|carplay|airdrop|siri|apple\s*podcasts)\b"
            ),
            2,
            "native_app_service_name",
        ),
        (
            re.compile(
                r"\b(app\s*crashes|app\s*crashing|app\s*keeps\s*closing|app\s*freezing|can't\s*download\s*apps?)\b"
            ),
            2,
            "app_malfunction_phrase",
        ),
    ],
    "feature_how_to": [
        (
            re.compile(
                r"\b(how\s*do\s*i|how\s*can\s*i|is\s*there\s*a\s*way\s*to|can\s*you\s*tell\s*me\s*how|how\s*to)\b"
            ),
            2,
            "how_to_inquiry_starter",
        ),
        (
            re.compile(r"\b(where\s*is\s*the\s*setting|how\s*does\s*one)\b"),
            1,
            "configuration_inquiry",
        ),
    ],
    "complaint_feedback": [
        (
            re.compile(
                r"\b(fix\s*your\s*shit|fix\s*this\s*shit|worst\s*phone|hate\s*apple|terrible\s*customer\s*service|worst\s*company|horrible\s*update)\b"
            ),
            2,
            "strong_complaint_phrase",
        ),
        (
            re.compile(
                r"\b(ridiculous|unacceptable|disappointed|waste\s*of\s*money|garbage)\b"
            ),
            1,
            "negative_sentiment_keyword",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Labeling Engine
# ---------------------------------------------------------------------------

@dataclass
class LabelResult:
    intent: str
    source: str
    reason: str
    confidence: str


def classify_customer_message(raw_text: str) -> LabelResult:
    """
    Apply deterministic pattern rules to assign a candidate intent.

    Rules:
      1. Empty or very short conversational mentions → unknown_other (fallback, low).
      2. Score all categories by matched weights.
      3. If top score >= 2 and clear winner → specific intent (rule, high).
      4. If top score == 1 and clear winner → specific intent (rule, medium).
      5. If top score >= 1 and multi-category tie/conflict → needs_review (needs_review, low).
      6. If score == 0 → unknown_other (fallback, low).
    """
    normalized = normalize_for_matching(raw_text)

    # Filter out empty or pure mentions
    words = normalized.split()
    if not words or len(normalized) < 8 or len(words) <= 1:
        return LabelResult(
            intent="unknown_other",
            source="fallback",
            reason="too_short_or_conversational",
            confidence="low",
        )

    scores: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}

    for intent_name, patterns in INTENT_PATTERNS.items():
        score = 0
        matched_reasons = []
        for pattern, weight, r_label in patterns:
            if pattern.search(normalized):
                score += weight
                matched_reasons.append(r_label)
        if score > 0:
            scores[intent_name] = score
            reasons[intent_name] = matched_reasons

    if not scores:
        return LabelResult(
            intent="unknown_other",
            source="fallback",
            reason="no_matching_rules_found",
            confidence="low",
        )

    # Sort categories by score descending
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_intent, top_score = sorted_scores[0]

    # Check for conflict / tie among top categories
    if len(sorted_scores) > 1:
        second_intent, second_score = sorted_scores[1]
        # Ambiguity condition: top score and second score are tied,
        # or top score is 2 and second score is 2 (competing strong signatures)
        if top_score == second_score or (top_score >= 2 and second_score >= 2):
            conflict_names = [k for k, v in sorted_scores if v >= second_score]
            return LabelResult(
                intent="needs_review",
                source="needs_review",
                reason=f"conflicting_rules_matched: {conflict_names}",
                confidence="low",
            )

    # Clear winner
    primary_reason = ", ".join(reasons[top_intent])
    confidence = "high" if top_score >= 2 else "medium"

    return LabelResult(
        intent=top_intent,
        source="rule",
        reason=f"matched: {primary_reason}",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Candidate Selection
# ---------------------------------------------------------------------------

@dataclass
class CandidateSelection:
    message: Optional[dict] = None
    reason: str = ""
    excluded: bool = False
    exclusion_reason: str = ""


def select_first_customer_message(messages: list[dict]) -> CandidateSelection:
    """
    Select the earliest chronologically customer message with non-empty text.

    Rules:
      1. Inspect messages where inbound == 'True'.
      2. If no customer messages exist, exclude thread.
      3. If customer messages exist, take the first with non-empty text.
      4. If first customer message is empty/whitespace, evaluate subsequent customer messages.
    """
    customer_msgs = [m for m in messages if str(m.get("inbound", "")).strip().lower() in ("true", "1")]

    if not customer_msgs:
        return CandidateSelection(
            excluded=True,
            exclusion_reason="no_customer_message",
        )

    for idx, m in enumerate(customer_msgs):
        text = str(m.get("text", "") or "").strip()
        if text and text.lower() != "nan":
            selection_reason = (
                "first_customer_message_chronological"
                if idx == 0
                else f"subsequent_customer_message_fallback (index {idx})"
            )
            return CandidateSelection(
                message=m,
                reason=selection_reason,
                excluded=False,
            )

    return CandidateSelection(
        excluded=True,
        exclusion_reason="all_customer_messages_empty",
    )


# ---------------------------------------------------------------------------
# Core Pipeline Execution
# ---------------------------------------------------------------------------

def process_thread_record(
    thread_obj: dict,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Process a single thread dict from JSONL.
    Returns (candidate_dict, exclusion_reason).
    """
    messages = thread_obj.get("messages", [])
    sel = select_first_customer_message(messages)

    if sel.excluded or not sel.message:
        return None, sel.exclusion_reason

    msg = sel.message
    raw_text = str(msg.get("text", "") or "").strip()
    label_res = classify_customer_message(raw_text)

    candidate_dict = {
        "thread_id": thread_obj.get("thread_id", ""),
        "tweet_id": str(msg.get("tweet_id", "")),
        "timestamp": str(msg.get("created_at", "")),
        "text": raw_text,
        "candidate_intent": label_res.intent,
        "label_source": label_res.source,
        "label_reason": label_res.reason,
        "label_confidence": label_res.confidence,
        "selection_reason": sel.reason,
        "thread_message_count": int(thread_obj.get("message_count", len(messages))),
        "customer_message_count": int(thread_obj.get("customer_message_count", 0)),
        "brand_message_count": int(thread_obj.get("brand_message_count", 0)),
        "has_missing_parent_link": bool(thread_obj.get("has_broken_links", False)),
        "has_missing_response_target": bool(
            thread_obj.get("link_metrics", {}).get("response_links_missing", 0) > 0
        ),
        "is_complete": bool(thread_obj.get("is_complete", False)),
        "ends_with_brand_reply": bool(thread_obj.get("ends_with_brand_reply", False)),
        "taxonomy_version": TAXONOMY_VERSION,
    }

    return candidate_dict, None


def run_intent_preparation(
    input_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    strict: bool = False,
) -> dict:
    """
    Execute full candidate extraction and preliminary intent classification.
    Produces CSV and metadata JSON.
    """
    t0 = time.time()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_csv = output_dir / OUTPUT_CSV_NAME
    output_meta = output_dir / OUTPUT_META_NAME

    if not overwrite:
        for p in (output_csv, output_meta):
            if p.exists():
                raise FileExistsError(
                    f"Output file already exists (use --overwrite): {p}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading threads from %s …", input_path)
    candidates: list[dict] = []
    exclusions: dict[str, int] = {}
    total_threads = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            total_threads += 1
            thread_obj = json.loads(line_str)
            cand, exc = process_thread_record(thread_obj)
            if cand:
                candidates.append(cand)
            else:
                ex_key = exc or "unknown_exclusion"
                exclusions[ex_key] = exclusions.get(ex_key, 0) + 1

    elapsed_extract = time.time() - t0
    log.info(
        "Processed %d threads | %d candidates | %d excluded | %.1fs",
        total_threads,
        len(candidates),
        sum(exclusions.values()),
        elapsed_extract,
    )

    # Sort deterministically by numeric thread root ID / tweet ID
    def _cand_sort_key(c: dict) -> tuple:
        tid = c["tweet_id"]
        try:
            return (0, int(tid))
        except ValueError:
            return (1, tid)

    candidates.sort(key=_cand_sort_key)

    # Check uniqueness
    tweet_ids = [c["tweet_id"] for c in candidates]
    thread_ids = [c["thread_id"] for c in candidates]
    dup_tweets = len(tweet_ids) - len(set(tweet_ids))
    dup_threads = len(thread_ids) - len(set(thread_ids))

    if dup_tweets > 0 or dup_threads > 0:
        msg = f"Duplicate detected: {dup_tweets} duplicate tweet_ids, {dup_threads} duplicate thread_ids"
        if strict:
            raise RuntimeError(msg)
        log.warning(msg)

    # Write CSV
    log.info("Writing CSV to %s …", output_csv)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(candidates)

    # Aggregate counts
    intent_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    conf_counts: dict[str, int] = {}

    for c in candidates:
        intent = c["candidate_intent"]
        src = c["label_source"]
        conf = c["label_confidence"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        source_counts[src] = source_counts.get(src, 0) + 1
        conf_counts[conf] = conf_counts.get(conf, 0) + 1

    # Helpers for relative paths
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    metadata = {
        "taxonomy_version": TAXONOMY_VERSION,
        "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dataset": _rel(input_path),
        "output_candidates_csv": _rel(output_csv),
        "output_metadata_json": _rel(output_meta),
        "total_threads_scanned": total_threads,
        "eligible_threads_count": len(candidates),
        "excluded_threads_count": sum(exclusions.values()),
        "exclusion_reasons": exclusions,
        "candidate_messages_count": len(candidates),
        "duplicate_candidate_tweet_ids": dup_tweets,
        "duplicate_candidate_thread_ids": dup_threads,
        "counts_by_candidate_intent": intent_counts,
        "counts_by_label_source": source_counts,
        "counts_by_confidence": conf_counts,
        "needs_review_count": intent_counts.get("needs_review", 0),
        "limitations": [
            "Candidate selection prefers the earliest chronological customer message; "
            "later turns with additional clarifications are preserved in the original thread.",
            "Labels are preliminary suggestions generated by deterministic lexical rules; "
            "they are NOT human-verified ground truth.",
            "ends_with_brand_reply and is_complete are preserved strictly as thread context "
            "and are NEVER used as intent or resolution labels.",
            "AppleSupport reply text is never inspected during classification to prevent data leakage.",
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info("Metadata written: %s", output_meta)
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5: Prepare customer-intent classification dataset from reconstructed threads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="PATH",
        help="Path to reconstructed threads JSONL",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        metavar="DIR",
        help="Output directory (default: data/processed/apple_support)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings as errors (exit 1)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        meta = run_intent_preparation(
            input_path=args.input,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            strict=args.strict,
        )
    except FileNotFoundError as exc:
        log.error("Input not found: %s", exc)
        sys.exit(1)
    except FileExistsError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    print()
    print("  ── Intent Dataset Preparation Complete ────────────────")
    print(f"  Taxonomy Version:        {meta['taxonomy_version']}")
    print(f"  Input threads:           {meta['total_threads_scanned']:,}")
    print(f"  Candidate messages:      {meta['candidate_messages_count']:,}")
    print(f"  Excluded threads:        {meta['excluded_threads_count']:,}")
    print(f"  Needs Review count:      {meta['needs_review_count']:,}")
    print(f"  Output CSV:              {meta['output_candidates_csv']}")
    print(f"  Metadata:                {meta['output_metadata_json']}")
    print(f"  Elapsed:                 {meta['elapsed_seconds']}s")
    print()


if __name__ == "__main__":
    main()
