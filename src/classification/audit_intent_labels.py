"""
src/classification/audit_intent_labels.py
=========================================
Phase 6 — Preliminary Intent Label Quality Audit and Human-Review Preparation.

PURPOSE
-------
Conducts a deterministic, reproducible quality audit of the preliminary intent
labels produced in Phase 5 and constructs a stratified, prioritized review
dataset for human annotation.

DESIGN PRINCIPLES
-----------------
1. Determinism:
   Fixed random seed (default 42) and explicit canonical pre-sorting by tweet_id
   guarantee 100% bitwise reproducible sample generation across environments.
2. Stratification:
   Multi-slice sampling guarantees coverage across all 11 intent domains,
   unknown_other, needs_review, all confidence levels, rule sources, rule
   overlaps, and dedicated high-risk cohorts (short text, URLs, mentions, emojis).
3. Review Prioritization:
   Assigns 'critical', 'high', 'medium', and 'normal' priority tiers to focus
   human annotation resources on high-uncertainty and high-impact edge cases.
4. Leakage Protection:
   Brand response text and future turns are never accessed or added.
5. Immutability:
   The original Phase 5 candidates CSV is strictly read-only and never modified.

OUTPUT ARTIFACTS
----------------
- data/processed/apple_support/apple_support_intent_review_sample.csv
- data/processed/apple_support/apple_support_intent_review_metadata.json

CLI USAGE
---------
.venv/bin/python -m src.classification.audit_intent_labels \\
    --input data/processed/apple_support/apple_support_intent_candidates.csv \\
    --output-dir data/processed/apple_support \\
    --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.classification.prepare_intent_dataset import (
    INTENT_PATTERNS,
    TAXONOMY_VERSION,
    VALID_CONFIDENCES,
    VALID_INTENTS,
    VALID_LABEL_SOURCES,
    normalize_for_matching,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("  %(levelname)s  %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_support"
    / "apple_support_intent_candidates.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "apple_support"
DEFAULT_OUTPUT_SAMPLE_CSV = "apple_support_intent_review_sample.csv"
DEFAULT_OUTPUT_METADATA_JSON = "apple_support_intent_review_metadata.json"
DEFAULT_OUTPUT_PILOT_CSV = "apple_support_intent_pilot_sample.csv"
DEFAULT_OUTPUT_PILOT_METADATA_JSON = "apple_support_intent_pilot_metadata.json"
DEFAULT_RANDOM_SEED = 42
DEFAULT_PILOT_SIZE = 150

# Column schema for the human-review output template
REVIEW_SAMPLE_COLUMNS = [
    "thread_id",
    "tweet_id",
    "created_at",
    "timestamp",
    "text",
    "cleaned_text",
    "preliminary_intent",
    "candidate_intent",
    "preliminary_confidence",
    "label_confidence",
    "preliminary_rule_id",
    "label_reason",
    "preliminary_label_source",
    "label_source",
    "matched_intents",
    "match_count",
    "rule_overlap_flag",
    "review_priority",
    "verified_intent",
    "verified_by",
    "verification_date",
    "verification_status",
    "notes",
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

# Regex patterns for high-risk slices
_URL_RE = re.compile(r"https?://\S+|t\.co/\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", re.IGNORECASE)
_EMOJI_OR_SPECIAL_RE = re.compile(r"[^\x00-\x7F]")
_AMBIGUOUS_KW_RE = re.compile(
    r"\b(update|app|apps|screen|battery|charge|apple|card|store|service)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Audit Feature Extraction
# ---------------------------------------------------------------------------

def compute_row_audit_features(raw_text: str) -> tuple[str, str, int, bool, dict[str, int]]:
    """
    Compute normalized text, matched categories, match count, overlap flag,
    and category match scores for a single customer text.
    """
    cleaned = normalize_for_matching(raw_text)
    scores: dict[str, int] = {}
    matched: list[str] = []

    for intent_name, patterns in INTENT_PATTERNS.items():
        intent_score = 0
        for pattern, weight, _ in patterns:
            if pattern.search(cleaned):
                intent_score += weight
        if intent_score > 0:
            scores[intent_name] = intent_score
            matched.append(intent_name)

    matched.sort()
    matched_intents_str = "|".join(matched)
    match_count = len(matched)
    overlap_flag = match_count > 1

    return cleaned, matched_intents_str, match_count, overlap_flag, scores


def assign_review_priority(
    intent: str,
    confidence: str,
    overlap_flag: bool,
    match_count: int,
    scores: dict[str, int],
) -> str:
    """
    Assign human review priority based on uncertainty and impact:
      - 'critical': rule overlap, ambiguous high-confidence prediction, or competing rules.
      - 'high': software_update, unknown_other, or needs_review.
      - 'medium': low-confidence predictions.
      - 'normal': other standard predictions.
    """
    # Critical: any rule overlap, multi-category match, or high-confidence with competing signal
    if overlap_flag or match_count > 1:
        return "critical"

    if confidence == "high" and len(scores) > 1:
        return "critical"

    # High: large volume or high ambiguity classes
    if intent in ("software_update", "unknown_other", "needs_review"):
        return "high"

    # Medium: low or moderate confidence predictions
    if confidence in ("low", "medium"):
        return "medium"

    # Normal: all others (clean high-confidence single-category predictions)
    return "normal"


# ---------------------------------------------------------------------------
# Deterministic Stratified Sampling
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    seed: int = DEFAULT_RANDOM_SEED
    sample_size_per_intent: int = 50
    sample_size_per_confidence: int = 100
    sample_size_per_source: int = 100
    sample_size_overlap: int = 150
    sample_size_high_risk: int = 100
    sample_size_edge_cases: int = 50


def sample_indices_from_slice(
    candidate_indices: list[int],
    target_count: int,
    rng: random.Random,
) -> list[int]:
    """Deterministically sample up to target_count indices from a candidate pool."""
    if len(candidate_indices) <= target_count:
        return list(candidate_indices)
    return rng.sample(candidate_indices, target_count)


def build_stratified_audit_sample(
    df: pd.DataFrame,
    config: SamplingConfig,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Create a deterministic stratified sample covering:
      - All preliminary intents
      - All confidence levels
      - All label sources
      - Overlap cases
      - High-risk cohorts (software_update, unknown_other, needs_review)
      - Edge-case slices (short messages, URLs, mentions, emojis, ambiguous words)

    Returns: (sampled_df, requested_sizes, actual_sizes)
    """
    # Deterministic sorting before any sampling
    # Sort by tweet_id numerically if possible, else string
    try:
        df["_sort_key"] = df["tweet_id"].astype(int)
    except Exception:
        df["_sort_key"] = df["tweet_id"].astype(str)

    sorted_df = df.sort_values(by="_sort_key").reset_index(drop=True)
    sorted_df = sorted_df.drop(columns=["_sort_key"])

    rng = random.Random(config.seed)
    requested_sizes: dict[str, int] = {}
    actual_sizes: dict[str, int] = {}
    selected_indices: set[int] = set()

    # -------------------------------------------------------------------------
    # Stratum A: Every Preliminary Intent
    # -------------------------------------------------------------------------
    all_intents = sorted(sorted_df["candidate_intent"].unique())
    for intent in all_intents:
        slice_name = f"intent_{intent}"
        pool = sorted_df[sorted_df["candidate_intent"] == intent].index.tolist()
        req = config.sample_size_per_intent
        sampled = sample_indices_from_slice(pool, req, rng)
        requested_sizes[slice_name] = req
        actual_sizes[slice_name] = len(sampled)
        selected_indices.update(sampled)

    # -------------------------------------------------------------------------
    # Stratum B: Every Confidence Level
    # -------------------------------------------------------------------------
    for conf in ("high", "medium", "low"):
        slice_name = f"confidence_{conf}"
        pool = sorted_df[sorted_df["label_confidence"] == conf].index.tolist()
        req = config.sample_size_per_confidence
        sampled = sample_indices_from_slice(pool, req, rng)
        requested_sizes[slice_name] = req
        actual_sizes[slice_name] = len(sampled)
        selected_indices.update(sampled)

    # -------------------------------------------------------------------------
    # Stratum C: Every Label Source
    # -------------------------------------------------------------------------
    for src in ("rule", "fallback", "needs_review"):
        slice_name = f"source_{src}"
        pool = sorted_df[sorted_df["label_source"] == src].index.tolist()
        req = config.sample_size_per_source
        sampled = sample_indices_from_slice(pool, req, rng)
        requested_sizes[slice_name] = req
        actual_sizes[slice_name] = len(sampled)
        selected_indices.update(sampled)

    # -------------------------------------------------------------------------
    # Stratum D: Rule Overlap Cases
    # -------------------------------------------------------------------------
    overlap_pool = sorted_df[sorted_df["rule_overlap_flag"] == True].index.tolist()
    req_ov = config.sample_size_overlap
    sampled_ov = sample_indices_from_slice(overlap_pool, req_ov, rng)
    requested_sizes["stratum_rule_overlap"] = req_ov
    actual_sizes["stratum_rule_overlap"] = len(sampled_ov)
    selected_indices.update(sampled_ov)

    # -------------------------------------------------------------------------
    # Stratum E: Dedicated High-Risk & Edge-Case Cohorts
    # -------------------------------------------------------------------------
    high_risk_slices: list[tuple[str, pd.Series, int]] = [
        (
            "high_risk_software_update",
            sorted_df["candidate_intent"] == "software_update",
            config.sample_size_high_risk,
        ),
        (
            "high_risk_unknown_other",
            sorted_df["candidate_intent"] == "unknown_other",
            config.sample_size_high_risk,
        ),
        (
            "high_risk_needs_review",
            sorted_df["candidate_intent"] == "needs_review",
            config.sample_size_high_risk,
        ),
        (
            "edge_case_short_message",
            sorted_df["cleaned_text"].apply(lambda t: len(str(t).split()) <= 5),
            config.sample_size_edge_cases,
        ),
        (
            "edge_case_has_url",
            sorted_df["text"].apply(lambda t: bool(_URL_RE.search(str(t)))),
            config.sample_size_edge_cases,
        ),
        (
            "edge_case_has_mention",
            sorted_df["text"].apply(lambda t: bool(_MENTION_RE.search(str(t)))),
            config.sample_size_edge_cases,
        ),
        (
            "edge_case_has_emoji_or_special",
            sorted_df["text"].apply(lambda t: bool(_EMOJI_OR_SPECIAL_RE.search(str(t)))),
            config.sample_size_edge_cases,
        ),
        (
            "edge_case_ambiguous_keywords",
            sorted_df["cleaned_text"].apply(lambda t: bool(_AMBIGUOUS_KW_RE.search(str(t)))),
            config.sample_size_edge_cases,
        ),
    ]

    for slice_name, mask, req_count in high_risk_slices:
        pool = sorted_df[mask].index.tolist()
        sampled = sample_indices_from_slice(pool, req_count, rng)
        requested_sizes[slice_name] = req_count
        actual_sizes[slice_name] = len(sampled)
        selected_indices.update(sampled)

    # Union all unique indices and sort canonically
    final_indices = sorted(selected_indices)
    sampled_df = sorted_df.iloc[final_indices].copy().reset_index(drop=True)

    return sampled_df, requested_sizes, actual_sizes


def build_pilot_sample(
    df: pd.DataFrame,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Construct a compact, deterministic annotation pilot sample (~150 records).
    Covers all 11 taxonomy domains + unknown_other + needs_review, all confidence
    tiers, rule overlaps, and high-risk edge cases (update collisions, short messages,
    vague complaints, URLs).
    """
    try:
        df["_sort_key"] = df["tweet_id"].astype(int)
    except Exception:
        df["_sort_key"] = df["tweet_id"].astype(str)

    sorted_df = df.sort_values(by="_sort_key").reset_index(drop=True)
    sorted_df = sorted_df.drop(columns=["_sort_key"])

    rng = random.Random(seed)
    requested_sizes: dict[str, int] = {}
    actual_sizes: dict[str, int] = {}
    selected_indices: set[int] = set()

    # Stratum A: 7 per intent across all candidates (12 * 7 = 84 target slots)
    for intent in sorted(sorted_df["candidate_intent"].unique()):
        slice_name = f"pilot_intent_{intent}"
        pool = sorted_df[sorted_df["candidate_intent"] == intent].index.tolist()
        req = 7
        sampled = sample_indices_from_slice(pool, req, rng)
        requested_sizes[slice_name] = req
        actual_sizes[slice_name] = len(sampled)
        selected_indices.update(sampled)

    # Stratum B: 20 rule overlap rows
    ov_pool = sorted_df[sorted_df["rule_overlap_flag"] == True].index.tolist()
    sampled_ov = sample_indices_from_slice(ov_pool, 20, rng)
    requested_sizes["pilot_rule_overlap"] = 20
    actual_sizes["pilot_rule_overlap"] = len(sampled_ov)
    selected_indices.update(sampled_ov)

    # Stratum C: software_update false-positive edge cases (cards, payment, order, app store downloads)
    su_mask = (sorted_df["candidate_intent"] == "software_update") & (
        sorted_df["text"].str.contains(r"\b(?:card|payment|order|shipping|app store|download)\b", case=False, na=False)
    )
    su_pool = sorted_df[su_mask].index.tolist()
    sampled_su = sample_indices_from_slice(su_pool, 12, rng)
    requested_sizes["pilot_software_update_edge_cases"] = 12
    actual_sizes["pilot_software_update_edge_cases"] = len(sampled_su)
    selected_indices.update(sampled_su)

    # Stratum D: unknown_other edge cases (conversational, vague, store)
    uo_mask = (sorted_df["candidate_intent"] == "unknown_other") & (
        sorted_df["text"].str.contains(r"\b(?:dm|help|store|genius|macbook|broken|glitch)\b", case=False, na=False)
    )
    uo_pool = sorted_df[uo_mask].index.tolist()
    sampled_uo = sample_indices_from_slice(uo_pool, 12, rng)
    requested_sizes["pilot_unknown_other_edge_cases"] = 12
    actual_sizes["pilot_unknown_other_edge_cases"] = len(sampled_uo)
    selected_indices.update(sampled_uo)

    # Stratum E: needs_review top conflict pairs
    nr_pool = sorted_df[sorted_df["candidate_intent"] == "needs_review"].index.tolist()
    sampled_nr = sample_indices_from_slice(nr_pool, 12, rng)
    requested_sizes["pilot_needs_review_conflicts"] = 12
    actual_sizes["pilot_needs_review_conflicts"] = len(sampled_nr)
    selected_indices.update(sampled_nr)

    # Stratum F: short / noisy text (<= 4 words)
    short_mask = sorted_df["cleaned_text"].apply(lambda x: len(str(x).split()) <= 4)
    short_pool = sorted_df[short_mask].index.tolist()
    sampled_short = sample_indices_from_slice(short_pool, 10, rng)
    requested_sizes["pilot_short_noisy"] = 10
    actual_sizes["pilot_short_noisy"] = len(sampled_short)
    selected_indices.update(sampled_short)

    # Stratum G: URL / media tweets
    url_mask = sorted_df["text"].str.contains(r"https?://|t\.co", case=False, na=False)
    url_pool = sorted_df[url_mask].index.tolist()
    sampled_url = sample_indices_from_slice(url_pool, 8, rng)
    requested_sizes["pilot_url_media"] = 8
    actual_sizes["pilot_url_media"] = len(sampled_url)
    selected_indices.update(sampled_url)

    final_indices = sorted(selected_indices)
    pilot_df = sorted_df.iloc[final_indices].copy().reset_index(drop=True)
    return pilot_df, requested_sizes, actual_sizes


# ---------------------------------------------------------------------------
# Formatting & Export Pipeline
# ---------------------------------------------------------------------------

def enrich_and_format_sample_df(sampled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all review columns, blank verification placeholders,
    and priority tiers are populated.
    """
    # Populate aliases
    sampled_df["preliminary_intent"] = sampled_df["candidate_intent"]
    sampled_df["preliminary_confidence"] = sampled_df["label_confidence"]
    sampled_df["preliminary_rule_id"] = sampled_df["label_reason"]
    sampled_df["preliminary_label_source"] = sampled_df["label_source"]
    sampled_df["created_at"] = sampled_df["timestamp"]

    # Initialize human review fields
    sampled_df["verified_intent"] = ""
    sampled_df["verified_by"] = ""
    sampled_df["verification_date"] = ""
    sampled_df["verification_status"] = "unreviewed"
    sampled_df["notes"] = ""

    # Reorder columns to match canonical review schema
    output_cols = [c for c in REVIEW_SAMPLE_COLUMNS if c in sampled_df.columns]
    for c in REVIEW_SAMPLE_COLUMNS:
        if c not in output_cols:
            sampled_df[c] = ""
            output_cols.append(c)

    return sampled_df[output_cols]


def compute_content_sha256(csv_path: Path) -> str:
    """Compute SHA-256 hash of the generated review CSV for exact verification."""
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_audit_sampling(
    input_csv: Path,
    output_dir: Path,
    sample_csv_name: str = DEFAULT_OUTPUT_SAMPLE_CSV,
    metadata_name: str = DEFAULT_OUTPUT_METADATA_JSON,
    seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
    config: Optional[SamplingConfig] = None,
) -> dict[str, Any]:
    """
    Execute end-to-end audit sample generation and metadata export.
    Returns metadata dict.
    """
    t0 = time.time()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input candidate CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / sample_csv_name
    out_meta = output_dir / metadata_name

    if not overwrite:
        for p in (out_csv, out_meta):
            if p.exists():
                raise FileExistsError(
                    f"Output file already exists (use --overwrite): {p}"
                )

    if config is None:
        config = SamplingConfig(seed=seed)
    else:
        config.seed = seed

    log.info("Loading candidate dataset from %s …", input_csv)
    df = pd.read_csv(input_csv, dtype=str)
    total_candidates = len(df)
    log.info("Loaded %d candidate records.", total_candidates)

    # Pre-compute audit features for all candidates
    log.info("Computing text normalization and pattern match audit features …")
    cleaned_texts = []
    matched_intents = []
    match_counts = []
    overlap_flags = []
    priorities = []

    for _, row in df.iterrows():
        raw_text = str(row.get("text", "") or "")
        intent = str(row.get("candidate_intent", "") or "")
        confidence = str(row.get("label_confidence", "") or "")
        cleaned, m_str, m_cnt, ov_flag, scores = compute_row_audit_features(raw_text)

        priority = assign_review_priority(
            intent=intent,
            confidence=confidence,
            overlap_flag=ov_flag,
            match_count=m_cnt,
            scores=scores,
        )

        cleaned_texts.append(cleaned)
        matched_intents.append(m_str)
        match_counts.append(m_cnt)
        overlap_flags.append(ov_flag)
        priorities.append(priority)

    df["cleaned_text"] = cleaned_texts
    df["matched_intents"] = matched_intents
    df["match_count"] = match_counts
    df["rule_overlap_flag"] = overlap_flags
    df["review_priority"] = priorities

    # Build stratified sample
    log.info("Extracting stratified audit samples with seed=%d …", config.seed)
    sampled_df, req_sizes, act_sizes = build_stratified_audit_sample(df, config)
    final_df = enrich_and_format_sample_df(sampled_df)

    log.info("Writing review sample to %s (%d rows) …", out_csv, len(final_df))
    final_df.to_csv(out_csv, index=False, encoding="utf-8")

    # Compute deterministic SHA-256 of output CSV
    content_hash = compute_content_sha256(out_csv)

    # Collect metadata
    intent_counts = final_df["candidate_intent"].value_counts().to_dict()
    conf_counts = final_df["label_confidence"].value_counts().to_dict()
    source_counts = final_df["label_source"].value_counts().to_dict()
    priority_counts = final_df["review_priority"].value_counts().to_dict()
    overlap_counts = {
        "overlap": int((final_df["rule_overlap_flag"] == True).sum()),
        "no_overlap": int((final_df["rule_overlap_flag"] == False).sum()),
    }

    elapsed = round(time.time() - t0, 2)

    metadata: dict[str, Any] = {
        "taxonomy_version": TAXONOMY_VERSION,
        "source_candidate_path": str(input_csv.relative_to(PROJECT_ROOT))
        if input_csv.is_relative_to(PROJECT_ROOT)
        else str(input_csv),
        "output_sample_path": str(out_csv.relative_to(PROJECT_ROOT))
        if out_csv.is_relative_to(PROJECT_ROOT)
        else str(out_csv),
        "output_metadata_path": str(out_meta.relative_to(PROJECT_ROOT))
        if out_meta.is_relative_to(PROJECT_ROOT)
        else str(out_meta),
        "random_seed": config.seed,
        "sampling_strategy": "deterministic_stratified_multi_slice",
        "requested_sample_sizes": req_sizes,
        "actual_sample_sizes": act_sizes,
        "total_candidates_scanned": total_candidates,
        "total_review_samples": len(final_df),
        "unique_tweet_ids": int(final_df["tweet_id"].nunique()),
        "unique_thread_ids": int(final_df["thread_id"].nunique()),
        "counts_by_preliminary_intent": intent_counts,
        "counts_by_confidence": conf_counts,
        "counts_by_label_source": source_counts,
        "counts_by_review_priority": priority_counts,
        "counts_by_overlap_status": overlap_counts,
        "deterministic_content_sha256": content_hash,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Review sample is designed for quality auditing and active learning; it is NOT an unstratified natural distribution test set.",
            "Verified labels must be assigned by human reviewers in verified_intent.",
            "Brand reply text is never inspected, preserving strict zero data leakage.",
            "Customer tweets reflect Twitter microblogging constraints and 2017 temporal context.",
        ],
        "elapsed_seconds": elapsed,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("Metadata saved to %s in %.1fs", out_meta, elapsed)
    return metadata


def run_pilot_sampling(
    input_csv: Path,
    output_dir: Path,
    pilot_csv_name: str = DEFAULT_OUTPUT_PILOT_CSV,
    metadata_name: str = DEFAULT_OUTPUT_PILOT_METADATA_JSON,
    seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Execute end-to-end pilot sample generation and metadata export (~150 records).
    Returns metadata dict.
    """
    t0 = time.time()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input candidate CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / pilot_csv_name
    out_meta = output_dir / metadata_name

    if not overwrite:
        for p in (out_csv, out_meta):
            if p.exists():
                raise FileExistsError(
                    f"Output file already exists (use --overwrite): {p}"
                )

    log.info("Loading candidate dataset for pilot from %s …", input_csv)
    df = pd.read_csv(input_csv, dtype=str)
    total_candidates = len(df)

    log.info("Computing audit features for pilot …")
    cleaned_texts = []
    matched_intents = []
    match_counts = []
    overlap_flags = []
    priorities = []

    for _, row in df.iterrows():
        raw_text = str(row.get("text", "") or "")
        intent = str(row.get("candidate_intent", "") or "")
        confidence = str(row.get("label_confidence", "") or "")
        cleaned, m_str, m_cnt, ov_flag, scores = compute_row_audit_features(raw_text)

        priority = assign_review_priority(
            intent=intent,
            confidence=confidence,
            overlap_flag=ov_flag,
            match_count=m_cnt,
            scores=scores,
        )

        cleaned_texts.append(cleaned)
        matched_intents.append(m_str)
        match_counts.append(m_cnt)
        overlap_flags.append(ov_flag)
        priorities.append(priority)

    df["cleaned_text"] = cleaned_texts
    df["matched_intents"] = matched_intents
    df["match_count"] = match_counts
    df["rule_overlap_flag"] = overlap_flags
    df["review_priority"] = priorities

    log.info("Extracting pilot sample (~150 records) with seed=%d …", seed)
    pilot_df, req_sizes, act_sizes = build_pilot_sample(df, seed=seed)
    final_df = enrich_and_format_sample_df(pilot_df)

    log.info("Writing pilot sample to %s (%d rows) …", out_csv, len(final_df))
    final_df.to_csv(out_csv, index=False, encoding="utf-8")

    content_hash = compute_content_sha256(out_csv)

    intent_counts = final_df["candidate_intent"].value_counts().to_dict()
    conf_counts = final_df["label_confidence"].value_counts().to_dict()
    source_counts = final_df["label_source"].value_counts().to_dict()
    priority_counts = final_df["review_priority"].value_counts().to_dict()
    overlap_counts = {
        "overlap": int((final_df["rule_overlap_flag"] == True).sum()),
        "no_overlap": int((final_df["rule_overlap_flag"] == False).sum()),
    }

    elapsed = round(time.time() - t0, 2)

    metadata: dict[str, Any] = {
        "taxonomy_version": TAXONOMY_VERSION,
        "source_candidate_path": str(input_csv.relative_to(PROJECT_ROOT))
        if input_csv.is_relative_to(PROJECT_ROOT)
        else str(input_csv),
        "output_pilot_path": str(out_csv.relative_to(PROJECT_ROOT))
        if out_csv.is_relative_to(PROJECT_ROOT)
        else str(out_csv),
        "output_metadata_path": str(out_meta.relative_to(PROJECT_ROOT))
        if out_meta.is_relative_to(PROJECT_ROOT)
        else str(out_meta),
        "random_seed": seed,
        "sampling_strategy": "deterministic_stratified_pilot",
        "requested_sample_sizes": req_sizes,
        "actual_sample_sizes": act_sizes,
        "total_candidates_scanned": total_candidates,
        "total_pilot_samples": len(final_df),
        "unique_tweet_ids": int(final_df["tweet_id"].nunique()),
        "unique_thread_ids": int(final_df["thread_id"].nunique()),
        "counts_by_preliminary_intent": intent_counts,
        "counts_by_confidence": conf_counts,
        "counts_by_label_source": source_counts,
        "counts_by_review_priority": priority_counts,
        "counts_by_overlap_status": overlap_counts,
        "deterministic_content_sha256": content_hash,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Pilot sample is designed for annotator calibration and guideline testing; it is NOT an unstratified natural distribution test set.",
            "Verified labels must be assigned by human reviewers in verified_intent without fabricating ground truth.",
            "Brand reply text is never inspected, preserving strict zero data leakage.",
            "Customer tweets reflect Twitter microblogging constraints and late 2017 temporal context.",
        ],
        "elapsed_seconds": elapsed,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("Pilot metadata saved to %s in %.1fs", out_meta, elapsed)
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 — Audit preliminary intent labels and prepare human-review samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Path to candidate CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-csv-name",
        type=str,
        default=DEFAULT_OUTPUT_SAMPLE_CSV,
        help=f"Sample CSV filename (default: {DEFAULT_OUTPUT_SAMPLE_CSV})",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default=DEFAULT_OUTPUT_METADATA_JSON,
        help=f"Metadata JSON filename (default: {DEFAULT_OUTPUT_METADATA_JSON})",
    )
    parser.add_argument(
        "--pilot-csv-name",
        type=str,
        default=DEFAULT_OUTPUT_PILOT_CSV,
        help=f"Pilot CSV filename (default: {DEFAULT_OUTPUT_PILOT_CSV})",
    )
    parser.add_argument(
        "--pilot-metadata-name",
        type=str,
        default=DEFAULT_OUTPUT_PILOT_METADATA_JSON,
        help=f"Pilot metadata JSON filename (default: {DEFAULT_OUTPUT_PILOT_METADATA_JSON})",
    )
    parser.add_argument(
        "--generate-pilot",
        action="store_true",
        help="Also generate the 150-record pilot annotation dataset",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Only generate the 150-record pilot annotation dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for deterministic sampling (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational log messages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quiet:
        log.setLevel(logging.WARNING)

    try:
        if not args.pilot_only:
            run_audit_sampling(
                input_csv=args.input,
                output_dir=args.output_dir,
                sample_csv_name=args.sample_csv_name,
                metadata_name=args.metadata_name,
                seed=args.seed,
                overwrite=args.overwrite,
            )

        if args.generate_pilot or args.pilot_only:
            run_pilot_sampling(
                input_csv=args.input,
                output_dir=args.output_dir,
                pilot_csv_name=args.pilot_csv_name,
                metadata_name=args.pilot_metadata_name,
                seed=args.seed,
                overwrite=args.overwrite,
            )

        return 0
    except Exception as err:
        log.error("Audit/pilot sampling failed: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
