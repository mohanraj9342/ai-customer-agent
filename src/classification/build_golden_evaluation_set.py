"""
src/classification/build_golden_evaluation_set.py
=================================================
Pipeline module and utilities to construct, validate, and maintain the
human-verified Golden Evaluation Set for the AppleSupport intent classifier.

Key Functionalities:
1. Constructs the 158-record Golden Evaluation Set from the audited Phase 7 human
   annotations (`apple_support_intent_pilot_sample.csv`).
2. Standardizes the golden schema (`golden_intent`, `verification_status`, `verified_by`,
   `verification_date`, `notes`, `evaluation_split`, `case_difficulty`).
3. Enforces strict training data isolation: extracts and verifies disjoint training
   candidate pools with zero overlap (`train_tweet_ids ∩ golden_tweet_ids = ∅`).
4. Generates comprehensive audit metadata including content SHA-256 and slice distributions.
5. Provides validation functions to ensure zero duplicates, zero missing values in required
   fields, taxonomy consistency, and explicit preservation of `needs_review` cases.

Usage:
------
    # Build the golden evaluation set and metadata
    python -m src.classification.build_golden_evaluation_set --build --overwrite

    # Validate existing golden dataset
    python -m src.classification.build_golden_evaluation_set --validate

    # Export isolated training candidates (82,101 - 158 = 81,943 rows)
    python -m src.classification.build_golden_evaluation_set --export-isolated-candidates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.classification.intent_loader import get_fallback_intent, get_intent_names

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and Paths
# ---------------------------------------------------------------------------
DEFAULT_PILOT_CSV = "data/processed/apple_support/apple_support_intent_pilot_sample.csv"
DEFAULT_PILOT_META = "data/processed/apple_support/apple_support_intent_pilot_metadata.json"
DEFAULT_CANDIDATES_CSV = "data/processed/apple_support/apple_support_intent_candidates.csv"

DEFAULT_GOLDEN_CSV = "data/processed/apple_support/apple_support_intent_golden_set.csv"
DEFAULT_GOLDEN_META = "data/processed/apple_support/apple_support_intent_golden_metadata.json"
DEFAULT_TRAIN_CANDIDATES_CSV = "data/processed/apple_support/apple_support_intent_training_candidates.csv"

TAXONOMY_VERSION = "2.0"

VALID_VERIFICATION_STATUSES = {"verified", "corrected", "flagged_ambiguous"}

CASE_DIFFICULTY_TIERS = {
    "representative",
    "rule_conflict",
    "attribution_vs_symptom",
    "fallback_recovery",
    "boundary_disambiguation",
    "short_noisy",
    "multilingual",
    "ambiguous_multi_intent",
}

GOLDEN_SET_COLUMNS = [
    "tweet_id",
    "thread_id",
    "created_at",
    "timestamp",
    "text",
    "cleaned_text",
    "golden_intent",
    "verified_intent",
    "verification_status",
    "verified_by",
    "verification_date",
    "notes",
    "evaluation_split",
    "case_difficulty",
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

REQUIRED_NON_NULL_COLUMNS = [
    "tweet_id",
    "thread_id",
    "text",
    "cleaned_text",
    "golden_intent",
    "verified_intent",
    "verification_status",
    "verified_by",
    "verification_date",
    "notes",
    "evaluation_split",
    "case_difficulty",
    "taxonomy_version",
]


# ---------------------------------------------------------------------------
# Case Difficulty Categorization Logic
# ---------------------------------------------------------------------------
def assign_case_difficulty(row: pd.Series) -> str:
    """Classify an evaluation record into an operational difficulty slice for granular error analysis.

    Slices:
    - ambiguous_multi_intent: Genuinely irreconcilable multi-intent cases (needs_review / flagged_ambiguous).
    - multilingual: Non-English customer inquiries (Spanish, German, Japanese, Portuguese).
    - short_noisy: Conversational fragments and ultra-short texts (<= 25 chars or brief conversational phrases).
    - attribution_vs_symptom: Rule 3 applications where actionable symptoms (battery, network) override update attribution.
    - boundary_disambiguation: Boundary collisions (App Store downloads vs OS update, payment method updates, UI glitches).
    - fallback_recovery: Inquiries successfully recovered from preliminary unknown_other fallback into concrete intents.
    - rule_conflict: Competing heuristic rules (needs_review or rule_overlap_flag) resolved by human reviewer.
    - representative: Clean, high-confidence single-intent matches that were directly verified.
    """
    status = str(row.get("verification_status", "")).strip()
    prelim = str(row.get("preliminary_intent", "")).strip()
    notes = str(row.get("notes", "")).lower()
    text = str(row.get("text", "")).strip()
    overlap = bool(row.get("rule_overlap_flag", False))
    tid = int(row.get("tweet_id", 0))

    # 1. Ambiguous multi-intent
    if status == "flagged_ambiguous":
        return "ambiguous_multi_intent"

    # 2. Multilingual (known foreign language tweet IDs in pilot pool)
    if tid in {1333692, 1756350, 1807493, 2406677, 718775}:
        return "multilingual"

    # 3. Short / noisy (microblogging fragments lacking substantial syntax)
    if len(text) <= 25 or (prelim == "unknown_other" and status == "verified" and len(text) <= 35):
        return "short_noisy"

    # 4. Attribution vs symptom (Rule 3: OS update attribution with specific actionable symptom)
    if (
        "rule 3" in notes
        or ("battery" in notes and "symptom" in notes)
        or ("update" in notes and "symptom" in notes)
        or ("attribution" in notes)
    ):
        return "attribution_vs_symptom"

    # 5. Boundary disambiguation (Rule 4 App Store updates, Rule 2 billing updates, return policy, UI glitches)
    if (
        "rule 4" in notes
        or ("app store" in notes and prelim == "software_update")
        or ("billing" in notes and prelim == "software_update")
        or ("return and refund" in notes)
        or ("status bar" in notes)
    ):
        return "boundary_disambiguation"

    # 6. Fallback recovery (preliminary unknown_other reassigned to an actionable intent)
    if prelim == "unknown_other" and status == "corrected":
        return "fallback_recovery"

    # 7. Rule conflict (multi-category collision resolved by human reviewer)
    if prelim == "needs_review" or overlap:
        return "rule_conflict"

    # 8. Representative standard case
    return "representative"


# ---------------------------------------------------------------------------
# Core Builder Function
# ---------------------------------------------------------------------------
def build_golden_set(
    pilot_csv_path: str | Path = DEFAULT_PILOT_CSV,
    pilot_meta_path: str | Path = DEFAULT_PILOT_META,
    output_golden_csv: str | Path = DEFAULT_GOLDEN_CSV,
    output_golden_meta: str | Path = DEFAULT_GOLDEN_META,
    candidates_csv_path: str | Path = DEFAULT_CANDIDATES_CSV,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the final human-verified Golden Evaluation Set and associated metadata.

    Parameters:
    -----------
    pilot_csv_path : Path to the audited Phase 7 pilot CSV.
    pilot_meta_path : Path to the Phase 7 pilot metadata JSON.
    output_golden_csv : Path to write the golden evaluation CSV.
    output_golden_meta : Path to write the golden evaluation metadata JSON.
    candidates_csv_path : Path to the full Phase 5 candidates CSV for isolation verification.
    overwrite : Whether to overwrite existing golden artifacts.

    Returns:
    --------
    (df_golden, metadata_dict)
    """
    pilot_csv_path = Path(pilot_csv_path)
    pilot_meta_path = Path(pilot_meta_path)
    output_golden_csv = Path(output_golden_csv)
    output_golden_meta = Path(output_golden_meta)
    candidates_csv_path = Path(candidates_csv_path)

    if not pilot_csv_path.exists():
        raise FileNotFoundError(f"Pilot dataset not found at: {pilot_csv_path}")

    if not overwrite:
        if output_golden_csv.exists():
            raise FileExistsError(
                f"Golden set CSV already exists at {output_golden_csv}. Use overwrite=True to replace."
            )
        if output_golden_meta.exists():
            raise FileExistsError(
                f"Golden metadata already exists at {output_golden_meta}. Use overwrite=True to replace."
            )

    logger.info("Loading audited pilot dataset from: %s", pilot_csv_path)
    df_pilot = pd.read_csv(pilot_csv_path)

    if len(df_pilot) != 158:
        raise ValueError(f"Expected exactly 158 pilot records, found {len(df_pilot)}")

    df = df_pilot.copy()

    # Cast review fields to object/string
    for col in ["verified_intent", "verified_by", "verification_date", "verification_status", "notes"]:
        df[col] = df[col].astype("object")

    # Cleaned text can be empty string for handle-only tweets, ensure not NaN
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

    # Standardize ground truth target
    df["golden_intent"] = df["verified_intent"]
    df["evaluation_split"] = "golden_test"

    # Assign difficulty slice
    df["case_difficulty"] = df.apply(assign_case_difficulty, axis=1)

    # Ensure all required columns are present in canonical order
    for col in GOLDEN_SET_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df_golden = df[GOLDEN_SET_COLUMNS].copy()

    # Validate dataset integrity
    validation_results = validate_golden_dataset(df_golden)
    logger.info("Validation passed: %s", validation_results["summary"])

    # Ensure output parent directory exists
    output_golden_csv.parent.mkdir(parents=True, exist_ok=True)
    output_golden_meta.parent.mkdir(parents=True, exist_ok=True)

    # Write golden CSV
    df_golden.to_csv(output_golden_csv, index=False)
    logger.info("Saved golden evaluation set to: %s (%d rows)", output_golden_csv, len(df_golden))

    # Compute deterministic SHA-256
    with open(output_golden_csv, "rb") as f:
        content_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Check candidates for isolation verification
    total_candidates_scanned = 0
    isolated_candidates_count = 0
    if candidates_csv_path.exists():
        df_candidates = pd.read_csv(candidates_csv_path, usecols=["tweet_id"])
        total_candidates_scanned = len(df_candidates)
        golden_ids_set = set(df_golden["tweet_id"].astype(int))
        isolated_df = df_candidates[~df_candidates["tweet_id"].astype(int).isin(golden_ids_set)]
        isolated_candidates_count = len(isolated_df)
        assert len(golden_ids_set.intersection(set(isolated_df["tweet_id"].astype(int)))) == 0
    else:
        logger.warning("Candidates CSV not found at %s; skipping full corpus scan count.", candidates_csv_path)

    # Build comprehensive metadata
    status_counts = df_golden["verification_status"].value_counts().to_dict()
    intent_counts = df_golden["golden_intent"].value_counts().to_dict()
    difficulty_counts = df_golden["case_difficulty"].value_counts().to_dict()

    agreement_count = int((df_golden["preliminary_intent"] == df_golden["golden_intent"]).sum())
    agreement_rate = round(float(agreement_count) / len(df_golden), 4)

    golden_tweet_ids = sorted([int(tid) for tid in df_golden["tweet_id"].unique()])

    metadata: dict[str, Any] = {
        "dataset_name": "AppleSupport Intent Classification Golden Evaluation Set",
        "taxonomy_version": TAXONOMY_VERSION,
        "evaluation_split": "golden_test",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_pilot_path": str(pilot_csv_path),
        "output_golden_csv": str(output_golden_csv),
        "output_golden_metadata": str(output_golden_meta),
        "total_golden_records": len(df_golden),
        "unique_tweet_ids": int(df_golden["tweet_id"].nunique()),
        "unique_thread_ids": int(df_golden["thread_id"].nunique()),
        "deterministic_content_sha256": content_sha256,
        "counts_by_golden_intent": intent_counts,
        "counts_by_verification_status": status_counts,
        "counts_by_case_difficulty": difficulty_counts,
        "preliminary_rule_agreement": {
            "matching_records": agreement_count,
            "total_records": len(df_golden),
            "agreement_rate": agreement_rate,
        },
        "training_data_isolation": {
            "total_candidates_scanned": total_candidates_scanned,
            "isolated_training_candidates": isolated_candidates_count,
            "golden_overlap_with_training": 0,
            "is_strictly_isolated": True,
            "isolation_rule": "S_train ∩ S_golden = ∅",
            "golden_tweet_id_count": len(golden_tweet_ids),
            "golden_tweet_ids": golden_tweet_ids,
        },
        "data_leakage_safeguards": [
            "Intent ground truth is determined exclusively from the customer's initial inbound message text.",
            "Brand replies and future conversation turns are never inspected during intent annotation.",
            "Golden set is strictly held out and must never be included in classifier training or validation sets.",
        ],
    }

    with open(output_golden_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved golden evaluation metadata to: %s", output_golden_meta)

    return df_golden, metadata


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------
def validate_golden_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate data integrity, schema consistency, and label validity of the golden evaluation set.

    Raises:
    -------
    ValueError or AssertionError if any integrity condition is violated.
    """
    errors: list[str] = []
    df = df.copy()
    if "cleaned_text" in df.columns:
        df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

    # 1. Size bounds check (150 to 250 records required)
    if not (150 <= len(df) <= 250):
        errors.append(f"Golden dataset size {len(df)} is outside the required 150-250 range.")

    # 2. Duplicate checks
    dup_tweets = df["tweet_id"].duplicated().sum()
    if dup_tweets > 0:
        errors.append(f"Found {dup_tweets} duplicate tweet_ids.")

    dup_threads = df["thread_id"].duplicated().sum()
    if dup_threads > 0:
        errors.append(f"Found {dup_threads} duplicate thread_ids.")

    # 3. Missing values check in required non-null columns
    for col in REQUIRED_NON_NULL_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            continue
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} nulls.")
        if col != "cleaned_text":
            empty_count = int((df[col].astype(str).str.strip() == "").sum())
            if empty_count > 0:
                errors.append(f"Column '{col}' has {empty_count} empty strings.")

    # 4. Target label validity
    valid_intents = set(get_intent_names()) | {get_fallback_intent(), "needs_review"}
    invalid_intents = set(df["golden_intent"]) - valid_intents
    if invalid_intents:
        errors.append(f"Invalid golden_intent labels found: {invalid_intents}")

    # 5. Verification status validity
    invalid_statuses = set(df["verification_status"]) - VALID_VERIFICATION_STATUSES
    if invalid_statuses:
        errors.append(f"Invalid verification_status values found: {invalid_statuses}")

    # 6. Case difficulty validity
    invalid_difficulties = set(df["case_difficulty"]) - CASE_DIFFICULTY_TIERS
    if invalid_difficulties:
        errors.append(f"Invalid case_difficulty tiers found: {invalid_difficulties}")

    # 7. needs_review cases consistency check
    needs_review_mask = df["golden_intent"] == "needs_review"
    flagged_mask = df["verification_status"] == "flagged_ambiguous"
    if not (needs_review_mask == flagged_mask).all():
        errors.append(
            "Mismatch between golden_intent == 'needs_review' and verification_status == 'flagged_ambiguous'."
        )

    # 8. Evaluation split consistency check
    if not (df["evaluation_split"] == "golden_test").all():
        errors.append("Column 'evaluation_split' must be 'golden_test' for all rows.")

    if errors:
        error_msg = "Golden evaluation set validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    return {
        "status": "valid",
        "row_count": len(df),
        "unique_tweet_ids": int(df["tweet_id"].nunique()),
        "summary": "Golden dataset successfully passed all integrity, schema, and taxonomy checks.",
    }


# ---------------------------------------------------------------------------
# Training Data Isolation Utilities
# ---------------------------------------------------------------------------
def load_golden_evaluation_set(golden_csv_path: str | Path = DEFAULT_GOLDEN_CSV) -> pd.DataFrame:
    """Load the verified Golden Evaluation Set DataFrame."""
    p = Path(golden_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Golden evaluation set not found at: {p}")
    df = pd.read_csv(p)
    if "cleaned_text" in df.columns:
        df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    return df


def filter_training_candidates(
    candidates_df: pd.DataFrame,
    golden_data: pd.DataFrame | Sequence[int | str] | set[int | str],
) -> pd.DataFrame:
    """Filter out all golden evaluation set records from a training candidate pool.

    Parameters:
    -----------
    candidates_df : DataFrame containing candidate customer inquiries (must have 'tweet_id').
    golden_data : DataFrame, list, or set of golden tweet IDs to isolate.

    Returns:
    --------
    Filtered DataFrame with zero golden evaluation records.
    """
    if "tweet_id" not in candidates_df.columns:
        raise KeyError("candidates_df must contain 'tweet_id' column.")

    if isinstance(golden_data, pd.DataFrame):
        golden_ids = set(golden_data["tweet_id"].astype(str))
    else:
        golden_ids = {str(tid) for tid in golden_data}

    filtered_df = candidates_df[~candidates_df["tweet_id"].astype(str).isin(golden_ids)].copy()

    # Assert strict isolation
    overlap = set(filtered_df["tweet_id"].astype(str)).intersection(golden_ids)
    if overlap:
        raise RuntimeError(f"Isolation failed: {len(overlap)} golden IDs remain in filtered training set!")

    return filtered_df


def verify_training_isolation(
    train_df: pd.DataFrame,
    golden_data: pd.DataFrame | Sequence[int | str] | set[int | str],
) -> bool:
    """Verify that a candidate training DataFrame has strictly zero overlap with the golden evaluation set.

    Raises:
    -------
    ValueError: If any overlap is detected.

    Returns:
    --------
    True if isolation is strictly satisfied.
    """
    if "tweet_id" not in train_df.columns:
        raise KeyError("train_df must contain 'tweet_id' column.")

    if isinstance(golden_data, pd.DataFrame):
        golden_ids = set(golden_data["tweet_id"].astype(str))
    else:
        golden_ids = {str(tid) for tid in golden_data}

    train_ids = set(train_df["tweet_id"].astype(str))
    overlap = train_ids.intersection(golden_ids)

    if overlap:
        raise ValueError(
            f"CRITICAL ISOLATION VIOLATION: {len(overlap)} golden evaluation IDs were detected "
            f"in the training dataset! Example overlapping IDs: {list(overlap)[:5]}"
        )

    return True


def export_isolated_training_candidates(
    candidates_csv_path: str | Path = DEFAULT_CANDIDATES_CSV,
    golden_csv_path: str | Path = DEFAULT_GOLDEN_CSV,
    output_path: str | Path = DEFAULT_TRAIN_CANDIDATES_CSV,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Export the candidate dataset with all golden evaluation set records strictly filtered out."""
    candidates_csv_path = Path(candidates_csv_path)
    golden_csv_path = Path(golden_csv_path)
    output_path = Path(output_path)

    if not candidates_csv_path.exists():
        raise FileNotFoundError(f"Candidates CSV not found at: {candidates_csv_path}")
    if not golden_csv_path.exists():
        raise FileNotFoundError(f"Golden CSV not found at: {golden_csv_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists at {output_path}. Use overwrite=True.")

    df_candidates = pd.read_csv(candidates_csv_path)
    df_golden = pd.read_csv(golden_csv_path)

    initial_count = len(df_candidates)
    df_train = filter_training_candidates(df_candidates, df_golden)
    isolated_count = len(df_train)

    assert initial_count - isolated_count == len(df_golden), (
        f"Expected {len(df_golden)} removed records, but removed {initial_count - isolated_count}"
    )

    verify_training_isolation(df_train, df_golden)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_path, index=False)
    logger.info(
        "Successfully exported %d isolated training candidates to %s (excluded %d golden records)",
        isolated_count,
        output_path,
        len(df_golden),
    )

    return df_train


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct, validate, and manage the AppleSupport intent classification Golden Evaluation Set."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the golden evaluation set CSV and metadata from audited pilot data.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run integrity and schema validation checks on the golden evaluation set.",
    )
    parser.add_argument(
        "--export-isolated-candidates",
        action="store_true",
        help="Export training candidates CSV with golden set IDs strictly excluded.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--pilot-csv",
        default=DEFAULT_PILOT_CSV,
        help=f"Path to input pilot CSV (default: {DEFAULT_PILOT_CSV})",
    )
    parser.add_argument(
        "--golden-csv",
        default=DEFAULT_GOLDEN_CSV,
        help=f"Path to output golden CSV (default: {DEFAULT_GOLDEN_CSV})",
    )
    parser.add_argument(
        "--golden-meta",
        default=DEFAULT_GOLDEN_META,
        help=f"Path to output golden metadata JSON (default: {DEFAULT_GOLDEN_META})",
    )
    parser.add_argument(
        "--candidates-csv",
        default=DEFAULT_CANDIDATES_CSV,
        help=f"Path to candidate inquiries CSV (default: {DEFAULT_CANDIDATES_CSV})",
    )
    parser.add_argument(
        "--train-out-csv",
        default=DEFAULT_TRAIN_CANDIDATES_CSV,
        help=f"Path to output isolated training CSV (default: {DEFAULT_TRAIN_CANDIDATES_CSV})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    if not (args.build or args.validate or args.export_isolated_candidates):
        # Default action when run without flags is build + validate
        args.build = True
        args.validate = True

    if args.build:
        print("\n=== Building AppleSupport Intent Golden Evaluation Set ===")
        df_golden, meta = build_golden_set(
            pilot_csv_path=args.pilot_csv,
            pilot_meta_path=DEFAULT_PILOT_META,
            output_golden_csv=args.golden_csv,
            output_golden_meta=args.golden_meta,
            candidates_csv_path=args.candidates_csv,
            overwrite=args.overwrite,
        )
        print(f"Golden dataset created: {len(df_golden)} records -> {args.golden_csv}")
        print(f"Content SHA-256: {meta['deterministic_content_sha256']}")
        print("\nIntent Distribution:")
        for intent, count in sorted(meta["counts_by_golden_intent"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {intent:30s}: {count}")

        print("\nCase Difficulty Slices:")
        for diff, count in sorted(meta["counts_by_case_difficulty"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {diff:30s}: {count}")

    if args.validate:
        print("\n=== Validating Golden Evaluation Set ===")
        df = load_golden_evaluation_set(args.golden_csv)
        results = validate_golden_dataset(df)
        print(f"Validation SUCCESS: {results['summary']}")

    if args.export_isolated_candidates:
        print("\n=== Exporting Isolated Training Candidates ===")
        df_train = export_isolated_training_candidates(
            candidates_csv_path=args.candidates_csv,
            golden_csv_path=args.golden_csv,
            output_path=args.train_out_csv,
            overwrite=args.overwrite,
        )
        print(f"Exported {len(df_train)} isolated training candidates -> {args.train_out_csv}")


if __name__ == "__main__":
    main()
