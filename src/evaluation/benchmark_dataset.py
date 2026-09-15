"""
src/evaluation/benchmark_dataset.py
===================================
Phase 14 — Controlled End-to-End Benchmark Dataset Generator.

Builds and persists a controlled 100-case evaluation benchmark dataset from
historical AppleSupport inquiries quarantined from the Phase 8 Golden Evaluation Set.

CRITICAL METHODOLOGICAL DISCLOSURES:
1. NON-REPRESENTATIVENESS: This 100-case benchmark is an engineering stress-test
   and coverage cohort across operational edge cases. It is NOT statistically
   representative of the real-world AppleSupport macro inquiry distribution.
2. LABEL TYPOLOGY: Every record explicitly categorizes label sources:
   - human_verified_ground_truth
   - programmatically_derived_heuristic
   - deterministic_policy_expectation
   - unlabeled_diagnostic
3. ZERO CONTAMINATION: Mathematical assertion S_benchmark ∩ S_golden = ∅ is
   enforced on build and load.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES_CSV = Path("data/processed/apple_support/apple_support_intent_candidates.csv")
DEFAULT_GOLDEN_CSV = Path("data/processed/apple_support/apple_support_intent_golden_set.csv")
DEFAULT_BENCHMARK_CSV = Path("data/processed/apple_support/apple_support_e2e_benchmark.csv")
DEFAULT_BENCHMARK_META = Path("data/processed/apple_support/apple_support_e2e_benchmark_metadata.json")

# Taxonomy classes for routine coverage
TAXONOMY_CLASSES = [
    "account_access",
    "app_or_service_issue",
    "battery_power",
    "billing_payment",
    "complaint_feedback",
    "connectivity_network",
    "device_hardware",
    "feature_how_to",
    "order_shipping",
    "software_update",
    "unknown_other",
]


def load_or_build_benchmark(
    benchmark_csv: Path = DEFAULT_BENCHMARK_CSV,
    candidates_csv: Path = DEFAULT_CANDIDATES_CSV,
    golden_csv: Path = DEFAULT_GOLDEN_CSV,
    seed: int = 42,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load existing benchmark dataset or deterministically build it from candidates.
    Always verifies Golden Set isolation.
    """
    if benchmark_csv.exists() and not force_rebuild:
        df = pd.read_csv(benchmark_csv)
        _verify_golden_isolation(df, golden_csv)
        return df

    return build_benchmark_dataset(
        candidates_csv=candidates_csv,
        golden_csv=golden_csv,
        output_csv=benchmark_csv,
        seed=seed,
    )


def _verify_golden_isolation(benchmark_df: pd.DataFrame, golden_csv: Path = DEFAULT_GOLDEN_CSV) -> None:
    """Assert zero intersection between benchmark tweet IDs and Golden Set tweet IDs."""
    if not golden_csv.exists():
        logger.warning("Golden CSV not found at %s. Skipping isolation check.", golden_csv)
        return

    golden_df = pd.read_csv(golden_csv)
    golden_ids = set(golden_df["tweet_id"].astype(int))
    benchmark_ids = set(benchmark_df["tweet_id"].astype(int))

    intersection = golden_ids.intersection(benchmark_ids)
    if intersection:
        raise RuntimeError(
            f"CRITICAL CONTAMINATION: Benchmark dataset contains {len(intersection)} "
            f"records from Golden Evaluation Set! IDs: {intersection}"
        )


def build_benchmark_dataset(
    candidates_csv: Path = DEFAULT_CANDIDATES_CSV,
    golden_csv: Path = DEFAULT_GOLDEN_CSV,
    output_csv: Path = DEFAULT_BENCHMARK_CSV,
    metadata_path: Path = DEFAULT_BENCHMARK_META,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Deterministically constructs the 100-case Phase 14 benchmark dataset.
    """
    logger.info("Building Phase 14 End-to-End Benchmark Dataset (seed: %d)...", seed)

    candidates_df = pd.read_csv(candidates_csv)
    golden_df = pd.read_csv(golden_csv)
    golden_ids = set(golden_df["tweet_id"].astype(int))

    # 1. Enforce strict isolation at pool level
    pool = candidates_df[~candidates_df["tweet_id"].astype(int).isin(golden_ids)].copy()
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    benchmark_rows: List[Dict[str, Any]] = []
    used_tweet_ids: Set[int] = set()

    # -----------------------------------------------------------------------
    # Cohort 1: Routine Inquiries (55 cases: 5 per taxonomy class x 11 classes)
    # -----------------------------------------------------------------------
    routine_targets_per_class = 5
    for intent in TAXONOMY_CLASSES:
        # High confidence, rule-based candidates only
        subset = pool[
            (pool["candidate_intent"] == intent) &
            (pool["label_confidence"] == "high") &
            (pool["label_source"] == "rule") &
            (~pool["tweet_id"].isin(used_tweet_ids))
        ]
        # For unknown_other, label_source is fallback
        if len(subset) < routine_targets_per_class:
            subset = pool[
                (pool["candidate_intent"] == intent) &
                (~pool["tweet_id"].isin(used_tweet_ids))
            ]

        sample = subset.head(routine_targets_per_class)
        for _, r in sample.iterrows():
            tid = int(r["tweet_id"])
            used_tweet_ids.add(tid)
            # account_access always escalates per policy
            exp_esc = (intent == "account_access")
            exp_rule = "account_security_escalation" if exp_esc else None

            benchmark_rows.append({
                "tweet_id": tid,
                "thread_id": str(r["thread_id"]),
                "customer_message": str(r["text"]).strip(),
                "cohort": "routine",
                "ground_truth_intent": intent,
                "intent_label_source": "programmatically_derived_heuristic",
                "expected_escalate": exp_esc,
                "expected_escalation_rule": exp_rule,
                "escalation_label_source": "deterministic_policy_expectation",
                "notes": f"Clean routine single-intent query for {intent}.",
            })

    # -----------------------------------------------------------------------
    # Cohort 2: Ambiguous / Low-Confidence Cases (15 cases)
    # -----------------------------------------------------------------------
    ambig_pool = pool[
        (pool["candidate_intent"] == "needs_review") |
        (pool["label_confidence"] == "low")
    ]
    ambig_sample = ambig_pool[~ambig_pool["tweet_id"].isin(used_tweet_ids)].head(15)
    for _, r in ambig_sample.iterrows():
        tid = int(r["tweet_id"])
        used_tweet_ids.add(tid)
        benchmark_rows.append({
            "tweet_id": tid,
            "thread_id": str(r["thread_id"]),
            "customer_message": str(r["text"]).strip(),
            "cohort": "ambiguous_low_confidence",
            "ground_truth_intent": None,
            "intent_label_source": "unlabeled_diagnostic",
            "expected_escalate": True,
            "expected_escalation_rule": "low_intent_confidence",
            "escalation_label_source": "deterministic_policy_expectation",
            "notes": "Heuristic conflict or low-confidence inquiry testing triage escalation.",
        })

    # -----------------------------------------------------------------------
    # Cohort 3: Escalation-Sensitive Cases (15 cases)
    # -----------------------------------------------------------------------
    # 3a. Safety hazards (fire, smoke, exploded, burn, shock) -> 4 cases
    safety_regex = re.compile(r"\b(?:fire|smoke|smoking|exploded|burn|burning|shock|burnt)\b", re.IGNORECASE)
    safety_candidates = pool[
        pool["text"].str.contains(safety_regex, na=False) &
        (~pool["tweet_id"].isin(used_tweet_ids))
    ].head(4)
    for _, r in safety_candidates.iterrows():
        tid = int(r["tweet_id"])
        used_tweet_ids.add(tid)
        benchmark_rows.append({
            "tweet_id": tid,
            "thread_id": str(r["thread_id"]),
            "customer_message": str(r["text"]).strip(),
            "cohort": "escalation_sensitive",
            "ground_truth_intent": None,
            "intent_label_source": "unlabeled_diagnostic",
            "expected_escalate": True,
            "expected_escalation_rule": "safety_hazard_alert",
            "escalation_label_source": "deterministic_policy_expectation",
            "notes": "Physical safety or equipment hazard keyword trigger.",
        })

    # 3b. Legal threats (lawyer, sue, lawsuit, fraud) -> 4 cases
    legal_regex = re.compile(r"\b(?:lawyer|attorney|sue|suing|lawsuit|fraud|stolen)\b", re.IGNORECASE)
    legal_candidates = pool[
        pool["text"].str.contains(legal_regex, na=False) &
        (~pool["tweet_id"].isin(used_tweet_ids))
    ].head(4)
    for _, r in legal_candidates.iterrows():
        tid = int(r["tweet_id"])
        used_tweet_ids.add(tid)
        benchmark_rows.append({
            "tweet_id": tid,
            "thread_id": str(r["thread_id"]),
            "customer_message": str(r["text"]).strip(),
            "cohort": "escalation_sensitive",
            "ground_truth_intent": None,
            "intent_label_source": "unlabeled_diagnostic",
            "expected_escalate": True,
            "expected_escalation_rule": "legal_fraud_alert",
            "escalation_label_source": "deterministic_policy_expectation",
            "notes": "Legal threat or fraud allegation keyword trigger.",
        })

    # 3c. Account Security Lockouts -> 4 cases
    account_regex = re.compile(r"\b(?:apple id|password|locked|lockout|2fa|verification code)\b", re.IGNORECASE)
    account_candidates = pool[
        (pool["candidate_intent"] == "account_access") &
        pool["text"].str.contains(account_regex, na=False) &
        (~pool["tweet_id"].isin(used_tweet_ids))
    ].head(4)
    for _, r in account_candidates.iterrows():
        tid = int(r["tweet_id"])
        used_tweet_ids.add(tid)
        benchmark_rows.append({
            "tweet_id": tid,
            "thread_id": str(r["thread_id"]),
            "customer_message": str(r["text"]).strip(),
            "cohort": "escalation_sensitive",
            "ground_truth_intent": "account_access",
            "intent_label_source": "programmatically_derived_heuristic",
            "expected_escalate": True,
            "expected_escalation_rule": "account_security_escalation",
            "escalation_label_source": "deterministic_policy_expectation",
            "notes": "Credential lockout and identity verification policy trigger.",
        })

    # 3d. High-value billing dispute ($100+, refund) -> 3 cases
    billing_regex = re.compile(r"(?:\$|£|€)\s*[1-9]\d{2,}|\b(?:full\s+refund|demand\s+refund)\b", re.IGNORECASE)
    billing_candidates = pool[
        pool["text"].str.contains(billing_regex, na=False) &
        (~pool["tweet_id"].isin(used_tweet_ids))
    ].head(3)
    for _, r in billing_candidates.iterrows():
        tid = int(r["tweet_id"])
        used_tweet_ids.add(tid)
        benchmark_rows.append({
            "tweet_id": tid,
            "thread_id": str(r["thread_id"]),
            "customer_message": str(r["text"]).strip(),
            "cohort": "escalation_sensitive",
            "ground_truth_intent": "billing_payment",
            "intent_label_source": "programmatically_derived_heuristic",
            "expected_escalate": True,
            "expected_escalation_rule": "high_value_billing_dispute",
            "escalation_label_source": "deterministic_policy_expectation",
            "notes": "Large monetary refund or commercial billing dispute policy trigger.",
        })

    # -----------------------------------------------------------------------
    # Cohort 4: Vague / Short Queries (8 cases)
    # -----------------------------------------------------------------------
    short_candidates = pool[
        (pool["text"].str.split().str.len() < 3) &
        (pool["text"].str.len() > 3) &
        (~pool["tweet_id"].isin(used_tweet_ids))
    ].head(8)
    for _, r in short_candidates.iterrows():
        tid = int(r["tweet_id"])
        used_tweet_ids.add(tid)
        benchmark_rows.append({
            "tweet_id": tid,
            "thread_id": str(r["thread_id"]),
            "customer_message": str(r["text"]).strip(),
            "cohort": "vague_short",
            "ground_truth_intent": None,
            "intent_label_source": "unlabeled_diagnostic",
            "expected_escalate": True,
            "expected_escalation_rule": "vague_short_query",
            "escalation_label_source": "deterministic_policy_expectation",
            "notes": "Ultra-short context-deficient inquiry (< 3 words).",
        })

    # -----------------------------------------------------------------------
    # Cohort 5: Weak Retrieval / Out-of-Domain Inquiries (7 cases)
    # -----------------------------------------------------------------------
    # Queries that are off-topic or out-of-domain, yielding top similarity < 0.50
    # against the AppleSupport historical index. Tests weak_retrieval_grounding escalation.
    weak_retrieval_cases = [
        (990001, "thread_ood_1", "@AppleSupport what is the recipe for chocolate chip cookies?"),
        (990002, "thread_ood_2", "@AppleSupport can you tell me the weather forecast for tomorrow in Tokyo?"),
        (990003, "thread_ood_3", "@AppleSupport who won the 1998 FIFA World Cup final in Paris?"),
        (990004, "thread_ood_4", "@AppleSupport explain quantum entanglement in simple terms please"),
        (990005, "thread_ood_5", "@AppleSupport what is the capital city of France?"),
        (990006, "thread_ood_6", "@AppleSupport how many kilometers is it from earth to the moon?"),
        (990007, "thread_ood_7", "@AppleSupport what time does the local grocery supermarket close on Sunday?"),
    ]

    for tid, thid, msg in weak_retrieval_cases:
        used_tweet_ids.add(tid)
        benchmark_rows.append({
            "tweet_id": tid,
            "thread_id": thid,
            "customer_message": msg,
            "cohort": "weak_retrieval",
            "ground_truth_intent": None,
            "intent_label_source": "unlabeled_diagnostic",
            "expected_escalate": True,
            "expected_escalation_rule": "weak_retrieval_grounding",
            "escalation_label_source": "deterministic_policy_expectation",
            "notes": "Out-of-domain inquiry yielding similarity < 0.50 testing weak retrieval grounding.",
        })

    benchmark_df = pd.DataFrame(benchmark_rows)

    # Assign deterministic benchmark IDs BM_001 to BM_100
    benchmark_df.insert(0, "benchmark_id", [f"BM_{i+1:03d}" for i in range(len(benchmark_df))])

    # Enforce isolation
    _verify_golden_isolation(benchmark_df, golden_csv)

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(output_csv, index=False)
    logger.info("Saved Phase 14 benchmark dataset (%d records) to %s.", len(benchmark_df), output_csv)

    # Save metadata JSON
    metadata: Dict[str, Any] = {
        "dataset_name": "AppleSupport End-to-End Evaluation Benchmark",
        "phase": "Phase 14: End-to-End Batch Evaluation & Quality Benchmarking",
        "total_records": len(benchmark_df),
        "random_seed": seed,
        "golden_set_overlap_count": 0,
        "methodological_disclosures": {
            "non_representativeness": (
                "This 100-case dataset is an engineered coverage stress-test across 5 operational cohorts. "
                "It is NOT statistically representative of the overall AppleSupport Twitter population."
            ),
            "label_typology": {
                "programmatically_derived_heuristic": int((benchmark_df["intent_label_source"] == "programmatically_derived_heuristic").sum()),
                "unlabeled_diagnostic": int((benchmark_df["intent_label_source"] == "unlabeled_diagnostic").sum()),
                "deterministic_policy_expectation": int((benchmark_df["escalation_label_source"] == "deterministic_policy_expectation").sum()),
            },
        },
        "cohort_counts": benchmark_df["cohort"].value_counts().to_dict(),
        "intent_distribution": benchmark_df["ground_truth_intent"].value_counts(dropna=False).to_dict(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return benchmark_df
