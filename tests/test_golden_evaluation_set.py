"""
tests/test_golden_evaluation_set.py
===================================
Unit and integration tests for the AppleSupport intent classification
Golden Evaluation Set and training data isolation mechanisms.

Validates:
1. Golden dataset existence, schema completeness, and size (150 <= N <= 250).
2. Zero duplicate tweet_ids or thread_ids.
3. Zero null or empty values across all mandatory verification columns.
4. Target intent label validity against Taxonomy Version 2.0.
5. Explicit preservation and consistency of `needs_review` / `flagged_ambiguous` cases.
6. Representation across all 8 operational difficulty slices.
7. Strict training data isolation (S_train ∩ S_golden = ∅) and violation detection.
8. Metadata synchronization and SHA-256 content verification.
9. Link hygiene and confidentiality compliance in documentation.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.classification.build_golden_evaluation_set import (
    CASE_DIFFICULTY_TIERS,
    GOLDEN_SET_COLUMNS,
    REQUIRED_NON_NULL_COLUMNS,
    VALID_VERIFICATION_STATUSES,
    build_golden_set,
    filter_training_candidates,
    load_golden_evaluation_set,
    validate_golden_dataset,
    verify_training_isolation,
)
from src.classification.intent_loader import get_fallback_intent, get_intent_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def golden_paths(repo_root: Path) -> tuple[Path, Path]:
    csv_path = repo_root / "data" / "processed" / "apple_support" / "apple_support_intent_golden_set.csv"
    meta_path = repo_root / "data" / "processed" / "apple_support" / "apple_support_intent_golden_metadata.json"
    return csv_path, meta_path


@pytest.fixture
def golden_df(golden_paths: tuple[Path, Path]) -> pd.DataFrame:
    csv_path, _ = golden_paths
    if not csv_path.exists():
        pytest.skip("Golden evaluation set CSV not present locally.")
    return load_golden_evaluation_set(csv_path)


@pytest.fixture
def golden_metadata(golden_paths: tuple[Path, Path]) -> dict:
    _, meta_path = golden_paths
    if not meta_path.exists():
        pytest.skip("Golden evaluation metadata JSON not present locally.")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Dataset Integrity & Schema Tests
# ---------------------------------------------------------------------------
class TestGoldenDatasetIntegrity:
    def test_golden_artifacts_exist(self, golden_paths: tuple[Path, Path]):
        csv_path, meta_path = golden_paths
        assert csv_path.exists(), f"Golden CSV missing at: {csv_path}"
        assert meta_path.exists(), f"Golden metadata missing at: {meta_path}"

    def test_golden_dataset_size_within_bounds(self, golden_df: pd.DataFrame):
        assert 150 <= len(golden_df) <= 250, (
            f"Golden dataset size {len(golden_df)} is outside required 150-250 range."
        )
        assert len(golden_df) == 158

    def test_zero_duplicate_ids(self, golden_df: pd.DataFrame):
        assert golden_df["tweet_id"].duplicated().sum() == 0, "Duplicate tweet_ids found in golden set!"
        assert golden_df["thread_id"].duplicated().sum() == 0, "Duplicate thread_ids found in golden set!"
        assert golden_df["tweet_id"].nunique() == len(golden_df)
        assert golden_df["thread_id"].nunique() == len(golden_df)

    def test_all_canonical_columns_present(self, golden_df: pd.DataFrame):
        for col in GOLDEN_SET_COLUMNS:
            assert col in golden_df.columns, f"Required golden column missing: {col}"

    def test_mandatory_verification_fields_non_null(self, golden_df: pd.DataFrame):
        for col in REQUIRED_NON_NULL_COLUMNS:
            assert col in golden_df.columns
            assert golden_df[col].notna().all(), f"Null values detected in column: {col}"
            if col != "cleaned_text":
                assert (golden_df[col].astype(str).str.strip() != "").all(), (
                    f"Empty string detected in mandatory column: {col}"
                )

    def test_target_intent_labels_valid(self, golden_df: pd.DataFrame):
        valid_intents = set(get_intent_names()) | {get_fallback_intent(), "needs_review"}
        dataset_intents = set(golden_df["golden_intent"])
        invalid_intents = dataset_intents - valid_intents
        assert not invalid_intents, f"Invalid golden_intent labels detected: {invalid_intents}"

    def test_verification_status_valid(self, golden_df: pd.DataFrame):
        dataset_statuses = set(golden_df["verification_status"])
        invalid_statuses = dataset_statuses - VALID_VERIFICATION_STATUSES
        assert not invalid_statuses, f"Invalid verification_status detected: {invalid_statuses}"

    def test_evaluation_split_constant(self, golden_df: pd.DataFrame):
        assert (golden_df["evaluation_split"] == "golden_test").all()

    def test_validator_function_passes(self, golden_df: pd.DataFrame):
        result = validate_golden_dataset(golden_df)
        assert result["status"] == "valid"
        assert result["row_count"] == 158


# ---------------------------------------------------------------------------
# 2. Ambiguity & Multi-Intent Preservation Tests
# ---------------------------------------------------------------------------
class TestAmbiguityPreservation:
    def test_needs_review_cases_flagged_ambiguous(self, golden_df: pd.DataFrame):
        needs_review_cases = golden_df[golden_df["golden_intent"] == "needs_review"]
        assert len(needs_review_cases) == 3, f"Expected exactly 3 needs_review cases, found {len(needs_review_cases)}"

        assert (needs_review_cases["verification_status"] == "flagged_ambiguous").all(), (
            "All needs_review cases must have verification_status == 'flagged_ambiguous'."
        )
        assert (needs_review_cases["case_difficulty"] == "ambiguous_multi_intent").all(), (
            "All needs_review cases must have case_difficulty == 'ambiguous_multi_intent'."
        )
        assert (needs_review_cases["notes"].astype(str).str.len() > 10).all(), (
            "Explanatory notes required for all flagged ambiguous cases."
        )

    def test_canonical_ambiguous_tweet_ids(self, golden_df: pd.DataFrame):
        expected_ambiguous_ids = {410510, 965710, 1066754}
        actual_ambiguous_ids = set(golden_df[golden_df["golden_intent"] == "needs_review"]["tweet_id"].astype(int))
        assert actual_ambiguous_ids == expected_ambiguous_ids

    def test_corrected_cases_have_different_intent(self, golden_df: pd.DataFrame):
        corrected_cases = golden_df[golden_df["verification_status"] == "corrected"]
        assert len(corrected_cases) == 79
        assert (corrected_cases["golden_intent"] != corrected_cases["preliminary_intent"]).all()

    def test_verified_cases_have_matching_intent(self, golden_df: pd.DataFrame):
        verified_cases = golden_df[golden_df["verification_status"] == "verified"]
        assert len(verified_cases) == 76
        assert (verified_cases["golden_intent"] == verified_cases["preliminary_intent"]).all()


# ---------------------------------------------------------------------------
# 3. Operational Difficulty Slices Tests
# ---------------------------------------------------------------------------
class TestCaseDifficultySlices:
    def test_all_difficulty_tiers_represented(self, golden_df: pd.DataFrame):
        dataset_tiers = set(golden_df["case_difficulty"])
        assert dataset_tiers == CASE_DIFFICULTY_TIERS, (
            f"Missing difficulty tiers: {CASE_DIFFICULTY_TIERS - dataset_tiers}"
        )

    def test_difficulty_counts_match_metadata(self, golden_df: pd.DataFrame, golden_metadata: dict):
        df_counts = golden_df["case_difficulty"].value_counts().to_dict()
        meta_counts = golden_metadata["counts_by_case_difficulty"]
        assert df_counts == meta_counts

    def test_symptom_attribution_slice_criteria(self, golden_df: pd.DataFrame):
        symptom_slice = golden_df[golden_df["case_difficulty"] == "attribution_vs_symptom"]
        assert len(symptom_slice) >= 20
        for _, row in symptom_slice.iterrows():
            notes = str(row["notes"]).lower()
            assert any(term in notes for term in ["rule 3", "symptom", "battery", "update", "attribution"])


# ---------------------------------------------------------------------------
# 4. Training Data Isolation Tests
# ---------------------------------------------------------------------------
class TestTrainingDataIsolation:
    def test_golden_ids_registered_in_metadata(self, golden_df: pd.DataFrame, golden_metadata: dict):
        iso_meta = golden_metadata["training_data_isolation"]
        assert iso_meta["is_strictly_isolated"] is True
        assert iso_meta["golden_tweet_id_count"] == 158

        registered_ids = set(iso_meta["golden_tweet_ids"])
        actual_ids = set(golden_df["tweet_id"].astype(int))
        assert registered_ids == actual_ids

    def test_filter_training_candidates_success(self, golden_df: pd.DataFrame, repo_root: Path):
        candidates_csv = repo_root / "data" / "processed" / "apple_support" / "apple_support_intent_candidates.csv"
        if not candidates_csv.exists():
            pytest.skip("Full candidates CSV not present locally.")

        df_candidates = pd.read_csv(candidates_csv, usecols=["tweet_id", "text"])
        initial_len = len(df_candidates)
        df_isolated = filter_training_candidates(df_candidates, golden_df)

        assert len(df_isolated) == initial_len - len(golden_df)
        assert len(set(df_isolated["tweet_id"]).intersection(set(golden_df["tweet_id"]))) == 0

    def test_verify_training_isolation_passes_on_disjoint_sets(self, golden_df: pd.DataFrame):
        synthetic_train = pd.DataFrame({"tweet_id": [99999901, 99999902, 99999903]})
        assert verify_training_isolation(synthetic_train, golden_df) is True

    def test_verify_training_isolation_raises_on_leakage(self, golden_df: pd.DataFrame):
        leaked_id = int(golden_df["tweet_id"].iloc[0])
        synthetic_train_with_leak = pd.DataFrame({"tweet_id": [99999901, leaked_id, 99999903]})
        with pytest.raises(ValueError, match="CRITICAL ISOLATION VIOLATION"):
            verify_training_isolation(synthetic_train_with_leak, golden_df)


# ---------------------------------------------------------------------------
# 5. Metadata Synchronization & Documentation Hygiene Tests
# ---------------------------------------------------------------------------
class TestMetadataAndDocHygiene:
    def test_content_sha256_matches_file(self, golden_paths: tuple[Path, Path], golden_metadata: dict):
        csv_path, _ = golden_paths
        with open(csv_path, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()
        assert golden_metadata["deterministic_content_sha256"] == computed_sha

    def test_intent_counts_match_metadata(self, golden_df: pd.DataFrame, golden_metadata: dict):
        df_intent_counts = golden_df["golden_intent"].value_counts().to_dict()
        meta_intent_counts = golden_metadata["counts_by_golden_intent"]
        assert df_intent_counts == meta_intent_counts

    def test_status_counts_match_metadata(self, golden_df: pd.DataFrame, golden_metadata: dict):
        df_status_counts = golden_df["verification_status"].value_counts().to_dict()
        meta_status_counts = golden_metadata["counts_by_verification_status"]
        assert df_status_counts == meta_status_counts

    def test_no_local_file_links_in_golden_doc(self, repo_root: Path):
        doc_path = repo_root / "docs" / "apple_support_intent_golden_evaluation_set.md"
        assert doc_path.exists(), "Golden evaluation set documentation missing!"
        content = doc_path.read_text(encoding="utf-8")
        assert "file://" not in content, "Forbidden local file:// URI detected in golden documentation!"
        assert "/home/" not in content, "Forbidden absolute path detected in golden documentation!"


# ---------------------------------------------------------------------------
# 6. Builder Execution & Overwrite Guard Tests
# ---------------------------------------------------------------------------
class TestBuilderExecution:
    def test_overwrite_guard(self, tmp_path: Path, golden_paths: tuple[Path, Path]):
        csv_path, meta_path = golden_paths
        out_csv = tmp_path / "test_golden.csv"
        out_meta = tmp_path / "test_golden_meta.json"

        # First build succeeds
        build_golden_set(
            pilot_csv_path=csv_path,
            pilot_meta_path=meta_path,
            output_golden_csv=out_csv,
            output_golden_meta=out_meta,
            overwrite=True,
        )
        assert out_csv.exists()

        # Re-run without overwrite raises FileExistsError
        with pytest.raises(FileExistsError):
            build_golden_set(
                pilot_csv_path=csv_path,
                pilot_meta_path=meta_path,
                output_golden_csv=out_csv,
                output_golden_meta=out_meta,
                overwrite=False,
            )
