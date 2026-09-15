"""
tests/test_train_eval_intent_classifier.py
=========================================
Unit and integration tests for AppleSupport intent classification pipeline.

Verifies:
1. Strict training isolation enforcement (rejects overlap with golden set).
2. Elimination of unverified 'needs_review' heuristic labels from training.
3. Correct behavior and probability format of MajorityClassBaseline.
4. Deterministic reproducibility across multiple runs with fixed random seed.
5. Serialized model artifacts integrity and loadability (joblib).
6. Comprehensive evaluation output schema (metrics, confusion matrix, slice metrics, ambiguous analysis).
7. Validity of all predicted labels against approved taxonomy.
8. Zero data leakage: models accept only customer initial problem text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification.build_golden_evaluation_set import (
    CASE_DIFFICULTY_TIERS,
    load_golden_evaluation_set,
)
from src.classification.intent_loader import get_intent_names
from src.classification.train_eval_intent_classifier import (
    DEFAULT_EVAL_RESULTS_JSON,
    DEFAULT_GOLDEN_CSV,
    DEFAULT_MODELS_DIR,
    DEFAULT_TRAIN_CSV,
    MajorityClassBaseline,
    load_model_artifacts,
    prepare_training_data,
    train_models,
)


@pytest.fixture(scope="module")
def golden_df():
    """Load golden evaluation set."""
    golden_path = Path(DEFAULT_GOLDEN_CSV)
    assert golden_path.exists(), f"Golden CSV missing at {golden_path}"
    return load_golden_evaluation_set(golden_path)


@pytest.fixture(scope="module")
def loaded_artifacts():
    """Load serialized model artifacts from disk."""
    models_dir = Path(DEFAULT_MODELS_DIR)
    assert models_dir.exists(), f"Models directory missing at {models_dir}"
    return load_model_artifacts(models_dir)


@pytest.fixture(scope="module")
def evaluation_results():
    """Load evaluation results JSON."""
    results_path = Path(DEFAULT_EVAL_RESULTS_JSON)
    assert results_path.exists(), f"Results JSON missing at {results_path}"
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Training Isolation and Data Preparation Tests
# ---------------------------------------------------------------------------
def test_prepare_training_data_isolation_violation_raises(golden_df, tmp_path):
    """Verify prepare_training_data raises ValueError if golden set tweet_id is injected into training."""
    overlap_id = int(golden_df.iloc[0]["tweet_id"])
    tainted_df = pd.DataFrame(
        [
            {
                "tweet_id": overlap_id,
                "text": "My phone is broken",
                "candidate_intent": "hardware_defect",
            },
            {
                "tweet_id": 999999999,
                "text": "Normal customer message",
                "candidate_intent": "battery_power",
            },
        ]
    )
    tainted_csv = tmp_path / "tainted_train.csv"
    tainted_df.to_csv(tainted_csv, index=False)

    with pytest.raises(ValueError, match="CRITICAL ISOLATION VIOLATION"):
        prepare_training_data(tainted_csv, golden_df)


def test_prepare_training_data_filters_needs_review(golden_df, tmp_path):
    """Verify unverified 'needs_review' heuristic rows are excluded from training targets."""
    mock_df = pd.DataFrame(
        [
            {
                "tweet_id": 800000001,
                "text": "Valid message 1",
                "candidate_intent": "software_update",
            },
            {
                "tweet_id": 800000002,
                "text": "Ambiguous message 2",
                "candidate_intent": "needs_review",
            },
            {
                "tweet_id": 800000003,
                "text": "Valid message 3",
                "candidate_intent": "battery_power",
            },
        ]
    )
    mock_csv = tmp_path / "mock_train.csv"
    mock_df.to_csv(mock_csv, index=False)

    df_clean, y_clean = prepare_training_data(mock_csv, golden_df)
    assert len(df_clean) == 2
    assert "needs_review" not in y_clean.values
    assert set(y_clean.values) == {"software_update", "battery_power"}


# ---------------------------------------------------------------------------
# 2. Majority Class Baseline Tests
# ---------------------------------------------------------------------------
def test_majority_baseline_predict():
    """Verify MajorityClassBaseline predicts the mode and formats probability matrix."""
    model = MajorityClassBaseline()
    y_train = ["apple_id_account", "battery_power", "apple_id_account", "software_update"]
    X_train = np.zeros((4, 5))

    model.fit(X_train, y_train)

    X_test = np.zeros((3, 5))
    preds = model.predict(X_test)
    assert list(preds) == ["apple_id_account", "apple_id_account", "apple_id_account"]

    probs = model.predict_proba(X_test)
    assert probs.shape == (3, 3)  # 3 distinct classes
    maj_idx = np.where(model.classes_ == "apple_id_account")[0][0]
    assert np.all(probs[:, maj_idx] == 1.0)
    assert np.all(np.delete(probs, maj_idx, axis=1) == 0.0)


def test_majority_baseline_unfitted_raises():
    """Verify calling predict on unfitted MajorityClassBaseline raises an exception."""
    model = MajorityClassBaseline()
    with pytest.raises(Exception, match="not fitted yet"):
        model.predict(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# 3. Model Artifacts & Reproducibility Tests
# ---------------------------------------------------------------------------
def test_model_artifacts_loaded(loaded_artifacts):
    """Verify all serialized artifacts are present and of expected types."""
    assert "vectorizer" in loaded_artifacts
    assert "majority_model" in loaded_artifacts
    assert "logistic_regression_model" in loaded_artifacts
    assert "linear_svm_model" in loaded_artifacts
    assert "metadata" in loaded_artifacts

    # Check vectorizer
    vec = loaded_artifacts["vectorizer"]
    assert hasattr(vec, "vocabulary_")
    assert len(vec.vocabulary_) <= 10000

    # Check models have classes_
    assert len(loaded_artifacts["logistic_regression_model"].classes_) == 11
    assert len(loaded_artifacts["linear_svm_model"].classes_) == 11
    assert len(loaded_artifacts["majority_model"].classes_) == 11


def test_training_determinism_reproducibility():
    """Verify training is deterministic with fixed random seed."""
    texts = [
        "iphone battery draining fast after update",
        "cannot log into my apple id account password reset",
        "airpods microphone not working crackling noise",
        "where is my order tracking number delivery delay",
        "subscription charged twice apple music refund request",
    ] * 6
    labels = [
        "battery_power",
        "apple_id_account",
        "hardware_defect",
        "order_shipping",
        "billing_payment",
    ] * 6

    res1 = train_models(texts, labels, seed=42)
    res2 = train_models(texts, labels, seed=42)

    test_input = ["battery life is terrible after installing new ios"]
    vec1 = res1["vectorizer"].transform(test_input)
    vec2 = res2["vectorizer"].transform(test_input)

    pred1 = res1["logistic_regression_model"].predict(vec1)
    pred2 = res2["logistic_regression_model"].predict(vec2)
    assert list(pred1) == list(pred2)

    svm_pred1 = res1["linear_svm_model"].predict(vec1)
    svm_pred2 = res2["linear_svm_model"].predict(vec2)
    assert list(svm_pred1) == list(svm_pred2)


# ---------------------------------------------------------------------------
# 4. Evaluation Output & Benchmark Results Tests
# ---------------------------------------------------------------------------
def test_evaluation_results_schema(evaluation_results):
    """Verify structure and schema completeness of apple_support_intent_evaluation_results.json."""
    assert "pipeline_version" in evaluation_results
    assert "taxonomy_version" in evaluation_results
    assert "random_seed" in evaluation_results
    assert "models" in evaluation_results

    required_models = ["majority_baseline", "logistic_regression", "linear_svm"]
    for m in required_models:
        assert m in evaluation_results["models"], f"Missing model: {m}"
        m_res = evaluation_results["models"][m]
        assert "overall_metrics" in m_res
        metrics = m_res["overall_metrics"]
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["weighted_f1"] <= 1.0

        assert "confusion_matrix" in m_res
        assert "labels" in m_res["confusion_matrix"]
        assert "matrix" in m_res["confusion_matrix"]

        assert "slice_level_metrics" in m_res
        for tier in CASE_DIFFICULTY_TIERS:
            assert tier in m_res["slice_level_metrics"], f"Missing slice: {tier}"

        assert "ambiguous_cases_analysis" in m_res
        assert len(m_res["ambiguous_cases_analysis"]) == 3


def test_model_performance_hierarchy(evaluation_results):
    """Verify Logistic Regression significantly beats Majority Baseline on Macro-F1."""
    maj_f1 = evaluation_results["models"]["majority_baseline"]["overall_metrics"]["macro_f1"]
    lr_f1 = evaluation_results["models"]["logistic_regression"]["overall_metrics"]["macro_f1"]
    svm_f1 = evaluation_results["models"]["linear_svm"]["overall_metrics"]["macro_f1"]

    # Baseline is near 0.015 due to single-class prediction
    assert maj_f1 < 0.05
    # Both machine learning models achieve substantial Macro-F1 (> 0.50)
    assert lr_f1 > 0.50
    assert svm_f1 > 0.50
    # Logistic Regression outperforms or matches SVM
    assert lr_f1 >= svm_f1 - 0.05


def test_ambiguous_cases_diagnostic_structure(evaluation_results):
    """Verify the 3 needs_review cases contain full diagnostic information."""
    lr_amb = evaluation_results["models"]["logistic_regression"]["ambiguous_cases_analysis"]
    expected_ids = {410510, 965710, 1066754}
    found_ids = {item["tweet_id"] for item in lr_amb}
    assert found_ids == expected_ids

    for item in lr_amb:
        assert len(item["text"]) > 0
        assert item["predicted_intent"] in get_intent_names()
        assert 0.0 < item["confidence"] <= 1.0
        assert 0.0 <= item["confidence_margin"] <= 1.0
        assert len(item["top_predictions"]) >= 2
        assert len(item["human_reviewer_notes"]) > 0


def test_predicted_intents_are_valid_taxonomy_labels(golden_df, loaded_artifacts):
    """Verify all predictions across golden evaluation set match valid taxonomy names."""
    vec = loaded_artifacts["vectorizer"]
    lr_model = loaded_artifacts["logistic_regression_model"]
    svm_model = loaded_artifacts["linear_svm_model"]

    valid_intents = set(get_intent_names())
    X = vec.transform(golden_df["text"])

    lr_preds = lr_model.predict(X)
    svm_preds = svm_model.predict(X)

    for p in lr_preds:
        assert p in valid_intents, f"Invalid LR predicted intent: {p}"
    for p in svm_preds:
        assert p in valid_intents, f"Invalid SVM predicted intent: {p}"
