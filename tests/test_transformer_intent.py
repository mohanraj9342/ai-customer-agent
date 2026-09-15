"""
tests/test_transformer_intent.py
================================
Unit and validation tests for Phase 10 DistilRoBERTa pipeline and Colab configuration.

Verifies:
1. Label mapping consistency with Taxonomy v2.0 (11 classes).
2. Training data preparation and strict golden set isolation.
3. GPU enforcement: verify_gpu_available() blocks CPU execution with RuntimeError.
4. Colab notebook integrity: JSON validity, GPU accelerator metadata, pre-flight checks.
5. Baseline comparison calculation logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.classification.build_golden_evaluation_set import load_golden_evaluation_set
from src.classification.intent_loader import get_intent_names
from src.classification.train_distilroberta import (
    DEFAULT_GOLDEN_CSV,
    DEFAULT_TRAIN_CSV,
    get_label_mappings,
    prepare_datasets,
    verify_gpu_available,
)
from src.classification.eval_transformer import compare_with_phase9_baselines


@pytest.fixture(scope="module")
def golden_df():
    return load_golden_evaluation_set(DEFAULT_GOLDEN_CSV)


def test_label_mappings_match_taxonomy():
    """Verify label2id and id2label contain exactly the 11 taxonomy classes."""
    label2id, id2label = get_label_mappings()
    taxonomy_intents = set(get_intent_names())

    assert len(label2id) == 11
    assert len(id2label) == 11
    assert set(label2id.keys()) == taxonomy_intents
    for i, name in id2label.items():
        assert label2id[name] == i


def test_prepare_datasets_enforces_isolation(golden_df, tmp_path):
    """Verify prepare_datasets raises ValueError if golden IDs leak into training candidates."""
    leak_id = int(golden_df.iloc[0]["tweet_id"])
    tainted_df = pd.DataFrame(
        [
            {"tweet_id": leak_id, "text": "battery dying", "candidate_intent": "battery_power"},
            {"tweet_id": 9990001, "text": "wifi not connecting", "candidate_intent": "connectivity_network"},
        ]
    )
    tainted_csv = tmp_path / "tainted_train.csv"
    tainted_df.to_csv(tainted_csv, index=False)

    with pytest.raises(ValueError, match="CRITICAL ISOLATION VIOLATION"):
        prepare_datasets(tainted_csv, golden_df)


def test_prepare_datasets_filters_needs_review(golden_df, tmp_path):
    """Verify prepare_datasets filters out unverified 'needs_review' heuristic rows."""
    mock_df = pd.DataFrame(
        [
            {"tweet_id": 900001, "text": "battery dead", "candidate_intent": "battery_power"},
            {"tweet_id": 900002, "text": "ambiguous message", "candidate_intent": "needs_review"},
            {"tweet_id": 900003, "text": "wifi issues", "candidate_intent": "connectivity_network"},
            {"tweet_id": 900004, "text": "refund please", "candidate_intent": "billing_payment"},
        ]
        * 10  # Repeat so stratified split has enough samples
    )
    mock_csv = tmp_path / "mock_train.csv"
    mock_df.to_csv(mock_csv, index=False)

    df_train, df_val, label2id, id2label, weights = prepare_datasets(mock_csv, golden_df, val_ratio=0.2, seed=42)

    assert "needs_review" not in df_train["candidate_intent"].values
    assert "needs_review" not in df_val["candidate_intent"].values
    assert len(weights) == 11
    assert len(df_train) + len(df_val) == 30  # 40 minus 10 needs_review


def test_verify_gpu_available_blocks_cpu():
    """Verify that verify_gpu_available raises RuntimeError if PyTorch or CUDA is unavailable.

    This ensures that silent fallback to local CPU training is strictly prevented.
    """
    try:
        import torch
        cuda_present = torch.cuda.is_available()
    except ImportError:
        cuda_present = False

    if not cuda_present:
        with pytest.raises(RuntimeError, match="(CUDA GPU is not available|PyTorch is not installed)"):
            verify_gpu_available()


def test_colab_notebook_structure():
    """Verify train_distilroberta_colab.ipynb is valid JSON and declares GPU accelerator."""
    nb_path = Path("notebooks/train_distilroberta_colab.ipynb")
    assert nb_path.exists(), "Colab notebook missing at notebooks/train_distilroberta_colab.ipynb"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert "metadata" in nb
    assert nb["metadata"].get("accelerator") == "GPU"
    assert nb["metadata"].get("colab", {}).get("gpuType") == "T4"

    # Verify key code cells exist
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5

    # Check that nvidia-smi pre-flight check exists in first code cell
    first_code = "".join(code_cells[0]["source"])
    assert "nvidia-smi" in first_code
    assert "torch.cuda.is_available" in first_code


def test_compare_with_phase9_baselines_logic(tmp_path):
    """Verify comparison logic accurately computes metric deltas vs Logistic Regression."""
    mock_baselines = {
        "models": {
            "logistic_regression": {
                "model_name": "TF-IDF + Logistic Regression",
                "overall_metrics": {"accuracy": 0.5935, "macro_f1": 0.6319, "weighted_f1": 0.5964},
                "per_intent_metrics": {
                    "battery_power": {"f1-score": 0.6977},
                    "connectivity_network": {"f1-score": 0.8571},
                },
            }
        }
    }
    b_file = tmp_path / "baselines.json"
    with open(b_file, "w", encoding="utf-8") as f:
        json.dump(mock_baselines, f)

    mock_transformer = {
        "overall_metrics": {"accuracy": 0.6800, "macro_f1": 0.7100, "weighted_f1": 0.6850},
        "per_intent_metrics": {
            "battery_power": {"f1-score": 0.7500},
            "connectivity_network": {"f1-score": 0.8800},
        },
    }

    comp = compare_with_phase9_baselines(mock_transformer, b_file)

    assert "distilroberta" in comp["models"]
    assert comp["models"]["distilroberta"]["metrics"]["macro_f1"] == 0.7100

    deltas = comp["per_intent_deltas_vs_logistic_regression"]
    assert "battery_power" in deltas
    assert deltas["battery_power"]["delta_f1"] == round(0.7500 - 0.6977, 4)
    assert deltas["connectivity_network"]["delta_f1"] == round(0.8800 - 0.8571, 4)
