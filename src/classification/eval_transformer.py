"""
src/classification/eval_transformer.py
======================================
Evaluation and Comparative Diagnostic Engine for Fine-Tuned Transformer Models.

Loads fine-tuned DistilRoBERTa model and tokenizer from disk, evaluates on the
human-verified Golden Evaluation Set, and compares directly against Phase 9 baselines
(Majority Baseline, TF-IDF + Linear SVM, TF-IDF + Logistic Regression).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.classification.build_golden_evaluation_set import load_golden_evaluation_set
from src.classification.train_distilroberta import (
    DEFAULT_GOLDEN_CSV,
    DEFAULT_OUTPUT_DIR,
    evaluate_transformer_on_golden_set,
    get_label_mappings,
)

logger = logging.getLogger(__name__)

DEFAULT_BASELINES_JSON = "data/processed/apple_support/apple_support_intent_evaluation_results.json"
DEFAULT_TRANSFORMER_RESULTS_JSON = "data/processed/apple_support/apple_support_distilroberta_evaluation_results.json"


def load_fine_tuned_model(model_dir: str | Path = DEFAULT_OUTPUT_DIR) -> tuple[Any, Any, Dict[str, Any]]:
    """Load fine-tuned transformer weights, tokenizer, and metadata from disk."""
    m_path = Path(model_dir)
    if not m_path.exists():
        raise FileNotFoundError(f"Fine-tuned model directory not found at: {m_path}")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("PyTorch and transformers must be installed to load the transformer model.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading tokenizer from: %s", m_path)
    tokenizer = AutoTokenizer.from_pretrained(str(m_path))

    logger.info("Loading model from: %s onto device: %s", m_path, device)
    model = AutoModelForSequenceClassification.from_pretrained(str(m_path)).to(device)

    meta_file = m_path / "model_metadata.json"
    metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, tokenizer, metadata


def compare_with_phase9_baselines(
    transformer_results: Dict[str, Any],
    baselines_json_path: str | Path = DEFAULT_BASELINES_JSON,
) -> Dict[str, Any]:
    """Generate comparative performance matrix across all models."""
    b_path = Path(baselines_json_path)
    baselines_data = {}
    if b_path.exists():
        with open(b_path, "r", encoding="utf-8") as f:
            baselines_data = json.load(f).get("models", {})

    comparison: Dict[str, Any] = {
        "models": {},
        "per_intent_deltas_vs_logistic_regression": {},
    }

    # Add baselines
    for key, data in baselines_data.items():
        comparison["models"][key] = {
            "name": data.get("model_name", key),
            "metrics": data.get("overall_metrics", {}),
        }

    # Add DistilRoBERTa
    trans_metrics = transformer_results.get("overall_metrics", {})
    comparison["models"]["distilroberta"] = {
        "name": "DistilRoBERTa (Transformer)",
        "metrics": trans_metrics,
    }

    # Compare per-intent F1 against Logistic Regression (best baseline)
    lr_data = baselines_data.get("logistic_regression", {})
    lr_per_intent = lr_data.get("per_intent_metrics", {})
    trans_per_intent = transformer_results.get("per_intent_metrics", {})

    deltas: Dict[str, Dict[str, float]] = {}
    for intent, t_scores in trans_per_intent.items():
        if isinstance(t_scores, dict) and "f1-score" in t_scores:
            t_f1 = t_scores["f1-score"]
            lr_scores = lr_per_intent.get(intent, {})
            lr_f1 = lr_scores.get("f1-score", 0.0) if isinstance(lr_scores, dict) else 0.0
            deltas[intent] = {
                "distilroberta_f1": round(float(t_f1), 4),
                "logistic_regression_f1": round(float(lr_f1), 4),
                "delta_f1": round(float(t_f1 - lr_f1), 4),
            }

    comparison["per_intent_deltas_vs_logistic_regression"] = deltas
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DistilRoBERTa against Golden Set.")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--golden-csv", default=DEFAULT_GOLDEN_CSV)
    parser.add_argument("--baselines-json", default=DEFAULT_BASELINES_JSON)
    parser.add_argument("--out-results", default=DEFAULT_TRANSFORMER_RESULTS_JSON)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    model, tokenizer, meta = load_fine_tuned_model(args.model_dir)
    golden_df = load_golden_evaluation_set(args.golden_csv)
    _, id2label = get_label_mappings()

    eval_results = evaluate_transformer_on_golden_set(
        model=model,
        tokenizer=tokenizer,
        golden_df=golden_df,
        id2label=id2label,
    )

    comparison = compare_with_phase9_baselines(eval_results, args.baselines_json)

    out_path = Path(args.out_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"evaluation": eval_results, "comparison": comparison}, f, indent=2)
    logger.info("Saved evaluation and comparison to: %s", out_path)


if __name__ == "__main__":
    main()
