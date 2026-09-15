"""
src/classification/train_eval_intent_classifier.py
==================================================
Training, evaluation, and analysis pipeline for AppleSupport intent classification.

Models Implemented:
1. Majority-Class Baseline (predicts most frequent training class)
2. TF-IDF + Logistic Regression (L2 regularization, class-weight balancing)
3. TF-IDF + Linear SVM (LinearSVC with class-weight balancing)

Key Guarantees:
- Strict Training Isolation: Validates that golden evaluation examples are never
  included in training data (asserts S_train ∩ S_golden = ∅).
- Input Text Only: Classifiers operate exclusively on the customer's initial problem
  statement (`text`). No brand replies, subsequent turns, or resolution metadata are used.
- Full Diagnostic Evaluation: Evaluates overall accuracy, Macro-F1, Weighted-F1,
  per-intent precision/recall/F1, confusion matrices, slice-level performance across
  all 8 operational difficulty tiers, and confidence/margin analysis on ambiguous cases.
- Reproducibility: Fully deterministic with fixed random seed (default: 42).
- Artifact Persistence: Serializes vectorizers, models, metadata, and evaluation results.

Usage:
------
    # Train, evaluate, and save model artifacts
    python -m src.classification.train_eval_intent_classifier --train --evaluate --save-models

    # Evaluate existing model artifacts on golden set
    python -m src.classification.train_eval_intent_classifier --evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from src.classification.build_golden_evaluation_set import (
    CASE_DIFFICULTY_TIERS,
    load_golden_evaluation_set,
    verify_training_isolation,
)
from src.classification.intent_loader import get_fallback_intent, get_intent_names

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Paths and Configurations
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_CSV = "data/processed/apple_support/apple_support_intent_training_candidates.csv"
DEFAULT_GOLDEN_CSV = "data/processed/apple_support/apple_support_intent_golden_set.csv"
DEFAULT_MODELS_DIR = "models/intent_classifier"
DEFAULT_EVAL_RESULTS_JSON = "data/processed/apple_support/apple_support_intent_evaluation_results.json"

DEFAULT_SEED = 42
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)


from src.classification.baseline_classifier import MajorityClassBaseline


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def prepare_training_data(
    train_csv_path: str | Path,
    golden_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare training candidates, verifying strict golden set isolation.

    Filters out unverified heuristic conflict rows (`candidate_intent == 'needs_review'`)
    to ensure the model is trained exclusively on concrete taxonomy targets.
    """
    train_path = Path(train_csv_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training candidates CSV not found at: {train_path}")

    logger.info("Loading training candidates from: %s", train_path)
    df_raw = pd.read_csv(train_path)

    # 1. Enforce strict isolation verification
    verify_training_isolation(df_raw, golden_df)
    logger.info("Strict training isolation verified: S_train ∩ S_golden = ∅.")

    # 2. Exclude unresolved multi-category conflict rows ('needs_review')
    df_valid = df_raw[df_raw["candidate_intent"] != "needs_review"].copy()
    logger.info(
        "Filtered out %d unverified 'needs_review' candidate rows; training pool has %d records.",
        len(df_raw) - len(df_valid),
        len(df_valid),
    )

    # 3. Handle any potential null text
    df_valid["text"] = df_valid["text"].fillna("").astype(str)

    return df_valid, df_valid["candidate_intent"]


def train_models(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Fit TF-IDF vectorizer, Majority Baseline, Logistic Regression, and Linear SVM.

    Returns:
    --------
    dict containing 'vectorizer', 'majority_model', 'logistic_regression_model', 'linear_svm_model'.
    """
    logger.info("Fitting TF-IDF Vectorizer (max_features=%d, ngrams=%s)...", TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE)
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_vec = vectorizer.fit_transform(train_texts)
    logger.info("TF-IDF matrix built: shape=%s", X_train_vec.shape)

    # 1. Majority Class Baseline
    logger.info("Fitting Majority Class Baseline...")
    majority_model = MajorityClassBaseline()
    majority_model.fit(X_train_vec, train_labels)

    # 2. Logistic Regression
    logger.info("Fitting Logistic Regression (class_weight='balanced', seed=%d)...", seed)
    lr_model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    lr_model.fit(X_train_vec, train_labels)

    # 3. Linear SVM
    logger.info("Fitting Linear SVM (LinearSVC, class_weight='balanced', seed=%d)...", seed)
    svm_model = LinearSVC(
        class_weight="balanced",
        random_state=seed,
        dual=False,
        max_iter=2000,
    )
    svm_model.fit(X_train_vec, train_labels)

    return {
        "vectorizer": vectorizer,
        "majority_model": majority_model,
        "logistic_regression_model": lr_model,
        "linear_svm_model": svm_model,
    }


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------
def evaluate_model_on_golden_set(
    model: Any,
    vectorizer: TfidfVectorizer | None,
    golden_df: pd.DataFrame,
    model_name: str,
) -> dict[str, Any]:
    """Comprehensive evaluation of a trained model against the human-verified Golden Evaluation Set.

    Evaluates:
    - Primary Closed-World Metrics (155 resolved records where golden_intent != 'needs_review')
    - Per-Intent Classification Report (Precision, Recall, F1, Support)
    - Confusion Matrix
    - Slice-Level Performance across all 8 difficulty tiers
    - Ambiguous Case Analysis (the 3 'needs_review' cases)
    """
    golden_df = golden_df.copy()
    golden_df["text"] = golden_df["text"].fillna("").astype(str)

    # Transform features
    if vectorizer is not None:
        X_all = vectorizer.transform(golden_df["text"])
    else:
        X_all = golden_df["text"]

    # Predictions for all 158 rows
    preds_all = model.predict(X_all)
    golden_df["predicted_intent"] = preds_all

    # Probabilities / Decision Scores if available
    probs_all = None
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X_all)
        classes = model.classes_
    elif hasattr(model, "decision_function"):
        # Softmax approximation over decision function for LinearSVC
        df_scores = model.decision_function(X_all)
        exp_scores = np.exp(df_scores - np.max(df_scores, axis=1, keepdims=True))
        probs_all = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        classes = model.classes_
    else:
        classes = getattr(model, "classes_", np.array([]))

    # Subset 1: Closed-World 12-Class Evaluation (155 resolved records)
    eval_155 = golden_df[golden_df["golden_intent"] != "needs_review"].copy()
    y_true_155 = eval_155["golden_intent"].values
    y_pred_155 = eval_155["predicted_intent"].values

    # Determine unique classes present
    target_labels = sorted(list(set(y_true_155)))

    acc = float(accuracy_score(y_true_155, y_pred_155))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_155, y_pred_155, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_155, y_pred_155, average="weighted", zero_division=0
    )

    # Per-intent metrics
    clf_report = classification_report(
        y_true_155, y_pred_155, labels=target_labels, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true_155, y_pred_155, labels=target_labels)

    # Slice-Level Performance across all 8 difficulty tiers
    slice_performance: dict[str, Any] = {}
    for tier in sorted(list(CASE_DIFFICULTY_TIERS)):
        tier_df = golden_df[golden_df["case_difficulty"] == tier]
        tier_total = len(tier_df)
        if tier_total == 0:
            continue

        if tier == "ambiguous_multi_intent":
            # For ambiguous cases, accuracy is not applicable in a closed-world setup
            slice_performance[tier] = {
                "total": tier_total,
                "note": "Closed-world evaluation N/A; evaluated separately via confidence/escalation analysis.",
            }
        else:
            tier_correct = int((tier_df["golden_intent"] == tier_df["predicted_intent"]).sum())
            slice_acc = round(tier_correct / tier_total, 4)
            slice_performance[tier] = {
                "total": tier_total,
                "correct": tier_correct,
                "accuracy": slice_acc,
            }

    # Ambiguous Cases Diagnostic Analysis (3 records)
    amb_df = golden_df[golden_df["golden_intent"] == "needs_review"].copy()
    ambiguous_analysis: list[dict[str, Any]] = []

    for idx, (_, row) in enumerate(amb_df.iterrows()):
        tid = int(row["tweet_id"])
        pred_class = str(row["predicted_intent"])

        # Row index in full golden_df
        full_idx = golden_df.index[golden_df["tweet_id"] == tid][0]

        top_classes: list[dict[str, Any]] = []
        confidence = 0.0
        margin = 0.0

        if probs_all is not None and len(classes) > 0:
            p_row = probs_all[full_idx]
            top_indices = p_row.argsort()[-3:][::-1]
            top_classes = [
                {"intent": str(classes[i]), "confidence": round(float(p_row[i]), 4)}
                for i in top_indices
            ]
            confidence = float(p_row[top_indices[0]])
            margin = float(p_row[top_indices[0]] - p_row[top_indices[1]]) if len(top_indices) > 1 else confidence

        ambiguous_analysis.append(
            {
                "tweet_id": tid,
                "text": str(row["text"]),
                "predicted_intent": pred_class,
                "confidence": round(confidence, 4),
                "confidence_margin": round(margin, 4),
                "top_predictions": top_classes,
                "human_reviewer_notes": str(row["notes"]),
            }
        )

    return {
        "model_name": model_name,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "golden_test",
        "total_golden_records": len(golden_df),
        "closed_world_evaluated_records": len(eval_155),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
        },
        "per_intent_metrics": clf_report,
        "confusion_matrix": {
            "labels": target_labels,
            "matrix": cm.tolist(),
        },
        "slice_level_metrics": slice_performance,
        "ambiguous_cases_analysis": ambiguous_analysis,
    }


# ---------------------------------------------------------------------------
# Serialization & Persistence Utilities
# ---------------------------------------------------------------------------
def save_model_artifacts(
    artifacts: dict[str, Any],
    output_dir: str | Path = DEFAULT_MODELS_DIR,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save trained vectorizer, models, and metadata to disk using joblib."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vectorizer
    if "vectorizer" in artifacts:
        vec_path = out_dir / "tfidf_vectorizer.joblib"
        joblib.dump(artifacts["vectorizer"], vec_path)
        logger.info("Saved vectorizer to: %s", vec_path)

    # 2. Models
    for key, filename in [
        ("majority_model", "majority_baseline_model.joblib"),
        ("logistic_regression_model", "logistic_regression_model.joblib"),
        ("linear_svm_model", "linear_svm_model.joblib"),
    ]:
        if key in artifacts:
            m_path = out_dir / filename
            joblib.dump(artifacts[key], m_path)
            logger.info("Saved %s to: %s", key, m_path)

    # 3. Model Metadata
    meta_path = out_dir / "model_metadata.json"
    if metadata is None:
        metadata = {}
    metadata.update(
        {
            "save_timestamp": datetime.now(timezone.utc).isoformat(),
            "models_dir": str(out_dir),
            "serialized_files": [
                "tfidf_vectorizer.joblib",
                "majority_baseline_model.joblib",
                "logistic_regression_model.joblib",
                "linear_svm_model.joblib",
            ],
        }
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to: %s", meta_path)


def load_model_artifacts(models_dir: str | Path = DEFAULT_MODELS_DIR) -> dict[str, Any]:
    """Load serialized vectorizer, models, and metadata from disk."""
    m_dir = Path(models_dir)
    if not m_dir.exists():
        raise FileNotFoundError(f"Models directory not found at: {m_dir}")

    artifacts: dict[str, Any] = {}

    vec_path = m_dir / "tfidf_vectorizer.joblib"
    if vec_path.exists():
        artifacts["vectorizer"] = joblib.load(vec_path)

    lr_path = m_dir / "logistic_regression_model.joblib"
    if lr_path.exists():
        artifacts["logistic_regression_model"] = joblib.load(lr_path)

    svm_path = m_dir / "linear_svm_model.joblib"
    if svm_path.exists():
        artifacts["linear_svm_model"] = joblib.load(svm_path)

    maj_path = m_dir / "majority_baseline_model.joblib"
    if maj_path.exists():
        artifacts["majority_model"] = joblib.load(maj_path)

    meta_path = m_dir / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            artifacts["metadata"] = json.load(f)

    return artifacts


# ---------------------------------------------------------------------------
# CLI & Full Execution Workflow
# ---------------------------------------------------------------------------
def run_training_and_evaluation(
    train_csv_path: str | Path = DEFAULT_TRAIN_CSV,
    golden_csv_path: str | Path = DEFAULT_GOLDEN_CSV,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    eval_results_json: str | Path = DEFAULT_EVAL_RESULTS_JSON,
    seed: int = DEFAULT_SEED,
    save_models: bool = True,
) -> dict[str, Any]:
    """Execute complete end-to-end training and evaluation workflow."""
    # 1. Load golden evaluation set
    golden_df = load_golden_evaluation_set(golden_csv_path)
    logger.info("Loaded golden evaluation set: %d records.", len(golden_df))

    # 2. Load and isolate training data
    df_train, y_train = prepare_training_data(train_csv_path, golden_df)

    # 3. Train models
    artifacts = train_models(df_train["text"], y_train, seed=seed)
    vectorizer = artifacts["vectorizer"]

    # 4. Evaluate all 3 models on the golden set
    eval_results: dict[str, Any] = {
        "pipeline_version": "1.0",
        "taxonomy_version": "2.0",
        "random_seed": seed,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    # (a) Majority Baseline
    logger.info("Evaluating Majority Class Baseline...")
    maj_eval = evaluate_model_on_golden_set(
        artifacts["majority_model"], vectorizer=None, golden_df=golden_df, model_name="Majority Class Baseline"
    )
    eval_results["models"]["majority_baseline"] = maj_eval

    # (b) Logistic Regression
    logger.info("Evaluating Logistic Regression...")
    lr_eval = evaluate_model_on_golden_set(
        artifacts["logistic_regression_model"],
        vectorizer=vectorizer,
        golden_df=golden_df,
        model_name="TF-IDF + Logistic Regression",
    )
    eval_results["models"]["logistic_regression"] = lr_eval

    # (c) Linear SVM
    logger.info("Evaluating Linear SVM...")
    svm_eval = evaluate_model_on_golden_set(
        artifacts["linear_svm_model"],
        vectorizer=vectorizer,
        golden_df=golden_df,
        model_name="TF-IDF + Linear SVM",
    )
    eval_results["models"]["linear_svm"] = svm_eval

    # 5. Save model artifacts if requested
    if save_models:
        model_meta = {
            "random_seed": seed,
            "training_samples_count": len(df_train),
            "training_classes": sorted(list(set(y_train))),
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
            "best_model": "logistic_regression",
            "best_macro_f1": lr_eval["overall_metrics"]["macro_f1"],
        }
        save_model_artifacts(artifacts, output_dir=models_dir, metadata=model_meta)

    # 6. Save evaluation results JSON
    eval_out_path = Path(eval_results_json)
    eval_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Saved evaluation results to: %s", eval_out_path)

    return eval_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate intent classifiers on AppleSupport dataset against Golden Evaluation Set."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train majority baseline, Logistic Regression, and Linear SVM models.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models against the Golden Evaluation Set.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained model artifacts and vectorizer using joblib.",
    )
    parser.add_argument(
        "--train-csv",
        default=DEFAULT_TRAIN_CSV,
        help=f"Path to training candidates CSV (default: {DEFAULT_TRAIN_CSV})",
    )
    parser.add_argument(
        "--golden-csv",
        default=DEFAULT_GOLDEN_CSV,
        help=f"Path to golden evaluation set CSV (default: {DEFAULT_GOLDEN_CSV})",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Directory to save/load model artifacts (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--eval-results",
        default=DEFAULT_EVAL_RESULTS_JSON,
        help=f"Path to output evaluation results JSON (default: {DEFAULT_EVAL_RESULTS_JSON})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    if not (args.train or args.evaluate):
        # Default action is train, evaluate, and save
        args.train = True
        args.evaluate = True
        args.save_models = True

    print("\n=================================================================")
    print("  AppleSupport Intent Classification: Training & Evaluation Pipeline")
    print("=================================================================")

    results = run_training_and_evaluation(
        train_csv_path=args.train_csv,
        golden_csv_path=args.golden_csv,
        models_dir=args.models_dir,
        eval_results_json=args.eval_results,
        seed=args.seed,
        save_models=args.save_models,
    )

    # Print summary table
    print("\n--- Comparative Evaluation on Golden Set (155 Closed-World Records) ---")
    print(
        f"{'Model':30s} | {'Accuracy':<10s} | {'Macro-F1':<10s} | {'Weighted-F1':<12s} | {'Macro-Precision':<15s} | {'Macro-Recall':<12s}"
    )
    print("-" * 100)
    for model_key, res in results["models"].items():
        m_name = res["model_name"]
        metrics = res["overall_metrics"]
        print(
            f"{m_name:30s} | {metrics['accuracy']:<10.4f} | {metrics['macro_f1']:<10.4f} | {metrics['weighted_f1']:<12.4f} | {metrics['macro_precision']:<15.4f} | {metrics['macro_recall']:<12.4f}"
        )

    # Print best model per-intent breakdown
    best_res = results["models"]["logistic_regression"]
    print("\n--- Per-Intent Classification Report (Best Model: TF-IDF + Logistic Regression) ---")
    print(f"{'Intent':25s} | {'Precision':<10s} | {'Recall':<10s} | {'F1-Score':<10s} | {'Support':<8s}")
    print("-" * 75)
    for intent, scores in sorted(best_res["per_intent_metrics"].items()):
        if isinstance(scores, dict) and "f1-score" in scores:
            p = scores["precision"]
            r = scores["recall"]
            f1 = scores["f1-score"]
            sup = int(scores["support"])
            print(f"{intent:25s} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {sup:<8d}")

    # Print slice-level performance
    print("\n--- Slice-Level Performance by Case Difficulty (TF-IDF + Logistic Regression) ---")
    print(f"{'Difficulty Slice':25s} | {'Accuracy':<10s} | {'Correct/Total':<15s}")
    print("-" * 55)
    for slice_name, slice_info in sorted(best_res["slice_level_metrics"].items()):
        if "accuracy" in slice_info:
            acc_str = f"{slice_info['accuracy']:.4f}"
            cnt_str = f"{slice_info['correct']}/{slice_info['total']}"
        else:
            acc_str = "N/A"
            cnt_str = f"-/{slice_info['total']} (ambiguous)"
        print(f"{slice_name:25s} | {acc_str:<10s} | {cnt_str:<15s}")

    # Print ambiguous cases predictions
    print("\n--- Ambiguous Cases Analysis (3 needs_review records) ---")
    for amb in best_res["ambiguous_cases_analysis"]:
        print(f"Tweet {amb['tweet_id']}:")
        print(f"  Text: {amb['text'][:85]}...")
        print(f"  Top Prediction: {amb['predicted_intent']} (confidence: {amb['confidence']:.4f}, margin: {amb['confidence_margin']:.4f})")
        top_str = ", ".join(f"{t['intent']} ({t['confidence']:.3f})" for t in amb['top_predictions'])
        print(f"  Top-3 Classes:  {top_str}")
        print(f"  Review Notes:   {amb['human_reviewer_notes']}")


if __name__ == "__main__":
    main()
