"""
src/classification/train_distilroberta.py
=========================================
Deep Transformer Fine-Tuning Pipeline (DistilRoBERTa) for AppleSupport Intent Classification.

Key Guarantees:
- Strict GPU Enforcement: Asserts torch.cuda.is_available() before training begins.
  Local CPU training is strictly prohibited per operational guidelines.
- Strict Training Isolation: Validates S_train ∩ S_golden = ∅ before fine-tuning.
- Input Text Exclusivity: Only initial customer problem statements (`text`) are used.
- Excludes 'needs_review' Heuristic Conflicts: Model is trained strictly on 11 concrete taxonomy targets.
- Class-Weight Balancing: Weighted Cross-Entropy loss addresses severe class imbalance.
- Artifact Persistence: Serializes fine-tuned weights, tokenizer, config, and training metadata.
- Golden Set Evaluation: Computes closed-world metrics, per-intent reports, slice metrics, and ambiguous diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.classification.build_golden_evaluation_set import (
    CASE_DIFFICULTY_TIERS,
    load_golden_evaluation_set,
    verify_training_isolation,
)
from src.classification.intent_loader import get_intent_names

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CSV = "data/processed/apple_support/apple_support_intent_training_candidates.csv"
DEFAULT_GOLDEN_CSV = "data/processed/apple_support/apple_support_intent_golden_set.csv"
DEFAULT_OUTPUT_DIR = "models/distilroberta_intent_classifier"
DEFAULT_EVAL_JSON = "data/processed/apple_support/apple_support_distilroberta_evaluation_results.json"

DEFAULT_MODEL_NAME = "distilroberta-base"
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_EPOCHS = 3
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01


def get_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return canonical label2id and id2label mappings for the 11 taxonomy classes."""
    intent_names = sorted(get_intent_names())
    label2id = {name: i for i, name in enumerate(intent_names)}
    id2label = {i: name for i, name in enumerate(intent_names)}
    return label2id, id2label


def verify_gpu_available() -> Dict[str, Any]:
    """Verify that a CUDA GPU is available for transformer training.

    Raises RuntimeError if CUDA is unavailable. Local CPU training is prohibited.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the current environment. "
            "Please ensure you are running in the Colab GPU runtime."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CRITICAL: CUDA GPU is not available! torch.cuda.is_available() returned False. "
            "Per operational requirements, DistilRoBERTa must be trained on a GPU in Google Colab. "
            "Local CPU training is strictly prohibited."
        )

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    total_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    gpu_info = {
        "cuda_available": True,
        "device_count": device_count,
        "device_name": device_name,
        "compute_capability": f"{device_capability[0]}.{device_capability[1]}",
        "total_memory_gb": total_memory_gb,
        "torch_version": torch.__version__,
    }
    logger.info("GPU Runtime Verified: %s (%s GB memory, CUDA capability %s)",
                device_name, total_memory_gb, gpu_info["compute_capability"])
    return gpu_info


def prepare_datasets(
    train_csv_path: str | Path,
    golden_df: pd.DataFrame,
    val_ratio: float = 0.10,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str], np.ndarray]:
    """Load training candidates, enforce isolation, filter out 'needs_review', and split train/val."""
    train_path = Path(train_csv_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training candidates CSV not found at: {train_path}")

    logger.info("Loading training candidates from: %s", train_path)
    df_raw = pd.read_csv(train_path)

    # 1. Strict mathematical isolation check: S_train ∩ S_golden = ∅
    verify_training_isolation(df_raw, golden_df)
    logger.info("Strict training isolation verified: zero overlap with golden set.")

    # 2. Exclude unresolved multi-category conflict rows ('needs_review')
    df_clean = df_raw[df_raw["candidate_intent"] != "needs_review"].copy()
    logger.info(
        "Filtered out %d 'needs_review' rows; clean candidate pool has %d records.",
        len(df_raw) - len(df_clean),
        len(df_clean),
    )

    # 3. Label mappings
    label2id, id2label = get_label_mappings()
    df_clean = df_clean[df_clean["candidate_intent"].isin(label2id.keys())].copy()
    df_clean["label"] = df_clean["candidate_intent"].map(label2id)
    df_clean["text"] = df_clean["text"].fillna("").astype(str)

    # 4. Compute balanced class weights for loss weighting
    class_counts = df_clean["label"].value_counts().sort_index()
    n_samples = len(df_clean)
    n_classes = len(label2id)
    weights = np.zeros(n_classes, dtype=np.float32)
    for c_id in range(n_classes):
        cnt = class_counts.get(c_id, 1)
        weights[c_id] = n_samples / (n_classes * cnt)
    logger.info("Computed balanced class weights for %d classes.", n_classes)

    # 5. Stratified train/validation split
    df_train, df_val = train_test_split(
        df_clean,
        test_size=val_ratio,
        random_state=seed,
        stratify=df_clean["label"],
    )
    logger.info("Dataset split: %d train samples, %d validation samples.", len(df_train), len(df_val))

    return df_train, df_val, label2id, id2label, weights


def train_distilroberta(
    train_csv_path: str | Path = DEFAULT_TRAIN_CSV,
    golden_csv_path: str | Path = DEFAULT_GOLDEN_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Fine-tune DistilRoBERTa on GPU using HuggingFace Trainer with class-weighted loss."""
    # 1. Enforce GPU availability
    gpu_info = verify_gpu_available()

    # 2. Load golden evaluation set
    golden_df = load_golden_evaluation_set(golden_csv_path)

    # 3. Prepare data and verify isolation
    df_train, df_val, label2id, id2label, class_weights = prepare_datasets(
        train_csv_path=train_csv_path,
        golden_df=golden_df,
        seed=seed,
    )

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)

    # 4. Tokenizer
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Convert to HuggingFace Datasets
    ds_train = Dataset.from_pandas(df_train[["text", "label"]])
    ds_val = Dataset.from_pandas(df_val[["text", "label"]])

    logger.info("Tokenizing train and validation datasets...")
    ds_train_tok = ds_train.map(tokenize_batch, batched=True)
    ds_val_tok = ds_val.map(tokenize_batch, batched=True)

    # 5. Model
    logger.info("Initializing sequence classification model: %s (%d classes)", model_name, len(label2id))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # 6. Custom Trainer with Class-Weighted Cross-Entropy Loss
    class WeightedTrainer(Trainer):
        def __init__(self, *args: Any, class_weights_tensor: torch.Tensor | None = None, **kwargs: Any) -> None:
            # Dynamically map tokenizer <-> processing_class across all transformers releases
            import inspect
            trainer_params = inspect.signature(Trainer.__init__).parameters
            if "tokenizer" in kwargs and "processing_class" in trainer_params:
                kwargs["processing_class"] = kwargs.pop("tokenizer")
            elif "processing_class" in kwargs and "tokenizer" in trainer_params:
                kwargs["tokenizer"] = kwargs.pop("processing_class")
            elif "tokenizer" in kwargs and "tokenizer" not in trainer_params and "processing_class" not in trainer_params:
                tok = kwargs.pop("tokenizer")
                self.processing_class = tok
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights_tensor

        def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if self.class_weights is not None and logits is not None and labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            else:
                loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }

    # 7. Training Arguments with Dynamic Signature Inspection (Universal Version Compatibility)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_path / "checkpoints"

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters

    args_dict: Dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "num_train_epochs": epochs,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_steps": 50,
        "seed": seed,
        "report_to": "none",
    }

    # fp16 support
    if "fp16" in sig and torch.cuda.is_available():
        args_dict["fp16"] = True

    # Version-safe evaluation strategy (eval_strategy in newer, evaluation_strategy in older)
    if "eval_strategy" in sig:
        args_dict["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        args_dict["evaluation_strategy"] = "epoch"

    # Version-safe save strategy
    if "save_strategy" in sig:
        args_dict["save_strategy"] = "epoch"

    # Version-safe warmup (warmup_ratio in newer, warmup_steps in older)
    total_steps = max(1, (len(ds_train_tok) // batch_size) * epochs)
    if "warmup_ratio" in sig:
        args_dict["warmup_ratio"] = DEFAULT_WARMUP_RATIO
    elif "warmup_steps" in sig:
        args_dict["warmup_steps"] = int(total_steps * DEFAULT_WARMUP_RATIO)

    # Filter arguments to strictly match supported parameters of installed transformers
    valid_args = {k: v for k, v in args_dict.items() if k in sig}
    training_args = TrainingArguments(**valid_args)

    # Resolve trainer parameters dynamically
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": ds_train_tok,
        "eval_dataset": ds_val_tok,
        "compute_metrics": compute_metrics,
        "class_weights_tensor": weights_tensor,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    logger.info("Starting DistilRoBERTa fine-tuning on GPU...")
    train_result = trainer.train()
    logger.info("Training completed. Global steps: %d, Loss: %.4f",
                train_result.global_step, train_result.training_loss)

    # 8. Save Model and Tokenizer Artifacts
    logger.info("Saving fine-tuned model and tokenizer to: %s", out_path)
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # 9. Save Training Metadata
    metadata = {
        "model_architecture": "DistilRoBERTa",
        "base_model_name": model_name,
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": id2label,
        "training_samples_count": len(df_train),
        "validation_samples_count": len(df_val),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "random_seed": seed,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_environment": gpu_info,
        "training_loss": float(train_result.training_loss),
        "global_steps": int(train_result.global_step),
    }
    with open(out_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model, tokenizer, metadata


def evaluate_transformer_on_golden_set(
    model: Any,
    tokenizer: Any,
    golden_df: pd.DataFrame,
    id2label: Dict[int, str],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, Any]:
    """Comprehensive evaluation of the fine-tuned DistilRoBERTa against the Golden Evaluation Set."""
    import torch
    from scipy.special import softmax

    golden_df = golden_df.copy()
    golden_df["text"] = golden_df["text"].fillna("").astype(str)

    model.eval()
    device = next(model.parameters()).device

    # Tokenize full golden set (158 rows)
    encoded = tokenizer(
        golden_df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()

    probs = softmax(logits, axis=1)
    pred_indices = np.argmax(probs, axis=1)
    golden_df["predicted_intent"] = [id2label[i] for i in pred_indices]

    classes = np.array([id2label[i] for i in range(len(id2label))])

    # Closed-World 11-Class Evaluation (155 resolved records)
    eval_155 = golden_df[golden_df["golden_intent"] != "needs_review"].copy()
    y_true_155 = eval_155["golden_intent"].values
    y_pred_155 = eval_155["predicted_intent"].values
    target_labels = sorted(list(set(y_true_155)))

    acc = float(accuracy_score(y_true_155, y_pred_155))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_155, y_pred_155, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_155, y_pred_155, average="weighted", zero_division=0
    )

    clf_report = classification_report(
        y_true_155, y_pred_155, labels=target_labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true_155, y_pred_155, labels=target_labels)

    # Slice-Level Performance across all 8 difficulty tiers
    slice_performance: Dict[str, Any] = {}
    for tier in sorted(list(CASE_DIFFICULTY_TIERS)):
        tier_df = golden_df[golden_df["case_difficulty"] == tier]
        tier_total = len(tier_df)
        if tier_total == 0:
            continue
        if tier == "ambiguous_multi_intent":
            slice_performance[tier] = {
                "total": tier_total,
                "note": "Evaluated separately via confidence and margin analysis.",
            }
        else:
            correct = int((tier_df["golden_intent"] == tier_df["predicted_intent"]).sum())
            slice_performance[tier] = {
                "total": tier_total,
                "correct": correct,
                "accuracy": round(correct / tier_total, 4),
            }

    # Ambiguous Cases Diagnostic Analysis (3 records)
    amb_df = golden_df[golden_df["golden_intent"] == "needs_review"].copy()
    ambiguous_analysis: List[Dict[str, Any]] = []

    for _, row in amb_df.iterrows():
        tid = int(row["tweet_id"])
        pred_class = str(row["predicted_intent"])
        full_idx = golden_df.index[golden_df["tweet_id"] == tid][0]

        p_row = probs[full_idx]
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
        "model_name": "DistilRoBERTa (Transformer)",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate DistilRoBERTa on GPU.")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--golden-csv", default=DEFAULT_GOLDEN_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-results", default=DEFAULT_EVAL_JSON)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    print("\n=================================================================")
    print("  AppleSupport DistilRoBERTa Fine-Tuning & Evaluation Pipeline")
    print("=================================================================")

    # Train model on GPU
    model, tokenizer, metadata = train_distilroberta(
        train_csv_path=args.train_csv,
        golden_csv_path=args.golden_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )

    # Evaluate on Golden Set
    golden_df = load_golden_evaluation_set(args.golden_csv)
    _, id2label = get_label_mappings()
    eval_results = evaluate_transformer_on_golden_set(
        model=model,
        tokenizer=tokenizer,
        golden_df=golden_df,
        id2label=id2label,
        max_length=args.max_length,
    )

    # Save evaluation results JSON
    eval_path = Path(args.eval_results)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Saved evaluation results to: %s", eval_path)

    # Print summary
    m = eval_results["overall_metrics"]
    print("\n--- DistilRoBERTa Performance on Golden Set (155 Closed-World Records) ---")
    print(f"Accuracy:        {m['accuracy']:.4f}")
    print(f"Macro-F1:        {m['macro_f1']:.4f}")
    print(f"Weighted-F1:     {m['weighted_f1']:.4f}")
    print(f"Macro-Precision: {m['macro_precision']:.4f}")
    print(f"Macro-Recall:    {m['macro_recall']:.4f}")


if __name__ == "__main__":
    main()
