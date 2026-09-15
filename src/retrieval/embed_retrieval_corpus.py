"""
src/retrieval/embed_retrieval_corpus.py
======================================
Phase 11 — Dense Vector Indexing Engine with all-MiniLM-L6-v2.

Encodes the historical customer inquiry texts in the retrieval corpus into
384-dimensional dense semantic vectors using sentence-transformers/all-MiniLM-L6-v2.
Applies L2 normalization to enable exact cosine similarity via matrix dot products.

Outputs:
- data/processed/apple_support/apple_support_retrieval_embeddings.npy (~125 MB)
- data/processed/apple_support/apple_support_embedding_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_CSV = "data/processed/apple_support/apple_support_retrieval_corpus.csv"
DEFAULT_OUTPUT_NPY = "data/processed/apple_support/apple_support_retrieval_embeddings.npy"
DEFAULT_OUTPUT_META = "data/processed/apple_support/apple_support_embedding_metadata.json"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 256


def embed_retrieval_corpus(
    corpus_csv_path: str | Path = DEFAULT_CORPUS_CSV,
    output_npy_path: str | Path = DEFAULT_OUTPUT_NPY,
    output_meta_path: str | Path = DEFAULT_OUTPUT_META,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Encode customer messages from retrieval corpus into normalized dense embeddings.
    """
    c_path = Path(corpus_csv_path)
    if not c_path.exists():
        raise FileNotFoundError(f"Retrieval corpus CSV not found at: {c_path}")

    logger.info("Loading retrieval corpus from: %s", c_path)
    df = pd.read_csv(c_path)
    if limit is not None and limit > 0:
        logger.info("Applying row limit of %d records for embedding generation.", limit)
        df = df.head(limit).copy()

    total_records = len(df)
    logger.info("Total customer messages to encode: %d", total_records)
    texts = df["customer_text"].fillna("").astype(str).tolist()

    logger.info("Loading SentenceTransformer model '%s' on device: %s", model_name, device)
    t0_load = time.time()
    model = SentenceTransformer(model_name, device=device)
    logger.info("Model loaded in %.2f seconds.", time.time() - t0_load)

    logger.info(
        "Starting dense vector encoding (batch_size=%d, normalize_embeddings=True)...",
        batch_size,
    )
    t0_encode = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    encode_duration = time.time() - t0_encode
    rate = total_records / max(0.001, encode_duration)
    logger.info(
        "Encoding complete! Processed %d records in %.2f seconds (%.1f records/sec).",
        total_records,
        encode_duration,
        rate,
    )

    # Validate shape and normalization
    if embeddings.shape != (total_records, 384):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}, expected ({total_records}, 384)")

    # Assert L2 unit norm (within float tolerance)
    norms = np.linalg.norm(embeddings, axis=1)
    norm_min, norm_max = float(norms.min()), float(norms.max())
    if not (0.99 <= norm_min <= 1.01 and 0.99 <= norm_max <= 1.01):
        logger.warning("Embeddings norm range [%.4f, %.4f] deviates from 1.0", norm_min, norm_max)

    # Save .npy file
    out_npy = Path(output_npy_path)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving embeddings matrix to: %s", out_npy)
    np.save(out_npy, embeddings.astype(np.float32))

    # Compute sha256 checksum
    with open(out_npy, "rb") as f:
        npy_sha256 = hashlib.sha256(f.read()).hexdigest()

    file_size_mb = out_npy.stat().st_size / (1024 * 1024)
    logger.info("Embeddings saved: %.2f MB | SHA256: %s", file_size_mb, npy_sha256[:16])

    metadata: Dict[str, Any] = {
        "pipeline_phase": "Phase 11: Historical Response Retrieval & Grounding",
        "embedding_model": model_name,
        "embedding_dimension": 384,
        "total_records_indexed": total_records,
        "batch_size": batch_size,
        "device": device,
        "normalized_l2": True,
        "encode_duration_seconds": round(encode_duration, 2),
        "records_per_second": round(rate, 1),
        "embeddings_file": str(out_npy),
        "embeddings_file_size_mb": round(file_size_mb, 2),
        "embeddings_sha256": npy_sha256,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_corpus_file": str(c_path),
    }

    out_meta = Path(output_meta_path)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved embedding metadata to: %s", out_meta)

    return embeddings, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense vector indexing for AppleSupport retrieval corpus.")
    parser.add_argument("--corpus-csv", default=DEFAULT_CORPUS_CSV)
    parser.add_argument("--out-npy", default=DEFAULT_OUTPUT_NPY)
    parser.add_argument("--out-meta", default=DEFAULT_OUTPUT_META)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    embed_retrieval_corpus(
        corpus_csv_path=args.corpus_csv,
        output_npy_path=args.out_npy,
        output_meta_path=args.out_meta,
        model_name=args.model_name,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
