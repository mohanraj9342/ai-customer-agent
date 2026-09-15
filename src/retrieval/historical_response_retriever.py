"""
src/retrieval/historical_response_retriever.py
=============================================
Phase 11 — Real-Time Semantic Historical Response Retriever & Grounding Engine.

Indexes customer messages and pairs them with grounded AppleSupport replies.
Given an incoming customer inquiry, retrieves top-k most semantically similar
historical inquiries and their verified brand replies using exact cosine similarity
over all-MiniLM-L6-v2 normalized embeddings.

Zero Generative Hallucination:
All retrieved replies are grounded, verbatim historical responses from AppleSupport.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_CSV = "data/processed/apple_support/apple_support_retrieval_corpus.csv"
DEFAULT_EMBEDDINGS_NPY = "data/processed/apple_support/apple_support_retrieval_embeddings.npy"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved historical customer-brand interaction record."""

    rank: int
    similarity_score: float
    thread_id: str
    customer_tweet_id: int
    customer_text: str
    brand_tweet_id: int
    brand_text: str
    inferred_intent: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoricalResponseRetriever:
    """
    In-memory vectorized semantic search engine over historical AppleSupport responses.
    """

    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS_CSV,
        embeddings_path: str | Path = DEFAULT_EMBEDDINGS_NPY,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        self.device = device

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus CSV not found at: {self.corpus_path}")
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings matrix not found at: {self.embeddings_path}")

        logger.info("Loading retrieval corpus from: %s", self.corpus_path)
        self.df = pd.read_csv(self.corpus_path)

        logger.info("Loading precomputed embeddings from: %s", self.embeddings_path)
        self.embeddings = np.load(self.embeddings_path)

        if len(self.df) != len(self.embeddings):
            raise ValueError(
                f"Mismatch between corpus rows ({len(self.df)}) and embeddings rows ({len(self.embeddings)})"
            )

        logger.info(
            "Retriever initialized with %d records (embedding_dim=%d).",
            len(self.df),
            self.embeddings.shape[1],
        )

        logger.info("Loading SentenceTransformer model '%s' on %s...", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def corpus_size(self) -> int:
        return len(self.df)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = -1.0,
        deduplicate_customer_text: bool = False,
        filter_intent: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k most semantically similar historical responses.

        Args:
            query: Input customer message string.
            top_k: Maximum number of historical records to return.
            min_score: Minimum similarity threshold (cosine score between -1.0 and 1.0).
            deduplicate_customer_text: If True, avoids returning multiple responses for
                                       identical customer inquiry text.
            filter_intent: Optional intent category to restrict candidates.

        Returns:
            List of RetrievalResult objects ranked by similarity score descending.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        # Encode and L2-normalize query vector
        q_vec = self.model.encode(
            cleaned_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Compute cosine similarity via dot product (since both vectors are unit normalized)
        # S = X @ q
        scores = np.dot(self.embeddings, q_vec)

        # Apply intent filter if specified
        if filter_intent:
            mask = self.df["inferred_intent"].values == filter_intent
            scores = np.where(mask, scores, -np.inf)

        # Apply minimum threshold
        valid_indices = np.where(scores >= min_score)[0]
        if len(valid_indices) == 0:
            return []

        # Sort candidate scores descending
        candidate_scores = scores[valid_indices]
        sorted_order = np.argsort(-candidate_scores)
        ranked_indices = valid_indices[sorted_order]

        results: List[RetrievalResult] = []
        seen_texts: set[str] = set()

        for idx in ranked_indices:
            score = float(scores[idx])
            row = self.df.iloc[idx]
            c_text = str(row["customer_text"])

            if deduplicate_customer_text:
                normalized = c_text.strip().lower()
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)

            results.append(
                RetrievalResult(
                    rank=len(results) + 1,
                    similarity_score=round(score, 4),
                    thread_id=str(row["thread_id"]),
                    customer_tweet_id=int(row["customer_tweet_id"]),
                    customer_text=c_text,
                    brand_tweet_id=int(row["brand_tweet_id"]),
                    brand_text=str(row["brand_text"]),
                    inferred_intent=str(row.get("inferred_intent", "unknown_other")),
                )
            )

            if len(results) >= top_k:
                break

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        min_score: float = -1.0,
        batch_size: int = 64,
    ) -> List[List[RetrievalResult]]:
        """Batch retrieval for evaluating across multiple test queries."""
        if not queries:
            return []

        # Encode queries in batch
        q_vecs = self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Scores matrix: (N_queries, N_corpus) = Q @ X.T
        similarity_matrix = np.dot(q_vecs, self.embeddings.T)

        all_results: List[List[RetrievalResult]] = []
        for i, q in enumerate(queries):
            cleaned = (q or "").strip()
            if not cleaned:
                all_results.append([])
                continue

            scores = similarity_matrix[i]
            valid_indices = np.where(scores >= min_score)[0]
            if len(valid_indices) == 0:
                all_results.append([])
                continue

            sorted_order = np.argsort(-scores[valid_indices])
            top_idx = valid_indices[sorted_order][:top_k]

            query_results: List[RetrievalResult] = []
            for rank_num, idx in enumerate(top_idx, start=1):
                row = self.df.iloc[idx]
                query_results.append(
                    RetrievalResult(
                        rank=rank_num,
                        similarity_score=round(float(scores[idx]), 4),
                        thread_id=str(row["thread_id"]),
                        customer_tweet_id=int(row["customer_tweet_id"]),
                        customer_text=str(row["customer_text"]),
                        brand_tweet_id=int(row["brand_tweet_id"]),
                        brand_text=str(row["brand_text"]),
                        inferred_intent=str(row.get("inferred_intent", "unknown_other")),
                    )
                )
            all_results.append(query_results)

        return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the AppleSupport Historical Response Retriever.")
    parser.add_argument("--query", type=str, required=True, help="Customer inquiry text to retrieve responses for.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top historical cases to retrieve.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum cosine similarity score.")
    parser.add_argument("--corpus-csv", default=DEFAULT_CORPUS_CSV)
    parser.add_argument("--embeddings-npy", default=DEFAULT_EMBEDDINGS_NPY)
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by customer text.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    retriever = HistoricalResponseRetriever(
        corpus_path=args.corpus_csv,
        embeddings_path=args.embeddings_npy,
    )

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        deduplicate_customer_text=args.dedup,
    )

    print("\n" + "=" * 80)
    print(f"Query: \"{args.query}\"")
    print(f"Top {len(results)} Historical Matches (from {retriever.corpus_size:,} isolated records):")
    print("=" * 80)

    for res in results:
        print(f"\n[Rank {res.rank}] Similarity: {res.similarity_score:.4f} | Intent: {res.inferred_intent}")
        print(f" Thread ID:       {res.thread_id}")
        print(f" Customer Tweet:  {res.customer_tweet_id}")
        print(f" Customer Query:  {res.customer_text}")
        print(f" Brand Tweet:     {res.brand_tweet_id}")
        print(f" Brand Reply:     {res.brand_text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
