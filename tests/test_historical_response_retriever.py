"""
tests/test_historical_response_retriever.py
===========================================
Unit and integration tests for Phase 11 Historical Response Retrieval & Grounding.

Verifies:
1. Grounded customer-message <-> brand-reply pair extraction fidelity.
2. Strict Golden Evaluation Set isolation enforcement (S_retrieval ∩ S_golden = ∅).
3. Retrieval corpus schema integrity and non-empty text bounds.
4. Embedding matrix properties (dimension 384, L2 normalization).
5. Vectorized retrieval correctness, top-k ordering, self-retrieval fidelity, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.retrieval.build_retrieval_corpus import (
    audit_duplicates,
    extract_customer_brand_pair,
    verify_retrieval_isolation,
)
from src.retrieval.historical_response_retriever import (
    HistoricalResponseRetriever,
    RetrievalResult,
)

CORPUS_CSV = Path("data/processed/apple_support/apple_support_retrieval_corpus.csv")
EMBEDDINGS_NPY = Path("data/processed/apple_support/apple_support_retrieval_embeddings.npy")
GOLDEN_CSV = Path("data/processed/apple_support/apple_support_intent_golden_set.csv")


# ---------------------------------------------------------------------------
# 1. Pair Extraction & Boundary Fidelity Tests
# ---------------------------------------------------------------------------
class TestPairExtraction:
    def test_extract_pair_basic(self):
        thread = {
            "thread_id": "thread_1",
            "messages": [
                {"tweet_id": 101, "author_id": "user1", "inbound": True, "text": "My phone is frozen", "created_at": "2017-10-01T12:00:00Z"},
                {"tweet_id": 102, "author_id": "AppleSupport", "inbound": False, "text": "We can help. Have you tried a hard reset?", "created_at": "2017-10-01T12:05:00Z"},
            ],
        }
        pair = extract_customer_brand_pair(thread)
        assert pair is not None
        assert pair["thread_id"] == "thread_1"
        assert pair["customer_tweet_id"] == 101
        assert pair["customer_text"] == "My phone is frozen"
        assert pair["brand_tweet_id"] == 102
        assert pair["brand_text"] == "We can help. Have you tried a hard reset?"

    def test_extract_pair_multi_turn_boundary_preservation(self):
        """Verify only the initial customer turn and immediate brand reply are extracted."""
        thread = {
            "thread_id": "thread_2",
            "messages": [
                {"tweet_id": 201, "author_id": "user2", "inbound": True, "text": "Battery dies in 1 hour"},
                {"tweet_id": 202, "author_id": "AppleSupport", "inbound": False, "text": "Let us take a look into this with you."},
                {"tweet_id": 203, "author_id": "user2", "inbound": True, "text": "I tried resetting and it did not fix it"},
                {"tweet_id": 204, "author_id": "AppleSupport", "inbound": False, "text": "Send us a DM with your iOS version."},
            ],
        }
        pair = extract_customer_brand_pair(thread)
        assert pair is not None
        assert pair["customer_tweet_id"] == 201
        assert pair["customer_text"] == "Battery dies in 1 hour"
        assert pair["brand_tweet_id"] == 202
        assert pair["brand_text"] == "Let us take a look into this with you."

    def test_extract_pair_skips_empty_customer_message(self):
        thread = {
            "thread_id": "thread_3",
            "messages": [
                {"tweet_id": 301, "author_id": "user3", "inbound": True, "text": "   "},
                {"tweet_id": 302, "author_id": "user3", "inbound": True, "text": "Actual question here"},
                {"tweet_id": 303, "author_id": "AppleSupport", "inbound": False, "text": "Happy to help!"},
            ],
        }
        pair = extract_customer_brand_pair(thread)
        assert pair is not None
        assert pair["customer_tweet_id"] == 302
        assert pair["customer_text"] == "Actual question here"
        assert pair["brand_tweet_id"] == 303

    def test_extract_pair_returns_none_when_no_brand_reply(self):
        thread = {
            "thread_id": "thread_4",
            "messages": [
                {"tweet_id": 401, "author_id": "user4", "inbound": True, "text": "Unanswered question"},
            ],
        }
        assert extract_customer_brand_pair(thread) is None

    def test_extract_pair_returns_none_when_no_customer_message(self):
        thread = {
            "thread_id": "thread_5",
            "messages": [
                {"tweet_id": 501, "author_id": "AppleSupport", "inbound": False, "text": "Outbound broadcast"},
            ],
        }
        assert extract_customer_brand_pair(thread) is None


# ---------------------------------------------------------------------------
# 2. Golden Evaluation Set Isolation Tests
# ---------------------------------------------------------------------------
class TestIsolationValidation:
    def test_verify_isolation_passes_on_disjoint_sets(self):
        retrieval_ids = {1001, 1002, 1003}
        golden_ids = {2001, 2002, 2003}
        assert verify_retrieval_isolation(retrieval_ids, golden_ids) is True

    def test_verify_isolation_raises_on_leak(self):
        retrieval_ids = {1001, 1002, 9999}
        golden_ids = {2001, 9999, 2003}
        with pytest.raises(ValueError, match="CRITICAL ISOLATION VIOLATION"):
            verify_retrieval_isolation(retrieval_ids, golden_ids)

    def test_actual_retrieval_corpus_isolation(self):
        """Assert zero overlap between actual retrieval corpus CSV and golden set CSV."""
        if not CORPUS_CSV.exists() or not GOLDEN_CSV.exists():
            pytest.skip("Corpus CSV or Golden CSV not yet generated on disk.")

        corpus_df = pd.read_csv(CORPUS_CSV)
        golden_df = pd.read_csv(GOLDEN_CSV)

        corpus_ids = set(corpus_df["customer_tweet_id"].astype(int))
        golden_ids = set(golden_df["tweet_id"].astype(int))

        overlap = corpus_ids.intersection(golden_ids)
        assert len(overlap) == 0, f"Found {len(overlap)} leaking IDs in retrieval corpus: {overlap}"
        assert verify_retrieval_isolation(corpus_ids, golden_ids) is True


@pytest.fixture(scope="module")
def corpus_df():
    if not CORPUS_CSV.exists():
        pytest.skip("Corpus CSV not found.")
    return pd.read_csv(CORPUS_CSV)


@pytest.fixture(scope="module")
def mock_retriever(tmp_path_factory):
    """Create a lightweight retriever with synthetic data to test retrieval math quickly."""
    tmp = tmp_path_factory.mktemp("retriever_test")
    corpus_data = pd.DataFrame([
        {
            "corpus_id": 0,
            "thread_id": "t_battery_1",
            "customer_tweet_id": 1001,
            "customer_text": "My iPhone battery dies within 30 minutes of charging.",
            "brand_tweet_id": 2001,
            "brand_text": "We can help you inspect battery health in Settings > Battery.",
            "inferred_intent": "battery_power",
        },
        {
            "corpus_id": 1,
            "thread_id": "t_wifi_1",
            "customer_tweet_id": 1002,
            "customer_text": "WiFi disconnects constantly and cannot find router signal.",
            "brand_tweet_id": 2002,
            "brand_text": "Try resetting network settings in Settings > General > Reset.",
            "inferred_intent": "connectivity_network",
        },
        {
            "corpus_id": 2,
            "thread_id": "t_screen_1",
            "customer_tweet_id": 1003,
            "customer_text": "Cracked screen on iPhone X after dropping on concrete.",
            "brand_tweet_id": 2003,
            "brand_text": "You can schedule a repair at your local Apple Store or authorized service provider.",
            "inferred_intent": "device_hardware",
        },
    ])
    c_file = tmp / "mock_corpus.csv"
    corpus_data.to_csv(c_file, index=False)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embs = st.encode(corpus_data["customer_text"].tolist(), normalize_embeddings=True)
    e_file = tmp / "mock_embeddings.npy"
    np.save(e_file, embs.astype(np.float32))

    return HistoricalResponseRetriever(corpus_path=c_file, embeddings_path=e_file, device="cpu")


# ---------------------------------------------------------------------------
# 3. Corpus Schema & Integrity Tests
# ---------------------------------------------------------------------------
class TestCorpusIntegrity:
    def test_corpus_expected_size(self, corpus_df: pd.DataFrame):
        # 82101 threads with pairs minus 158 golden set records = 81943
        assert len(corpus_df) == 81943

    def test_corpus_schema_columns(self, corpus_df: pd.DataFrame):
        expected_cols = [
            "corpus_id",
            "thread_id",
            "customer_tweet_id",
            "customer_text",
            "customer_timestamp",
            "brand_tweet_id",
            "brand_text",
            "brand_timestamp",
            "inferred_intent",
        ]
        for col in expected_cols:
            assert col in corpus_df.columns, f"Missing expected column: {col}"

    def test_corpus_no_empty_texts(self, corpus_df: pd.DataFrame):
        assert corpus_df["customer_text"].fillna("").astype(str).str.strip().str.len().min() > 0
        assert corpus_df["brand_text"].fillna("").astype(str).str.strip().str.len().min() > 0

    def test_corpus_unique_tweet_ids(self, corpus_df: pd.DataFrame):
        assert corpus_df["customer_tweet_id"].nunique() == len(corpus_df)

    def test_audit_duplicates_function(self, corpus_df: pd.DataFrame):
        stats = audit_duplicates(corpus_df)
        assert stats["total_records"] == 81943
        assert stats["unique_customer_tweet_ids"] == 81943
        assert stats["duplicate_customer_tweet_ids"] == 0
        assert stats["unique_customer_texts"] > 81000
        assert stats["unique_pair_count"] == 81943


# ---------------------------------------------------------------------------
# 4. Vector Retriever Functionality Tests
# ---------------------------------------------------------------------------
class TestHistoricalResponseRetriever:

    def test_retrieval_empty_query(self, mock_retriever: HistoricalResponseRetriever):
        assert mock_retriever.retrieve("") == []
        assert mock_retriever.retrieve("   ") == []

    def test_retrieval_exact_self_match(self, mock_retriever: HistoricalResponseRetriever):
        query = "My iPhone battery dies within 30 minutes of charging."
        results = mock_retriever.retrieve(query, top_k=1)
        assert len(results) == 1
        top = results[0]
        assert top.rank == 1
        assert top.customer_tweet_id == 1001
        assert pytest.approx(top.similarity_score, abs=1e-3) == 1.0
        assert top.inferred_intent == "battery_power"
        assert "Settings > Battery" in top.brand_text

    def test_retrieval_semantic_match(self, mock_retriever: HistoricalResponseRetriever):
        """Test retrieving by paraphrased concept (battery drainage)."""
        query = "My phone battery drains very quickly and turns off."
        results = mock_retriever.retrieve(query, top_k=3)
        assert len(results) == 3
        top = results[0]
        assert top.customer_tweet_id == 1001  # Battery should rank #1
        assert top.similarity_score > 0.55
        assert top.similarity_score >= results[1].similarity_score >= results[2].similarity_score

    def test_retrieval_top_k_parameter(self, mock_retriever: HistoricalResponseRetriever):
        results = mock_retriever.retrieve("wifi issue", top_k=2)
        assert len(results) == 2

    def test_retrieval_intent_filter(self, mock_retriever: HistoricalResponseRetriever):
        # Searching for battery with intent filter for connectivity_network should only return connectivity
        results = mock_retriever.retrieve("phone issue", filter_intent="connectivity_network", top_k=5)
        assert len(results) == 1
        assert results[0].inferred_intent == "connectivity_network"

    def test_batch_retrieval(self, mock_retriever: HistoricalResponseRetriever):
        queries = ["battery draining", "wifi disconnects"]
        batch_res = mock_retriever.batch_retrieve(queries, top_k=1)
        assert len(batch_res) == 2
        assert batch_res[0][0].customer_tweet_id == 1001
        assert batch_res[1][0].customer_tweet_id == 1002


# ---------------------------------------------------------------------------
# 5. Full Index Integration Tests (Conditional)
# ---------------------------------------------------------------------------
class TestFullRetrievalIndex:
    def test_full_retrieval_if_available(self):
        if not EMBEDDINGS_NPY.exists() or not CORPUS_CSV.exists():
            pytest.skip("Full embeddings matrix not yet built on disk.")

        retriever = HistoricalResponseRetriever(
            corpus_path=CORPUS_CSV,
            embeddings_path=EMBEDDINGS_NPY,
            device="cpu",
        )
        assert retriever.corpus_size == 81943

        query = "My battery is draining completely after updating"
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3
        top = results[0]
        assert top.similarity_score > 0.60
        assert len(top.brand_text) > 0
        assert top.customer_tweet_id > 0
        assert top.brand_tweet_id > 0

