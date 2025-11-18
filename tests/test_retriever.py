"""Tests for the document retrieval module."""

import tempfile
from pathlib import Path

import pytest

from rag.ingest import DocumentIngester
from rag.retriever import DocumentRetriever


class TestDocumentRetriever:
    """Test cases for DocumentRetriever class."""

    @pytest.fixture
    def indexed_kb(self):
        """Create and index a temporary knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "kb"
            persist_dir = Path(tmpdir) / "chroma_db"
            kb_path.mkdir()

            # Create test documents with distinct content
            (kb_path / "python.md").write_text(
                "# Python Programming\n\n"
                "Python is a high-level programming language known for its "
                "simplicity and readability. It supports multiple programming "
                "paradigms including procedural, object-oriented, and functional "
                "programming. Python uses indentation for code blocks."
            )

            (kb_path / "rag.md").write_text(
                "# RAG Systems\n\n"
                "RAG stands for Retrieval-Augmented Generation. It combines "
                "information retrieval with text generation to produce more "
                "accurate and grounded responses. RAG systems first retrieve "
                "relevant documents from a knowledge base, then use them as "
                "context for a language model."
            )

            (kb_path / "vectors.md").write_text(
                "# Vector Embeddings\n\n"
                "Vector embeddings are dense numerical representations of text "
                "that capture semantic meaning. Similar texts have similar "
                "embeddings. They are used in semantic search, recommendation "
                "systems, and clustering applications."
            )

            # Ingest documents
            ingester = DocumentIngester(
                kb_path=kb_path,
                persist_dir=persist_dir,
                chunk_size=200,
                chunk_overlap=20,
            )
            ingester.ingest(collection_name="test_collection")

            yield persist_dir, "test_collection"

    def test_basic_retrieval(self, indexed_kb):
        """Test basic document retrieval."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=3,
        )

        results = retriever.retrieve(
            "What is Python?",
            collection_name=collection_name,
        )

        assert len(results) > 0
        assert all("content" in doc for doc in results)
        assert all("score" in doc for doc in results)
        assert all("metadata" in doc for doc in results)

        # Scores should be in descending order
        scores = [doc["score"] for doc in results]
        assert scores == sorted(scores, reverse=True)

    def test_semantic_relevance(self, indexed_kb):
        """Test that retrieval returns semantically relevant documents."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=3,
            score_threshold=0.0,  # Don't filter by threshold for this test
        )

        # Query about RAG should return RAG-related content
        results = retriever.retrieve(
            "How does retrieval augmented generation work?",
            collection_name=collection_name,
        )

        assert len(results) > 0
        # Top result should contain RAG-related content
        top_content = results[0]["content"].lower()
        assert "rag" in top_content or "retrieval" in top_content

    def test_top_k_limit(self, indexed_kb):
        """Test that top_k limits the number of results."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=1,
            score_threshold=0.0,
        )

        results = retriever.retrieve(
            "programming",
            collection_name=collection_name,
        )

        assert len(results) <= 1

    def test_score_threshold_filtering(self, indexed_kb):
        """Test that score threshold filters low-quality results."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=10,
            score_threshold=0.5,  # High threshold
        )

        results = retriever.retrieve(
            "Python programming language",
            collection_name=collection_name,
        )

        # All results should meet threshold
        for doc in results:
            assert doc["score"] >= 0.5

    def test_metadata_preservation(self, indexed_kb):
        """Test that metadata is preserved in retrieved documents."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=3,
        )

        results = retriever.retrieve(
            "vector embeddings",
            collection_name=collection_name,
        )

        for doc in results:
            assert "source" in doc["metadata"]
            assert "chunk_index" in doc["metadata"]
            assert doc["metadata"]["source"].endswith(".md")

    def test_last_results_caching(self, indexed_kb):
        """Test that last results are cached."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=2,
        )

        results = retriever.retrieve(
            "Python",
            collection_name=collection_name,
        )

        cached = retriever.get_last_results()
        assert cached == results

    def test_format_sources(self, indexed_kb):
        """Test formatting of sources for display."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=2,
        )

        retriever.retrieve(
            "RAG systems",
            collection_name=collection_name,
        )

        formatted = retriever.format_sources()

        assert "Retrieved Sources:" in formatted
        assert "Score:" in formatted

    def test_nonexistent_collection(self, indexed_kb):
        """Test handling of nonexistent collection."""
        persist_dir, _ = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
        )

        results = retriever.retrieve(
            "test query",
            collection_name="nonexistent",
        )

        assert results == []

    def test_mmr_retrieval(self, indexed_kb):
        """Test MMR retrieval for diversity."""
        persist_dir, collection_name = indexed_kb

        retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=3,
            use_mmr=True,
            mmr_diversity=0.5,
            score_threshold=0.0,
        )

        results = retriever.retrieve(
            "programming concepts",
            collection_name=collection_name,
        )

        # Should return results (MMR shouldn't break retrieval)
        assert len(results) > 0

        # Results should have ranks assigned
        ranks = [doc["rank"] for doc in results]
        assert ranks == list(range(1, len(results) + 1))


class TestRetrieverEdgeCases:
    """Test edge cases for the retriever."""

    @pytest.fixture
    def empty_collection(self):
        """Create an empty ChromaDB collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "chroma_db"
            kb_path = Path(tmpdir) / "kb"
            kb_path.mkdir()

            # Create empty ingester to initialize empty collection
            ingester = DocumentIngester(
                kb_path=kb_path,
                persist_dir=persist_dir,
            )
            ingester.ingest(collection_name="empty_collection")

            yield persist_dir

    def test_empty_query(self, empty_collection):
        """Test handling of empty query string."""
        retriever = DocumentRetriever(
            persist_dir=empty_collection,
        )

        # Should not raise, but may return empty or poor results
        results = retriever.retrieve(
            "",
            collection_name="empty_collection",
        )

        assert isinstance(results, list)

    def test_very_long_query(self, empty_collection):
        """Test handling of very long query."""
        retriever = DocumentRetriever(
            persist_dir=empty_collection,
        )

        long_query = "test " * 1000
        results = retriever.retrieve(
            long_query,
            collection_name="empty_collection",
        )

        assert isinstance(results, list)
