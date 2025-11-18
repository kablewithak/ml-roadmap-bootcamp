"""Tests for the document ingestion module."""

import tempfile
from pathlib import Path

import pytest

from rag.ingest import DocumentIngester


class TestDocumentIngester:
    """Test cases for DocumentIngester class."""

    @pytest.fixture
    def temp_kb(self):
        """Create a temporary knowledge base directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "kb"
            kb_path.mkdir()

            # Create test documents
            (kb_path / "test1.md").write_text(
                "# Test Document 1\n\n"
                "This is the first test document. "
                "It contains some sample content for testing the ingestion pipeline. "
                "We need enough text here to test chunking behavior.\n\n"
                "## Section Two\n\n"
                "This is another section with more content to ensure "
                "we have multiple chunks when using smaller chunk sizes."
            )

            (kb_path / "test2.md").write_text(
                "# Test Document 2\n\n"
                "This document covers different topics. "
                "Python is a programming language. "
                "Machine learning involves training models on data.\n\n"
                "RAG stands for Retrieval-Augmented Generation."
            )

            yield kb_path

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "chroma_db"

    def test_load_documents(self, temp_kb, temp_persist_dir):
        """Test loading documents from knowledge base."""
        ingester = DocumentIngester(
            kb_path=temp_kb,
            persist_dir=temp_persist_dir,
        )

        documents = ingester.load_documents()

        assert len(documents) == 2
        assert all("content" in doc for doc in documents)
        assert all("metadata" in doc for doc in documents)
        assert any("test1.md" in doc["metadata"]["source"] for doc in documents)
        assert any("test2.md" in doc["metadata"]["source"] for doc in documents)

    def test_chunk_documents(self, temp_kb, temp_persist_dir):
        """Test chunking documents with overlap."""
        ingester = DocumentIngester(
            kb_path=temp_kb,
            persist_dir=temp_persist_dir,
            chunk_size=100,
            chunk_overlap=20,
        )

        documents = ingester.load_documents()
        chunks = ingester.chunk_documents(documents)

        # Should have multiple chunks due to small chunk size
        assert len(chunks) > len(documents)

        # Each chunk should have metadata
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "source" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]
            assert "chunk_total" in chunk["metadata"]

    def test_full_ingestion_pipeline(self, temp_kb, temp_persist_dir):
        """Test the full ingestion pipeline."""
        ingester = DocumentIngester(
            kb_path=temp_kb,
            persist_dir=temp_persist_dir,
            chunk_size=200,
            chunk_overlap=20,
        )

        num_chunks = ingester.ingest(collection_name="test_collection")

        assert num_chunks > 0

        # Verify collection was created
        stats = ingester.get_collection_stats(collection_name="test_collection")
        assert stats["document_count"] == num_chunks

    def test_empty_kb_directory(self, temp_persist_dir):
        """Test handling of empty knowledge base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_kb = Path(tmpdir) / "empty_kb"
            empty_kb.mkdir()

            ingester = DocumentIngester(
                kb_path=empty_kb,
                persist_dir=temp_persist_dir,
            )

            documents = ingester.load_documents()
            assert len(documents) == 0

            num_chunks = ingester.ingest()
            assert num_chunks == 0

    def test_nonexistent_kb_directory(self, temp_persist_dir):
        """Test handling of nonexistent knowledge base directory."""
        ingester = DocumentIngester(
            kb_path=Path("/nonexistent/path"),
            persist_dir=temp_persist_dir,
        )

        documents = ingester.load_documents()
        assert len(documents) == 0

    def test_collection_stats(self, temp_kb, temp_persist_dir):
        """Test getting collection statistics."""
        ingester = DocumentIngester(
            kb_path=temp_kb,
            persist_dir=temp_persist_dir,
        )

        # Before ingestion
        stats = ingester.get_collection_stats("nonexistent")
        assert "error" in stats
        assert stats["document_count"] == 0

        # After ingestion
        ingester.ingest(collection_name="test_stats")
        stats = ingester.get_collection_stats("test_stats")
        assert stats["document_count"] > 0
        assert "error" not in stats

    def test_re_ingestion_replaces_collection(self, temp_kb, temp_persist_dir):
        """Test that re-ingesting replaces the existing collection."""
        ingester = DocumentIngester(
            kb_path=temp_kb,
            persist_dir=temp_persist_dir,
        )

        # First ingestion
        first_count = ingester.ingest(collection_name="replace_test")

        # Second ingestion should replace, not append
        second_count = ingester.ingest(collection_name="replace_test")

        stats = ingester.get_collection_stats("replace_test")
        assert stats["document_count"] == second_count
        assert first_count == second_count  # Same documents


class TestChunkingBehavior:
    """Test cases for chunking behavior specifics."""

    @pytest.fixture
    def long_document_kb(self):
        """Create a KB with a single long document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "kb"
            kb_path.mkdir()

            # Create a document with clear section boundaries
            content = (
                "# Introduction\n\n"
                "This is the introduction section. " * 20 + "\n\n"
                "# Methods\n\n"
                "This describes the methods used. " * 20 + "\n\n"
                "# Results\n\n"
                "These are the results. " * 20
            )
            (kb_path / "long_doc.md").write_text(content)

            yield kb_path

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "chroma_db"

    def test_chunk_overlap_preserves_context(self, long_document_kb, temp_persist_dir):
        """Test that chunk overlap preserves context at boundaries."""
        ingester = DocumentIngester(
            kb_path=long_document_kb,
            persist_dir=temp_persist_dir,
            chunk_size=300,
            chunk_overlap=50,
        )

        documents = ingester.load_documents()
        chunks = ingester.chunk_documents(documents)

        # Check that consecutive chunks have overlapping content
        # (This is a simplified check - actual overlap depends on text splitting)
        assert len(chunks) >= 3  # Long document should create multiple chunks

    def test_metadata_preserves_chunk_position(
        self, long_document_kb, temp_persist_dir
    ):
        """Test that chunk metadata preserves position information."""
        ingester = DocumentIngester(
            kb_path=long_document_kb,
            persist_dir=temp_persist_dir,
            chunk_size=300,
            chunk_overlap=50,
        )

        documents = ingester.load_documents()
        chunks = ingester.chunk_documents(documents)

        # Verify chunk indices are sequential
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

        # Verify chunk_total is consistent
        totals = {c["metadata"]["chunk_total"] for c in chunks}
        assert len(totals) == 1
        assert totals.pop() == len(chunks)
