"""Tests for the RAG pipeline module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.ingest import DocumentIngester
from rag.pipeline import RAGPipeline, PipelineError


class TestRAGPipeline:
    """Test cases for RAGPipeline class."""

    @pytest.fixture
    def indexed_kb(self):
        """Create and index a temporary knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "kb"
            persist_dir = Path(tmpdir) / "chroma_db"
            kb_path.mkdir()

            # Create test documents
            (kb_path / "python.md").write_text(
                "# Python Programming\n\n"
                "Python is a high-level programming language. "
                "It emphasizes code readability with significant indentation. "
                "Python supports multiple paradigms including procedural, "
                "object-oriented, and functional programming."
            )

            (kb_path / "rag.md").write_text(
                "# RAG Systems\n\n"
                "RAG stands for Retrieval-Augmented Generation. "
                "It enhances LLM responses by first retrieving relevant "
                "documents from a knowledge base. The retrieved context "
                "is then used to generate more accurate answers."
            )

            # Ingest documents
            ingester = DocumentIngester(
                kb_path=kb_path,
                persist_dir=persist_dir,
                chunk_size=300,
                chunk_overlap=30,
            )
            ingester.ingest(collection_name="test_collection")

            yield persist_dir, "test_collection"

    def test_pipeline_initialization(self, indexed_kb):
        """Test pipeline initializes correctly."""
        persist_dir, collection_name = indexed_kb

        pipeline = RAGPipeline(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        assert pipeline.retriever is not None
        assert pipeline.llm is not None
        assert pipeline.collection_name == collection_name

    @patch("rag.pipeline.OllamaLLM")
    def test_ask_retrieves_and_generates(self, mock_llm_class, indexed_kb):
        """Test that ask() retrieves documents and generates answer."""
        persist_dir, collection_name = indexed_kb

        # Mock LLM to avoid actual Ollama call
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Python is a programming language."
        mock_llm_class.return_value = mock_llm

        pipeline = RAGPipeline(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        answer = pipeline.ask("What is Python?")

        # Verify retrieval happened
        assert len(pipeline._last_sources) > 0

        # Verify LLM was called
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args

        # Check prompt contains context
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "Context:" in prompt
        assert "Question:" in prompt

        # Verify answer is returned
        assert answer == "Python is a programming language."

    @patch("rag.pipeline.OllamaLLM")
    def test_prompt_includes_sources(self, mock_llm_class, indexed_kb):
        """Test that generated prompt includes source information."""
        persist_dir, collection_name = indexed_kb

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Test answer"
        mock_llm_class.return_value = mock_llm

        pipeline = RAGPipeline(
            persist_dir=persist_dir,
            collection_name=collection_name,
            top_k=2,
        )

        pipeline.ask("What is RAG?")

        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]

        # Should have source markers
        assert "[Source" in prompt

    @patch("rag.pipeline.OllamaLLM")
    def test_get_last_results(self, mock_llm_class, indexed_kb):
        """Test getting results from last query."""
        persist_dir, collection_name = indexed_kb

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Test answer"
        mock_llm_class.return_value = mock_llm

        pipeline = RAGPipeline(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        pipeline.ask("What is Python?")

        results = pipeline.get_last_results()

        assert results["query"] == "What is Python?"
        assert results["answer"] == "Test answer"
        assert len(results["sources"]) > 0

    @patch("rag.pipeline.OllamaLLM")
    def test_format_response(self, mock_llm_class, indexed_kb):
        """Test formatting response with sources."""
        persist_dir, collection_name = indexed_kb

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Python is great."
        mock_llm_class.return_value = mock_llm

        pipeline = RAGPipeline(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        pipeline.ask("What is Python?")
        formatted = pipeline.format_response(include_sources=True)

        assert "Python is great." in formatted
        assert "Sources:" in formatted

    @patch("rag.pipeline.OllamaLLM")
    def test_no_sources_returns_helpful_message(self, mock_llm_class):
        """Test that missing sources returns a helpful message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "chroma_db"
            kb_path = Path(tmpdir) / "kb"
            kb_path.mkdir()

            # Create empty collection
            ingester = DocumentIngester(
                kb_path=kb_path,
                persist_dir=persist_dir,
            )
            ingester.ingest(collection_name="empty_collection")

            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            pipeline = RAGPipeline(
                persist_dir=persist_dir,
                collection_name="empty_collection",
            )

            answer = pipeline.ask("What is something?")

            # Should return helpful message without calling LLM
            assert "couldn't find" in answer.lower()
            mock_llm.generate.assert_not_called()


class TestPipelineHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def indexed_kb(self):
        """Create and index a temporary knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "kb"
            persist_dir = Path(tmpdir) / "chroma_db"
            kb_path.mkdir()

            (kb_path / "test.md").write_text("# Test\n\nTest content.")

            ingester = DocumentIngester(
                kb_path=kb_path,
                persist_dir=persist_dir,
            )
            ingester.ingest(collection_name="test_collection")

            yield persist_dir

    @patch("rag.pipeline.OllamaLLM")
    def test_health_check_with_valid_collection(self, mock_llm_class, indexed_kb):
        """Test health check with valid collection."""
        mock_llm = MagicMock()
        mock_llm.check_health.return_value = {"healthy": True}
        mock_llm_class.return_value = mock_llm

        pipeline = RAGPipeline(
            persist_dir=indexed_kb,
            collection_name="test_collection",
        )

        health = pipeline.check_health()

        assert health["retriever"] is True
        assert health["collection_count"] > 0

    @patch("rag.pipeline.OllamaLLM")
    def test_health_check_with_missing_collection(self, mock_llm_class, indexed_kb):
        """Test health check with missing collection."""
        mock_llm = MagicMock()
        mock_llm.check_health.return_value = {"healthy": True}
        mock_llm_class.return_value = mock_llm

        pipeline = RAGPipeline(
            persist_dir=indexed_kb,
            collection_name="nonexistent",
        )

        health = pipeline.check_health()

        assert health["retriever"] is False
        assert "retriever_error" in health


class TestPipelineErrorHandling:
    """Test error handling in pipeline."""

    @pytest.fixture
    def indexed_kb(self):
        """Create and index a temporary knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "kb"
            persist_dir = Path(tmpdir) / "chroma_db"
            kb_path.mkdir()

            (kb_path / "test.md").write_text("# Test\n\nTest content here.")

            ingester = DocumentIngester(
                kb_path=kb_path,
                persist_dir=persist_dir,
            )
            ingester.ingest(collection_name="test_collection")

            yield persist_dir

    @patch("rag.pipeline.OllamaLLM")
    def test_llm_error_raises_pipeline_error(self, mock_llm_class, indexed_kb):
        """Test that LLM errors are wrapped in PipelineError."""
        from rag.llm import LLMError

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = LLMError("Connection failed")
        mock_llm_class.return_value = mock_llm

        pipeline = RAGPipeline(
            persist_dir=indexed_kb,
            collection_name="test_collection",
        )

        with pytest.raises(PipelineError) as exc_info:
            pipeline.ask("What is test?")

        assert "Connection failed" in str(exc_info.value)
