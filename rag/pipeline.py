"""RAG pipeline module for RAG Assistant.

Orchestrates retrieval and generation for question answering.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from rag.config import Config
from rag.llm import LLMError, OllamaLLM
from rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

# Default system prompt for RAG QA
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question using ONLY the information in the context below
- If the context doesn't contain enough information, say so honestly
- Be concise but thorough
- Reference specific sources when possible (e.g., "According to python_basics.md...")
- Do not make up information that isn't in the context"""


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "kb_docs",
        top_k: Optional[int] = None,
        use_mmr: Optional[bool] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            persist_dir: Path to ChromaDB persistence directory.
            collection_name: Name of the ChromaDB collection.
            top_k: Number of documents to retrieve.
            use_mmr: Whether to use MMR for diversity.
            model: Ollama model name.
            temperature: LLM sampling temperature.
            system_prompt: Custom system prompt.
        """
        self.collection_name = collection_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Initialize retriever
        self.retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=top_k,
            use_mmr=use_mmr,
        )

        # Initialize LLM
        self.llm = OllamaLLM(
            model=model,
            temperature=temperature,
        )

        # Store last query results
        self._last_query: Optional[str] = None
        self._last_sources: list[dict] = []
        self._last_answer: Optional[str] = None

        logger.info(
            f"Initialized RAG pipeline: "
            f"top_k={self.retriever.top_k}, "
            f"model={self.llm.model}"
        )

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Answer a question using RAG.

        Args:
            question: The user's question.
            top_k: Override default top_k for this query.
            temperature: Override default temperature for this query.

        Returns:
            The generated answer.

        Raises:
            PipelineError: If retrieval or generation fails.
        """
        logger.info(f"Processing question: '{question[:100]}...'")
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        try:
            sources = self.retriever.retrieve(
                query=question,
                collection_name=self.collection_name,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise PipelineError(f"Failed to retrieve documents: {e}") from e

        if not sources:
            logger.warning("No relevant documents found")
            self._last_query = question
            self._last_sources = []
            self._last_answer = (
                "I couldn't find any relevant information in the knowledge base "
                "to answer this question. Please try rephrasing or check if the "
                "topic is covered in the KB."
            )
            return self._last_answer

        # Step 2: Build prompt with context
        prompt = self._build_prompt(question, sources)

        # Step 3: Generate answer
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
            )
        except LLMError as e:
            logger.error(f"Generation failed: {e}")
            raise PipelineError(f"Failed to generate answer: {e}") from e

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f}s")

        # Store results
        self._last_query = question
        self._last_sources = sources
        self._last_answer = answer

        return answer

    def _build_prompt(self, question: str, sources: list[dict]) -> str:
        """Build the prompt with context from retrieved sources.

        Args:
            question: The user's question.
            sources: Retrieved documents with content and metadata.

        Returns:
            Formatted prompt with context.
        """
        # Format context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            source_name = source["metadata"].get("source", "Unknown")
            content = source["content"]
            context_parts.append(f"[Source {i}: {source_name}]\n{content}")

        context = "\n\n".join(context_parts)

        # Build final prompt
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        logger.debug(f"Built prompt with {len(sources)} sources ({len(prompt)} chars)")

        return prompt

    def get_last_results(self) -> dict:
        """Get results from the last query.

        Returns:
            Dictionary with query, sources, and answer.
        """
        return {
            "query": self._last_query,
            "sources": self._last_sources,
            "answer": self._last_answer,
        }

    def get_sources(self) -> list[dict]:
        """Get sources from the last query.

        Returns:
            List of source documents.
        """
        return self._last_sources

    def format_response(self, include_sources: bool = True) -> str:
        """Format the last response for display.

        Args:
            include_sources: Whether to include source information.

        Returns:
            Formatted response string.
        """
        if not self._last_answer:
            return "No answer available. Run a query first."

        parts = [self._last_answer]

        if include_sources and self._last_sources:
            parts.append("\n\n" + "=" * 40)
            parts.append("Sources:")
            for source in self._last_sources:
                source_name = source["metadata"].get("source", "Unknown")
                chunk_idx = source["metadata"].get("chunk_index", "?")
                score = source["score"]
                parts.append(f"  - {source_name} (chunk {chunk_idx + 1}, score: {score:.2f})")

        return "\n".join(parts)

    def check_health(self) -> dict:
        """Check health of pipeline components.

        Returns:
            Health status dictionary.
        """
        health = {
            "healthy": False,
            "retriever": False,
            "llm": False,
        }

        # Check retriever (ChromaDB collection exists)
        try:
            stats = self.retriever.chroma_client.get_collection(self.collection_name)
            health["retriever"] = True
            health["collection_count"] = stats.count()
        except ValueError:
            health["retriever_error"] = f"Collection '{self.collection_name}' not found"

        # Check LLM
        llm_health = self.llm.check_health()
        health["llm"] = llm_health["healthy"]
        if not llm_health["healthy"]:
            health["llm_error"] = llm_health.get("error", "LLM not available")

        health["healthy"] = health["retriever"] and health["llm"]

        return health


class PipelineError(Exception):
    """Custom exception for pipeline errors."""

    pass


def create_pipeline(
    collection_name: str = "kb_docs",
    top_k: Optional[int] = None,
    model: Optional[str] = None,
) -> RAGPipeline:
    """Convenience function to create a pipeline.

    Args:
        collection_name: ChromaDB collection name.
        top_k: Number of documents to retrieve.
        model: Ollama model name.

    Returns:
        Configured RAGPipeline instance.
    """
    return RAGPipeline(
        collection_name=collection_name,
        top_k=top_k,
        model=model,
    )


def ask_question(
    question: str,
    collection_name: str = "kb_docs",
    top_k: Optional[int] = None,
) -> str:
    """Convenience function to ask a question.

    Args:
        question: The user's question.
        collection_name: ChromaDB collection name.
        top_k: Number of documents to retrieve.

    Returns:
        Generated answer.
    """
    pipeline = create_pipeline(collection_name=collection_name, top_k=top_k)
    return pipeline.ask(question)


if __name__ == "__main__":
    # Configure logging for standalone run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create pipeline
    pipeline = create_pipeline()

    # Check health
    print("Checking pipeline health...")
    health = pipeline.check_health()

    if not health["healthy"]:
        print("Pipeline health check failed:")
        if not health["retriever"]:
            print(f"  - Retriever: {health.get('retriever_error', 'Not ready')}")
            print("  - Run: python -m rag.cli ingest")
        if not health["llm"]:
            print(f"  - LLM: {health.get('llm_error', 'Not ready')}")
        exit(1)

    print(f"Pipeline healthy. Collection has {health['collection_count']} documents.\n")

    # Test query
    question = "What is RAG?"
    print(f"Question: {question}\n")

    answer = pipeline.ask(question)
    print(f"Answer:\n{pipeline.format_response()}")
