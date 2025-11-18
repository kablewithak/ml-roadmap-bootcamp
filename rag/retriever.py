"""Retriever module for RAG Assistant.

Handles semantic search and document retrieval from ChromaDB.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.config import Config

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieves relevant documents from vector store."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_mmr: Optional[bool] = None,
        mmr_diversity: Optional[float] = None,
    ):
        """Initialize the retriever with configuration.

        Args:
            persist_dir: Path to ChromaDB persistence directory.
            embedding_model: Name of sentence-transformers model.
            top_k: Number of documents to retrieve.
            score_threshold: Minimum similarity score threshold.
            use_mmr: Whether to use Maximal Marginal Relevance.
            mmr_diversity: MMR diversity factor (0=max relevance, 1=max diversity).
        """
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.top_k = top_k or Config.RETRIEVER_TOP_K
        self.score_threshold = score_threshold or Config.RETRIEVER_SCORE_THRESHOLD
        self.use_mmr = use_mmr if use_mmr is not None else Config.USE_MMR
        self.mmr_diversity = mmr_diversity or Config.MMR_DIVERSITY

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Cache for last retrieval results
        self._last_results: list[dict] = []

    def retrieve(
        self,
        query: str,
        collection_name: str = "kb_docs",
        top_k: Optional[int] = None,
        use_mmr: Optional[bool] = None,
    ) -> list[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            collection_name: Name of the ChromaDB collection.
            top_k: Override default top_k.
            use_mmr: Override default MMR setting.

        Returns:
            List of retrieved documents with content, metadata, and scores.
        """
        k = top_k or self.top_k
        mmr = use_mmr if use_mmr is not None else self.use_mmr

        logger.info(f"Retrieving top-{k} documents for query: '{query[:50]}...'")
        start_time = time.time()

        try:
            collection = self.chroma_client.get_collection(collection_name)
        except ValueError as e:
            logger.error(f"Collection '{collection_name}' not found: {e}")
            return []

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Query ChromaDB
        # Retrieve more candidates for MMR filtering
        n_candidates = k * 3 if mmr else k

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_candidates, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Process results
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity score (ChromaDB uses L2 by default)
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i]
                score = 1 - distance  # Convert to similarity

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                documents.append(
                    {
                        "content": doc,
                        "metadata": metadata,
                        "score": score,
                        "rank": i + 1,
                    }
                )

        # Apply MMR if requested
        if mmr and len(documents) > k:
            documents = self._apply_mmr(query_embedding, documents, k)

        # Apply score threshold
        documents = [
            doc for doc in documents if doc["score"] >= self.score_threshold
        ]

        # Limit to top_k
        documents = documents[:k]

        # Update ranks after filtering
        for i, doc in enumerate(documents):
            doc["rank"] = i + 1

        elapsed = time.time() - start_time
        logger.info(
            f"Retrieved {len(documents)} documents in {elapsed:.3f}s "
            f"(threshold={self.score_threshold})"
        )

        # Cache results
        self._last_results = documents

        return documents

    def _apply_mmr(
        self,
        query_embedding: list[float],
        documents: list[dict],
        k: int,
    ) -> list[dict]:
        """Apply Maximal Marginal Relevance for diversity.

        Args:
            query_embedding: The query embedding vector.
            documents: Candidate documents with scores.
            k: Number of documents to select.

        Returns:
            Reranked documents balancing relevance and diversity.
        """
        if len(documents) <= k:
            return documents

        lambda_param = 1 - self.mmr_diversity  # Convert diversity to lambda

        # Get embeddings for all documents
        doc_texts = [doc["content"] for doc in documents]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)

        selected_indices = []
        remaining_indices = list(range(len(documents)))

        for _ in range(k):
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance to query (cosine similarity)
                relevance = np.dot(query_vec, doc_vecs[idx])

                # Maximum similarity to already selected documents
                if selected_indices:
                    selected_vecs = doc_vecs[selected_indices]
                    similarities = np.dot(selected_vecs, doc_vecs[idx])
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return selected documents in order
        return [documents[i] for i in selected_indices]

    def get_last_results(self) -> list[dict]:
        """Get the results from the last retrieval.

        Returns:
            List of documents from the last query.
        """
        return self._last_results

    def format_sources(self, documents: Optional[list[dict]] = None) -> str:
        """Format retrieved documents as a sources string.

        Args:
            documents: Documents to format, or use last results.

        Returns:
            Formatted string with sources.
        """
        docs = documents or self._last_results

        if not docs:
            return "No sources retrieved."

        lines = ["Retrieved Sources:", "=" * 40]

        for doc in docs:
            source = doc["metadata"].get("source", "Unknown")
            chunk_idx = doc["metadata"].get("chunk_index", "?")
            chunk_total = doc["metadata"].get("chunk_total", "?")
            score = doc["score"]

            lines.append(f"\n[{doc['rank']}] {source} (chunk {chunk_idx + 1}/{chunk_total})")
            lines.append(f"    Score: {score:.3f}")
            lines.append(f"    Preview: {doc['content'][:150]}...")

        return "\n".join(lines)


def retrieve_documents(
    query: str,
    persist_dir: Optional[Path] = None,
    collection_name: str = "kb_docs",
    top_k: Optional[int] = None,
) -> list[dict]:
    """Convenience function to retrieve documents.

    Args:
        query: The search query.
        persist_dir: Path to ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
        top_k: Number of documents to retrieve.

    Returns:
        List of retrieved documents.
    """
    retriever = DocumentRetriever(persist_dir=persist_dir, top_k=top_k)
    return retriever.retrieve(query, collection_name)


if __name__ == "__main__":
    # Configure logging for standalone run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    query = "What is RAG?"
    results = retrieve_documents(query)

    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")

    for doc in results:
        print(f"[{doc['rank']}] Score: {doc['score']:.3f}")
        print(f"    Source: {doc['metadata'].get('source', 'Unknown')}")
        print(f"    Content: {doc['content'][:200]}...")
        print()
