"""Document ingestion module for RAG Assistant.

Handles loading, chunking, embedding, and storing documents in ChromaDB.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.config import Config

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Ingests documents from knowledge base into vector store."""

    def __init__(
        self,
        kb_path: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the ingester with configuration.

        Args:
            kb_path: Path to knowledge base directory.
            persist_dir: Path for ChromaDB persistence.
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between chunks in characters.
            embedding_model: Name of sentence-transformers model.
        """
        self.kb_path = kb_path or Config.KB_PATH
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        elapsed = time.time() - start_time
        logger.info(f"Embedding model loaded in {elapsed:.2f}s")

        # Initialize ChromaDB client
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def load_documents(self) -> list[dict]:
        """Load all markdown documents from the knowledge base.

        Returns:
            List of documents with content and metadata.
        """
        documents = []

        if not self.kb_path.exists():
            logger.warning(f"Knowledge base path does not exist: {self.kb_path}")
            return documents

        md_files = list(self.kb_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files in {self.kb_path}")

        for file_path in md_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": file_path.name,
                            "path": str(file_path),
                        },
                    }
                )
                logger.debug(f"Loaded: {file_path.name} ({len(content)} chars)")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        return documents

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Split documents into chunks with overlap.

        Args:
            documents: List of documents with content and metadata.

        Returns:
            List of chunks with content and metadata.
        """
        chunks = []

        for doc in documents:
            text_chunks = self.text_splitter.split_text(doc["content"])

            for i, chunk_text in enumerate(text_chunks):
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": i,
                            "chunk_total": len(text_chunks),
                        },
                    }
                )

        logger.info(
            f"Split {len(documents)} documents into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return chunks

    def embed_and_store(
        self, chunks: list[dict], collection_name: str = "kb_docs"
    ) -> int:
        """Embed chunks and store in ChromaDB.

        Args:
            chunks: List of chunks with content and metadata.
            collection_name: Name of the ChromaDB collection.

        Returns:
            Number of chunks stored.
        """
        if not chunks:
            logger.warning("No chunks to embed and store")
            return 0

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except ValueError:
            pass  # Collection doesn't exist

        # Create new collection
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Prepare data for ChromaDB
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        start_time = time.time()
        embeddings = self.embeddings.embed_documents(documents)
        elapsed = time.time() - start_time
        logger.info(f"Embeddings generated in {elapsed:.2f}s")

        # Store in ChromaDB
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Stored {len(chunks)} chunks in collection '{collection_name}'")

        return len(chunks)

    def ingest(self, collection_name: str = "kb_docs") -> int:
        """Run the full ingestion pipeline.

        Args:
            collection_name: Name of the ChromaDB collection.

        Returns:
            Number of chunks ingested.
        """
        logger.info("Starting ingestion pipeline...")
        start_time = time.time()

        # Load documents
        documents = self.load_documents()
        if not documents:
            logger.error("No documents found to ingest")
            return 0

        # Chunk documents
        chunks = self.chunk_documents(documents)

        # Embed and store
        num_chunks = self.embed_and_store(chunks, collection_name)

        elapsed = time.time() - start_time
        logger.info(f"Ingestion complete: {num_chunks} chunks in {elapsed:.2f}s")

        return num_chunks

    def get_collection_stats(self, collection_name: str = "kb_docs") -> dict:
        """Get statistics about a collection.

        Args:
            collection_name: Name of the ChromaDB collection.

        Returns:
            Dictionary with collection statistics.
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()
            return {
                "collection_name": collection_name,
                "document_count": count,
                "persist_dir": str(self.persist_dir),
            }
        except ValueError:
            return {
                "collection_name": collection_name,
                "document_count": 0,
                "persist_dir": str(self.persist_dir),
                "error": "Collection not found",
            }


def ingest_knowledge_base(
    kb_path: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    collection_name: str = "kb_docs",
) -> int:
    """Convenience function to ingest knowledge base.

    Args:
        kb_path: Path to knowledge base directory.
        persist_dir: Path for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Number of chunks ingested.
    """
    ingester = DocumentIngester(kb_path=kb_path, persist_dir=persist_dir)
    return ingester.ingest(collection_name)


if __name__ == "__main__":
    # Configure logging for standalone run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run ingestion
    num_chunks = ingest_knowledge_base()
    print(f"\nIngested {num_chunks} chunks")
