"""Configuration management for RAG Assistant.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration for RAG Assistant."""

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    KB_PATH: Path = BASE_DIR / os.getenv("KB_PATH", "kb")
    CHROMA_PERSIST_DIR: Path = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

    # LLM Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Retrieval Configuration
    RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "4"))
    RETRIEVER_SCORE_THRESHOLD: float = float(
        os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.3")
    )
    USE_MMR: bool = os.getenv("USE_MMR", "false").lower() == "true"
    MMR_DIVERSITY: float = float(os.getenv("MMR_DIVERSITY", "0.5"))

    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Web UI
    WEB_HOST: str = os.getenv("WEB_HOST", "127.0.0.1")
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8000"))

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.KB_PATH.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_summary(cls) -> dict:
        """Return a summary of current configuration for logging."""
        return {
            "ollama_model": cls.OLLAMA_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "top_k": cls.RETRIEVER_TOP_K,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "use_mmr": cls.USE_MMR,
        }


# Singleton instance
config = Config()


def get_config() -> Config:
    """Get the configuration instance."""
    return config
