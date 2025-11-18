# Usage Guide

This guide covers how to install, configure, and use the RAG Assistant.

## Installation

### Prerequisites

1. Python 3.10 or higher
2. Ollama installed and running

### Steps

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install package with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Copy environment configuration
cp .env.example .env
```

## Configuration

Edit `.env` to customize settings. Key parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model to use |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `RETRIEVER_TOP_K` | `4` | Number of chunks to retrieve |

## CLI Commands

### Ingest Knowledge Base

Process documents in `kb/` and create embeddings:

```bash
python -m rag.cli ingest
```

### Ask Questions

Query the knowledge base:

```bash
python -m rag.cli ask "What is RAG?"
```

### Show Sources

Display sources for the last query:

```bash
python -m rag.cli show-sources
```

## Web Interface

Start the web server:

```bash
uvicorn rag.web.app:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

## Testing

Run the test suite:

```bash
pytest -v
```

With coverage:

```bash
pytest --cov=rag --cov-report=html
```

## Evaluation

Run the evaluation harness:

```bash
python -m rag.eval_harness
```

Results are saved to `eval/report.md`.

## Quickstart Example Session

Here's a complete example session showing the RAG pipeline in action:

```bash
# Step 1: Ingest the knowledge base
$ python -m rag.cli ingest

2024-01-15 10:00:01 - rag.ingest - INFO - Loading embedding model: all-MiniLM-L6-v2
2024-01-15 10:00:03 - rag.ingest - INFO - Embedding model loaded in 2.15s
2024-01-15 10:00:03 - rag.ingest - INFO - Found 4 markdown files in kb
2024-01-15 10:00:03 - rag.ingest - INFO - Split 4 documents into 12 chunks (size=500, overlap=50)
2024-01-15 10:00:04 - rag.ingest - INFO - Generating embeddings for 12 chunks...
2024-01-15 10:00:04 - rag.ingest - INFO - Embeddings generated in 0.45s
2024-01-15 10:00:04 - rag.ingest - INFO - Stored 12 chunks in collection 'kb_docs'
2024-01-15 10:00:04 - rag.ingest - INFO - Ingestion complete: 12 chunks in 3.21s

Ingested 12 chunks from 4 documents.

# Step 2: Ask a question
$ python -m rag.cli ask "What is RAG and how does it work?"

2024-01-15 10:01:00 - rag.retriever - INFO - Retrieving top-4 documents...
2024-01-15 10:01:00 - rag.retriever - INFO - Retrieved 4 documents in 0.052s
2024-01-15 10:01:00 - rag.llm - INFO - Generating response: model=llama3.2:3b
2024-01-15 10:01:05 - rag.llm - INFO - Generated 256 chars in 4.89s (52.1 tokens/s)

RAG stands for Retrieval-Augmented Generation. According to glossary_ai_rag.md,
it's a technique that enhances LLM responses by retrieving relevant information
from a knowledge base before generation. The system first finds specific
documents related to your question, then includes them in the prompt to help
the language model generate more accurate and grounded responses with citations.

========================================
Sources:
  - glossary_ai_rag.md (chunk 1, score: 0.82)
  - project_faq.md (chunk 1, score: 0.71)
  - glossary_ai_rag.md (chunk 2, score: 0.65)
  - python_basics.md (chunk 3, score: 0.41)

# Step 3: Show sources with more detail
$ python -m rag.cli show-sources

Retrieved Sources:
========================================

[1] glossary_ai_rag.md (chunk 1/4)
    Score: 0.823
    Preview: RAG (Retrieval-Augmented Generation): A technique that enhances
    LLM responses by retrieving relevant information from a knowledge base...

[2] project_faq.md (chunk 1/6)
    Score: 0.714
    Preview: This is a RAG (Retrieval-Augmented Generation) Assistant that
    answers questions using a local knowledge base. It combines vector search...
```

## Observability

The pipeline logs timing information at each step:

- **Embedding model loading**: Time to load sentence-transformers model (~2s first time)
- **Chunk embedding**: Time to generate embeddings for all chunks
- **Retrieval**: Time to query ChromaDB and rank results (<100ms typically)
- **Generation**: Time for LLM to generate response (varies by model/hardware)
- **Tokens/second**: Generation speed metric

Enable debug logging for more detail:

```bash
LOG_LEVEL=DEBUG python -m rag.cli ask "your question"
```

---

*Last updated: Block 8 - Pipeline implementation*
