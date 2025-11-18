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

## Observability

*Logging and timing information will be added here.*

---

*Last updated: Initial project setup*
