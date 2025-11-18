# RAG Assistant

A question-answering assistant over a local knowledge base using Retrieval-Augmented Generation (RAG).

## Features

- **Local-first**: All processing happens on your machine using Ollama
- **Open-source stack**: sentence-transformers, ChromaDB, LangChain
- **CLI & Web UI**: Interact via command line or browser
- **Evaluation harness**: Built-in testing with precision metrics

## Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | CPU-friendly, widely used |
| Vector Store | ChromaDB | Local, zero-cost |
| LLM | Ollama + Llama 3.2 3B | Local serving, easy model swapping |
| Orchestration | LangChain | Industry mindshare |
| Interface | Typer CLI + FastAPI/HTMX | Progressive enhancement |

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running
   ```bash
   ollama pull llama3.2:3b
   ```

### Installation

```bash
# Clone and enter repo
git clone <repo-url>
cd rag-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env
```

### Usage

```bash
# Ingest knowledge base
python -m rag.cli ingest

# Ask a question
python -m rag.cli ask "What is RAG?"

# Show sources for last query
python -m rag.cli show-sources
```

## Project Structure

```
rag-assistant/
  rag/
    __init__.py
    config.py          # Configuration management
    ingest.py          # Document ingestion & embedding
    retriever.py       # Vector search & retrieval
    llm.py             # LLM adapter (Ollama)
    pipeline.py        # End-to-end RAG pipeline
    eval_harness.py    # Evaluation metrics
    cli.py             # Command-line interface
    web/               # Web interface
      app.py
      templates/
  kb/                  # Knowledge base documents
  tests/               # Test suite
  docs/                # Documentation
  eval/                # Evaluation results
```

## Testing

```bash
pytest -v
```

## Evaluation

```bash
python -m rag.eval_harness
```

Results are saved to `eval/report.md`.

## Configuration

See `.env.example` for available configuration options.

## Documentation

- [Design Decisions](docs/decisions.md)
- [Usage Guide](docs/usage.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT
