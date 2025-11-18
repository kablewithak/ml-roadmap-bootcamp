# RAG Assistant

A question-answering assistant over a local knowledge base using Retrieval-Augmented Generation (RAG).

## Features

- **Local-first**: All processing happens on your machine using Ollama
- **Open-source stack**: sentence-transformers, ChromaDB, LangChain
- **CLI & Web UI**: Interact via command line or browser
- **Evaluation harness**: Built-in testing with precision metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐ │
│  │   KB    │───▶│  Ingest  │───▶│ ChromaDB│    │         │ │
│  │  (md)   │    │  + Embed │    │ (store) │    │         │ │
│  └─────────┘    └──────────┘    └────┬────┘    │         │ │
│                                      │         │  Ollama │ │
│  ┌─────────┐    ┌──────────┐    ┌────▼────┐    │   LLM   │ │
│  │  Query  │───▶│ Retrieve │───▶│ Context │───▶│         │ │
│  └─────────┘    │  (top-k) │    │ + Query │    │         │ │
│                 └──────────┘    └────┬────┘    └────┬────┘ │
│                                      │              │      │
│                                      └──────────────┘      │
│                                           Answer           │
└─────────────────────────────────────────────────────────────┘
```

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

## Web Interface

Start the web server:

```bash
uvicorn rag.web.app:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## CLI Commands

```bash
# Show all commands
python -m rag.cli --help

# Ingest documents
python -m rag.cli ingest

# Ask a question
python -m rag.cli ask "What is RAG?"

# Show detailed sources
python -m rag.cli show-sources

# Check system health
python -m rag.cli health

# Show configuration
python -m rag.cli config
```

## Documentation

- [Design Decisions](docs/decisions.md) - Stack choices and trade-offs
- [Usage Guide](docs/usage.md) - Installation and examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues
- [Checklist](docs/checklist.md) - Quality audit
- [Portfolio](docs/portfolio.md) - Project summary

## License

MIT
