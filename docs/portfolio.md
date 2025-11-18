# Portfolio Documentation

## Project Summary

**RAG Assistant** is a local-first question-answering system that demonstrates end-to-end ML engineering skills: embeddings, vector search, LLM orchestration, and production-quality code.

## LinkedIn Post Draft

---

**Just shipped my first RAG system from scratch!**

Built a complete Retrieval-Augmented Generation assistant using an open-source stack:

- sentence-transformers for embeddings
- ChromaDB for vector storage
- Ollama + Llama 3.2 for local LLM
- LangChain for orchestration
- FastAPI + HTMX for the web interface

Key learnings:

**1. Chunking matters more than you think**
Finding the right chunk size (500 chars) and overlap (50) was critical. Too small = lost context, too big = noisy retrieval.

**2. Local LLMs are surprisingly capable**
Llama 3.2 3B handles RAG QA well when properly prompted. Temperature 0.7 balances creativity with grounding.

**3. Evaluation needs to be built-in**
Created a harness with 8 test queries measuring hit rate and nugget recall. Without this, "it seems to work" is dangerously vague.

**4. Observability is non-negotiable**
Added timing logs for embedding, retrieval, and generation. This made parameter tuning data-driven instead of guesswork.

The system answers questions over a local knowledge base with citations, runs entirely on my machine (privacy!), and includes CLI + web interfaces.

Code: [GitHub Link]

Next up: PDF/HTML loaders, better evaluation metrics, and Docker packaging.

#MachineLearning #RAG #Python #LLM #OpenSource

---

## Key Accomplishments

### Technical Skills Demonstrated

1. **Vector Databases**: Implemented ChromaDB integration with proper persistence
2. **Embeddings**: Used sentence-transformers with normalization
3. **LLM Integration**: Built Ollama adapter with error handling and health checks
4. **Prompt Engineering**: Designed system prompts for grounded answers with citations
5. **RAG Pipeline**: End-to-end retrieve-then-generate architecture
6. **Evaluation**: Metrics-based harness with precision@k style measurements
7. **CLI Development**: Rich Typer interface with multiple commands
8. **Web Development**: FastAPI + HTMX for dynamic UI without heavy JS

### Software Engineering Practices

1. **Configuration Management**: Environment-based config with sensible defaults
2. **Error Handling**: Custom exceptions with helpful messages
3. **Logging & Observability**: Timing metrics throughout pipeline
4. **Testing**: Unit tests for all core modules
5. **Documentation**: Design decisions, usage guides, troubleshooting
6. **Code Quality**: Pre-commit hooks with black, ruff, mypy

### Project Statistics

- **Lines of Code**: ~2,500
- **Test Coverage**: Key paths covered
- **Documentation**: 5 markdown files
- **Commits**: Atomic, descriptive messages
- **Time to Complete**: 3 days (22 blocks × 45 min)

## What I Learned

1. **Abstraction is key**: Separating ingest/retrieve/llm/pipeline made testing and iteration much easier.

2. **Defaults matter**: Spent time tuning chunk_size=500, overlap=50, top_k=4, temperature=0.7 - these defaults define the user experience.

3. **Evaluation prevents regression**: The 8-query test harness caught issues when I changed chunking parameters.

4. **Local LLMs need patience**: First load is slow (~2s for embeddings, ~5s for LLM), but subsequent calls are fast.

5. **Documentation is a feature**: The troubleshooting guide would have saved me hours during development.

## Next Steps

1. **Add document loaders**: PDF, HTML, DOCX support
2. **Improve evaluation**: Faithfulness metrics, automated scoring
3. **Add authentication**: Basic auth for web interface
4. **Dockerize**: Single-command deployment
5. **Cloud vector DB**: Pinecone/Weaviate option for scale
6. **Streaming responses**: Better UX for longer generations

## Repository Structure

```
rag-assistant/
├── rag/
│   ├── config.py        # Configuration management
│   ├── ingest.py        # Document ingestion
│   ├── retriever.py     # Vector search
│   ├── llm.py           # Ollama adapter
│   ├── pipeline.py      # RAG orchestration
│   ├── eval_harness.py  # Evaluation metrics
│   ├── cli.py           # Command-line interface
│   └── web/             # FastAPI application
├── kb/                  # Knowledge base documents
├── tests/               # Test suite
├── docs/                # Documentation
└── eval/                # Evaluation results
```

---

*Project completed as part of ML Roadmap Bootcamp*
