# Design Decisions

This document captures key architectural and implementation decisions for the RAG Assistant project.

## Stack Selection

### Embeddings: sentence-transformers (all-MiniLM-L6-v2)

**Decision**: Use `all-MiniLM-L6-v2` as the default embedding model.

**Rationale**:
- CPU-friendly: Runs efficiently without GPU, making it accessible
- Small footprint: ~80MB model size
- Good quality: Achieves strong performance on semantic similarity tasks
- Widely used: Large community, good documentation, battle-tested

**Trade-offs**:
- Not the highest quality embeddings available (but sufficient for local KB)
- 384 dimensions (smaller than some alternatives like 768 or 1536)

**Future work**: Support swapping to larger models like `all-mpnet-base-v2` or API-based embeddings.

### Vector Store: ChromaDB

**Decision**: Use ChromaDB for vector storage and retrieval.

**Rationale**:
- Zero-cost: Fully open-source, runs locally
- Simple API: Easy to integrate with LangChain
- Persistent: Supports saving to disk
- Good defaults: Built-in distance metrics and metadata filtering

**Trade-offs**:
- Not suitable for massive scale (millions of documents)
- Less feature-rich than Pinecone/Weaviate for production use

**Future work**: Abstract vector store interface to support swapping to cloud providers.

### LLM: Ollama + Llama 3.2 3B

**Decision**: Use Ollama to serve Llama 3.2 3B locally.

**Rationale**:
- Privacy: All data stays on your machine
- Cost: No API fees
- Learning: Teaches local LLM serving patterns
- Flexibility: Easy model swapping (just `ollama pull`)

**Trade-offs**:
- Lower quality than GPT-4/Claude for complex reasoning
- Requires decent hardware for 7B+ models
- Slower inference than cloud APIs

**Future work**: Add provider abstraction (Ollama/OpenAI/Anthropic).

### Orchestration: LangChain

**Decision**: Use LangChain for RAG orchestration.

**Rationale**:
- Industry standard: Widely used, good for career skills
- Comprehensive: Built-in document loaders, splitters, retrievers
- Extensible: Easy to customize components

**Trade-offs**:
- Heavy dependency
- Rapid API changes (breaking changes between versions)
- Abstraction overhead

**Future work**: Consider LlamaIndex or custom implementation for comparison.

## Architecture Decisions

### Configuration via Environment Variables

**Decision**: All configuration through `.env` file and environment variables.

**Rationale**:
- Security: Secrets not in code
- Flexibility: Easy to change without code changes
- Standard pattern: Well-understood in industry

### Modular Design

**Decision**: Separate modules for ingest, retriever, llm, pipeline.

**Rationale**:
- Testability: Each component testable in isolation
- Maintainability: Clear responsibilities
- Extensibility: Easy to swap implementations

### CLI-First Interface

**Decision**: Build CLI first, web UI second.

**Rationale**:
- Faster iteration
- Forces clean API design
- Useful for automation/scripting

## Parameter Decisions

### Chunking: 500 characters, 50 overlap

**Decision**: Default chunk size of 500 with 50-character overlap.

**Rationale**:
- Balances context and retrieval precision
- Overlap prevents losing information at boundaries

**Future work**: Document parameter sweep results in this file.

### LLM Parameters

**Decision**: Use temperature=0.7, top_p=0.9, max_tokens=512 as defaults.

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | 0.7 | Balanced creativity/coherence for QA tasks. Lower (0.3) for more deterministic, higher (0.9) for creative. |
| `top_p` | 0.9 | Nucleus sampling threshold. 0.9 keeps 90% probability mass, reducing unlikely tokens while maintaining variety. |
| `max_tokens` | 512 | Sufficient for detailed answers without excessive generation. Increase for longer explanations. |

**Rationale**:
- **Temperature 0.7**: RAG QA benefits from some creativity in phrasing while staying factual. We're not doing pure extraction (would use 0.1) or creative writing (would use 0.9+).
- **Top-p 0.9**: Works well with temperature 0.7 to filter out low-probability tokens that could derail coherence.
- **Max tokens 512**: Most QA answers fit in 200-400 tokens; 512 gives headroom without runaway generation.

**Trade-offs**:
- Higher temperature can cause hallucinations or drifting from source material
- Lower max_tokens may truncate complex multi-part answers
- These are heuristics; optimal values depend on your specific KB and questions

**Future work**: Add streaming support for better UX on longer responses.

### Retrieval: top_k=4, score_threshold=0.3

**Decision**: Retrieve 4 chunks by default with a minimum similarity threshold of 0.3.

**Rationale**:
- **top_k=4**: Provides enough context without overwhelming the LLM's context window. Most questions are answered by 2-3 relevant chunks.
- **score_threshold=0.3**: Filters out poor matches. Cosine similarity below 0.3 typically indicates weak relevance.

**Trade-offs**:
- Higher k provides more context but includes less relevant chunks
- Threshold too high may miss partial matches; too low allows noise

---

*Last updated: Block 4 - Ollama setup and LLM parameters*
