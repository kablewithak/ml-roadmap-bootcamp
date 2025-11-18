# Troubleshooting Guide

Common issues and solutions for RAG Assistant.

## Ollama Issues

### "Connection refused" error

**Symptom**: Error connecting to Ollama at localhost:11434

**Solutions**:
1. Ensure Ollama is running:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags

   # Start Ollama (if not running)
   ollama serve
   ```

2. Check the port:
   ```bash
   # Verify Ollama is listening
   netstat -an | grep 11434
   ```

### Model not found

**Symptom**: "model not found" error when asking questions

**Solution**: Pull the model first:
```bash
ollama pull llama3.2:3b
```

### Slow response times

**Symptom**: LLM takes >30 seconds to respond

**Solutions**:
1. Try a smaller model: `OLLAMA_MODEL=llama3.2:1b`
2. Reduce `LLM_MAX_TOKENS` in `.env`
3. Ensure sufficient RAM (8GB+ recommended)

## ChromaDB Issues

### "No such collection" error

**Symptom**: Error when querying before ingestion

**Solution**: Run ingestion first:
```bash
python -m rag.cli ingest
```

### Stale index

**Symptom**: New documents not appearing in results

**Solution**: Re-run ingestion to rebuild the index:
```bash
python -m rag.cli ingest --force
```

## Embedding Issues

### Out of memory

**Symptom**: OOM error during ingestion

**Solutions**:
1. Reduce batch size for embedding
2. Process documents in smaller batches

## General Issues

### Import errors

**Symptom**: ModuleNotFoundError

**Solutions**:
1. Ensure virtual environment is activated
2. Reinstall: `pip install -e ".[dev]"`

### Configuration not loading

**Symptom**: Default values used instead of `.env` settings

**Solutions**:
1. Ensure `.env` file exists in project root
2. Check for typos in variable names
3. Restart the application after changes

---

## Ollama Setup Guide

### Installation

**Windows**:
1. Download installer from https://ollama.ai
2. Run the installer (follows standard Windows installation)
3. Ollama runs as a system service automatically

**Linux**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS**:
```bash
brew install ollama
```

### Pulling Models

```bash
# Pull the recommended model (Llama 3.2 3B - 2GB download)
ollama pull llama3.2:3b

# Alternative: Larger 7B model (if you have 16GB+ RAM)
ollama pull llama3.2:7b

# Verify installation
ollama list
```

### Example Session Output

```
$ ollama list
NAME           ID              SIZE      MODIFIED
llama3.2:3b    a80c4f17acd5    2.0 GB    2 hours ago

$ ollama run llama3.2:3b "What is RAG in AI?"
RAG stands for Retrieval-Augmented Generation. It's a technique that
combines information retrieval with text generation. The system first
retrieves relevant documents from a knowledge base, then uses those
documents as context for a language model to generate more accurate
and grounded responses...
```

### Verifying Ollama is Running

```bash
# Check API is responding
curl http://localhost:11434/api/tags

# Expected output: {"models":[{"name":"llama3.2:3b",...}]}
```

### Model Selection Guide

| Model | Size | RAM Required | Use Case |
|-------|------|-------------|----------|
| llama3.2:1b | 1.3 GB | 4GB | Testing, quick responses |
| llama3.2:3b | 2.0 GB | 8GB | **Recommended** - good quality/speed balance |
| llama3.2:7b | 4.7 GB | 16GB | Higher quality, slower |

### Command Reference

```bash
# List downloaded models
ollama list

# Pull a model
ollama pull llama3.2:3b

# Run interactive chat
ollama run llama3.2:3b

# Show model details
ollama show llama3.2:3b

# Remove a model
ollama rm llama3.2:3b

# Start server (usually auto-started)
ollama serve
```

---

*Last updated: Block 4 - Ollama setup documentation*
