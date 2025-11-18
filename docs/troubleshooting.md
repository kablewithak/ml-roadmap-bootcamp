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

## Ollama Setup on Windows

### Installation

1. Download from https://ollama.ai
2. Run the installer
3. Ollama runs as a system service automatically

### Pulling Models

```powershell
# Pull the recommended model
ollama pull llama3.2:3b

# Verify it downloaded
ollama list
```

### Example Session

```powershell
PS> ollama list
NAME           ID           SIZE   MODIFIED
llama3.2:3b    abc123...    2.0GB  2 hours ago

PS> ollama run llama3.2:3b "Hello, what can you do?"
Hello! I'm a helpful AI assistant...
```

---

*Last updated: Initial project setup*
