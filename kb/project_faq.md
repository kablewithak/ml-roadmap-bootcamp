# Project FAQ

## What is this project?

This is a RAG (Retrieval-Augmented Generation) Assistant that answers questions using a local knowledge base. It combines vector search for finding relevant documents with a local LLM (Ollama) for generating natural language answers. The system is designed to be privacy-preserving since all processing happens on your machine.

The project demonstrates key skills for ML/AI engineering: working with embeddings, vector databases, LLM orchestration, and building end-to-end pipelines. It uses an open-source stack including sentence-transformers for embeddings, ChromaDB for vector storage, and LangChain for orchestration.

## How do I get started?

First, install Ollama from https://ollama.ai and pull the language model: `ollama pull llama3.2:3b`. Then create a Python virtual environment and install dependencies with `pip install -e ".[dev]"`. Copy `.env.example` to `.env` to configure the application. Run `python -m rag.cli ingest` to process the knowledge base, then `python -m rag.cli ask "your question"` to query it.

If you want to modify the knowledge base, add or edit markdown files in the `kb/` directory, then re-run the ingestion command. The system will create new embeddings and update the ChromaDB index. You can customize retrieval parameters like chunk size and top-k in the `.env` file.

## What can I ask?

You can ask any question related to the content in your knowledge base. The system will find the most relevant document chunks and use them to generate an answer. Questions about Python basics, AI/ML concepts, project setup, and company policies are covered by the default KB files.

The quality of answers depends on: (1) whether relevant information exists in the KB, (2) the chunk size and overlap settings, (3) the number of chunks retrieved (top-k), and (4) the LLM's ability to synthesize information. If answers are poor, try adjusting these parameters or adding more detailed content to the KB.

## How do I evaluate quality?

The project includes an evaluation harness that runs predefined queries and checks if expected information appears in the results. Run `python -m rag.eval_harness` to generate a report. The harness computes hit@k metrics showing whether correct chunks are retrieved in the top results.

For manual evaluation, use the `show-sources` command to see which chunks were retrieved for the last query. Check if these chunks are relevant and if the answer accurately reflects their content. Document qualitative observations in the evaluation report to track improvements over time.

## What are the system requirements?

The project requires Python 3.10+ and Ollama for LLM serving. Recommended hardware: 8GB+ RAM for the 3B model, 16GB+ for 7B models. CPU is sufficient for the embedding model (all-MiniLM-L6-v2), though a GPU will speed up ingestion for large knowledge bases.

Storage requirements depend on your knowledge base size. ChromaDB stores embeddings efficiently, but expect roughly 1-2MB per 1000 document chunks. The Ollama model files are stored separately and range from 2GB (3B) to 4GB (7B) in size.
