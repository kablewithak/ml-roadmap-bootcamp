# AI and RAG Glossary

## Core Concepts

**RAG (Retrieval-Augmented Generation)**: A technique that enhances LLM responses by retrieving relevant information from a knowledge base before generation. Instead of relying solely on the model's training data, RAG systems find specific documents and include them in the prompt, leading to more accurate and up-to-date answers with citations.

**Embedding**: A dense vector representation of text that captures semantic meaning. Similar texts have similar embeddings (close in vector space). Embeddings are created by neural network models trained on large text corpora. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings.

**Vector Database**: A database optimized for storing and searching high-dimensional vectors. Unlike traditional databases that search by exact match, vector databases find similar items using distance metrics like cosine similarity or Euclidean distance. ChromaDB, Pinecone, and Weaviate are popular choices.

**LLM (Large Language Model)**: A neural network trained on vast amounts of text to generate human-like text. Examples include GPT-4, Claude, Llama, and Mistral. LLMs can answer questions, write code, summarize documents, and perform many language tasks based on their training.

## Retrieval Concepts

**Chunking**: The process of splitting documents into smaller pieces for embedding and retrieval. Chunks should be small enough for precise retrieval but large enough to contain meaningful context. Typical sizes range from 200-1000 characters with some overlap.

**Top-k Retrieval**: Retrieving the k most similar documents to a query based on embedding similarity. Higher k values provide more context but may include less relevant results. Common values are k=3 to k=10 depending on the use case.

**MMR (Maximal Marginal Relevance)**: A retrieval strategy that balances relevance with diversity. Instead of just taking the most similar documents, MMR also considers how different each result is from already-selected results, reducing redundancy in retrieved content.

**Semantic Search**: Searching based on meaning rather than keyword matching. A query for "programming languages" would match documents about "coding" or "software development" even without those exact words, because the embeddings capture semantic similarity.

## LLM Concepts

**Prompt Engineering**: The practice of crafting effective prompts to get desired outputs from LLMs. This includes system prompts (instructions for behavior), few-shot examples (showing desired format), and careful wording of questions. Good prompts are clear, specific, and provide necessary context.

**Temperature**: A parameter controlling randomness in LLM outputs. Temperature 0 gives deterministic outputs (always the most likely token), while higher values (0.7-1.0) introduce more variety. Lower temperatures are better for factual tasks; higher for creative tasks.

**Context Window**: The maximum number of tokens an LLM can process at once, including both the prompt and the response. Models have different context limits: 4K, 8K, 32K, or 128K tokens. Longer contexts allow more retrieved documents but increase cost and latency.

**Hallucination**: When an LLM generates plausible-sounding but incorrect information. RAG helps reduce hallucinations by grounding responses in retrieved documents. However, LLMs can still misinterpret or incorrectly synthesize retrieved information.

## Evaluation Concepts

**Precision@k**: The fraction of retrieved documents that are relevant, out of the top k results. If 3 out of 4 retrieved chunks are relevant, precision@4 is 0.75. Higher precision means less noise in retrieved results.

**Recall**: The fraction of all relevant documents that were retrieved. High recall means the system found most of the relevant information. There's often a trade-off between precision and recall.

**Faithfulness**: How accurately the generated answer reflects the retrieved documents. A faithful answer only includes information present in the sources, without adding invented details. This is harder to measure automatically than retrieval quality.
