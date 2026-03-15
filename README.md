# CodeChat RAG Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline for codebase understanding. Given a code repository, it indexes all source files and answers natural-language questions about the code using semantic search + an LLM.

---

## Architecture

```
Repository Files
      │
      ▼
 ingestion.py        ← loads supported source files (.py, .js, .ts, .java, …)
      │
      ▼
 chunking.py         ← splits files into function/class-level chunks (AST + tree-sitter)
      │
      ▼
 embeddings.py       ← generates semantic embeddings (SentenceTransformers)
      │
      ▼
 vector_store.py     ← stores and retrieves vectors (FAISS / InMemory)
      │
      ▼
 prompt_builder.py   ← assembles context + user query into an LLM prompt
      │
      ▼
 rag_pipeline.py     ← orchestrates all stages; supports Ollama, OpenAI, Anthropic
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Index a repository
```bash
python3 rag_pipeline.py index --repo /path/to/repo --index-path ./saved_index
```

### 3. Query the codebase
```bash
python3 rag_pipeline.py query \
  --index-path ./saved_index \
  --query "How does authentication work?"
```

### 4. Interactive mode
```bash
python3 rag_pipeline.py interactive --index-path ./saved_index
```

### 5. Validate pipeline (no LLM required)
```bash
python3 test_pipeline.py
```

---

## Python API

```python
from rag_pipeline import RAGPipeline, RAGConfig

config = RAGConfig(
    llm_provider="ollama",   # or "openai" / "anthropic"
    llm_model="llama3.2",
    top_k=5,
)

pipeline = RAGPipeline(config)
pipeline.index_repository("/path/to/repo")        # run once

response = pipeline.query("How does login work?")
print(response.answer)
print(response.retrieved_chunks)                  # [(ChunkMetadata, score), ...]
```

---

## Supported Languages

| Language   | Chunker          |
|------------|-----------------|
| Python     | stdlib `ast`    |
| JavaScript | tree-sitter     |
| TypeScript | tree-sitter     |
| Java       | tree-sitter     |
| JSX / TSX  | tree-sitter     |

---

## LLM Providers

| Provider   | Config                    | Requirement            |
|------------|---------------------------|------------------------|
| Ollama     | `llm_provider="ollama"`   | `ollama serve` running |
| OpenAI     | `llm_provider="openai"`   | `OPENAI_API_KEY` env var |
| Anthropic  | `llm_provider="anthropic"`| `ANTHROPIC_API_KEY` env var |

---

## Running Tests

```bash
pip install pytest pytest-mock faiss-cpu
python3 -m pytest tests/ -v
```

**122 tests — 0 failures.**

---

## Project Structure

```
rag/
├── ingestion.py          # file loading
├── chunking.py           # code chunking
├── embeddings.py         # embedding generation
├── vector_store.py       # FAISS vector store
├── prompt_builder.py     # prompt assembly
├── rag_pipeline.py       # pipeline orchestration
├── test_pipeline.py      # end-to-end validation (no LLM)
├── requirements.txt
├── pytest.ini
└── tests/
    ├── conftest.py
    ├── mock_repo/         # sample Python + JS files for tests
    ├── test_ingestion.py
    ├── test_chunking.py
    ├── test_embeddings.py
    ├── test_vector_store.py
    ├── test_prompt_builder.py
    └── test_rag_pipeline.py
```
