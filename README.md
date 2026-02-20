# RAG from Scratch

A research paper Q&A system built piece by piece — no LangChain, no LlamaIndex.

## Setup

```bash
uv sync
```

That's it. `uv` handles the virtualenv, dependencies, everything.

## Quick Start

```bash
# Grab a paper
wget https://arxiv.org/pdf/1706.03762 -O papers/attention.pdf

# Parse and see structure
uv run rag-parse papers/attention.pdf

# Chunk with all 3 strategies and compare
uv run rag-chunk papers/attention.pdf

# Build index, compare dense vs sparse retrieval
uv run rag-index papers/attention.pdf

# Hybrid search + reranking (the full pipeline)
uv run rag-search papers\attention.pdf "What is multi-head attention?"

# Ask questions — full RAG pipeline with cited answers
uv run rag-ask papers\attention.pdf "What is multi-head attention?"

# Interactive mode — keep asking without re-indexing
uv run rag-ask papers\attention.pdf

# Evaluate — measure retrieval + answer quality
uv run rag-eval papers\attention.pdf --model claude

# Compare models
uv run rag-eval papers\attention.pdf --model claude --model deepseek
```

**First time setup for generation:**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

## Project Structure

```
rag-from-scratch/
├── pyproject.toml          # Dependencies, CLI scripts
├── papers/                 # Drop PDFs here
├── index/                  # Saved FAISS index (auto-created)
└── src/rag/
    ├── __init__.py
    ├── pdf_parser.py       # Module 1: PDF → structured sections
    ├── chunkers.py         # Module 1: Three chunking strategies
    ├── compare.py          # Module 1: Run & compare chunkers
    ├── embedder.py         # Module 2: Text → vectors
    ├── vector_store.py     # Module 2: FAISS index + search
    ├── sparse.py           # Module 2: BM25 keyword search
    ├── index_and_search.py # Module 2: Build index & compare retrieval
    ├── hybrid.py           # Module 3: RRF fusion + cross-encoder reranking
    ├── generator.py        # Module 4: Multi-provider LLM, citations, faithfulness
    ├── ask.py              # Module 4: Full pipeline CLI
    ├── eval_data.py        # Module 5: Test set format + sample QA pairs
    └── evaluate.py         # Module 5: Retrieval & answer quality metrics
```

## Modules

| Module | Status | What it covers |
|--------|--------|----------------|
| 1. Chunking | ✅ | PDF parsing, fixed/semantic/hierarchical chunking |
| 2. Embedding | ✅ | Vector encoding, FAISS index, dense vs sparse (BM25) |
| 3. Retrieval | ✅ | Hybrid search (RRF), cross-encoder reranking |
| 4. Generation | ✅ | Claude API, citation grounding, faithfulness check |
| 5. Evaluation | ✅ | Retrieval metrics (Recall, MRR), answer quality, model comparison |

## Why from Scratch?

Frameworks like LangChain hide the decisions that matter. When your RAG gives wrong answers, you need to know: is it the chunking? The embedding model? The retrieval? The prompt? Building each piece yourself means you understand the full chain.