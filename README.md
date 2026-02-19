# Module 1: Document Ingestion & Chunking

## Why Start Here

Most RAG tutorials jump straight to embeddings and vector databases. That's backwards. Chunking is where most RAG systems silently fail — you get bad answers and blame the LLM or the embedding model, when the real problem is that your chunks are garbage.

Think of it this way: if you feed the LLM a chunk that says *"ture uses multi-head attention to process"* because your chunker split the word "architecture" across two chunks, no amount of GPT-4 is going to save you.

## Setup

```bash
pip install pymupdf sentence-transformers
```

Grab a paper to test with:
```bash
# Attention Is All You Need
wget https://arxiv.org/pdf/1706.03762 -O attention.pdf

# Or any paper you're interested in
```

## Files

| File | What it does |
|------|-------------|
| `pdf_parser.py` | Extracts text from research paper PDFs, detects sections, cleans artifacts |
| `chunkers.py` | Three chunking strategies: Fixed, Semantic, Hierarchical |
| `compare.py` | Runs all three on a paper and shows why the differences matter |

## Run It

```bash
python compare.py attention.pdf
```

This will show you:
- Document structure (detected sections)
- Side-by-side stats for all three strategies
- Concrete examples of boundary problems (fixed-size)
- Context loss demonstration (fixed vs hierarchical)

## The Three Strategies

### 1. Fixed-Size Chunking
Split every N characters with M overlap. Fast, dumb, predictable.

```
"The transformer architecture uses multi-head atten|tion to process sequences"
                                                   ^
                                          chunk boundary cuts here
```

**Use when:** You need a quick baseline or your documents have no structure.

### 2. Semantic Chunking
Split at paragraph and sentence boundaries. Never cuts mid-sentence.

```
Paragraph 1: "The transformer architecture uses multi-head attention..."  → Chunk 1
Paragraph 2: "Training was performed on 8 NVIDIA P100 GPUs..."           → Chunk 2
```

**Use when:** Default choice. Better than fixed-size in almost every case.

### 3. Hierarchical Chunking
Section-aware — each chunk knows which section it belongs to.

```
[Section: 3.2 Multi-Head Attention]
Multi-head attention allows the model to jointly attend to information
from different representation subspaces at different positions...
```

**Use when:** Structured documents (papers, docs, legal). The section header in the chunk text dramatically improves both embedding quality and LLM comprehension.

## What to Notice

When you run `compare.py`, pay attention to:

1. **Chunk count varies wildly** — Fixed-size gives you the most chunks (smallest avg size). Hierarchical gives fewer, more meaningful chunks.

2. **Size variance** — Fixed has near-zero variance (by design). Semantic and Hierarchical have high variance because they respect natural boundaries.

3. **Boundary quality** — Look at where chunks start and end. Fixed chunks will start/end mid-thought. Semantic chunks always end at sentence boundaries.

4. **Context preservation** — Hierarchical chunks carry `[Section: ...]` headers. When these get embedded in Module 2, the vector will capture both the content AND its location in the paper.

## Exercises

Before moving to Module 2, try these:

1. **Change chunk sizes** — Run fixed-size with 500, 1000, 2000 chars. How does chunk count change? At what size do chunks start containing multiple topics?

2. **Try different papers** — A math-heavy paper (lots of equations) vs a survey paper (lots of text). Which chunker handles each better?

3. **Inspect the worst chunks** — Find the shortest and longest chunks from each strategy. Are the short ones useful? Are the long ones focused?

4. **Break the section detector** — Find a paper where section detection fails. Why did it fail? What pattern would fix it?

## What's Next: Module 2

Module 2 will take these chunks and embed them — convert text to vectors. You'll see:
- How different chunks produce different embeddings
- Why the hierarchical context header improves retrieval
- Dense vs sparse retrieval and when each wins
- Building a FAISS index from scratch

The quality of Module 2's output is entirely determined by what we built here. Bad chunks → bad embeddings → bad retrieval → wrong answers. That's the RAG quality chain.
