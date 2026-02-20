"""
index_and_search.py — Build index from a paper, compare dense vs sparse
========================================================================

This is the "aha moment" script. You'll see:
  1. Same paper, same chunks → different retrieval results
  2. Queries where dense search wins (semantic matching)
  3. Queries where BM25 wins (exact keyword matching)
  4. How hierarchical chunks retrieve better than fixed chunks

Usage:
  uv run rag-index papers/attention.pdf
"""

import sys
from pathlib import Path

from rag.pdf_parser import parse_pdf
from rag.chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    HierarchicalChunker,
)
from rag.embedder import Embedder
from rag.vector_store import VectorStore, print_results
from rag.sparse import BM25Index


# Queries designed to show the strengths/weaknesses of each method
DEMO_QUERIES = [
    # Semantic — dense search should win
    {
        "query": "How does the model handle long-range dependencies in sequences?",
        "note": "SEMANTIC: No exact keywords match, but the concept is about attention.",
    },
    # Keyword — BM25 should win
    {
        "query": "BLEU score English to German",
        "note": "KEYWORD: Specific metrics and proper nouns. BM25 finds exact terms.",
    },
    # Mixed — both should contribute
    {
        "query": "What is multi-head attention and why is it used?",
        "note": "MIXED: 'multi-head attention' is both a keyword and a concept.",
    },
    # Section-specific — hierarchical chunks should win
    {
        "query": "training details and hyperparameters",
        "note": "STRUCTURAL: Hierarchical chunks with [Section: Training] context help here.",
    },
]


def compare_retrieval(
    query: str,
    dense_results: list[dict],
    sparse_results: list[dict],
    note: str = "",
):
    """Side-by-side comparison of dense vs sparse results for a query."""
    print(f"\n{'='*70}")
    if note:
        print(f"  {note}")
    print(f"  Query: {query!r}")
    print(f"{'='*70}")

    print(f"\n  {'DENSE (embedding similarity)':<38} {'SPARSE (BM25 keywords)':<38}")
    print(f"  {'-'*36}   {'-'*36}")

    max_rows = max(len(dense_results), len(sparse_results))
    for i in range(min(max_rows, 5)):
        left = ""
        right = ""

        if i < len(dense_results):
            d = dense_results[i]
            section = d["chunk"].metadata.get("section", "—")[:20]
            left = f"#{d['rank']} {d['score']:.3f} [{section}]"

        if i < len(sparse_results):
            s = sparse_results[i]
            section = s["chunk"].metadata.get("section", "—")[:20]
            right = f"#{s['rank']} {s['score']:.3f} [{section}]"

        print(f"  {left:<38} {right:<38}")

    # Check overlap — are they finding the same chunks?
    dense_ids = {d["chunk"].chunk_id for d in dense_results[:5]}
    sparse_ids = {s["chunk"].chunk_id for s in sparse_results[:5]}
    overlap = dense_ids & sparse_ids
    print(f"\n  Overlap in top-5: {len(overlap)}/5 chunks found by both methods")


def compare_chunk_strategies(
    query: str,
    fixed_store: VectorStore,
    hier_store: VectorStore,
):
    """Show how chunk strategy affects retrieval quality."""
    print(f"\n{'='*70}")
    print(f"  CHUNK STRATEGY COMPARISON")
    print(f"  Query: {query!r}")
    print(f"{'='*70}")

    fixed_results = fixed_store.search(query, top_k=3)
    hier_results = hier_store.search(query, top_k=3)

    print(f"\n  Fixed-size chunks:")
    for r in fixed_results:
        preview = r["chunk"].text[:100].replace('\n', ' ')
        print(f"    #{r['rank']} score={r['score']:.3f}: {preview}...")

    print(f"\n  Hierarchical chunks:")
    for r in hier_results:
        preview = r["chunk"].text[:100].replace('\n', ' ')
        print(f"    #{r['rank']} score={r['score']:.3f}: {preview}...")


def main():
    """Entry point for `uv run rag-index <paper.pdf>`"""
    if len(sys.argv) < 2:
        print("Usage: uv run rag-index papers/attention.pdf")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    print(f"\n{'='*70}")
    print(f"  MODULE 2: Embedding & Retrieval Demo")
    print(f"  Paper: {filepath.name}")
    print(f"{'='*70}")

    # Step 1: Parse
    print("\n[1/5] Parsing PDF...")
    doc = parse_pdf(filepath)
    print(f"  {doc.total_pages} pages, {len(doc.sections)} sections, {len(doc.raw_text):,} chars")

    # Step 2: Chunk with two strategies
    print("\n[2/5] Chunking...")
    fixed_chunker = FixedSizeChunker(chunk_size=1000, overlap=200)
    fixed_chunks = fixed_chunker.chunk(doc.raw_text, {"source": doc.filename})

    hier_chunker = HierarchicalChunker(max_chunk_size=1500, include_context_header=True)
    hier_chunks = hier_chunker.chunk(doc.sections, {"source": doc.filename})
    print(f"  Fixed: {len(fixed_chunks)} chunks | Hierarchical: {len(hier_chunks)} chunks")

    # Step 3: Embed and index
    print("\n[3/5] Embedding & indexing...")
    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    # Dense stores — one per chunk strategy
    fixed_store = VectorStore(embedder)
    fixed_store.add_chunks(fixed_chunks)

    hier_store = VectorStore(embedder)
    hier_store.add_chunks(hier_chunks)

    # Sparse index — hierarchical chunks only
    bm25 = BM25Index()
    bm25.add_chunks(hier_chunks)

    # Step 4: Compare dense vs sparse
    print("\n[4/5] Dense vs Sparse retrieval comparison...")
    for demo in DEMO_QUERIES:
        dense_results = hier_store.search(demo["query"], top_k=5)
        sparse_results = bm25.search(demo["query"], top_k=5)
        compare_retrieval(demo["query"], dense_results, sparse_results, demo["note"])

    # Step 5: Compare chunk strategies
    print("\n[5/5] Chunk strategy comparison...")
    compare_chunk_strategies(
        "How does the model process input sequences?",
        fixed_store,
        hier_store,
    )

    # Save the hierarchical index for Module 3
    index_dir = Path("index")
    hier_store.save(index_dir)

    # Takeaways
    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. Dense search finds semantically similar content even without
     exact keyword matches. Great for "how does X work?" questions.

  2. BM25 finds exact terms. Essential for "what is the BLEU score?"
     type questions with specific numbers, names, acronyms.

  3. Neither method alone is sufficient. Module 3 combines them
     (hybrid search) for the best of both.

  4. Hierarchical chunks with section headers retrieve better because
     the embedding captures WHAT the text says AND WHERE it is.

  5. Index saved to ./index/ — Module 3 will load it for hybrid search.
    """)