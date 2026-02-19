"""
compare.py — Run all three chunking strategies on a paper and compare
=====================================================================

Usage:
  uv run rag-chunk paper.pdf
"""

import sys
from pathlib import Path

from rag.pdf_parser import parse_pdf, print_structure
from rag.chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    HierarchicalChunker,
    print_comparison,
)


def demonstrate_boundary_problem(fixed_chunks: list, semantic_chunks: list):
    """Show why fixed-size chunking breaks sentences."""
    print(f"\n{'='*70}")
    print("BOUNDARY PROBLEM DEMONSTRATION")
    print(f"{'='*70}")

    found = False
    for i, chunk in enumerate(fixed_chunks[:-1]):
        text = chunk.text
        if text and text[-1] not in '.!?:)"\'':
            next_chunk = fixed_chunks[i + 1]
            print(f"\n  Fixed chunk #{chunk.chunk_id} ENDS with:")
            print(f"    ...{text[-80:]!r}")
            print(f"\n  Fixed chunk #{next_chunk.chunk_id} STARTS with:")
            print(f"    {next_chunk.text[:80]!r}...")
            print(f"\n  → Sentence split across boundary!")
            found = True
            break

    if not found:
        print("\n  (No obvious boundary splits found)")

    if found and semantic_chunks:
        target_text = fixed_chunks[i].text[:50]
        for sc in semantic_chunks:
            if target_text in sc.text:
                print(f"\n  Semantic chunk #{sc.chunk_id} keeps the full paragraph:")
                print(f"    {sc.text[:150]!r}...")
                print(f"\n  → Complete thought preserved.")
                break


def demonstrate_context_loss(fixed_chunks: list, hierarchical_chunks: list):
    """Show how hierarchical chunking preserves section context."""
    print(f"\n{'='*70}")
    print("CONTEXT LOSS DEMONSTRATION")
    print(f"{'='*70}")

    if fixed_chunks:
        fc = fixed_chunks[min(5, len(fixed_chunks) - 1)]
        print(f"\n  Fixed chunk #{fc.chunk_id}:")
        print(f"    text: {fc.text[:120]!r}...")
        print(f"    metadata: {fc.metadata}")
        print(f"    → No section context.")

    if hierarchical_chunks:
        hc = hierarchical_chunks[min(5, len(hierarchical_chunks) - 1)]
        print(f"\n  Hierarchical chunk #{hc.chunk_id}:")
        print(f"    text: {hc.text[:120]!r}...")
        print(f"    metadata: {hc.metadata}")
        print(f"    → Section context in text AND metadata.")


def main():
    """Entry point for `uv run rag-chunk <paper.pdf>`"""
    if len(sys.argv) < 2:
        print("Usage: uv run rag-chunk <paper.pdf>")
        print("\nGrab a paper first:")
        print("  wget https://arxiv.org/pdf/1706.03762 -O papers/attention.pdf")
        print("  uv run rag-chunk papers/attention.pdf")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    print(f"\nParsing: {filepath.name}")
    print("=" * 70)

    # Parse
    doc = parse_pdf(filepath)
    print_structure(doc)

    # Chunk with all three strategies
    print(f"\nChunking with three strategies...")

    fixed = FixedSizeChunker(chunk_size=1000, overlap=200)
    fixed_chunks = fixed.chunk(doc.raw_text, {"source": doc.filename})

    semantic = SemanticChunker(max_chunk_size=1500, min_chunk_size=100)
    semantic_chunks = semantic.chunk(doc.raw_text, {"source": doc.filename})

    hierarchical = HierarchicalChunker(max_chunk_size=1500, include_context_header=True)
    hier_chunks = hierarchical.chunk(doc.sections, {"source": doc.filename})

    # Compare
    print_comparison({
        "Fixed (1000/200)": fixed_chunks,
        "Semantic (1500)": semantic_chunks,
        "Hierarchical": hier_chunks,
    })

    # Demonstrate problems
    demonstrate_boundary_problem(fixed_chunks, semantic_chunks)
    demonstrate_context_loss(fixed_chunks, hier_chunks)

    # Takeaways
    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. Fixed-size is a baseline, not a solution. Predictable but
     destroys sentence boundaries and has zero structural awareness.

  2. Semantic chunking respects natural text boundaries. Should be
     your default over fixed-size.

  3. Hierarchical is best for structured docs. The section header
     helps both the embedding model AND the LLM understand context.

  4. Chunk BOUNDARIES matter more than chunk SIZE.

  Next: Module 2 (Embedding & Vector Storage) — where you'll see
  how these chunk quality differences affect retrieval.
    """)