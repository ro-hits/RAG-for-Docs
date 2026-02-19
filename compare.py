"""
compare.py — Run all three chunking strategies on a paper and compare
=====================================================================

Usage:
  python compare.py <paper.pdf>

This script:
  1. Parses the PDF into sections
  2. Runs all three chunkers
  3. Shows stats comparison
  4. Demonstrates WHY the differences matter for retrieval

The point isn't to find the "best" chunker. It's to understand
what each one does to your text so you can make informed decisions
when your RAG system gives bad answers.
"""

import sys
from pathlib import Path

from pdf_parser import parse_pdf, print_structure
from chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    HierarchicalChunker,
    print_comparison,
)


def demonstrate_boundary_problem(fixed_chunks: list, semantic_chunks: list):
    """
    Show a concrete example of why fixed-size chunking breaks sentences.

    This is the #1 reason fixed-size chunking hurts retrieval quality.
    If a user asks "What is the attention mechanism?" and the answer
    sentence is split across two chunks, NEITHER chunk will be retrieved
    because neither contains the full answer.
    """
    print(f"\n{'='*70}")
    print("BOUNDARY PROBLEM DEMONSTRATION")
    print(f"{'='*70}")

    # Find a fixed chunk that ends mid-sentence
    found = False
    for i, chunk in enumerate(fixed_chunks[:-1]):
        text = chunk.text
        # Check if chunk ends without sentence-ending punctuation
        if text and text[-1] not in '.!?:)"\'':
            next_chunk = fixed_chunks[i + 1]
            print(f"\n  Fixed chunk #{chunk.chunk_id} ENDS with:")
            print(f"    ...{text[-80:]!r}")
            print(f"\n  Fixed chunk #{next_chunk.chunk_id} STARTS with:")
            print(f"    {next_chunk.text[:80]!r}...")
            print(f"\n  → Sentence split across boundary! Neither chunk has the full thought.")
            found = True
            break

    if not found:
        print("\n  (No obvious boundary splits found in this document)")

    # Show how semantic chunker handles the same region
    if found and semantic_chunks:
        # Find the semantic chunk covering the same region
        target_pos = fixed_chunks[i].metadata.get("char_start", 0)
        for sc in semantic_chunks:
            if sc.text[:50] in fixed_chunks[i].text or fixed_chunks[i].text[:50] in sc.text:
                print(f"\n  Semantic chunk #{sc.chunk_id} keeps the full paragraph:")
                print(f"    {sc.text[:150]!r}...")
                print(f"\n  → Complete thought preserved. Better for retrieval.")
                break


def demonstrate_context_loss(fixed_chunks: list, hierarchical_chunks: list):
    """
    Show how hierarchical chunking preserves WHERE text came from.

    When the LLM gets a chunk that says "The results show 95% accuracy",
    it needs to know: accuracy of WHAT? Which experiment? Which section?
    """
    print(f"\n{'='*70}")
    print("CONTEXT LOSS DEMONSTRATION")
    print(f"{'='*70}")

    # Find a fixed chunk with no section info
    if fixed_chunks:
        fc = fixed_chunks[min(5, len(fixed_chunks) - 1)]
        print(f"\n  Fixed chunk #{fc.chunk_id}:")
        print(f"    text: {fc.text[:120]!r}...")
        print(f"    metadata: {fc.metadata}")
        print(f"    → No section context. LLM doesn't know where this came from.")

    # Same-ish content from hierarchical
    if hierarchical_chunks:
        hc = hierarchical_chunks[min(5, len(hierarchical_chunks) - 1)]
        print(f"\n  Hierarchical chunk #{hc.chunk_id}:")
        print(f"    text: {hc.text[:120]!r}...")
        print(f"    metadata: {hc.metadata}")
        print(f"    → Section context embedded in text AND metadata.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare.py <paper.pdf>")
        print("\nDownload a paper to test with:")
        print("  wget https://arxiv.org/pdf/1706.03762 -O attention.pdf")
        print("  python compare.py attention.pdf")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    print(f"\nParsing: {filepath.name}")
    print("=" * 70)

    # Step 1: Parse PDF
    doc = parse_pdf(filepath)
    print_structure(doc)

    # Step 2: Run all three chunkers
    print(f"\nChunking with three strategies...")

    # Strategy 1: Fixed size
    fixed = FixedSizeChunker(chunk_size=1000, overlap=200)
    fixed_chunks = fixed.chunk(doc.raw_text, {"source": doc.filename})

    # Strategy 2: Semantic
    semantic = SemanticChunker(max_chunk_size=1500, min_chunk_size=100)
    semantic_chunks = semantic.chunk(doc.raw_text, {"source": doc.filename})

    # Strategy 3: Hierarchical (uses sections from parser)
    hierarchical = HierarchicalChunker(max_chunk_size=1500, include_context_header=True)
    hierarchical_chunks = hierarchical.chunk(doc.sections, {"source": doc.filename})

    # Step 3: Compare stats
    print_comparison({
        "Fixed (1000/200)": fixed_chunks,
        "Semantic (1500)": semantic_chunks,
        "Hierarchical": hierarchical_chunks,
    })

    # Step 4: Show why it matters
    demonstrate_boundary_problem(fixed_chunks, semantic_chunks)
    demonstrate_context_loss(fixed_chunks, hierarchical_chunks)

    # Step 5: Key takeaways
    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. Fixed-size is a baseline, not a solution. It's predictable but
     destroys sentence boundaries and has zero structural awareness.

  2. Semantic chunking respects natural text boundaries. It costs
     nothing extra and should be your default over fixed-size.

  3. Hierarchical chunking is best for structured docs (papers, docs,
     legal). The section context header is the secret weapon — it
     helps both the embedding model AND the LLM understand context.

  4. Chunk SIZE matters less than chunk BOUNDARIES. A 800-char chunk
     with a complete thought beats a 1000-char chunk split mid-sentence.

  5. In Module 2 (Embedding), we'll see how these different chunks
     produce different embeddings and how that affects retrieval.
    """)


if __name__ == "__main__":
    main()
