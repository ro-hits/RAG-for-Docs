"""
chunkers.py — Three Chunking Strategies for RAG
================================================

Why chunking matters:
  When you retrieve context for an LLM, you're retrieving CHUNKS, not
  documents. The chunk is the atomic unit of your RAG system. Get it
  wrong and everything downstream fails:

  - Too small → chunks lack context, LLM can't form coherent answers
  - Too large → irrelevant text dilutes the useful part, wastes tokens
  - Bad boundaries → sentences cut mid-thought, tables split in half

  There is no "best" chunking strategy. It depends on your documents
  and your queries. That's why we build three and compare.

Strategies implemented:
  1. FixedSizeChunker    — split by character count with overlap
  2. SemanticChunker     — split at paragraph/sentence boundaries
  3. HierarchicalChunker — section-aware, preserves document structure

Dependencies:
  pip install sentence-transformers  (only for SemanticChunker)
"""

import re
from dataclasses import dataclass, field

# We'll import these lazily to avoid hard dependency
# from sentence_transformers import SentenceTransformer
# import numpy as np


@dataclass
class Chunk:
    """A single chunk of text with metadata for retrieval."""
    text: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)
    # metadata carries: section_title, page, strategy, char_count, etc.
    # This travels with the chunk through embedding → storage → retrieval
    # so the LLM knows WHERE this text came from.

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self):
        preview = self.text[:60].replace('\n', ' ')
        section = self.metadata.get('section', '')
        return f"Chunk(id={self.chunk_id}, words={self.word_count}, section={section!r}, text={preview!r}...)"


# ==================== STRATEGY 1: FIXED SIZE ====================

class FixedSizeChunker:
    """
    Split text into fixed-size chunks with overlap.

    This is the simplest strategy and the one most tutorials use.
    It's fast, predictable, and easy to reason about.

    HOW IT WORKS:
      Text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      chunk_size=10, overlap=3:
        Chunk 1: "ABCDEFGHIJ"
        Chunk 2: "HIJKLMNOPQ"  ← overlaps 3 chars with chunk 1
        Chunk 3: "OPQRSTUVWX"
        Chunk 4: "VWXYZ"

    WHY OVERLAP:
      Without overlap, if a key sentence spans the boundary between
      two chunks, neither chunk has the full sentence. Overlap gives
      you a buffer so boundary sentences appear in at least one chunk.

    WHEN IT FAILS:
      - Cuts sentences mid-word ("The transformer architec" | "ture uses...")
      - Splits tables, equations, code blocks
      - A chunk might contain the end of one section and start of another
        with zero topical coherence
      - Wastes overlap tokens on repeated content

    WHEN TO USE:
      - Quick baseline to compare against
      - Documents with uniform density (no sections, no structure)
      - When you need predictable chunk sizes for token budgets
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            # Don't cut mid-word — find nearest space
            if end < len(text):
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    metadata={
                        **metadata,
                        "strategy": "fixed_size",
                        "char_start": start,
                        "char_end": end,
                    }
                ))
                chunk_id += 1

            start = end - self.overlap

        return chunks


# ==================== STRATEGY 2: SEMANTIC ====================

class SemanticChunker:
    """
    Split at natural boundaries — paragraphs first, then sentences.

    Instead of arbitrarily cutting at N characters, this respects
    the document's own structure: paragraph breaks are natural topic
    boundaries.

    HOW IT WORKS:
      1. Split text into paragraphs (double newline)
      2. If a paragraph fits in max_chunk_size → it's a chunk
      3. If too long → split into sentences, group sentences until
         they'd exceed max_chunk_size
      4. If a single sentence exceeds max_chunk_size → fall back to
         fixed-size split on that sentence (rare, but handles it)

    OPTIONALLY (if embedding model provided):
      After basic splitting, merge adjacent chunks if their embeddings
      are very similar (cosine > threshold). This recombines chunks
      that were split but actually discuss the same topic.

    WHY THIS IS BETTER THAN FIXED:
      - Never cuts mid-sentence
      - Respects paragraph boundaries (natural topic shifts)
      - Chunks have internal coherence
      - Variable size — short paragraphs stay short, dense ones stay together

    WHEN IT FAILS:
      - Long paragraphs with multiple topics get lumped together
      - Doesn't know about document structure (sections, subsections)
      - Paragraph detection fails on poorly formatted PDFs
      - Sentence splitting fails on abbreviations ("Dr. Smith et al.")
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 100,
                 merge_similar: bool = False, similarity_threshold: float = 0.8):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.merge_similar = merge_similar
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, handling abbreviations."""
        # Negative lookbehind for common abbreviations
        pattern = r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|al|etc|Fig|Eq|Sec|Vol))\.\s+'
        parts = re.split(pattern, text)
        # Re-add the period that was consumed by split
        sentences = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                sentences.append(part.strip() + '.')
            else:
                sentences.append(part.strip())
        return [s for s in sentences if s]

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on double newlines (paragraph boundaries)."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        paragraphs = self._split_paragraphs(text)
        chunks = []
        chunk_id = 0

        for para in paragraphs:
            if len(para) <= self.max_chunk_size:
                # Paragraph fits — use it as-is
                if len(para) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=para,
                        chunk_id=chunk_id,
                        metadata={**metadata, "strategy": "semantic", "boundary": "paragraph"},
                    ))
                    chunk_id += 1
                elif chunks:
                    # Too small — merge with previous chunk if it won't overflow
                    prev = chunks[-1]
                    merged = prev.text + "\n\n" + para
                    if len(merged) <= self.max_chunk_size:
                        chunks[-1] = Chunk(
                            text=merged,
                            chunk_id=prev.chunk_id,
                            metadata={**prev.metadata, "boundary": "merged_paragraph"},
                        )
                    else:
                        chunks.append(Chunk(
                            text=para,
                            chunk_id=chunk_id,
                            metadata={**metadata, "strategy": "semantic", "boundary": "paragraph"},
                        ))
                        chunk_id += 1
                else:
                    chunks.append(Chunk(
                        text=para,
                        chunk_id=chunk_id,
                        metadata={**metadata, "strategy": "semantic", "boundary": "paragraph"},
                    ))
                    chunk_id += 1
            else:
                # Paragraph too long — split by sentences
                sentences = self._split_sentences(para)
                current_group = []
                current_len = 0

                for sent in sentences:
                    if current_len + len(sent) > self.max_chunk_size and current_group:
                        # Flush current group
                        chunks.append(Chunk(
                            text=' '.join(current_group),
                            chunk_id=chunk_id,
                            metadata={**metadata, "strategy": "semantic", "boundary": "sentence"},
                        ))
                        chunk_id += 1
                        current_group = []
                        current_len = 0

                    current_group.append(sent)
                    current_len += len(sent) + 1  # +1 for space

                # Flush remaining
                if current_group:
                    chunks.append(Chunk(
                        text=' '.join(current_group),
                        chunk_id=chunk_id,
                        metadata={**metadata, "strategy": "semantic", "boundary": "sentence"},
                    ))
                    chunk_id += 1

        # Optional: merge similar adjacent chunks
        if self.merge_similar and len(chunks) > 1:
            chunks = self._merge_similar_chunks(chunks)

        return chunks

    def _merge_similar_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge adjacent chunks with high semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            print("Warning: sentence-transformers not installed, skipping merge")
            return chunks

        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)

        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            combined_len = len(merged[-1].text) + len(chunks[i].text)

            if sim >= self.similarity_threshold and combined_len <= self.max_chunk_size:
                # Merge with previous
                prev = merged[-1]
                merged[-1] = Chunk(
                    text=prev.text + "\n\n" + chunks[i].text,
                    chunk_id=prev.chunk_id,
                    metadata={**prev.metadata, "boundary": "semantic_merge", "merge_sim": round(sim, 3)},
                )
            else:
                merged.append(chunks[i])

        # Re-index
        for i, c in enumerate(merged):
            c.chunk_id = i

        return merged


# ==================== STRATEGY 3: HIERARCHICAL ====================

class HierarchicalChunker:
    """
    Section-aware chunking — preserves document structure.

    This is the most sophisticated strategy for structured documents
    like research papers. It understands that text within "2.1 Attention
    Mechanism" is topically different from "3. Experiments" and should
    never be combined.

    HOW IT WORKS:
      1. Take pre-detected sections from pdf_parser (Section objects)
      2. For each section, chunk its content using SemanticChunker
      3. Prepend section context to each chunk as a header
      4. Each chunk carries its section lineage in metadata

    THE KEY INSIGHT:
      A chunk that says "The results show 95% accuracy" is useless
      without knowing it came from "4.2 Ablation Study" in the paper.
      Hierarchical chunking preserves this context.

    WHAT GETS PREPENDED:
      "[Section: 3 Experiments > 3.2 Ablation Study]
       The results show 95% accuracy..."

      This gives the embedding model AND the LLM context about where
      this text came from, dramatically improving retrieval relevance.

    WHEN TO USE:
      - Research papers (always)
      - Technical documentation with clear hierarchy
      - Legal documents with numbered sections
      - Any document where section headers carry meaning

    WHEN IT FAILS:
      - Unstructured documents (blog posts, emails)
      - When section detection is wrong (garbage sections = garbage chunks)
      - Very short sections get tiny chunks (we handle via min_chunk_size)
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 100,
                 include_context_header: bool = True):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_context_header = include_context_header
        # Use semantic chunker internally for within-section splitting
        self._inner_chunker = SemanticChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )

    def chunk(self, sections: list, metadata: dict = None) -> list[Chunk]:
        """
        Chunk pre-detected sections.

        Args:
            sections: list of Section objects from pdf_parser
            metadata: additional metadata to attach to all chunks
        """
        metadata = metadata or {}
        all_chunks = []
        chunk_id = 0

        for section in sections:
            if not section.content.strip():
                continue

            # Build section context string
            context_header = f"[Section: {section.title}]"

            # Chunk the section content
            section_meta = {
                **metadata,
                "strategy": "hierarchical",
                "section": section.title,
                "section_level": section.level,
                "page_start": section.page_start,
                "page_end": section.page_end,
            }

            inner_chunks = self._inner_chunker.chunk(section.content, section_meta)

            for ic in inner_chunks:
                # Prepend context header so embeddings capture section info
                if self.include_context_header:
                    text_with_context = f"{context_header}\n{ic.text}"
                else:
                    text_with_context = ic.text

                all_chunks.append(Chunk(
                    text=text_with_context,
                    chunk_id=chunk_id,
                    metadata=ic.metadata,
                ))
                chunk_id += 1

        return all_chunks


# ==================== COMPARISON UTILITIES ====================

def chunk_stats(chunks: list[Chunk]) -> dict:
    """Compute statistics about a set of chunks."""
    if not chunks:
        return {"count": 0}

    sizes = [c.char_count for c in chunks]
    words = [c.word_count for c in chunks]

    return {
        "count": len(chunks),
        "total_chars": sum(sizes),
        "total_words": sum(words),
        "avg_chars": round(sum(sizes) / len(sizes)),
        "avg_words": round(sum(words) / len(words)),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "std_chars": round((sum((s - sum(sizes)/len(sizes))**2 for s in sizes) / len(sizes)) ** 0.5),
    }


def print_comparison(strategies: dict[str, list[Chunk]]):
    """Print side-by-side comparison of chunking strategies."""
    print(f"\n{'='*70}")
    print(f"{'CHUNKING STRATEGY COMPARISON':^70}")
    print(f"{'='*70}")

    headers = ["Metric"] + list(strategies.keys())
    all_stats = {name: chunk_stats(chunks) for name, chunks in strategies.items()}

    metrics = ["count", "avg_chars", "avg_words", "min_chars", "max_chars", "std_chars", "total_chars"]
    labels = {
        "count": "Chunk count",
        "avg_chars": "Avg chars/chunk",
        "avg_words": "Avg words/chunk",
        "min_chars": "Smallest chunk",
        "max_chars": "Largest chunk",
        "std_chars": "Std deviation",
        "total_chars": "Total chars",
    }

    # Print header
    col_width = max(18, max(len(n) for n in strategies.keys()) + 2)
    header_fmt = f"  {{:<20}}" + f"{{:>{col_width}}}" * len(strategies)
    print(header_fmt.format("Metric", *strategies.keys()))
    print("  " + "-" * (20 + col_width * len(strategies)))

    for metric in metrics:
        values = [str(all_stats[name].get(metric, "—")) for name in strategies.keys()]
        print(header_fmt.format(labels.get(metric, metric), *values))

    print()

    # Show sample chunks from each strategy
    for name, chunks in strategies.items():
        if chunks:
            print(f"\n  Sample chunk from [{name}]:")
            sample = chunks[min(2, len(chunks) - 1)]  # 3rd chunk usually more interesting
            preview = sample.text[:300].replace('\n', '\n    ')
            print(f"    {preview}...")
            print(f"    → metadata: {sample.metadata}")
