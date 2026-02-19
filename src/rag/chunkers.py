"""
chunkers.py — Three Chunking Strategies for RAG
================================================

Strategies:
  1. FixedSizeChunker    — split by character count with overlap
  2. SemanticChunker     — split at paragraph/sentence boundaries
  3. HierarchicalChunker — section-aware, preserves document structure

There is no "best" chunking strategy. It depends on your documents
and your queries. That's why we build three and compare.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text with metadata for retrieval."""
    text: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)

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

    HOW IT WORKS:
      Text gets sliced every N characters. Overlap means the last M
      characters of chunk N appear as the first M of chunk N+1.

    WHY OVERLAP:
      Without it, sentences at boundaries get split and neither chunk
      has the full thought. Overlap is a band-aid, not a fix.

    WHEN IT FAILS:
      - Cuts sentences mid-word
      - Splits tables, equations, code blocks
      - Mixes content from different sections in one chunk
      - Wastes tokens on repeated overlap content

    WHEN TO USE:
      - Quick baseline to compare against
      - Documents with no structure (flat text)
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

    HOW IT WORKS:
      1. Split text into paragraphs (double newline)
      2. If a paragraph fits in max_chunk_size → it's a chunk
      3. If too long → split into sentences, group until limit
      4. Tiny paragraphs get merged with neighbors

    WHY THIS IS BETTER THAN FIXED:
      - Never cuts mid-sentence
      - Respects paragraph boundaries (natural topic shifts)
      - Chunks have internal coherence

    WHEN IT FAILS:
      - Long paragraphs with multiple topics get lumped together
      - Doesn't know about document structure (sections)
      - Paragraph detection fails on poorly formatted PDFs
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 100,
                 merge_similar: bool = False, similarity_threshold: float = 0.8):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.merge_similar = merge_similar
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Split into sentences, handling abbreviations."""
        abbreviations = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr",
                         "vs", "al", "etc", "fig", "eq", "sec", "vol"}

        # Split on period + whitespace
        parts = re.split(r'\.\s+', text)
        sentences = []
        buffer = ""

        for i, part in enumerate(parts):
            buffer = f"{buffer}. {part}" if buffer else part

            # Last part — flush
            if i == len(parts) - 1:
                sentences.append(buffer.strip())
                break

            # Check if this part ends with an abbreviation
            last_word = part.strip().rsplit(None, 1)[-1].lower().rstrip('.') if part.strip() else ""
            if last_word in abbreviations:
                continue  # don't split here, keep buffering

            sentences.append(buffer.strip() + '.')
            buffer = ""

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
                if len(para) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=para,
                        chunk_id=chunk_id,
                        metadata={**metadata, "strategy": "semantic", "boundary": "paragraph"},
                    ))
                    chunk_id += 1
                elif chunks:
                    # Too small — merge with previous if it fits
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
                            text=para, chunk_id=chunk_id,
                            metadata={**metadata, "strategy": "semantic", "boundary": "paragraph"},
                        ))
                        chunk_id += 1
                else:
                    chunks.append(Chunk(
                        text=para, chunk_id=chunk_id,
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
                        chunks.append(Chunk(
                            text=' '.join(current_group),
                            chunk_id=chunk_id,
                            metadata={**metadata, "strategy": "semantic", "boundary": "sentence"},
                        ))
                        chunk_id += 1
                        current_group = []
                        current_len = 0

                    current_group.append(sent)
                    current_len += len(sent) + 1

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
                prev = merged[-1]
                merged[-1] = Chunk(
                    text=prev.text + "\n\n" + chunks[i].text,
                    chunk_id=prev.chunk_id,
                    metadata={**prev.metadata, "boundary": "semantic_merge", "merge_sim": round(sim, 3)},
                )
            else:
                merged.append(chunks[i])

        for i, c in enumerate(merged):
            c.chunk_id = i

        return merged


# ==================== STRATEGY 3: HIERARCHICAL ====================

class HierarchicalChunker:
    """
    Section-aware chunking — preserves document structure.

    THE KEY INSIGHT:
      A chunk that says "The results show 95% accuracy" is useless
      without knowing it came from "4.2 Ablation Study".
      Hierarchical chunking preserves this context by prepending
      section headers.

    HOW IT WORKS:
      1. Take pre-detected sections from pdf_parser
      2. Chunk each section's content using SemanticChunker
      3. Prepend [Section: ...] header to each chunk
      4. Section lineage travels in metadata

    WHEN TO USE:
      - Research papers (always)
      - Technical docs with clear hierarchy
      - Legal documents with numbered sections
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 100,
                 include_context_header: bool = True):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_context_header = include_context_header
        self._inner_chunker = SemanticChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )

    def chunk(self, sections: list, metadata: dict = None) -> list[Chunk]:
        """
        Chunk pre-detected sections.

        Args:
            sections: list of Section objects from pdf_parser
            metadata: additional metadata for all chunks
        """
        metadata = metadata or {}
        all_chunks = []
        chunk_id = 0

        for section in sections:
            if not section.content.strip():
                continue

            context_header = f"[Section: {section.title}]"

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

    col_width = max(18, max(len(n) for n in strategies.keys()) + 2)
    header_fmt = f"  {{:<20}}" + f"{{:>{col_width}}}" * len(strategies)
    print(header_fmt.format("Metric", *strategies.keys()))
    print("  " + "-" * (20 + col_width * len(strategies)))

    for metric in metrics:
        values = [str(all_stats[name].get(metric, "—")) for name in strategies.keys()]
        print(header_fmt.format(labels.get(metric, metric), *values))

    print()

    for name, chunks in strategies.items():
        if chunks:
            print(f"\n  Sample chunk from [{name}]:")
            sample = chunks[min(2, len(chunks) - 1)]
            preview = sample.text[:300].replace('\n', '\n    ')
            print(f"    {preview}...")
            print(f"    → metadata: {sample.metadata}")