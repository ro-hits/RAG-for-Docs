"""
sparse.py — BM25 keyword-based retrieval
=========================================

Why you need this alongside dense search:
  Dense embeddings are great at semantic similarity — "car" matches
  "automobile". But they're TERRIBLE at exact keyword matching.

  Ask: "What does equation 3.2 say?"
  Dense search: returns chunks about equations in general
  BM25 search: returns the chunk containing "equation 3.2" exactly

  Ask: "What is the BLEU score on WMT 2014?"
  Dense search: returns chunks about translation quality metrics
  BM25 search: returns the chunk with "BLEU" and "WMT 2014" literally

  For technical papers with specific terminology, model names, metrics,
  and numbered references — BM25 is essential. In Module 3 we'll
  combine both (hybrid search) to get the best of both worlds.

How BM25 works (simplified):
  1. Tokenize all chunks into words
  2. Build inverse document frequency (IDF) — rare words score higher
  3. For a query, score each chunk by:
     - How many query terms appear in the chunk (term frequency)
     - How rare those terms are across all chunks (IDF)
     - Normalized by chunk length (so long chunks don't dominate)

  It's basically "smart keyword matching" — invented in the 1990s
  and still competitive with neural methods on keyword-heavy queries.

Usage:
  from rag.sparse import BM25Index
  bm25 = BM25Index()
  bm25.add_chunks(chunks)
  results = bm25.search("BLEU score WMT 2014", top_k=5)
"""

import re
import math
from collections import Counter

from rag.chunkers import Chunk


# Simple tokenizer — good enough for English research papers
def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove short tokens."""
    tokens = re.findall(r'[a-z0-9]+(?:\.[0-9]+)*', text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """
    BM25 sparse retrieval index.

    Parameters k1 and b control the scoring:
      - k1 (default 1.5): term frequency saturation
        Higher = more weight to repeated terms
        Lower = diminishing returns for repeated terms
      - b (default 0.75): length normalization
        b=1.0 → fully normalize by doc length
        b=0.0 → ignore doc length entirely
        0.75 is the standard default and works well for most cases

    These defaults are from the original BM25 paper (Robertson et al., 1995)
    and rarely need tuning.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.tokenized: list[list[str]] = []
        self.doc_freqs: Counter = Counter()  # term → num docs containing it
        self.avg_dl: float = 0.0             # average document length
        self.n_docs: int = 0

    def add_chunks(self, chunks: list[Chunk]):
        """Tokenize chunks and build IDF statistics."""
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            self.tokenized.append(tokens)
            self.chunks.append(chunk)

            # Count document frequency (unique terms per doc)
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.n_docs = len(self.chunks)
        self.avg_dl = sum(len(t) for t in self.tokenized) / self.n_docs if self.n_docs else 0

        print(f"BM25 index: {self.n_docs} docs, {len(self.doc_freqs)} unique terms, "
              f"avg {self.avg_dl:.0f} tokens/doc")

    def _idf(self, term: str) -> float:
        """Inverse document frequency for a term."""
        df = self.doc_freqs.get(term, 0)
        # Smooth IDF to avoid zero/negative values
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        """BM25 score for a single document against query tokens."""
        doc_tokens = self.tokenized[doc_idx]
        doc_len = len(doc_tokens)
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search chunks by BM25 keyword relevance.

        Returns list of {chunk, score, rank} sorted by descending score.
        """
        if not self.chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Score all documents
        scores = [(i, self._score_doc(query_tokens, i)) for i in range(self.n_docs)]

        # Sort by score descending, take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        results = []
        for rank, (idx, score) in enumerate(top, 1):
            if score <= 0:
                break
            results.append({
                "chunk": self.chunks[idx],
                "score": score,
                "rank": rank,
            })

        return results