"""
embedder.py — Convert text chunks into vectors
===============================================

Why this matters for RAG:
  Retrieval works by finding chunks whose vectors are "close" to the
  query vector in high-dimensional space. The embedding model decides
  what "close" means — it defines your system's notion of similarity.

  Two chunks about "attention mechanisms" should have similar vectors.
  A chunk about "training hyperparameters" should be far away. If your
  embedding model doesn't capture this, retrieval fails silently —
  you get plausible-looking but wrong context fed to the LLM.

Key concepts:
  - Dense embeddings: every dimension has a value (384-1536 floats)
  - Cosine similarity: angle between vectors (not magnitude)
  - Normalization: L2-normalize so dot product = cosine similarity
  - Batching: encode many chunks at once (GPU-friendly)

Model choice matters:
  - all-MiniLM-L6-v2: fast, 384 dims, good baseline
  - all-mpnet-base-v2: slower, 768 dims, better quality
  - For research papers, mpnet is worth the extra time

Usage:
  from rag.embedder import Embedder
  emb = Embedder()
  vectors = emb.embed_chunks(chunks)
  query_vec = emb.embed_query("What is multi-head attention?")
"""

import time
import numpy as np
from pathlib import Path

from rag.chunkers import Chunk


class Embedder:
    """
    Encode text into dense vectors using sentence-transformers.

    WHY sentence-transformers:
      Regular transformers (BERT, etc.) give you token-level embeddings.
      You'd need to pool them yourself (CLS token? mean pooling?).
      Sentence-transformers are fine-tuned specifically for sentence-level
      similarity — they already know how to compress a paragraph into
      a single meaningful vector.

    IMPORTANT DETAIL — query vs document encoding:
      Some models encode queries differently from documents. For our
      models (MiniLM, mpnet) this doesn't matter — same encoder for both.
      But if you switch to asymmetric models (like E5, BGE), you'd need
      to prefix queries with "query: " and documents with "passage: ".
      We handle this with the query_prefix/doc_prefix params.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size
        self.normalize = normalize

        print(f"Loading embedding model: {model_name}...", end=" ", flush=True)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"done ({self.dim} dims)")

    def embed_texts(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Encode a list of texts into vectors.

        Args:
            texts: list of strings to encode
            is_query: if True, prepend query_prefix (for asymmetric models)

        Returns:
            np.ndarray of shape (len(texts), dim), dtype float32
        """
        prefix = self.query_prefix if is_query else self.doc_prefix
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 50,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Encode chunks, using their .text field."""
        texts = [c.text for c in chunks]
        start = time.time()
        vectors = self.embed_texts(texts, is_query=False)
        elapsed = time.time() - start
        print(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s "
              f"({len(chunks)/elapsed:.0f} chunks/s, {self.dim} dims)")
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query. Returns shape (dim,)."""
        return self.embed_texts([query], is_query=True)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def pairwise_similarities(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity of query against all chunk vectors. Returns shape (n_chunks,)."""
    # If normalized, dot product = cosine similarity
    return chunk_vecs @ query_vec