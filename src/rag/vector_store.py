"""
vector_store.py — FAISS-backed vector storage and retrieval
===========================================================

Why FAISS:
  FAISS (Facebook AI Similarity Search) is the industry standard for
  dense vector search. It's written in C++ with Python bindings, so
  it's fast — millions of vectors searched in milliseconds.

  For our use case (hundreds to thousands of chunks), a flat index
  is perfect. No approximation needed. When you scale to millions,
  you'd switch to IVF or HNSW indexes for speed.

What this module does:
  1. Build a FAISS index from chunk embeddings
  2. Search by query vector → top-k most similar chunks
  3. Save/load index + metadata to disk (persist across sessions)
  4. BM25 sparse index for keyword search (hybrid in Module 3)

Key concept — FAISS only stores vectors, not text:
  FAISS maps index positions to vectors. We maintain a parallel list
  of Chunk objects so we can map: query → vector search → index positions
  → original chunks with text and metadata.

Usage:
  from rag.vector_store import VectorStore
  store = VectorStore(embedder)
  store.add_chunks(chunks)
  results = store.search("What is attention?", top_k=5)
"""

import json
import pickle
import numpy as np
from pathlib import Path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from rag.chunkers import Chunk
from rag.embedder import Embedder


class VectorStore:
    """
    Dense vector store backed by FAISS.

    HOW IT WORKS:
      - Chunks get embedded into vectors (via Embedder)
      - Vectors go into a FAISS index (flat L2 or inner product)
      - Query comes in → embed query → search index → return top-k chunks
      - Metadata (section, page, source) travels with the chunks

    INDEX TYPES:
      - IndexFlatIP: exact inner product search (cosine sim if normalized)
        Best for < 100k vectors. No approximation, 100% recall.
      - IndexIVFFlat: approximate search with inverted file index
        Better for 100k-10M vectors. Faster but may miss some results.
      - IndexHNSW: graph-based approximate search
        Best for 1M+ vectors. Very fast, high recall.

      We use IndexFlatIP because for research papers (hundreds of chunks),
      exact search is fast enough and we don't want approximation errors.
    """

    def __init__(self, embedder: Embedder):
        if not HAS_FAISS:
            raise ImportError("pip install faiss-cpu")

        self.embedder = embedder
        self.dim = embedder.dim
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

        # Inner product index (= cosine similarity when vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dim)

    def add_chunks(self, chunks: list[Chunk]):
        """
        Embed chunks and add to the index.

        This is the main entry point. Give it chunks, it handles
        embedding and indexing.
        """
        if not chunks:
            return

        vectors = self.embedder.embed_chunks(chunks)

        # Add to FAISS index
        self.index.add(vectors)

        # Store chunks and vectors for later reference
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        print(f"Index now has {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for chunks most similar to query.

        Returns list of dicts:
          {chunk, score, rank}
        sorted by descending similarity.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = self.embedder.embed_query(query)
        query_vec = query_vec.reshape(1, -1)

        # FAISS search returns (distances, indices) arrays
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "rank": rank,
            })

        return results

    def search_with_vector(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search with a pre-computed query vector."""
        if self.index.ntotal == 0:
            return []

        query_vec = query_vec.reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx == -1:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "rank": rank,
            })
        return results

    # ==================== PERSISTENCE ====================

    def save(self, directory: str | Path):
        """
        Save index + chunks to disk.

        Creates:
          directory/index.faiss    — the FAISS index
          directory/chunks.pkl     — chunk objects with metadata
          directory/config.json    — model name, dim, count
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(directory / "index.faiss"))

        # Save chunks (pickle because they have complex metadata)
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        # Save config
        config = {
            "model_name": self.embedder.model_name,
            "dim": self.dim,
            "num_chunks": len(self.chunks),
        }
        with open(directory / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved index ({self.index.ntotal} vectors) to {directory}")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder = None) -> "VectorStore":
        """
        Load a saved index from disk.

        If embedder is None, creates one using the saved model name.
        """
        directory = Path(directory)

        # Load config
        with open(directory / "config.json") as f:
            config = json.load(f)

        # Create or validate embedder
        if embedder is None:
            embedder = Embedder(model_name=config["model_name"])
        elif embedder.dim != config["dim"]:
            raise ValueError(
                f"Embedder dim {embedder.dim} != saved dim {config['dim']}. "
                f"Use model: {config['model_name']}"
            )

        store = cls(embedder)

        # Load FAISS index
        store.index = faiss.read_index(str(directory / "index.faiss"))

        # Load chunks
        with open(directory / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)

        print(f"Loaded index ({store.index.ntotal} vectors, {len(store.chunks)} chunks) from {directory}")
        return store

    # ==================== DIAGNOSTICS ====================

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            "num_vectors": self.index.ntotal,
            "num_chunks": len(self.chunks),
            "dim": self.dim,
            "model": self.embedder.model_name,
        }


def print_results(results: list[dict], query: str, max_preview: int = 150):
    """Pretty-print search results."""
    print(f"\n{'─'*70}")
    print(f"Query: {query!r}")
    print(f"{'─'*70}")

    for r in results:
        chunk = r["chunk"]
        score = r["score"]
        section = chunk.metadata.get("section", "—")
        strategy = chunk.metadata.get("strategy", "—")
        preview = chunk.text[:max_preview].replace('\n', ' ')

        print(f"\n  #{r['rank']}  score={score:.4f}  section={section!r}  strategy={strategy}")
        print(f"      {preview}...")