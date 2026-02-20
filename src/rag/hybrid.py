"""
hybrid.py — Hybrid Retrieval + Reranking
=========================================

Why hybrid:
  You just saw it in Module 2 — dense and BM25 find DIFFERENT chunks
  for the same query. 2-3 overlap out of 5. That means each method
  alone is missing relevant content the other finds.

  Hybrid search combines both result lists into one, getting the
  best of semantic understanding (dense) and keyword precision (BM25).

How — Reciprocal Rank Fusion (RRF):
  The simplest and most robust way to combine two ranked lists.
  No score normalization needed (BM25 scores range 0-20+, dense
  scores range 0-1 — you can't just average them).

  RRF score = sum of 1/(k + rank) for each method that found the chunk

  k=60 is the standard constant from the original paper (Cormack et al., 2009).
  It dampens the difference between rank 1 and rank 2 while still
  favoring higher-ranked results.

  Example:
    Chunk A: dense rank 1, BM25 rank 3
    RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

    Chunk B: dense rank 2, BM25 rank 1
    RRF = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252

    Chunk B wins — found high in both lists.

    Chunk C: dense rank 5, BM25 not found
    RRF = 1/(60+5) + 0 = 0.01538

    Chunk C ranks lower — only one method found it.

Why reranking:
  Retrieval (dense, BM25, hybrid) is FAST but APPROXIMATE. It uses
  dot products or term matching — cheap operations that scale.

  A cross-encoder is SLOW but PRECISE. It takes (query, chunk) as
  a single input and runs full transformer attention across both.
  This means it can understand relationships between query words
  and chunk words that a dot product can't capture.

  The pipeline:
    1. Hybrid retrieval → top 20 candidates (fast, broad)
    2. Cross-encoder reranking → re-score those 20 → top 5 (slow, precise)

  This is the single biggest quality improvement in RAG and most
  people skip it because tutorials don't cover it.

Usage:
  uv run rag-search papers/attention.pdf "What is multi-head attention?"
"""

import sys
from pathlib import Path

from rag.chunkers import Chunk
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.sparse import BM25Index


# ==================== RECIPROCAL RANK FUSION ====================

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 20,
) -> list[dict]:
    """
    Combine multiple ranked result lists using RRF.

    Args:
        result_lists: list of search result lists, each containing
                      dicts with "chunk" and "rank" keys
        k: RRF constant (default 60, from the original paper)
        top_n: how many fused results to return

    Returns:
        Merged list sorted by RRF score, deduplicated by chunk_id
    """
    # Accumulate RRF scores by chunk_id
    chunk_scores: dict[int, float] = {}
    chunk_map: dict[int, Chunk] = {}
    # Track which methods found each chunk
    chunk_sources: dict[int, list[str]] = {}

    for list_idx, results in enumerate(result_lists):
        method = f"method_{list_idx}"
        for r in results:
            cid = r["chunk"].chunk_id
            rank = r["rank"]
            rrf_score = 1.0 / (k + rank)

            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_score
            chunk_map[cid] = r["chunk"]

            if cid not in chunk_sources:
                chunk_sources[cid] = []
            chunk_sources[cid].append(f"{method}@{rank}")

    # Sort by fused score
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (cid, score) in enumerate(sorted_chunks[:top_n], 1):
        results.append({
            "chunk": chunk_map[cid],
            "score": score,
            "rank": rank,
            "found_by": chunk_sources[cid],
        })

    return results


# ==================== CROSS-ENCODER RERANKING ====================

class CrossEncoderReranker:
    """
    Re-score (query, chunk) pairs using a cross-encoder model.

    WHY THIS IS DIFFERENT FROM EMBEDDINGS:
      Embedding (bi-encoder): encode query and chunk SEPARATELY,
        then compare with dot product. Fast but limited — can't see
        word-level interactions between query and chunk.

      Cross-encoder: encode query AND chunk TOGETHER as one input.
        Full attention across both. Sees every word interaction.
        Much more accurate, but O(n) forward passes instead of O(1).

    That's why we use it as a reranker on top-20, not as the primary
    retrieval method — 20 forward passes is fine, 10,000 is not.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      Trained on MS MARCO passage ranking. Small, fast, effective.
      Outputs a relevance score (not calibrated probability, just ranking).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        print(f"Loading reranker: {model_name}...", end=" ", flush=True)
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print("done")

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        """
        Re-score and re-rank results using cross-encoder.

        Args:
            query: the search query
            results: list of dicts with "chunk" key
            top_k: how many to return after reranking

        Returns:
            Re-ranked list with updated scores and ranks
        """
        if not results:
            return []

        # Build (query, chunk_text) pairs
        pairs = [(query, r["chunk"].text) for r in results]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)
            r["original_rank"] = r["rank"]

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        # Assign new ranks
        for i, r in enumerate(reranked[:top_k], 1):
            r["rank"] = i

        return reranked[:top_k]


# ==================== FULL PIPELINE ====================

class HybridRetriever:
    """
    Full retrieval pipeline: dense + BM25 → RRF fusion → cross-encoder reranking.

    This is the production-grade retrieval setup. Three stages:
      1. RECALL: Cast a wide net with both dense and sparse search
      2. FUSE: Combine results with RRF
      3. PRECISION: Rerank top candidates with cross-encoder
    """

    def __init__(
        self,
        dense_store: VectorStore,
        bm25_index: BM25Index,
        reranker: CrossEncoderReranker | None = None,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        fusion_top_n: int = 20,
        final_top_k: int = 5,
    ):
        self.dense_store = dense_store
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_top_n = fusion_top_n
        self.final_top_k = final_top_k

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Full pipeline: dense + sparse → RRF → rerank.

        Returns list of {chunk, score, rank, found_by, rerank_score}
        """
        top_k = top_k or self.final_top_k

        # Stage 1: Recall — retrieve from both methods
        dense_results = self.dense_store.search(query, top_k=self.dense_top_k)
        sparse_results = self.bm25_index.search(query, top_k=self.sparse_top_k)

        # Stage 2: Fuse with RRF
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_n=self.fusion_top_n,
        )

        # Stage 3: Rerank (if reranker available)
        if self.reranker:
            final = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            final = fused[:top_k]

        return final


# ==================== CLI DEMO ====================

def print_pipeline_results(query: str, results: list[dict]):
    """Pretty-print the full pipeline results."""
    print(f"\n{'─'*70}")
    print(f"  Query: {query!r}")
    print(f"{'─'*70}")

    for r in results:
        chunk = r["chunk"]
        section = chunk.metadata.get("section", "—")
        found_by = r.get("found_by", [])
        rerank_score = r.get("rerank_score")
        original_rank = r.get("original_rank")
        preview = chunk.text[:120].replace('\n', ' ')

        rank_info = f"#{r['rank']}"
        if original_rank and original_rank != r["rank"]:
            rank_info += f" (was #{original_rank})"

        score_info = f"rrf={r['score']:.4f}"
        if rerank_score is not None:
            score_info += f"  rerank={rerank_score:.3f}"

        print(f"\n  {rank_info}  {score_info}")
        print(f"  section={section!r}  found_by={found_by}")
        print(f"  {preview}...")


def main():
    """Entry point for `uv run rag-search <paper.pdf> <query>`"""
    if len(sys.argv) < 3:
        print('Usage: uv run rag-search papers/attention.pdf "What is attention?"')
        print("\nRun rag-index first to build the index, or this will build it fresh.")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    query = sys.argv[2]

    print(f"\n{'='*70}")
    print(f"  MODULE 3: Hybrid Retrieval + Reranking")
    print(f"  Paper: {filepath.name}")
    print(f"  Query: {query!r}")
    print(f"{'='*70}")

    # Check for saved index
    index_dir = Path("index")
    if (index_dir / "index.faiss").exists():
        print("\n[1/4] Loading saved index...")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        dense_store = VectorStore.load(index_dir, embedder)
    else:
        print("\n[1/4] No saved index — building from scratch...")
        from rag.pdf_parser import parse_pdf
        from rag.chunkers import HierarchicalChunker

        doc = parse_pdf(filepath)
        chunker = HierarchicalChunker(max_chunk_size=1500, include_context_header=True)
        chunks = chunker.chunk(doc.sections, {"source": doc.filename})

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        dense_store = VectorStore(embedder)
        dense_store.add_chunks(chunks)
        dense_store.save(index_dir)

    # Build BM25 (always fresh — it's fast)
    print("\n[2/4] Building BM25 index...")
    bm25 = BM25Index()
    bm25.add_chunks(dense_store.chunks)

    # Load reranker
    print("\n[3/4] Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    # Build hybrid retriever
    retriever = HybridRetriever(
        dense_store=dense_store,
        bm25_index=bm25,
        reranker=reranker,
        dense_top_k=20,
        sparse_top_k=20,
        fusion_top_n=15,
        final_top_k=5,
    )

    # Search
    print("\n[4/4] Searching...")
    results = retriever.search(query)
    print_pipeline_results(query, results)

    # Show the pipeline effect
    print(f"\n{'='*70}")
    print("WHAT HAPPENED")
    print(f"{'='*70}")

    # Also run dense-only and BM25-only for comparison
    dense_only = dense_store.search(query, top_k=5)
    sparse_only = bm25.search(query, top_k=5)

    dense_sections = [r["chunk"].metadata.get("section", "?")[:25] for r in dense_only]
    sparse_sections = [r["chunk"].metadata.get("section", "?")[:25] for r in sparse_only]
    hybrid_sections = [r["chunk"].metadata.get("section", "?")[:25] for r in results]

    print(f"\n  Dense only:  {dense_sections}")
    print(f"  BM25 only:   {sparse_sections}")
    print(f"  Hybrid+RR:   {hybrid_sections}")

    reranked_count = sum(1 for r in results if r.get("original_rank") and r["original_rank"] != r["rank"])
    print(f"\n  Reranker moved {reranked_count}/{len(results)} chunks to different positions.")
    print(f"  This is the precision step — reranking catches cases where RRF")
    print(f"  ranked something high just because both methods found it, even")
    print(f"  though it wasn't actually the best answer.")