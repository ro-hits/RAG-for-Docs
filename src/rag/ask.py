"""
ask.py — Full RAG pipeline: question in, cited answer out
==========================================================

Usage:
  uv run rag-ask papers/attention.pdf "What is multi-head attention?"

  # Switch models with --model
  uv run rag-ask papers/attention.pdf "query" --model claude
  uv run rag-ask papers/attention.pdf "query" --model glm5
  uv run rag-ask papers/attention.pdf "query" --model deepseek
  uv run rag-ask papers/attention.pdf "query" --model llama3

  # Interactive mode (keep asking without re-indexing)
  uv run rag-ask papers/attention.pdf --model glm5

  # List available presets
  uv run rag-ask --list-models
"""

import sys
from pathlib import Path

from rag.pdf_parser import parse_pdf
from rag.chunkers import HierarchicalChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.sparse import BM25Index
from rag.hybrid import HybridRetriever, CrossEncoderReranker
from rag.generator import RAGGenerator, print_response, list_presets


def parse_args(argv: list[str]) -> dict:
    """
    Simple arg parser (no argparse dependency for clarity).

    Parses:
      rag-ask <paper.pdf> [query] [--model preset] [--list-models]
    """
    args = {
        "filepath": None,
        "query": None,
        "model": "claude",  # default preset
        "list_models": False,
    }

    positional = []
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            args["model"] = argv[i + 1]
            i += 2
        elif argv[i] == "--list-models":
            args["list_models"] = True
            i += 1
        elif argv[i].startswith("--"):
            i += 1  # skip unknown flags
        else:
            positional.append(argv[i])
            i += 1

    if len(positional) >= 1:
        args["filepath"] = positional[0]
    if len(positional) >= 2:
        args["query"] = positional[1]

    return args


def build_pipeline(filepath: Path, model_preset: str) -> tuple[HybridRetriever, RAGGenerator]:
    """Build the full retrieval + generation pipeline."""

    index_dir = Path("index")

    # Step 1: Load or build index
    if (index_dir / "index.faiss").exists():
        print("\n[1/4] Loading saved index...")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        dense_store = VectorStore.load(index_dir, embedder)
    else:
        print("\n[1/4] Building index from scratch...")
        doc = parse_pdf(filepath)
        print(f"  {doc.total_pages} pages, {len(doc.sections)} sections")

        chunker = HierarchicalChunker(max_chunk_size=1500, include_context_header=True)
        chunks = chunker.chunk(doc.sections, {"source": doc.filename})
        print(f"  {len(chunks)} chunks")

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        dense_store = VectorStore(embedder)
        dense_store.add_chunks(chunks)
        dense_store.save(index_dir)

    # Step 2: BM25
    print("\n[2/4] Building BM25 index...")
    bm25 = BM25Index()
    bm25.add_chunks(dense_store.chunks)

    # Step 3: Reranker
    print("\n[3/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    retriever = HybridRetriever(
        dense_store=dense_store,
        bm25_index=bm25,
        reranker=reranker,
        dense_top_k=20,
        sparse_top_k=20,
        fusion_top_n=15,
        final_top_k=5,
    )

    # Step 4: Generator (uses preset for model selection)
    print(f"\n[4/4] Initializing generator (preset: {model_preset})...")
    generator = RAGGenerator(preset=model_preset)

    return retriever, generator


def ask(query: str, retriever: HybridRetriever, generator: RAGGenerator):
    """Run the full pipeline for a single question."""
    print(f"\n{'─'*70}")
    print(f"  Retrieving...")
    results = retriever.search(query, top_k=5)

    print(f"  Top {len(results)} chunks:")
    for r in results:
        section = r["chunk"].metadata.get("section", "?")[:30]
        found = r.get("found_by", [])
        print(f"    #{r['rank']} [{section}] found_by={found}")

    print(f"\n  Generating answer...")
    response = generator.generate(query, results)
    print_response(response)
    return response


def main():
    """Entry point for `uv run rag-ask`"""
    args = parse_args(sys.argv[1:])

    # List models and exit
    if args["list_models"]:
        print(list_presets())
        print("\nUsage: uv run rag-ask papers/paper.pdf \"query\" --model <preset>")
        sys.exit(0)

    if not args["filepath"]:
        print("Usage:")
        print('  uv run rag-ask papers/attention.pdf "What is attention?"')
        print('  uv run rag-ask papers/attention.pdf "query" --model glm5')
        print('  uv run rag-ask papers/attention.pdf --model deepseek')
        print('  uv run rag-ask --list-models')
        sys.exit(1)

    filepath = Path(args["filepath"])
    model_preset = args["model"]

    print(f"\n{'='*70}")
    print(f"  RAG FROM SCRATCH — Full Pipeline")
    print(f"  Paper: {filepath.name}")
    print(f"  Model: {model_preset}")
    print(f"{'='*70}")

    retriever, generator = build_pipeline(filepath, model_preset)

    # Single query mode
    if args["query"]:
        ask(args["query"], retriever, generator)
        return

    # Interactive mode
    print(f"\n{'='*70}")
    print(f"  Ready! Ask questions about the paper. (model: {model_preset})")
    print(f"  Type 'quit' to stop, 'switch <preset>' to change model.")
    print(f"{'='*70}")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Allow switching models mid-session
        if user_input.lower().startswith("switch "):
            new_preset = user_input.split(None, 1)[1].strip()
            try:
                generator = RAGGenerator(preset=new_preset)
                print(f"  Switched to {new_preset}")
            except ValueError as e:
                print(f"  Error: {e}")
            continue

        if user_input.lower() == "models":
            print(list_presets())
            continue

        ask(user_input, retriever, generator)