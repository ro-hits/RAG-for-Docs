"""
evaluate.py — RAG Evaluation Framework
========================================

The question nobody asks: "Is my RAG actually good?"

Most people build a RAG, try a few queries, see answers that look
reasonable, and ship it. That's like deploying a model without a
test set. You have NO idea if changes improve things or break them.

This module gives you numbers. Two kinds:

1. RETRIEVAL METRICS — Did the right chunks get retrieved?
   - Recall@k: what fraction of expected sections appeared in top-k?
   - MRR (Mean Reciprocal Rank): how high did the first relevant chunk rank?
   - Precision@k: what fraction of top-k chunks were relevant?

2. ANSWER QUALITY METRICS — Did the LLM answer correctly?
   - Citation accuracy: did it cite chunks, and were they valid?
   - Faithfulness: are all claims grounded in context?
   - No-answer detection: did it correctly refuse unanswerable questions?

Why both matter:
  You can have perfect retrieval and bad answers (LLM ignores context).
  You can have bad retrieval and good answers (LLM gets lucky or hallucinates).
  You need to measure BOTH to know where your pipeline is failing.

Usage:
  uv run rag-eval papers/attention.pdf --model claude
  uv run rag-eval papers/attention.pdf --model claude --model glm5   # compare
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

from rag.eval_data import TestCase, TestSet, load_test_set


# ==================== RETRIEVAL METRICS ====================

@dataclass
class RetrievalScore:
    """Retrieval metrics for a single query."""
    question: str
    expected_sections: list[str]
    retrieved_sections: list[str]
    recall_at_k: float          # fraction of expected found in top-k
    mrr: float                  # 1/rank of first relevant result (0 if none)
    precision_at_k: float       # fraction of top-k that were relevant
    k: int

    def __repr__(self):
        return (f"Recall@{self.k}={self.recall_at_k:.2f} "
                f"MRR={self.mrr:.2f} "
                f"Precision@{self.k}={self.precision_at_k:.2f}")


def _section_match(expected: str, retrieved: str) -> bool:
    """
    Fuzzy match between expected section name and retrieved section.

    Expected might be truncated ("6.1 Machine Translat") or partial.
    Retrieved comes from chunk metadata.
    """
    exp = expected.lower().strip()
    ret = retrieved.lower().strip()
    # Either one is a prefix of the other
    return exp in ret or ret in exp


def score_retrieval(test_case: TestCase, results: list[dict], k: int = 5) -> RetrievalScore:
    """
    Score retrieval quality for a single test case.

    Args:
        test_case: has expected_sections
        results: from hybrid retriever, each with chunk.metadata["section"]
        k: evaluate top-k results
    """
    top_k = results[:k]
    retrieved_sections = [
        r["chunk"].metadata.get("section", "") for r in top_k
    ]
    expected = test_case.expected_sections

    if not expected:
        # Unanswerable question — no sections expected
        return RetrievalScore(
            question=test_case.question,
            expected_sections=expected,
            retrieved_sections=retrieved_sections,
            recall_at_k=1.0,  # vacuously true
            mrr=0.0,
            precision_at_k=0.0,
            k=k,
        )

    # Recall@k: how many expected sections appear in retrieved?
    found = 0
    for exp in expected:
        for ret in retrieved_sections:
            if _section_match(exp, ret):
                found += 1
                break
    recall = found / len(expected)

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for rank, ret in enumerate(retrieved_sections, 1):
        for exp in expected:
            if _section_match(exp, ret):
                mrr = 1.0 / rank
                break
        if mrr > 0:
            break

    # Precision@k: how many retrieved are relevant?
    relevant_count = 0
    for ret in retrieved_sections:
        for exp in expected:
            if _section_match(exp, ret):
                relevant_count += 1
                break
    precision = relevant_count / len(top_k) if top_k else 0.0

    return RetrievalScore(
        question=test_case.question,
        expected_sections=expected,
        retrieved_sections=retrieved_sections,
        recall_at_k=recall,
        mrr=mrr,
        precision_at_k=precision,
        k=k,
    )


# ==================== ANSWER QUALITY METRICS ====================

@dataclass
class AnswerScore:
    """Answer quality metrics for a single query."""
    question: str
    answer_type: str
    has_citations: bool         # did the answer contain [N] markers?
    citation_count: int
    faithful: bool              # all claims grounded in context?
    no_answer_correct: bool     # for unanswerable: did it refuse correctly?
    answer_length: int          # chars in answer

    def __repr__(self):
        return (f"citations={self.citation_count} "
                f"faithful={self.faithful} "
                f"no_answer_correct={self.no_answer_correct}")


def score_answer(test_case: TestCase, response) -> AnswerScore:
    """
    Score answer quality for a single test case.

    Args:
        test_case: has answer_type, expected_answer
        response: RAGResponse from generator
    """
    is_unanswerable = test_case.answer_type == "unanswerable"

    # For unanswerable: did the model correctly refuse?
    if is_unanswerable:
        no_answer_correct = response.no_answer
    else:
        no_answer_correct = not response.no_answer  # should NOT refuse

    return AnswerScore(
        question=test_case.question,
        answer_type=test_case.answer_type,
        has_citations=len(response.citations) > 0,
        citation_count=len(response.citations),
        faithful=response.faithful,
        no_answer_correct=no_answer_correct,
        answer_length=len(response.answer),
    )


# ==================== AGGREGATE RESULTS ====================

@dataclass
class EvalResult:
    """Full evaluation results for a model on a test set."""
    model_name: str
    test_set_name: str
    retrieval_scores: list[RetrievalScore] = field(default_factory=list)
    answer_scores: list[AnswerScore] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def avg_recall(self) -> float:
        scores = [s.recall_at_k for s in self.retrieval_scores]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_mrr(self) -> float:
        scores = [s.mrr for s in self.retrieval_scores]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_precision(self) -> float:
        scores = [s.precision_at_k for s in self.retrieval_scores]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def citation_rate(self) -> float:
        """Fraction of answers that contained at least one citation."""
        answerable = [s for s in self.answer_scores if s.answer_type != "unanswerable"]
        if not answerable:
            return 0.0
        return sum(1 for s in answerable if s.has_citations) / len(answerable)

    @property
    def faithfulness_rate(self) -> float:
        answerable = [s for s in self.answer_scores if s.answer_type != "unanswerable"]
        if not answerable:
            return 0.0
        return sum(1 for s in answerable if s.faithful) / len(answerable)

    @property
    def no_answer_accuracy(self) -> float:
        """How often did the model correctly handle unanswerable questions?"""
        all_scores = self.answer_scores
        if not all_scores:
            return 0.0
        return sum(1 for s in all_scores if s.no_answer_correct) / len(all_scores)

    def summary_dict(self) -> dict:
        return {
            "model": self.model_name,
            "test_cases": len(self.retrieval_scores),
            "avg_recall@5": round(self.avg_recall, 3),
            "avg_mrr": round(self.avg_mrr, 3),
            "avg_precision@5": round(self.avg_precision, 3),
            "citation_rate": round(self.citation_rate, 3),
            "faithfulness": round(self.faithfulness_rate, 3),
            "no_answer_accuracy": round(self.no_answer_accuracy, 3),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


# ==================== EVAL RUNNER ====================

def run_eval(
    test_set: TestSet,
    retriever,
    generator,
    model_name: str = "",
    k: int = 5,
    verbose: bool = True,
) -> EvalResult:
    """
    Run full evaluation: retrieval + answer quality for each test case.

    This is the core loop:
      For each question in test set:
        1. Retrieve top-k chunks
        2. Score retrieval (did we get the right sections?)
        3. Generate answer
        4. Score answer (citations, faithfulness, no-answer detection)
    """
    result = EvalResult(
        model_name=model_name or generator.model,
        test_set_name=test_set.name,
    )

    total = len(test_set)
    start = time.time()

    for i, case in enumerate(test_set.cases, 1):
        if verbose:
            print(f"\n  [{i}/{total}] {case.question[:60]}...")
            print(f"    type={case.answer_type}, difficulty={case.difficulty}")

        # Retrieve
        results = retriever.search(case.question, top_k=k)
        ret_score = score_retrieval(case, results, k=k)
        result.retrieval_scores.append(ret_score)

        if verbose:
            print(f"    retrieval: {ret_score}")

        # Generate
        response = generator.generate(case.question, results)
        ans_score = score_answer(case, response)
        result.answer_scores.append(ans_score)

        if verbose:
            print(f"    answer: {ans_score}")
            preview = response.answer[:100].replace('\n', ' ')
            print(f"    response: {preview}...")

    result.elapsed_seconds = time.time() - start
    return result


# ==================== DISPLAY ====================

def print_eval_summary(result: EvalResult):
    """Print evaluation summary for a single model."""
    s = result.summary_dict()

    print(f"\n{'='*60}")
    print(f"  EVAL: {s['model']} on {result.test_set_name}")
    print(f"{'='*60}")
    print(f"  Test cases: {s['test_cases']}")
    print(f"  Time: {s['elapsed_seconds']}s")
    print()
    print(f"  RETRIEVAL:")
    print(f"    Recall@5:      {s['avg_recall@5']:.3f}  (did we find the right sections?)")
    print(f"    MRR:           {s['avg_mrr']:.3f}  (how high did the first relevant rank?)")
    print(f"    Precision@5:   {s['avg_precision@5']:.3f}  (what fraction of top-5 was relevant?)")
    print()
    print(f"  ANSWER QUALITY:")
    print(f"    Citation rate: {s['citation_rate']:.3f}  (fraction of answers with [N] citations)")
    print(f"    Faithfulness:  {s['faithfulness']:.3f}  (all claims grounded in context?)")
    print(f"    No-answer acc: {s['no_answer_accuracy']:.3f}  (handled unanswerable correctly?)")

    # Per-difficulty breakdown
    for diff in ["easy", "medium", "hard"]:
        cases_idx = [i for i, c in enumerate(result.retrieval_scores)
                     if i < len(result.answer_scores)]
        # Filter by difficulty from the test set this was run on
        # We don't have direct access, so skip if we can't determine
        pass

    # Show worst retrieval cases
    worst = sorted(result.retrieval_scores, key=lambda s: s.recall_at_k)[:3]
    if worst and worst[0].recall_at_k < 1.0:
        print(f"\n  WORST RETRIEVAL (lowest recall):")
        for w in worst:
            if w.recall_at_k < 1.0:
                print(f"    Recall={w.recall_at_k:.2f}: {w.question[:55]}...")
                print(f"      expected: {w.expected_sections}")
                print(f"      got:      {w.retrieved_sections[:3]}")


def print_comparison(results: list[EvalResult]):
    """Side-by-side comparison of multiple models."""
    if not results:
        return

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")

    metrics = [
        ("avg_recall@5", "Recall@5"),
        ("avg_mrr", "MRR"),
        ("avg_precision@5", "Precision@5"),
        ("citation_rate", "Citation Rate"),
        ("faithfulness", "Faithfulness"),
        ("no_answer_accuracy", "No-Answer Acc"),
        ("elapsed_seconds", "Time (s)"),
    ]

    # Header
    model_names = [r.model_name[:20] for r in results]
    col_w = max(14, max(len(n) for n in model_names) + 2)
    header = f"  {'Metric':<18}" + "".join(f"{n:>{col_w}}" for n in model_names)
    print(header)
    print("  " + "-" * (18 + col_w * len(results)))

    for key, label in metrics:
        values = []
        for r in results:
            s = r.summary_dict()
            values.append(f"{s[key]:.3f}" if isinstance(s[key], float) else str(s[key]))
        row = f"  {label:<18}" + "".join(f"{v:>{col_w}}" for v in values)
        print(row)

    # Highlight winner per metric
    print()
    print("  WINNERS:")
    for key, label in metrics:
        if key == "elapsed_seconds":
            # Lower is better for time
            best = min(results, key=lambda r: r.summary_dict()[key])
        else:
            best = max(results, key=lambda r: r.summary_dict()[key])
        val = best.summary_dict()[key]
        print(f"    {label:<18} → {best.model_name} ({val})")


# ==================== CLI ====================

def main():
    """Entry point for `uv run rag-eval`"""
    if len(sys.argv) < 2:
        print("Usage:")
        print('  uv run rag-eval papers/attention.pdf --model claude')
        print('  uv run rag-eval papers/attention.pdf --model claude --model glm5')
        print('  uv run rag-eval papers/attention.pdf --model llama3 --test-set eval.json')
        sys.exit(1)

    # Parse args
    filepath = None
    models = []
    test_set_path = None
    verbose = True
    i = 0
    args = sys.argv[1:]
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            models.append(args[i + 1])
            i += 2
        elif args[i] == "--test-set" and i + 1 < len(args):
            test_set_path = args[i + 1]
            i += 2
        elif args[i] == "--quiet":
            verbose = False
            i += 1
        elif not args[i].startswith("--"):
            filepath = args[i]
            i += 1
        else:
            i += 1

    if not filepath:
        print("Error: provide a PDF path")
        sys.exit(1)

    if not models:
        models = ["claude"]

    filepath = Path(filepath)

    # Load test set
    test_set = load_test_set(test_set_path)
    print(f"\n{'='*70}")
    print(f"  MODULE 5: RAG Evaluation")
    print(f"  Paper: {filepath.name}")
    print(f"  Test set: {test_set.name} ({len(test_set)} cases)")
    print(f"  Models: {models}")
    print(f"{'='*70}")

    # Build retrieval pipeline (shared across models — retrieval is model-independent)
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore
    from rag.sparse import BM25Index
    from rag.hybrid import HybridRetriever, CrossEncoderReranker
    from rag.generator import RAGGenerator

    index_dir = Path("index")
    if (index_dir / "index.faiss").exists():
        print("\nLoading saved index...")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        dense_store = VectorStore.load(index_dir, embedder)
    else:
        print("\nBuilding index...")
        from rag.pdf_parser import parse_pdf
        from rag.chunkers import HierarchicalChunker
        doc = parse_pdf(filepath)
        chunker = HierarchicalChunker(max_chunk_size=1500, include_context_header=True)
        chunks = chunker.chunk(doc.sections, {"source": doc.filename})
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        dense_store = VectorStore(embedder)
        dense_store.add_chunks(chunks)
        dense_store.save(index_dir)

    bm25 = BM25Index()
    bm25.add_chunks(dense_store.chunks)

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

    # Run eval for each model
    all_results = []
    for model_preset in models:
        print(f"\n{'─'*70}")
        print(f"  Evaluating: {model_preset}")
        print(f"{'─'*70}")

        generator = RAGGenerator(preset=model_preset)
        result = run_eval(
            test_set=test_set,
            retriever=retriever,
            generator=generator,
            model_name=model_preset,
            verbose=verbose,
        )
        all_results.append(result)
        print_eval_summary(result)

    # If multiple models, show comparison
    if len(all_results) > 1:
        print_comparison(all_results)

    print(f"\n{'='*70}")
    print("WHAT THESE NUMBERS TELL YOU")
    print(f"{'='*70}")
    print("""
  Recall@5 < 0.8?  → Your retrieval is missing relevant chunks.
    Fix: try different chunking, adjust chunk size, check section detection.

  MRR < 0.5?  → Relevant chunks rank too low.
    Fix: reranker may need tuning, or embedding model isn't capturing the query.

  Citation rate < 0.9?  → The LLM isn't citing its sources.
    Fix: strengthen the citation instruction in the system prompt.

  Faithfulness < 0.8?  → The LLM is making unsupported claims.
    Fix: use a stronger model, reduce context noise, add "only use context" emphasis.

  No-answer accuracy < 0.8?  → The LLM hallucinates on unanswerable questions.
    Fix: add explicit "say I don't know" examples in the prompt, use lower temperature.
    """)