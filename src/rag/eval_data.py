"""
eval_data.py — Evaluation dataset for RAG
==========================================

The hardest part of RAG evaluation is having something to evaluate
AGAINST. You need:
  - Questions with known answers
  - Expected source sections (which chunks SHOULD be retrieved?)
  - Ground truth to compare the LLM's answer against

Three ways to build a test set:
  1. MANUAL — write QA pairs by hand (best quality, slow)
  2. LLM-GENERATED — use an LLM to read the paper and generate QA pairs
     (fast, decent quality, needs human review)
  3. HYBRID — LLM generates, human reviews and corrects (best tradeoff)

This module handles the data format and provides a sample test set
for the "Attention Is All You Need" paper so you can run evals
immediately.

Usage:
  from rag.eval_data import load_test_set, ATTENTION_TEST_SET
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class TestCase:
    """
    A single evaluation question.

    Fields:
      question:         the query to ask
      expected_answer:  gold standard answer (for answer quality eval)
      expected_sections: which paper sections should be retrieved
                         (for retrieval eval — did we find the right chunks?)
      answer_type:      "factual" (specific fact), "conceptual" (explanation),
                        "unanswerable" (not in the paper — tests hallucination)
      difficulty:       "easy" (answer in one place), "medium" (needs synthesis),
                        "hard" (spread across sections or requires inference)
    """
    question: str
    expected_answer: str
    expected_sections: list[str] = field(default_factory=list)
    answer_type: str = "factual"       # factual | conceptual | unanswerable
    difficulty: str = "easy"           # easy | medium | hard
    tags: list[str] = field(default_factory=list)


@dataclass
class TestSet:
    """Collection of test cases with metadata."""
    name: str
    paper: str
    cases: list[TestCase] = field(default_factory=list)
    description: str = ""

    def __len__(self):
        return len(self.cases)

    def by_type(self, answer_type: str) -> list[TestCase]:
        return [c for c in self.cases if c.answer_type == answer_type]

    def by_difficulty(self, difficulty: str) -> list[TestCase]:
        return [c for c in self.cases if c.difficulty == difficulty]

    def save(self, filepath: str | Path):
        """Save test set to JSON."""
        filepath = Path(filepath)
        data = {
            "name": self.name,
            "paper": self.paper,
            "description": self.description,
            "cases": [asdict(c) for c in self.cases],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.cases)} test cases to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "TestSet":
        """Load test set from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        cases = [TestCase(**c) for c in data["cases"]]
        return cls(
            name=data["name"],
            paper=data["paper"],
            description=data.get("description", ""),
            cases=cases,
        )


# ==================== SAMPLE TEST SET ====================
# Hand-crafted for "Attention Is All You Need" (Vaswani et al., 2017)
#
# These cover:
#   - Factual questions (specific numbers, names)
#   - Conceptual questions (how/why things work)
#   - Cross-section questions (answer requires multiple sections)
#   - Unanswerable questions (not in the paper — should say "I don't know")

ATTENTION_TEST_SET = TestSet(
    name="attention_is_all_you_need",
    paper="attention.pdf",
    description="Evaluation set for Vaswani et al., 2017 — Attention Is All You Need",
    cases=[
        # === FACTUAL / EASY ===
        TestCase(
            question="What BLEU score did the Transformer achieve on the WMT 2014 English-to-German translation task?",
            expected_answer="The Transformer achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.",
            expected_sections=["6.1 Machine Translat", "Abstract"],
            answer_type="factual",
            difficulty="easy",
            tags=["results", "metrics"],
        ),
        TestCase(
            question="How many encoder and decoder layers does the base Transformer model have?",
            expected_answer="The base Transformer model has 6 encoder layers and 6 decoder layers (N=6).",
            expected_sections=["3 Model Architecture"],
            answer_type="factual",
            difficulty="easy",
            tags=["architecture"],
        ),
        TestCase(
            question="What hardware was used to train the model?",
            expected_answer="The model was trained on 8 NVIDIA P100 GPUs.",
            expected_sections=["5.2 Hardware and Sch", "5 Training"],
            answer_type="factual",
            difficulty="easy",
            tags=["training"],
        ),
        TestCase(
            question="What optimizer was used for training?",
            expected_answer="The Adam optimizer was used with beta1=0.9, beta2=0.98, and epsilon=10^-9.",
            expected_sections=["5.3 Optimizer"],
            answer_type="factual",
            difficulty="easy",
            tags=["training", "optimizer"],
        ),
        TestCase(
            question="What is the dimensionality of the model (d_model) in the base configuration?",
            expected_answer="The model dimensionality d_model is 512 in the base configuration.",
            expected_sections=["3 Model Architecture"],
            answer_type="factual",
            difficulty="easy",
            tags=["architecture", "hyperparameters"],
        ),

        # === CONCEPTUAL / MEDIUM ===
        TestCase(
            question="How does multi-head attention work and why is it used instead of single-head attention?",
            expected_answer="Multi-head attention runs multiple attention functions in parallel, each with different learned projections. This allows the model to attend to information from different representation subspaces at different positions. A single attention head averages these, inhibiting this ability.",
            expected_sections=["3.2 Attention", "3.2.2 Multi-Head Att"],
            answer_type="conceptual",
            difficulty="medium",
            tags=["attention", "architecture"],
        ),
        TestCase(
            question="Why does the Transformer use positional encoding, and what approach was chosen?",
            expected_answer="Since the Transformer contains no recurrence or convolution, positional encodings are added to give the model information about token positions. The authors use sine and cosine functions of different frequencies.",
            expected_sections=["3.5 Positional Encod"],
            answer_type="conceptual",
            difficulty="medium",
            tags=["architecture", "positional"],
        ),
        TestCase(
            question="What are the three different ways attention is used in the Transformer?",
            expected_answer="Attention is used in three ways: (1) encoder self-attention, where each position attends to all positions in the previous encoder layer; (2) decoder self-attention with masking, preventing leftward information flow; (3) encoder-decoder attention, where decoder queries attend to encoder output.",
            expected_sections=["3.2.3 Applications o"],
            answer_type="conceptual",
            difficulty="medium",
            tags=["attention", "architecture"],
        ),
        TestCase(
            question="Why is self-attention preferred over recurrent layers for sequence modeling?",
            expected_answer="Self-attention connects all positions with O(1) sequential operations vs O(n) for recurrence, enabling more parallelization. It also has shorter maximum path lengths between dependencies, making it easier to learn long-range relationships.",
            expected_sections=["4 Why Self-Attention"],
            answer_type="conceptual",
            difficulty="medium",
            tags=["attention", "comparison"],
        ),

        # === CROSS-SECTION / HARD ===
        TestCase(
            question="How does the Transformer's training efficiency compare to previous state-of-the-art models?",
            expected_answer="The Transformer requires significantly less training cost than previous models. The big Transformer model was trained in 3.5 days on 8 GPUs, while achieving better BLEU scores than all previous ensembles. The base model also surpasses all previous models at a fraction of the training cost.",
            expected_sections=["6.1 Machine Translat", "5.2 Hardware and Sch", "Abstract"],
            answer_type="factual",
            difficulty="hard",
            tags=["results", "training", "comparison"],
        ),
        TestCase(
            question="What regularization techniques were applied during training and what were their effects?",
            expected_answer="Three regularization techniques were used: (1) residual dropout with rate 0.1 applied to sub-layer outputs and positional encoding sums; (2) attention dropout; (3) label smoothing of value 0.1, which hurt perplexity but improved accuracy and BLEU score.",
            expected_sections=["5.4 Regularization"],
            answer_type="factual",
            difficulty="hard",
            tags=["training", "regularization"],
        ),

        # === UNANSWERABLE (tests hallucination resistance) ===
        TestCase(
            question="What is the carbon footprint of training the Transformer model?",
            expected_answer="The paper does not discuss the carbon footprint or environmental impact of training.",
            expected_sections=[],
            answer_type="unanswerable",
            difficulty="easy",
            tags=["unanswerable", "hallucination_test"],
        ),
        TestCase(
            question="How does the Transformer perform on speech recognition tasks?",
            expected_answer="The paper does not evaluate the Transformer on speech recognition. It focuses on machine translation and English constituency parsing.",
            expected_sections=[],
            answer_type="unanswerable",
            difficulty="easy",
            tags=["unanswerable", "hallucination_test"],
        ),
    ],
)


def load_test_set(path: str | Path = None) -> TestSet:
    """Load test set from file, or return the built-in sample."""
    if path:
        return TestSet.load(path)
    return ATTENTION_TEST_SET