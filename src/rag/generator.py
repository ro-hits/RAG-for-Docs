"""
generator.py — LLM-powered answer generation with citations
============================================================

This is where the full RAG loop closes:
  Question → Retrieve chunks → Feed to LLM → Cited answer

Provider-agnostic:
  Supports any LLM through a simple adapter pattern:
  - Anthropic Claude (native SDK)
  - OpenAI-compatible APIs (GPT, GLM-5 via OpenRouter, DeepSeek, etc.)
  - Ollama (local models, no API key needed)

  All three share the same interface: system prompt + user message → text.
  Adding a new provider is ~20 lines of code.

API keys (PowerShell):
  $env:ANTHROPIC_API_KEY = "sk-ant-..."           # Claude
  $env:OPENAI_API_KEY = "sk-..."                  # OpenAI / OpenRouter
  $env:OPENROUTER_API_KEY = "sk-or-..."           # OpenRouter specifically

Usage:
  from rag.generator import RAGGenerator
  gen = RAGGenerator(preset="claude")
  gen = RAGGenerator(preset="glm5")
  gen = RAGGenerator(preset="llama3")
  gen = RAGGenerator(provider="openai", model="my-model", base_url="http://...")
"""

import os
import re
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from rag.chunkers import Chunk


# ==================== DATA STRUCTURES ====================

@dataclass
class Citation:
    """A single citation linking a claim to a source chunk."""
    chunk_id: int
    section: str
    claim: str

    def __repr__(self):
        return f"[{self.chunk_id}] {self.section}: {self.claim[:60]}..."


@dataclass
class RAGResponse:
    """Full response with answer, citations, and diagnostics."""
    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks_used: list[int] = field(default_factory=list)
    chunks_provided: list[int] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    faithful: bool = True
    no_answer: bool = False
    usage: dict = field(default_factory=dict)

    @property
    def chunks_unused(self) -> list[int]:
        return [c for c in self.chunks_provided if c not in self.chunks_used]

    def summary(self) -> str:
        lines = [
            f"Query: {self.query!r}",
            f"Provider: {self.provider} | Model: {self.model}",
            f"Faithful: {self.faithful}",
            f"No answer: {self.no_answer}",
            f"Chunks provided: {len(self.chunks_provided)} | used: {len(self.chunks_used)} | unused: {len(self.chunks_unused)}",
            f"Citations: {len(self.citations)}",
        ]
        if self.usage:
            inp = self.usage.get('input_tokens', '?')
            out = self.usage.get('output_tokens', '?')
            lines.append(f"Tokens: {inp} in, {out} out")
        return "\n".join(lines)


# ==================== PROMPT (shared across all providers) ====================

SYSTEM_PROMPT = """You are a precise research assistant that answers questions based ONLY on the provided context chunks from a research paper.

Rules:
1. Answer the question using ONLY information from the provided context chunks.
2. After each claim or fact, cite the source chunk using [N] where N is the chunk number.
3. If the context does not contain enough information to answer the question, say: "The provided context does not contain sufficient information to answer this question." and explain what's missing.
4. Do NOT use any knowledge outside the provided chunks. If you know something but it's not in the chunks, do not include it.
5. Be concise. Don't repeat the question. Don't add unnecessary preamble.
6. If chunks contain conflicting information, note the conflict and cite both sources.

Example format:
The model uses multi-head attention to process sequences in parallel [1]. Training was performed on 8 GPUs for 12 hours [3]. The authors report a BLEU score of 28.4 on the English-to-German task [2]."""


def build_context_block(results: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    blocks = []
    for r in results:
        chunk = r["chunk"]
        idx = chunk.chunk_id
        section = chunk.metadata.get("section", "Unknown")
        text = chunk.text
        text = re.sub(r'^\[Section:[^\]]+\]\n?', '', text).strip()
        block = f"--- Chunk [{idx}] | Section: {section} ---\n{text}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_user_message(query: str, context_block: str) -> str:
    """Build user message. Query after context (recency bias)."""
    return f"""Context chunks from the paper:

{context_block}

---

Question: {query}

Answer the question using ONLY the context above. Cite sources using [N] notation."""


# ==================== CITATION EXTRACTION ====================

def extract_citations(answer: str, chunks_provided: list[int]) -> list[dict]:
    """Parse [N] citation markers from the answer."""
    citations = []
    for match in re.finditer(r'\[(\d+(?:\s*,\s*\d+)*)\]', answer):
        ids = [int(x.strip()) for x in match.group(1).split(',')]
        start = max(0, match.start() - 80)
        context = answer[start:match.start()].strip()
        for chunk_id in ids:
            citations.append({
                "chunk_id": chunk_id,
                "position": match.start(),
                "claim": context,
                "valid": chunk_id in chunks_provided,
            })
    return citations


def check_faithfulness(answer: str, citations: list[dict], chunks_provided: list[int]) -> dict:
    """Basic faithfulness heuristic — checks for invalid citations and uncited sentences."""
    invalid_citations = [c for c in citations if not c["valid"]]
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    uncited_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue
        if any(phrase in sent.lower() for phrase in [
            "context does not", "not enough information",
            "cannot answer", "not mentioned", "no information",
        ]):
            continue
        if not re.search(r'\[\d+\]', sent):
            uncited_sentences.append(sent)
    return {
        "is_faithful": len(invalid_citations) == 0 and len(uncited_sentences) == 0,
        "invalid_citations": invalid_citations,
        "uncited_sentences": uncited_sentences,
        "total_citations": len(citations),
        "unique_chunks_cited": len(set(c["chunk_id"] for c in citations)),
    }


# ==================== LLM BACKENDS ====================

class LLMBackend(ABC):
    """
    Abstract base for LLM providers.

    Every backend implements one method: call().
    Takes system prompt + user message, returns (text, usage_dict).

    This is the adapter pattern — swap providers without touching
    any other code.
    """

    @abstractmethod
    def call(self, system: str, user: str) -> tuple[str, dict]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ClaudeBackend(LLMBackend):
    """Anthropic Claude via native SDK."""

    def __init__(self, model: str, max_tokens: int, temperature: float):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set.\n"
                "PowerShell: $env:ANTHROPIC_API_KEY = 'sk-ant-...'"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "claude"

    def call(self, system: str, user: str) -> tuple[str, dict]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return text, usage


class OpenAIBackend(LLMBackend):
    """
    OpenAI-compatible API — covers GPT, GLM-5, DeepSeek, Mistral, etc.

    Any provider that speaks /v1/chat/completions works here:
      - OpenAI:      https://api.openai.com/v1
      - OpenRouter:  https://openrouter.ai/api/v1
      - DeepSeek:    https://api.deepseek.com/v1
      - Z.ai (GLM):  https://api.z.ai/v1
      - Together:    https://api.together.xyz/v1
      - Local vLLM:  http://localhost:8000/v1
    """

    def __init__(self, model: str, max_tokens: int, temperature: float,
                 base_url: str | None = None, api_key: str | None = None):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        # Smart API key resolution
        if api_key is None:
            if base_url and "openrouter" in base_url:
                api_key = os.environ.get("OPENROUTER_API_KEY")
            elif base_url and "deepseek" in base_url:
                api_key = os.environ.get("DEEPSEEK_API_KEY")
            elif base_url and "z.ai" in base_url:
                api_key = os.environ.get("ZAI_API_KEY")
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Set one of:\n"
                "  $env:OPENAI_API_KEY = '...'\n"
                "  $env:OPENROUTER_API_KEY = '...'\n"
                "  $env:DEEPSEEK_API_KEY = '...'\n"
                "  $env:ZAI_API_KEY = '...'"
            )

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._base_url = base_url or "openai"

    @property
    def name(self) -> str:
        for keyword in ["openrouter", "deepseek", "z.ai", "together"]:
            if keyword in self._base_url:
                return keyword
        return "openai"

    def call(self, system: str, user: str) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return text, usage


class OllamaBackend(LLMBackend):
    """
    Ollama for local models — no API key, no cost, full privacy.

    ollama pull llama3.1
    Then it just works at localhost:11434.
    """

    def __init__(self, model: str, max_tokens: int, temperature: float,
                 host: str = "http://localhost:11434"):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        self.client = openai.OpenAI(
            base_url=f"{host}/v1",
            api_key="ollama",  # Ollama ignores this but SDK requires it
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "ollama"

    def call(self, system: str, user: str) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return text, usage


# ==================== PRESETS ====================

PRESETS = {
    # --- Anthropic ---
    "claude":       {"provider": "claude",  "model": "claude-sonnet-4-20250514"},
    "claude-haiku": {"provider": "claude",  "model": "claude-haiku-4-5-20251001"},
    "claude-opus":  {"provider": "claude",  "model": "claude-opus-4-6"},

    # --- GLM-5 via OpenRouter ---
    "glm5":         {"provider": "openai",  "model": "zai/glm-5",
                     "base_url": "https://openrouter.ai/api/v1"},

    # --- DeepSeek ---
    "deepseek":     {"provider": "openai",  "model": "deepseek-chat",
                     "base_url": "https://api.deepseek.com/v1"},

    # --- OpenAI ---
    "gpt4o":        {"provider": "openai",  "model": "gpt-4o"},
    "gpt4o-mini":   {"provider": "openai",  "model": "gpt-4o-mini"},

    # --- Local (Ollama) ---
    "llama3":       {"provider": "ollama",  "model": "llama3.1"},
    "mistral":      {"provider": "ollama",  "model": "mistral"},
    "qwen":         {"provider": "ollama",  "model": "qwen2.5"},
}


def list_presets() -> str:
    """List available model presets."""
    lines = ["Available presets:"]
    for name, cfg in PRESETS.items():
        provider = cfg["provider"]
        model = cfg["model"]
        url = cfg.get("base_url", "")
        extra = f"  ({url})" if url else ""
        lines.append(f"  {name:<16} {provider:<8} {model}{extra}")
    return "\n".join(lines)


# ==================== GENERATOR (provider-agnostic) ====================

def create_backend(
    provider: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMBackend:
    """Factory — create the right backend from provider string."""
    if provider == "claude":
        return ClaudeBackend(model, max_tokens, temperature)
    elif provider == "openai":
        return OpenAIBackend(model, max_tokens, temperature, base_url, api_key)
    elif provider == "ollama":
        return OllamaBackend(model, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use: claude, openai, ollama")


class RAGGenerator:
    """
    Generate answers from retrieved chunks using any LLM.

    Completely decoupled from the LLM provider. Handles context
    trimming, prompt construction, citation parsing, faithfulness
    checking. The LLMBackend handles the actual API call.
    """

    def __init__(
        self,
        preset: str | None = None,
        provider: str = "claude",
        model: str = "claude-sonnet-4-20250514",
        max_context_tokens: int = 4000,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        if preset:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset: {preset!r}\n{list_presets()}")
            cfg = PRESETS[preset]
            provider = cfg["provider"]
            model = cfg["model"]
            base_url = cfg.get("base_url", base_url)

        self.backend = create_backend(
            provider=provider, model=model,
            max_tokens=max_output_tokens, temperature=temperature,
            base_url=base_url, api_key=api_key,
        )
        self.model = model
        self.provider = provider
        self.max_context_tokens = max_context_tokens
        print(f"Generator ready: {self.backend.name}/{model}")

    def _trim_context(self, results: list[dict]) -> list[dict]:
        """Trim context to fit token budget (~4 chars per token)."""
        char_budget = self.max_context_tokens * 4
        trimmed = []
        total_chars = 0
        for r in results:
            chunk_chars = len(r["chunk"].text)
            if total_chars + chunk_chars > char_budget:
                break
            trimmed.append(r)
            total_chars += chunk_chars
        if len(trimmed) < len(results):
            print(f"  Context trimmed: {len(results)} → {len(trimmed)} chunks "
                  f"({total_chars:,} chars, ~{total_chars//4:,} tokens)")
        return trimmed

    def generate(self, query: str, results: list[dict]) -> RAGResponse:
        """Generate a cited answer. Works identically across all backends."""
        if not results:
            return RAGResponse(
                query=query,
                answer="No context chunks were retrieved. Cannot answer.",
                no_answer=True, model=self.model, provider=self.provider,
            )

        results = self._trim_context(results)
        chunks_provided = [r["chunk"].chunk_id for r in results]

        context_block = build_context_block(results)
        user_message = build_user_message(query, context_block)

        print(f"  Calling {self.backend.name}/{self.model}...", end=" ", flush=True)
        answer, usage = self.backend.call(SYSTEM_PROMPT, user_message)
        print("done")

        raw_citations = extract_citations(answer, chunks_provided)
        citations = []
        chunks_used = set()
        for rc in raw_citations:
            if rc["valid"]:
                chunk = next((r["chunk"] for r in results if r["chunk"].chunk_id == rc["chunk_id"]), None)
                section = chunk.metadata.get("section", "Unknown") if chunk else "Unknown"
                citations.append(Citation(chunk_id=rc["chunk_id"], section=section, claim=rc["claim"]))
                chunks_used.add(rc["chunk_id"])

        faith = check_faithfulness(answer, raw_citations, chunks_provided)

        no_answer_phrases = [
            "does not contain sufficient information", "context does not",
            "cannot answer", "not mentioned in the provided", "no information about",
        ]
        no_answer = any(phrase in answer.lower() for phrase in no_answer_phrases)

        return RAGResponse(
            query=query, answer=answer, citations=citations,
            chunks_used=list(chunks_used), chunks_provided=chunks_provided,
            model=self.model, provider=self.backend.name,
            faithful=faith["is_faithful"], no_answer=no_answer, usage=usage,
        )


# ==================== DISPLAY ====================

def print_response(resp: RAGResponse):
    """Pretty-print a RAG response with citations and diagnostics."""
    print(f"\n{'='*70}")
    print(f"  ANSWER")
    print(f"{'='*70}")
    print(f"\n{resp.answer}")

    print(f"\n{'─'*70}")
    print(f"  DIAGNOSTICS")
    print(f"{'─'*70}")
    print(f"  {resp.summary()}")

    if resp.citations:
        print(f"\n  Citations:")
        for c in resp.citations:
            print(f"    [{c.chunk_id}] {c.section}: {c.claim[:70]}...")

    if resp.chunks_unused:
        print(f"\n  Unused chunks (in context but not cited): {resp.chunks_unused}")
        print(f"  → These chunks were retrieved but the LLM didn't find them relevant.")

    if not resp.faithful:
        print(f"\n  ⚠ FAITHFULNESS WARNING: Some claims may not be grounded in context.")