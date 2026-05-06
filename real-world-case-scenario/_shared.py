"""
Shared utilities for all RAG chatbots.

Handles:
  - Loading and chunking markdown documents from context/
  - Ollama embeddings (nomic-embed-text)
  - Ollama chat (llama3.2)
  - Formatted output helpers
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import ollama
except ImportError:
    raise SystemExit("Install ollama client: pip install ollama")

CONTEXT_DIR = Path(__file__).parent / "context"
EMBED_MODEL  = "nomic-embed-text"   # ollama pull nomic-embed-text
CHAT_MODEL   = "glm-5:cloud"         # ollama pull glm-5:cloud
TOP_K        = 5                    # chunks to retrieve per query
CHUNK_MIN_CHARS = 150               # ignore very short paragraphs


# ── document loading ──────────────────────────────────────────────────────────

def load_documents() -> list[dict]:
    """
    Load every markdown file from context/ and split into paragraph chunks.
    Returns a list of dicts: {id, source, text}.
    """
    docs = []
    for md_file in sorted(CONTEXT_DIR.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in raw.split("\n\n")]
        paragraphs = [p for p in paragraphs if len(p) >= CHUNK_MIN_CHARS]
        for i, para in enumerate(paragraphs):
            docs.append({
                "id":     f"{md_file.stem}__{i:03d}",
                "source": md_file.stem.replace("_", " ").title(),
                "text":   para,
            })
    return docs


# ── embeddings ────────────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    """Embed a single string using Ollama nomic-embed-text."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(resp["embedding"], dtype=np.float32)


def embed_corpus(docs: list[dict], verbose: bool = True) -> np.ndarray:
    """
    Embed all document chunks. Returns float32 array of shape (n_docs, dim).
    """
    embeddings = []
    for i, doc in enumerate(docs):
        if verbose and (i % 20 == 0):
            print(f"  Embedding {i+1}/{len(docs)} …", flush=True)
        embeddings.append(embed(doc["text"]))
    return np.stack(embeddings)


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve_by_indices(docs: list[dict], indices: list[int]) -> list[dict]:
    """Return docs at the given indices (filtering out -1 / invalid)."""
    return [docs[i] for i in indices if 0 <= i < len(docs)]


# ── LLM answer generation ─────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"You are a helpful assistant. Answer the following question using ONLY "
        f"the context provided below. If the answer is not in the context, say "
        f"'I don't have enough information on that topic.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    prompt = build_prompt(question, chunks)
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]


# ── display helpers ───────────────────────────────────────────────────────────

def print_header(method: str):
    print(f"\n{'═'*60}")
    print(f"  RAG Chatbot — {method}")
    print(f"  Embedding : {EMBED_MODEL}")
    print(f"  LLM       : {CHAT_MODEL}")
    print(f"{'═'*60}\n")


def print_retrieved(chunks: list[dict], elapsed_ms: float):
    print(f"\n  Retrieved {len(chunks)} chunks in {elapsed_ms:.1f} ms:")
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:120].replace("\n", " ")
        print(f"  [{i}] ({c['source']}) {preview}…")
    print()


def chat_loop(retrieve_fn):
    """
    Interactive chat loop.
    retrieve_fn(query: str) → (chunks: list[dict], elapsed_ms: float)
    """
    print("Type your question (or 'quit' to exit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question or question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        chunks, elapsed_ms = retrieve_fn(question)
        print_retrieved(chunks, elapsed_ms)

        print("Assistant: ", end="", flush=True)
        answer = generate_answer(question, chunks)
        print(answer)
        print()
