"""
RAG Chatbot — Baseline HNSW (via ChromaDB)

ChromaDB uses hnswlib internally, providing a production-grade vector store.
This is the reference baseline: standard HNSW with no early termination or
adaptive ef selection.

Usage:
    python real-world-case-scenario/chatbot_hnsw.py

Requirements:
    pip install ollama chromadb
    ollama pull nomic-embed-text
    ollama pull llama3.2
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    load_documents, embed, embed_corpus,
    retrieve_by_indices, generate_answer,
    print_header, print_retrieved, chat_loop,
    TOP_K, EMBED_MODEL,
)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise SystemExit("Install chromadb: pip install chromadb")


def build_index(docs: list[dict]) -> chromadb.Collection:
    print("  Building ChromaDB (HNSW) index …")
    client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("rag_hnsw")
    except Exception:
        pass

    collection = client.create_collection(
        name="rag_hnsw",
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    print(f"  Embedding {len(docs)} chunks with {EMBED_MODEL} …")
    t0 = time.perf_counter()
    embeddings = embed_corpus(docs, verbose=True)
    print(f"  Embedded in {time.perf_counter()-t0:.1f}s")

    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings.tolist(),
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )
    print(f"  Index ready — {len(docs)} chunks\n")
    return collection, docs, embeddings


def make_retrieve_fn(collection, docs):
    def retrieve(question: str):
        q_emb = embed(question).tolist()
        t0 = time.perf_counter()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        retrieved = []
        for doc_id, text, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            retrieved.append({"id": doc_id, "source": meta["source"], "text": text})

        return retrieved, elapsed_ms

    return retrieve


def main():
    print_header("Baseline HNSW (ChromaDB)")

    docs = load_documents()
    print(f"  Loaded {len(docs)} document chunks from context/\n")

    collection, docs, embeddings = build_index(docs)
    retrieve_fn = make_retrieve_fn(collection, docs)
    chat_loop(retrieve_fn)


if __name__ == "__main__":
    main()
