"""
RAG Chatbot — Ada-ef (Adaptive Expansion Factor)

Uses our custom C++ HNSW-AdaEF index. An offline calibration table maps
query difficulty (distance distribution statistics) to the minimum efSearch
needed to achieve a target recall. At query time, the appropriate ef is
selected automatically — no fixed efSearch parameter required.

Usage:
    python real-world-case-scenario/chatbot_adaef.py [--target-recall 0.90]

Requirements:
    pip install ollama
    ollama pull nomic-embed-text
    ollama pull llama3.2
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    load_documents, embed, embed_corpus,
    generate_answer, print_header, print_retrieved, chat_loop,
    TOP_K,
)

import numpy as np

# Load Ada-ef .so via spec
_adaef_so = ROOT / "hnsw_adaef_cpp" / "src" / "build" / "hnsw_edaef.cpython-311-darwin.so"
if not _adaef_so.exists():
    raise SystemExit(f"Ada-ef .so not found at {_adaef_so}\nBuild it first.")

_adaef_spec = importlib.util.spec_from_file_location("hnswAdaEF_cpp", _adaef_so)
hnswAdaEF = importlib.util.module_from_spec(_adaef_spec)
_adaef_spec.loader.exec_module(hnswAdaEF)


def build_index(docs: list[dict], embeddings: np.ndarray,
                M: int, efC: int, metric: str,
                k: int, target_recall: float):
    dim = embeddings.shape[1]
    ef_values = [20, 50, 75, 100, 150, 200, 300]

    print(f"  Building Ada-ef C++ index (M={M}, efC={efC}) …")
    t0 = time.perf_counter()
    idx = hnswAdaEF.HNSWAdaEFIndex(
        dim=int(dim), M=int(M), efConstruction=int(efC), metric=str(metric)
    )
    idx.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    print(f"  Index built in {time.perf_counter()-t0:.1f}s")

    print(f"  Building offline calibration table (target_recall={target_recall}) …")
    t0 = time.perf_counter()
    idx.build_adaef_offline(
        k=int(k),
        target_recall=float(target_recall),
        ef_values=ef_values,
    )
    print(f"  Calibration table ready in {time.perf_counter()-t0:.1f}s\n")
    return idx


def make_retrieve_fn(idx, docs, k, target_recall):
    def retrieve(question: str):
        q_emb = embed(question)
        xq = np.ascontiguousarray(q_emb[np.newaxis, :], dtype=np.float32)

        t0 = time.perf_counter()
        _, I = idx.search(xq, k=int(k), target_recall=float(target_recall))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        indices = I[0].tolist()
        chunks = [docs[i] for i in indices if 0 <= i < len(docs)]
        return chunks, elapsed_ms

    return retrieve


def main():
    parser = argparse.ArgumentParser(description="Ada-ef RAG chatbot")
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="Target recall for adaptive ef selection (default: 0.90)")
    parser.add_argument("--M",   type=int, default=16)
    parser.add_argument("--efC", type=int, default=200)
    args = parser.parse_args()

    print_header(f"Ada-ef (target_recall={args.target_recall})")

    docs = load_documents()
    print(f"  Loaded {len(docs)} document chunks from context/\n")

    print("  Embedding corpus …")
    embeddings = embed_corpus(docs, verbose=True)
    print(f"  Corpus shape: {embeddings.shape}\n")

    idx = build_index(docs, embeddings, args.M, args.efC, "cosine",
                      TOP_K, args.target_recall)

    retrieve_fn = make_retrieve_fn(idx, docs, TOP_K, args.target_recall)
    chat_loop(retrieve_fn)


if __name__ == "__main__":
    main()
