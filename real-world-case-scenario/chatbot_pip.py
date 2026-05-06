"""
RAG Chatbot — PiP (Point in Polytope)

Uses our custom C++ HNSW-PiP index. Candidates during search are pruned
geometrically: if a candidate lies within the convex hull of the current
k-NN result set, it cannot improve recall and is discarded without computing
exact distance.

No training data required — PiP is parameterised only by pip_gamma and
pip_delta set at index construction.

Usage:
    python real-world-case-scenario/chatbot_pip.py [--gamma 95] [--delta 10] [--ef 100]

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

# Load PiP .so via spec (module name differs from filename)
_pip_so = ROOT / "hnsw_pip_cpp" / "src" / "build" / "hnsw_pip.cpython-311-darwin.so"
if not _pip_so.exists():
    raise SystemExit(f"PiP .so not found at {_pip_so}\nBuild it first.")

_pip_spec = importlib.util.spec_from_file_location("hnswPip_cpp", _pip_so)
hnswPip = importlib.util.module_from_spec(_pip_spec)
_pip_spec.loader.exec_module(hnswPip)


def build_index(docs: list[dict], embeddings: np.ndarray,
                M: int, efC: int, gamma: float, delta: int, metric: str):
    dim = embeddings.shape[1]
    print(f"  Building PiP C++ index (M={M}, efC={efC}, γ={gamma}, Δ={delta}) …")
    t0 = time.perf_counter()
    idx = hnswPip.HNSWPiPIndex(
        dim=int(dim), M=int(M), efConstruction=int(efC),
        metric=str(metric), pip_gamma=float(gamma), pip_delta=int(delta),
    )
    idx.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    print(f"  Index built in {time.perf_counter()-t0:.1f}s\n")
    return idx


def make_retrieve_fn(idx, docs, k, ef):
    def retrieve(question: str):
        q_emb = embed(question)
        xq = np.ascontiguousarray(q_emb[np.newaxis, :], dtype=np.float32)

        t0 = time.perf_counter()
        _, I = idx.search(xq, k=int(k), efSearch=int(ef))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        indices = I[0].tolist()
        chunks = [docs[i] for i in indices if 0 <= i < len(docs)]
        return chunks, elapsed_ms

    return retrieve


def main():
    parser = argparse.ArgumentParser(description="PiP RAG chatbot")
    parser.add_argument("--gamma", type=float, default=95.0,
                        help="PiP gamma — stability percentile (default: 95)")
    parser.add_argument("--delta", type=int, default=10,
                        help="PiP delta — minimum stable iterations (default: 10)")
    parser.add_argument("--ef",  type=int, default=100,
                        help="efSearch (default: 100)")
    parser.add_argument("--M",   type=int, default=16)
    parser.add_argument("--efC", type=int, default=200)
    args = parser.parse_args()

    print_header(f"PiP (γ={args.gamma}, Δ={args.delta}, efSearch={args.ef})")

    docs = load_documents()
    print(f"  Loaded {len(docs)} document chunks from context/\n")

    print("  Embedding corpus …")
    embeddings = embed_corpus(docs, verbose=True)
    print(f"  Corpus shape: {embeddings.shape}\n")

    idx = build_index(docs, embeddings, args.M, args.efC,
                      args.gamma, args.delta, "l2")

    retrieve_fn = make_retrieve_fn(idx, docs, TOP_K, args.ef)
    chat_loop(retrieve_fn)


if __name__ == "__main__":
    main()
