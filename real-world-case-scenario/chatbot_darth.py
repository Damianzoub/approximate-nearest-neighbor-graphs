"""
RAG Chatbot — DARTH (Dynamic Approximate Recall Threshold)

Uses our custom C++ HNSW-DARTH index. A LightGBM predictor is trained
on-the-fly from the document corpus and predicts per-query whether the
search can terminate early while still achieving the target recall Rt.

Usage:
    python real-world-case-scenario/chatbot_darth.py [--Rt 0.90] [--ef 100]

Requirements:
    pip install ollama lightgbm
    ollama pull nomic-embed-text
    ollama pull llama3.2
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hnswDarth_cpp" / "src" / "build"))
sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    load_documents, embed, embed_corpus,
    generate_answer, print_header, print_retrieved, chat_loop,
    TOP_K,
)

import numpy as np

try:
    import hnswDarth_cpp
except ImportError:
    raise SystemExit(
        "hnswDarth_cpp not found. Build it:\n"
        "  cd hnswDarth_cpp/src/build && cmake .. && make"
    )

from darth.collect_training_data import collect_training_data
from darth.train_predictor import train as train_predictor
from darth.predictor import LGBMPredictor


def build_index(docs: list[dict], embeddings: np.ndarray,
                M: int, efC: int, metric: str):
    dim = embeddings.shape[1]
    print(f"  Building DARTH C++ index (M={M}, efC={efC}, dim={dim}) …")
    t0 = time.perf_counter()
    idx = hnswDarth_cpp.HNSWDarthIndex(
        dim=int(dim), M=int(M), efConstruction=int(efC), metric=metric
    )
    idx.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    print(f"  Index built in {time.perf_counter()-t0:.1f}s")
    return idx


MIN_CORPUS_FOR_PREDICTOR = 1000  # corpus too small → skip predictor


def train_predictor_from_corpus(embeddings: np.ndarray,
                                k: int, efC: int,
                                model_path: Path, csv_path: Path):
    """
    Train a DARTH predictor from the document corpus.
    Returns None if the corpus is too small to produce meaningful training data.
    """
    if len(embeddings) < MIN_CORPUS_FOR_PREDICTOR:
        print(f"  [info] Corpus too small ({len(embeddings)} chunks < {MIN_CORPUS_FOR_PREDICTOR}) "
              f"for predictor training — using DARTH without early termination.")
        return None

    if model_path.exists():
        print(f"  [skip] Predictor already cached: {model_path.name}")
        return LGBMPredictor(str(model_path))

    n_learn = min(500, len(embeddings))
    rng = np.random.default_rng(42)
    idx_learn = rng.choice(len(embeddings), size=n_learn, replace=False)
    xlearn = embeddings[idx_learn]

    print(f"  Computing exact GT for {n_learn} learn queries …")
    try:
        import faiss
        fi = faiss.IndexFlatL2(embeddings.shape[1])
        fi.add(embeddings.astype(np.float32))
        _, gt_learn = fi.search(xlearn.astype(np.float32), k)
        gt_learn = gt_learn.astype(np.int32)
    except ImportError:
        print("  [warn] faiss not found — using numpy brute-force GT (slow)")
        gt_learn = np.empty((n_learn, k), dtype=np.int32)
        for i, q in enumerate(xlearn):
            dists = ((embeddings - q) ** 2).sum(1)
            gt_learn[i] = np.argsort(dists)[:k]

    if not csv_path.exists():
        print(f"  Collecting DARTH training data …")
        from hsnw_constructionDARTH import HNSW_DARTH
        py_idx = HNSW_DARTH(
            dim=embeddings.shape[1], M=16, efConstruction=efC, metric="l2"
        )
        for i, v in enumerate(embeddings):
            py_idx._insert_(v, i)
            if (i + 1) % 1000 == 0:
                print(f"    inserted {i+1}/{len(embeddings)}", flush=True)
        collect_training_data(
            py_idx, xlearn, gt_learn,
            k=k, efSearch=efC,
            output_path=str(csv_path),
            logging_interval=50, verbose=True,
        )
        del py_idx

    print(f"  Training LightGBM predictor …")
    train_predictor(
        input_csv=str(csv_path),
        output_model=str(model_path),
        n_estimators=100,
        val_fraction=0.1,
        verbose=True,
    )
    return LGBMPredictor(str(model_path))


def make_retrieve_fn(idx, predictor, docs, k, ef, Rt):
    xq_buf = np.empty((1,), dtype=np.float32)  # resized per query

    def retrieve(question: str):
        q_emb = embed(question)
        xq = np.ascontiguousarray(q_emb[np.newaxis, :], dtype=np.float32)

        t0 = time.perf_counter()
        _, I = idx.search_darth(
            xq,
            k=int(k),
            efSearch=int(ef),
            Rt=float(Rt),
            predictor=predictor,
            ipi=int(ef),
            mpi=20,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        indices = I[0].tolist()
        chunks = [docs[i] for i in indices if 0 <= i < len(docs)]
        return chunks, elapsed_ms

    return retrieve


def main():
    parser = argparse.ArgumentParser(description="DARTH RAG chatbot")
    parser.add_argument("--Rt", type=float, default=0.90,
                        help="Target recall threshold (default: 0.90)")
    parser.add_argument("--ef", type=int, default=100,
                        help="efSearch (default: 100)")
    parser.add_argument("--M",  type=int, default=16)
    parser.add_argument("--efC", type=int, default=200)
    args = parser.parse_args()

    print_header(f"DARTH (Rt={args.Rt}, efSearch={args.ef})")

    docs = load_documents()
    print(f"  Loaded {len(docs)} document chunks from context/\n")

    print("  Embedding corpus …")
    embeddings = embed_corpus(docs, verbose=True)
    print(f"  Corpus shape: {embeddings.shape}\n")

    model_dir = ROOT / "predictor_models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"realworld_darth_M{args.M}_efC{args.efC}_k{TOP_K}.txt"
    csv_path   = model_dir / f"train_data_realworld_M{args.M}_efC{args.efC}_k{TOP_K}.csv"

    idx = build_index(docs, embeddings, args.M, args.efC, "l2")
    predictor = train_predictor_from_corpus(
        embeddings, TOP_K, args.efC, model_path, csv_path
    )

    retrieve_fn = make_retrieve_fn(idx, predictor, docs, TOP_K, args.ef, args.Rt)
    chat_loop(retrieve_fn)


if __name__ == "__main__":
    main()
