"""
Embed MS MARCO passages and save as .fvecs/.ivecs for ANN benchmarking.

Pipeline:
  1. Read passages from passages.jsonl (produced by download_msmarco_text.py)
  2. Embed with sentence-transformers all-MiniLM-L6-v2  (384-dim, L2 metric)
  3. Split 50k passages → 40k base / 5k learn / 5k query
  4. Compute exact top-100 ground truth (query vs base) via faiss IndexFlatL2
  5. Write msmarco_base.fvecs / msmarco_learn.fvecs /
             msmarco_query.fvecs / msmarco_groundtruth.ivecs

The learn set is used by run_pipeline.py to collect DARTH predictor training data.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH   = PROJECT_ROOT / "Dataset" / "msmarco_text" / "passages.jsonl"
OUT_DIR      = PROJECT_ROOT / "Datasets" / "msmarco_text"

NUM_BASE  = 40_000
NUM_LEARN =  5_000
NUM_QUERY =  5_000
TOP_K     =    100
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# I/O helpers (mirror of read_fvecs / read_ivecs but for writing)
# ---------------------------------------------------------------------------

def write_fvecs(path: Path, data: np.ndarray) -> None:
    data = data.astype("float32")
    n, d = data.shape
    header = np.full((n, 1), d, dtype="int32")
    interleaved = np.hstack([header.view("float32"), data])
    interleaved.tofile(path)


def write_ivecs(path: Path, data: np.ndarray) -> None:
    data = data.astype("int32")
    n, d = data.shape
    header = np.full((n, 1), d, dtype="int32")
    interleaved = np.hstack([header, data])
    interleaved.tofile(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_passages(limit: int) -> list[str]:
    passages = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            if len(passages) >= limit:
                break
            obj = json.loads(line)
            text = obj.get("passage", "").strip()
            if text:
                passages.append(text)
    return passages


def embed(passages: list[str], model: SentenceTransformer) -> np.ndarray:
    print(f"Embedding {len(passages):,} passages (batch={BATCH_SIZE})...")
    vecs = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vecs.astype("float32")


def compute_groundtruth(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    print(f"Computing exact top-{k} ground truth with faiss IndexFlatL2...")
    d = base.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(base)
    _, I = index.search(query, k)
    return I.astype("int32")


def main() -> None:
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found. Run download_msmarco_text.py first.")
        sys.exit(1)

    total_needed = NUM_BASE + NUM_LEARN + NUM_QUERY
    print(f"Loading up to {total_needed:,} passages from {JSONL_PATH}...")
    passages = load_passages(total_needed)

    if len(passages) < total_needed:
        print(
            f"WARNING: only {len(passages):,} passages available "
            f"(need {total_needed:,}). Adjusting split proportionally."
        )
        n = len(passages)
        num_base  = int(n * 0.80)
        num_learn = int(n * 0.10)
        num_query = n - num_base - num_learn
    else:
        num_base, num_learn, num_query = NUM_BASE, NUM_LEARN, NUM_QUERY

    print(f"Split: {num_base:,} base  +  {num_learn:,} learn  +  {num_query:,} query")

    model = SentenceTransformer(MODEL_NAME)
    all_vecs = embed(passages, model)

    base  = all_vecs[:num_base]
    learn = all_vecs[num_base : num_base + num_learn]
    query = all_vecs[num_base + num_learn : num_base + num_learn + num_query]

    groundtruth = compute_groundtruth(base, query, TOP_K)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_path  = OUT_DIR / "msmarco_base.fvecs"
    learn_path = OUT_DIR / "msmarco_learn.fvecs"
    query_path = OUT_DIR / "msmarco_query.fvecs"
    gt_path    = OUT_DIR / "msmarco_groundtruth.ivecs"

    print(f"Writing {base_path} ...")
    write_fvecs(base_path, base)

    print(f"Writing {learn_path} ...")
    write_fvecs(learn_path, learn)

    print(f"Writing {query_path} ...")
    write_fvecs(query_path, query)

    print(f"Writing {gt_path} ...")
    write_ivecs(gt_path, groundtruth)

    print("\nDone.")
    print(f"  base       : {base.shape}   → {base_path}")
    print(f"  learn      : {learn.shape}   → {learn_path}")
    print(f"  query      : {query.shape}   → {query_path}")
    print(f"  groundtruth: {groundtruth.shape} → {gt_path}")


if __name__ == "__main__":
    main()
