"""
Convert ann-benchmarks HDF5 files → .fvecs / .ivecs for the DARTH pipeline.

Supports:
  sift128   sift-128-euclidean.hdf5   (L2, 128-dim, 1M train, 10K test)
  glove100  glove-100-angular.hdf5    (cosine, 100-dim, 1.18M train, 10K test)

HDF5 layout (ann-benchmarks standard):
  train      → base vectors  (n_train, dim)
  test       → query vectors (10 000, dim)
  neighbors  → ground truth  (10 000, 100)  — indices into train

Output (written to Datasets/<name>/):
  <name>_base.fvecs          first --max-base vectors of train
  <name>_learn.fvecs         next --n-learn vectors (NOT in base)
  <name>_query.fvecs         test vectors
  <name>_groundtruth.ivecs   top-100 GT for query against the base subset

For glove100 the vectors are L2-normalised before writing so that L2 search
is equivalent to cosine similarity (||a-b||² = 2 - 2·cos(a,b) for unit vectors).

Usage
-----
    # prepare both (defaults)
    OMP_NUM_THREADS=1 python scripts/prepare_ann_hdf5.py --dataset sift128 glove100

    # smaller base for quick testing
    OMP_NUM_THREADS=1 python scripts/prepare_ann_hdf5.py --dataset sift128 --max-base 100000
"""

import argparse
from pathlib import Path

import faiss
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset registry ──────────────────────────────────────────────────────────
REGISTRY = {
    "sift128": {
        "hdf5":    PROJECT_ROOT / "Dataset" / "sift128ann"  / "sift-128-euclidean.hdf5",
        "out_dir": PROJECT_ROOT / "Datasets" / "sift128ann",
        "metric":  "l2",
        "normalize": False,
    },
    "glove100": {
        "hdf5":    PROJECT_ROOT / "Dataset" / "glove100ann" / "glove-100-angular.hdf5",
        "out_dir": PROJECT_ROOT / "Datasets" / "glove100ann",
        "metric":  "cosine",
        "normalize": True,   # L2-normalise so L2 dist ≡ cosine
    },
}


# ── fvecs / ivecs writers ─────────────────────────────────────────────────────

def write_fvecs(path: Path, data: np.ndarray) -> None:
    data = data.astype("float32")
    n, d = data.shape
    out = np.empty((n, d + 1), dtype="int32")
    out[:, 0] = d
    out[:, 1:] = data.view("int32")
    out.tofile(path)
    print(f"  wrote {path.name}  {data.shape}  ({path.stat().st_size/1e6:.1f} MB)")


def write_ivecs(path: Path, data: np.ndarray) -> None:
    data = data.astype("int32")
    n, d = data.shape
    out = np.empty((n, d + 1), dtype="int32")
    out[:, 0] = d
    out[:, 1:] = data
    out.tofile(path)
    print(f"  wrote {path.name}  {data.shape}  ({path.stat().st_size/1e6:.1f} MB)")


# ── Ground truth via faiss ────────────────────────────────────────────────────

def compute_gt(base: np.ndarray, query: np.ndarray, k: int, normalize: bool) -> np.ndarray:
    d = base.shape[1]
    if normalize:
        index = faiss.IndexFlatIP(d)   # inner product on unit vectors = cosine
    else:
        index = faiss.IndexFlatL2(d)
    print(f"  Building faiss index ({len(base):,} vectors, dim={d}) …")
    index.add(base)
    print(f"  Searching top-{k} for {len(query):,} queries …")
    _, I = index.search(query, k)
    return I.astype("int32")


# ── Main preparation ──────────────────────────────────────────────────────────

def prepare(name: str, max_base: int, n_learn: int, top_k: int) -> None:
    cfg = REGISTRY[name]
    hdf5_path = cfg["hdf5"]
    out_dir   = cfg["out_dir"]
    normalize = cfg["normalize"]

    if not hdf5_path.exists():
        print(f"ERROR: {hdf5_path} not found.")
        print(f"  Run:  python scripts/download_all_ann.py --only {name}")
        raise SystemExit(1)

    print(f"\n{'═'*60}")
    print(f"  Preparing {name}  (normalize={normalize})")
    print(f"{'═'*60}")

    # ── Load from HDF5 ────────────────────────────────────────────────────────
    with h5py.File(hdf5_path) as f:
        train_full = f["train"][:]       # (n_train, dim)
        test_full  = f["test"][:]        # (10000,   dim)
        # HDF5 neighbors are valid only for full train — we'll recompute for subset

    train_full = train_full.astype("float32")
    test_full  = test_full.astype("float32")

    print(f"  Loaded: train={train_full.shape}  test={test_full.shape}")

    # ── Normalise for cosine datasets ─────────────────────────────────────────
    if normalize:
        norms = np.linalg.norm(train_full, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        train_full /= norms
        norms = np.linalg.norm(test_full, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        test_full /= norms
        print("  Vectors L2-normalised (cosine → inner product search)")

    # ── Split: base | learn  (non-overlapping) ────────────────────────────────
    n_train = len(train_full)
    actual_base  = min(max_base, n_train - n_learn)
    actual_learn = min(n_learn, n_train - actual_base)

    base  = train_full[:actual_base]
    learn = train_full[actual_base : actual_base + actual_learn]

    print(f"  Split: {len(base):,} base  +  {len(learn):,} learn  +  {len(test_full):,} query")

    # ── Ground truth ──────────────────────────────────────────────────────────
    print(f"\n  Computing GT for query set (top-{top_k}) …")
    gt_query = compute_gt(base, test_full, top_k, normalize)

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Writing to {out_dir}/")
    write_fvecs(out_dir / f"{name}_base.fvecs",  base)
    write_fvecs(out_dir / f"{name}_learn.fvecs", learn)
    write_fvecs(out_dir / f"{name}_query.fvecs", test_full)
    write_ivecs(out_dir / f"{name}_groundtruth.ivecs", gt_query)

    print(f"\n  Done — {name} ready in {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert ann-benchmarks HDF5 → fvecs/ivecs.")
    parser.add_argument("--dataset", nargs="+", default=["sift128", "glove100"],
                        choices=list(REGISTRY),
                        help="Dataset(s) to prepare (default: sift128 glove100)")
    parser.add_argument("--max-base", type=int, default=200_000,
                        help="Max base vectors to use (default: 200000). "
                             "Use a smaller value for quick tests.")
    parser.add_argument("--n-learn", type=int, default=10_000,
                        help="Learn vectors taken from train after base (default: 10000)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of ground-truth neighbours (default: 100)")
    args = parser.parse_args()

    for name in args.dataset:
        prepare(name, max_base=args.max_base, n_learn=args.n_learn, top_k=args.top_k)

    print("\n=== All datasets prepared ===")


if __name__ == "__main__":
    main()
