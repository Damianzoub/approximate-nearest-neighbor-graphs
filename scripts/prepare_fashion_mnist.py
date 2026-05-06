"""
Prepare Fashion-MNIST as .fvecs/.ivecs for ANN benchmarking.

Reads the raw IDX files downloaded by torchvision (no torch dependency here),
so faiss can load cleanly without OMP conflicts.

Pipeline:
  1. Read raw IDX files from Dataset/fashion_mnist/FashionMNIST/raw/
  2. Flatten 28x28 → 784-dim float32, normalise to [0, 1]
  3. Split train (60k) → 55k base + 5k learn; test (10k) → query
  4. Compute exact top-100 ground truth (query vs base) via faiss IndexFlatL2
  5. Write fashion_mnist_{base,learn,query}.fvecs + fashion_mnist_groundtruth.ivecs

Run download_fashion_mnist.py first (or just let torchvision pull the files once).
"""

import struct
import gzip
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR  = PROJECT_ROOT / "Dataset"  / "fashion_mnist" / "FashionMNIST" / "raw"
OUT_DIR  = PROJECT_ROOT / "Datasets" / "fashion_mnist"

NUM_BASE  = 55_000
NUM_LEARN =  5_000
TOP_K     =    100


def read_idx_images(path: Path) -> np.ndarray:
    """Read IDX3 image file (gz or plain)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype("float32") / 255.0


def write_fvecs(path: Path, data: np.ndarray) -> None:
    data = data.astype("float32")
    n, d = data.shape
    out = np.empty((n, d + 1), dtype="int32")
    out[:, 0] = d
    out[:, 1:] = data.view("int32")
    out.tofile(path)


def write_ivecs(path: Path, data: np.ndarray) -> None:
    data = data.astype("int32")
    n, d = data.shape
    out = np.empty((n, d + 1), dtype="int32")
    out[:, 0] = d
    out[:, 1:] = data
    out.tofile(path)


def compute_groundtruth(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    print(f"Computing exact top-{k} ground truth with faiss IndexFlatL2...")
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, I = index.search(query, k)
    return I.astype("int32")


def main() -> None:
    train_path = RAW_DIR / "train-images-idx3-ubyte"
    test_path  = RAW_DIR / "t10k-images-idx3-ubyte"

    for p in (train_path, test_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run download_fashion_mnist.py first.")
            raise SystemExit(1)

    print(f"Reading {train_path} ...")
    xb_full = read_idx_images(train_path)   # (60000, 784)
    print(f"Reading {test_path} ...")
    xq      = read_idx_images(test_path)    # (10000, 784)

    base  = xb_full[:NUM_BASE]
    learn = xb_full[NUM_BASE : NUM_BASE + NUM_LEARN]
    print(f"Split: {base.shape[0]:,} base  +  {learn.shape[0]:,} learn  +  {xq.shape[0]:,} query")

    groundtruth = compute_groundtruth(base, xq, TOP_K)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_path  = OUT_DIR / "fashion_mnist_base.fvecs"
    learn_path = OUT_DIR / "fashion_mnist_learn.fvecs"
    query_path = OUT_DIR / "fashion_mnist_query.fvecs"
    gt_path    = OUT_DIR / "fashion_mnist_groundtruth.ivecs"

    print(f"Writing {base_path} ...")
    write_fvecs(base_path, base)

    print(f"Writing {learn_path} ...")
    write_fvecs(learn_path, learn)

    print(f"Writing {query_path} ...")
    write_fvecs(query_path, xq)

    print(f"Writing {gt_path} ...")
    write_ivecs(gt_path, groundtruth)

    print("\nDone.")
    print(f"  base       : {base.shape}   → {base_path}")
    print(f"  learn      : {learn.shape}    → {learn_path}")
    print(f"  query      : {xq.shape}  → {query_path}")
    print(f"  groundtruth: {groundtruth.shape} → {gt_path}")


if __name__ == "__main__":
    main()
