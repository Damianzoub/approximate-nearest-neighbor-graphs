"""
Download full GloVe-100 from ann-benchmarks and convert to fvecs/ivecs.

Steps:
  1. Download glove-100-angular.hdf5 (~485 MB)
  2. Extract train (1.18M x 100) as base vectors
  3. Extract test (10K x 100) as query vectors
  4. Take first 5000 train vectors as learn set
  5. Compute L2 ground truth (k=100) with FAISS
  6. Write fvecs/ivecs files to Datasets/glove100_full/

Usage:
    python scripts/prepare_glove100_full.py
"""

import struct
import sys
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "Datasets" / "glove100_full"
HDF5   = OUTDIR / "glove-100-angular.hdf5"
URL    = "https://ann-benchmarks.com/glove-100-angular.hdf5"

N_LEARN = 5000
K_GT    = 100


def write_fvecs(path: Path, X: np.ndarray):
    X = X.astype(np.float32)
    n, d = X.shape
    with open(path, "wb") as f:
        for row in X:
            f.write(struct.pack("i", d))
            f.write(row.tobytes())
    print(f"  wrote {path.name}  ({n} x {d})")


def write_ivecs(path: Path, X: np.ndarray):
    X = X.astype(np.int32)
    n, d = X.shape
    with open(path, "wb") as f:
        for row in X:
            f.write(struct.pack("i", d))
            f.write(row.tobytes())
    print(f"  wrote {path.name}  ({n} x {d})")


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded")
        return
    print(f"  Downloading {url} ...")
    t0 = time.perf_counter()

    def _progress(block, bsize, total):
        done = block * bsize
        pct  = min(done / total * 100, 100) if total > 0 else 0
        mb   = done / 1e6
        sys.stdout.write(f"\r  {pct:.1f}%  {mb:.0f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n  Done in {time.perf_counter() - t0:.0f}s")


def compute_gt_l2(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    import faiss
    print(f"  Computing L2 ground truth (n={len(xq)}, k={k}) with FAISS ...")
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb.astype(np.float32))
    _, I = index.search(xq.astype(np.float32), k)
    print(f"  GT done in {time.perf_counter() - t0:.1f}s")
    return I.astype(np.int32)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    download(URL, HDF5)

    print("  Loading HDF5 ...")
    with h5py.File(HDF5, "r") as f:
        xb = f["train"][:]   # (1183514, 100)
        xq = f["test"][:]    # (10000, 100)

    xb = xb.astype(np.float32)
    xq = xq.astype(np.float32)
    print(f"  base={xb.shape}  query={xq.shape}")

    xl = xb[:N_LEARN]

    gt_path = OUTDIR / "glove100_groundtruth.ivecs"
    if gt_path.exists():
        print(f"  [skip] groundtruth already exists")
    else:
        gt = compute_gt_l2(xb, xq, K_GT)
        write_ivecs(gt_path, gt)

    write_fvecs(OUTDIR / "glove100_base.fvecs",  xb)
    write_fvecs(OUTDIR / "glove100_query.fvecs", xq)
    write_fvecs(OUTDIR / "glove100_learn.fvecs", xl)

    print("\n  All done.")


if __name__ == "__main__":
    main()
