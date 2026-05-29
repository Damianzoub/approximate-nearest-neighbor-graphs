"""
Train DARTH predictors for multiple (dataset, M, efC, k) combinations.

Skips any combination whose model file already exists.
Predictors are saved to predictor_models/{dataset}_M{M}_efC{efC}_k{k}.txt

Usage
-----
    # Train all missing predictors with defaults
    python experiments/train_predictors.py

    # Specific dataset / sweep
    python experiments/train_predictors.py --datasets siftsmall --M 8 16 --efC 128 200 --k 10 20

    # All three datasets, full sweep
    python experiments/train_predictors.py --datasets siftsmall fashion_mnist msmarco --M 8 16 --efC 128 200 --k 10 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.read_files import read_fvecs
from darth.collect_training_data_cpp import collect_training_data_cpp
from darth.train_predictor import train as train_predictor

# ── dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "siftsmall": {
        "base":  "Datasets/siftsmall/siftsmall_base.fvecs",
        "learn": "Datasets/siftsmall/siftsmall_learn.fvecs",
        "metric": "l2",
        "max_learn": 5000,
        "default_M":   [8, 16],
        "default_efC": [128, 200],
    },
    "fashion_mnist": {
        "base":  "Datasets/fashion_mnist/fashion_mnist_base.fvecs",
        "learn": "Datasets/fashion_mnist/fashion_mnist_learn.fvecs",
        "metric": "l2",
        "max_learn": 3000,
        "default_M":   [16],
        "default_efC": [200],
    },
    "msmarco_text": {
        "base":  "Datasets/msmarco_text/msmarco_base.fvecs",
        "learn": "Datasets/msmarco_text/msmarco_learn.fvecs",
        "metric": "l2",
        "max_learn": 3000,
        "default_M":   [16],
        "default_efC": [200],
    },
    # "sift128": {   # cut-down version (200K of 1M) — excluded
    #     "base":  "Datasets/sift128ann/sift128_base.fvecs",
    #     "learn": "Datasets/sift128ann/sift128_learn.fvecs",
    #     "metric": "l2",
    #     "max_learn": 5000,
    #     "default_M":   [16],
    #     "default_efC": [200],
    # },
    # "glove100": {  # cut-down version (200K of 1.18M) — excluded
    #     "base":  "Datasets/glove100ann/glove100_base.fvecs",
    #     "learn": "Datasets/glove100ann/glove100_learn.fvecs",
    #     "metric": "l2",
    #     "max_learn": 5000,
    #     "default_M":   [16],
    #     "default_efC": [200],
    # },
    "glove100_full": {
        "base":  "Datasets/glove100_full/glove100_base.fvecs",
        "learn": "Datasets/glove100_full/glove100_learn.fvecs",
        "metric": "l2",
        "max_learn": 5000,
        "default_M":   [8, 16],
        "default_efC": [128, 200],
    },
    "sift1m": {
        "base":  "Datasets/sift/sift_base.fvecs",
        "learn": "Datasets/sift/sift_learn.fvecs",
        "metric": "l2",
        "max_learn": 5000,
        "default_M":   [8, 16],
        "default_efC": [128, 200],
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def exact_gt_faiss(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    try:
        import faiss
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb.astype(np.float32))
        _, I = index.search(xq.astype(np.float32), k)
        return I.astype(np.int32)
    except ImportError:
        print("  [warn] faiss not found — using numpy brute-force (slow)")
        gt = np.empty((len(xq), k), dtype=np.int32)
        batch = 500
        for s in range(0, len(xq), batch):
            e = min(s + batch, len(xq))
            q = xq[s:e].astype(np.float32)
            diffs = (q**2).sum(1, keepdims=True) + (xb**2).sum(1) - 2.0 * (q @ xb.T)
            gt[s:e] = np.argsort(diffs, 1)[:, :k]
        return gt




def train_one(dataset_name: str, M: int, efC: int, k: int,
              model_dir: Path, max_learn: int, dry_run: bool = False):
    cfg = DATASETS[dataset_name]

    model_path = model_dir / f"{dataset_name}_M{M}_efC{efC}_k{k}.txt"
    csv_path   = model_dir / f"train_data_{dataset_name}_M{M}_efC{efC}_k{k}.csv"

    if model_path.exists():
        print(f"  [skip] {model_path.name} already exists")
        return

    print(f"\n{'─'*55}")
    print(f"  {dataset_name}  M={M}  efC={efC}  k={k}")
    print(f"{'─'*55}")

    if dry_run:
        print("  [dry-run] would train this model")
        return

    base_path  = ROOT / cfg["base"]
    learn_path = ROOT / cfg["learn"]

    if not base_path.exists():
        print(f"  [skip] base file not found: {base_path}")
        return
    if not learn_path.exists():
        print(f"  [skip] learn file not found: {learn_path}")
        return

    xb     = read_fvecs(str(base_path))
    xlearn = read_fvecs(str(learn_path))[:max_learn]
    print(f"  base={xb.shape}  learn={xlearn.shape}")

    if not csv_path.exists():
        print(f"  Computing learn-set GT (n={len(xlearn)}) …")
        t0 = time.perf_counter()
        gt_learn = exact_gt_faiss(xb, xlearn, k)
        print(f"  GT done in {time.perf_counter()-t0:.1f}s")

        print(f"  Collecting training data → {csv_path.name}")
        collect_training_data_cpp(
            xb, xlearn, gt_learn,
            k=k, efSearch=efC,
            M=M, efConstruction=efC,
            output_path=str(csv_path),
            metric=cfg["metric"],
            ipi=5, verbose=True,
        )

    print(f"  Training predictor → {model_path.name}")
    train_predictor(
        input_csv=str(csv_path),
        output_model=str(model_path),
        n_estimators=100,
        val_fraction=0.1,
        verbose=True,
    )
    print(f"  Saved: {model_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DARTH predictors.")
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Datasets to train on (default: all)")
    parser.add_argument("--M",   nargs="+", type=int, default=None,
                        help="M values (default: per-dataset defaults)")
    parser.add_argument("--efC", nargs="+", type=int, default=None,
                        help="efConstruction values (default: per-dataset defaults)")
    parser.add_argument("--k",   nargs="+", type=int, default=[10, 20],
                        help="k values (default: 10 20)")
    parser.add_argument("--max-learn", type=int, default=None,
                        help="Max learn vectors (default: per-dataset)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be trained without doing it")
    args = parser.parse_args()

    model_dir = ROOT / "predictor_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        M_vals   = args.M   or cfg["default_M"]
        efC_vals = args.efC or cfg["default_efC"]
        k_vals   = args.k
        max_learn = args.max_learn or cfg["max_learn"]

        print(f"\n{'═'*55}")
        print(f"  Dataset : {dataset_name}")
        print(f"  M       : {M_vals}")
        print(f"  efC     : {efC_vals}")
        print(f"  k       : {k_vals}")
        print(f"  max_learn: {max_learn}")
        print(f"{'═'*55}")

        for M in M_vals:
            for efC in efC_vals:
                for k in k_vals:
                    train_one(dataset_name, M, efC, k, model_dir, max_learn, args.dry_run)
                    total += 1

    print(f"\n  All done — processed {total} (dataset, M, efC, k) combos")


if __name__ == "__main__":
    main()
