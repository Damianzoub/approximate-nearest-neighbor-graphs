"""
DARTH end-to-end pipeline for siftsmall.

Stages
------
1. Load dataset (base, learn, query, groundtruth)
2. Compute exact ground-truth for the learn set via brute-force L2 search
3. Build HNSW_DARTH index on base vectors
4. Collect training data  →  predictor_models/train_data.csv
5. Train LightGBM model   →  predictor_models/<dataset>_M{M}_efC{efC}_k{k}.txt
6. Evaluate on test queries: baseline HNSW vs DARTH recall + timing

Supports sweeping over multiple M, efC, and k values.  The index is rebuilt
only when M or efC changes, so k-sweeps within the same (M, efC) pair are cheap.

Usage
-----
    python darth/run_pipeline.py                          # all defaults
    python darth/run_pipeline.py --M 8 16 32 --efC 128 200 --k 10 20
    python darth/run_pipeline.py --dataset msmarco_text --M 16 --efC 200 --k 10 20
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.read_files import read_fvecs, read_ivecs
from hsnw_constructionDARTH import HNSW_DARTH
from darth.collect_training_data import collect_training_data
from darth.train_predictor import train as train_predictor
from darth.predictor import LGBMPredictor


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "fashion_mnist": {
        "base":  "Datasets/fashion_mnist/fashion_mnist_base.fvecs",
        "learn": "Datasets/fashion_mnist/fashion_mnist_learn.fvecs",
        "query": "Datasets/fashion_mnist/fashion_mnist_query.fvecs",
        "gt":    "Datasets/fashion_mnist/fashion_mnist_groundtruth.ivecs",
    },
    "msmarco_text": {
        "base":  "Datasets/msmarco_text/msmarco_base.fvecs",
        "learn": "Datasets/msmarco_text/msmarco_learn.fvecs",
        "query": "Datasets/msmarco_text/msmarco_query.fvecs",
        "gt":    "Datasets/msmarco_text/msmarco_groundtruth.ivecs",
    },
    "sift128": {
        "base":  "Datasets/sift128ann/sift128_base.fvecs",
        "learn": "Datasets/sift128ann/sift128_learn.fvecs",
        "query": "Datasets/sift128ann/sift128_query.fvecs",
        "gt":    "Datasets/sift128ann/sift128_groundtruth.ivecs",
    },
    "glove100": {
        "base":  "Datasets/glove100ann/glove100_base.fvecs",
        "learn": "Datasets/glove100ann/glove100_learn.fvecs",
        "query": "Datasets/glove100ann/glove100_query.fvecs",
        "gt":    "Datasets/glove100ann/glove100_groundtruth.ivecs",
    },
    "siftsmall": {
        "base":  "Datasets/siftsmall/siftsmall_base.fvecs",
        "learn": "Datasets/siftsmall/siftsmall_learn.fvecs",
        "query": "Datasets/siftsmall/siftsmall_query.fvecs",
        "gt":    "Datasets/siftsmall/siftsmall_groundtruth.ivecs",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exact_gt(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """Brute-force exact L2 ground truth: returns (nq, k) index array."""
    print(f"  Computing exact GT for {len(xq)} queries vs {len(xb)} base vectors …")
    gt = np.empty((len(xq), k), dtype=np.int32)
    batch = 500
    for start in range(0, len(xq), batch):
        end = min(start + batch, len(xq))
        q_chunk = xq[start:end].astype(np.float32)
        diffs = (
            (q_chunk ** 2).sum(1, keepdims=True)
            + (xb ** 2).sum(1)
            - 2.0 * (q_chunk @ xb.T)
        )
        gt[start:end] = np.argsort(diffs, axis=1)[:, :k]
    return gt


def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = sum(
        len(set(I[i, :k].tolist()) & set(gt[i, :k].tolist()))
        for i in range(len(I))
    )
    return hits / (len(I) * k)


def build_index(xb: np.ndarray, M: int, efC: int, metric: str = "l2") -> HNSW_DARTH:
    print(f"  Building HNSW_DARTH index (M={M}, efC={efC}, n={len(xb)}) …")
    t0 = time.time()
    hnsw = HNSW_DARTH(dim=xb.shape[1], M=M, efConstruction=efC, metric=metric)
    for i, v in enumerate(xb):
        hnsw._insert_(v, i)
        if (i + 1) % 2000 == 0:
            print(f"    inserted {i+1}/{len(xb)}", flush=True)
    print(f"  Index built in {time.time()-t0:.1f}s")
    return hnsw


def search_baseline(hnsw: HNSW_DARTH, xq: np.ndarray, k: int, efSearch: int):
    t0 = time.time()
    D, I = hnsw.search_darth(xq, k=k, efSearch=efSearch, Rt=0.0,
                              predictor=None, ipi=efSearch+1, mpi=efSearch+1)
    elapsed = time.time() - t0
    return I, elapsed


def search_darth(hnsw: HNSW_DARTH, xq: np.ndarray, k: int, efSearch: int,
                 Rt: float, predictor, ipi: int, mpi: int):
    t0 = time.time()
    D, I = hnsw.search_darth(xq, k=k, efSearch=efSearch, Rt=Rt,
                              predictor=predictor, ipi=ipi, mpi=mpi)
    elapsed = time.time() - t0
    return I, elapsed


# ---------------------------------------------------------------------------
# Single (M, efC, k) run — index must already be built
# ---------------------------------------------------------------------------

def run_one(
    hnsw: HNSW_DARTH,
    dataset_name: str,
    xlearn: np.ndarray,
    gt_learn: np.ndarray,
    xq: np.ndarray,
    gt_q: np.ndarray,
    M: int,
    efC: int,
    efSearch: int,
    k: int,
    Rt: float,
    ipi: int,
    mpi: int,
    n_estimators: int,
    logging_interval: int,
    model_dir: str,
    verbose: bool = True,
) -> dict:
    TRAIN_CSV = f"{model_dir}/train_data_{dataset_name}_M{M}_efC{efC}_k{k}.csv"
    MODEL_TXT = f"{model_dir}/{dataset_name}_M{M}_efC{efC}_k{k}.txt"

    # ── 4. Collect training data ─────────────────────────────────────────────
    if verbose:
        print(f"\n── Stage 4: Collect training data (M={M}, efC={efC}, k={k}) ──")
    if os.path.exists(TRAIN_CSV):
        if verbose:
            print(f"  CSV exists, skipping: {TRAIN_CSV}")
    else:
        collect_training_data(
            hnsw, xlearn, gt_learn,
            k=k,
            efSearch=efSearch,
            output_path=TRAIN_CSV,
            logging_interval=logging_interval,
            verbose=verbose,
        )

    # ── 5. Train predictor ───────────────────────────────────────────────────
    if verbose:
        print(f"\n── Stage 5: Train LightGBM predictor ───────────────────────────")
    if os.path.exists(MODEL_TXT):
        if verbose:
            print(f"  Model exists, skipping: {MODEL_TXT}")
    else:
        train_predictor(
            input_csv=TRAIN_CSV,
            output_model=MODEL_TXT,
            n_estimators=n_estimators,
            val_fraction=0.1,
            verbose=verbose,
        )

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n── Stage 6: Evaluate (M={M}, efC={efC}, k={k}) ─────────────────")
    predictor = LGBMPredictor(MODEL_TXT)

    if verbose:
        print(f"  Running baseline (efSearch={efSearch}) …")
    I_base, t_base = search_baseline(hnsw, xq, k, efSearch)
    r_base = recall_at_k(I_base, gt_q, k)

    if verbose:
        print(f"  Running DARTH (Rt={Rt}, ipi={ipi}, mpi={mpi}) …")
    I_darth, t_darth = search_darth(hnsw, xq, k, efSearch, Rt, predictor, ipi, mpi)
    r_darth = recall_at_k(I_darth, gt_q, k)

    nq = len(xq)
    speedup = t_base / t_darth if t_darth > 0 else float("inf")

    return dict(
        dataset=dataset_name, M=M, efC=efC, k=k, nq=nq,
        r_base=r_base, qps_base=nq/t_base,
        r_darth=r_darth, qps_darth=nq/t_darth,
        speedup=speedup,
        model=MODEL_TXT, csv=TRAIN_CSV,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    dataset: str = "fashion_mnist",
    M_list: list = None,
    efC_list: list = None,
    efSearch: int = 200,
    k_list: list = None,
    Rt: float = 0.90,
    ipi: int = 200,
    mpi: int = 20,
    n_estimators: int = 100,
    logging_interval: int = 2,
    max_train_queries: int = 5000,
) -> None:
    if M_list is None:
        M_list = [16]
    if efC_list is None:
        efC_list = [200]
    if k_list is None:
        k_list = [10]

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}")

    paths = DATASETS[dataset]
    MODEL_DIR = "predictor_models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n── Stage 1: Load data ──────────────────────────────────────────")
    xb     = read_fvecs(paths["base"])
    xlearn = read_fvecs(paths["learn"])[:max_train_queries]
    xq     = read_fvecs(paths["query"])
    gt_q   = read_ivecs(paths["gt"])
    print(f"  base={xb.shape}  learn={xlearn.shape}  query={xq.shape}  gt={gt_q.shape}")

    # ── 2. Exact GT for learn set (use largest k needed) ────────────────────
    k_max = max(k_list)
    print(f"\n── Stage 2: Exact GT for learn set (k={k_max}) ────────────────")
    gt_learn = exact_gt(xb, xlearn, k_max)
    print(f"  gt_learn={gt_learn.shape}")

    all_results = []

    # Sweep over (M, efC) pairs — rebuild index only when they change
    for M, efC in itertools.product(M_list, efC_list):
        print(f"\n{'═'*60}")
        print(f"  Building index for M={M}, efC={efC} …")
        print(f"{'═'*60}")
        hnsw = build_index(xb, M=M, efC=efC)

        # Sweep over k within the same index
        for k in k_list:
            # Slice gt_learn to current k
            gt_learn_k = gt_learn[:, :k]
            result = run_one(
                hnsw=hnsw,
                dataset_name=dataset,
                xlearn=xlearn,
                gt_learn=gt_learn_k,
                xq=xq,
                gt_q=gt_q,
                M=M, efC=efC,
                efSearch=efSearch,
                k=k,
                Rt=Rt, ipi=ipi, mpi=mpi,
                n_estimators=n_estimators,
                logging_interval=logging_interval,
                model_dir=MODEL_DIR,
                verbose=True,
            )
            all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*78}")
    print(f"  SWEEP SUMMARY  —  dataset={dataset}  efSearch={efSearch}  Rt={Rt}")
    print(f"{'═'*78}")
    hdr = f"{'M':>4} {'efC':>5} {'k':>3} | {'Base R@k':>8} {'Base QPS':>9} | {'DARTH R@k':>9} {'DARTH QPS':>10} | {'Speedup':>7}"
    print(hdr)
    print("─" * len(hdr))
    for r in all_results:
        print(
            f"{r['M']:>4} {r['efC']:>5} {r['k']:>3} | "
            f"{r['r_base']:>8.4f} {r['qps_base']:>9.1f} | "
            f"{r['r_darth']:>9.4f} {r['qps_darth']:>10.1f} | "
            f"{r['speedup']:>7.2f}x"
        )
    print(f"{'═'*78}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DARTH end-to-end pipeline with optional sweep over M, efC, k."
    )
    parser.add_argument("--dataset",    type=str,   default="fashion_mnist",
                        choices=list(DATASETS),
                        help="Dataset to use (default: fashion_mnist)")
    parser.add_argument("--M",          type=int,   nargs="+", default=[16],
                        metavar="M",
                        help="One or more M values (default: 16)")
    parser.add_argument("--efC",        type=int,   nargs="+", default=[200],
                        metavar="efC",
                        help="One or more efConstruction values (default: 200)")
    parser.add_argument("--efSearch",   type=int,   default=200)
    parser.add_argument("--k",          type=int,   nargs="+", default=[10],
                        metavar="k",
                        help="One or more k values, e.g. --k 10 20 (default: 10)")
    parser.add_argument("--Rt",         type=float, default=0.90)
    parser.add_argument("--ipi",        type=int,   default=200)
    parser.add_argument("--mpi",        type=int,   default=20)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--logging-interval", type=int, default=2)
    parser.add_argument("--max-train-queries", type=int, default=5000)
    args = parser.parse_args()

    run(
        dataset=args.dataset,
        M_list=args.M,
        efC_list=args.efC,
        efSearch=args.efSearch,
        k_list=args.k,
        Rt=args.Rt,
        ipi=args.ipi,
        mpi=args.mpi,
        n_estimators=args.n_estimators,
        logging_interval=args.logging_interval,
        max_train_queries=args.max_train_queries,
    )
