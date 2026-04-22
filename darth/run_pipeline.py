"""
DARTH end-to-end pipeline for siftsmall.

Stages
------
1. Load siftsmall (base, learn, query, groundtruth)
2. Compute exact ground-truth for the learn set via brute-force L2 search
3. Build HNSW_DARTH index on base vectors
4. Collect training data  →  predictor_models/train_data.csv
5. Train LightGBM model   →  predictor_models/siftsmall_M{M}_efC{efC}_k{k}.txt
6. Evaluate on test queries: baseline HNSW vs DARTH recall + timing

Usage
-----
    python darth/run_pipeline.py                  # all defaults
    python darth/run_pipeline.py --Rt 0.90 --k 10
"""

import argparse
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
# Helpers
# ---------------------------------------------------------------------------

def exact_gt(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """Brute-force exact L2 ground truth: returns (nq, k) index array."""
    print(f"  Computing exact GT for {len(xq)} queries vs {len(xb)} base vectors …")
    gt = np.empty((len(xq), k), dtype=np.int32)
    # batch to avoid huge memory allocations
    batch = 500
    for start in range(0, len(xq), batch):
        end = min(start + batch, len(xq))
        # squared L2 via ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
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
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    M: int = 16,
    efC: int = 200,
    efSearch: int = 200,
    k: int = 10,
    Rt: float = 0.90,
    ipi: int = 200,
    mpi: int = 20,
    n_estimators: int = 100,
    logging_interval: int = 2,
    max_train_queries: int = 5000,
) -> None:

    BASE  = "Datasets/sift/sift_base.fvecs"
    LEARN = "Datasets/sift/sift_learn.fvecs"
    QUERY = "Datasets/sift/sift_query.fvecs"
    GT    = "Datasets/sift/sift_groundtruth.ivecs"

    MODEL_DIR = "predictor_models"
    TRAIN_CSV = f"{MODEL_DIR}/train_data_M{M}_efC{efC}_k{k}.csv"
    MODEL_TXT = f"{MODEL_DIR}/sift_M{M}_efC{efC}_k{k}.txt"

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n── Stage 1: Load data ──────────────────────────────────────────")
    xb    = read_fvecs(BASE)
    xlearn = read_fvecs(LEARN)[:max_train_queries]
    xq    = read_fvecs(QUERY)
    gt_q  = read_ivecs(GT)
    print(f"  base={xb.shape}  learn={xlearn.shape}  query={xq.shape}  gt={gt_q.shape}")

    # ── 2. Exact GT for learn set ────────────────────────────────────────────
    print("\n── Stage 2: Exact GT for learn set ────────────────────────────")
    gt_learn = exact_gt(xb, xlearn, k)
    print(f"  gt_learn={gt_learn.shape}")

    # ── 3. Build index ───────────────────────────────────────────────────────
    print("\n── Stage 3: Build HNSW index ───────────────────────────────────")
    hnsw = build_index(xb, M=M, efC=efC)

    # ── 4. Collect training data ─────────────────────────────────────────────
    print("\n── Stage 4: Collect training data ──────────────────────────────")
    if os.path.exists(TRAIN_CSV):
        print(f"  CSV already exists, skipping collection: {TRAIN_CSV}")
    else:
        collect_training_data(
            hnsw, xlearn, gt_learn,
            k=k,
            efSearch=efSearch,
            output_path=TRAIN_CSV,
            logging_interval=logging_interval,
            verbose=True,
        )

    # ── 5. Train predictor ───────────────────────────────────────────────────
    print("\n── Stage 5: Train LightGBM predictor ───────────────────────────")
    if os.path.exists(MODEL_TXT):
        print(f"  Model already exists, skipping training: {MODEL_TXT}")
    else:
        train_predictor(
            input_csv=TRAIN_CSV,
            output_model=MODEL_TXT,
            n_estimators=n_estimators,
            val_fraction=0.1,
            verbose=True,
        )

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    print("\n── Stage 6: Evaluate on test queries ───────────────────────────")
    predictor = LGBMPredictor(MODEL_TXT)

    print(f"  Running baseline (efSearch={efSearch}) …")
    I_base, t_base = search_baseline(hnsw, xq, k, efSearch)
    r_base = recall_at_k(I_base, gt_q, k)

    print(f"  Running DARTH (Rt={Rt}, ipi={ipi}, mpi={mpi}) …")
    I_darth, t_darth = search_darth(hnsw, xq, k, efSearch, Rt, predictor, ipi, mpi)
    r_darth = recall_at_k(I_darth, gt_q, k)

    nq = len(xq)
    print()
    print("═" * 50)
    print(f"  Queries         : {nq}")
    print(f"  k               : {k}")
    print(f"  Target recall   : {Rt}")
    print()
    print(f"  Baseline recall : {r_base:.4f}   QPS: {nq/t_base:.1f}")
    print(f"  DARTH recall    : {r_darth:.4f}   QPS: {nq/t_darth:.1f}")
    speedup = t_base / t_darth if t_darth > 0 else float("inf")
    print(f"  Speedup         : {speedup:.2f}x")
    print("═" * 50)
    print(f"\n  Model saved to  : {MODEL_TXT}")
    print(f"  Training CSV    : {TRAIN_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARTH end-to-end pipeline on siftsmall.")
    parser.add_argument("--M",           type=int,   default=16)
    parser.add_argument("--efC",         type=int,   default=200)
    parser.add_argument("--efSearch",    type=int,   default=200)
    parser.add_argument("--k",           type=int,   default=10)
    parser.add_argument("--Rt",          type=float, default=0.90)
    parser.add_argument("--ipi",         type=int,   default=200)
    parser.add_argument("--mpi",         type=int,   default=20)
    parser.add_argument("--n-estimators",type=int,   default=100)
    parser.add_argument("--logging-interval", type=int, default=2)
    parser.add_argument("--max-train-queries", type=int, default=5000)
    args = parser.parse_args()

    run(
        M=args.M, efC=args.efC, efSearch=args.efSearch,
        k=args.k, Rt=args.Rt, ipi=args.ipi, mpi=args.mpi,
        n_estimators=args.n_estimators,
        logging_interval=args.logging_interval,
        max_train_queries=args.max_train_queries,
    )
