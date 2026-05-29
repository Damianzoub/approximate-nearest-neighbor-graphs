"""
Fast training-data collector for the DARTH recall predictor using the C++ index.

Instead of building a slow Python HNSW, this module builds a C++ HNSWDarthIndex
and uses search_darth_collect() to log feature snapshots + top-k node IDs at
every `ipi` candidate pops.  Recall at each snapshot is computed from the
returned node IDs against the exact ground truth.

The efSearch-sized result set (as used in the fixed search_darth) is used
throughout, so the training distribution matches inference exactly.

Output CSV columns (same as collect_training_data.py):
  qid, nstep, ndis, ninserts, firstNN, closestNN, furthestNN,
  meanNN, varNN, p25NN, p50NN, p75NN, r
"""

import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hnswDarth_cpp" / "src" / "build"))

import hnswDarth_cpp


def collect_training_data_cpp(
    xb: np.ndarray,
    xq: np.ndarray,
    groundtruth: np.ndarray,
    k: int,
    efSearch: int,
    M: int,
    efConstruction: int,
    output_path: str,
    metric: str = "l2",
    ipi: int = 5,
    max_queries: Optional[int] = None,
    verbose: bool = True,
) -> None:
    """
    Build a C++ HNSW index on xb and collect DARTH training data by running
    search_darth_collect on xq.

    Parameters
    ----------
    xb : np.ndarray (n, dim)
        Base vectors.
    xq : np.ndarray (nq, dim)
        Training query vectors.
    groundtruth : np.ndarray (nq, k_gt)
        Exact k nearest-neighbour IDs for each query (k_gt >= k).
    k : int
        Number of nearest neighbours for the recall target.
    efSearch : int
        Beam width used at search time (controls result-set size).
    M, efConstruction : int
        HNSW graph parameters.
    output_path : str
        Destination CSV file.
    metric : str
        "l2" or "cosine".
    ipi : int
        Log one row every ipi candidate pops (initial prediction interval).
    max_queries : int, optional
        Cap the number of queries processed.
    verbose : bool
        Print progress.
    """
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    groundtruth = np.asarray(groundtruth, dtype=np.int32)

    nq = len(xq)
    if max_queries is not None:
        nq = min(nq, max_queries)
    xq = xq[:nq]

    if verbose:
        print(f"  Building C++ HNSW (M={M}, efC={efConstruction}, n={len(xb)}) …", flush=True)
    t0 = time.perf_counter()
    idx = hnswDarth_cpp.HNSWDarthIndex(dim=int(xb.shape[1]), M=M,
                                        efConstruction=efConstruction, metric=metric)
    idx.add(xb)
    if verbose:
        print(f"  Built in {time.perf_counter()-t0:.1f}s", flush=True)

    fieldnames = [
        "qid", "nstep", "ndis", "ninserts",
        "firstNN", "closestNN", "furthestNN",
        "meanNN", "varNN", "p25NN", "p50NN", "p75NN", "r",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_rows = 0
    t1 = time.perf_counter()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for qi in range(nq):
            gt_set = set(int(x) for x in groundtruth[qi, :k])
            pending_rows: list[dict] = []

            def _callback(feats: dict, topk_ids: list, _qi=qi, _gt=gt_set):
                r = len(set(topk_ids) & _gt) / k
                row = {
                    "qid":       _qi,
                    "nstep":     feats["nstep"],
                    "ndis":      feats["ndis"],
                    "ninserts":  feats["ninserts"],
                    "firstNN":   feats["firstNN"],
                    "closestNN": feats["closestNN"],
                    "furthestNN":feats["furthestNN"],
                    "meanNN":    feats["meanNN"],
                    "varNN":     feats["varNN"],
                    "p25NN":     feats["p25NN"],
                    "p50NN":     feats["p50NN"],
                    "p75NN":     feats["p75NN"],
                    "r":         r,
                }
                pending_rows.append(row)

            idx.search_darth_collect(
                xq[qi:qi+1],
                k=efSearch,          # efSearch-sized result set so features match inference
                efSearch=efSearch,
                ipi=ipi,
                callback=_callback,
            )

            writer.writerows(pending_rows)
            total_rows += len(pending_rows)

            if verbose and (qi + 1) % 500 == 0:
                elapsed = time.perf_counter() - t1
                print(f"  [{qi+1}/{nq}] queries done, {total_rows} rows, "
                      f"{elapsed:.0f}s elapsed", flush=True)

    if verbose:
        print(f"  Done. {total_rows} training rows → {output_path}")
