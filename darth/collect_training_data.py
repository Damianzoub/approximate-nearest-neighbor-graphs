"""
Training-data collector for the DARTH recall predictor.

Runs the HNSW_DARTH index in data-collection mode: for every training query,
it performs a full beam search (no early termination) and logs the 11 DARTH
features together with the actual recall@k at every `logging_interval`
distance computations.

Output CSV columns (canonical names used by train_predictor.py):
  qid, nstep, ndis, ninserts, firstNN, closestNN, furthestNN,
  meanNN, varNN, p25NN, p50NN, p75NN, r

Usage
-----
From code:
    from darth.collect_training_data import collect_training_data
    collect_training_data(hnsw, xq_train, gt, k=10, efSearch=200,
                          output_path="predictor_models/train_data.csv")

From CLI:
    python -m darth.collect_training_data \\
        --base   Datasets/siftsmall/siftsmall_base.fvecs \\
        --train  Datasets/siftsmall/siftsmall_learn.fvecs \\
        --gt     Datasets/siftsmall/siftsmall_groundtruth.ivecs \\
        --k 10 --efSearch 200 --M 16 --efC 200 \\
        --output predictor_models/train_data.csv
"""

import heapq
import csv
import os
import sys
import argparse
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core data-collection search
# ---------------------------------------------------------------------------

def _collect_search_layer(
    hnsw,
    q_vec: np.ndarray,
    ep_id: int,
    layer: int,
    efSearch: int,
    k: int,
    gt_ids_set: set,
    logging_interval: int = 2,
) -> list[dict]:
    """
    Full HNSW beam search with per-step feature logging.

    Mirrors _search_layer_darth from hsnw_constructionDARTH.py but:
      - Never breaks early (Rt = -inf)
      - Logs features + recall@k every `logging_interval` distance computations

    Parameters
    ----------
    hnsw : HNSW_DARTH
        A built Python HNSW_DARTH index.
    q_vec : np.ndarray
        Query vector.
    ep_id : int
        Entry-point node id at `layer`.
    layer : int
        Which layer to search (always 0 for DARTH training).
    efSearch : int
        Beam width.
    k : int
        Number of nearest neighbours.
    gt_ids_set : set
        Set of the k true nearest-neighbour IDs for this query.
    logging_interval : int
        Log one row every this many distance computations (after heap has k items).

    Returns
    -------
    list[dict]
        One dict per logged step, with keys matching FEATURE_COLUMNS + 'r'.
    """
    rows: list[dict] = []

    NBL = ep_id
    firstNN = float(hnsw.dist(q_vec, hnsw.vectors[NBL]))

    # max-heap of (-dist, id) — top-k result set
    result_set = [(-firstNN, NBL)]
    heapq.heapify(result_set)

    # min-heap of (dist, id) — candidate queue
    cand_queue = [(firstNN, NBL)]
    heapq.heapify(cand_queue)

    visited = {NBL}

    ndis = 1
    nstep = 0
    ninserts = 1
    idis = 0

    def get_maxdist() -> float:
        if len(result_set) < k:
            return float("inf")
        return -result_set[0][0]

    def try_add(node_id: int, d: float) -> None:
        nonlocal ninserts
        if len(result_set) < k:
            heapq.heappush(result_set, (-d, node_id))
            ninserts += 1
        elif d < -result_set[0][0]:
            heapq.heapreplace(result_set, (-d, node_id))
            ninserts += 1

    def compute_recall() -> float:
        current_ids = {node_id for (_, node_id) in result_set}
        return len(current_ids & gt_ids_set) / k

    def extract_features() -> Optional[dict]:
        if not result_set:
            return None
        dists = np.array([-neg_d for (neg_d, _) in result_set], dtype=np.float64)
        return {
            "nstep":      nstep,
            "ndis":       ndis,
            "ninserts":   ninserts,
            "firstNN":    firstNN,
            "closestNN":  float(dists.min()),
            "furthestNN": float(dists.max()),
            "meanNN":     float(dists.mean()),
            "varNN":      float(dists.var()),
            "p25NN":      float(np.percentile(dists, 25)),
            "p50NN":      float(np.percentile(dists, 50)),
            "p75NN":      float(np.percentile(dists, 75)),
        }

    while cand_queue:
        dc, c_id = heapq.heappop(cand_queue)

        if dc > get_maxdist():
            break

        ndis += 1
        idis += 1
        try_add(c_id, dc)

        layer_neighbors = hnsw.layers[layer].get(c_id, set())
        for nb in layer_neighbors:
            if nb in visited:
                continue
            visited.add(nb)
            nd = float(hnsw.dist(q_vec, hnsw.vectors[nb]))
            ndis += 1
            if nd < get_maxdist() or len(cand_queue) < efSearch:
                heapq.heappush(cand_queue, (nd, nb))

        # log once per interval (only after the heap has k candidates)
        if ninserts >= k and (idis % logging_interval) == 0:
            feats = extract_features()
            if feats is not None:
                feats["r"] = compute_recall()
                rows.append(feats)

        nstep += 1

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_training_data(
    hnsw,
    queries: np.ndarray,
    groundtruth: np.ndarray,
    k: int,
    efSearch: int,
    output_path: str,
    logging_interval: int = 2,
    max_queries: Optional[int] = None,
    verbose: bool = True,
) -> None:
    """
    Run data collection over `queries` and write a training CSV.

    Parameters
    ----------
    hnsw : HNSW_DARTH
        A fully built Python HNSW_DARTH instance.
    queries : np.ndarray, shape (nq, dim)
        Training query vectors (use learn set, not test set).
    groundtruth : np.ndarray, shape (nq, k_gt)
        Ground-truth nearest-neighbour IDs for each query.
        k_gt must be >= k.
    k : int
        Number of nearest neighbours the index is searched for.
    efSearch : int
        Beam width (same value used at search time).
    output_path : str
        Destination CSV file path.
    logging_interval : int
        Log one row every this many distance computations (default 2,
        matching the official DARTH repo).
    max_queries : int, optional
        Cap the number of queries processed (useful for quick tests).
    verbose : bool
        Print progress.
    """
    queries = np.asarray(queries, dtype=np.float32)
    groundtruth = np.asarray(groundtruth, dtype=np.int32)

    nq = len(queries)
    if max_queries is not None:
        nq = min(nq, max_queries)

    fieldnames = [
        "qid", "nstep", "ndis", "ninserts",
        "firstNN", "closestNN", "furthestNN",
        "meanNN", "varNN", "p25NN", "p50NN", "p75NN", "r",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_rows = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for qi in range(nq):
            q_vec = queries[qi].astype(float)
            gt_ids_set = set(int(x) for x in groundtruth[qi, :k])

            # greedy descent to layer 0
            ep = hnsw.entry_id
            for lc in range(hnsw.maxlevel, 0, -1):
                ep = hnsw._search_layer_greedy(q_vec, ep, lc, ef=1)

            rows = _collect_search_layer(
                hnsw, q_vec, ep, layer=0,
                efSearch=efSearch, k=k,
                gt_ids_set=gt_ids_set,
                logging_interval=logging_interval,
            )

            for row in rows:
                row["qid"] = qi
                writer.writerow({fn: row[fn] for fn in fieldnames})
            total_rows += len(rows)

            if verbose and (qi + 1) % 500 == 0:
                print(f"  [{qi+1}/{nq}] queries done, {total_rows} rows logged", flush=True)

    if verbose:
        print(f"Done. {total_rows} training rows written to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect DARTH training data by running HNSW in logging mode."
    )
    parser.add_argument("--base",    required=True, help=".fvecs base vector file")
    parser.add_argument("--train",   required=True, help=".fvecs training query file")
    parser.add_argument("--gt",      required=True, help=".ivecs ground-truth file")
    parser.add_argument("--output",  required=True, help="Output CSV path")
    parser.add_argument("--k",        type=int, default=10)
    parser.add_argument("--efSearch", type=int, default=200)
    parser.add_argument("--M",        type=int, default=16)
    parser.add_argument("--efC",      type=int, default=200,
                        help="efConstruction for HNSW build")
    parser.add_argument("--logging-interval", type=int, default=2)
    parser.add_argument("--max-queries",      type=int, default=None)
    parser.add_argument("--metric",  default="l2", choices=["l2", "cosine"])
    args = parser.parse_args()

    # import here so the module is importable without these on path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.read_files import read_fvecs, read_ivecs
    from hsnw_constructionDARTH import HNSW_DARTH

    print(f"Loading base vectors from {args.base} …")
    xb = read_fvecs(args.base)
    dim = xb.shape[1]

    print(f"Building HNSW (M={args.M}, efC={args.efC}, dim={dim}) …")
    hnsw = HNSW_DARTH(dim=dim, M=args.M, efConstruction=args.efC, metric=args.metric)
    for i, v in enumerate(xb):
        hnsw._insert_(v, i)
    print(f"  Index built: {len(xb)} vectors")

    print(f"Loading training queries from {args.train} …")
    xq = read_fvecs(args.train)

    print(f"Loading ground truth from {args.gt} …")
    gt = read_ivecs(args.gt)

    print(f"Collecting training data (k={args.k}, efSearch={args.efSearch}, "
          f"logging_interval={args.logging_interval}) …")
    collect_training_data(
        hnsw, xq, gt,
        k=args.k,
        efSearch=args.efSearch,
        output_path=args.output,
        logging_interval=args.logging_interval,
        max_queries=args.max_queries,
        verbose=True,
    )


if __name__ == "__main__":
    _main()
