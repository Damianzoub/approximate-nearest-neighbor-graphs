"""
Shared experiment logic used by all dataset-specific scripts.
Uses C++ bindings for index build and search; Python DARTH only for predictor training.

Columns written to every all_results.csv:
  dataset, method, M, efConstruction, efSearch, k,
  Rt, pip_gamma, pip_delta,
  Recall@k, QPS, Time_total_s, n_queries

Failure analysis CSV (DARTH-specific) columns:
  dataset, query_idx, baseline_recall, darth_recall,
  is_failure, difficulty_1nn_dist, query_norm
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── C++ bindings ───────────────────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "hnswDarth_cpp"  / "src" / "build"))

import hnswDarth_cpp   # HNSW-DARTH  (HNSWDarthIndex)

# PiP loaded via spec (module filename doesn't match module name)
_pip_so = ROOT / "hnsw_pip_cpp" / "src" / "build" / "hnsw_pip.cpython-311-darwin.so"
_pip_spec = importlib.util.spec_from_file_location("hnswPip_cpp", _pip_so)
hnswPip_cpp = importlib.util.module_from_spec(_pip_spec)
_pip_spec.loader.exec_module(hnswPip_cpp)

_adaef_so = ROOT / "hnsw_adaef_cpp" / "src" / "build" / "hnsw_edaef.cpython-311-darwin.so"
try:
    _adaef_spec = importlib.util.spec_from_file_location("hnswAdaEF_cpp", _adaef_so)
    hnswAdaEF_cpp = importlib.util.module_from_spec(_adaef_spec)
    _adaef_spec.loader.exec_module(hnswAdaEF_cpp)
    _HAS_ADAEF_CPP = True
except Exception:
    _HAS_ADAEF_CPP = False
    print("[warn] hnswAdaEF_cpp not found — Ada-ef experiments will be skipped")

# Python DARTH — used only for predictor training data collection
from hsnw_constructionDARTH import HNSW_DARTH
from darth.collect_training_data import collect_training_data
from darth.train_predictor import train as train_predictor
from darth.predictor import LGBMPredictor
from utils.read_files import read_fvecs, read_ivecs


# ── constants ──────────────────────────────────────────────────────────────────

EF_SEARCH_VALUES = [10, 20, 50, 100, 150, 200]
K_VALUES         = [10, 20]
RT_VALUES        = [0.80, 0.85, 0.90, 0.95]
PIP_GAMMA_VALUES = [90, 95, 99]
PIP_DELTA_VALUES = [5, 10, 20]
ADAEF_RT_VALUES  = [0.80, 0.85, 0.90, 0.95]

N_WARMUP  = 2
N_MEASURE = 5


# ── helpers ────────────────────────────────────────────────────────────────────

def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    """Per-query Recall@k — vectorised. Returns shape (n_queries,)."""
    I_k  = I[:, :k]
    gt_k = gt[:, :k]
    hits = (I_k[:, :, None] == gt_k[:, None, :]).any(axis=2).sum(axis=1)
    return hits.astype(np.float32) / k


def mean_recall(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    return float(recall_at_k(I, gt, k).mean())


def measure_search(search_fn, xq: np.ndarray, n_warmup: int, n_measure: int):
    """
    Warm up then time search_fn(xq) for n_measure rounds.
    search_fn must accept xq (full query matrix) and return (D, I).
    Returns (I_last, mean_time_seconds).
    """
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    for _ in range(n_warmup):
        search_fn(xq)
    times = []
    I_out = None
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _, I_out = search_fn(xq)
        times.append(time.perf_counter() - t0)
    return I_out, float(np.mean(times))


def row(dataset, method, M, efC, efSearch, k, recall, t_s, nq,
        Rt=None, pip_gamma=None, pip_delta=None) -> dict:
    return {
        "dataset":        dataset,
        "method":         method,
        "M":              M,
        "efConstruction": efC,
        "efSearch":       efSearch,
        "k":              k,
        "Rt":             Rt,
        "pip_gamma":      pip_gamma,
        "pip_delta":      pip_delta,
        "Recall@k":       round(recall, 6),
        "QPS":            round(nq / t_s, 2) if t_s > 0 else 0.0,
        "Time_total_s":   round(t_s, 6),
        "n_queries":      nq,
    }


def exact_gt_faiss(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """Exact L2 GT via faiss. Falls back to numpy if faiss unavailable."""
    try:
        import faiss
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb.astype(np.float32))
        _, I = index.search(xq.astype(np.float32), k)
        return I.astype(np.int32)
    except ImportError:
        print("  [warn] faiss not found — falling back to numpy brute-force GT (slow)")
        return _exact_gt_numpy(xb, xq, k)


def _exact_gt_numpy(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    gt = np.empty((len(xq), k), dtype=np.int32)
    batch = 500
    for s in range(0, len(xq), batch):
        e = min(s + batch, len(xq))
        q = xq[s:e].astype(np.float32)
        diffs = (q**2).sum(1, keepdims=True) + (xb**2).sum(1) - 2.0 * (q @ xb.T)
        gt[s:e] = np.argsort(diffs, 1)[:, :k]
    return gt


# ── C++ index builders ────────────────────────────────────────────────────────

def build_darth_cpp(xb: np.ndarray, M: int, efC: int, metric: str = "l2"):
    """Build C++ HNSW-DARTH index (used for Baseline + DARTH benchmarks)."""
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    print(f"  [C++/DARTH] Building M={M} efC={efC} n={len(xb)} …")
    t0 = time.perf_counter()
    idx = hnswDarth_cpp.HNSWDarthIndex(
        dim=xb.shape[1], M=int(M), efConstruction=int(efC), metric=str(metric)
    )
    idx.add(xb)
    print(f"  [C++/DARTH] Built in {time.perf_counter()-t0:.1f}s")
    return idx


def build_pip_cpp(xb: np.ndarray, M: int, efC: int,
                  gamma: float, delta: int, metric: str = "l2"):
    """Build C++ HNSW-PiP index for a single (gamma, delta) config."""
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    print(f"  [C++/PiP] Building M={M} efC={efC} γ={gamma} Δ={delta} …")
    t0 = time.perf_counter()
    idx = hnswPip_cpp.HNSWPiPIndex(
        dim=xb.shape[1], M=int(M), efConstruction=int(efC),
        metric=str(metric), pip_gamma=float(gamma), pip_delta=int(delta)
    )
    idx.add(xb)
    print(f"  [C++/PiP] Built in {time.perf_counter()-t0:.1f}s")
    return idx


def build_adaef_cpp(xb: np.ndarray, M: int, efC: int,
                    metric: str = "l2",
                    offline_k: int = 10, offline_target_recall: float = 0.90,
                    offline_ef_values: list | None = None):
    """Build C++ HNSW-AdaEF index with prebuilt offline recall table."""
    if not _HAS_ADAEF_CPP:
        raise ImportError("hnswAdaEF_cpp not available")
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    if offline_ef_values is None:
        offline_ef_values = [50, 75, 100, 150, 200, 300, 400, 500]
    print(f"  [C++/Ada-ef] Building M={M} efC={efC} …")
    t0 = time.perf_counter()
    idx = hnswAdaEF_cpp.HNSWAdaEFIndex(
        dim=xb.shape[1], M=int(M), efConstruction=int(efC), metric=str(metric)
    )
    idx.add(xb)
    print(f"  [C++/Ada-ef] Built in {time.perf_counter()-t0:.1f}s  — building offline table …")
    idx.build_adaef_offline(
        k=int(offline_k),
        target_recall=float(offline_target_recall),
        ef_values=offline_ef_values,
    )
    print("  [C++/Ada-ef] Offline table ready")
    return idx


# ── Python DARTH (predictor training only) ────────────────────────────────────

def build_darth_python(xb: np.ndarray, M: int, efC: int, metric: str = "l2") -> HNSW_DARTH:
    """Build Python HNSW-DARTH index — only needed to collect predictor training data."""
    print(f"  [Py/DARTH] Building M={M} efC={efC} n={len(xb)} …")
    t0 = time.perf_counter()
    idx = HNSW_DARTH(dim=xb.shape[1], M=M, efConstruction=efC, metric=metric)
    for i, v in enumerate(xb):
        idx._insert_(v, i)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(xb)}", flush=True)
    print(f"  [Py/DARTH] Built in {time.perf_counter()-t0:.1f}s")
    return idx


def ensure_predictor(xb: np.ndarray, xlearn: np.ndarray, gt_learn: np.ndarray,
                     dataset: str, M: int, efC: int, k: int,
                     model_dir: Path) -> LGBMPredictor:
    """
    Train and cache a LightGBM predictor.
    Builds a Python DARTH index on xb only if the training CSV is not yet cached.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = model_dir / f"train_data_{dataset}_M{M}_efC{efC}_k{k}.csv"
    model_path = model_dir / f"{dataset}_M{M}_efC{efC}_k{k}.txt"

    if model_path.exists():
        print(f"  [skip] Predictor already exists: {model_path.name}")
        return LGBMPredictor(str(model_path))

    if not csv_path.exists():
        print(f"\n  Predictor training CSV not cached — building Python DARTH index …")
        darth_py = build_darth_python(xb, M, efC)
        print(f"  Collecting training data → {csv_path.name}")
        collect_training_data(
            darth_py, xlearn, gt_learn,
            k=k, efSearch=efC,
            output_path=str(csv_path),
            logging_interval=5, verbose=True,
        )
        del darth_py

    if not model_path.exists():
        print(f"  Training predictor → {model_path.name}")
        train_predictor(
            input_csv=str(csv_path),
            output_model=str(model_path),
            n_estimators=100,
            val_fraction=0.1,
            verbose=True,
        )

    return LGBMPredictor(str(model_path))


# ── per-method experiment runners (all use C++ indexes) ───────────────────────

def run_baseline(darth_cpp_idx, xq, gt, dataset, M, efC, k_values, ef_values):
    results = []
    nq = len(xq)
    for k in k_values:
        for ef in ef_values:
            fn = lambda q, _ef=ef, _k=k: darth_cpp_idx.search_darth(
                q, k=int(_k), efSearch=int(_ef),
                Rt=0.0, predictor=None, ipi=int(_ef)+1, mpi=int(_ef)+1
            )
            I, t = measure_search(fn, xq, N_WARMUP, N_MEASURE)
            rec = mean_recall(I, gt, k)
            results.append(row(dataset, "Baseline-HNSW", M, efC, ef, k, rec, t, nq))
            print(f"    Baseline ef={ef:>4} k={k} → R@k={rec:.4f}  QPS={nq/t:.0f}")
    return results


def run_darth(darth_cpp_idx, predictor, xq, gt, dataset, M, efC,
              k_values, ef_values, rt_values, ipi=200, mpi=20):
    results = []
    nq = len(xq)
    for k in k_values:
        for ef in ef_values:
            for Rt in rt_values:
                fn = lambda q, _ef=ef, _Rt=Rt, _k=k: darth_cpp_idx.search_darth(
                    q, k=int(_k), efSearch=int(_ef),
                    Rt=float(_Rt), predictor=predictor,
                    ipi=int(ipi), mpi=int(mpi)
                )
                I, t = measure_search(fn, xq, N_WARMUP, N_MEASURE)
                rec = mean_recall(I, gt, k)
                results.append(row(dataset, "DARTH", M, efC, ef, k, rec, t, nq, Rt=Rt))
                print(f"    DARTH   ef={ef:>4} k={k} Rt={Rt} → R@k={rec:.4f}  QPS={nq/t:.0f}")
    return results


def run_pip(xb, xq, gt, dataset, M, efC, metric,
            k_values, ef_values, gamma_values, delta_values):
    """Builds a fresh C++ PiP index for each (gamma, delta) combo."""
    results = []
    nq = len(xq)
    for gamma in gamma_values:
        for delta in delta_values:
            pidx = build_pip_cpp(xb, M, efC, gamma, delta, metric)
            for k in k_values:
                for ef in ef_values:
                    fn = lambda q, _ef=ef, _k=k: pidx.search(
                        q, k=int(_k), efSearch=int(_ef)
                    )
                    I, t = measure_search(fn, xq, N_WARMUP, N_MEASURE)
                    rec = mean_recall(I, gt, k)
                    results.append(row(dataset, "PiP", M, efC, ef, k, rec, t, nq,
                                       pip_gamma=gamma, pip_delta=delta))
                    print(f"    PiP     ef={ef:>4} k={k} γ={gamma} Δ={delta} → R@k={rec:.4f}  QPS={nq/t:.0f}")
            del pidx
    return results


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-10)
    return (x / norms).astype(np.float32)


def run_adaef(xb, xq, gt, dataset, M, efC, metric, k_values, rt_values):
    """
    Builds C++ Ada-ef index (cosine only) then sweeps k and target recall.
    Vectors are L2-normalised so cosine ≡ L2 on the unit sphere.
    GT is recomputed on normalised vectors for fair recall measurement.
    """
    results = []
    nq = len(xq)

    # Ada-ef C++ only supports cosine — normalise regardless of dataset metric
    xb_n = _l2_normalise(xb)
    xq_n = _l2_normalise(xq)

    # Recompute GT on normalised vectors (cosine = L2 on unit sphere)
    max_k = max(k_values)
    print(f"  [Ada-ef] Computing cosine GT for {nq} queries …")
    gt_cos = exact_gt_faiss(xb_n, xq_n, max_k)

    try:
        aidx = build_adaef_cpp(xb_n, M, efC, metric="cosine")
    except Exception as e:
        print(f"  [Ada-ef] build failed: {e}")
        return results

    for k in k_values:
        for Rt in rt_values:
            fn = lambda q, _Rt=Rt, _k=k: aidx.search(
                q, k=int(_k), target_recall=float(_Rt)
            )
            I, t = measure_search(fn, xq_n, N_WARMUP, N_MEASURE)
            rec = mean_recall(I, gt_cos, k)
            results.append(row(dataset, "Ada-ef", M, efC, 0, k, rec, t, nq, Rt=Rt))
            print(f"    Ada-ef  k={k} Rt={Rt} → R@k={rec:.4f}  QPS={nq/t:.0f}")
    del aidx
    return results


# ── failure analysis ───────────────────────────────────────────────────────────

def run_failure_analysis(darth_cpp_idx, predictor,
                         xq: np.ndarray, xb: np.ndarray, gt: np.ndarray,
                         dataset: str, M: int, efC: int,
                         k: int = 10, efSearch: int = 200,
                         Rt: float = 0.90, ipi: int = 200, mpi: int = 20
                         ) -> pd.DataFrame:
    """
    Per-query Baseline vs DARTH comparison at a fixed operating point.
    Flags queries where DARTH recall < Rt but Baseline recall >= Rt.
    """
    xq = np.ascontiguousarray(xq, dtype=np.float32)

    print(f"\n  [Failure] Baseline (efSearch={efSearch}) …")
    _, I_base = darth_cpp_idx.search_darth(
        xq, k=int(k), efSearch=int(efSearch),
        Rt=0.0, predictor=None, ipi=int(efSearch)+1, mpi=int(efSearch)+1
    )

    print(f"  [Failure] DARTH (Rt={Rt}) …")
    _, I_darth = darth_cpp_idx.search_darth(
        xq, k=int(k), efSearch=int(efSearch),
        Rt=float(Rt), predictor=predictor, ipi=int(ipi), mpi=int(mpi)
    )

    r_base  = recall_at_k(I_base,  gt, k)
    r_darth = recall_at_k(I_darth, gt, k)

    true_nn_idx = gt[:, 0]
    difficulty  = np.linalg.norm(xq - xb[true_nn_idx], axis=1).astype(np.float32)
    query_norms = np.linalg.norm(xq, axis=1).astype(np.float32)

    is_failure = (r_darth < Rt) & (r_base >= Rt)

    df = pd.DataFrame({
        "dataset":             dataset,
        "query_idx":           np.arange(len(xq)),
        "M":                   M,
        "efConstruction":      efC,
        "efSearch":            efSearch,
        "k":                   k,
        "Rt":                  Rt,
        "baseline_recall":     r_base.round(6),
        "darth_recall":        r_darth.round(6),
        "is_failure":          is_failure,
        "difficulty_1nn_dist": difficulty.round(6),
        "query_norm":          query_norms.round(6),
    })

    n_fail = int(is_failure.sum())
    print(f"  [Failure] {n_fail}/{len(xq)} failures ({100*n_fail/len(xq):.1f}%)")
    return df


# ── main entry point ───────────────────────────────────────────────────────────

def run_dataset_experiment(
    dataset_name: str,
    paths: dict,
    M_values: list,
    efC_values: list,
    metric: str = "l2",
    max_learn: int = 5000,
    run_pip_flag: bool = True,
    run_adaef_flag: bool = True,
):
    """
    Full benchmark for one dataset. Saves:
      results_csv/{dataset_name}/all_results.csv
      results_csv/{dataset_name}/failure_analysis.csv
    """
    out_dir   = ROOT / "results_csv" / dataset_name
    model_dir = ROOT / "predictor_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Dataset : {dataset_name}")
    print(f"{'═'*60}")

    xb     = read_fvecs(paths["base"])
    xlearn = read_fvecs(paths["learn"])[:max_learn]
    xq     = read_fvecs(paths["query"])
    gt     = read_ivecs(paths["gt"])

    print(f"  base={xb.shape}  learn={xlearn.shape}  query={xq.shape}  gt={gt.shape}")

    all_rows: list[dict]      = []
    failure_dfs: list         = []

    for M in M_values:
        for efC in efC_values:
            print(f"\n{'─'*50}")
            print(f"  M={M}  efC={efC}")
            print(f"{'─'*50}")

            # ── predictor (cached after first run) ────────────────────────────
            k_pred = 10
            csv_path = model_dir / f"train_data_{dataset_name}_M{M}_efC{efC}_k{k_pred}.csv"
            if not csv_path.exists():
                print(f"\n  Computing learn-set GT (faiss, n={len(xlearn)}) …")
                t0 = time.perf_counter()
                gt_learn = exact_gt_faiss(xb, xlearn, k_pred)
                print(f"  Learn GT done in {time.perf_counter()-t0:.1f}s")
            else:
                gt_learn = None  # already cached — ensure_predictor won't need it

            pred = ensure_predictor(xb, xlearn, gt_learn, dataset_name,
                                    M, efC, k_pred, model_dir)

            # ── C++ DARTH index (Baseline + DARTH) ───────────────────────────
            didx = build_darth_cpp(xb, M, efC, metric)

            print("\n  → Baseline HNSW")
            all_rows += run_baseline(didx, xq, gt, dataset_name, M, efC,
                                     K_VALUES, EF_SEARCH_VALUES)

            print("\n  → DARTH")
            all_rows += run_darth(didx, pred, xq, gt, dataset_name, M, efC,
                                  K_VALUES, [50, 100, 200], RT_VALUES)

            print("\n  → Failure analysis")
            fdf = run_failure_analysis(didx, pred, xq, xb, gt,
                                       dataset_name, M, efC,
                                       k=10, efSearch=200, Rt=0.90)
            failure_dfs.append(fdf)
            del didx

            # ── C++ PiP ──────────────────────────────────────────────────────
            if run_pip_flag:
                print("\n  → PiP")
                all_rows += run_pip(xb, xq, gt, dataset_name, M, efC, metric,
                                    K_VALUES, [50, 100, 200],
                                    PIP_GAMMA_VALUES, PIP_DELTA_VALUES)

            # ── C++ Ada-ef ───────────────────────────────────────────────────
            if run_adaef_flag:
                print("\n  → Ada-ef")
                all_rows += run_adaef(xb, xq, gt, dataset_name, M, efC, metric,
                                      K_VALUES, ADAEF_RT_VALUES)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    results_path = out_dir / "all_results.csv"
    pd.DataFrame(all_rows).to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")

    if failure_dfs:
        failure_path = out_dir / "failure_analysis.csv"
        pd.concat(failure_dfs, ignore_index=True).to_csv(failure_path, index=False)
        print(f"  Saved: {failure_path}")

    print(f"\n  Done — {dataset_name}")
