"""
Full benchmark: Baseline-HNSW, DARTH, PiP, Ada-ef, faiss-hnsw
across all datasets. Results saved to benchmark_results/{dataset}/results.csv.
Plots saved to benchmark_results/{dataset}/recall_vs_qps.png|pdf.

Usage:
    python benchmark/run_benchmark.py                        # all datasets
    python benchmark/run_benchmark.py --datasets siftsmall   # one dataset
    python benchmark/run_benchmark.py --no-pip --no-adaef   # skip slow methods
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hnswDarth_cpp" / "src" / "build"))

# ── C++ bindings ──────────────────────────────────────────────────────────────
import hnswDarth_cpp

def _load_so(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pip_so    = ROOT / "hnsw_pip_cpp"   / "src" / "build" / "hnsw_pip.cpython-311-darwin.so"
_adaef_so  = ROOT / "hnsw_adaef_cpp" / "src" / "build" / "hnsw_edaef.cpython-311-darwin.so"

hnswPip_cpp   = _load_so(_pip_so,   "hnswPip_cpp")
hnswAdaEF_cpp = _load_so(_adaef_so, "hnswAdaEF_cpp")

import faiss
faiss.omp_set_num_threads(1)

LGBMPredictor = hnswDarth_cpp.LGBMPredictor  # C++ native predictor — castable by search_darth

# ── I/O helpers ───────────────────────────────────────────────────────────────
def _read_fvecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()

def _read_ivecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()

# ── Metrics ───────────────────────────────────────────────────────────────────
N_WARMUP  = 3
N_MEASURE = 1   # single full-batch timing is stable enough; increase for noisy hardware

def _recall(I_pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Vectorised Recall@k."""
    I_k  = I_pred[:, :k]
    gt_k = gt[:, :k]
    hits = (I_k[:, :, None] == gt_k[:, None, :]).any(axis=2).sum()
    return float(hits) / (len(gt_k) * k)

def _measure(search_fn, xq: np.ndarray, gt: np.ndarray, k: int):
    """Return (recall, qps). search_fn(xq) -> I (n, k)."""
    nq = len(xq)
    wq = xq[:min(nq, 64)]
    for _ in range(N_WARMUP):
        search_fn(wq)
    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        I  = search_fn(xq)
        times.append(time.perf_counter() - t0)
    t = min(times)
    return _recall(I, gt, k), nq / t

def _l2norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-10)
    return (x / norms).astype(np.float32)

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "siftsmall": {
        "base":   ROOT / "Datasets/siftsmall/siftsmall_base.fvecs",
        "query":  ROOT / "Datasets/siftsmall/siftsmall_query.fvecs",
        "gt":     ROOT / "Datasets/siftsmall/siftsmall_groundtruth.ivecs",
        "metric": "l2",
        "pip_gamma": 95.0, "pip_delta": 20,  # best-recall config from prior sweep
    },
    "fashion_mnist": {
        "base":   ROOT / "Datasets/fashion_mnist/fashion_mnist_base.fvecs",
        "query":  ROOT / "Datasets/fashion_mnist/fashion_mnist_query.fvecs",
        "gt":     ROOT / "Datasets/fashion_mnist/fashion_mnist_groundtruth.ivecs",
        "metric": "l2",
        "pip_gamma": 95.0, "pip_delta": 20,
    },
    "msmarco_text": {
        "base":   ROOT / "Datasets/msmarco_text/msmarco_base.fvecs",
        "query":  ROOT / "Datasets/msmarco_text/msmarco_query.fvecs",
        "gt":     ROOT / "Datasets/msmarco_text/msmarco_groundtruth.ivecs",
        "metric": "l2",
        "pip_gamma": 99.0, "pip_delta": 20,
    },
    "sift128": {
        "base":   ROOT / "Datasets/sift128ann/sift128_base.fvecs",
        "query":  ROOT / "Datasets/sift128ann/sift128_query.fvecs",
        "gt":     ROOT / "Datasets/sift128ann/sift128_groundtruth.ivecs",
        "metric": "l2",
        "pip_gamma": 99.0, "pip_delta": 20,
    },
    "glove100": {
        "base":   ROOT / "Datasets/glove100ann/glove100_base.fvecs",
        "query":  ROOT / "Datasets/glove100ann/glove100_query.fvecs",
        "gt":     ROOT / "Datasets/glove100ann/glove100_groundtruth.ivecs",
        "metric": "l2",
        "pip_gamma": 99.0, "pip_delta": 20,
    },
}

M_VALUES       = [8, 16]
EFC_VALUES     = [128, 200]
EF_SEARCH      = [10, 20, 50, 100, 150, 200]
K_VALUES       = [10, 20]
DARTH_RT       = 0.90
ADAEF_RT_VALUES = [0.90]

MODEL_DIR = ROOT / "predictor_models"


# ── Index builders ────────────────────────────────────────────────────────────
def _build_darth_cpp(xb, M, efC, metric):
    idx = hnswDarth_cpp.HNSWDarthIndex(
        dim=int(xb.shape[1]), M=int(M), efConstruction=int(efC), metric=metric
    )
    idx.add(np.ascontiguousarray(xb, dtype=np.float32))
    return idx

def _build_pip_cpp(xb, M, efC, metric, gamma, delta):
    idx = hnswPip_cpp.HNSWPiPIndex(
        dim=int(xb.shape[1]), M=int(M), efConstruction=int(efC),
        metric=metric, pip_gamma=float(gamma), pip_delta=int(delta)
    )
    idx.add(np.ascontiguousarray(xb, dtype=np.float32))
    return idx

def _build_adaef_cpp(xb_n, M, efC, k, rt_values):
    ef_cal = [20, 50, 75, 100, 150, 200, 300]
    idx = hnswAdaEF_cpp.HNSWAdaEFIndex(
        dim=int(xb_n.shape[1]), M=int(M), efConstruction=int(efC), metric="cosine"
    )
    idx.add(np.ascontiguousarray(xb_n, dtype=np.float32))
    # use max k for calibration table
    idx.build_adaef_offline(k=int(k), target_recall=float(max(rt_values)), ef_values=ef_cal)
    return idx

def _build_faiss_hnsw(xb, M, efC):
    idx = faiss.IndexHNSWFlat(xb.shape[1], M)
    idx.hnsw.efConstruction = efC
    idx.add(xb.astype(np.float32))
    return idx


# ── Per-method sweep ──────────────────────────────────────────────────────────
def _row(dataset, method, M, efC, ef, k, recall, qps, **extra):
    return {"dataset": dataset, "method": method, "M": M, "efConstruction": efC,
            "efSearch": ef, "k": k, "Recall@k": round(recall, 5),
            "QPS": round(qps, 1), **extra}


def _sweep_baseline_and_darth(xb, xq, gt, dataset, M, efC, metric, predictor):
    rows = []
    print(f"  [C++] Building Baseline/DARTH M={M} efC={efC} …", flush=True)
    t0 = time.perf_counter()
    idx = _build_darth_cpp(xb, M, efC, metric)
    print(f"  [C++] Built in {time.perf_counter()-t0:.1f}s", flush=True)

    for ef in EF_SEARCH:
        for k in K_VALUES:
            # Baseline
            fn = lambda q, _ef=ef, _k=k: idx.search(
                np.ascontiguousarray(q, dtype=np.float32), k=int(_k), efSearch=int(_ef))[1]
            rec, qps = _measure(fn, xq, gt, k)
            rows.append(_row(dataset, "Baseline-HNSW", M, efC, ef, k, rec, qps))
            print(f"    Baseline  ef={ef:4d} k={k} → R@{k}={rec:.4f}  QPS={qps:.0f}", flush=True)

            # DARTH
            if predictor is not None:
                fn2 = lambda q, _ef=ef, _k=k: idx.search_darth(
                    np.ascontiguousarray(q, dtype=np.float32),
                    k=int(_k), efSearch=int(_ef), Rt=DARTH_RT,
                    predictor=predictor, ipi=int(_ef), mpi=20)[1]
                rec2, qps2 = _measure(fn2, xq, gt, k)
                rows.append(_row(dataset, "DARTH", M, efC, ef, k, rec2, qps2, Rt=DARTH_RT))
                print(f"    DARTH     ef={ef:4d} k={k} → R@{k}={rec2:.4f}  QPS={qps2:.0f}", flush=True)

    del idx
    return rows


def _sweep_pip(xb, xq, gt, dataset, M, efC, metric, gamma, delta):
    rows = []
    print(f"  [PiP] Building M={M} efC={efC} γ={gamma} Δ={delta} …", flush=True)
    t0 = time.perf_counter()
    idx = _build_pip_cpp(xb, M, efC, metric, gamma, delta)
    print(f"  [PiP] Built in {time.perf_counter()-t0:.1f}s", flush=True)

    for ef in EF_SEARCH:
        for k in K_VALUES:
            fn = lambda q, _ef=ef, _k=k: idx.search(
                np.ascontiguousarray(q, dtype=np.float32), k=int(_k), efSearch=int(_ef))[1]
            rec, qps = _measure(fn, xq, gt, k)
            rows.append(_row(dataset, "PiP", M, efC, ef, k, rec, qps,
                             pip_gamma=gamma, pip_delta=delta))
            print(f"    PiP       ef={ef:4d} k={k} → R@{k}={rec:.4f}  QPS={qps:.0f}", flush=True)

    del idx
    return rows


def _sweep_adaef(xb, xq, dataset, M, efC):
    rows = []
    xb_n = _l2norm(xb)
    xq_n = _l2norm(xq)
    # recompute GT on normalised vectors (cosine-equivalent)
    print(f"  [Ada-ef] Computing cosine GT …", flush=True)
    fi = faiss.IndexFlatL2(xb_n.shape[1])
    fi.add(xb_n)
    _, gt_cos_raw = fi.search(xq_n, max(K_VALUES))
    gt_cos = gt_cos_raw.astype(np.int32)

    for k in K_VALUES:
        print(f"  [Ada-ef] Building M={M} efC={efC} k={k} …", flush=True)
        t0 = time.perf_counter()
        idx = _build_adaef_cpp(xb_n, M, efC, k, ADAEF_RT_VALUES)
        print(f"  [Ada-ef] Built in {time.perf_counter()-t0:.1f}s", flush=True)

        for Rt in ADAEF_RT_VALUES:
            fn = lambda q, _Rt=Rt, _k=k: idx.search(
                np.ascontiguousarray(q, dtype=np.float32),
                k=int(_k), target_recall=float(_Rt))[1]
            rec, qps = _measure(fn, xq_n, gt_cos, k)
            rows.append(_row(dataset, "Ada-ef", M, efC, 0, k, rec, qps, Rt=Rt))
            print(f"    Ada-ef    Rt={Rt}  k={k} → R@{k}={rec:.4f}  QPS={qps:.0f}", flush=True)

        del idx
    return rows



def _sweep_faiss_hnsw(xb, xq, gt, dataset, M, efC):
    rows = []
    print(f"  [faiss-hnsw] Building M={M} efC={efC} …", flush=True)
    t0 = time.perf_counter()
    idx = _build_faiss_hnsw(xb, M, efC)
    print(f"  [faiss-hnsw] Built in {time.perf_counter()-t0:.1f}s", flush=True)

    for ef in EF_SEARCH:
        idx.hnsw.efSearch = ef
        for k in K_VALUES:
            fn = lambda q, _k=k: idx.search(
                np.ascontiguousarray(q, dtype=np.float32), int(_k))[1]  # ef set on index above
            rec, qps = _measure(fn, xq, gt, k)
            rows.append(_row(dataset, "faiss-hnsw", M, efC, ef, k, rec, qps))
            print(f"    faiss-hnsw ef={ef:4d} k={k} → R@{k}={rec:.4f}  QPS={qps:.0f}", flush=True)

    del idx
    return rows


# ── Per-dataset orchestration ─────────────────────────────────────────────────
def run_dataset(name: str, cfg: dict,
                run_pip: bool, run_adaef: bool,
                run_faiss: bool) -> pd.DataFrame:
    print(f"\n{'═'*55}")
    print(f"  Dataset: {name}")
    print(f"{'═'*55}")

    xb = _read_fvecs(str(cfg["base"]))
    xq = _read_fvecs(str(cfg["query"]))
    gt = _read_ivecs(str(cfg["gt"]))
    metric = cfg["metric"]
    print(f"  xb={xb.shape}  xq={xq.shape}  gt={gt.shape}")

    all_rows = []

    for M in M_VALUES:
        for efC in EFC_VALUES:
            # Load predictor — fall back to any trained efC for the same M
            model_path = MODEL_DIR / f"{name}_M{M}_efC{efC}_k10.txt"
            if not model_path.exists():
                # try other efC values in descending order (prefer larger efC)
                for fallback_efC in sorted(EFC_VALUES, reverse=True):
                    fb = MODEL_DIR / f"{name}_M{M}_efC{fallback_efC}_k10.txt"
                    if fb.exists():
                        model_path = fb
                        print(f"  [info] No predictor for efC={efC}, using efC={fallback_efC} predictor")
                        break
            predictor = LGBMPredictor(str(model_path)) if model_path.exists() else None
            if predictor is None:
                print(f"  [warn] No predictor found for M={M} — DARTH disabled")

            # Baseline + DARTH (share one index build)
            all_rows += _sweep_baseline_and_darth(xb, xq, gt, name, M, efC, metric, predictor)

            # PiP — single best config (no grid sweep)
            if run_pip:
                all_rows += _sweep_pip(xb, xq, gt, name, M, efC, metric,
                                       cfg["pip_gamma"], cfg["pip_delta"])

            # Ada-ef
            if run_adaef:
                all_rows += _sweep_adaef(xb, xq, name, M, efC)

            if run_faiss:
                all_rows += _sweep_faiss_hnsw(xb, xq, gt, name, M, efC)

    return pd.DataFrame(all_rows)


# ── Plotting ──────────────────────────────────────────────────────────────────
METHOD_STYLE = {
    "Baseline-HNSW": dict(color="#333333", marker="o",  ls="-",  lw=2.5, ms=8,  label="Baseline-HNSW (C++)"),
    "DARTH":         dict(color="#e63946", marker="s",  ls="-",  lw=2.5, ms=8,  label="DARTH (Rt=0.90)"),
    "PiP":           dict(color="#2a9d8f", marker="^",  ls="-",  lw=2.5, ms=9,  label="PiP"),
    "Ada-ef":        dict(color="#f4a261", marker="D",  ls="--", lw=2.0, ms=10, label="Ada-ef (Rt=0.90)"),
    "faiss-hnsw":    dict(color="#8338ec", marker="P",  ls="--", lw=2.0, ms=8,  label="faiss-hnsw"),
}

def _pareto(df_m: pd.DataFrame) -> pd.DataFrame:
    pts = df_m.sort_values("QPS").copy()
    best_rec, keep = -1.0, []
    for _, row in pts.iterrows():
        if row["Recall@k"] > best_rec:
            best_rec = row["Recall@k"]
            keep.append(True)
        else:
            keep.append(False)
    return pts[keep]


def _plot_recall_vs_efsearch(sub: pd.DataFrame, k: int, M: int, efC: int,
                              dataset: str, out_dir: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7))

    ef_vals = sorted(sub[sub["method"] == "Baseline-HNSW"]["efSearch"].unique())
    ef_min, ef_max = ef_vals[0], ef_vals[-1]

    for method, style in METHOD_STYLE.items():
        mdf = sub[sub["method"] == method].copy()
        if mdf.empty:
            continue
        if method == "Ada-ef":
            rec = mdf["Recall@k"].iloc[0]
            ax.hlines(rec, ef_min, ef_max,
                      colors=style["color"], linestyles="dashed", lw=style["lw"])
            ax.scatter([ef_max], [rec], color=style["color"],
                       marker=style["marker"], s=120, zorder=6)
            ax.annotate(f" Ada-ef", (ef_max, rec),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=10, color=style["color"], va="center", fontweight="bold")
            ax.plot([], [], color=style["color"], ls="--", lw=style["lw"],
                    marker=style["marker"], markersize=style["ms"], label=style["label"])
            continue
        mdf = mdf.sort_values("efSearch")
        ax.plot(mdf["efSearch"], mdf["Recall@k"],
                color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=style["lw"], markersize=style["ms"],
                label=style["label"])

    ax.axhline(0.90, color="gray", ls=":", lw=1.5, label="Target = 0.90")
    ax.set_xlabel("efSearch", fontsize=13)
    ax.set_ylabel(f"Recall@{k}", fontsize=13)
    ax.set_title(f"Recall@{k} vs efSearch  —  {dataset}  (M={M}, efC={efC})", fontsize=14)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95, edgecolor="#cccccc")
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    stem = f"recall_M{M}_efC{efC}_k{k}"
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  Saved {stem}.png / .pdf")


def _plot_recall_vs_qps(sub: pd.DataFrame, k: int, M: int, efC: int,
                         dataset: str, out_dir: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7))

    for method, style in METHOD_STYLE.items():
        mdf = sub[sub["method"] == method]
        if mdf.empty:
            continue
        if method == "Ada-ef":
            ax.scatter(mdf["QPS"], mdf["Recall@k"],
                       color=style["color"], marker=style["marker"],
                       s=160, zorder=6, label=style["label"])
            for _, r in mdf.iterrows():
                ax.annotate(f"Ada-ef", (r["QPS"], r["Recall@k"]),
                            textcoords="offset points", xytext=(8, 0),
                            fontsize=10, color=style["color"], va="center", fontweight="bold")
        else:
            pf = _pareto(mdf)
            ax.plot(pf["QPS"], pf["Recall@k"],
                    color=style["color"], marker=style["marker"],
                    ls=style["ls"], lw=style["lw"], markersize=style["ms"],
                    label=style["label"])

    ax.axhline(0.90, color="gray", ls=":", lw=1.5, label="Target = 0.90")
    ax.set_xscale("log")
    ax.set_xlabel("QPS (log scale)", fontsize=13)
    ax.set_ylabel(f"Recall@{k}", fontsize=13)
    ax.set_title(f"Recall@{k} vs QPS  —  {dataset}  (M={M}, efC={efC})", fontsize=14)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95, edgecolor="#cccccc")
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()

    stem = f"recall_vs_qps_M{M}_efC{efC}_k{k}"
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  Saved {stem}.png / .pdf")


def plot_dataset(df: pd.DataFrame, out_dir: Path, dataset: str):
    import matplotlib
    matplotlib.use("Agg")

    for M in df["M"].unique():
        for efC in df["efConstruction"].unique():
            for k in df["k"].unique():
                sub = df[(df["M"] == M) & (df["efConstruction"] == efC) & (df["k"] == k)]
                if sub.empty:
                    continue
                _plot_recall_vs_efsearch(sub, k, M, efC, dataset, out_dir)
                _plot_recall_vs_qps(sub, k, M, efC, dataset, out_dir)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--no-pip",   action="store_true")
    parser.add_argument("--no-adaef", action="store_true")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args()

    out_root = ROOT / "benchmark_results"

    for name in args.datasets:
        if name not in DATASETS:
            print(f"[skip] Unknown dataset: {name}")
            continue

        cfg = DATASETS[name]
        df  = run_dataset(name, cfg,
                          run_pip=not args.no_pip,
                          run_adaef=not args.no_adaef,
                          run_faiss=not args.no_faiss)

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved {csv_path}")

        print(f"  Generating plots …")
        plot_dataset(df, out_dir, name)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
