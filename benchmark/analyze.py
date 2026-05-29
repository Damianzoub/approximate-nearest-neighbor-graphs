"""
Simple analysis of benchmark results.

Usage:
    python benchmark/analyze.py                        # all datasets
    python benchmark/analyze.py --datasets siftsmall   # one dataset
    python benchmark/analyze.py --recall-target 0.90   # custom recall threshold
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
RESULTS    = ROOT / "benchmark_results"
RECALL_TARGET = 0.90


def load_all(datasets: list[str]) -> pd.DataFrame:
    frames = []
    for d in datasets:
        p = RESULTS / d / "results.csv"
        if not p.exists():
            print(f"[skip] No results for {d}")
            continue
        df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary_table(df: pd.DataFrame, recall_target: float) -> pd.DataFrame:
    """For each (dataset, method): best QPS at or above recall_target, plus build time."""
    rows = []
    for (dataset, method), g in df.groupby(["dataset", "method"]):
        above = g[g["Recall@k"] >= recall_target]
        if above.empty:
            best_qps   = float("nan")
            best_recall = g["Recall@k"].max()
        else:
            best_row    = above.loc[above["QPS"].idxmax()]
            best_qps    = best_row["QPS"]
            best_recall = best_row["Recall@k"]

        build_t = g["build_time_s"].dropna().iloc[0] if "build_time_s" in g.columns and not g["build_time_s"].dropna().empty else float("nan")
        rows.append({
            "dataset":        dataset,
            "method":         method,
            f"best_QPS@R≥{recall_target}": round(best_qps, 1) if not pd.isna(best_qps) else "—",
            "best_Recall@k":  round(best_recall, 4),
            "build_time_s":   build_t,
        })
    return pd.DataFrame(rows).sort_values(["dataset", f"best_QPS@R≥{recall_target}"],
                                          ascending=[True, False])


def speedup_over_baseline(df: pd.DataFrame, recall_target: float) -> pd.DataFrame:
    """QPS speedup of each method vs Baseline-HNSW at the same recall target."""
    rows = []
    for dataset, g in df.groupby("dataset"):
        base = g[g["method"] == "Baseline-HNSW"]
        base_above = base[base["Recall@k"] >= recall_target]
        if base_above.empty:
            continue
        base_qps = base_above.loc[base_above["QPS"].idxmax(), "QPS"]

        for method, mg in g.groupby("method"):
            if method == "Baseline-HNSW":
                continue
            above = mg[mg["Recall@k"] >= recall_target]
            if above.empty:
                speedup = float("nan")
            else:
                speedup = above["QPS"].max() / base_qps
            rows.append({"dataset": dataset, "method": method,
                         "speedup_vs_baseline": round(speedup, 3) if not pd.isna(speedup) else "—"})
    return pd.DataFrame(rows).sort_values(["dataset", "method"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=[p.name for p in RESULTS.iterdir() if p.is_dir()])
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    args = parser.parse_args()

    df = load_all(args.datasets)
    if df.empty:
        print("No results found.")
        return

    print(f"\n{'═'*65}")
    print(f"  SUMMARY — best QPS at Recall ≥ {args.recall_target}")
    print(f"{'═'*65}")
    s = summary_table(df, args.recall_target)
    print(s.to_string(index=False))

    print(f"\n{'═'*65}")
    print(f"  QPS SPEEDUP vs Baseline-HNSW (at Recall ≥ {args.recall_target})")
    print(f"{'═'*65}")
    sp = speedup_over_baseline(df, args.recall_target)
    print(sp.to_string(index=False))

    # Save combined CSV and summary
    out = RESULTS / "analysis_summary.csv"
    s.to_csv(out, index=False)
    print(f"\n  Saved {out}")

    out2 = RESULTS / "speedup_summary.csv"
    sp.to_csv(out2, index=False)
    print(f"  Saved {out2}")


if __name__ == "__main__":
    main()
