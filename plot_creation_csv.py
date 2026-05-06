"""
Generate all thesis plots from experiment CSVs.

Reads:
  results_csv/{dataset}/all_results.csv
  results_csv/{dataset}/failure_analysis.csv

Writes plots to:
  plot_{dataset}/                   ← method comparison plots
  plot_{dataset}/failure_analysis/  ← failure case plots

Usage
-----
    python plot_creation_csv.py                        # all datasets
    python plot_creation_csv.py --dataset siftsmall    # one dataset
    python plot_creation_csv.py --dataset siftsmall fashion_mnist
    python plot_creation_csv.py --no-failure           # skip failure plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

DATASETS = ["siftsmall", "fashion_mnist", "msmarco_text", "sift128", "glove100"]

METHOD_STYLE = {
    "Baseline-HNSW": dict(color="#2166ac", marker="o",  linestyle="-",  linewidth=2),
    "DARTH":         dict(color="#d6604d", marker="s",  linestyle="--", linewidth=2),
    "PiP":           dict(color="#4dac26", marker="^",  linestyle="-.", linewidth=2),
    "Ada-ef":        dict(color="#b2abd2", marker="D",  linestyle=":",  linewidth=2),
}

FIG_DPI  = 200
SAVE_FMT = ["png", "pdf"]


# ── utilities ──────────────────────────────────────────────────────────────────

def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in SAVE_FMT:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def styled_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)
    return ax


def _darth_slice(g):
    s = g[g["Rt"] == 0.90]
    return s if not s.empty else g


def _pip_slice(g):
    s = g[(g["pip_gamma"] == 95) & (g["pip_delta"] == 10)]
    return s if not s.empty else g


# ── Plot 1: Recall@k vs efSearch ───────────────────────────────────────────────

def plot_recall_vs_efsearch(df: pd.DataFrame, out_dir: Path, dataset: str):
    for k in df["k"].unique():
        for M in df["M"].unique():
            for efC in df["efConstruction"].unique():
                sub = df[(df["k"] == k) & (df["M"] == M)
                         & (df["efConstruction"] == efC) & (df["efSearch"] > 0)]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 5))
                for method, style in METHOD_STYLE.items():
                    g = sub[sub["method"] == method].copy()
                    if method == "DARTH":
                        g = _darth_slice(g)
                    elif method == "PiP":
                        g = _pip_slice(g)
                    if g.empty:
                        continue
                    ax.plot(g.sort_values("efSearch")["efSearch"],
                            g.sort_values("efSearch")["Recall@k"],
                            label=method, **style)
                ax.axhline(0.90, color="grey", linestyle=":", linewidth=0.8,
                           label="Rt=0.90")
                styled_ax(ax, xlabel="efSearch", ylabel=f"Recall@{k}",
                          title=f"Recall@{k} vs efSearch — {dataset} (M={M}, efC={efC})")
                savefig(fig, out_dir / f"recall_vs_efsearch_M{M}_efC{efC}_k{k}")


# ── Plot 2: Recall–QPS Pareto curve ───────────────────────────────────────────

def plot_recall_vs_qps(df: pd.DataFrame, out_dir: Path, dataset: str):
    for k in df["k"].unique():
        for M in df["M"].unique():
            for efC in df["efConstruction"].unique():
                sub = df[(df["k"] == k) & (df["M"] == M) & (df["efConstruction"] == efC)]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 5))
                for method, style in METHOD_STYLE.items():
                    g = sub[sub["method"] == method].copy()
                    if method == "DARTH":
                        g = _darth_slice(g)
                    elif method == "PiP":
                        g = _pip_slice(g)
                    if g.empty:
                        continue
                    g = g.sort_values("Recall@k")
                    ax.plot(g["QPS"], g["Recall@k"], label=method, **style)
                    for _, r in g.iterrows():
                        lbl = f"ef={int(r['efSearch'])}" if r["efSearch"] > 0 else f"Rt={r['Rt']}"
                        ax.annotate(lbl, (r["QPS"], r["Recall@k"]),
                                    textcoords="offset points", xytext=(4, 3), fontsize=6)
                ax.axhline(0.90, color="grey", linestyle=":", linewidth=0.8)
                ax.set_xscale("log")
                styled_ax(ax, xlabel="QPS (log)", ylabel=f"Recall@{k}",
                          title=f"Recall@{k}–QPS — {dataset} (M={M}, efC={efC})")
                savefig(fig, out_dir / f"recall_vs_qps_M{M}_efC{efC}_k{k}")


# ── Plot 3: DARTH Speedup bar chart ───────────────────────────────────────────

def plot_speedup(df: pd.DataFrame, out_dir: Path, dataset: str):
    for k in df["k"].unique():
        for M in df["M"].unique():
            for efC in df["efConstruction"].unique():
                base = (df[(df["method"] == "Baseline-HNSW") & (df["k"] == k)
                           & (df["M"] == M) & (df["efConstruction"] == efC)
                           & (df["efSearch"] > 0)]
                        .set_index("efSearch")["QPS"])
                darth = (df[(df["method"] == "DARTH") & (df["Rt"] == 0.90)
                            & (df["k"] == k) & (df["M"] == M)
                            & (df["efConstruction"] == efC) & (df["efSearch"] > 0)]
                         .set_index("efSearch")["QPS"])
                common = sorted(set(base.index) & set(darth.index))
                if not common:
                    continue
                speedups = [darth[e] / base[e] for e in common]
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar([str(e) for e in common], speedups,
                              color="#d6604d", edgecolor="white")
                ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
                for bar, val in zip(bars, speedups):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.02,
                            f"{val:.2f}×", ha="center", va="bottom", fontsize=9)
                ax.set_xlabel("efSearch", fontsize=11)
                ax.set_ylabel("Speedup (DARTH / Baseline)", fontsize=11)
                ax.set_title(f"DARTH Speedup at Rt=0.90 — {dataset} (M={M}, efC={efC}, k={k})")
                ax.grid(True, alpha=0.3, axis="y")
                fig.tight_layout()
                savefig(fig, out_dir / f"speedup_M{M}_efC{efC}_k{k}")


# ── Plot 4: PiP parameter heatmap ─────────────────────────────────────────────

def plot_pip_heatmap(df: pd.DataFrame, out_dir: Path, dataset: str):
    pip = df[(df["method"] == "PiP") & (df["efSearch"] == 200) & (df["k"] == 10)]
    if pip.empty:
        return
    for M in pip["M"].unique():
        for efC in pip["efConstruction"].unique():
            sub = pip[(pip["M"] == M) & (pip["efConstruction"] == efC)]
            gammas = sorted(sub["pip_gamma"].dropna().unique())
            deltas = sorted(sub["pip_delta"].dropna().unique())
            if not gammas or not deltas:
                continue
            matrix = np.full((len(deltas), len(gammas)), np.nan)
            for i, d in enumerate(deltas):
                for j, g in enumerate(gammas):
                    r = sub[(sub["pip_gamma"] == g) & (sub["pip_delta"] == d)]
                    if not r.empty:
                        matrix[i, j] = r["Recall@k"].values[0]
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.70, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(gammas)))
            ax.set_xticklabels([f"γ={g}" for g in gammas])
            ax.set_yticks(range(len(deltas)))
            ax.set_yticklabels([f"Δ={d}" for d in deltas])
            for i in range(len(deltas)):
                for j in range(len(gammas)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                                fontsize=9,
                                color="black" if matrix[i, j] > 0.85 else "white")
            plt.colorbar(im, ax=ax, label="Recall@10")
            ax.set_title(f"PiP Recall@10 by (γ, Δ) — {dataset} (M={M}, efC={efC})")
            fig.tight_layout()
            savefig(fig, out_dir / f"pip_heatmap_M{M}_efC{efC}")


# ── Plot 5: DARTH Rt sensitivity ───────────────────────────────────────────────

def plot_darth_rt_sensitivity(df: pd.DataFrame, out_dir: Path, dataset: str):
    darth = df[(df["method"] == "DARTH") & (df["efSearch"] == 200) & (df["k"] == 10)]
    base  = df[(df["method"] == "Baseline-HNSW") & (df["efSearch"] == 200) & (df["k"] == 10)]
    if darth.empty or base.empty:
        return
    for M in darth["M"].unique():
        for efC in darth["efConstruction"].unique():
            d_sub  = darth[(darth["M"] == M) & (darth["efConstruction"] == efC)].sort_values("Rt")
            b_qps  = base[(base["M"] == M) & (base["efConstruction"] == efC)]["QPS"]
            if d_sub.empty or b_qps.empty:
                continue
            b_qps_val = float(b_qps.iloc[0])
            fig, ax1 = plt.subplots(figsize=(7, 5))
            ax2 = ax1.twinx()
            ax1.plot(d_sub["Rt"], d_sub["Recall@k"], "s--",
                     color="#d6604d", linewidth=2, label="DARTH Recall@10")
            ax1.axhline(0.90, color="grey", linestyle=":", linewidth=0.8)
            ax2.plot(d_sub["Rt"], d_sub["QPS"] / b_qps_val, "D-.",
                     color="#b2abd2", linewidth=2, label="Speedup vs Baseline")
            ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            ax1.set_xlabel("Target Recall (Rt)", fontsize=11)
            ax1.set_ylabel("Actual Recall@10", fontsize=11, color="#d6604d")
            ax2.set_ylabel("Speedup", fontsize=11, color="#b2abd2")
            ax1.set_title(f"DARTH: Recall & Speedup vs Rt — {dataset} (M={M}, efC={efC})")
            lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
            lbls  = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
            ax1.legend(lines, lbls, fontsize=9)
            ax1.grid(True, alpha=0.3)
            savefig(fig, out_dir / f"darth_rt_sensitivity_M{M}_efC{efC}")


# ── Plot 6: Unified comparison at Recall ≥ 0.90 ───────────────────────────────

def plot_unified_comparison(df: pd.DataFrame, out_dir: Path, dataset: str):
    TARGET, k = 0.90, 10
    records = []
    for method in METHOD_STYLE:
        sub = df[(df["method"] == method) & (df["k"] == k)]
        if method == "DARTH":
            sub = sub[sub["Rt"] == 0.90]
        elif method == "PiP":
            sub = sub[(sub["pip_gamma"] == 95) & (sub["pip_delta"] == 10)]
        if sub.empty:
            continue
        above = sub[sub["Recall@k"] >= TARGET]
        chosen = (above.sort_values("QPS", ascending=False).iloc[0]
                  if not above.empty
                  else sub.sort_values("Recall@k", ascending=False).iloc[0])
        records.append({"method": method, "QPS": chosen["QPS"],
                        "Recall@k": chosen["Recall@k"]})
    if not records:
        return
    res    = pd.DataFrame(records).sort_values("QPS", ascending=False)
    colors = [METHOD_STYLE[m]["color"] for m in res["method"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(res["method"], res["QPS"], color=colors, edgecolor="white")
    for bar, (_, r) in zip(bars, res.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                f"R={r['Recall@k']:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("QPS", fontsize=11)
    ax.set_title(f"QPS at Recall@10 ≥ {TARGET} — {dataset}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, out_dir / "unified_comparison_at_recall_90")


# ── Failure analysis plots ─────────────────────────────────────────────────────

def plot_failure_analysis(fa: pd.DataFrame, out_dir: Path, dataset: str):
    fa_dir = out_dir / "failure_analysis"
    fa_dir.mkdir(parents=True, exist_ok=True)

    failures = fa[fa["is_failure"]]
    normal   = fa[~fa["is_failure"]]
    n_fail, n_total = len(failures), len(fa)

    # FA-1: scatter baseline vs DARTH recall
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(normal["baseline_recall"],   normal["darth_recall"],
               alpha=0.25, s=8,  color="#2166ac", label="Normal queries")
    ax.scatter(failures["baseline_recall"], failures["darth_recall"],
               alpha=0.8,  s=20, color="#d6604d",
               label=f"DARTH failures ({n_fail}/{n_total})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="y = x")
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=0.8, label="Rt=0.90")
    styled_ax(ax, xlabel="Baseline Recall@10", ylabel="DARTH Recall@10",
              title=f"Per-query recall: Baseline vs DARTH — {dataset}")
    savefig(fig, fa_dir / "scatter_baseline_vs_darth")

    # FA-2: difficulty & norm distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, xlabel in [
        (axes[0], "difficulty_1nn_dist", "Distance to true 1-NN"),
        (axes[1], "query_norm",          "Query L2 norm"),
    ]:
        ax.hist(fa[col],       bins=40, alpha=0.5, density=True,
                color="#2166ac", label="All queries")
        ax.hist(failures[col], bins=40, alpha=0.7, density=True,
                color="#d6604d", label=f"Failures ({n_fail})")
        styled_ax(ax, xlabel=xlabel, ylabel="Density",
                  title=xlabel.replace("_", " ").title())
    fig.suptitle(f"DARTH failure characterisation — {dataset}", fontsize=13)
    fig.tight_layout()
    savefig(fig, fa_dir / "difficulty_distribution")

    # FA-3: failure rate by difficulty decile
    fa = fa.copy()
    fa["decile"] = pd.qcut(fa["difficulty_1nn_dist"], q=10,
                           labels=False, duplicates="drop")
    stats = fa.groupby("decile")["is_failure"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(stats.index + 1, stats.values, color="#d6604d", edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Difficulty decile (1=easiest, 10=hardest)", fontsize=11)
    ax.set_ylabel("Failure rate (%)", fontsize=11)
    ax.set_title(f"DARTH failure rate by difficulty decile — {dataset}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, fa_dir / "failure_rate_by_decile")

    # FA-4: CDF of per-query recall
    fig, ax = plt.subplots(figsize=(7, 5))
    for col, label, color in [
        ("baseline_recall", "Baseline HNSW", "#2166ac"),
        ("darth_recall",    "DARTH",         "#d6604d"),
    ]:
        vals = np.sort(fa[col].values)
        ax.plot(vals, np.arange(1, len(vals)+1) / len(vals),
                label=label, color=color, linewidth=2)
    ax.axvline(0.90, color="grey", linestyle=":", linewidth=0.8, label="Rt=0.90")
    styled_ax(ax, xlabel="Recall@10", ylabel="CDF",
              title=f"CDF of per-query Recall@10 — {dataset}")
    savefig(fig, fa_dir / "recall_cdf")

    print(f"    Failure plots → {fa_dir}/")


# ── orchestrator ───────────────────────────────────────────────────────────────

def make_plots(dataset: str, skip_failure: bool = False):
    res_dir = ROOT / "results_csv" / dataset
    out_dir = ROOT / f"plot_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv = res_dir / "all_results.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found — run the experiment first.")
        return

    df = pd.read_csv(csv)
    print(f"\n{'═'*55}")
    print(f"  Plotting: {dataset}  ({len(df)} rows)")
    print(f"  Methods : {sorted(df['method'].unique())}")
    print(f"{'═'*55}")

    plot_recall_vs_efsearch(df, out_dir, dataset)
    plot_recall_vs_qps(df, out_dir, dataset)
    plot_speedup(df, out_dir, dataset)
    plot_pip_heatmap(df, out_dir, dataset)
    plot_darth_rt_sensitivity(df, out_dir, dataset)
    plot_unified_comparison(df, out_dir, dataset)
    print(f"  Method comparison plots → {out_dir}/")

    if not skip_failure:
        fa_csv = res_dir / "failure_analysis.csv"
        if fa_csv.exists():
            plot_failure_analysis(pd.read_csv(fa_csv), out_dir, dataset)
        else:
            print(f"  [skip] failure_analysis.csv not found for {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Generate all thesis plots from CSVs.")
    parser.add_argument("--dataset", nargs="*", default=None,
                        choices=DATASETS, metavar="DATASET",
                        help="Dataset(s) to plot (default: all)")
    parser.add_argument("--no-failure", action="store_true",
                        help="Skip failure analysis plots")
    args = parser.parse_args()

    for ds in (args.dataset or DATASETS):
        make_plots(ds, skip_failure=args.no_failure)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
