import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "benchmark_results_1M.csv"
OUT_DIR = "plots1Mvectors"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# sanity checks
needed = {"Method", "M", "efConstruction", "efSearch", "k", "Recall@k", "QPS"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing columns: {missing}. "
                     f"Make sure you saved the full dataframe with Method/M/efConstruction/efSearch/k.")

# Ensure consistent ordering
df = df.sort_values(["M", "efConstruction", "k", "Method", "efSearch"])

methods_order = ["hnswlib", "faiss-hnsw", "HNSW NEW"]
df["Method"] = pd.Categorical(df["Method"], categories=methods_order, ordered=True)

groups = df.groupby(["M", "efConstruction", "k"], sort=True)

for (M, efC, k), g in groups:
    # ---- Plot 1: Recall vs efSearch ----
    plt.figure()
    for method in methods_order:
        gm = g[g["Method"] == method]
        if gm.empty:
            continue
        plt.plot(gm["efSearch"], gm["Recall@k"], marker="o", label=method)

    plt.xlabel("efSearch")
    plt.ylabel("Recall@k")
    plt.title(f"Recall@{k} vs efSearch (M={M}, efC={efC})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"recall_M{M}_efC{efC}_k{k}.png"), dpi=200)
    plt.close()

    # ---- Plot 2: QPS vs efSearch ----
    plt.figure()
    for method in methods_order:
        gm = g[g["Method"] == method]
        if gm.empty:
            continue
        plt.plot(gm["efSearch"], gm["QPS"], marker="o", label=method)

    plt.xlabel("efSearch")
    plt.ylabel("QPS")
    plt.title(f"QPS vs efSearch (M={M}, efC={efC}, k={k})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"qps_M{M}_efC{efC}_k{k}.png"), dpi=200)
    plt.close()

        # ---- Plot 3: Recall vs QPS (trade-off plot) ----
    plt.figure()
    for method in methods_order:
        gm = g[g["Method"] == method]
        if gm.empty:
            continue
        plt.plot(gm["Recall@k"],gm["QPS"], marker="o", label=method)

    plt.xlabel("Recall@k")
    plt.ylabel("QPS")
    plt.title(f"Recall@{k} vs QPS (M={M}, efC={efC})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"recall_vs_qps_M{M}_efC{efC}_k{k}.png"), dpi=200)
    plt.close()


print(f"Saved plots to: {OUT_DIR}/")
