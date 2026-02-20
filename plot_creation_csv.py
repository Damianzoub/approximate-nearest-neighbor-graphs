import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "DarthPerformance_vs_others.csv"
OUT_DIR = "HNSWDARTH_comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# sanity checks
needed = {"Method", "M", "efConstruction", "efSearch", "k", "Recall@k", "QPS"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(
        f"CSV is missing columns: {missing}. "
        f"Make sure you saved the full dataframe with Method/M/efConstruction/efSearch/k."
    )

# Normalize method names a bit (optional but helps if you have small variations)
df["Method"] = df["Method"].astype(str).str.strip()

# Try to detect the DARTH method name(s) present in the CSV
# Anything containing 'darth' (case-insensitive) will be treated as DARTH
darth_methods = sorted(df[df["Method"].str.contains("darth", case=False, na=False)]["Method"].unique().tolist())

# Base order you want
base_methods_order = ["hnswlib", "faiss-hnsw", "HNSW NEW"]

# Add DARTH methods (if any) to the order. If none exist, we still keep plotting others.
methods_order = base_methods_order + darth_methods

# Also include any other methods found in the CSV (optional; keeps plots complete)
# Put them at the end so you don't lose anything if you renamed methods.
other_methods = [m for m in df["Method"].unique().tolist() if m not in methods_order]
methods_order = methods_order + other_methods

# Make categorical with ALL methods we plan to plot (so nothing becomes NaN)
df["Method"] = pd.Categorical(df["Method"], categories=methods_order, ordered=True)

# Ensure consistent ordering
df = df.sort_values(["M", "efConstruction", "k", "Method", "efSearch"])

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
        plt.plot(gm["Recall@k"], gm["QPS"], marker="o", label=method)

    plt.xlabel("Recall@k")
    plt.ylabel("QPS")
    plt.title(f"Recall@{k} vs QPS (M={M}, efC={efC})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"recall_vs_qps_M{M}_efC{efC}_k{k}.png"), dpi=200)
    plt.close()

print(f"Saved plots to: {OUT_DIR}/")
print("Methods plotted (in order):", methods_order)
print("Detected DARTH method(s):", darth_methods)