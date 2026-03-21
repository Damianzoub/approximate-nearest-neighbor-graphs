import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results_csv/my_algos_cpp_benchmark_results.csv"
OUT_DIR = "my_algorithms_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

needed = {"Method", "M", "efConstruction", "efSearch", "k", "Recall@k", "QPS"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(
        f"CSV is missing columns: {missing}. "
        f"Make sure you saved the full dataframe with Method/M/efConstruction/efSearch/k."
    )

df["Method"] = df["Method"].astype(str).str.strip()

adaef_methods = sorted(df[df["Method"].str.contains("adaef", case=False, na=False)]["Method"].unique().tolist())
darth_methods = sorted(df[df["Method"].str.contains("darth", case=False, na=False)]["Method"].unique().tolist())

base_methods_order = ["hnswlib", "faiss-hnsw", "HNSW NEW"]
methods_order = base_methods_order + darth_methods

other_methods = [m for m in df["Method"].unique().tolist() if m not in methods_order and m not in adaef_methods]
methods_order = methods_order + other_methods

df["Method"] = pd.Categorical(df["Method"], categories=methods_order + adaef_methods, ordered=True)
df = df.sort_values(["M", "efConstruction", "k", "Method", "efSearch"])

groups = df.groupby(["M", "efConstruction", "k"], sort=True)

for (M, efC, k), g in groups:
    # ---- Plot 1: Recall vs efSearch (excluding adaef) ----
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

    # ---- Plot 2: QPS vs efSearch (excluding adaef) ----
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

    # ---- Plot 3: Recall vs QPS (trade-off plot - all methods) ----
    plt.figure()
    for method in methods_order + adaef_methods:
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

    # ---- Plot 4: AdaEF Recall vs target_recall ----
    if adaef_methods:
        plt.figure()
        for method in adaef_methods:
            gm = g[g["Method"] == method]
            if gm.empty:
                continue
            plt.plot(gm["target_recall"], gm["Recall@k"], marker="s", label=method, linestyle="--")

        plt.xlabel("target_recall")
        plt.ylabel("Recall@k")
        plt.title(f"AdaEF: Recall@{k} vs target_recall (M={M}, efC={efC})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"adaef_recall_M{M}_efC{efC}_k{k}.png"), dpi=200)
        plt.close()

        # ---- Plot 5: AdaEF QPS vs target_recall ----
        plt.figure()
        for method in adaef_methods:
            gm = g[g["Method"] == method]
            if gm.empty:
                continue
            plt.plot(gm["target_recall"], gm["QPS"], marker="s", label=method, linestyle="--")

        plt.xlabel("target_recall")
        plt.ylabel("QPS")
        plt.title(f"AdaEF: QPS vs target_recall (M={M}, efC={efC}, k={k})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"adaef_qps_M{M}_efC{efC}_k{k}.png"), dpi=200)
        plt.close()

print(f"Saved plots to: {OUT_DIR}/")
print("Methods plotted (in order):", methods_order)
print("Detected DARTH method(s):", darth_methods)
print("Detected AdaEF method(s):", adaef_methods)
