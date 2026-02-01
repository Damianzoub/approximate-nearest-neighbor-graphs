import pandas as pd

def create_results_df(rows):
    df = pd.DataFrame(rows)
    recall_cols = [c for c in df.columns if c.startswith("Recall@")]

    order_cols = [
        "Method",
        "M",
        "efConstruction",
        "efSearch",
        "k",
        *recall_cols,
        "QPS",
        "Total Time (s)"
    ]

    df = df[order_cols].sort_values(
        ["Method","M","efConstruction","efSearch","k"]
    )
    df = df.set_index(["Method", "M", "efConstruction", "efSearch", "k"])
    return df

def fancy_display(df):
    fmt = {
        "QPS":" {:,.2f}",
        "Total Time (s)" : "{:.4f}"
    }

    for col in df.columns:
        if col.startswith("Recall@"):
            fmt[col] = "{:.4f}"

    return (
        df.style
        .format(fmt)
        .set_caption("ANN Evaluation: QPS and Recall@k (higher is better)")
    )