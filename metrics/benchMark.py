import time 
import numpy as np

def recall_at_k(I_true,I_pred,k):
    hits = 0
    for t,p in zip(I_true,I_pred):
        hits += len(set(t[:k]).intersection(p[:k]))
    return hits/ (I_true.shape[0]*k)

def measure(search_fn, Xq, k, I_true=None, warmup=1):
    # warmup (important for fair QPS)
    for _ in range(warmup):
        search_fn(Xq[:min(len(Xq), 64)], k)

    t0 = time.perf_counter()
    D, I = search_fn(Xq, k)
    t1 = time.perf_counter()

    total_s = t1 - t0
    qps = len(Xq) / total_s if total_s > 0 else float("inf")
    rec = recall_at_k(I_true, I, k) if I_true is not None else np.nan

    return {
        "QPS": qps,
        "Total Time (s)": total_s,
        "Recall@k": rec,
    }