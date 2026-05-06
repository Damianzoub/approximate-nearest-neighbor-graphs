# Plots Guide — `plot_creation_csv.py`

Each plot is saved as both `.png` and `.pdf` in `plot_{dataset}/` (method comparison plots) and `plot_{dataset}/failure_analysis/` (failure case plots).

Run with:
```bash
python plot_creation_csv.py                        # all datasets
python plot_creation_csv.py --dataset siftsmall    # one dataset
python plot_creation_csv.py --no-failure           # skip failure plots
```

---

## Method Comparison Plots (`plot_{dataset}/`)

### 1. `recall_vs_efsearch_M{M}_efC{efC}_k{k}`

**What it shows:** How Recall@k changes as efSearch increases, for each method.

**Why it matters:** efSearch controls how many candidates HNSW explores. A well-designed early-termination method (DARTH) or pruning method (PiP) should reach its maximum recall faster (at lower efSearch) than the Baseline, meaning fewer distance computations for the same quality.

**What to look for:**
- DARTH and PiP curves should plateau earlier than Baseline-HNSW.
- Ada-ef does not have a fixed efSearch (it adapts), so it appears as a single point.
- The horizontal dashed line at 0.90 marks the target recall operating point used throughout the thesis.

---

### 2. `recall_vs_qps_M{M}_efC{efC}_k{k}`

**What it shows:** The recall–throughput Pareto frontier for each method. X-axis is Queries Per Second (log scale), Y-axis is Recall@k.

**Why it matters:** This is the primary benchmark in the ANN literature — a method is better if it achieves higher recall at the same QPS, or higher QPS at the same recall. The Pareto-optimal curve dominates everything to its left and below.

**What to look for:**
- Methods whose curves are pushed toward the top-right corner are superior.
- DARTH and PiP should be to the right of Baseline-HNSW at high recall (≥ 0.90), showing they are faster at the same quality.
- Ada-ef appears as individual points (one per target recall), showing where it sits on the frontier.
- Annotations show the efSearch or Rt value at each operating point.

---

### 3. `speedup_M{M}_efC{efC}_k{k}`

**What it shows:** DARTH's QPS speedup relative to Baseline-HNSW at the same efSearch value, evaluated at Rt=0.90.

**Why it matters:** Directly quantifies DARTH's computational saving — how many times faster it is than vanilla HNSW for the same configuration.

**What to look for:**
- Bars above 1.0× indicate DARTH is faster than Baseline.
- Larger speedups at higher efSearch values are expected because there is more search effort for DARTH to save.
- If speedup is close to 1.0 or below, DARTH is not effective for this dataset/configuration.

---

### 4. `pip_heatmap_M{M}_efC{efC}`

**What it shows:** PiP Recall@10 for all combinations of `pip_gamma` (x-axis) and `pip_delta` (y-axis), at efSearch=200.

**Why it matters:** PiP has two hyperparameters that control the aggressiveness of candidate pruning. This heatmap reveals the sensitivity to these parameters and identifies the best configuration for each dataset.

**What to look for:**
- Green cells (high recall) in the top-right indicate that high gamma and delta are safe.
- Red cells in the bottom-left indicate that aggressive pruning (low gamma, low delta) degrades recall.
- The best operating point for PiP can be read off directly from this plot.

---

### 5. `darth_rt_sensitivity_M{M}_efC{efC}`

**What it shows:** How DARTH's actual Recall@10 (left axis) and speedup over Baseline (right axis) change as the target recall threshold Rt is varied, at efSearch=200.

**Why it matters:** Rt is DARTH's primary control parameter. This plot shows whether the predictor reliably achieves the requested recall and what the cost-benefit tradeoff is — lower Rt gives more speedup but risks lower actual recall.

**What to look for:**
- Actual recall should closely track Rt — if it drops well below the dashed line, the predictor is over-aggressive.
- Speedup should increase as Rt decreases (more early terminations).
- A well-calibrated DARTH achieves actual recall ≈ Rt with speedup > 1.0 across the range.

---

### 6. `unified_comparison_at_recall_90`

**What it shows:** Bar chart of QPS for each method at the best operating point that achieves Recall@10 ≥ 0.90. This is the single-number summary used in the thesis results table.

**Why it matters:** Provides a single fair comparison at a fixed quality target. Methods that achieve 0.90 recall at higher QPS are strictly better for production deployment.

**What to look for:**
- The tallest bar is the fastest method at this quality level.
- Recall annotations on each bar confirm that the 0.90 threshold is met.
- If a method has no bar, it could not reach 0.90 recall in this configuration.

---

## Failure Analysis Plots (`plot_{dataset}/failure_analysis/`)

These plots are DARTH-specific. They analyse per-query behaviour at the fixed operating point efSearch=200, Rt=0.90.

A **failure** is defined as: a query where DARTH recall < 0.90 but Baseline recall ≥ 0.90. These are cases where DARTH's predictor incorrectly declared early termination.

---

### FA-1. `scatter_baseline_vs_darth`

**What it shows:** Scatter plot of per-query Baseline recall (x-axis) vs DARTH recall (y-axis). Failures are highlighted in red.

**Why it matters:** Shows the correlation between Baseline and DARTH recall at the query level. Most queries should cluster along the diagonal (y = x), meaning DARTH and Baseline agree. Failures (red) appear below the Rt=0.90 line.

**What to look for:**
- Queries that fall below the dashed diagonal: DARTH is doing worse than Baseline.
- The density of red points relative to blue indicates the overall failure rate.
- Queries far below the diagonal are severe failures worth investigating.

---

### FA-2. `difficulty_distribution`

**What it shows:** Side-by-side density histograms comparing all queries (blue) vs failure queries (red) on two difficulty axes:
- **Distance to true 1-NN** — queries with a distant nearest neighbour are harder (search must go deeper).
- **Query L2 norm** — captures whether query magnitude is related to failure.

**Why it matters:** Identifies which types of queries DARTH fails on. If failures concentrate in the high-difficulty tail, this is expected and justifies the design. Unexpected failure patterns suggest predictor weaknesses.

**What to look for:**
- Failures should skew toward higher 1-NN distances (harder queries).
- If failures are uniformly distributed across difficulty, the predictor is not working as intended.

---

### FA-3. `failure_rate_by_decile`

**What it shows:** Bar chart of failure rate (%) in each decile of query difficulty (sorted by distance to true 1-NN). Decile 1 = easiest queries, decile 10 = hardest.

**Why it matters:** The ideal DARTH behaviour is: 0% failure rate on easy queries (decile 1–5), increasing failure rate only on hard queries (decile 8–10). This plot directly validates whether the predictor has learned query difficulty.

**What to look for:**
- Low or zero failure rate in deciles 1–5: good — easy queries are handled correctly.
- Rising failure rate toward decile 10: expected — hard queries are where early termination is risky.
- Flat or inverted pattern: the predictor is not calibrated to query difficulty.

---

### FA-4. `recall_cdf`

**What it shows:** Cumulative Distribution Function of per-query Recall@10 for Baseline-HNSW (blue) and DARTH (red).

**Why it matters:** Shows the full distribution of per-query recall quality — not just the average. A good DARTH should have its CDF close to the Baseline CDF for the majority of queries, with only a small tail of low-recall queries.

**What to look for:**
- The two curves should be close together for recall values ≥ 0.90.
- The gap between the curves at low recall values (< 0.80) quantifies how many queries DARTH handles poorly.
- A steep CDF at high recall means most queries achieve near-perfect recall.
