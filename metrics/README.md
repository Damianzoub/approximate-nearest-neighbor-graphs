# Per-Query Benchmarking & Failure Analysis

This directory contains the infrastructure for detailed per-query analysis of ANNS algorithms, specifically designed to understand **when and why early termination methods fail**.

## Overview

The goal of this analysis is to:

1. **Identify** which queries fail across different methods
2. **Categorize** failures by type (boundary, sparse, high-dispersion, outlier)
3. **Quantify** the impact of early termination on recall
4. **Design** targeted improvements based on findings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Failure Analysis Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │ Per-Query       │    │ Query           │               │
│  │ Benchmark       │    │ Difficulty      │               │
│  │                 │    │ Analyzer        │               │
│  │ • Recall        │    │                 │               │
│  │ • Latency      │    │ • d_ratio       │               │
│  │ • Termination   │    │ • Local density │               │
│  │ • Features      │    │ • Isolation     │               │
│  └────────┬────────┘    └────────┬────────┘               │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                     │
│           ┌─────────────────────┐                          │
│           │ Failure Analyzer     │                          │
│           │                     │                          │
│           │ • Pattern detection │                          │
│           │ • Cross-method     │                          │
│           │ • Recommendations   │                          │
│           └─────────────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files

### Core Modules

| File | Description |
|------|-------------|
| `per_query_benchmark.py` | Main benchmarking class for collecting per-query metrics |
| `query_difficulty.py` | Query characterization and failure analysis |
| `instrumented_search.py` | Instrumented wrappers for search algorithms |

### Scripts

| File | Description |
|------|-------------|
| `run_per_query_experiments.py` | Main experiment runner |

## Usage

### Quick Start

```bash
# Run with default settings (siftsmall, 100 queries)
python run_per_query_experiments.py

# More comprehensive test
python run_per_query_experiments.py --dataset sift --max-queries 500 --k 10
```

### Command Line Options

```bash
python run_per_query_experiments.py [OPTIONS]

Options:
  -d, --dataset TEXT     Dataset name (default: siftsmall)
  -n, --max-queries INT  Maximum queries to test (default: 100)
  -k, --k INT            k for k-NN search (default: 10)
  -o, --output-dir TEXT  Output directory (default: results_csv/per_query)
  -s, --seed INT         Random seed (default: 42)
```

### Python API

```python
from metrics.per_query_benchmark import PerQueryBenchmark, PerQueryResult
from metrics.query_difficulty import QueryDifficultyAnalyzer, FailureAnalyzer

# Create benchmark
benchmark = PerQueryBenchmark()

# Run queries and collect results
for i, query in enumerate(queries):
    result = benchmark.run_query(
        search_fn=wrapped_search,
        query=query,
        k=10,
        query_id=i,
        ground_truth=ground_truth[i]
    )

# Analyze query difficulty
analyzer = QueryDifficultyAnalyzer(X_base)
chars = analyzer.analyze_batch(queries, ground_truth)

# Run failure analysis
failure_analyzer = FailureAnalyzer(chars, results_df)
patterns = failure_analyzer.analyze_failure_patterns()
recommendations = failure_analyzer.recommend_improvements()
```

## Metrics Collected

### Per-Query Results

| Metric | Description |
|--------|-------------|
| `query_id` | Unique query identifier |
| `method` | Algorithm/method name |
| `recall` | Recall@k achieved |
| `latency_ms` | Query latency in milliseconds |
| `terminated_early` | Whether early termination was triggered |
| `actual_ef_used` | Actual exploration factor used |
| `num_iterations` | Number of search iterations |
| `num_distance_computations` | Distance calculations performed |
| `num_nodes_visited` | Graph nodes visited |
| `distance_ratio` | d_k / d_1 (dispersion) |

### Query Characteristics

| Characteristic | Description | Impact on Difficulty |
|----------------|-------------|---------------------|
| `d1nn` | Distance to 1-NN | High value → isolated query |
| `dknn` | Distance to k-NN | Used for dispersion |
| `d_ratio` | dknn / d1nn | High → neighbors far apart |
| `local_density` | Points in 2*d1nn radius | Low → sparse region |
| `isolation_score` | d1nn / global_mean_dist | High → outlier |
| `intrinsic_dim` | Variance of nn distances | High → high-dimensional |
| `difficulty` | Classification (easy/medium/hard/very_hard) | - |

## Failure Types

### 1. High Dispersion (`high_dispersion`)

**Definition:** The k nearest neighbors are far apart from each other (d_ratio > 2.5)

**Why it fails:** Early termination may stop before finding all scattered neighbors.

**Example:** Query in a region where neighbors belong to different clusters.

### 2. Sparse Region (`sparse_region`)

**Definition:** Few points in the local neighborhood (local_density < 10)

**Why it fails:** The search may exhaust available nearby nodes quickly, leading to premature termination.

**Example:** Query in a gap between clusters.

### 3. Outlier (`outlier`)

**Definition:** Query is far from all data points (isolation_score > 3.0)

**Why it fails:** The algorithm may terminate early thinking it has found good results, when it hasn't.

**Example:** Query for a concept not well-represented in the dataset.

### 4. Boundary (`boundary`)

**Definition:** Query near cluster boundaries

**Why it fails:** Neighbors may come from multiple clusters with varying densities.

**Example:** Query on the edge of two overlapping clusters.

## Output Files

After running experiments, you'll find:

```
results_csv/per_query/
├── per_query_results_<dataset>_<timestamp>.csv
│   └── All per-query metrics (one row per query × method)
│
├── query_characteristics_<dataset>_<timestamp>.csv
│   └── Query difficulty features (one row per query)
│
└── analysis_<dataset>_<timestamp>.json
    └── Failure patterns and recommendations
```

## Analysis Output

### Summary Statistics

```json
{
  "total_failures": 15,
  "failure_rate": 0.15,
  "by_method": {
    "hnsw_ef50": {"count": 5, "rate": 0.05},
    "darth_Rt0.90": {"count": 18, "rate": 0.18}
  },
  "by_failure_type": {
    "high_dispersion": 8,
    "sparse_region": 4,
    "boundary": 3
  },
  "cross_method_failure_rate": 0.02
}
```

### Recommendations

The system generates actionable recommendations:

```
1. DARTH fails most on high-dispersion queries (8/15 failures).
   Consider: Disable early termination for queries with d_ratio > 2.0

2. PiP has high failure rate on sparse regions (4/15 failures).
   Consider: Increase pip_delta in low-density areas

3. 2% of queries fail in ALL methods (fundamentally hard).
   Consider: Use higher efSearch as default for these queries
```

## Integration with Thesis

### Chapter 4: Failure Analysis Section

```latex
\section{Ανάλυση Αποτυχιών}

\subsection{Μεθοδολογία}
Πραγματοποιήθηκε ανάλυση ανά ερώτημα για την κατανόηση 
των περιπτώσεων αποτυχίας των μεθόδων πρόωρου τερματισμού.

\subsection{Χαρακτηριστικά Δυσκολίας}
\begin{itemize}
    \item \textbf{d\_ratio}: Λόγος απόστασης $d_k / d_1$
    \item \textbf{Τοπική πυκνότητα}: Πλήθος σημείων στη γειτονιά
    \item \textbf{Βαθμός απομόνωσης}: Κανονικοποιημένη απόσταση
\end{itemize}

\subsection{Τύποι Αποτυχίας}
\begin{table}[h]
    \centering
    \caption{Κατηγοριοποίηση αποτυχιών}
    \begin{tabular}{l|c|p{5cm}}
        \toprule
        Τύπος & Πλήθος & Περιγραφή \\
        \midrule
        High Dispersion & 8 & Κοντινοί γείτονες μακριά μεταξύ τους \\
        Sparse Region & 4 & Αραιή περιοχή \\
        Boundary & 3 & Οριακό ερώτημα \\
        \bottomrule
    \end{tabular}
\end{table}
```

## Next Steps

After running the analysis:

1. **Review failure patterns** in the generated JSON
2. **Identify root causes** for each failure type
3. **Design targeted improvements**:
   - Hybrid termination criteria
   - Query-adaptive parameters
   - Robustness to edge cases
4. **Validate improvements** by rerunning experiments

## Dependencies

```
numpy
pandas
scipy (for pdist in difficulty analysis)
```

## Notes

- The analysis is **hardware-independent** - metrics like distance computations are stable across platforms
- Per-query analysis is **computationally expensive** - use `--max-queries` for initial exploration
- Ground truth is **required** for recall calculation
- The system is designed for **offline analysis** - not for production use

## References

- DARTH: Dynamic Adaptive Re-termination
- PiP: Patience in Proximity
- Query difficulty estimation based on intrinsic dimensionality and local density
