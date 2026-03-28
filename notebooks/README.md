# Jupyter Notebooks for ANNS Thesis

This directory contains the Jupyter notebooks used for experiments and analysis in my thesis on **Approximate Nearest Neighbor Search (ANNS)**.

## Table of Contents

1. [01_hnsw_baseline.ipynb](#01_hnsw_baselineipynb)
2. [02_darth.ipynb](#02_darthipynb)
3. [03_pip.ipynb](#03_pipipynb)
4. [04_adaef.ipynb](#04_adaefipynb)
5. [05_unified_comparison.ipynb](#05_unified_comparisonipynb)
6. [06_query_difficulty.ipynb](#06_query_difficultyipynb)

---

## Prerequisites

### C++ Build Setup

Some notebooks use C++ implementations wrapped via pybind11 for performance. **Before running any notebook**, you must build the C++ modules:

```bash
# Build HNSW C++ module
cd hnsw_cpp/src
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native -ffast-math" ..
cmake --build . -j
cd ../../..

# Build DARTH C++ module (if using DARTH)
cd hnswDarth_cpp/src
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native -ffast-math" ..
cmake --build . -j
cd ../../..
```

> **Note**: The notebooks will attempt to import these modules automatically. If the build directories don't exist, the imports will fail.

### Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- numpy
- pandas
- matplotlib
- h5py (for some datasets)

---

## Notebook Descriptions

### 01_hnsw_baseline.ipynb

**Purpose**: Establish the plain HNSW baseline performance.

**What it does**:
- Loads the dataset and ground truth
- Builds a standard HNSW index
- Evaluates recall and QPS across different `efSearch` values
- Creates baseline comparison plots

**Key Questions Answered**:
- What is the baseline recall-QPS tradeoff for HNSW?
- How does `efSearch` affect performance?

---

### 02_darth.ipynb

**Purpose**: Evaluate DARTH (Dynamic Adaptive Re-termination using History).

**What it does**:
- Implements early termination based on history features
- Compares against HNSW baseline
- Analyzes the effect of the `Rt` (re-termination threshold) parameter

**Key Questions Answered**:
- Can DARTH reduce search effort without sacrificing recall?
- How does the `Rt` parameter affect the recall-efficiency tradeoff?

**Background**: DARTH uses a predictor trained on search history features to decide when to terminate search early. The `Rt` parameter controls the termination threshold.

---

### 03_pip.ipynb

**Purpose**: Evaluate PiP (Patience in Proximity).

**What it does**:
- Implements saturation-based early termination
- Tests multiple `gamma` (γ) and `delta` (Δ) parameter combinations
- Compares PiP against plain HNSW

**Key Questions Answered**:
- Does PiP reduce search effort?
- How much recall is lost with different patience settings?
- What is the tradeoff between patience parameters and performance?

**Background**: PiP monitors how the top-k result set changes across iterations. When the overlap (φ) between consecutive iterations exceeds γ for Δ consecutive iterations, search terminates early.

---

### 04_adaef.ipynb

**Purpose**: Evaluate Ada-ef (Adaptive Exploration Factor).

**What it does**:
- Separates offline and online phases
- Builds an ef-estimation table based on query difficulty
- Tests different target recall values
- Tracks per-query chosen ef values

**Key Questions Answered**:
- Do all queries need the same `efSearch`?
- Can adaptive ef reduce latency while preserving recall?
- How does query-wise chosen ef vary across the workload?

**Background**: Ada-ef uses the Fundamental Distributional assumption (FDL) to estimate query difficulty. It then looks up the appropriate ef in an offline-computed table to achieve the target recall.

---

### 05_unified_comparison.ipynb

**Purpose**: Central comparison notebook for the thesis.

**What it does**:
- Compares HNSW, DARTH, PiP, and Ada-ef
- Creates unified plots
- Generates summary tables
- Produces the main thesis results

**Key Questions Answered**:
- Which method gives the best recall-speed tradeoff?
- Which method reduces unnecessary search effort?
- Which method is most stable across the workload?

**Output Files**:
- `results_csv/05_unified_comparison_master.csv` - All experimental results
- `results_csv/05_unified_comparison_ranked.csv` - Ranked summary
- `plot_results/05_*.png` - Comparison plots

---

### 06_query_difficulty.ipynb

**Purpose**: Analyze why some queries are harder than others.

**What it does**:
- Runs per-query experiments
- Creates difficulty buckets (easy/medium/hard)
- Compares method performance across difficulty levels
- Explains why adaptive methods help

**Key Questions Answered**:
- Why are some queries easy and others hard?
- How does query difficulty affect recall and latency?
- Why do adaptive methods (PiP, DARTH, Ada-ef) help?

**Output Files**:
- `results_csv/06_per_query_analysis.csv` - Per-query metrics
- `results_csv/06_difficulty_comparison.csv` - Comparison by difficulty
- `plot_results/06_*.png` - Analysis plots

---

## Dataset

The notebooks use the **siftsmall** dataset by default, located in `../Datasets/siftsmall/`.

To use a different dataset:
1. Add the dataset files to `../Datasets/`
2. Modify the `DATASET_NAME` variable in each notebook

---

## Output Directories

All results are saved to:

- `../results_csv/` - CSV files with experimental results
- `../plot_results/` - PNG plots

---

## Running the Notebooks

1. Ensure C++ modules are built (see Prerequisites above)
2. Open Jupyter:
   ```bash
   cd ..
   jupyter notebook
   ```
3. Navigate to the `notebooks/` directory
4. Open and run notebooks in order (recommended):
   1. `01_hnsw_baseline.ipynb` - Establish baseline
   2. `02_darth.ipynb` - Evaluate DARTH
   3. `03_pip.ipynb` - Evaluate PiP
   4. `04_adaef.ipynb` - Evaluate Ada-ef
   5. `05_unified_comparison.ipynb` - Unified comparison
   6. `06_query_difficulty.ipynb` - Difficulty analysis

---

## Algorithm Summary

| Algorithm | Early Termination | Adaptation | Offline Phase |
|-----------|-------------------|------------|---------------|
| HNSW | No | No | No |
| DARTH | Yes (history-based) | Yes | Optional |
| PiP | Yes (saturation-based) | No | No |
| Ada-ef | Yes (difficulty-based) | Yes | Yes |

---

## Troubleshooting

**ImportError: No module named 'hnsw_cpp'**
- The C++ module hasn't been built. See Prerequisites above.

**MemoryError during index construction**
- Try with a smaller dataset or reduce the number of base vectors.

**Low recall values**
- Increase `efSearch` or adjust the adaptive method parameters.

---

## License

This project is for academic research purposes as part of a thesis.
