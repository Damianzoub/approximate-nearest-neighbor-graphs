# Approximate Nearest Neighbor Graphs

## Overview

This repository contains implementations and benchmarks for my thesis on **Approximate Nearest Neighbor Search (ANNS)**. ANN graphs are used to efficiently find approximate nearest neighbors in high-dimensional spaces, making them useful for applications like recommendation systems, image retrieval, clustering, and NLP.

## Implemented Algorithms

### 1. HNSW (Hierarchical Navigable Small World)

**Description**: The foundational ANN graph algorithm using a multi-layer hierarchical structure with small-world navigation.

**Key Features**:
- Multi-layer construction with probability-based level assignment
- Greedy search on upper layers, beam search on bottom layer
- Parameters: `M` (connections per node), `efConstruction`, `efSearch`

**Files**:
- `hnsw_construction.py` - Pure Python implementation
- `hnsw_cpp/` - C++ optimized implementation

### 2. DARTH (Dynamic Adaptive Re-termination)

**Description**: Early termination strategy using history features to predict when search can safely stop.

**Key Features**:
- Monitors search history features (distance statistics, iteration count)
- Uses a predictor to estimate termination confidence
- Parameters: `Rt` (re-termination threshold), `ipi` (initial prediction interval), `mpi` (minimum prediction interval)

**Files**:
- `hsnw_constructionDARTH.py` - Python implementation
- `hnswDarth_cpp/` - C++ optimized implementation

### 3. PiP (Patience in Proximity)

**Description**: Saturation-based early termination that monitors when the top-k result set stops changing.

**Key Features**:
- Tracks overlap between consecutive iterations: φ_h,l(q) = 100 × |N_{h-1,l}(q) ∩ N_{h,l}(q)| / k
- Stops when φ ≥ γ for Δ consecutive iterations
- Parameters: `pip_gamma` (γ, saturation threshold), `pip_delta` (Δ, patience)

**Files**:
- `hnsw_pip.py` - Python implementation

### 4. Ada-ef (Adaptive Exploration Factor)

**Description**: Adapts the exploration factor based on estimated query difficulty using the Fundamental Distributional assumption (FDL).

**Key Features**:
- **Offline phase**: Builds ef-estimation table from sampled queries
- **Online phase**: Estimates query difficulty and looks up appropriate ef
- Parameters: `target_recall`, `adaef_bins`, `adaef_delta`

**Files**:
- `hnsw_adaef.py` - Python implementation

---

## C++ Build Setup

The C++ implementations provide significant speedups for index construction and search. **Before running any notebook or benchmark, you must build the C++ modules.**

### Build Instructions

#### HNSW C++ Module

```bash
cd hnsw_cpp/src
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native -ffast-math" ..
cmake --build . -j
cd ../../..
```

#### DARTH C++ Module

```bash
cd hnswDarth_cpp/src
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native -ffast-math" ..
cmake --build . -j
cd ../../..
```

### Build Options

| Flag | Description |
|------|-------------|
| `-O3` | Maximum optimization |
| `-DNDEBUG` | Disable assertions |
| `-march=native` | CPU-specific optimizations |
| `-ffast-math` | Fast floating-point math |
| `-j` | Parallel build (uses all cores) |

---

## Project Structure

```
approximate-nearest-neighbor-graphs/
├── hnsw_construction.py      # HNSW Python implementation
├── hsnw_constructionDARTH.py # DARTH Python implementation
├── hnsw_pip.py               # PiP Python implementation
├── hnsw_adaef.py             # Ada-ef Python implementation
├── hnsw_cpp/                 # HNSW C++ source
├── hnswDarth_cpp/            # DARTH C++ source
├── testing/                  # Testing and comparison scripts
├── utils/                    # Utility functions
├── metrics/                  # Benchmarking metrics
├── notebooks/                # Jupyter notebooks
├── results_csv/              # CSV results
├── plot_results/             # Generated plots
├── Datasets/                 # Dataset files
└── README.md                # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CMake 3.15+ (for C++ modules)
- C++ compiler with C++17 support
- Required libraries (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/approximate-nearest-neighbor-graphs.git
   cd approximate-nearest-neighbor-graphs
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build C++ modules (see C++ Build Setup above)

4. Download or prepare your dataset in `Datasets/`

### Running Experiments

#### Using Jupyter Notebooks (Recommended)

```bash
jupyter notebook
```

Then navigate to `notebooks/` and run the notebooks in order:
1. `01_hnsw_baseline.ipynb` - Baseline HNSW
2. `02_darth.ipynb` - DARTH evaluation
3. `03_pip.ipynb` - PiP evaluation
4. `04_adaef.ipynb` - Ada-ef evaluation
5. `05_unified_comparison.ipynb` - Unified comparison
6. `06_query_difficulty.ipynb` - Query difficulty analysis

#### Using Python Scripts

```bash
# Run benchmarks
python testing/run_comparison.py

# Generate plots
python plot_creation_csv.py
```

---

## Datasets

The project uses ANN-benchmark format datasets:

- **siftsmall**: Small SIFT subset (default)
- **sift**: Full SIFT dataset
- **gist**: GIST descriptors
- **glove**: GloVe embeddings

Place datasets in `Datasets/<dataset_name>/` with files:
- `<name>_base.fvecs` - Base vectors
- `<name>_query.fvecs` - Query vectors
- `<name>_groundtruth.ivecs` - Ground truth

Or use the download scripts in `scripts/`.

---

## Benchmark Metrics

The following metrics are computed:

| Metric | Description |
|--------|-------------|
| Recall@K | Fraction of true neighbors found in top-K |
| QPS | Queries per second |
| Latency | Time per query (ms) |
| Build Time | Index construction time |

---

## Algorithm Comparison

| Algorithm | Early Termination | Adaptive | Offline Phase | Best For |
|-----------|-------------------|----------|--------------|----------|
| HNSW | No | No | No | Baseline, simple use cases |
| DARTH | Yes | Yes | Optional | Reducing effort on easy queries |
| PiP | Yes | No | No | Simple early stopping |
| Ada-ef | Yes | Yes | Yes | Consistent recall targets |

---

## Key Research Questions

This thesis investigates:

1. **Trade-offs**: How do different early termination strategies affect the recall-efficiency trade-off?

2. **Query Difficulty**: Why are some queries inherently harder than others, and can we identify them?

3. **Adaptation**: Can adaptive methods that adjust to query difficulty outperform fixed-parameter methods?

4. **Stability**: Which methods provide consistent performance across different query types?

---

## Troubleshooting

### ImportError: No module named 'hnsw_cpp'

The C++ module hasn't been built. Run the build commands in the **C++ Build Setup** section above.

### ImportError: No module named 'hnswDarth_cpp'

The DARTH C++ module hasn't been built. Run the DARTH build commands above.

### Out of Memory

- Reduce dataset size
- Use a machine with more RAM
- Process in batches

### Low Recall

- Increase `efSearch`
- Adjust adaptive method parameters
- Check metric compatibility (some methods use cosine by default)

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work is part of my thesis on Approximate Nearest Neighbor Search. Special thanks to:
- The original HNSW paper authors
- Authors of the DARTH, PiP, and Ada-ef papers
- The ANN-benchmark community for datasets
