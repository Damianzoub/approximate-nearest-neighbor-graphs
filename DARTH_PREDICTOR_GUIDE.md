# DARTH Predictor — Complete Implementation Guide

**DARTH** = *Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search*  
**Paper**: Chatzakis et al., SIGMOD 2026 (arXiv 2505.19001)  
**Official repo**: [MChatzakis/DARTH](https://github.com/MChatzakis/DARTH)  
**This project's implementation**: `darth/`, `hnswDarth_cpp/`, `hsnw_constructionDARTH.py`

---

## Table of Contents

1. [Architecture Summary](#1-architecture-summary)
2. [Where the Predictor Lives (Official Repo)](#2-where-the-predictor-lives-official-repo)
3. [How the Predictor Works](#3-how-the-predictor-works)
4. [Python Files to Study](#4-python-files-to-study)
5. [C++ / Header Files to Study](#5-c--header-files-to-study)
6. [How to Download and Build the Official Repo](#6-how-to-download-and-build-the-official-repo)
7. [This Project's Implementation](#7-this-projects-implementation)
8. [Start-Here Sequence](#8-start-here-sequence)
9. [How to Port to a New Project](#9-how-to-port-to-a-new-project)
10. [Risks and Known Gaps](#10-risks-and-known-gaps)

---

## 1. Architecture Summary

```
OFFLINE (Python)
─────────────────────────────────────────────────────
hsnw_constructionDARTH.py      ← HNSW_DARTH with _search_layer_darth

darth/collect_training_data.py ← run full HNSW, log features + recall@k per step
       ↓ saves CSV
darth/train_predictor.py       ← LGBMRegressor(n_estimators=100).fit(X, y)
       ↓ saves .txt (LightGBM text-model format)

ONLINE (C++ or Python)
─────────────────────────────────────────────────────
darth/predictor.py             ← LGBMPredictor loads .txt model, predict(feats) → float
       ↓ passed as callable
hnswDarth_cpp (pybind11)       ← PyPredictor bridges Python callable → C++ IPredictor
       ↓
HNSW_DARTH::search_layer_darth ← every N distance-calcs:
                                     features = darth_extract_features(result_heap)
                                     if predictor.predict(features) >= Rt: break
```

---

## 2. Where the Predictor Lives (Official Repo)

### Primary definition file
**`faiss/impl/DeclarativeRecall.h`** — contains all `typedef struct` definitions.
This is the single most important file in the official repo.

```
faiss/impl/DeclarativeRecall.h   ← struct DARTHPredictorHNSW, DARTHPredictorIVF,
                                     DeclarativeRecallDataCollectorHNSW,
                                     DeclarativeRecallDataManager
                                     includes <LightGBM/c_api.h>
                                     member: BoosterHandle booster

faiss/impl/DeclarativeRecall.cpp ← LGBM_BoosterCreateFromModelfile()
                                     LGBM_BoosterPredictForMatSingleRow()
                                     adaptive interval formula
                                     feature assembly

faiss/impl/HNSW.h               ← forward decls + search_DARTH() signatures
faiss/impl/HNSW.cpp             ← search_from_candidates_DARTH(), interval counter,
                                     predict_recall() call, early-exit check

faiss/IndexHNSW.h/cpp           ← public IndexHNSW::search_DARTH(DARTHPredictorHNSW)

hnsw-test/hnsw_test.cpp         ← CLI driver, constructs DARTHPredictorHNSW,
                                     --predictor-model-path argument
```

### Training / Python side
```
notebooks_scripts/predictor_training.py   ← trains LightGBM, saves .txt model
notebooks_scripts/predictor_validation.py ← evaluates model on validation queries
notebooks_scripts/extract_data_stats.py   ← parses training CSV stats
experiments/hnsw_training_data_generation.sh ← runs hnsw_test in data-collection mode
experiments/hnsw_darth_test.sh            ← runs DARTH experiments
experiments/tuning.py                     ← binary-searches optimal ipi/mpi per dataset
```

---

## 3. How the Predictor Works

### What it predicts
The GBDT predicts **current recall@k** — what fraction of the true k nearest neighbours are already in the result heap at a given point during HNSW graph traversal.

### When it is called
Inside the beam-search loop on layer 0, a counter `idis` tracks distance computations since the last prediction. When `idis % prediction_interval == 0`, the predictor is invoked. Before the heap has k elements (`inserts < k`), the call is skipped and 0.0 is returned.

### Input features — the 11-column vector

| # | C++ field | Python key (bindings) | Python key (pure-Python path) | Description |
|---|-----------|----------------------|-------------------------------|-------------|
| 1 | `nstep` | `nstep` | `nstep` | Search-step counter |
| 2 | `ndis` | `ndis` | `ndis` | Distance computations so far |
| 3 | `ninserts` | `ninserts` | `inserts` | Candidates inserted into heap |
| 4 | `firstNN` | `firstNN` | `firstNN` | Distance to first neighbour found |
| 5 | `closestNN` | `closestNN` | `closestNN` | Minimum distance in current heap |
| 6 | `furthestNN` | `furthestNN` | `furthestNN` | Maximum distance in current heap |
| 7 | `meanNN` | `meanNN` | `avg` | Mean of heap distances |
| 8 | `varNN` | `varNN` | `var` | Variance of heap distances |
| 9 | `p25NN` | `p25NN` | `perc25` | 25th-percentile heap distance |
| 10 | `p50NN` | `p50NN` | `med` | Median heap distance |
| 11 | `p75NN` | `p75NN` | `perc75` | 75th-percentile heap distance |

### Adaptive prediction intervals
After each prediction the interval for the next call shrinks as recall approaches the target:

```
prediction_interval = mpi + (ipi - mpi) × (Rt − predicted_recall)
```

- `ipi` = initial prediction interval (e.g. 200) — used when recall is far from target
- `mpi` = minimum prediction interval (e.g. 0 or 10) — floor as recall converges
- Clamped to `[mpi, ipi]`

When predicted recall is far below target → large interval → infrequent (cheap) prediction.  
When predicted recall is close to target → small interval → frequent prediction → precise stop.

### Early termination condition
```cpp
if (predicted_recall >= Rt) {
    break;   // exit the beam-search loop
}
```

The current heap contents become the final result.

### LightGBM training parameters (from official repo)
```python
lgb.LGBMRegressor(
    objective='regression',
    n_estimators=100,
    random_state=42,
    verbose=-1
)
model.booster_.save_model("model.txt")  # LightGBM text format
```

### LightGBM C++ inference (from official repo)
```cpp
// load once in constructor
LGBM_BoosterCreateFromModelfile(model_path, &out_iterations, &booster);

// called per prediction
LGBM_BoosterPredictForMatSingleRow(
    booster, feature_array, C_API_DTYPE_FLOAT64,
    11, 1, C_API_PREDICT_NORMAL, 0, -1, "", &out_len, out_result);
```

---

## 4. Python Files to Study

### Official repo
| File | Role |
|------|------|
| `notebooks_scripts/predictor_training.py` | Trains `lgb.LGBMRegressor`, saves `.txt` model. Features: `step, dists, inserts, first_nn_dist, nn_dist, furthest_dist, avg_dist, variance, percentile_25, percentile_50, percentile_75`. Target: `r` (recall@k). |
| `notebooks_scripts/predictor_validation.py` | Loads `.txt` model, evaluates on validation queries. |
| `notebooks_scripts/extract_data_stats.py` | Parses logged training CSVs, computes statistics. |
| `experiments/hnsw_training_data_generation.sh` | Calls `hnsw_test --mode early-stop-training --logging-interval 2` to produce training CSVs. |
| `experiments/hnsw_darth_test.sh` | Full DARTH benchmark. Model path: `predictor_models/darth/{DS}_..._all_feats.txt`. |
| `experiments/tuning.py` | Binary-searches optimal `ipi`/`mpi` values per dataset and target recall. |

### This project
| File | Role |
|------|------|
| `hsnw_constructionDARTH.py` | Pure-Python `HNSW_DARTH` with `_search_layer_darth`, `darth_extract_features`. |
| `darth/collect_training_data.py` | Runs full HNSW search per query, logs features + recall@k at every `logging_interval` steps. Outputs CSV. |
| `darth/train_predictor.py` | Reads CSV, trains LightGBM, saves `.txt` model. |
| `darth/predictor.py` | `LGBMPredictor` — loads `.txt`, implements `predict(feats_dict)`. Used directly in Python or via C++ bindings. |

---

## 5. C++ / Header Files to Study

### Official repo
| File | Role |
|------|------|
| `faiss/impl/DeclarativeRecall.h` | **Start here.** All struct definitions. `BoosterHandle booster` member. |
| `faiss/impl/DeclarativeRecall.cpp` | Model loading, feature assembly, prediction call, adaptive interval. |
| `faiss/impl/HNSW.h` | Forward declarations, method signatures. |
| `faiss/impl/HNSW.cpp` | `search_from_candidates_DARTH` — the actual search loop with predictor wiring. |
| `faiss/IndexHNSW.h/cpp` | Public `IndexHNSW::search_DARTH()` entry point. |
| `hnsw-test/hnsw_test.cpp` | CLI driver with `--predictor-model-path` and `DARTHPredictorHNSW` construction. |

### This project
| File | Role |
|------|------|
| `hnswDarth_cpp/include/hnswDarth.h` | `HNSW_DARTH` class. `DarthFeatures` struct (11 members). `IPredictor` interface. `search_layer_darth()` and `darth_extract_features()` signatures. |
| `hnswDarth_cpp/src/hnswDarth.cpp` | Full `search_layer_darth` implementation, adaptive interval, `darth_extract_features`, `query_darth`, `search_darth`. |
| `hnswDarth_cpp/src/bindings.cpp` | `PyPredictor` — bridges Python callable to `IPredictor`. `HNSWDarthIndex` pybind11 class with `search_darth()`. |
| `hnswDarth_cpp/src/CMakeLists.txt` | Builds `hnswDarth_cpp.so` with pybind11. |

---

## 6. How to Download and Build the Official Repo

```bash
git clone https://github.com/MChatzakis/DARTH.git
cd DARTH

# Install LightGBM to /HOME/lightgbm-install (or edit CMakeLists.txt)
# Install FAISS prerequisites: OpenBLAS/MKL, LAPACK, C++17 compiler

cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
make -C build -j faiss
make -C build -j hnsw_test
make -C build -j ivf_test

# Python deps
pip install -r requirements.txt   # lightgbm, pandas, numpy, dask, scikit-learn

# 1. Generate training data
bash experiments/hnsw_training_data_generation.sh

# 2. Train
cd notebooks_scripts && python predictor_training.py
# → ../predictor_models/lightgbm/{DS}_..._all_feats.txt

# 3. Move model to expected path
# experiments/hnsw_darth_test.sh looks for predictor_models/darth/*.txt

# 4. Run DARTH
bash experiments/hnsw_darth_test.sh
```

**Model filename convention:**
```
{DS}_M{M}_efC{efC}_efS{efS}_s{train_queries}_k{k}_nestim{n_estim}_li{li}_all_feats.txt
```

---

## 7. This Project's Implementation

### File map

```
hsnw_constructionDARTH.py          Pure-Python HNSW with DARTH search
hnswDarth_cpp/
  include/hnswDarth.h              C++ HNSW_DARTH + IPredictor interface
  src/hnswDarth.cpp                C++ search_layer_darth implementation
  src/bindings.cpp                 pybind11 bridge: PyPredictor + HNSWDarthIndex
  src/CMakeLists.txt               builds hnswDarth_cpp.so

darth/
  __init__.py                      exports LGBMPredictor
  predictor.py                     LGBMPredictor — load .txt model, predict()
  collect_training_data.py         run HNSW in logging mode → CSV
  train_predictor.py               CSV → LightGBM model → .txt

predictor_models/
  .gitkeep                         placeholder (models not committed)
```

### Python-only prototype
```python
from darth.predictor import LGBMPredictor

predictor = LGBMPredictor("predictor_models/siftsmall_k10.txt")

# With pure-Python HNSW_DARTH
from hsnw_constructionDARTH import HNSW_DARTH
hnsw = HNSW_DARTH(dim=128, M=16, efConstruction=200)
# ... add vectors ...
ids = hnsw.query_darth(q, k=10, efSearch=200, Rt=0.90, predictor=predictor)
```

### C++ production path
```python
import sys
sys.path.insert(0, "hnswDarth_cpp/src/build")
import hnswDarth_cpp

from darth.predictor import LGBMPredictor
predictor = LGBMPredictor("predictor_models/siftsmall_k10.txt")

index = hnswDarth_cpp.HNSWDarthIndex(dim=128, M=16, efConstruction=200)
index.add(xb)
D, I = index.search_darth(xq, k=10, efSearch=200, Rt=0.90, predictor=predictor)
```

The `PyPredictor` in `bindings.cpp` wraps the Python callable and calls it per prediction step inside the C++ search loop — zero extra overhead from Python object allocation per query.

### Python↔C++ bridge (model format)

| Step | Where | What |
|------|-------|------|
| Train | `darth/train_predictor.py` | `lgb.LGBMRegressor.fit()` |
| Save | `darth/train_predictor.py` | `model.booster_.save_model("model.txt")` — **LightGBM text format** |
| Load (Python) | `darth/predictor.py` | `lgb.Booster(model_file="model.txt")` |
| Load (C++ official repo) | `DeclarativeRecall.cpp` | `LGBM_BoosterCreateFromModelfile("model.txt", ...)` |
| Infer (Python) | `LGBMPredictor.predict()` | `model.predict(feature_array)` |
| Infer (C++ official repo) | `DeclarativeRecall.cpp` | `LGBM_BoosterPredictForMatSingleRow(...)` |

The `.txt` file is interchangeable — train in Python, load in C++ with no conversion step.

---

## 8. Start-Here Sequence

```bash
# 0. Install LightGBM (not in requirements.txt by default)
pip install lightgbm

# 1. Build the C++ module (or skip if using pure Python)
cd hnswDarth_cpp/src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
cd ../../..

# 2. Build the HNSW index and collect training data
python -c "
from hsnw_constructionDARTH import HNSW_DARTH
from utils.load_datasets import load_fvecs, load_ivecs
from darth.collect_training_data import collect_training_data

xb = load_fvecs('Datasets/siftsmall/siftsmall_base.fvecs')
xq = load_fvecs('Datasets/siftsmall/siftsmall_learn.fvecs')[:5000]
gt = load_ivecs('Datasets/siftsmall/siftsmall_groundtruth.ivecs')

hnsw = HNSW_DARTH(dim=128, M=16, efConstruction=200)
for i, v in enumerate(xb): hnsw._insert_(v, i)

collect_training_data(hnsw, xq, gt, k=10, efSearch=200,
                      output_path='predictor_models/train_data.csv')
"

# 3. Train the predictor
python darth/train_predictor.py \
  --input predictor_models/train_data.csv \
  --output predictor_models/siftsmall_M16_k10.txt \
  --n-estimators 100

# 4. Use the predictor
python -c "
from darth.predictor import LGBMPredictor
from hsnw_constructionDARTH import HNSW_DARTH
from utils.load_datasets import load_fvecs, load_ivecs

xb = load_fvecs('Datasets/siftsmall/siftsmall_base.fvecs')
xq = load_fvecs('Datasets/siftsmall/siftsmall_query.fvecs')

hnsw = HNSW_DARTH(dim=128, M=16, efConstruction=200)
for i, v in enumerate(xb): hnsw._insert_(v, i)

predictor = LGBMPredictor('predictor_models/siftsmall_M16_k10.txt')
ids = hnsw.query_darth(xq[0], k=10, efSearch=200, Rt=0.90, predictor=predictor)
print(ids)
"
```

---

## 9. How to Port to a New Project

### Minimal requirements
1. **An HNSW search loop** that exposes per-step counters: `ndis`, `nstep`, `ninserts`, `firstNN`, and the current result heap (for percentile stats).
2. **LightGBM** installed (`pip install lightgbm`).
3. **The three `darth/` scripts** — copy them verbatim.

### Minimum code to add to your HNSW
Inside your beam-search loop on layer 0, after every `ipi` distance computations:
```python
if inserts >= k and (idis % pi_int) == 0:
    features = extract_11_features(result_heap, ndis, nstep, firstNN, inserts)
    predicted_recall = predictor.predict(features)
    if predicted_recall >= target_recall:
        break
    # update interval
    pi = mpi + (ipi - mpi) * (target_recall - predicted_recall)
    pi = max(mpi, min(ipi, pi))
    idis = 0
```

### Custom C++ wrapper (no FAISS)
```cpp
#include <LightGBM/c_api.h>

struct DARTHPredictor {
    BoosterHandle booster;

    explicit DARTHPredictor(const char* model_path) {
        int iters;
        LGBM_BoosterCreateFromModelfile(model_path, &iters, &booster);
    }

    // feature order must match training: nstep,ndis,ninserts,firstNN,
    //   closestNN,furthestNN,meanNN,varNN,p25NN,p50NN,p75NN
    float predict(double* features, int n_features = 11) {
        int64_t out_len;
        double out;
        LGBM_BoosterPredictForMatSingleRow(
            booster, features, C_API_DTYPE_FLOAT64,
            n_features, 1, C_API_PREDICT_NORMAL, 0, -1, "", &out_len, &out);
        return (float)out;
    }
};
```
Link with: `-llightgbm`

---

## 10. Risks and Known Gaps

### Pre-trained models are not in the repo
`predictor_models/` is a placeholder. You must run the full collect → train pipeline before DARTH actually predicts anything. With a `DummyPredictor` (returns 0.0), `search_darth` degrades to standard beam search that never terminates early.

### Model is config-specific
One model per `(dataset, M, efConstruction, efSearch, k)` tuple. A model trained on SIFT/M=16/k=10 will likely give poor predictions on a different dataset or different k. Always retrain when the index config changes.

### `lightgbm` is not in `requirements.txt`
Install separately: `pip install lightgbm`. If you want to pin: `pip install "lightgbm>=3.3"`.

### Training data volume
The official repo uses 10,000–20,000 training queries. With `siftsmall` (10,000 base vectors) you have fewer queries to collect from. Use `siftsmall_learn.fvecs` (25,000 learn vectors) as the training query set, not the 100-query test set.

### Feature name mismatch between Python paths
The pure-Python `darth_extract_features` in `hsnw_constructionDARTH.py` returns keys like `avg`, `var`, `med`, `perc25`, `perc75`. The C++ `bindings.cpp` `PyPredictor` passes `meanNN`, `varNN`, `p50NN`, `p25NN`, `p75NN`. The `LGBMPredictor` in `darth/predictor.py` handles both via an internal alias map — do not change key names in either file without updating `predictor.py`.

### Thread safety
The Python `LGBMPredictor` is not thread-safe for concurrent `predict()` calls (LightGBM's `Booster.predict()` holds the GIL). The C++ bindings acquire the GIL for every prediction call (`py::gil_scoped_acquire`). For multi-threaded C++ search, create one `LGBMPredictor` instance per thread, or switch to a pure C++ predictor using `LGBM_BoosterPredictForMatSingleRow` which is thread-safe.

### `DeclarativeRecallDataCollectorIVF.get_observation_data_str` is commented out
The IVF data-collection logging code in the official repo is inside a `/* */` block — the IVF training pipeline appears incomplete. Only HNSW is fully operational end-to-end.

### No validation split in `train_predictor.py`
The training script uses all data for training. Run `darth/train_predictor.py` with `--val-fraction 0.1` to hold out a validation set and print MAE — the flag is supported.
