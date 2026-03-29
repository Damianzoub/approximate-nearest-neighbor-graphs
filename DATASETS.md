# Dataset Documentation

This document tracks all datasets used in the ANNS thesis experiments, including their sources, versions, and download instructions.

## Standard ANN Benchmark Datasets

All datasets follow the ANN-benchmark format:
- `<name>_base.fvecs` - Base vectors (index)
- `<name>_query.fvecs` - Query vectors
- `<name>_groundtruth.ivecs` - Ground truth neighbors

### Dataset Overview

| Dataset | Dimension | Base Size | Query Size | Type | Source |
|---------|-----------|-----------|------------|------|--------|
| siftsmall | 128 | 10,000 | 100 | SIFT | ANN-benchmark |
| sift | 128 | 1,000,000 | 10,000 | SIFT | ANN-benchmark |
| glove100ann | 100 | 1,183,514 | 10,000 | Word Embeddings | ANN-benchmark |
| gist960ann | 960 | 1,000,000 | 1,000 | GIST | ANN-benchmark |
| fashion_mnist | 784 | 60,000 | 10,000 | Images | Fashion-MNIST |
| glove6b | 50/100/200/300 | 6B tokens | Various | Word Embeddings | GloVe |
| msmarco_text | 768 | ~8.8M | 101,093 | Text Embeddings | MS MARCO |

---

## Dataset Details

### SIFT (Scale-Invariant Feature Transform)

**Description**: Image feature descriptors commonly used for image retrieval and object recognition.

**Characteristics**:
- Dense, clustered distribution
- Well-suited for HNSW (navigable)
- Standard benchmark dataset

**Download**:
```bash
python scripts/download_sift.py
python scripts/download_siftsmall.py
```

**File Hashes** (for verification):
- sift_base.fvecs: SHA256 = `a3f9...` (pending)
- sift_query.fvecs: SHA256 = `b4e8...` (pending)

---

### GIST (Generic Image Signatures)

**Description**: Global image descriptors capturing structural properties of scenes.

**Characteristics**:
- High dimensionality (960D)
- Dense distribution
- Challenging for exact NN search

**Download**:
```bash
python scripts/download_gist960ann.py
```

---

### GloVe (Global Vectors for Word Representation)

**Description**: Pre-trained word embeddings trained on large corpora.

**Characteristics**:
- Moderate dimensionality (100D)
- Sparse in original space, dense in embedding space
- Good for text similarity tasks

**Download**:
```bash
python scripts/download_glove100ann.py
python scripts/download_glove6b.py
```

---

### Fashion-MNIST

**Description**: Zalando's article images converted to feature vectors.

**Characteristics**:
- Image classification dataset
- 28x28 grayscale images = 784D
- More structured than MNIST

**Download**:
```bash
python scripts/download_fashion_mnist.py
```

---

### MS MARCO (Microsoft Machine Reading Comprehension)

**Description**: Real user queries from Bing search engine with passage embeddings.

**Characteristics**:
- Real-world queries
- Passage-level retrieval
- Mixed difficulty distribution

**Download**:
```bash
python scripts/download_msmarco_text.py
```

---

## Dataset Version Tracking

| Dataset | Version | Download Date | Download URL | Hash |
|---------|---------|--------------|--------------|------|
| siftsmall | 1.0 | 2024-03-27 | ANN-benchmark | - |
| sift | 1.0 | 2024-03-27 | ANN-benchmark | - |
| glove100ann | 1.0 | 2024-03-27 | ANN-benchmark | - |
| gist960ann | 1.0 | 2024-03-27 | ANN-benchmark | - |
| fashion_mnist | 1.0 | 2024-03-27 | Fashion-MNIST | - |
| glove6b | 6B | 2024-03-27 | Stanford GloVe | - |
| msmarco_text | v1 | 2024-03-27 | MS MARCO | - |

---

## Adding New Datasets

To add a new dataset:

1. **Download** the dataset in ANN-benchmark format
2. **Place** in `Datasets/<name>/` directory
3. **Update** `configs/experiments.yaml` with dataset configuration
4. **Document** in this file with:
   - Description
   - Characteristics
   - Download command
   - Hash for verification

### Example Directory Structure

```
Datasets/
├── new_dataset/
│   ├── new_dataset_base.fvecs
│   ├── new_dataset_query.fvecs
│   └── new_dataset_groundtruth.ivecs
```

### Format Conversion

If you have raw data, convert to fvecs/ivecs format:

```python
import numpy as np

def save_fvecs(filename, data):
    """Save data in fvecs format."""
    with open(filename, 'wb') as f:
        for vec in data:
            f.write(np.array([len(vec)], dtype=np.int32).tobytes())
            f.write(vec.astype(np.float32).tobytes())

def save_ivecs(filename, data):
    """Save data in ivecs format."""
    with open(filename, 'wb') as f:
        for vec in data:
            f.write(np.array([len(vec)], dtype=np.int32).tobytes())
            f.write(vec.astype(np.int32).tobytes())
```

---

## Recommended Datasets for Different Experiments

| Experiment Type | Recommended Datasets |
|-----------------|---------------------|
| Quick Testing | siftsmall |
| Standard Benchmark | sift, glove100ann |
| High Dimensional | gist960ann |
| Text/Embedding Tasks | glove100ann, msmarco_text |
| Image Retrieval | fashion_mnist, sift |
| Scalability Testing | sift, glove100ann |

---

## External Sources

- **ANN-benchmark**: https://github.com/erikbern/ann-benchmarks
- **Fashion-MNIST**: https://github.com/zalandoresearch/fashion-mnist
- **GloVe**: https://nlp.stanford.edu/projects/glove
- **MS MARCO**: https://microsoft.github.io/msmarco
