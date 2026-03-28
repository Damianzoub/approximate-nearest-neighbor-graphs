# FAISS: Billion-Scale Similarity Search with GPUs

## 0. Algorithm Explanation (Beginner-Friendly)

### What is the problem?
We have billions of vectors and need to find similar ones fast.

X CPU-based methods are too slow for billions
X Exact search is impossible at this scale

---

### Core intuition (VERY important)

Think of it like organizing a library:

- Books = vectors
- Finding similar books by hand = too slow
- Using an index (catalog) = fast

FAISS combines:
- GPU parallelism (many calculations at once)
- Smart indexing (navigate efficiently)
- Quantization (compress vectors)

---

### Structure

FAISS is a library with multiple index types:

- IVF (Inverted File Index)
- PQ (Product Quantization)
- HNSW (can use GPU-accelerated version)
- Composite indexes (combine multiple techniques)

---

### Key Components

#### 1. Index Structure

FAISS builds indexes for fast search:

- IVF: Divide vectors into clusters, search relevant clusters
- PQ: Compress vectors into codes
- HNSW: Multi-layer graph navigation

#### 2. GPU Acceleration

GPU excels at:
- Batch distance calculations
- Parallel nearest neighbor search
- Efficient memory operations

#### 3. Search Process

1. Load index to GPU memory
2. For query batch:
   - Calculate distances in parallel
   - Find top-k candidates
   - Return results
3. Optionally: refine on CPU

---

### Key Parameters

- nlist -> number of clusters (IVF)
- m -> subvector dimensions (PQ)
- nprobe -> clusters to search (IVF)
- efSearch -> for HNSW indexes

Trade-offs:
- More clusters = more precision but slower
- Smaller codes = less memory but lower recall

---

## 1. Problem the Paper Solves

CPU-based ANN methods cannot scale to billions of vectors.

FAISS solves:
- GPU-accelerated similarity search
- Memory-efficient billion-scale indexing
- Batch query processing

---

## 2. Core Idea

Combine GPU parallelism with smart indexing for billion-scale search.

Key innovations:
- Product Quantization for compression
- IVF for clustering
- GPU kernels for fast computation

---

## 3. How It Works (Mechanism)

### Index Building
1. Cluster vectors using k-means
2. Assign each vector to nearest centroid
3. Store centroid list and inverted lists
4. Optionally: train PQ codes

### GPU Search
1. Load query batch to GPU
2. For each query:
   - Calculate distance to centroids
   - Select nprobe nearest clusters
   - Search vectors in those clusters
   - Use PQ for fast distance computation
3. Return top-k results

### Product Quantization
1. Split vector into m subvectors
2. Quantize each subvector independently
3. Store as m-byte code
4. Distance = sum of subvector distances (approximated)

---

## 4. Key Formula / Signal

PQ distance (asymmetric):
d(x, y) ~ sum(d_sub(x_sub, c_sub))

Where:
- x_sub = subvector of x
- c_sub = nearest centroid for subvector

IVF selection:
Select nprobe clusters with smallest centroid distance.

---

## 5. Where It Fits (Big Picture)

Category:
- GPU-accelerated ANN library

Relation:

- HNSW, DARTH, PiP, AdaEF -> algorithms
- FAISS -> library that implements these algorithms (plus more)

FAISS provides:
- CPU and GPU implementations
- Multiple index types
- Production-ready code

---

## 6. Strengths

- Can search billions of vectors
- GPU provides massive speedup
- Multiple index types for different use cases
- Well-tested and maintained (by Meta)

---

## 7. Weaknesses / Limitations

- Requires GPU with enough memory
- PQ introduces quantization error
- Index building can be slow
- Configuration complexity (many parameters)

---

## 8. Implementation Notes

FAISS provides ready-to-use implementations:

- IVF-PQ index (most common)
- HNSW index (recent addition)
- Composite indexes (IVF-HNSW, etc.)

GPU-specific considerations:
- Batch size affects performance
- Memory bandwidth is bottleneck
- Mixed precision can help

---

## 9. Experimental Impact

Performance numbers from paper:
- 1 billion vectors searchable
- QPS: 10,000+ on GPU
- Recall: tunable via nprobe
- Memory: 4-20 bytes per vector

Comparison:
- CPU HNSW: ~1,000 QPS
- FAISS GPU: ~10,000+ QPS

---

## 10. Comparison to Custom Implementations

vs our HNSW:
- FAISS = production-ready, GPU-accelerated
- Our HNSW = educational, simpler

vs DARTH/PiP/AdaEF:
- These algorithms can be implemented in FAISS
- FAISS provides efficient building blocks

---

## 11. Key Insight

GPU + smart indexing = billion-scale search.

FAISS shows that with right data structures and hardware, ANN at billion scale is practical.

---

## 12. Open Questions / Ideas

- Can custom algorithms (DARTH, PiP, AdaEF) be implemented in FAISS?
- How to handle dynamic updates?
- Can we use multiple GPUs efficiently?
- What about distributed search?

---

## 13. Relationship to Other Papers

FAISS implements concepts from many papers:

- IVF from Inverted File Index papers
- PQ from Product Quantization paper
- HNSW integration from HNSW paper

The other papers in this collection (DARTH, PiP, AdaEF) could potentially be implemented in FAISS for production use.
