# Hierarchical Navigable Small World (HNSW)

## 0. Algorithm Explanation (Beginner-Friendly)

### What is the problem?
We have millions of vectors (points in space).
Given a query, we want to find the closest ones.

X Exact search = too slow
Y HNSW = fast approximate search

---

### Core intuition (VERY important)

Think of it like Google Maps:

- Top layer -> highways (fast, rough navigation)
- Lower layers -> streets (more detailed)
- Bottom layer -> exact local search

HNSW moves:
1. Fast at top (big jumps)
2. Slowly at bottom (fine search)

---

### Structure

HNSW is a **multi-layer graph**:

- Each node = a vector
- Each node connects to neighbors
- Upper layers = fewer nodes
- Bottom layer = all nodes

---

### Two phases

#### 1. Index Construction (building the graph)

For each new point:

1. Assign random level (higher = rarer)
2. Start from top layer
3. Greedy search -> find closest node
4. Go layer by layer down
5. At each layer:
   - connect to M nearest neighbors

---

#### 2. Search (query time)

1. Start from top layer
2. Greedy search -> find closest node
3. Move down layers
4. At bottom:
   - use **efSearch candidates**
   - explore neighbors
   - return top-k

---

### Key Parameters

- M -> number of neighbors (graph quality)
- efConstruction -> accuracy during building
- efSearch -> accuracy during search

Bigger efSearch = better recall but slower

---

## 1. Problem the Paper Solves

Exact nearest neighbor search is too slow in high dimensions.

HNSW solves:
- Fast search
- High recall
- Scalable indexing

---

## 2. Core Idea

Use a hierarchical graph structure to:
- Navigate quickly globally
- Refine locally

Combine:
- Greedy search (fast)
- Multi-layer graph (efficient)

---

## 3. How It Works (Mechanism)

### Build Phase
- Insert points one by one
- Assign random level
- Connect to neighbors using greedy search

### Search Phase
- Start at top layer
- Move downward
- At base layer:
  - explore efSearch candidates
  - maintain best results

Key components:
- Priority queue
- Neighbor list
- Graph layers

---

## 4. Key Formula / Signal

No single formula -- but key concept:

Distance:
d(q, x)

Search condition:
Continue exploring while:
candidate_distance < worst_best_distance

Meaning:
Only explore if it can improve results

---

## 5. Where It Fits (Big Picture)

Category:
- Graph-based ANN

Relation:

- HNSW -> base algorithm
- DARTH -> adds adaptive stopping
- PiP -> adds early termination
- AdaEF -> changes ef dynamically

---

## 6. Strengths

- Very high recall
- Efficient search
- Widely used in industry (FAISS, Milvus)

---

## 7. Weaknesses / Limitations

- efSearch must be tuned manually
- Same ef for all queries (bad)
- Can over-search or under-search

---

## 8. My Implementation Notes

- Implemented from scratch
- Used:
  - M = 16
  - efConstruction = 200
  - efSearch = [20, 50, 100]

Simplifications:
- No SIMD
- No parallelism

---

## 9. Experimental Impact

- Baseline recall
- Used as comparison for:
  - DARTH
  - PiP
  - AdaEF

---

## 10. Comparison to Other Methods

- vs DARTH:
  - HNSW = fixed search
  - DARTH = adaptive stopping

- vs PiP:
  - HNSW = full search
  - PiP = early termination

- vs AdaEF:
  - HNSW = static ef
  - AdaEF = dynamic ef

---

## 11. Key Insight

HNSW is powerful but:

The real bottleneck is **search control (efSearch)**

All later methods (DARTH, PiP, AdaEF) try to fix this.

---

## 12. Open Questions / Ideas

- Can we replace efSearch entirely?
- Can we predict difficulty of queries?
- Can we combine:
  - PiP (early stop)
  - AdaEF (adaptive ef)?

This leads to a hybrid approach
