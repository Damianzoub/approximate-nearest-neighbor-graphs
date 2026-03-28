# PiP: Probe in Parallel for HNSW

## 0. Algorithm Explanation (Beginner-Friendly)

### What is the problem?
HNSW searches one region at a time, but:
- The best answer might be in a different region
- Sequential search misses parallel opportunities
- Fixed efSearch cannot adapt to query difficulty

X Missing distant clusters
X Wasting time on easy queries

---

### Core intuition (VERY important)

Think of it like searching for treasure:

- HNSW = one explorer going deeper into one cave
- PiP = multiple explorers probing different caves in parallel

PiP launches parallel probes to explore multiple regions simultaneously.

---

### Structure

PiP adds parallel exploration to HNSW:

- Multiple entry points at top layer
- Parallel search from each entry
- Merge results at the end

---

### Two phases

#### 1. Index Construction (same as HNSW)

PiP uses standard HNSW building:
- Multi-layer graph
- M neighbors per node
- efConstruction parameter

---

#### 2. Search (parallel probing)

Key innovation: Multiple entry points.

For search:
1. Select K entry points (using heuristics)
2. Launch parallel searches from each entry
3. Merge all results
4. Return top-k

---

### Key Parameters

- K (number of probes) -> how many entry points
- Probe selection strategy -> how to pick entry points
- Merge strategy -> how to combine results

More probes = more recall but more computation

---

## 1. Problem the Paper Solves

HNSW can miss clusters far from the initial entry point.

PiP solves:
- Missing nearby clusters
- Single-threaded search limitation
- Poor exploration of graph topology

---

## 2. Core Idea

Use multiple entry points to explore different regions of the graph in parallel.

If the query has neighbors in multiple clusters, parallel probes find them all.

---

## 3. How It Works (Mechanism)

### Entry Point Selection
1. Start from top layer of HNSW
2. Use heuristics to select K diverse entry points:
   - Geographic spread
   - Distance from query
   - Graph connectivity
3. These become starting points for parallel search

### Parallel Search
1. Launch K independent HNSW searches
2. Each probe explores its local region
3. All probes use smaller efSearch (optional)
4. Collect results from all probes

### Result Merging
1. Combine all candidates from all probes
2. Re-rank by distance to query
3. Return top-k

Key components:
- Probe scheduler
- Parallel execution engine
- Result merger

---

## 4. Key Formula / Signal

Diversity score for entry points:
diversity(p) = min(dist(p, other_selected_points))

Goal: Maximize diversity among selected entry points.

Combined candidate set:
C_total = union(C_1, C_2, ..., C_K)

---

## 5. Where It Fits (Big Picture)

Category:
- Graph-based ANN with parallel exploration

Relation:
- HNSW -> single entry point search
- PiP -> multiple entry point search

Extends HNSW with:
- Parallel exploration
- Multi-region search
- Result merging

---

## 6. Strengths

- Explores multiple regions simultaneously
- Better recall for queries near cluster boundaries
- Natural parallelism (embarrassingly parallel)
- Can combine with other optimizations

---

## 7. Weaknesses / Limitations

- More computation (K searches)
- Entry point selection overhead
- Diminishing returns for high K
- May duplicate work on overlapping regions

---

## 8. My Implementation Notes

- Built on top of HNSW implementation
- Added:
  - Entry point selection algorithm
  - Parallel search wrapper
  - Result merging logic

Parameters used:
- K = [2, 4, 8] probes
- Entry selection = distance-based diversity

---

## 9. Experimental Impact

- Improves recall on queries near multiple clusters
- Parallel execution reduces wall-clock time
- Best for:
  - Queries with neighbors in different regions
  - High-dimensional data with clustered structure

---

## 10. Comparison to Other Methods

- vs HNSW:
  - HNSW = single entry point
  - PiP = multiple entry points

- vs DARTH:
  - DARTH = adaptive stopping
  - PiP = parallel exploration

- vs AdaEF:
  - AdaEF = adaptive efSearch
  - PiP = parallel search paths

---

## 11. Key Insight

Single entry points limit exploration.

Multiple probes can find neighbors in distant clusters that sequential search misses.

---

## 12. Open Questions / Ideas

- How to select optimal K automatically?
- Can probes share intermediate results?
- How does this interact with adaptive ef?
- Can we use learning for entry point selection?

---

## 13. Relationship to Other Papers

Combines well with:
- DARTH (early termination on each probe)
- AdaEF (adaptive ef per probe)

Future work:
- Smart probe selection using ML
- Probe coordination to reduce overlap
