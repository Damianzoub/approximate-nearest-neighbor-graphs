# HNSW — structure, algorithms, simple explanation, and drawbacks

## What HNSW builds (simple)
HNSW (Hierarchical Navigable Small World) incrementally builds a multi-layer proximity graph that approximates the k-nearest-neighbor graph for a dataset. Each node is a data point; edges connect nearby points. Higher layers are sparser and connect long-range shortcuts, lower layers are denser and capture fine local neighborhoods. The structure is designed to allow fast approximate nearest-neighbor search by navigating from sparse to dense layers.

## Structure (concise)
- Layers: each node has a random maximum level `L`. The node appears in all layers 0..L.
- Entry point: a single node (or small set) used to start searches at the top layer.
- Graph per layer: an undirected graph where each node keeps up to `M` neighbors (or `M_max`/`M0` depending on layer).
- Heuristic neighbor selection: when adding edges the algorithm uses a selection heuristic to choose neighbors that improve search quality.

## Core algorithms (high level)
1. Insertion / Construction
    - Pick a random level `L` for the new node (geometric distribution).
    - Starting from the entry point at the highest layer, do greedy search to find the closest node at that layer.
    - For each layer from top down to 0:
      - Use a local search (beam/greedy with candidate list of size `efConstruction`) to find nearest candidates.
      - Select up to `M` neighbors from those candidates using the heuristic (diversify neighbors).
      - Link the new node with selected neighbors (and update neighbors’ lists).
    - If the new node has a higher level than current entry, update the global entry point.

2. Query / Search
    - Start at the entry point on the top layer.
    - Greedy descent on each higher layer: move to closer neighbors until no improvement.
    - At layer 0, run a best-first search (priority queue) with beam width `ef` to refine nearest neighbors and return top-k results.

3. Neighbor selection heuristic
    - Prefer neighbors that are close to the node but also diversify to avoid redundant, tightly clustered links (improves navigability).

## Simple step-by-step explanation
- Imagine building a map of cities with highways and local streets.
- Top layers are highways (few connections, long reach); bottom layer is local streets (many nearby links).
- For each new city (point), you find where it fits on highways first (greedy), then create local street connections to nearby cities.
- To find the nearest cities, you start on the highway network, move toward the right area, then switch to local streets to find the exact neighbors.

## Drawbacks and limitations
- Memory: storing neighbor lists per node can be memory-heavy (especially with large `M` and high-dimensional data).
- Indexing cost: construction (especially with large `efConstruction`) can be expensive for very large datasets or frequent inserts.
- Parameter sensitivity: search quality and speed depend on tuning `M`, `ef`, and `efConstruction`.
- Non-determinism: random levels and insertion order affect final graph and performance.
- Deletions: not as simple as insertions; may require complex re-linking or rebuilding parts of the graph.
- High-dimensional data: like other ANN methods, performance degrades in very high dimensions (distance concentration).
- Approximation: results are approximate; guarantee of exact nearest neighbors is not provided.

## Practical tips
- Tune `M` for memory vs connectivity (common values 6–48).
- Use larger `efConstruction` for better graph quality; use smaller `ef` at query time for speed.
- Batch construction or parallelism helps for large datasets.
- Persist the graph or use incremental checkpoints if inserts are frequent.

References: the above is a high-level summary of the HNSW approach (hierarchical small-world proximity graph and greedy/beam search algorithms) — not verbatim from any single source.