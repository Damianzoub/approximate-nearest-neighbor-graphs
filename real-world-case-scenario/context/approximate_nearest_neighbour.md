# Approximate Nearest Neighbour Search

Nearest neighbour search finds the most similar items to a query in a large collection. In the exact setting, this requires computing the distance between the query and every item — feasible for small datasets but prohibitively expensive at the billion-scale databases used in modern recommendation systems, search engines, and vector databases. Approximate Nearest Neighbour (ANN) algorithms sacrifice a small amount of accuracy to achieve orders-of-magnitude speedup.

## The Curse of Dimensionality

In low dimensions, space-partitioning structures like k-d trees enable fast exact nearest neighbour search. However, as dimensionality grows, these structures degrade to brute-force performance — a phenomenon known as the curse of dimensionality. Modern embeddings from neural networks (text, image, audio) typically have 128 to 1536 dimensions, rendering tree-based methods ineffective.

ANN algorithms address this by exploring only a promising subset of the dataset, guided by graph traversal, hashing, or quantisation.

## Hierarchical Navigable Small World (HNSW)

HNSW, introduced by Malkov and Yashunin in 2020, builds a multi-layer proximity graph. The upper layers contain few nodes and long-range edges for fast coarse navigation; the bottom layer contains all nodes and short-range edges for precise local search. During query, the algorithm enters from the top layer, greedily descends to the bottom, and expands a candidate list using a beam search controlled by the efSearch parameter.

HNSW achieves state-of-the-art recall-throughput tradeoffs and is the backbone of production vector databases including Weaviate, Milvus, Qdrant, and Chroma. The efConstruction parameter controls index quality at build time; M controls the number of bidirectional links per node and thus graph connectivity and memory usage.

## DARTH — Dynamic Approximate Recall Threshold

DARTH enhances HNSW with learned early termination. A LightGBM predictor, trained on internal search statistics (number of distance computations, nearest-neighbour distance distribution, variance), predicts whether the current result set has already achieved the target recall threshold Rt. When the predictor is confident enough, the search terminates early, saving computation without a recall penalty for easy queries while continuing to search for difficult ones.

DARTH is particularly effective for query sets with heterogeneous difficulty, where many queries are easy and can be resolved quickly while a minority of hard queries require full search depth. The predictor is trained offline on a learn set representative of the query distribution.

## PiP — Point in Polytope

PiP prunes HNSW candidates by checking whether a candidate is within the convex hull (polytope) of the current k nearest neighbours. If the candidate lies inside this polytope, it cannot improve the result set and is discarded without computing the exact distance. PiP is parameterised by pip_gamma (the confidence percentile for the stability check) and pip_delta (the minimum number of stable iterations before pruning).

PiP is particularly effective for cosine similarity search on unit-sphere embeddings, where the polytope geometry is most constraining. It requires no training data, making it easy to deploy.

## Ada-ef — Adaptive Expansion Factor

Ada-ef adapts the efSearch parameter per query based on the query's distance distribution relative to precomputed statistics. Rather than using a fixed efSearch for all queries, Ada-ef builds an offline calibration table mapping expected recall to the required efSearch at different operating points. At query time, it selects the minimum efSearch expected to achieve the target recall for that query's characteristics.

This reduces unnecessary computation for easy queries without sacrificing recall on hard queries, providing a principled approach to the recall-throughput tradeoff.

## Applications

ANN search powers recommendation engines (finding similar items, users, or content), semantic search (retrieving documents relevant to a query), retrieval-augmented generation (providing LLMs with relevant context), face recognition (matching faces against a database), drug discovery (finding molecules similar to a lead compound), and anomaly detection (identifying points far from their nearest neighbours).

Vector databases — purpose-built systems for storing and querying high-dimensional embeddings — are now a critical component of modern AI infrastructure, and efficient ANN algorithms like HNSW are at their core.
