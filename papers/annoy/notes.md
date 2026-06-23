# Annoy: Approximate Nearest Neighbors Oh Yeah

**Author:** Erik Bernhardsson  
**Type:** Open-source software library (no formal academic paper)  
**Year:** 2015  
**GitHub:** https://github.com/spotify/annoy

## Citation

No formal academic paper exists. Standard citation is:

> Bernhardsson, E. (2015). Annoy: Approximate Nearest Neighbors in C++/Python.
> GitHub repository: https://github.com/spotify/annoy

## Summary

Annoy builds a forest of random binary trees (originally random projection trees, later hierarchical 2-means trees) for approximate nearest neighbor search. Optimized for memory usage and loading/saving to disk. Originally developed at Spotify for music recommendations. Easy to build an index, but recall is lower than graph-based methods (like HNSW) for a given computational budget.
