"""
Per-Query Benchmark Module for ANNS Failure Analysis.

This module provides detailed per-query metrics collection for analyzing
when and why early termination methods fail.
"""

import time
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class PerQueryResult:
    """Container for per-query benchmark results."""
    
    # Query identification
    query_id: int
    method: str
    
    # Basic results
    recall: float
    latency_ms: float
    true_neighbors_found: int
    
    # Early termination info
    terminated_early: bool = False
    actual_ef_used: int = 0
    max_ef_allowed: int = 0
    
    # Search statistics
    num_iterations: int = 0
    num_distance_computations: int = 0
    num_nodes_visited: int = 0
    
    # Result quality
    distance_to_best: float = 0.0
    distance_to_worst_in_topk: float = 0.0
    distance_ratio: float = 0.0  # dk/d1
    
    # Method-specific fields (stored as JSON string)
    method_specific: str = "{}"
    
    # Query characteristics (for categorization)
    query_d1nn_distance: float = 0.0  # Distance to 1-NN
    query_dknn_distance: float = 0.0   # Distance to k-NN
    query_local_density: float = 0.0   # Points in neighborhood
    
    # Failure classification
    failure_type: str = "none"  # none, boundary, sparse, outlier, high_dispersion
    recall_gap_from_oracle: float = 0.0  # Difference from oracle (high efSearch)


class PerQueryBenchmark:
    """
    Benchmark that collects per-query metrics for failure analysis.
    
    Usage:
        benchmark = PerQueryBenchmark()
        
        # Wrap any search function
        wrapped_search = benchmark.wrap_search(
            hnsw_index.search,
            method_name="hnsw",
            max_ef=100
        )
        
        # Run queries
        for i, query in enumerate(queries):
            result = benchmark.run_query(wrapped_search, query, k=10, query_id=i)
            results.append(result)
        
        # Save results
        benchmark.save_to_csv("results.csv")
    """
    
    def __init__(self, track_distances: bool = True):
        self.results: List[PerQueryResult] = []
        self.track_distances = track_distances
        self._current_method_stats = {}
        
    def wrap_search(
        self, 
        search_fn: Callable, 
        method_name: str,
        max_ef: int = 100,
        **method_params
    ) -> Callable:
        """
        Wrap a search function to track detailed metrics.
        
        Args:
            search_fn: The search function to wrap
            method_name: Name of the method (hnsw, darth, pip, adaef)
            max_ef: Maximum efSearch allowed
            **method_params: Additional parameters for the search function
            
        Returns:
            Wrapped function that tracks metrics
        """
        def wrapped(query: np.ndarray, k: int, **kwargs):
            # Reset counters
            self._current_method_stats = {
                "method": method_name,
                "num_distance_computations": 0,
                "num_nodes_visited": 0,
                "num_iterations": 0,
                "actual_ef_used": 0,
                "terminated_early": False,
                "method_specific": {},
                **method_params
            }
            
            # Run search with timing
            start_time = time.perf_counter()
            result = search_fn(query.reshape(1, -1), k, **kwargs)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            
            # Extract results
            if isinstance(result, tuple):
                distances, indices = result
            else:
                indices = result
                distances = None
            
            return {
                "indices": indices[0] if indices.ndim > 1 else indices,
                "distances": distances[0] if distances is not None and distances.ndim > 1 else distances,
                "latency_ms": latency_ms,
                "stats": self._current_method_stats
            }
            
        return wrapped
    
    def run_query(
        self,
        search_fn: Callable,
        query: np.ndarray,
        k: int,
        query_id: int,
        ground_truth: Optional[np.ndarray] = None,
        method_name: str = "unknown",
        **kwargs
    ) -> PerQueryResult:
        """
        Run a single query and record detailed metrics.
        
        Args:
            search_fn: Wrapped search function
            query: Query vector (1D or 2D)
            k: Number of neighbors
            query_id: Unique query identifier
            ground_truth: Ground truth indices (for recall calculation)
            method_name: Name of the method
            **kwargs: Additional arguments for search
            
        Returns:
            PerQueryResult object
        """
        # Ensure query is 2D for search
        query_2d = query.reshape(1, -1) if query.ndim == 1 else query
        
        # Run search
        start_time = time.perf_counter()
        result = search_fn(query, k, **kwargs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        indices = result["indices"]
        stats = result.get("stats", {})
        
        # Calculate recall if ground truth is available
        recall = 0.0
        true_neighbors_found = 0
        if ground_truth is not None:
            true_k = ground_truth[:k]
            found_k = indices[:k] if len(indices) >= k else indices
            true_neighbors_found = len(set(true_k).intersection(set(found_k)))
            recall = true_neighbors_found / k
        
        # Calculate distance statistics
        distance_to_best = 0.0
        distance_to_worst = 0.0
        if result.get("distances") is not None and len(result["distances"]) > 0:
            all_dists = result["distances"]
            distance_to_best = float(all_dists[0]) if len(all_dists) > 0 else 0.0
            distance_to_worst = float(all_dists[min(k-1, len(all_dists)-1)]) if len(all_dists) > 0 else 0.0
        
        # Calculate distance ratio (k-th / 1st)
        distance_ratio = distance_to_worst / distance_to_best if distance_to_best > 0 else 1.0
        
        # Get query characteristics from ground truth
        query_d1nn_dist = 0.0
        query_dknn_dist = 0.0
        if ground_truth is not None:
            # These would need the actual distances, placeholder for now
            query_dknn_dist = distance_to_worst
        
        # Create result object
        per_query_result = PerQueryResult(
            query_id=query_id,
            method=stats.get("method", method_name),
            recall=recall,
            latency_ms=latency_ms,
            true_neighbors_found=true_neighbors_found,
            terminated_early=stats.get("terminated_early", False),
            actual_ef_used=stats.get("actual_ef_used", 0),
            max_ef_allowed=stats.get("max_ef_allowed", 0),
            num_iterations=stats.get("num_iterations", 0),
            num_distance_computations=stats.get("num_distance_computations", 0),
            num_nodes_visited=stats.get("num_nodes_visited", 0),
            distance_to_best=distance_to_best,
            distance_to_worst_in_topk=distance_to_worst,
            distance_ratio=distance_ratio,
            method_specific=json.dumps(stats.get("method_specific", {})),
            query_d1nn_distance=query_d1nn_dist,
            query_dknn_distance=query_dknn_dist,
            query_local_density=stats.get("local_density", 0.0),
            failure_type="none",
            recall_gap_from_oracle=0.0
        )
        
        self.results.append(per_query_result)
        return per_query_result
    
    def classify_failures(
        self, 
        recall_threshold: float = 0.9,
        d1nn_dist_threshold: Optional[float] = None,
        local_density_threshold: Optional[float] = None
    ) -> None:
        """
        Classify query failures based on observed characteristics.
        
        Args:
            recall_threshold: Minimum acceptable recall
            d1nn_dist_threshold: Threshold for outlier detection
            local_density_threshold: Threshold for sparse region detection
        """
        for result in self.results:
            if result.recall >= recall_threshold:
                result.failure_type = "none"
                continue
            
            # Classify failure type
            if result.distance_ratio > 2.0:
                result.failure_type = "high_dispersion"
            elif result.query_local_density < local_density_threshold:
                result.failure_type = "sparse"
            elif result.distance_to_best > d1nn_dist_threshold:
                result.failure_type = "outlier"
            else:
                result.failure_type = "boundary"
    
    def compute_oracle_gaps(
        self, 
        oracle_results: Dict[int, float]  # query_id -> oracle recall
    ) -> None:
        """Compute recall gap from oracle for all results."""
        for result in self.results:
            if result.query_id in oracle_results:
                result.recall_gap_from_oracle = oracle_results[result.query_id] - result.recall
    
    def save_to_csv(self, filepath: str) -> None:
        """Save results to CSV file."""
        import pandas as pd
        
        records = []
        for r in self.results:
            record = asdict(r)
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(records)} results to {filepath}")
    
    def save_to_json(self, filepath: str) -> None:
        """Save results to JSON file."""
        records = [asdict(r) for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} results to {filepath}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all queries."""
        if not self.results:
            return {}
        
        recalls = [r.recall for r in self.results]
        latencies = [r.latency_ms for r in self.results]
        
        summary = {
            "total_queries": len(self.results),
            "mean_recall": np.mean(recalls),
            "std_recall": np.std(recalls),
            "min_recall": np.min(recalls),
            "median_recall": np.median(recalls),
            "p95_recall": np.percentile(recalls, 95),
            "p5_recall": np.percentile(recalls, 5),
            
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            
            "failure_count": sum(1 for r in self.results if r.failure_type != "none"),
            "early_termination_count": sum(1 for r in self.results if r.terminated_early),
            
            "failure_types": self._count_failure_types(),
            "methods": list(set(r.method for r in self.results))
        }
        
        return summary
    
    def _count_failure_types(self) -> Dict[str, int]:
        """Count failures by type."""
        counts = {}
        for r in self.results:
            ft = r.failure_type
            counts[ft] = counts.get(ft, 0) + 1
        return counts
    
    def filter_by_method(self, method: str) -> List[PerQueryResult]:
        """Get results for a specific method."""
        return [r for r in self.results if r.method == method]
    
    def filter_by_failure_type(self, failure_type: str) -> List[PerQueryResult]:
        """Get results with specific failure type."""
        return [r for r in self.results if r.failure_type == failure_type]
    
    def compare_methods(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across methods."""
        methods = set(r.method for r in self.results)
        comparison = {}
        
        for method in methods:
            method_results = self.filter_by_method(method)
            if not method_results:
                continue
            
            recalls = [r.recall for r in method_results]
            latencies = [r.latency_ms for r in method_results]
            
            comparison[method] = {
                "n_queries": len(method_results),
                "mean_recall": np.mean(recalls),
                "std_recall": np.std(recalls),
                "mean_latency_ms": np.mean(latencies),
                "n_early_terminations": sum(1 for r in method_results if r.terminated_early),
                "n_failures": sum(1 for r in method_results if r.failure_type != "none"),
                "mean_actual_ef": np.mean([r.actual_ef_used for r in method_results if r.actual_ef_used > 0]) if any(r.actual_ef_used > 0 for r in method_results) else 0,
            }
        
        return comparison
    
    def get_hard_queries(self, recall_threshold: float = 0.8) -> List[int]:
        """Get IDs of queries that consistently fail across methods."""
        all_query_ids = set(r.query_id for r in self.results)
        hard_queries = []
        
        for qid in all_query_ids:
            query_results = [r for r in self.results if r.query_id == qid]
            if all(r.recall < recall_threshold for r in query_results):
                hard_queries.append(qid)
        
        return hard_queries
    
    def reset(self) -> None:
        """Clear all results."""
        self.results = []
        self._current_method_stats = {}


class InstrumentedSearchMixin:
    """
    Mixin class that adds instrumentation tracking to search algorithms.
    
    Use this to wrap HNSW search methods for detailed metric collection.
    """
    
    def __init__(self):
        super().__init__()
        self._reset_instrumentation()
    
    def _reset_instrumentation(self):
        """Reset all instrumentation counters."""
        self._num_distance_computations = 0
        self._num_nodes_visited = 0
        self._num_iterations = 0
        self._terminated_early = False
        self._method_specific = {}
    
    def _count_distance(self):
        """Call this after each distance computation."""
        self._num_distance_computations += 1
    
    def _count_node_visit(self):
        """Call this when visiting a node."""
        self._num_nodes_visited += 1
    
    def _count_iteration(self):
        """Call this at the start of each iteration."""
        self._num_iterations += 1
    
    def _set_terminated_early(self, value: bool = True):
        """Set early termination flag."""
        self._terminated_early = value
    
    def _set_method_specific(self, key: str, value: Any):
        """Store method-specific data."""
        self._method_specific[key] = value
    
    def get_instrumentation_stats(self) -> Dict[str, Any]:
        """Get current instrumentation stats."""
        return {
            "num_distance_computations": self._num_distance_computations,
            "num_nodes_visited": self._num_nodes_visited,
            "num_iterations": self._num_iterations,
            "terminated_early": self._terminated_early,
            "method_specific": self._method_specific.copy()
        }
