"""
Instrumented Search Functions for Per-Query Benchmarking.

This module provides instrumented versions of HNSW search algorithms
that track detailed per-query metrics for failure analysis.
"""

import numpy as np
import heapq
from typing import Dict, Any, Tuple, List, Optional
import time


class InstrumentedHNSW:
    """
    Instrumented HNSW that tracks per-query search statistics.
    
    Tracks:
    - Number of distance computations
    - Number of nodes visited
    - Number of iterations
    - Actual efSearch used
    - Whether early termination occurred
    """
    
    def __init__(self, hnsw_instance):
        """
        Initialize with an existing HNSW instance.
        
        Args:
            hnsw_instance: Any HNSW-like instance with search method
        """
        self.hnsw = hnsw_instance
        self._reset_stats()
    
    def _reset_stats(self):
        """Reset all instrumentation counters."""
        self._num_distance_computations = 0
        self._num_nodes_visited = 0
        self._num_iterations = 0
        self._actual_ef_used = 0
        self._terminated_early = False
        self._method_specific = {}
    
    def search_with_instrumentation(
        self, 
        query: np.ndarray, 
        k: int, 
        efSearch: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform search with detailed metric tracking.
        
        Args:
            query: Query vector
            k: Number of neighbors to return
            efSearch: Maximum efSearch to use
            
        Returns:
            Tuple of (distances, indices, stats_dict)
        """
        self._reset_stats()
        
        # Call the underlying search
        D, I = self.hnsw.search(query.reshape(1, -1), k, efSearch)
        
        distances = D[0] if D is not None else np.array([])
        indices = I[0] if I is not None else np.array([])
        
        stats = self.get_stats()
        return distances, indices, stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current instrumentation stats."""
        return {
            "num_distance_computations": self._num_distance_computations,
            "num_nodes_visited": self._num_nodes_visited,
            "num_iterations": self._num_iterations,
            "actual_ef_used": self._actual_ef_used,
            "terminated_early": self._terminated_early,
            "method": "hnsw",
            "method_specific": self._method_specific.copy()
        }


class InstrumentedDARTH:
    """
    Instrumented DARTH that tracks early termination behavior.
    
    Additional tracking beyond HNSW:
    - DARTH prediction score (Rp)
    - History features (ndis, nstep, etc.)
    - Number of prediction checks
    """
    
    def __init__(self, darth_instance):
        self.darth = darth_instance
        self._reset_stats()
    
    def _reset_stats(self):
        self._num_distance_computations = 0
        self._num_nodes_visited = 0
        self._num_iterations = 0
        self._actual_ef_used = 0
        self._terminated_early = False
        self._final_Rp = 0.0
        self._num_prediction_checks = 0
        self._all_features = []
        self._method_specific = {}
    
    def search_with_instrumentation(
        self,
        query: np.ndarray,
        k: int,
        efSearch: int,
        Rt: float,
        predictor=None,
        ipi: int = 200,
        mpi: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform DARTH search with detailed metric tracking.
        """
        self._reset_stats()
        
        # Call DARTH search
        indices = self.darth.query_darth(
            query, k, efSearch, Rt, predictor, ipi, mpi
        )
        
        # Calculate distances
        distances = np.array([
            self.darth.dist(query, self.darth.vectors[idx]) 
            if idx >= 0 else np.inf 
            for idx in indices
        ])
        
        stats = self.get_stats()
        return distances, np.array(indices), stats
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_distance_computations": self._num_distance_computations,
            "num_nodes_visited": self._num_nodes_visited,
            "num_iterations": self._num_iterations,
            "actual_ef_used": self._actual_ef_used,
            "terminated_early": self._terminated_early,
            "final_Rp": self._final_Rp,
            "num_prediction_checks": self._num_prediction_checks,
            "method": "darth",
            "method_specific": {
                "Rt_threshold": self._method_specific.get("Rt", 0.95),
                "all_features": self._all_features
            }
        }


class InstrumentedPiP:
    """
    Instrumented PiP that tracks saturation-based termination.
    
    Additional tracking:
    - Saturation curve (phi values over iterations)
    - Final saturation value
    - Iterations before termination
    """
    
    def __init__(self, pip_instance):
        self.pip = pip_instance
        self._reset_stats()
    
    def _reset_stats(self):
        self._num_distance_computations = 0
        self._num_nodes_visited = 0
        self._num_iterations = 0
        self._actual_ef_used = 0
        self._terminated_early = False
        self._saturation_curve = []
        self._final_saturation = 0.0
        self._iterations_before_termination = 0
        self._method_specific = {}
    
    def search_with_instrumentation(
        self,
        query: np.ndarray,
        k: int,
        efSearch: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform PiP search with detailed metric tracking.
        """
        self._reset_stats()
        
        # Call PiP search
        indices = self.pip.search_pip(query.reshape(1, -1), k, efSearch)
        
        # Calculate distances
        distances = np.array([
            self.pip.dist(query, self.pip.vectors[idx])
            if idx >= 0 else np.inf
            for idx in indices[0]
        ])
        
        stats = self.get_stats()
        return distances, indices[0], stats
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_distance_computations": self._num_distance_computations,
            "num_nodes_visited": self._num_nodes_visited,
            "num_iterations": self._num_iterations,
            "actual_ef_used": self._actual_ef_used,
            "terminated_early": self._terminated_early,
            "saturation_curve": self._saturation_curve,
            "final_saturation": self._final_saturation,
            "iterations_before_termination": self._iterations_before_termination,
            "method": "pip",
            "method_specific": {
                "pip_gamma": self._method_specific.get("gamma", 95.0),
                "pip_delta": self._method_specific.get("delta", 20),
                "saturation_curve": self._saturation_curve
            }
        }


class InstrumentedAdaEf:
    """
    Instrumented Ada-ef that tracks adaptive ef selection.
    
    Additional tracking:
    - Query difficulty estimate
    - Chosen ef value
    - Target recall
    """
    
    def __init__(self, adaef_instance):
        self.adaef = adaef_instance
        self._reset_stats()
    
    def _reset_stats(self):
        self._num_distance_computations = 0
        self._num_nodes_visited = 0
        self._num_iterations = 0
        self._actual_ef_used = 0
        self._terminated_early = False
        self._query_difficulty = 0.0
        self._chosen_ef = 0
        self._target_recall = 0.0
        self._method_specific = {}
    
    def search_with_instrumentation(
        self,
        query: np.ndarray,
        k: int,
        target_recall: float
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform Ada-ef search with detailed metric tracking.
        """
        self._reset_stats()
        
        # Call Ada-ef search
        result = self.adaef.search_adaef(query.reshape(1, -1), k, target_recall)
        
        indices = result[0] if isinstance(result, tuple) else result
        chosen_ef = self._method_specific.get("chosen_ef", 0)
        
        # Calculate distances
        distances = np.array([
            self.adaef.dist(query, self.adaef.vectors[idx])
            if idx >= 0 else np.inf
            for idx in indices
        ])
        
        stats = self.get_stats()
        return distances, indices, stats
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_distance_computations": self._num_distance_computations,
            "num_nodes_visited": self._num_nodes_visited,
            "num_iterations": self._num_iterations,
            "actual_ef_used": self._actual_ef_used,
            "terminated_early": self._terminated_early,
            "query_difficulty": self._query_difficulty,
            "chosen_ef": self._chosen_ef,
            "target_recall": self._target_recall,
            "method": "adaef",
            "method_specific": self._method_specific.copy()
        }


def create_instrumented_search(index, method: str):
    """
    Factory function to create instrumented search for any method.
    
    Args:
        index: The index instance
        method: Method name ('hnsw', 'darth', 'pip', 'adaef')
        
    Returns:
        Instrumented search instance
    """
    instrumented_classes = {
        "hnsw": InstrumentedHNSW,
        "darth": InstrumentedDARTH,
        "pip": InstrumentedPiP,
        "adaef": InstrumentedAdaEf,
    }
    
    if method not in instrumented_classes:
        raise ValueError(f"Unknown method: {method}")
    
    return instrumented_classes[method](index)
