"""
Query Difficulty Analysis Module for ANNS Failure Analysis.

This module provides tools for analyzing why certain queries are hard
and why early termination methods fail on specific query types.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QueryCharacteristics:
    """Characteristics that define query difficulty."""
    query_id: int
    
    # Distance-based features
    d1nn: float = 0.0           # Distance to 1-NN
    dknn: float = 0.0           # Distance to k-NN
    d_ratio: float = 0.0        # dknn / d1nn (dispersion)
    
    # Distribution features
    local_density: float = 0.0  # Points within 2*d1nn
    isolation_score: float = 0.0  # Distance to nearest cluster centroid
    
    # Structural features
    intrinsic_dim: float = 0.0   # Estimated intrinsic dimensionality
    cluster_boundary: bool = False
    
    # Difficulty classification
    difficulty: str = "unknown"  # easy, medium, hard
    difficulty_score: float = 0.0


class QueryDifficultyAnalyzer:
    """
    Analyzes query difficulty based on dataset and ground truth characteristics.
    
    A query is considered "hard" if:
    1. The k-NN are far apart (high dispersion)
    2. The query is in a sparse region
    3. The query is near cluster boundaries
    4. The query is an outlier
    """
    
    def __init__(self, X_base: np.ndarray, k_analysis: int = 10):
        """
        Initialize the analyzer.
        
        Args:
            X_base: Base dataset vectors (N x D)
            k_analysis: k for analysis (typically 10 or 20)
        """
        self.X_base = X_base
        self.k_analysis = k_analysis
        self.N, self.D = X_base.shape
        
        # Precompute useful statistics
        self._precompute_statistics()
    
    def _precompute_statistics(self):
        """Precompute dataset statistics for analysis."""
        # Mean and std for normalization
        self.mean = np.mean(self.X_base, axis=0)
        self.std = np.std(self.X_base, axis=0)
        
        # Global distance statistics for normalization
        sample_size = min(10000, self.N)
        sample_indices = np.random.choice(self.N, sample_size, replace=False)
        sample_data = self.X_base[sample_indices]
        
        # Compute pairwise distances for a sample
        from scipy.spatial.distance import pdist
        if sample_size > 1:
            self.global_mean_dist = float(np.mean(pdist(sample_data)))
        else:
            self.global_mean_dist = 1.0
    
    def compute_query_characteristics(
        self,
        query: np.ndarray,
        query_id: int,
        ground_truth: np.ndarray,
        X_base: Optional[np.ndarray] = None
    ) -> QueryCharacteristics:
        """
        Compute characteristics for a single query.
        
        Args:
            query: Query vector
            query_id: Query identifier
            ground_truth: Ground truth indices (k_nn nearest neighbors)
            X_base: Base dataset (if different from initialization)
            
        Returns:
            QueryCharacteristics object
        """
        X = X_base if X_base is not None else self.X_base
        
        char = QueryCharacteristics(query_id=query_id)
        
        # Distance to k-NN
        knn_vectors = X[ground_truth[:self.k_analysis]]
        
        # d1nn: distance to nearest neighbor
        char.d1nn = float(np.linalg.norm(query - knn_vectors[0]))
        
        # dknn: distance to k-th nearest neighbor
        char.dknn = float(np.linalg.norm(query - knn_vectors[-1]))
        
        # d_ratio: dispersion of nearest neighbors
        # High ratio = neighbors are far apart = harder query
        if char.d1nn > 0:
            char.d_ratio = char.dknn / char.d1nn
        else:
            char.d_ratio = 1.0
        
        # Local density: count points within 2*d1nn
        distances_to_all = np.linalg.norm(X - query, axis=1)
        char.local_density = int(np.sum(distances_to_all <= 2 * char.d1nn))
        
        # Isolation score: normalized distance to nearest neighbor
        char.isolation_score = char.d1nn / self.global_mean_dist
        
        # Estimate intrinsic dimensionality using variance of distances
        nn_distances = np.linalg.norm(knn_vectors - query, axis=1)
        if len(nn_distances) > 2:
            char.intrinsic_dim = float(np.var(nn_distances) / (np.mean(nn_distances) + 1e-6))
        else:
            char.intrinsic_dim = 0.0
        
        # Classify difficulty
        char.difficulty, char.difficulty_score = self._classify_difficulty(char)
        
        return char
    
    def _classify_difficulty(
        self, 
        char: QueryCharacteristics
    ) -> Tuple[str, float]:
        """
        Classify query difficulty based on characteristics.
        
        Returns:
            Tuple of (difficulty_label, difficulty_score)
            difficulty_score: 0 = easy, 1 = very hard
        """
        score = 0.0
        
        # Factor 1: Dispersion (d_ratio)
        if char.d_ratio > 3.0:
            score += 0.4
        elif char.d_ratio > 2.0:
            score += 0.25
        elif char.d_ratio > 1.5:
            score += 0.1
        
        # Factor 2: Local density (normalized)
        density_threshold = self.N * 0.001  # 0.1% of dataset
        if char.local_density < 5:
            score += 0.3
        elif char.local_density < density_threshold * 0.1:
            score += 0.15
        
        # Factor 3: Isolation
        if char.isolation_score > 3.0:
            score += 0.3
        elif char.isolation_score > 2.0:
            score += 0.15
        
        # Factor 4: High intrinsic dimension
        if char.intrinsic_dim > 2.0:
            score += 0.2
        elif char.intrinsic_dim > 1.0:
            score += 0.1
        
        # Normalize score
        score = min(1.0, score)
        
        # Classify
        if score < 0.2:
            difficulty = "easy"
        elif score < 0.4:
            difficulty = "medium"
        elif score < 0.6:
            difficulty = "hard"
        else:
            difficulty = "very_hard"
        
        return difficulty, score
    
    def analyze_batch(
        self,
        X_queries: np.ndarray,
        ground_truth: np.ndarray,  # shape: (n_queries, k)
        X_base: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Analyze difficulty for a batch of queries.
        
        Args:
            X_queries: Query vectors (n_queries x D)
            ground_truth: Ground truth indices (n_queries x k)
            X_base: Base dataset
            
        Returns:
            DataFrame with query characteristics
        """
        results = []
        
        for i, (query, gt) in enumerate(zip(X_queries, ground_truth)):
            char = self.compute_query_characteristics(
                query, i, gt, X_base
            )
            results.append({
                "query_id": char.query_id,
                "d1nn": char.d1nn,
                "dknn": char.dknn,
                "d_ratio": char.d_ratio,
                "local_density": char.local_density,
                "isolation_score": char.isolation_score,
                "intrinsic_dim": char.intrinsic_dim,
                "difficulty": char.difficulty,
                "difficulty_score": char.difficulty_score,
            })
        
        return pd.DataFrame(results)


class FailureAnalyzer:
    """
    Analyzes why early termination methods fail on specific queries.
    """
    
    def __init__(
        self,
        query_characteristics: pd.DataFrame,
        per_query_results: pd.DataFrame
    ):
        """
        Initialize failure analyzer.
        
        Args:
            query_characteristics: DataFrame from QueryDifficultyAnalyzer
            per_query_results: DataFrame from PerQueryBenchmark
        """
        self.query_chars = query_characteristics
        self.results = per_query_results
        self._merged = None
    
    def merge_data(self) -> pd.DataFrame:
        """Merge characteristics with results for analysis."""
        self._merged = pd.merge(
            self.results,
            self.query_chars,
            on="query_id",
            how="left"
        )
        return self._merged
    
    def identify_failures(
        self,
        recall_threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        Identify failed queries across methods.
        
        Returns:
            DataFrame of failed queries with failure types
        """
        if self._merged is None:
            self.merge_data()
        
        failures = self._merged[self._merged["recall"] < recall_threshold].copy()
        
        # Classify failure type
        def classify_failure(row):
            if row["d_ratio"] > 2.5:
                return "high_dispersion"
            elif row["local_density"] < 10:
                return "sparse_region"
            elif row["isolation_score"] > 3.0:
                return "outlier"
            elif row["difficulty_score"] > 0.4:
                return "hard_query"
            else:
                return "boundary"
        
        failures["failure_type"] = failures.apply(classify_failure, axis=1)
        return failures
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in failures across methods.
        
        Returns:
            Dictionary with failure statistics
        """
        if self._merged is None:
            self.merge_data()
        
        failures = self.identify_failures()
        
        patterns = {
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(self._merged),
            "by_method": {},
            "by_failure_type": {},
            "cross_method_failures": []
        }
        
        # Failures by method
        for method in failures["method"].unique():
            method_failures = failures[failures["method"] == method]
            patterns["by_method"][method] = {
                "count": len(method_failures),
                "rate": len(method_failures) / len(self._merged[self._merged["method"] == method])
            }
        
        # Failures by type
        for ftype in failures["failure_type"].unique():
            type_failures = failures[failures["failure_type"] == ftype]
            patterns["by_failure_type"][ftype] = len(type_failures)
        
        # Cross-method failures (queries that fail in ALL methods)
        queries_by_method = failures.groupby("query_id")["method"].nunique()
        total_methods = self._merged["method"].nunique()
        cross_failures = queries_by_method[queries_by_method == total_methods].index.tolist()
        patterns["cross_method_failures"] = cross_failures
        patterns["cross_method_failure_rate"] = len(cross_failures) / len(self._merged["query_id"].unique())
        
        return patterns
    
    def get_method_comparison(self) -> pd.DataFrame:
        """
        Compare methods on hard vs easy queries.
        
        Returns:
            DataFrame with comparison statistics
        """
        if self._merged is None:
            self.merge_data()
        
        # Split by difficulty
        easy = self._merged[self._merged["difficulty"].isin(["easy", "medium"])]
        hard = self._merged[self._merged["difficulty"].isin(["hard", "very_hard"])]
        
        comparison = []
        for method in self._merged["method"].unique():
            method_easy = easy[easy["method"] == method]
            method_hard = hard[hard["method"] == method]
            
            comparison.append({
                "method": method,
                "easy_mean_recall": method_easy["recall"].mean() if len(method_easy) > 0 else np.nan,
                "easy_std_recall": method_easy["recall"].std() if len(method_easy) > 0 else np.nan,
                "easy_mean_latency_ms": method_easy["latency_ms"].mean() if len(method_easy) > 0 else np.nan,
                "hard_mean_recall": method_hard["recall"].mean() if len(method_hard) > 0 else np.nan,
                "hard_std_recall": method_hard["recall"].std() if len(method_hard) > 0 else np.nan,
                "hard_mean_latency_ms": method_hard["latency_ms"].mean() if len(method_hard) > 0 else np.nan,
                "recall_gap_easy_vs_hard": (
                    method_easy["recall"].mean() - method_hard["recall"].mean()
                    if len(method_easy) > 0 and len(method_hard) > 0 else np.nan
                )
            })
        
        return pd.DataFrame(comparison)
    
    def recommend_improvements(self) -> List[str]:
        """
        Generate recommendations based on failure analysis.
        
        Returns:
            List of recommendation strings
        """
        patterns = self.analyze_failure_patterns()
        recommendations = []
        
        # Check for high-dispersion failures
        if patterns["by_failure_type"].get("high_dispersion", 0) > patterns["total_failures"] * 0.3:
            recommendations.append(
                "Consider increasing efSearch for queries with high neighbor dispersion. "
                "These queries have k-NN that are far apart from each other."
            )
        
        # Check for sparse region failures
        if patterns["by_failure_type"].get("sparse_region", 0) > patterns["total_failures"] * 0.2:
            recommendations.append(
                "Sparse region queries need more exploration. "
                "Consider disabling early termination for queries in low-density areas."
            )
        
        # Check for cross-method failures
        if patterns["cross_method_failure_rate"] > 0.1:
            recommendations.append(
                f"{patterns['cross_method_failure_rate']:.1%} of queries fail in ALL methods. "
                "These are fundamentally hard queries - consider using higher efSearch as default."
            )
        
        # Compare methods
        comparison = self.get_method_comparison()
        for _, row in comparison.iterrows():
            if row["recall_gap_easy_vs_hard"] > 0.15:
                recommendations.append(
                    f"{row['method']} has large recall gap ({row['recall_gap_easy_vs_hard']:.2f}) "
                    f"between easy and hard queries. Consider adaptive parameters."
                )
        
        return recommendations
