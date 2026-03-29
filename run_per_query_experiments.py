#!/usr/bin/env python3
"""
Per-Query Experiment Runner for ANNS Failure Analysis.

This script runs comprehensive per-query benchmarks across all methods
and produces detailed analysis of when and why early termination fails.

Usage:
    python run_per_query_experiments.py
    python run_per_query_experiments.py --dataset siftsmall --max-queries 100
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.read_files import read_fvecs, read_ivecs
from utils.reproducibility import set_seed
from metrics.per_query_benchmark import PerQueryBenchmark, PerQueryResult
from metrics.query_difficulty import QueryDifficultyAnalyzer, FailureAnalyzer


def load_dataset(dataset_name: str, base_dir: Path) -> Dict[str, Any]:
    """Load a dataset from the standard location."""
    dataset_path = base_dir / "Datasets" / dataset_name
    
    base_file = dataset_path / f"{dataset_name}_base.fvecs"
    query_file = dataset_path / f"{dataset_name}_query.fvecs"
    gt_file = dataset_path / f"{dataset_name}_groundtruth.ivecs"
    
    if not all(f.exists() for f in [base_file, query_file, gt_file]):
        raise FileNotFoundError(f"Dataset files not found in {dataset_path}")
    
    print(f"Loading dataset: {dataset_name}")
    Xb = read_fvecs(str(base_file))
    Xq = read_fvecs(str(query_file))
    I_gt = read_ivecs(str(gt_file))
    
    return {
        "name": dataset_name,
        "X_base": Xb,
        "X_query": Xq,
        "ground_truth": I_gt,
        "num_base": Xb.shape[0],
        "num_query": Xq.shape[0],
        "dimension": Xb.shape[1],
        "k_max": I_gt.shape[1]
    }


def build_indices(dataset: Dict[str, Any], M: int = 16, efC: int = 200) -> Dict[str, Any]:
    """Build all index structures."""
    from hnsw_construction import HNSW_NEW
    from hsnw_constructionDARTH import HNSW_DARTH, DummyPredictor
    from hnsw_pip import HNSW_PiP
    from hnsw_adaef import HNSW_AdaEF
    
    Xb = dataset["X_base"]
    dim = dataset["dimension"]
    
    indices = {}
    
    # HNSW
    print("  Building HNSW index...")
    indices["hnsw"] = HNSW_NEW(dim=dim, M=M, efConstruction=efC)
    for i, vec in enumerate(Xb):
        indices["hnsw"]._insert_(vec, i)
    
    # DARTH
    print("  Building DARTH index...")
    indices["darth"] = HNSW_DARTH(dim=dim, M=M, efConstruction=efC)
    for i, vec in enumerate(Xb):
        indices["darth"]._insert_(vec, i)
    
    # PiP
    print("  Building PiP index...")
    indices["pip"] = HNSW_PiP(
        dim=dim, M=M, efConstruction=efC,
        pip_gamma=95.0, pip_delta=20
    )
    for i, vec in enumerate(Xb):
        indices["pip"]._insert_(vec, i)
    
    # Ada-ef
    print("  Building Ada-ef index...")
    indices["adaef"] = HNSW_AdaEF(dim=dim, M=M, efConstruction=efC)
    for i, vec in enumerate(Xb):
        indices["adaef"]._insert_(vec, i)
    
    return indices


def run_hnsw_benchmark(
    index,
    Xq: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    ef_values: List[int],
    max_queries: int
) -> List[PerQueryResult]:
    """Run HNSW per-query benchmark."""
    benchmark = PerQueryBenchmark()
    results = []
    
    Xq_subset = Xq[:max_queries]
    gt_subset = ground_truth[:max_queries]
    
    for ef in ef_values:
        print(f"    Testing HNSW with efSearch={ef}...")
        
        for i, (query, gt) in enumerate(zip(Xq_subset, gt_subset)):
            start = datetime.now()
            D, I = index.search(query.reshape(1, -1), k, ef)
            elapsed = (datetime.now() - start).total_seconds() * 1000
            
            # Calculate recall
            true_k = gt[:k]
            found_k = I[0][:k] if len(I[0]) >= k else I[0]
            recall = len(set(true_k).intersection(set(found_k))) / k
            
            result = PerQueryResult(
                query_id=i,
                method=f"hnsw_ef{ef}",
                recall=recall,
                latency_ms=elapsed,
                true_neighbors_found=len(set(true_k).intersection(set(found_k))),
                terminated_early=False,
                actual_ef_used=ef,
                max_ef_allowed=ef,
                num_iterations=0,
                num_distance_computations=0,
                num_nodes_visited=0,
                distance_to_best=float(D[0][0]) if len(D[0]) > 0 else 0.0,
                distance_to_worst_in_topk=float(D[0][min(k-1, len(D[0])-1)]) if len(D[0]) > 0 else 0.0,
                distance_ratio=1.0,
            )
            results.append(result)
    
    return results


def run_darth_benchmark(
    index,
    Xq: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    Rt_values: List[float],
    max_queries: int
) -> List[PerQueryResult]:
    """Run DARTH per-query benchmark."""
    from hsnw_constructionDARTH import DummyPredictor
    
    results = []
    Xq_subset = Xq[:max_queries]
    gt_subset = ground_truth[:max_queries]
    
    for Rt in Rt_values:
        print(f"    Testing DARTH with Rt={Rt}...")
        
        for i, (query, gt) in enumerate(zip(Xq_subset, gt_subset)):
            start = datetime.now()
            I = index.query_darth(
                query, k, 
                efSearch=100, 
                Rt=Rt, 
                predictor=DummyPredictor(),
                ipi=200, 
                mpi=20
            )
            elapsed = (datetime.now() - start).total_seconds() * 1000
            
            # Calculate recall
            true_k = gt[:k]
            found_k = I[:k] if len(I) >= k else I
            recall = len(set(true_k).intersection(set(found_k))) / k
            
            result = PerQueryResult(
                query_id=i,
                method=f"darth_Rt{Rt}",
                recall=recall,
                latency_ms=elapsed,
                true_neighbors_found=len(set(true_k).intersection(set(found_k))),
                terminated_early=True,  # DARTH always uses early termination
                actual_ef_used=0,  # Would need instrumentation
                max_ef_allowed=100,
                num_iterations=0,
                num_distance_computations=0,
                num_nodes_visited=0,
            )
            results.append(result)
    
    return results


def run_pip_benchmark(
    index,
    Xq: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    gamma_values: List[float],
    max_queries: int
) -> List[PerQueryResult]:
    """Run PiP per-query benchmark."""
    results = []
    Xq_subset = Xq[:max_queries]
    gt_subset = ground_truth[:max_queries]
    
    for gamma in gamma_values:
        print(f"    Testing PiP with gamma={gamma}...")
        
        # Rebuild with new gamma
        from hnsw_pip import HNSW_PiP
        pip_index = HNSW_PiP(
            dim=index.dim, 
            M=index.M, 
            efConstruction=index.efConstruction,
            pip_gamma=gamma, 
            pip_delta=20
        )
        for i, vec in enumerate(index.vectors.values()):
            pip_index._insert_(vec, i)
        
        for i, (query, gt) in enumerate(zip(Xq_subset, gt_subset)):
            start = datetime.now()
            D,I = pip_index.search(query.reshape(1,-1),k,efSearch=100)
            elapsed = (datetime.now() - start).total_seconds() * 1000
            
            # Calculate recall
            true_k = gt[:k]
            found_k = I[0][:k] if len(I[0]) >= k else I[0]
            recall = len(set(true_k).intersection(set(found_k))) / k
            
            result = PerQueryResult(
                query_id=i,
                method=f"pip_gamma{gamma}",
                recall=recall,
                latency_ms=elapsed,
                true_neighbors_found=len(set(true_k).intersection(set(found_k))),
                terminated_early=True,
                actual_ef_used=0,
                max_ef_allowed=100,
                num_iterations=0,
                num_distance_computations=0,
                num_nodes_visited=0,
            )
            results.append(result)
    
    return results


def run_adaef_benchmark(
    index,
    Xq: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    target_recall_values: List[float],
    max_queries: int
) -> List[PerQueryResult]:
    """Run Ada-ef per-query benchmark."""
    results = []
    Xq_subset = Xq[:max_queries]
    gt_subset = ground_truth[:max_queries]
    
    for target_recall in target_recall_values:
        print(f"    Testing Ada-ef with target_recall={target_recall}...")
        
        index.build_adaef_offline(k=k, target_recall=target_recall)
        
        for i, (query, gt) in enumerate(zip(Xq_subset, gt_subset)):
            start = datetime.now()
            D, I = index.search(query.reshape(1, -1), k, target_recall)
            elapsed = (datetime.now() - start).total_seconds() * 1000
            
            true_k = gt[:k]
            found_k = I[0][:k] if len(I[0]) >= k else I[0]
            recall = len(set(true_k).intersection(set(found_k))) / k
            
            result = PerQueryResult(
                query_id=i,
                method=f"adaef_target{target_recall}",
                recall=recall,
                latency_ms=elapsed,
                true_neighbors_found=len(set(true_k).intersection(set(found_k))),
                terminated_early=True,
                actual_ef_used=0,
                max_ef_allowed=0,
                num_iterations=0,
                num_distance_computations=0,
                num_nodes_visited=0,
            )
            results.append(result)
    
    return results


def compute_query_characteristics(
    dataset: Dict[str, Any],
    max_queries: int
) -> pd.DataFrame:
    """Compute query difficulty characteristics."""
    analyzer = QueryDifficultyAnalyzer(dataset["X_base"], k_analysis=10)
    
    Xq = dataset["X_query"][:max_queries]
    gt = dataset["ground_truth"][:max_queries]
    
    chars_df = analyzer.analyze_batch(Xq, gt)
    return chars_df


def run_analysis(
    per_query_results: List[PerQueryResult],
    query_chars: pd.DataFrame
) -> Dict[str, Any]:
    """Run failure analysis on results."""
    # Convert to DataFrame
    results_df = pd.DataFrame([vars(r) for r in per_query_results])
    
    # Create analyzer
    analyzer = FailureAnalyzer(query_chars, results_df)
    
    # Run analysis
    patterns = analyzer.analyze_failure_patterns()
    comparison = analyzer.get_method_comparison()
    recommendations = analyzer.recommend_improvements()
    
    return {
        "patterns": patterns,
        "comparison": comparison.to_dict(),
        "recommendations": recommendations
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run per-query ANNS benchmarks for failure analysis"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="siftsmall",
        help="Dataset name (default: siftsmall)"
    )
    parser.add_argument(
        "--max-queries", "-n",
        type=int,
        default=100,
        help="Maximum queries to test (default: 100)"
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=10,
        help="k for k-NN search (default: 10)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results_csv/per_query",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Per-Query ANNS Failure Analysis")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset(args.dataset, base_dir)
    print(f"  Base: {dataset['num_base']:,} vectors")
    print(f"  Query: {dataset['num_query']:,} vectors")
    print(f"  Dimension: {dataset['dimension']}")
    print(f"  Testing on: {args.max_queries} queries")
    print()
    
    # Build indices
    print("Building indices...")
    indices = build_indices(dataset)
    print()
    
    # Run benchmarks
    all_results = []
    
    # HNSW with different efSearch values
    hnsw_results = run_hnsw_benchmark(
        indices["hnsw"],
        dataset["X_query"],
        dataset["ground_truth"],
        args.k,
        ef_values=[20, 50, 100],
        max_queries=args.max_queries
    )
    all_results.extend(hnsw_results)
    
    # DARTH with different Rt values
    darth_results = run_darth_benchmark(
        indices["darth"],
        dataset["X_query"],
        dataset["ground_truth"],
        args.k,
        Rt_values=[0.85, 0.90, 0.95],
        max_queries=args.max_queries
    )
    all_results.extend(darth_results)
    
    # PiP with different gamma values
    pip_results = run_pip_benchmark(
        indices["pip"],
        dataset["X_query"],
        dataset["ground_truth"],
        args.k,
        gamma_values=[90.0, 95.0, 99.0],
        max_queries=args.max_queries
    )
    all_results.extend(pip_results)
    
    # Ada-ef with different target recall values
    adaef_results = run_adaef_benchmark(
        indices["adaef"],
        dataset["X_query"],
        dataset["ground_truth"],
        args.k,
        target_recall_values=[0.90, 0.95, 0.99],
        max_queries=args.max_queries
    )
    all_results.extend(adaef_results)
    
    print()
    print(f"Total results collected: {len(all_results)}")
    
    # Compute query characteristics
    print("Computing query difficulty characteristics...")
    query_chars = compute_query_characteristics(dataset, args.max_queries)
    print()
    
    # Run analysis
    print("Running failure analysis...")
    analysis = run_analysis(all_results, query_chars)
    print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save per-query results
    results_df = pd.DataFrame([vars(r) for r in all_results])
    results_file = output_dir / f"per_query_results_{args.dataset}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved per-query results to: {results_file}")
    
    # Save query characteristics
    chars_file = output_dir / f"query_characteristics_{args.dataset}_{timestamp}.csv"
    query_chars.to_csv(chars_file, index=False)
    print(f"Saved query characteristics to: {chars_file}")
    
    # Save analysis
    analysis_file = output_dir / f"analysis_{args.dataset}_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Saved analysis to: {analysis_file}")
    
    # Print summary
    print()
    print("=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal queries analyzed: {args.max_queries}")
    print(f"Total failures (recall < 0.9): {analysis['patterns']['total_failures']}")
    print(f"Failure rate: {analysis['patterns']['failure_rate']:.1%}")
    
    print("\nFailures by method:")
    for method, stats in analysis['patterns']['by_method'].items():
        print(f"  {method}: {stats['count']} ({stats['rate']:.1%})")
    
    print("\nFailures by type:")
    for ftype, count in analysis['patterns']['by_failure_type'].items():
        print(f"  {ftype}: {count}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
