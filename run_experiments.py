"""
Automated Experiment Runner for ANNS Benchmarking.

This script provides a unified interface for running reproducible
experiments based on YAML configuration files.

Usage:
    python run_experiments.py --config configs/experiments.yaml --profile full_benchmark
    python run_experiments.py --config configs/experiments.yaml --dataset sift --algorithm hnsw
    python run_experiments.py --quick-test
"""

import argparse
import yaml
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.reproducibility import set_seed, create_experiment_id, log_experiment_info
from utils.read_files import read_fvecs, read_ivecs
from metrics.performance import (
    PerformanceMonitor, get_memory_usage, get_system_info,
    estimate_index_size, BenchmarkResult
)
from testing.comparing_cpp_algorithms import (
    build_hnsw_cpp, hnsw_cpp_search_fn,
    build_hnsw_darth_cpp, hnsw_darth_cpp_search_fn,
    build_hnsw_pip_cpp, hnsw_pip_cpp_search_fn,
    build_hnsw_adaef_cpp, hnsw_adaef_cpp_search_fn,
)
from metrics.benchMark import measure


class ExperimentRunner:
    """
    Main experiment runner class.
    
    Handles loading configuration, running benchmarks,
    and saving results with full reproducibility metadata.
    """
    
    def __init__(self, config_path: str, profile: Optional[str] = None):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to YAML configuration file
            profile: Optional profile name to override default settings
        """
        self.config = self._load_config(config_path)
        self.profile = profile or self.config.get("experiment", {}).get("name", "default")
        self.base_dir = Path(__file__).parent.parent
        self.output_dir = self.base_dir / self.config.get("metrics", {}).get("output_dir", "results_csv")
        self.plots_dir = self.base_dir / self.config.get("metrics", {}).get("plots_dir", "plot_results")
        
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        self.system_info = get_system_info()
        self.results = []
        self.metadata = {
            "config_path": str(config_path),
            "profile": self.profile,
            "system_info": self.system_info,
            "experiments": []
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _merge_with_profile(self, profile_name: str) -> Dict[str, Any]:
        """Merge base config with profile-specific settings."""
        base = self.config.copy()
        profiles = self.config.get("profiles", {})
        
        if profile_name in profiles:
            profile = profiles[profile_name]
            for key, value in profile.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key].update(value)
                else:
                    base[key] = value
                    
        return base
    
    def _load_dataset(self, dataset_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load a dataset based on configuration."""
        name = dataset_config.get("name")
        base_path = self.base_dir / dataset_config.get("base", "")
        query_path = self.base_dir / dataset_config.get("query", "")
        gt_path = self.base_dir / dataset_config.get("gt", "")
        
        if not all(p.exists() for p in [base_path, query_path, gt_path]):
            print(f"  Warning: Dataset '{name}' files not found, skipping...")
            return None
            
        print(f"  Loading {name}...")
        xb = read_fvecs(str(base_path))
        xq = read_fvecs(str(query_path))
        gt = read_ivecs(str(gt_path))
        
        return {
            "name": name,
            "xbase": xb,
            "xquery": xq,
            "groundtruth": gt,
            "dimension": xb.shape[1],
            "num_base": xb.shape[0],
            "num_query": xq.shape[0],
        }
    
    def _run_hnsw_benchmark(
        self,
        dataset: Dict[str, Any],
        params: Dict[str, Any],
        seed: int
    ) -> List[Dict[str, Any]]:
        """Run HNSW baseline benchmark."""
        results = []
        M = params.get("M", 16)
        efC = params.get("efConstruction", 200)
        
        print(f"    Building HNSW (M={M}, efC={efC})...")
        monitor = PerformanceMonitor(track_memory=True)
        monitor.start()
        idx = build_hnsw_cpp(dataset["xbase"], M=M, efC=efC)
        monitor.stop()
        build_stats = monitor.get_stats()
        
        for efS in params.get("efSearch", [50, 100]):
            for k in self.config.get("metrics", {}).get("k_values", [10]):
                for run in range(self.config.get("experiment", {}).get("num_runs", 1)):
                    set_seed(seed + run)
                    
                    search_fn = hnsw_cpp_search_fn(idx, efS=efS)
                    metrics = measure(search_fn, dataset["xquery"], k, I_true=dataset["groundtruth"])
                    
                    result = {
                        "method": "hnsw",
                        "M": M,
                        "efConstruction": efC,
                        "efSearch": efS,
                        "k": k,
                        "run": run,
                        **metrics,
                        **build_stats,
                        "index_size_mb": estimate_index_size(
                            dataset["num_base"], dataset["dimension"], M
                        )
                    }
                    results.append(result)
                    
        return results
    
    def _run_darth_benchmark(
        self,
        dataset: Dict[str, Any],
        params: Dict[str, Any],
        seed: int
    ) -> List[Dict[str, Any]]:
        """Run DARTH benchmark."""
        results = []
        M = params.get("M", 16)
        efC = params.get("efConstruction", 200)
        
        print(f"    Building DARTH (M={M}, efC={efC})...")
        idx = build_hnsw_darth_cpp(dataset["xbase"], M=M, efC=efC)
        
        for efS in params.get("efSearch", [50, 100]):
            for Rt in params.get("Rt", [0.90, 0.95]):
                for k in self.config.get("metrics", {}).get("k_values", [10]):
                    for run in range(self.config.get("experiment", {}).get("num_runs", 1)):
                        set_seed(seed + run)
                        
                        search_fn = hnsw_darth_cpp_search_fn(
                            idx, efS=efS, Rt=Rt,
                            ipi=params.get("ipi", [200])[0],
                            mpi=params.get("mpi", [20])[0],
                            predictor=None
                        )
                        metrics = measure(search_fn, dataset["xquery"], k, I_true=dataset["groundtruth"])
                        
                        result = {
                            "method": "darth",
                            "M": M,
                            "efConstruction": efC,
                            "efSearch": efS,
                            "Rt": Rt,
                            "k": k,
                            "run": run,
                            **metrics
                        }
                        results.append(result)
                        
        return results
    
    def _run_pip_benchmark(
        self,
        dataset: Dict[str, Any],
        params: Dict[str, Any],
        seed: int
    ) -> List[Dict[str, Any]]:
        """Run PiP benchmark."""
        results = []
        M = params.get("M", 16)
        efC = params.get("efConstruction", 200)
        
        for gamma in params.get("pip_gamma", [95]):
            for delta in params.get("pip_delta", [10]):
                print(f"    Building PiP (M={M}, gamma={gamma}, delta={delta})...")
                idx = build_hnsw_pip_cpp(
                    dataset["xbase"], M=M, efC=efC,
                    pip_gamma=gamma, pip_delta=delta
                )
                
                for efS in params.get("efSearch", [50, 100]):
                    for k in self.config.get("metrics", {}).get("k_values", [10]):
                        for run in range(self.config.get("experiment", {}).get("num_runs", 1)):
                            set_seed(seed + run)
                            
                            search_fn = hnsw_pip_cpp_search_fn(idx, efS=efS)
                            metrics = measure(search_fn, dataset["xquery"], k, I_true=dataset["groundtruth"])
                            
                            result = {
                                "method": "pip",
                                "M": M,
                                "efConstruction": efC,
                                "efSearch": efS,
                                "pip_gamma": gamma,
                                "pip_delta": delta,
                                "k": k,
                                "run": run,
                                **metrics
                            }
                            results.append(result)
                            
        return results
    
    def _run_adaef_benchmark(
        self,
        dataset: Dict[str, Any],
        params: Dict[str, Any],
        seed: int
    ) -> List[Dict[str, Any]]:
        """Run Ada-ef benchmark."""
        results = []
        M = params.get("M", 16)
        efC = params.get("efConstruction", 200)
        
        print(f"    Building Ada-ef (M={M}, efC={efC})...")
        idx = build_hnsw_adaef_cpp(dataset["xbase"], M=M, efC=efC)
        
        for target_recall in params.get("target_recall", [0.90, 0.95, 0.99]):
            for k in self.config.get("metrics", {}).get("k_values", [10]):
                for run in range(self.config.get("experiment", {}).get("num_runs", 1)):
                    set_seed(seed + run)
                    
                    search_fn = hnsw_adaef_cpp_search_fn(idx, target_recall=target_recall)
                    metrics = measure(search_fn, dataset["xquery"], k, I_true=dataset["groundtruth"])
                    
                    result = {
                        "method": "adaef",
                        "M": M,
                        "efConstruction": efC,
                        "target_recall": target_recall,
                        "k": k,
                        "run": run,
                        **metrics
                    }
                    results.append(result)
                    
        return results
    
    def run_experiment(
        self,
        dataset_name: Optional[str] = None,
        algorithm: Optional[str] = None,
        profile: Optional[str] = None
    ) -> None:
        """
        Run the experiment suite.
        
        Args:
            dataset_name: Specific dataset to run (or None for all)
            algorithm: Specific algorithm to run (hnsw, darth, pip, adaef)
            profile: Configuration profile to use
        """
        config = self._merge_with_profile(profile or self.profile)
        seed = config.get("experiment", {}).get("seed", 42)
        
        print("=" * 60)
        print(f"ANNS Experiment Runner")
        print(f"Profile: {profile or self.profile}")
        print(f"Seed: {seed}")
        print("=" * 60)
        
        datasets_to_run = []
        for name, ds_config in config.get("datasets", {}).items():
            if dataset_name is None or name == dataset_name:
                datasets_to_run.append((name, ds_config))
        
        for ds_name, ds_config in datasets_to_run:
            print(f"\nLoading dataset: {ds_name}")
            dataset = self._load_dataset(ds_config)
            if dataset is None:
                continue
                
            print(f"  Base: {dataset['num_base']:,} vectors")
            print(f"  Query: {dataset['num_query']:,} vectors")
            print(f"  Dimension: {dataset['dimension']}")
            
            exp_id = create_experiment_id(self.profile, ds_name, seed)
            
            for alg_name, alg_params in [("hnsw", config.get("hnsw", {})),
                                          ("darth", config.get("darth", {})),
                                          ("pip", config.get("pip", {})),
                                          ("adaef", config.get("adaef", {}))]:
                if algorithm is not None and alg_name != algorithm:
                    continue
                    
                if not alg_params.get("enabled", False):
                    continue
                    
                print(f"\n  Running {alg_name.upper()}...")
                
                if alg_name == "hnsw":
                    results = self._run_hnsw_benchmark(dataset, alg_params, seed)
                elif alg_name == "darth":
                    results = self._run_darth_benchmark(dataset, alg_params, seed)
                elif alg_name == "pip":
                    results = self._run_pip_benchmark(dataset, alg_params, seed)
                elif alg_name == "adaef":
                    results = self._run_adaef_benchmark(dataset, alg_params, seed)
                    
                self.results.extend(results)
                
        self._save_results()
        print("\n" + "=" * 60)
        print(f"Experiments complete. Results saved to {self.output_dir}")
        print("=" * 60)
        
    def _save_results(self) -> None:
        """Save results to CSV and JSON."""
        import pandas as pd
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = self.output_dir / f"experiments_{self.profile}_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        meta_path = self.output_dir / f"experiments_{self.profile}_{timestamp}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"Metadata saved to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated ANNS Experiment Runner"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/experiments.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        help="Experiment profile to use (overrides config default)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Specific dataset to run (default: all)"
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["hnsw", "darth", "pip", "adaef"],
        help="Specific algorithm to run (default: all)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test profile"
    )
    
    args = parser.parse_args()
    
    profile = "quick_test" if args.quick_test else args.profile
    
    runner = ExperimentRunner(args.config, profile=profile)
    runner.run_experiment(
        dataset_name=args.dataset,
        algorithm=args.algorithm,
        profile=profile
    )


if __name__ == "__main__":
    main()
