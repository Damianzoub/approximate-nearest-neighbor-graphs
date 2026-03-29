"""
Reproducibility utilities for ANNS benchmarking.

This module provides functions to ensure reproducible experiments
by controlling random seeds across all relevant libraries.
"""

import os
import random
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np


def get_random_state_info() -> Dict[str, Any]:
    """
    Get information about current random state sources.
    
    Returns:
        Dictionary with information about random state
    """
    return {
        "python_random": random.getstate()[1][:5] if random.getstate()[1] else None,
        "numpy_state": np.random.get_state()[1][:5] if np.random.get_state()[1] else None,
        "env_pyhash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def set_seed(seed: int = 42, set_env: bool = True) -> None:
    """
    Set random seed for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: The random seed to use (default: 42)
        set_env: Whether to set PYTHONHASHSEED environment variable
    """
    if set_env:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except ImportError:
        pass


def get_experiment_hash(config: Dict[str, Any]) -> str:
    """
    Generate a unique hash for an experiment configuration.
    
    Args:
        config: Dictionary containing experiment configuration
        
    Returns:
        8-character hash string
    """
    import json
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def create_experiment_id(
    experiment_name: str,
    dataset_name: str,
    seed: Optional[int] = None,
    timestamp: bool = True
) -> str:
    """
    Create a unique experiment identifier.
    
    Args:
        experiment_name: Name of the experiment
        dataset_name: Name of the dataset used
        seed: Random seed used (if any)
        timestamp: Whether to include timestamp
        
    Returns:
        Unique experiment identifier string
    """
    parts = [experiment_name, dataset_name]
    
    if seed is not None:
        parts.append(f"seed{seed}")
    
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(ts)
    
    return "_".join(parts)


def log_experiment_info(
    experiment_name: str,
    dataset_name: str,
    parameters: Dict[str, Any],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Log experiment information for reproducibility.
    
    Args:
        experiment_name: Name of the experiment
        dataset_name: Name of the dataset
        parameters: Experiment parameters
        output_file: Optional file to write log to
        
    Returns:
        Dictionary with experiment metadata
    """
    info = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "parameters": parameters,
        "random_state": get_random_state_info(),
    }
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    return info


class ExperimentContext:
    """
    Context manager for reproducible experiments.
    
    Usage:
        with ExperimentContext("my_experiment", seed=42):
            # Run experiment code here
            pass
    """
    
    def __init__(
        self,
        name: str,
        seed: int = 42,
        log_file: Optional[str] = None
    ):
        self.name = name
        self.seed = seed
        self.log_file = log_file
        self.initial_state = None
        
    def __enter__(self):
        self.initial_state = get_random_state_info()
        set_seed(self.seed)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            log_experiment_info(
                self.name,
                "",
                {"seed": self.seed},
                self.log_file
            )
        return False


def verify_reproducibility(
    test_fn,
    seed: int = 42,
    runs: int = 3
) -> bool:
    """
    Verify that a function produces identical results across runs.
    
    Args:
        test_fn: Function to test (should return hashable results)
        seed: Random seed to use
        runs: Number of times to run the test
        
    Returns:
        True if all runs produce identical results
    """
    results = []
    
    for _ in range(runs):
        set_seed(seed)
        result = test_fn()
        results.append(result)
    
    return all(r == results[0] for r in results)
