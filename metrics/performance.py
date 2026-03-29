"""
Memory and performance measurement utilities for ANNS benchmarking.

This module provides tools for measuring memory usage, build time,
and other performance metrics.
"""

import time
import os
import sys
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MemoryStats:
    """Container for memory statistics."""
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    peak_rss_mb: float = 0.0
    available_mb: float = 0.0
    percent_used: float = 0.0


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    recall: float = 0.0
    qps: float = 0.0
    latency_ms: float = 0.0
    build_time_s: float = 0.0
    memory_stats: MemoryStats = field(default_factory=MemoryStats)
    index_size_mb: float = 0.0
    num_visited: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall": self.recall,
            "qps": self.qps,
            "latency_ms": self.latency_ms,
            "build_time_s": self.build_time_s,
            "memory_rss_mb": self.memory_stats.rss_mb,
            "memory_vms_mb": self.memory_stats.vms_mb,
            "memory_peak_rss_mb": self.memory_stats.peak_rss_mb,
            "memory_available_mb": self.memory_stats.available_mb,
            "memory_percent": self.memory_stats.percent_used,
            "index_size_mb": self.index_size_mb,
            "num_visited": self.num_visited,
        }


def get_memory_usage() -> MemoryStats:
    """
    Get current memory usage statistics.
    
    Returns:
        MemoryStats object with current memory info
    """
    if not PSUTIL_AVAILABLE:
        return MemoryStats()
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm_stat = psutil.virtual_memory()
    
    return MemoryStats(
        rss_mb=mem_info.rss / (1024 * 1024),
        vms_mb=mem_info.vms / (1024 * 1024),
        peak_rss_mb=mem_info.rss / (1024 * 1024),
        available_mb=vm_stat.available / (1024 * 1024),
        percent_used=vm_stat.percent,
    )


def get_peak_memory() -> float:
    """
    Get peak memory usage of current process in MB.
    
    Returns:
        Peak RSS memory in MB
    """
    if not PSUTIL_AVAILABLE:
        return 0.0
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for experiment metadata.
    
    Returns:
        Dictionary with system/hardware info
    """
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
    }
    
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        info.update({
            "total_memory_gb": mem.total / (1024 ** 3),
            "available_memory_gb": mem.available / (1024 ** 3),
        })
        
        try:
            info["cpu_freq_mhz"] = psutil.cpu_freq().current
        except (AttributeError, OSError):
            pass
    
    return info


def estimate_index_size(
    num_vectors: int,
    dimension: int,
    M: int,
    bytes_per_float: int = 4
) -> float:
    """
    Estimate HNSW index size in MB.
    
    Args:
        num_vectors: Number of vectors in the index
        dimension: Dimensionality of vectors
        M: Number of connections per node
        bytes_per_float: Bytes per float (4 for float32)
        
    Returns:
        Estimated index size in MB
    """
    vector_size = num_vectors * dimension * bytes_per_float / (1024 * 1024)
    
    layer_factor = 1 / (1 - 1 / (M + 1))
    neighbor_size = num_vectors * M * 8 / (1024 * 1024)
    
    overhead = num_vectors * 16 / (1024 * 1024)
    
    total_mb = vector_size + neighbor_size * layer_factor + overhead
    return total_mb


@contextmanager
def measure_time():
    """
    Context manager to measure execution time.
    
    Usage:
        with measure_time() as timer:
            # code to time
        print(f"Elapsed: {timer.elapsed}s")
    """
    class Timer:
        def __init__(self):
            self.start = time.perf_counter()
            self.end = None
            
        @property
        def elapsed(self) -> float:
            if self.end:
                return self.end - self.start
            return time.perf_counter() - self.start
    
    yield Timer()
    Timer.end = time.perf_counter()


@contextmanager
def measure_memory():
    """
    Context manager to measure peak memory usage.
    
    Usage:
        with measure_memory() as mem:
            # code to measure
        print(f"Peak memory: {mem.peak_mb:.2f} MB")
    """
    if not PSUTIL_AVAILABLE:
        class DummyMem:
            peak_mb = 0.0
        yield DummyMem()
        return
    
    process = psutil.Process(os.getpid())
    initial_rss = process.memory_info().rss
    
    class MemTracker:
        def __init__(self):
            self.peak_mb = 0.0
            
        def update(self):
            current = process.memory_info().rss
            if current > self.peak_mb * 1024 * 1024:
                self.peak_mb = current / (1024 * 1024)
    
    tracker = MemTracker()
    yield tracker
    tracker.update()


class PerformanceMonitor:
    """
    Monitor for tracking performance during benchmark runs.
    
    Usage:
        monitor = PerformanceMonitor()
        monitor.start()
        # run benchmarks
        monitor.stop()
        stats = monitor.get_stats()
    """
    
    def __init__(self, track_memory: bool = True):
        self.track_memory = track_memory
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = 0.0
        self.iterations = 0
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.perf_counter()
        if self.track_memory:
            self.start_memory = get_memory_usage()
            self.peak_memory = self.start_memory.rss_mb
            
    def stop(self):
        """Stop monitoring."""
        self.end_time = time.perf_counter()
        if self.track_memory:
            current_mem = get_memory_usage()
            self.peak_memory = max(self.peak_memory, current_mem.rss_mb)
            
    def tick(self):
        """Record an iteration."""
        self.iterations += 1
        if self.track_memory:
            current_mem = get_memory_usage()
            self.peak_memory = max(self.peak_memory, current_mem.rss_mb)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        elapsed = 0.0
        if self.start_time and self.end_time:
            elapsed = self.end_time - self.start_time
            
        return {
            "elapsed_s": elapsed,
            "iterations": self.iterations,
            "avg_iteration_s": elapsed / max(self.iterations, 1),
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": self.peak_memory - (self.start_memory.rss_mb if self.start_memory else 0),
        }


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    num_runs: int = 5,
    warmup_runs: int = 2,
    track_memory: bool = True
) -> Tuple[BenchmarkResult, Dict[str, Any]]:
    """
    Benchmark a function with multiple runs and statistics.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        track_memory: Whether to track memory usage
        
    Returns:
        Tuple of (BenchmarkResult with averages, dict with all run data)
    """
    if kwargs is None:
        kwargs = {}
    
    all_results = []
    
    for i in range(num_runs + warmup_runs):
        monitor = PerformanceMonitor(track_memory=track_memory)
        
        monitor.start()
        result = func(*args, **kwargs)
        monitor.stop()
        
        stats = monitor.get_stats()
        stats["run"] = i
        
        if i >= warmup_runs:
            all_results.append(stats)
    
    avg_result = BenchmarkResult()
    
    if all_results:
        avg_result.build_time_s = sum(r["elapsed_s"] for r in all_results) / len(all_results)
        avg_result.latency_ms = sum(r["avg_iteration_s"] for r in all_results) / len(all_results) * 1000
        
        if track_memory:
            avg_result.memory_stats = MemoryStats(
                peak_rss_mb=sum(r["peak_memory_mb"] for r in all_results) / len(all_results)
            )
    
    return avg_result, all_results
