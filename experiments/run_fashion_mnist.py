"""
Experiment script for Fashion-MNIST (55K base, 10K queries, 784-dim, L2).

Results  → results_csv/fashion_mnist/all_results.csv
Failures → results_csv/fashion_mnist/failure_analysis.csv

Usage
-----
    python experiments/run_fashion_mnist.py
    python experiments/run_fashion_mnist.py --no-pip --no-adaef
    python experiments/run_fashion_mnist.py --max-learn 3000   # faster
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments._core import run_dataset_experiment

PATHS = {
    "base":  str(ROOT / "Datasets/fashion_mnist/fashion_mnist_base.fvecs"),
    "learn": str(ROOT / "Datasets/fashion_mnist/fashion_mnist_learn.fvecs"),
    "query": str(ROOT / "Datasets/fashion_mnist/fashion_mnist_query.fvecs"),
    "gt":    str(ROOT / "Datasets/fashion_mnist/fashion_mnist_groundtruth.ivecs"),
}

# Fashion-MNIST is large (55K, 784-dim) — keep M sweep small to control build time
M_VALUES   = [16]
EFC_VALUES = [200]


def main():
    parser = argparse.ArgumentParser(description="Run experiments on Fashion-MNIST.")
    parser.add_argument("--no-pip",   action="store_true", help="Skip PiP experiments")
    parser.add_argument("--no-adaef", action="store_true", help="Skip Ada-ef experiments")
    parser.add_argument("--max-learn", type=int, default=3000,
                        help="Max learn vectors for predictor training (default: 3000)")
    args = parser.parse_args()

    run_dataset_experiment(
        dataset_name="fashion_mnist",
        paths=PATHS,
        M_values=M_VALUES,
        efC_values=EFC_VALUES,
        metric="l2",
        max_learn=args.max_learn,
        run_pip_flag=not args.no_pip,
        run_adaef_flag=not args.no_adaef,
    )


if __name__ == "__main__":
    main()
