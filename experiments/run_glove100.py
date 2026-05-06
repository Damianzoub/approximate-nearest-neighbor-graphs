"""
Experiment script for GloVe-100 (200K base, 10K queries, 100-dim, L2).
Vectors are L2-normalised GloVe embeddings (cosine via L2 on unit sphere).

Results  → results_csv/glove100/all_results.csv
Failures → results_csv/glove100/failure_analysis.csv

Usage
-----
    python experiments/run_glove100.py
    python experiments/run_glove100.py --no-pip --no-adaef
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments._core import run_dataset_experiment

PATHS = {
    "base":  str(ROOT / "Datasets/glove100ann/glove100_base.fvecs"),
    "learn": str(ROOT / "Datasets/glove100ann/glove100_learn.fvecs"),
    "query": str(ROOT / "Datasets/glove100ann/glove100_query.fvecs"),
    "gt":    str(ROOT / "Datasets/glove100ann/glove100_groundtruth.ivecs"),
}

M_VALUES   = [16]
EFC_VALUES = [200]


def main():
    parser = argparse.ArgumentParser(description="Run experiments on GloVe-100.")
    parser.add_argument("--no-pip",    action="store_true", help="Skip PiP experiments")
    parser.add_argument("--no-adaef", action="store_true", help="Skip Ada-ef experiments")
    parser.add_argument("--max-learn", type=int, default=5000,
                        help="Max learn vectors for predictor training (default: 5000)")
    args = parser.parse_args()

    run_dataset_experiment(
        dataset_name="glove100",
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
