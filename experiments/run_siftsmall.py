"""
Experiment script for SIFTsmall (10K base, 100 queries, 128-dim, L2).

Results  → results_csv/siftsmall/all_results.csv
Failures → results_csv/siftsmall/failure_analysis.csv

Usage
-----
    python experiments/run_siftsmall.py
    python experiments/run_siftsmall.py --no-pip --no-adaef   # DARTH only
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments._core import run_dataset_experiment

PATHS = {
    "base":  str(ROOT / "Datasets/siftsmall/siftsmall_base.fvecs"),
    "learn": str(ROOT / "Datasets/siftsmall/siftsmall_learn.fvecs"),
    "query": str(ROOT / "Datasets/siftsmall/siftsmall_query.fvecs"),
    "gt":    str(ROOT / "Datasets/siftsmall/siftsmall_groundtruth.ivecs"),
}

# Siftsmall is small — sweep multiple (M, efC) combos without too much wait
M_VALUES   = [8, 16]
EFC_VALUES = [128, 200]


def main():
    parser = argparse.ArgumentParser(description="Run experiments on SIFTsmall.")
    parser.add_argument("--no-pip",   action="store_true", help="Skip PiP experiments")
    parser.add_argument("--no-adaef", action="store_true", help="Skip Ada-ef experiments")
    parser.add_argument("--max-learn", type=int, default=5000,
                        help="Max learn vectors for predictor training (default: 5000)")
    args = parser.parse_args()

    run_dataset_experiment(
        dataset_name="siftsmall",
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
