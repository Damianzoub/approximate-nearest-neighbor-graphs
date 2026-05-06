"""
Experiment script for MS MARCO (40K base, 5K queries, 384-dim, L2).
Vectors are sentence-transformer embeddings (all-MiniLM-L6-v2).

Results  → results_csv/msmarco_text/all_results.csv
Failures → results_csv/msmarco_text/failure_analysis.csv

Usage
-----
    python experiments/run_msmarco.py
    python experiments/run_msmarco.py --no-pip --no-adaef
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments._core import run_dataset_experiment

PATHS = {
    "base":  str(ROOT / "Datasets/msmarco_text/msmarco_base.fvecs"),
    "learn": str(ROOT / "Datasets/msmarco_text/msmarco_learn.fvecs"),
    "query": str(ROOT / "Datasets/msmarco_text/msmarco_query.fvecs"),
    "gt":    str(ROOT / "Datasets/msmarco_text/msmarco_groundtruth.ivecs"),
}

M_VALUES   = [16]
EFC_VALUES = [200]


def main():
    parser = argparse.ArgumentParser(description="Run experiments on MS MARCO.")
    parser.add_argument("--no-pip",   action="store_true", help="Skip PiP experiments")
    parser.add_argument("--no-adaef", action="store_true", help="Skip Ada-ef experiments")
    parser.add_argument("--max-learn", type=int, default=3000,
                        help="Max learn vectors for predictor training (default: 3000)")
    args = parser.parse_args()

    run_dataset_experiment(
        dataset_name="msmarco_text",
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
