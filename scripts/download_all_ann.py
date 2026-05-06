"""
Download all ANN-benchmark datasets used in the thesis.

Approximate sizes (check disk space first!):
  sift-128-euclidean.hdf5    ~470 MB   (1M vectors, 128-dim)
  glove-100-angular.hdf5    ~462 MB   (1.2M vectors, 100-dim, cosine)
  gist-960-euclidean.hdf5   ~3.7 GB   (1M vectors, 960-dim)  ← needs space
  glove.6B.zip              ~860 MB   zip  (~1.5 GB extracted)

Usage
-----
    python scripts/download_all_ann.py                   # all datasets
    python scripts/download_all_ann.py --skip gist960 glove6b   # skip large ones
"""

import argparse
import sys
from pathlib import Path

# ── Make individual scripts importable ────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import download_gist960ann
import download_sift128ann
import download_glove100ann
import download_glove6b


REGISTRY = {
    "sift128":  download_sift128ann,
    "glove100": download_glove100ann,
    "gist960":  download_gist960ann,
    "glove6b":  download_glove6b,
}

SIZES = {
    "sift128":  "~470 MB",
    "glove100": "~462 MB",
    "gist960":  "~3.7 GB  ← large, check disk space first",
    "glove6b":  "~860 MB zip / ~1.5 GB extracted",
}


def main():
    parser = argparse.ArgumentParser(description="Download ANN benchmark datasets.")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=list(REGISTRY),
        metavar="DATASET",
        help="Datasets to skip, e.g. --skip gist960 glove6b",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        choices=list(REGISTRY),
        metavar="DATASET",
        help="Download only these datasets",
    )
    args = parser.parse_args()

    targets = list(REGISTRY) if args.only is None else args.only
    targets = [t for t in targets if t not in args.skip]

    print("=== ANN Dataset Downloader ===")
    print(f"  Will download: {targets}\n")
    for name in targets:
        print(f"  {name:<10}  {SIZES[name]}")
    print()

    for name in targets:
        print(f"\n{'─'*50}")
        print(f"  Dataset: {name}")
        print(f"{'─'*50}")
        REGISTRY[name].main()

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
