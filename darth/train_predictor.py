"""
Train a LightGBM recall predictor from a DARTH training CSV.

The CSV must have been produced by darth/collect_training_data.py.
The trained model is saved as a LightGBM text file (.txt) which can be:
  - loaded by LGBMPredictor (Python)
  - loaded by the official DARTH C++ code via LGBM_BoosterCreateFromModelfile

Usage
-----
    python darth/train_predictor.py \\
        --input  predictor_models/train_data.csv \\
        --output predictor_models/siftsmall_M16_k10.txt

Optional flags:
    --n-estimators   100       number of GBDT trees (default: 100)
    --val-fraction   0.1       fraction held out for validation MAE (0 = no split)
    --seed           42
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from darth.predictor import FEATURE_COLUMNS


def train(
    input_csv: str,
    output_model: str,
    n_estimators: int = 100,
    val_fraction: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """
    Load a training CSV, fit LGBMRegressor, save the model as a .txt file.

    Parameters
    ----------
    input_csv : str
        Path to CSV produced by collect_training_data.py.
    output_model : str
        Output path for the LightGBM text model (.txt).
    n_estimators : int
        Number of boosting rounds (trees). Paper uses 100.
    val_fraction : float
        Fraction of queries held out for MAE reporting (0 = no hold-out).
    seed : int
        Random seed.
    verbose : bool
        Print progress and metrics.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "lightgbm is required. Install with: pip install lightgbm"
        ) from e

    if verbose:
        print(f"Loading training data from {input_csv} …")

    df = pd.read_csv(input_csv)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}. "
            f"Expected columns from collect_training_data.py: {FEATURE_COLUMNS}"
        )
    if "r" not in df.columns:
        raise ValueError("CSV must have a 'r' column (recall@k target).")

    if verbose:
        nq = df["qid"].nunique() if "qid" in df.columns else "?"
        print(f"  {len(df)} rows, {nq} unique queries")

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df["r"].values.astype(np.float64)

    X_train, y_train = X, y
    X_val, y_val = None, None

    if val_fraction > 0.0:
        # split by query id so no query leaks across train/val
        if "qid" in df.columns:
            qids = df["qid"].values
            unique_qids = np.unique(qids)
            rng = np.random.default_rng(seed)
            rng.shuffle(unique_qids)
            n_val_q = max(1, int(len(unique_qids) * val_fraction))
            val_qids = set(unique_qids[:n_val_q])
            val_mask = np.array([q in val_qids for q in qids])
            X_train, y_train = X[~val_mask], y[~val_mask]
            X_val,   y_val   = X[val_mask],  y[val_mask]
        else:
            n_val = max(1, int(len(X) * val_fraction))
            X_train, y_train = X[n_val:], y[n_val:]
            X_val,   y_val   = X[:n_val], y[:n_val]

        if verbose:
            print(f"  Train rows: {len(X_train)}, Val rows: {len(X_val)}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        random_state=seed,
        verbose=-1,
    )

    if verbose:
        print(f"Training LGBMRegressor (n_estimators={n_estimators}) …")

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    if verbose:
        print(f"  Training done in {elapsed:.1f}s")

    if X_val is not None and verbose:
        y_pred = model.predict(X_val)
        mae = float(np.mean(np.abs(y_pred - y_val)))
        rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
        print(f"  Val MAE:  {mae:.4f}")
        print(f"  Val RMSE: {rmse:.4f}")

    if verbose:
        importances = sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: -x[1],
        )
        print("  Feature importances:")
        for feat, imp in importances:
            print(f"    {feat:<14} {imp}")

    os.makedirs(os.path.dirname(os.path.abspath(output_model)), exist_ok=True)
    model.booster_.save_model(output_model)

    if verbose:
        print(f"Model saved to {output_model}")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM DARTH recall predictor."
    )
    parser.add_argument("--input",   required=True, help="Training CSV from collect_training_data.py")
    parser.add_argument("--output",  required=True, help="Output .txt model path")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    train(
        input_csv=args.input,
        output_model=args.output,
        n_estimators=args.n_estimators,
        val_fraction=args.val_fraction,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    _main()
