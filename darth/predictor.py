"""
LGBMPredictor — wraps a trained LightGBM recall-prediction model.

Compatible with two calling conventions:
  - C++ path (bindings.cpp PyPredictor): keys are C++ field names
    {'ndis', 'nstep', 'ninserts', 'firstNN', 'closestNN', 'furthestNN',
     'meanNN', 'varNN', 'p25NN', 'p50NN', 'p75NN'}
  - Python path (hsnw_constructionDARTH.py darth_extract_features): keys are
    {'ndis', 'inserts', 'nstep', 'firstNN', 'closestNN', 'furthestNN',
     'avg', 'var', 'med', 'perc25', 'perc75'}

The model is a LightGBM text file (.txt) saved with
  model.booster_.save_model(path)
and can also be loaded by the official repo's C++ code via
  LGBM_BoosterCreateFromModelfile(path, &iters, &booster)
"""

import numpy as np

# Canonical feature order — must match the column order used in train_predictor.py
FEATURE_COLUMNS = [
    "nstep",
    "ndis",
    "ninserts",
    "firstNN",
    "closestNN",
    "furthestNN",
    "meanNN",
    "varNN",
    "p25NN",
    "p50NN",
    "p75NN",
]

# Maps canonical name → all accepted aliases (C++ keys and Python keys)
_ALIASES: dict[str, list[str]] = {
    "nstep":      ["nstep", "step"],
    "ndis":       ["ndis", "dists"],
    "ninserts":   ["ninserts", "inserts"],
    "firstNN":    ["firstNN", "first_nn_dist"],
    "closestNN":  ["closestNN", "nn_dist"],
    "furthestNN": ["furthestNN", "furthest_dist"],
    "meanNN":     ["meanNN", "avg", "avg_dist"],
    "varNN":      ["varNN", "var", "variance"],
    "p25NN":      ["p25NN", "perc25", "percentile_25"],
    "p50NN":      ["p50NN", "med", "percentile_50"],
    "p75NN":      ["p75NN", "perc75", "percentile_75"],
}

# Pre-built lookup: any key name → canonical name
_KEY_TO_CANONICAL: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _ALIASES.items()
    for alias in aliases
}


class LGBMPredictor:
    """
    Loads a LightGBM recall-predictor model and exposes a predict() callable
    compatible with both the pure-Python HNSW_DARTH and the C++ pybind11
    HNSWDarthIndex.

    Parameters
    ----------
    model_path : str
        Path to a LightGBM text model file (.txt) produced by
        model.booster_.save_model(model_path).

    Example
    -------
    >>> predictor = LGBMPredictor("predictor_models/siftsmall_M16_k10.txt")
    >>> # pure-Python path
    >>> ids = hnsw.query_darth(q, k=10, efSearch=200, Rt=0.90, predictor=predictor)
    >>> # C++ pybind11 path
    >>> D, I = index.search_darth(xq, k=10, efSearch=200, Rt=0.90, predictor=predictor)
    """

    def __init__(self, model_path: str) -> None:
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "lightgbm is required for LGBMPredictor. "
                "Install it with: pip install lightgbm"
            ) from e

        self.model_path = model_path
        self._model = lgb.Booster(model_file=model_path)
        self._buf = np.empty((1, len(FEATURE_COLUMNS)), dtype=np.float64)

    def predict(self, feats: dict) -> float:
        """
        Predict recall@k given the current search-state feature dict.

        Parameters
        ----------
        feats : dict
            Dict with at least the 11 search-state features.
            Accepts either C++ field names (meanNN, varNN, …) or Python
            aliases (avg, var, …) — see FEATURE_COLUMNS / _ALIASES above.

        Returns
        -------
        float
            Predicted recall in [0, 1].
        """
        for col_idx, col in enumerate(FEATURE_COLUMNS):
            val = None
            for alias in _ALIASES[col]:
                if alias in feats:
                    val = feats[alias]
                    break
            if val is None:
                raise KeyError(
                    f"Feature '{col}' not found in feats dict. "
                    f"Accepted aliases: {_ALIASES[col]}. Got keys: {list(feats.keys())}"
                )
            self._buf[0, col_idx] = float(val)

        out = self._model.predict(self._buf)
        return float(np.clip(out[0], 0.0, 1.0))

    def __repr__(self) -> str:
        return f"LGBMPredictor(model_path={self.model_path!r})"
