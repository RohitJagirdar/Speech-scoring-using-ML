"""
Stage 4: compare three regressors with 5-fold cross-validation, then refit
the winner on all 769 training clips.

Candidates:
  Ridge           - linear, heavily regularised; strong when features > samples
  LightGBM        - gradient-boosted trees; picks up non-linear interactions
  Ridge+LightGBM  - plain average of the two; their mistakes only partly overlap

Every candidate is wrapped as StandardScaler -> model in one sklearn Pipeline,
so the scaler is refit inside each fold (no leakage) and saved with the model.
"""
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_RESULTS_PATH,
    FEATURE_IMPORTANCE_PATH,
    LGBM_PARAMS,
    MAX_SCORE,
    MIN_SCORE,
    MODEL_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RIDGE_ALPHA,
)
from src.data_utils import build_feature_matrix, save_atomic


def make_candidates() -> dict:
    """The three models we compare. Fresh, unfitted objects every call."""
    return {
        "Ridge": Ridge(alpha=RIDGE_ALPHA),
        "LightGBM": LGBMRegressor(**LGBM_PARAMS),
        "Ridge+LightGBM": VotingRegressor([
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
            ("lightgbm", LGBMRegressor(**LGBM_PARAMS)),
        ]),
    }


def with_scaler(model) -> Pipeline:
    return make_pipeline(StandardScaler(), model)


def clip_scores(preds: np.ndarray) -> np.ndarray:
    """The rubric runs 0-5, so anything outside that is guaranteed to be wrong."""
    return np.clip(preds, MIN_SCORE, MAX_SCORE)


def stratify_bins(y: np.ndarray) -> np.ndarray:
    """
    Group scores into bands so every fold gets a similar score distribution.

    Bands are the whole-number part of the score (0, 2, 3, 4, 5). Scores 1.0-1.5
    are folded into band 0 because there are only 4 of them, and
    StratifiedKFold needs at least N_SPLITS members per band.
    """
    bands = np.floor(y).astype(int)
    bands[bands == 1] = 0
    return bands


def cross_validate(X: np.ndarray, y: np.ndarray, candidates: dict) -> pd.DataFrame:
    """Mean and spread of Pearson r and RMSE across folds, best model first."""
    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, stratify_bins(y)))
    rows = []
    for name, model in candidates.items():
        pearson, rmse = [], []
        for train_idx, val_idx in folds:
            # clone() gives a fresh, unfitted copy, so no fold can leak into the next.
            fitted = with_scaler(clone(model)).fit(X[train_idx], y[train_idx])
            preds = clip_scores(fitted.predict(X[val_idx]))
            pearson.append(pearsonr(y[val_idx], preds)[0])
            rmse.append(root_mean_squared_error(y[val_idx], preds))
        rows.append({
            "model": name,
            "mean_pearson": round(float(np.mean(pearson)), 4),
            "std_pearson": round(float(np.std(pearson)), 4),
            "mean_rmse": round(float(np.mean(rmse)), 4),
        })
    return pd.DataFrame(rows).sort_values("mean_pearson", ascending=False, ignore_index=True)


def feature_importance(pipeline: Pipeline, feature_names: list) -> pd.DataFrame:
    """
    One importance score per feature, whatever model won.

    Ridge: size of each coefficient (fair to compare because inputs are scaled).
    LightGBM: how often a feature is used to split. For the average model we
    normalise both to sum to 1 and average them.
    """
    model = pipeline[-1]
    parts = list(model.named_estimators_.values()) if isinstance(model, VotingRegressor) else [model]
    scores = []
    for part in parts:
        raw = np.abs(part.coef_) if hasattr(part, "coef_") else part.feature_importances_.astype(float)
        scores.append(raw / raw.sum())
    df = pd.DataFrame({"feature": feature_names, "importance": np.mean(scores, axis=0)})
    return df.sort_values("importance", ascending=False, ignore_index=True)


def run_cross_validation_and_train() -> pd.DataFrame:
    """Stage entry point: compare candidates with CV, refit and save the winner."""
    X, y, _, feature_names = build_feature_matrix("train")
    print(f"[train] {X.shape[0]} clips x {X.shape[1]} features")

    candidates = make_candidates()
    cv = cross_validate(X, y, candidates)
    print(cv.to_string(index=False))

    best = cv.loc[0, "model"]
    final = with_scaler(clone(candidates[best])).fit(X, y)
    print(f"[train] Winner: {best}. Refit on all {len(y)} training clips.")

    bundle = {"pipeline": final, "model_name": best, "feature_names": feature_names}
    save_atomic(MODEL_PATH, lambda p: p.write_bytes(pickle.dumps(bundle)))
    save_atomic(CV_RESULTS_PATH, lambda p: cv.to_csv(p, index=False))
    importance = feature_importance(final, feature_names)
    save_atomic(FEATURE_IMPORTANCE_PATH, lambda p: importance.to_csv(p, index=False))
    return cv


if __name__ == "__main__":
    run_cross_validation_and_train()
