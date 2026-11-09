"""
How much does each half of the model contribute?

Runs the same cross-validation as training, but on text-only, audio-only and
both feature sets, using Ridge (the model that wins on the full set).

    python -m src.ablation
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import RIDGE_ALPHA
from src.data_utils import build_feature_matrix
from src.train import cross_validate


def run_ablation() -> pd.DataFrame:
    """CV score of Ridge on text-only, audio-only and all features."""
    X, y, _, names = build_feature_matrix("train")
    is_audio = np.array([n.startswith("emb_") for n in names])
    feature_sets = {
        "text only (20)": ~is_audio,
        "audio only (768)": is_audio,
        "text + audio (788)": np.ones_like(is_audio),
    }
    rows = []
    for label, columns in feature_sets.items():
        result = cross_validate(X[:, columns], y, {"Ridge": Ridge(alpha=RIDGE_ALPHA)})
        rows.append({"features": label, **result.drop(columns="model").iloc[0].to_dict()})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run_ablation()
