"""
Stage 5: score the 216 test clips and write the submission CSV.
"""
import pickle

import pandas as pd

from src.config import MODEL_PATH, SUBMISSION_PATH
from src.data_utils import build_feature_matrix, save_atomic
from src.train import clip_scores


def load_model_bundle() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def generate_test_predictions() -> pd.DataFrame:
    """Stage entry point: scores for the test clips -> submission.csv."""
    bundle = load_model_bundle()
    X, _, filenames, feature_names = build_feature_matrix("test")

    # The model only makes sense if the columns arrive in the same order it was trained on.
    if feature_names != bundle["feature_names"]:
        raise ValueError("Test features don't match the trained model's features. Retrain with `python main.py`.")

    # The pipeline applies the scaler that was fit on train data, then the model.
    preds = clip_scores(bundle["pipeline"].predict(X))
    submission = pd.DataFrame({"filename": filenames, "label": preds})
    save_atomic(SUBMISSION_PATH, lambda p: submission.to_csv(p, index=False))

    print(f"[predict] {len(submission)} predictions from {bundle['model_name']} -> {SUBMISSION_PATH}")
    print(f"[predict] min {preds.min():.3f} | mean {preds.mean():.3f} | max {preds.max():.3f}")
    return submission


if __name__ == "__main__":
    generate_test_predictions()
