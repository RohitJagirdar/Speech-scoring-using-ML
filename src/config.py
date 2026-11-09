"""
Every path and knob the pipeline uses, in one place.

If you want to try a different Whisper size or a stronger Ridge penalty,
this is the only file you should need to touch.
"""
from pathlib import Path

# --- Where things live -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Dataset_Final"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
AUDIO_DIRS = {"train": DATA_DIR / "train", "test": DATA_DIR / "test"}

# Everything the pipeline produces goes here, so deleting this folder = clean slate.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRANSCRIPTS_PATH = ARTIFACTS_DIR / "transcripts.json"
NLP_FEATURES_PATH = ARTIFACTS_DIR / "nlp_features.csv"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npz"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"  # scaler + regressor saved together
CV_RESULTS_PATH = ARTIFACTS_DIR / "cv_results.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
SUBMISSION_PATH = ARTIFACTS_DIR / "submission.csv"

# --- Pretrained models -------------------------------------------------------
WHISPER_MODEL_NAME = "base"  # "small"/"medium" are more accurate but slower
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
AUDIO_SAMPLE_RATE = 16_000  # both Whisper and wav2vec2 expect 16 kHz mono
EMBEDDING_DIM = 768  # hidden size of wav2vec2-base

# --- Training ----------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42
RIDGE_ALPHA = 100.0  # strong penalty: 788 features vs only 769 training clips
LGBM_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    colsample_bytree=0.8,  # each tree sees 80% of the features
    random_state=RANDOM_STATE,
    verbose=-1,
)

# --- Score range of the grading rubric ---------------------------------------
MIN_SCORE = 0.0
MAX_SCORE = 5.0
