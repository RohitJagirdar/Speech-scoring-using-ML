"""
Shared helpers for reading the dataset and the cached artifacts.

The one quirk worth knowing: 212 filenames (e.g. audio_12.wav) exist in BOTH
the train and test folders, but they are different recordings. So every clip
is identified by a key that includes its split, like "train/audio_12.wav".
Using the bare filename as a dictionary key would silently overwrite data.
"""
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AUDIO_DIRS,
    EMBEDDINGS_PATH,
    NLP_FEATURES_PATH,
    TEST_CSV,
    TRAIN_CSV,
    TRANSCRIPTS_PATH,
)

SPLIT_CSVS = {"train": TRAIN_CSV, "test": TEST_CSV}
ID_COLUMNS = ("unique_key", "filename")  # columns in nlp_features.csv that are not features


@dataclass(frozen=True)
class AudioItem:
    """One audio clip: where it lives and the key we store its features under."""

    split: str
    filename: str

    @property
    def key(self) -> str:
        return make_key(self.split, self.filename)

    @property
    def path(self) -> Path:
        return AUDIO_DIRS[self.split] / self.filename


def make_key(split: str, filename: str) -> str:
    return f"{split}/{filename}"


def load_labels(split: str) -> pd.DataFrame:
    """The split's CSV: columns `filename` and `label` (label is -1 for test)."""
    return pd.read_csv(SPLIT_CSVS[split])


def load_manifest(splits: tuple[str, ...] = ("train", "test")) -> list[AudioItem]:
    """Every clip we need to process, train first and then test."""
    return [AudioItem(split, name) for split in splits for name in load_labels(split)["filename"]]


def require_audio_files(items: list[AudioItem]) -> None:
    """Fail before a long GPU run starts, not 40 minutes into it."""
    missing = [str(item.path) for item in items if not item.path.exists()]
    if missing:
        preview = "\n  ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} audio files are missing, e.g.:\n  {preview}")


# --- Writing artifacts safely --------------------------------------------------

def save_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """
    Write to a temp file first, then swap it into place.

    If the process dies mid-write we keep the old file instead of a
    half-written one that would look "done" but be corrupt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp_" + path.name)  # keep the suffix (numpy cares about .npz)
    write(tmp)
    os.replace(tmp, path)


def save_json(path: Path, data: dict) -> None:
    save_atomic(path, lambda p: p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"))


# --- Reading cached artifacts --------------------------------------------------

def load_transcripts(path: Path = TRANSCRIPTS_PATH) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    data = np.load(path)
    return dict(zip(data["keys"], data["embeddings"], strict=True))  # strict: lengths must match


def build_feature_matrix(
    split: str,
    nlp_path: Path = NLP_FEATURES_PATH,
    emb_path: Path = EMBEDDINGS_PATH,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    """
    Join the text features and the audio embeddings for one split.

    Returns (X, y, filenames, feature_names). y is None for the test split.
    Raises if any clip is missing a feature row: training on a quietly
    shrunken dataset or predicting from zero-filled rows is worse than crashing.
    """
    labels = load_labels(split)
    keys = [make_key(split, name) for name in labels["filename"]]

    nlp = pd.read_csv(nlp_path).set_index("unique_key")
    nlp_cols = [c for c in nlp.columns if c not in ID_COLUMNS]
    embeddings = load_embeddings(emb_path)

    missing = [k for k in keys if k not in nlp.index or k not in embeddings]
    if missing:
        raise KeyError(f"{len(missing)} {split} clips have no features (e.g. {missing[:3]}). "
                       "Run `python main.py --status` to see which stage is incomplete.")

    X = np.hstack([
        nlp.loc[keys, nlp_cols].to_numpy(dtype=np.float32),
        np.stack([embeddings[k] for k in keys]).astype(np.float32),
    ])
    y = labels["label"].to_numpy(dtype=np.float32) if split == "train" else None
    emb_cols = [f"emb_{i}" for i in range(X.shape[1] - len(nlp_cols))]
    return X, y, labels["filename"].tolist(), nlp_cols + emb_cols
