"""
The pipeline as a list of stages, each with an explicit status.

Before this file existed, main.py decided whether to run a step with
`force or not path.exists()`. That hid two states:
  * an interrupted transcription leaves a *partial* transcripts.json, which
    "exists", so the step was skipped and ~half the clips got empty features;
  * re-running an early step didn't re-run the later ones, so you could
    train on features computed from old transcripts.

Now every stage reports one of four states, and anything not READY gets run:

    MISSING     - output file isn't there yet
    INCOMPLETE  - output exists but doesn't cover every clip (interrupted run)
    STALE       - an input file is newer than the output (upstream re-ran)
    READY       - nothing to do

Running a stage moves it to READY and makes its outputs newer than the next
stage's, so that stage becomes STALE and runs too. Same idea as `make`.
"""
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_utils import load_manifest


class Status(Enum):
    MISSING = "missing"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    READY = "ready"


@dataclass
class Stage:
    name: str
    description: str
    entry_point: str  # "module:function", imported only when the stage runs
    outputs: list[Path]
    inputs: list[Path] = field(default_factory=list)
    covers_all_clips: Callable[[], bool] | None = None  # None = no per-clip check

    def status(self) -> Status:
        if not all(p.exists() for p in self.outputs):
            return Status.MISSING
        if self.covers_all_clips and not self.covers_all_clips():
            return Status.INCOMPLETE
        oldest_output = min(p.stat().st_mtime for p in self.outputs)
        if any(p.exists() and p.stat().st_mtime > oldest_output for p in self.inputs):
            return Status.STALE
        return Status.READY

    def run(self) -> None:
        # Lazy import: `--status` shouldn't have to load torch and Whisper.
        module_name, func_name = self.entry_point.split(":")
        getattr(importlib.import_module(module_name), func_name)()


# --- "Does this artifact cover every clip?" checks ---------------------------

def _all_keys() -> set:
    return {item.key for item in load_manifest()}


def _transcripts_complete() -> bool:
    return _all_keys() <= json.loads(config.TRANSCRIPTS_PATH.read_text(encoding="utf-8")).keys()


def _nlp_features_complete() -> bool:
    return _all_keys() <= set(pd.read_csv(config.NLP_FEATURES_PATH, usecols=["unique_key"])["unique_key"])


def _embeddings_complete() -> bool:
    return _all_keys() <= set(np.load(config.EMBEDDINGS_PATH)["keys"])


SRC = Path(__file__).resolve().parent

# The cheap stages list their own source file as an input, so editing a regex
# or a hyperparameter marks them STALE. Transcribe and embed deliberately don't:
# they depend only on the audio (the coverage check notices new clips), and a
# touched file should never trigger a 30-minute GPU re-run by accident.
STAGES = [
    Stage("transcribe", "Speech to text with Whisper", "src.transcribe:transcribe_audio_corpus",
          outputs=[config.TRANSCRIPTS_PATH], covers_all_clips=_transcripts_complete),
    Stage("nlp_features", "20 grammar and fluency features from the text", "src.nlp_features:compute_nlp_feature_table",
          outputs=[config.NLP_FEATURES_PATH], inputs=[config.TRANSCRIPTS_PATH, SRC / "nlp_features.py"],
          covers_all_clips=_nlp_features_complete),
    Stage("embeddings", "768-dim wav2vec2 vector per clip", "src.embeddings:compute_audio_embeddings",
          outputs=[config.EMBEDDINGS_PATH], covers_all_clips=_embeddings_complete),
    Stage("train", "Cross-validate 3 models, refit the best", "src.train:run_cross_validation_and_train",
          outputs=[config.MODEL_PATH, config.CV_RESULTS_PATH, config.FEATURE_IMPORTANCE_PATH],
          inputs=[config.NLP_FEATURES_PATH, config.EMBEDDINGS_PATH, config.TRAIN_CSV,
                  SRC / "train.py", SRC / "config.py"]),
    Stage("predict", "Score the test clips", "src.predict:generate_test_predictions",
          outputs=[config.SUBMISSION_PATH],
          inputs=[config.MODEL_PATH, config.NLP_FEATURES_PATH, config.EMBEDDINGS_PATH, config.TEST_CSV]),
]
STAGE_NAMES = [s.name for s in STAGES]


def print_status() -> None:
    print(f"{'stage':<14}{'status':<12}what it does")
    for stage in STAGES:
        print(f"{stage.name:<14}{stage.status().value:<12}{stage.description}")


def run_pipeline(force: bool = False, only: str | None = None) -> None:
    """Run every stage that isn't READY (or all of them with force, or just `only`)."""
    for i, stage in enumerate(STAGES, start=1):
        if only and stage.name != only:
            continue
        status = stage.status()
        header = f"\n[{i}/{len(STAGES)}] {stage.name} ({status.value})"
        if status is Status.READY and not (force or only):
            print(f"{header} - up to date, skipping")
            continue
        print(f"{header} - {stage.description}")
        stage.run()
