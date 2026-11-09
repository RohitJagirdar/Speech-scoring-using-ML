# Spoken Grammar Scoring


This project predicts a **0–5 grammar score** for a recording of someone speaking English. It uses two signals: **what** the speaker said (a Whisper transcript turned into text features) and **how** they said it (wav2vec2 audio embeddings).

**Result:** Ridge regression on 788 features, with 5-fold cross-validation over 769 training clips, scores **Pearson r = 0.857 ± 0.022** and **RMSE = 0.638**.

## Contents

- [Overview](#overview)
- [Results](#results)
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Pipeline stages and status](#pipeline-stages-and-status)
- [Verification checks](#verification-checks)
- [Configuration](#configuration)
- [Testing and CI](#testing-and-ci)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Things worth knowing about the data](#things-worth-knowing-about-the-data)
- [Limitations and future work](#limitations-and-future-work)

## Overview

| | |
|---|---|
| **Input** | A `.wav` recording of spoken English (16 kHz mono after loading) |
| **Output** | A score from 0 to 5, matching a human grading rubric |
| **Data** | 769 labelled training clips and 216 unlabelled test clips |
| **Metric** | Pearson correlation with the human scores (RMSE is also reported) |
| **Approach** | Pretrained speech models extract features, then a small regression model learns the score |

Training a speech model from scratch on 769 clips isn't realistic. Instead, two frozen pretrained models turn each clip into numbers, and a heavily regularised linear model learns the score from those numbers.

## Results

Cross-validated scores of the three candidate models (5 stratified folds, seed 42):

| Model | Pearson r | RMSE |
|---|---|---|
| **Ridge (α = 100)** | **0.857 ± 0.022** | **0.638** |
| Ridge + LightGBM (average) | 0.856 ± 0.020 | 0.642 |
| LightGBM | 0.828 ± 0.021 | 0.695 |

Ridge wins. It is also the simplest of the three, so it is the model that gets refit on all the training data and saved.

**Ablation:** which features carry the signal? (Ridge, same folds; run with `python -m src.ablation`)

| Features | Pearson r | RMSE |
|---|---|---|
| text only (20) | 0.613 | 0.984 |
| audio only (768) | 0.832 | 0.689 |
| text + audio (788) | 0.857 | 0.638 |

Audio does most of the work, and text adds a small but consistent gain. The grammar regexes almost never fire, because Whisper's decoder tends to "correct" grammar while it transcribes. The text features that matter are vocabulary size, length and readability. The full ranking is in [artifacts/feature_importance.csv](artifacts/feature_importance.csv).

## How it works

```
 .wav ──► Whisper ──► transcript ──► 20 text features ──┐
   │                                                     ├──► 788 features ──► StandardScaler ──► Ridge ──► clip to [0, 5]
   └────► wav2vec2 ──► mean over time ──► 768 numbers ───┘
```

### Text features (20)

Whisper `base` transcribes each clip. Each transcript then becomes:

- **Length and structure:** word count, sentence count, average sentence length, character count
- **Vocabulary:** unique word count, type-token ratio (unique words ÷ total words)
- **Fluency:** filler word count and rate (`um`, `uh`, `like`, …)
- **Spelling:** misspelled word count and rate (pyspellchecker)
- **Grammar:** 7 regex rules (double negatives, `they was`, `he do`, `more better`, `goed`, …), one count per rule plus a total
- **Readability:** Flesch reading ease and Flesch–Kincaid grade (textstat)

### Audio features (768)

The frozen [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base) model turns a clip into one 768-number vector for every ~20 ms of audio. Averaging those vectors over time gives a single fixed-size vector per clip. It captures pronunciation, rhythm and hesitation, which a transcript loses.

### Model

Each candidate is an sklearn `Pipeline(StandardScaler → model)`. Because the scaler is fit inside each cross-validation fold, no statistics leak from the validation fold, and the scaler is saved together with the model. Predictions are clipped to the rubric range [0, 5].

## Getting started

### Prerequisites

- **Python 3.10+**
- **CUDA GPU (optional):** makes Whisper and wav2vec2 roughly 10× faster. Everything also runs on CPU.

Tested with Python 3.14, torch 2.11, transformers 5.17, scikit-learn 1.9, LightGBM 4.7 and pandas 3.0.

### Installation

```bash
git clone https://github.com/RohitJagirdar/Speech-scoring-using-ML.git
cd Speech-scoring-using-ML

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

For a GPU build of torch, install it first from [pytorch.org](https://pytorch.org/get-started/locally/), then run the command above.

### Data

The dataset is not in the repo. Put it in the project root like this:

```
Dataset_Final/
├── train.csv              # filename,label   (label = 0–5 score)
├── test.csv               # filename,label   (label = -1, unknown)
├── sample_submission.csv
├── train/*.wav            # 769 clips
└── test/*.wav             # 216 clips
```

The pipeline checks that every file listed in the CSVs exists before it starts, and stops with a clear message if any are missing.

## Usage

```bash
python main.py --status         # show what's done and what's stale; runs nothing
python main.py                  # run only what's out of date, then verify
python main.py --stage train    # re-run one stage, then verify
python main.py --force          # re-run everything (Whisper + wav2vec2 are the slow part)
python -m src.ablation          # text vs audio comparison
```

Example `--status` output once everything has run:

```
stage         status      what it does
transcribe    ready       Speech to text with Whisper
nlp_features  ready       20 grammar and fluency features from the text
embeddings    ready       768-dim wav2vec2 vector per clip
train         ready       Cross-validate 3 models, refit the best
predict       ready       Score the test clips
```

`main.py` exits with code `0` if every verification check passes and `1` if any fails, so it can be used in scripts.

### Outputs

Everything the pipeline produces goes to `artifacts/`. Deleting that folder gives you a clean slate.

| File | Produced by | Committed? |
|---|---|---|
| `transcripts.json` | transcribe | no (regenerated) |
| `nlp_features.csv` | nlp_features | no |
| `embeddings.npz` | embeddings | no (2.8 MB) |
| `model.pkl` | train | no |
| `cv_results.csv` | train | yes |
| `feature_importance.csv` | train | yes |
| `submission.csv` | predict | yes |

## Pipeline stages and status

| Stage | What it does | Reads | Writes |
|---|---|---|---|
| `transcribe` | Whisper `base`, resumable (checkpoints every 20 clips) | audio | `transcripts.json` |
| `nlp_features` | the 20 text features above | transcripts | `nlp_features.csv` |
| `embeddings` | wav2vec2 mean-pooled vector per clip | audio | `embeddings.npz` |
| `train` | 5-fold CV of 3 models; refits the best on all data | features, embeddings | `model.pkl`, `cv_results.csv`, `feature_importance.csv` |
| `predict` | scores the 216 test clips | model, features, embeddings | `submission.csv` |

Every stage is in exactly one of four states:

| Status | Meaning | What `main.py` does |
|---|---|---|
| **missing** | output doesn't exist | runs it |
| **incomplete** | output exists but doesn't cover every clip (e.g. transcription was interrupted) | runs it, picking up where it stopped |
| **stale** | an input file, or the stage's own source code, changed after the output was written | runs it |
| **ready** | output is complete and newer than everything it depends on | skips it |

Staleness works like `make`: it compares file modification times. `transcribe` and `embeddings` read raw audio, which never changes, so for them the only question is whether every clip is covered. Re-running one stage therefore makes the stages after it stale, and the next `python main.py` brings them up to date. The logic is in [src/pipeline.py](src/pipeline.py).

Outputs are written to a temporary file and then renamed into place. A crash mid-write never leaves a half-written artifact that looks finished.

## Verification checks

After every run, [src/verify.py](src/verify.py) checks the artifacts and prints **PASS**, **WARN** or **FAIL** for each:

| Check | What it confirms |
|---|---|
| audio files | every clip listed in the CSVs exists; reports how many names appear in both splits |
| transcripts | one transcript per clip; warns about near-empty ones |
| nlp features | no missing values |
| embeddings | no NaNs; warns about all-zero vectors |
| no test leakage | the saved scaler saw exactly as many rows as the training set, no test rows |
| submission | columns match `sample_submission.csv`, filenames match `test.csv`, all scores within [0, 5] |

On the current data, all checks pass, with one WARN: 6 clips transcribe to (almost) nothing.

## Configuration

All paths and hyperparameters live in [src/config.py](src/config.py):

| Setting | Default | Notes |
|---|---|---|
| `WHISPER_MODEL_NAME` | `"base"` | `"small"` / `"medium"` are more accurate but slower |
| `WAV2VEC2_MODEL_NAME` | `"facebook/wav2vec2-base"` | change `EMBEDDING_DIM` to match |
| `N_SPLITS` | `5` | cross-validation folds |
| `RANDOM_STATE` | `42` | seeds the folds and LightGBM |
| `RIDGE_ALPHA` | `100.0` | strong penalty: 788 features vs 769 clips |
| `LGBM_PARAMS` | 100 trees, lr 0.05, depth 4 | |

Changing `config.py` marks the `train` stage as stale, so a new `RIDGE_ALPHA` takes effect on the next run. For changes to the Whisper or wav2vec2 settings, re-run that stage with `--stage transcribe` / `--stage embeddings` (they are slow, so they never re-run on their own).

## Testing and CI

```bash
pytest            # 9 unit tests + 2 artifact tests
ruff check .      # lint (settings in pyproject.toml)
```

- **Unit tests** ([tests/test_units.py](tests/test_units.py)) cover clip keys, the grammar rules (on both wrong and correct English), empty transcripts, score clipping, fold stratification, the stage state machine and atomic saves. They need no GPU, audio or model.
- **Artifact tests** ([tests/test_artifacts.py](tests/test_artifacts.py)) check dataset sizes and that every verification check passes. They skip themselves until the pipeline has been run.

[GitHub Actions](.github/workflows/ci.yml) runs `ruff check .` and `pytest` on every push and pull request. Since the unit tests don't need them, CI doesn't install torch or Whisper, which keeps it fast.

## Project structure

```
.
├── main.py                  CLI entry point
├── src/
│   ├── config.py            paths and hyperparameters
│   ├── data_utils.py        manifest, split-prefixed keys, feature-matrix join, atomic saves
│   ├── pipeline.py          stages and their status (the state machine)
│   ├── transcribe.py        stage 1: Whisper
│   ├── nlp_features.py      stage 2: text features
│   ├── embeddings.py        stage 3: wav2vec2
│   ├── train.py             stage 4: cross-validation + final model
│   ├── predict.py           stage 5: test predictions
│   ├── verify.py            PASS / WARN / FAIL checks
│   └── ablation.py          text vs audio experiment
├── tests/
│   ├── test_units.py
│   └── test_artifacts.py
├── artifacts/               pipeline outputs (only the small result tables are committed)
├── Dataset_Final/           dataset (not committed)
├── .github/workflows/ci.yml
├── pyproject.toml           project metadata, pytest + ruff settings
└── requirements.txt
```

## Design decisions

- **Frozen pretrained models, not fine-tuning.** With 769 clips, fine-tuning a 95M-parameter model would overfit. Frozen features plus a linear model are cheap, stable and easy to inspect.
- **Ridge over boosted trees.** With 788 features and 769 samples, a strongly regularised linear model generalises best. LightGBM was tried and lost, and averaging the two didn't beat Ridge alone.
- **Stratified folds.** Scores are uneven (only 4 clips between 1.0 and 1.5), so folds are stratified by score band. Every fold then sees a similar spread of scores.
- **Scaler inside the pipeline.** Fitting `StandardScaler` per fold avoids leaking validation statistics, and saving it with the model means predictions use exactly the same scaling.
- **Stages with explicit status.** Whisper and wav2vec2 take a long time, so each stage saves its output, can resume after an interruption, and only re-runs when something it depends on has changed.
- **Clipping instead of rounding.** Rounding predictions to the nearest 0.5 was tested and made RMSE worse (0.661 vs 0.638), so predictions are only clipped to [0, 5].

## Things worth knowing about the data

- **212 filenames exist in both `train/` and `test/`**, but they are different recordings. Every clip is therefore keyed by split, as `train/audio_12.wav` or `test/audio_12.wav`. Keying on the filename alone would silently mix them up.
- **Scores 1.0–1.5 are rare** (4 clips), so for stratified folds they're grouped with 0.
- **6 training clips transcribe to nothing.** The 37 clips scored 0 are *not* silent (median 77 words), so a 0 seems to mean something other than "no speech".

## Limitations and future work

- **Whisper hides grammar errors.** Its language model smooths out mistakes, so the regex rules see little. A more literal transcriber, or a model trained to detect grammar errors, could help the text side.
- **Mean pooling throws away timing.** Averaging over time loses where pauses and hesitations happen. Attention pooling, or using several wav2vec2 layers, may capture more.
- **Small dataset.** 769 clips limits how complex a model can be. More data, or fine-tuning with strong regularisation, are the obvious next steps.
- **Pickle model file.** `model.pkl` uses Python's `pickle`, so only load model files you created yourself.
