"""
Final step: sanity-check every artifact and the submission before you hand it in.

Each check gives PASS, WARN or FAIL plus a one-line reason. WARN means
"worth a look but not wrong" (e.g. a few clips with no speech). Only a FAIL
makes the pipeline exit with an error code.
"""
import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.config import MODEL_PATH, NLP_FEATURES_PATH, SAMPLE_SUBMISSION_CSV, SUBMISSION_PATH
from src.data_utils import load_embeddings, load_labels, load_manifest, load_transcripts


class Verdict(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    verdict: Verdict
    detail: str


def verdict(failed: bool, warn: bool = False) -> Verdict:
    return Verdict.FAIL if failed else Verdict.WARN if warn else Verdict.PASS


def check_audio() -> tuple[Verdict, str]:
    items = load_manifest()
    missing = sum(not item.path.exists() for item in items)
    shared = set(load_labels("train")["filename"]) & set(load_labels("test")["filename"])
    detail = f"{len(items) - missing}/{len(items)} present; {len(shared)} names in both splits (kept apart by key)"
    return verdict(missing > 0), detail


def check_transcripts() -> tuple[Verdict, str]:
    keys = {item.key for item in load_manifest()}
    transcripts = load_transcripts()
    missing = len(keys - transcripts.keys())
    near_empty = sum(len(transcripts.get(k, "").strip()) < 5 for k in keys)
    detail = f"{len(keys) - missing}/{len(keys)} clips; {near_empty} near-empty (silent or inaudible)"
    return verdict(missing > 0, warn=near_empty > 0), detail


def check_nlp_features() -> tuple[Verdict, str]:
    df = pd.read_csv(NLP_FEATURES_PATH)
    nulls = int(df.isnull().sum().sum())
    return verdict(nulls > 0), f"{df.shape[0]} rows x {df.shape[1] - 2} features, {nulls} nulls"


def check_embeddings() -> tuple[Verdict, str]:
    matrix = np.stack(list(load_embeddings().values()))
    nans = int(np.isnan(matrix).sum())
    zero_rows = int(np.all(matrix == 0, axis=1).sum())
    return verdict(nans > 0, warn=zero_rows > 0), f"{matrix.shape}, {nans} NaNs, {zero_rows} all-zero rows"


def check_no_leakage() -> tuple[Verdict, str]:
    """The saved scaler must have seen exactly the training rows, never test rows."""
    bundle = pickle.loads(MODEL_PATH.read_bytes())
    seen = int(bundle["pipeline"][0].n_samples_seen_)
    n_train = len(load_labels("train"))
    return verdict(seen != n_train), f"{bundle['model_name']}; scaler fit on {seen} rows (train has {n_train})"


def check_submission() -> tuple[Verdict, str]:
    sub = pd.read_csv(SUBMISSION_PATH)
    expected_cols = list(pd.read_csv(SAMPLE_SUBMISSION_CSV, nrows=0).columns)
    test_names = set(load_labels("test")["filename"])
    problems = []
    if list(sub.columns) != expected_cols:
        problems.append(f"columns {list(sub.columns)} != {expected_cols}")
    if set(sub["filename"]) != test_names or len(sub) != len(test_names):
        problems.append("filenames don't match test.csv")
    if sub["label"].isnull().any() or not sub["label"].between(0, 5).all():
        problems.append("scores missing or outside 0-5")
    summary = f"{len(sub)} rows, scores {sub['label'].min():.2f}-{sub['label'].max():.2f}"
    return verdict(bool(problems)), "; ".join(problems) or summary


CHECKS = {
    "audio files": check_audio,
    "transcripts": check_transcripts,
    "nlp features": check_nlp_features,
    "embeddings": check_embeddings,
    "no test leakage": check_no_leakage,
    "submission": check_submission,
}


def run_checks() -> list[CheckResult]:
    results = []
    for name, check in CHECKS.items():
        try:
            results.append(CheckResult(name, *check()))
        except Exception as e:  # a missing artifact is a failed check, not a crash
            results.append(CheckResult(name, Verdict.FAIL, f"{type(e).__name__}: {e}"))
    return results


def run_full_pipeline_verification() -> bool:
    """Print the report. True if nothing FAILed."""
    results = run_checks()
    print("\n[verify]")
    for r in results:
        print(f"  {r.verdict.value:<5} {r.name:<16} {r.detail}")
    passed = all(r.verdict is not Verdict.FAIL for r in results)
    print("  -> all good" if passed else "  -> fix the FAIL lines above")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run_full_pipeline_verification() else 1)
