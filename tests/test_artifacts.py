"""
Checks on the real pipeline outputs. Skipped automatically until you've run
`python main.py`, so a fresh clone still gets a green `pytest`.
"""
import pytest

from src.config import MODEL_PATH, SUBMISSION_PATH
from src.data_utils import load_labels
from src.verify import Verdict, run_checks

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SUBMISSION_PATH.exists()),
    reason="artifacts not built yet; run `python main.py` first",
)


def test_dataset_sizes():
    assert len(load_labels("train")) == 769
    assert len(load_labels("test")) == 216


def test_every_verification_check_passes():
    failures = [f"{r.name}: {r.detail}" for r in run_checks() if r.verdict is Verdict.FAIL]
    assert not failures, failures
