"""
Fast tests: no GPU, no audio, no trained model needed. Run with `pytest`.
"""
import os

import numpy as np
import pytest

from src.data_utils import AudioItem, make_key, save_atomic
from src.nlp_features import extract_features_from_text
from src.pipeline import Stage, Status
from src.train import clip_scores, stratify_bins

# --- keys: the train/test filename collision ---------------------------------

def test_same_filename_in_both_splits_gets_different_keys():
    assert AudioItem("train", "audio_1.wav").key != AudioItem("test", "audio_1.wav").key
    assert make_key("test", "audio_1.wav") == "test/audio_1.wav"


# --- text features -------------------------------------------------------------

def test_grammar_rules_fire_on_known_mistakes():
    f = extract_features_from_text("They was happy. I don't have no money. He buyed it. It was more better.")
    assert f["aux_verb_agreement"] == 1
    assert f["double_negative"] == 1
    assert f["irregular_verb_error"] == 1
    assert f["redundant_comparative"] == 1
    assert f["grammar_pattern_errors"] == 4


def test_grammar_rules_ignore_correct_english():
    # Each of these tripped an earlier version of the rules.
    f = extract_features_from_text("You are right. The good thing is we can go. I could express myself.")
    assert f["grammar_pattern_errors"] == 0


def test_fillers_and_ratios():
    f = extract_features_from_text("um I like like dogs")
    assert f["word_count"] == 5
    assert f["filler_word_count"] == 3  # "um", "like", "like"
    assert f["filler_word_rate"] == pytest.approx(0.6)


def test_empty_transcript_does_not_crash():
    f = extract_features_from_text("")
    assert f["word_count"] == 0 and f["type_token_ratio"] == 0
    assert len(f) == 20


# --- training helpers ------------------------------------------------------------

def test_scores_are_clipped_to_rubric():
    assert clip_scores(np.array([-1.0, 2.5, 7.0])).tolist() == [0.0, 2.5, 5.0]


def test_rare_low_scores_are_merged_into_one_band():
    bands = stratify_bins(np.array([0.0, 1.0, 1.5, 2.5, 5.0]))
    assert bands.tolist() == [0, 0, 0, 2, 5]


# --- the stage state machine --------------------------------------------------------

def _touch(path, mtime):
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_stage_moves_through_every_status(tmp_path):
    src, out = tmp_path / "input.txt", tmp_path / "output.txt"
    complete = {"value": True}
    stage = Stage("demo", "", "unused:unused", outputs=[out], inputs=[src],
                  covers_all_clips=lambda: complete["value"])

    _touch(src, 1_000)
    assert stage.status() is Status.MISSING

    _touch(out, 2_000)
    assert stage.status() is Status.READY

    complete["value"] = False  # e.g. transcription interrupted halfway
    assert stage.status() is Status.INCOMPLETE

    complete["value"] = True
    _touch(src, 3_000)  # upstream re-ran after us
    assert stage.status() is Status.STALE


def test_save_atomic_leaves_no_temp_file(tmp_path):
    target = tmp_path / "result.json"
    save_atomic(target, lambda p: p.write_text("{}"))
    assert target.read_text() == "{}"
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]
