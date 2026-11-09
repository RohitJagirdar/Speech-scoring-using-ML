"""
Stage 2: describe *what* was said with 20 readable numbers per transcript.

These are deliberately simple (counts, ratios, regex rules) so every feature
can be explained in one sentence, which a 768-dim embedding cannot.
"""
import re
import string
from typing import Any

import pandas as pd
import textstat
from spellchecker import SpellChecker

from src.config import NLP_FEATURES_PATH
from src.data_utils import load_manifest, load_transcripts, save_atomic

FILLER_WORDS = {"um", "uh", "er", "ah", "hmm", "like", "basically", "literally"}

# Each rule targets one common spoken-English mistake. They are cheap and transparent,
# but on this data they rarely fire: Whisper's decoder is itself a language model and
# tends to "tidy up" grammar while transcribing. Most of the signal ends up coming
# from fluency/vocabulary features and the audio embeddings instead.
GRAMMAR_RULES = {
    # "I don't have no time"
    "double_negative": r"\b(don\'?t|doesn\'?t|didn\'?t|won\'?t|can\'?t|never|no)\b(?:\s+\w+){0,3}\s+\b(no|nothing|nobody|never|nowhere|neither)\b",  # noqa: E501
    # "they was", "you was". Only subject-first: "the thing is we can..." is fine English.
    "aux_verb_agreement": r"\b(we|they|you)\s+(is|was)\b",
    # "he do", "she have"
    "subject_verb_mismatch": r"\b(he|she|it)\s+(do|have|go)\b",
    # "dont", "cant" (Whisper usually adds apostrophes, so this fires rarely)
    "missing_apostrophe": r"\b(dont|doesnt|didnt|wont|cant|couldnt|shouldnt|wouldnt|isnt|arent|wasnt|werent)\b",
    # "more better"
    "redundant_comparative": r"\bmore\s+(better|faster|slower|bigger|smaller|harder|easier|taller|shorter)\b",
    # "goed", "buyed"
    "irregular_verb_error": r"\b(goed|eated|runned|buyed|bringed|catched|teached|writed|speaked)\b",
    # sentence ending on a preposition: "...the place I went to."
    "dangling_preposition": r"\b(in|on|at|with|by|from|about|to)\s*[.?!]",
}
COMPILED_RULES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in GRAMMAR_RULES.items()}

_spell = SpellChecker()  # loads a word list once; slow to create, fast to use


def extract_features_from_text(text: str) -> dict[str, Any]:
    """All 20 features for one transcript. Empty text gives (mostly) zeros."""
    text = text or ""
    words = text.split()
    word_count = len(words)
    per_word = max(word_count, 1)  # avoids dividing by zero on silent clips

    # Lowercased words with punctuation stripped, used for vocabulary and spelling.
    clean_words = [w.strip(string.punctuation).lower() for w in words]
    clean_words = [w for w in clean_words if w.isalpha()]
    unique_words = set(clean_words)

    sentences = [s for s in re.split(r"[.?!]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)

    filler_count = sum(1 for w in words if w.lower() in FILLER_WORDS)
    misspelled = _spell.unknown(clean_words)  # a set, so each distinct misspelling counts once
    rule_hits = {name: len(rule.findall(text)) for name, rule in COMPILED_RULES.items()}

    try:
        flesch_ease = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
    except Exception:  # textstat can choke on empty or odd strings
        flesch_ease = flesch_grade = 0.0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(word_count / sentence_count, 4),
        "unique_word_count": len(unique_words),
        "type_token_ratio": round(len(unique_words) / per_word, 6),  # vocabulary variety
        "filler_word_count": filler_count,
        "filler_word_rate": round(filler_count / per_word, 6),
        "spell_error_count": len(misspelled),
        "spell_error_rate": round(len(misspelled) / per_word, 6),
        "grammar_pattern_errors": sum(rule_hits.values()),
        **rule_hits,
        "flesch_reading_ease": round(flesch_ease, 4),
        "flesch_kincaid_grade": round(flesch_grade, 4),
        "char_count": len(text),
    }


def compute_nlp_feature_table() -> pd.DataFrame:
    """Stage entry point: features for every clip -> nlp_features.csv."""
    transcripts = load_transcripts()
    rows = [
        {"unique_key": item.key, "filename": item.filename, **extract_features_from_text(transcripts.get(item.key, ""))}
        for item in load_manifest()
    ]
    df = pd.DataFrame(rows)
    save_atomic(NLP_FEATURES_PATH, lambda p: df.to_csv(p, index=False))
    print(f"[nlp_features] Saved {df.shape[0]} rows x {df.shape[1] - 2} features to {NLP_FEATURES_PATH}")
    return df


if __name__ == "__main__":
    compute_nlp_feature_table()
