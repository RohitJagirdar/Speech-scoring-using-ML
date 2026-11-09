"""
Stage 1: turn every audio clip into text with OpenAI Whisper.

This is the slowest stage (roughly 20-40 min for 985 clips on a GPU), so it is
resumable: transcripts are checkpointed to disk every few clips, and a rerun
only transcribes the clips that are not in the file yet.
"""

import librosa
import torch
import whisper
from tqdm import tqdm

from src.config import AUDIO_SAMPLE_RATE, TRANSCRIPTS_PATH, WHISPER_MODEL_NAME
from src.data_utils import load_manifest, load_transcripts, require_audio_files, save_json

CHECKPOINT_EVERY = 20  # clips; a crash loses at most this much work


def transcribe_audio_corpus(model_name: str = WHISPER_MODEL_NAME) -> dict[str, str]:
    """Transcribe every train and test clip. Returns {key: transcript}."""
    items = load_manifest()
    require_audio_files(items)

    transcripts = load_transcripts()
    todo = [item for item in items if item.key not in transcripts]
    if not todo:
        print(f"[transcribe] All {len(items)} clips already transcribed.")
        return transcripts
    print(f"[transcribe] {len(items) - len(todo)} cached, {len(todo)} to go.")

    # Half precision roughly doubles speed on a GPU but isn't supported on CPU.
    use_fp16 = torch.cuda.is_available()
    model = whisper.load_model(model_name)

    for done, item in enumerate(tqdm(todo, desc="Whisper", unit="clip"), start=1):
        # Loading with librosa means we don't need ffmpeg installed.
        audio, _ = librosa.load(item.path, sr=AUDIO_SAMPLE_RATE, mono=True)
        result = model.transcribe(audio, fp16=use_fp16, language="en")
        transcripts[item.key] = result["text"].strip()

        if done % CHECKPOINT_EVERY == 0:
            save_json(TRANSCRIPTS_PATH, transcripts)

    save_json(TRANSCRIPTS_PATH, transcripts)
    print(f"[transcribe] Saved {len(transcripts)} transcripts to {TRANSCRIPTS_PATH}")
    return transcripts


if __name__ == "__main__":
    transcribe_audio_corpus()
