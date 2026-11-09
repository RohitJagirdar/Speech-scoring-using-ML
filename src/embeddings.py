"""
Stage 3: describe *how* each clip sounds with a 768-number wav2vec2 vector.

We use wav2vec2 as a frozen feature extractor (no fine-tuning): with only 769
labelled clips, fine-tuning a 95M-parameter model would just memorise them.
"""
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.config import AUDIO_SAMPLE_RATE, EMBEDDINGS_PATH, WAV2VEC2_MODEL_NAME
from src.data_utils import AudioItem, load_manifest, require_audio_files, save_atomic


def embed_clip(item: AudioItem, extractor, model, device) -> np.ndarray:
    """One clip -> one 768-dim vector (the average of wav2vec2's per-frame outputs)."""
    waveform, _ = librosa.load(item.path, sr=AUDIO_SAMPLE_RATE, mono=True)
    inputs = extractor(waveform, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        frames = model(inputs.input_values.to(device)).last_hidden_state  # (1, T, 768), ~1 frame per 20 ms

    # Mean pooling: clips have different lengths, the regressor needs a fixed size.
    return frames.mean(dim=1).squeeze(0).cpu().numpy()


def compute_audio_embeddings(model_name: str = WAV2VEC2_MODEL_NAME) -> np.ndarray:
    """Stage entry point: one vector per clip -> embeddings.npz."""
    items = load_manifest()
    require_audio_files(items)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()  # eval() turns off dropout

    # No try/except on purpose: a broken clip should stop the run with its name,
    # not turn into a row of zeros that the model silently learns from.
    vectors = [embed_clip(item, extractor, model, device) for item in tqdm(items, desc="wav2vec2", unit="clip")]
    embeddings = np.stack(vectors).astype(np.float32)

    save_atomic(EMBEDDINGS_PATH, lambda p: np.savez_compressed(
        p, keys=np.array([item.key for item in items]), embeddings=embeddings))
    print(f"[embeddings] Saved {embeddings.shape} to {EMBEDDINGS_PATH}")
    return embeddings


if __name__ == "__main__":
    compute_audio_embeddings()
