# Handles loading any audio format and converting to standard format
# Supports: mp3, wav, flac, m4a, ogg, aac, wma
# All audio is resampled to 16kHz mono before processing

import numpy as np
import librosa
import os


SAMPLE_RATE = 16000
MAX_DURATION = 6      # seconds
MIN_DURATION = 0.5    # seconds


def load_audio(file_path):
    """
    Load any audio format and return a clean 16kHz mono numpy array.
    Handles: mp3, wav, flac, m4a, ogg, aac and more via librosa/soundfile.
    Applies silence trimming and volume normalisation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    print(f"Loading {ext} file: {os.path.basename(file_path)}")

    # librosa handles most formats natively
    # for m4a/aac it falls back to audioread which uses ffmpeg
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load audio file. Make sure ffmpeg is installed "
            f"for mp3/m4a support.\nError: {e}"
        )

    if len(y) == 0:
        raise ValueError("Audio file is empty or could not be decoded.")

    # Trim leading and trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # If too much was trimmed, use original
    if len(y_trimmed) < SAMPLE_RATE * MIN_DURATION:
        y_trimmed = y

    # Normalise volume
    max_val = np.max(np.abs(y_trimmed))
    if max_val > 1e-6:
        y_trimmed = y_trimmed / max_val * 0.9

    # Enforce duration limits
    if len(y_trimmed) < SAMPLE_RATE:
        y_trimmed = np.pad(y_trimmed, (0, SAMPLE_RATE - len(y_trimmed)))
    if len(y_trimmed) > SAMPLE_RATE * MAX_DURATION:
        y_trimmed = y_trimmed[:SAMPLE_RATE * MAX_DURATION]

    duration = len(y_trimmed) / SAMPLE_RATE
    print(f"Audio loaded: {duration:.2f}s at {SAMPLE_RATE}Hz mono")

    return y_trimmed.astype(np.float32), SAMPLE_RATE


def get_audio_info(file_path):
    """Return basic info about an audio file without full processing."""
    y, sr = librosa.load(file_path, sr=None, mono=False)
    channels = 1 if y.ndim == 1 else y.shape[0]
    duration = y.shape[-1] / sr
    return {
        "path":      file_path,
        "format":    os.path.splitext(file_path)[1].lower(),
        "duration":  round(duration, 2),
        "sample_rate": sr,
        "channels":  channels
    }