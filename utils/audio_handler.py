# utils/audio_handler.py
# Audio loading for the training pipeline.
#
# WHAT WAS REMOVED AND WHY:
#   get_audio_info()  — never called by trainer, app_inference, or any
#                       other file. Was a utility left over from early
#                       development that was never integrated.
#
# WHAT IS KEPT AND WHY:
#   load_audio()      — called by trainer.py for every training sample.
#
# NOTE:
#   app_inference.py has its own _load_audio() with pydub support and
#   RMS normalisation for compressed audio (WhatsApp/Gmail robustness).
#   This function is intentionally simpler — training files are clean
#   FLAC so the extra handling is not needed here.

import os
import numpy as np
import librosa


SAMPLE_RATE  = 16000
MAX_DURATION = 6      # seconds — caps very long files during training
MIN_DURATION = 0.5    # seconds — minimum after silence trimming


def load_audio(file_path):
    """
    Load any audio file and return a clean 16kHz mono numpy array.

    Steps:
        1. Load with librosa (handles wav, flac natively; mp3/m4a via ffmpeg)
        2. Trim leading/trailing silence (top_db=20)
        3. Normalise volume to peak 0.9
        4. Pad if too short, trim if too long

    Returns: (y, sr) — float32 numpy array, sample rate (always 16000)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load audio file. Make sure ffmpeg is installed "
            f"for mp3/m4a support.\nError: {e}"
        )

    if len(y) == 0:
        raise ValueError("Audio file is empty or could not be decoded.")

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) < SAMPLE_RATE * MIN_DURATION:
        y_trimmed = y   # keep original if too much was trimmed

    # Normalise volume
    max_val = np.max(np.abs(y_trimmed))
    if max_val > 1e-6:
        y_trimmed = y_trimmed / max_val * 0.9

    # Enforce duration limits
    if len(y_trimmed) < SAMPLE_RATE:
        y_trimmed = np.pad(y_trimmed, (0, SAMPLE_RATE - len(y_trimmed)))
    if len(y_trimmed) > SAMPLE_RATE * MAX_DURATION:
        y_trimmed = y_trimmed[:SAMPLE_RATE * MAX_DURATION]

    return y_trimmed.astype(np.float32), SAMPLE_RATE
