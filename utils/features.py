# utils/features.py
# Feature extraction and disk caching for the training pipeline.
#
# WHAT WAS REMOVED AND WHY:
#   extract_lfcc()         — never called by trainer or inference. LFCC was
#                            an early experiment, mel spectrogram was chosen.
#   save_spectrogram_plot()— never called anywhere in the pipeline.
#                            Our app_inference.py generates its own report.
#   IMAGENET_MEAN/STD      — ImageNet normalisation constants. Not used
#                            because LCNN uses single-channel mel, not RGB.
#   scipy.fftpack dct      — only used by extract_lfcc (removed above).
#   matplotlib/gridspec    — only used by save_spectrogram_plot (removed above).
#
# WHAT IS KEPT AND WHY:
#   extract_mel()          — called by trainer.py for every training sample.
#   get_cache_path()       — called by trainer.py to find cached features.
#   load_from_cache()      — called by trainer.py to skip recomputing features.
#   save_to_cache()        — called by trainer.py after computing features.
#   IMG_SIZE               — imported by export_onnx.py for dummy input shape.

import os
import hashlib
import numpy as np
import librosa
import torch
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS      = 128
IMG_SIZE    = 128      # LCNN input size — also used by export_onnx.py
HOP_LENGTH  = 128
N_FFT       = 2048


# ── Mel Spectrogram ───────────────────────────────────────────────────────────
def extract_mel(y, sr=SAMPLE_RATE, img_size=IMG_SIZE):
    """
    Convert audio waveform to log-mel spectrogram image.

    Steps:
        1. Compute mel spectrogram (N_MELS frequency bands over time)
        2. Convert power to decibels (log scale — matches human hearing)
        3. Normalise to 0–1 range
        4. Resize to img_size x img_size (128x128 for LCNN)
        5. Return as (1, H, W) — single channel, ready for LCNN

    Why mel spectrogram:
        Mel scale matches how humans perceive pitch — closer bands at low
        frequencies, wider at high. This makes the spectrogram a compact
        visual representation of the voice's frequency content over time.
        The LCNN treats it as an image and learns which visual patterns
        indicate real vs synthetic speech.
    """
    mel     = librosa.feature.melspectrogram(
                  y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)

    img = Image.fromarray((log_mel * 255).astype(np.uint8))
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0

    return img[np.newaxis, :, :]   # (1, H, W)


# ── Disk Caching ──────────────────────────────────────────────────────────────
# The ASVspoof2019 dataset has ~25,000 files. Computing mel spectrograms
# from scratch every epoch would take hours. These functions cache each
# file's features to disk after the first computation so subsequent epochs
# load instantly.

def get_cache_path(cache_dir, fname, augmented=False):
    """
    Generate a unique cache file path for a given audio filename.
    Uses MD5 hash of the filename so paths don't get too long.
    Augmented samples get a separate cache key (_aug suffix).
    """
    suffix = "_aug" if augmented else ""
    key    = hashlib.md5(fname.encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}{suffix}.pt")


def load_from_cache(cache_path):
    """Load cached features if they exist. Returns None if not cached yet."""
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    return None


def save_to_cache(cache_path, mel, lfcc):
    """
    Save mel spectrogram to disk cache.
    lfcc parameter kept for API compatibility but saved as empty tensor
    since LFCC extraction was removed.
    """
    torch.save({
        "mel":  torch.tensor(mel, dtype=torch.float32),
        "lfcc": torch.zeros(1, dtype=torch.float32)   # placeholder
    }, cache_path)
