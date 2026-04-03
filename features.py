# Feature extraction: Mel-spectrogram and LFCC
# Includes disk caching and spectrogram visualisation

import os
import hashlib
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.fftpack import dct
import torch


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS      = 128
IMG_SIZE    = 128      # LCNN works well at 128x128
HOP_LENGTH  = 128
N_FFT       = 2048
N_LFCC      = 60

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Mel-spectrogram ───────────────────────────────────────────────────────────
def extract_mel(y, sr=SAMPLE_RATE, img_size=IMG_SIZE):
    """
    Convert audio to log-mel spectrogram image.
    Returns: numpy array (1, H, W) — single channel for LCNN
    """
    mel     = librosa.feature.melspectrogram(
                  y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)

    img = Image.fromarray((log_mel * 255).astype(np.uint8))
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0

    return img[np.newaxis, :, :]   # (1, H, W)


# ── LFCC ──────────────────────────────────────────────────────────────────────
def extract_lfcc(y, sr=SAMPLE_RATE, n_lfcc=N_LFCC):
    """
    Extract LFCC — Linear Frequency Cepstral Coefficients.
    Better than MFCC for catching high-frequency AI artifacts.
    Returns: numpy array (n_lfcc,)
    """
    n_filter = 128
    fft_mag  = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    centers  = np.linspace(0, sr / 2, n_filter + 2)
    filters  = np.zeros((n_filter, len(freqs)))

    for i in range(n_filter):
        left, center, right = centers[i], centers[i+1], centers[i+2]
        for j, f in enumerate(freqs):
            if left <= f <= center:
                filters[i, j] = (f - left) / (center - left + 1e-8)
            elif center < f <= right:
                filters[i, j] = (right - f) / (right - center + 1e-8)

    linear_spec = filters @ fft_mag
    log_spec    = np.log(linear_spec + 1e-8)
    lfcc        = dct(log_spec, type=2, axis=0, norm='ortho')[:n_lfcc]
    lfcc        = (lfcc - lfcc.mean()) / (lfcc.std() + 1e-6)
    return lfcc.mean(axis=1).astype(np.float32)


# ── Visualisation ─────────────────────────────────────────────────────────────
def save_spectrogram_plot(y, sr, output_path, title="Audio Analysis",
                           prediction=None, confidence=None, verdict=None):
    """
    Generate a detailed spectrogram analysis plot and save as PNG.
    Shows: waveform, mel-spectrogram, and LFCC heatmap side by side.
    This is what gets shown to the user after they upload audio.
    """
    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    text_color  = '#e0e0e0'
    accent      = '#00d4aa' if (verdict == "REAL") else \
                  '#ff4444' if (verdict == "FAKE") else '#ffaa00'

    # ── Title and verdict ────────────────────────────────────────────────────
    if prediction is not None:
        verdict_text = f"{verdict}  —  {confidence*100:.1f}% confidence"
        fig.suptitle(f"{title}\n{verdict_text}",
                     fontsize=14, color=accent, fontweight='bold', y=0.98)
    else:
        fig.suptitle(title, fontsize=14, color=text_color, y=0.98)

    # ── Waveform ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    times = np.linspace(0, len(y) / sr, len(y))
    ax1.plot(times, y, color=accent, linewidth=0.4, alpha=0.8)
    ax1.fill_between(times, y, alpha=0.15, color=accent)
    ax1.set_facecolor('#1a1a1a')
    ax1.set_title("Waveform", color=text_color, fontsize=10)
    ax1.set_xlabel("Time (s)", color=text_color, fontsize=8)
    ax1.set_ylabel("Amplitude", color=text_color, fontsize=8)
    ax1.tick_params(colors=text_color, labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')
    ax1.grid(alpha=0.15, color='#444444')

    # ── Mel-spectrogram ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    mel     = librosa.feature.melspectrogram(
                  y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    img2    = ax2.imshow(log_mel, aspect='auto', origin='lower',
                          cmap='magma', interpolation='nearest')
    ax2.set_facecolor('#1a1a1a')
    ax2.set_title("Mel-spectrogram", color=text_color, fontsize=10)
    ax2.set_xlabel("Time frames", color=text_color, fontsize=8)
    ax2.set_ylabel("Mel frequency", color=text_color, fontsize=8)
    ax2.tick_params(colors=text_color, labelsize=7)
    plt.colorbar(img2, ax=ax2).ax.tick_params(colors=text_color, labelsize=6)

    # ── LFCC heatmap ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    n_filter = 64
    fft_mag  = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    centers  = np.linspace(0, sr / 2, n_filter + 2)
    filters  = np.zeros((n_filter, len(freqs)))
    for i in range(n_filter):
        left, center, right = centers[i], centers[i+1], centers[i+2]
        for j, f in enumerate(freqs):
            if left <= f <= center:
                filters[i, j] = (f - left) / (center - left + 1e-8)
            elif center < f <= right:
                filters[i, j] = (right - f) / (right - center + 1e-8)
    linear_spec = filters @ fft_mag
    log_spec    = np.log(linear_spec + 1e-8)
    lfcc_full   = dct(log_spec, type=2, axis=0, norm='ortho')[:N_LFCC]

    img3 = ax3.imshow(lfcc_full, aspect='auto', origin='lower',
                       cmap='viridis', interpolation='nearest')
    ax3.set_facecolor('#1a1a1a')
    ax3.set_title("LFCC (Linear Frequency)", color=text_color, fontsize=10)
    ax3.set_xlabel("Time frames", color=text_color, fontsize=8)
    ax3.set_ylabel("LFCC coefficient", color=text_color, fontsize=8)
    ax3.tick_params(colors=text_color, labelsize=7)
    plt.colorbar(img3, ax=ax3).ax.tick_params(colors=text_color, labelsize=6)

    # ── Spectral centroid ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT,
                                                   hop_length=HOP_LENGTH)[0]
    t_frames = librosa.frames_to_time(
        np.arange(len(centroid)), sr=sr, hop_length=HOP_LENGTH)
    ax4.plot(t_frames, centroid, color='#7b68ee', linewidth=0.8)
    ax4.fill_between(t_frames, centroid, alpha=0.2, color='#7b68ee')
    ax4.set_facecolor('#1a1a1a')
    ax4.set_title("Spectral centroid", color=text_color, fontsize=10)
    ax4.set_xlabel("Time (s)", color=text_color, fontsize=8)
    ax4.set_ylabel("Hz", color=text_color, fontsize=8)
    ax4.tick_params(colors=text_color, labelsize=7)
    for spine in ax4.spines.values():
        spine.set_edgecolor('#333333')
    ax4.grid(alpha=0.15, color='#444444')

    plt.savefig(output_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Spectrogram saved: {output_path}")
    return output_path


# ── Disk caching ──────────────────────────────────────────────────────────────
def get_cache_path(cache_dir, fname, augmented=False):
    suffix = "_aug" if augmented else ""
    key    = hashlib.md5(fname.encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}{suffix}.pt")


def load_from_cache(cache_path):
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)
    return None


def save_to_cache(cache_path, mel, lfcc):
    torch.save({
        "mel":  torch.tensor(mel,  dtype=torch.float32),
        "lfcc": torch.tensor(lfcc, dtype=torch.float32)
    }, cache_path)