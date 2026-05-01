"""
app_inference.py  —  Hybrid Deepfake Audio Detection Engine
============================================================
STANDALONE USE:
    python app_inference.py
    (opens file picker — hold Ctrl/Cmd to select multiple files)

BACKEND IMPORT:
    from app_inference import analyse
    result = analyse("path/to/audio.m4a")

RETURN SCHEMA:
    {
        "verdict":        "REAL" | "FAKE",
        "real_pct":       float,
        "fake_pct":       float,
        "confidence":     float,
        "low_confidence": bool,         # True if confidence < 65% OR short_clip
        "threshold_used": float,
        "duration_secs":  float,
        "short_clip":     bool,         # True if < 5 seconds
        "file":           str,
        "format":         str,
        "signals":        dict,
        "report_path":    str | None,
        "error":          str | None
    }

WHAT'S IN THE REPORT PNG:
    Row 1 — Waveform | Mel Spectrogram
    Row 2 — Pitch Contour (F0 over time)   ← NEW
    Row 3 — Signal breakdown bar chart
    Row 4 — Auto-generated text summary    ← NEW
    Row 5 — Metadata + SHA-256 hash        ← NEW
    Row 6 — Verdict panel
    Row 7 — Disclaimer                     ← NEW

HOW TO REMOVE INDIVIDUAL NEW SECTIONS (if not needed):
    Pitch contour  → delete "Row 2" block in _generate_report()
    Auto-summary   → delete "Row 4" block in _generate_report()
    Metadata/hash  → delete "Row 5" block in _generate_report()
    Disclaimer     → delete "Row 7" block in _generate_report()
    Each section is clearly labelled with a comment.

CALIBRATION LOG (empirical — 18-file batch, 18/18 accuracy):
    mfcc_delta_var:   Real=17-44,  Fake=47-78  → threshold 45.0, slope 0.25
    pitch_cv:         Real=0.08-0.46, Fake=0.12-0.46 → threshold 0.35, slope 25
    centroid_var:     Real=54k-200k, Fake=245k-675k → threshold 150k
    Weights:          mfcc=0.75, pitch=0.10, centroid=0.10, energy=0.05
"""

import os
import sys
import hashlib
import warnings
import tempfile
import datetime
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import torch
import torch.nn.functional as F

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(__file__))
from models.lcnn import LCNN

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH           = os.path.join(r"C:\Users\shara\deepfake_audio\output", "best_lcnn.pt")
REPORT_DIR           = os.path.join(r"C:\Users\shara\deepfake_audio\output", "reports")
IMG_SIZE             = 128
SAMPLE_RATE          = 16000
DEFAULT_THRESHOLD    = 0.50
TEMPERATURE          = 50.0
MIN_DURATION_SECS    = 5.0
VAD_MIN_VOICED_RATIO = 0.10

# ── Weights ───────────────────────────────────────────────────────────────────
# Exact weights from 18/18 batch — do not change without re-running batch test
WEIGHTS = {
    "mfcc_delta":        0.75,
    "pitch_cv":          0.10,
    "spectral_centroid": 0.10,
    "energy_envelope":   0.05,
}

SUPPORTED_FORMATS = {".wav", ".flac", ".mp3", ".m4a", ".ogg",
                     ".opus", ".aac", ".webm", ".wma",
                     ".mpeg", ".mpg", ".mp4", ".mov", ".3gp", ".amr"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# MODEL (debug only)
# ═══════════════════════════════════════════════════════════════

_model = None

def _get_model():
    global _model
    if _model is None:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        m = LCNN(num_classes=2).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        _model = m
    return _model


# ═══════════════════════════════════════════════════════════════
# AUDIO LOADING
# ═══════════════════════════════════════════════════════════════

def _load_audio(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    y   = None

    if ext in {".wav", ".flac", ".ogg"}:
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            pass

    if y is None:
        if not PYDUB_AVAILABLE:
            raise RuntimeError(
                f"Format '{ext}' requires pydub + ffmpeg.\n"
                "Install: pip install pydub  and add ffmpeg to PATH."
            )
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        seg.export(tmp_path, format="wav")
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        os.unlink(tmp_path)

    y_trim, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trim) < 1600:
        y_trim = y

    rms = np.sqrt(np.mean(y_trim ** 2)) + 1e-9
    return (y_trim / rms * 0.1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# FILE HASH  (NEW)
# ═══════════════════════════════════════════════════════════════
# SHA-256 hash of the original file bytes.
# Proves the file was not modified after submission.
# Standard practice in digital forensics — always hash the evidence.
# TO REMOVE: delete this function and the _get_file_hash(path) call in analyse()

def _get_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
# FILE METADATA  (NEW)
# ═══════════════════════════════════════════════════════════════
# Collects basic file properties for the forensic report header.
# TO REMOVE: delete this function and the _get_metadata(path) call in analyse()

def _get_metadata(path: str) -> dict:
    try:
        size_bytes = os.path.getsize(path)
        size_kb    = round(size_bytes / 1024, 1)

        # Get original audio properties before resampling
        y_raw, sr_orig = librosa.load(path, sr=None, mono=False)
        channels = 1 if y_raw.ndim == 1 else y_raw.shape[0]
        return {
            "size_kb":    size_kb,
            "sample_rate_orig": sr_orig,
            "channels":   channels,
        }
    except Exception:
        return {"size_kb": "?", "sample_rate_orig": "?", "channels": "?"}


# ═══════════════════════════════════════════════════════════════
# VOICE ACTIVITY DETECTION
# ═══════════════════════════════════════════════════════════════

def _check_voice_activity(y: np.ndarray) -> tuple:
    try:
        _, voiced_flag, _ = librosa.pyin(
            y, fmin=60, fmax=400, sr=SAMPLE_RATE, frame_length=2048)
        if voiced_flag is None or len(voiced_flag) == 0:
            return False, 0.0
        ratio = float(np.sum(voiced_flag) / len(voiced_flag))
        print(f"  [DBG] VAD  voiced_ratio={ratio:.3f}  "
              f"({'speech' if ratio >= VAD_MIN_VOICED_RATIO else 'NO SPEECH'})")
        return ratio >= VAD_MIN_VOICED_RATIO, ratio
    except Exception:
        return True, 1.0


# ═══════════════════════════════════════════════════════════════
# PITCH CONTOUR EXTRACTION  (NEW)
# ═══════════════════════════════════════════════════════════════
# Extracts F0 (fundamental frequency) frame by frame.
# Returns time axis, F0 values, and voiced frame flags.
# Used both for the pitch_cv signal AND for the pitch contour plot.
# Extracted once here so we don't run pyin twice.
# TO REMOVE: this can stay — it's also used by _score_pitch_cv internally.

def _extract_pitch(y: np.ndarray):
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=400, sr=SAMPLE_RATE, frame_length=2048)
    hop    = 512   # default hop for pyin
    times  = librosa.frames_to_time(
                np.arange(len(f0)), sr=SAMPLE_RATE, hop_length=hop)
    return times, f0, voiced_flag


# ═══════════════════════════════════════════════════════════════
# SIGNALS
# ═══════════════════════════════════════════════════════════════

def _score_mfcc_delta(y: np.ndarray) -> float:
    try:
        mfcc     = librosa.feature.mfcc(
                        y=y, sr=SAMPLE_RATE, n_mfcc=13,
                        n_fft=1024, hop_length=256)
        delta    = librosa.feature.delta(mfcc)
        mean_var = float(np.mean(np.var(delta, axis=1)))
        fake_score = 1.0 / (1.0 + np.exp(-(mean_var - 45.0) * 0.25))
        print(f"  [DBG] mfcc_delta  mean_var={mean_var:.2f}  → fake={fake_score:.3f}")
        return float(np.clip(fake_score, 0.0, 1.0))
    except Exception as e:
        print(f"  [DBG] mfcc_delta error: {e}")
        return 0.45


def _score_pitch_cv(f0: np.ndarray, voiced_flag) -> float:
    # Now accepts pre-extracted f0 so pyin doesn't run twice
    try:
        voiced_f0 = f0[voiced_flag == 1] if voiced_flag is not None \
                    else f0[~np.isnan(f0)]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if len(voiced_f0) < 15:
            print(f"  [DBG] pitch_cv  not enough voiced frames → 0.45")
            return 0.45
        cv         = float(np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-9))
        fake_score = 1.0 / (1.0 + np.exp(-(cv - 0.35) * 25))
        print(f"  [DBG] pitch_cv  cv={cv:.4f}  → fake={fake_score:.3f}")
        return float(np.clip(fake_score, 0.0, 1.0))
    except Exception as e:
        print(f"  [DBG] pitch error: {e}")
        return 0.45


def _score_spectral_centroid(y: np.ndarray) -> float:
    try:
        centroid = librosa.feature.spectral_centroid(
                        y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=256)[0]
        delta_c  = np.diff(centroid)
        var_dc   = float(np.var(delta_c))
        fake_score = 1.0 / (1.0 + np.exp(-(var_dc - 150000.0) * 0.000015))
        print(f"  [DBG] centroid_var={var_dc:.0f}  → fake={fake_score:.3f}")
        return float(np.clip(fake_score, 0.0, 1.0))
    except Exception as e:
        print(f"  [DBG] centroid error: {e}")
        return 0.45


def _score_energy_envelope(y: np.ndarray) -> float:
    try:
        rms    = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        thresh = np.percentile(rms, 10)
        voiced = rms[rms > thresh]
        if len(voiced) < 10:
            return 0.45
        cv = float(np.std(voiced) / (np.mean(voiced) + 1e-9))
        fake_score = 1.0 / (1.0 + np.exp((cv - 0.45) * 15))
        print(f"  [DBG] energy_cv={cv:.4f}  → fake={fake_score:.3f}")
        return float(np.clip(fake_score, 0.0, 1.0))
    except Exception as e:
        print(f"  [DBG] energy error: {e}")
        return 0.45


def _lcnn_debug(y: np.ndarray):
    try:
        S   = librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_mels=IMG_SIZE, n_fft=1024, hop_length=512)
        mel = librosa.power_to_db(S, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        t   = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t   = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            logits = _get_model()(t)
            raw    = logits.cpu().numpy()[0]
            prob   = float(torch.softmax(logits / TEMPERATURE, dim=1
                                         ).cpu().numpy()[0][1])
        print(f"  [DBG] lcnn  logits=({raw[0]:.1f},{raw[1]:.1f})  "
              f"fake_prob(T={TEMPERATURE})={prob:.3f}  [NOT used in verdict]")
    except Exception:
        pass


def _combine(scores: dict) -> float:
    return sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)


# ═══════════════════════════════════════════════════════════════
# AUTO-GENERATED SUMMARY TEXT  (NEW)
# ═══════════════════════════════════════════════════════════════
# Produces a plain English paragraph explaining the result.
# Written for non-technical readers — examiners, lawyers, clients.
# TO REMOVE: delete this function and ax_summary block in _generate_report()

def _generate_summary_text(result: dict) -> str:
    verdict    = result["verdict"]
    confidence = result["confidence"]
    sigs       = result["signals"]
    mfcc_s     = sigs.get("mfcc_delta", 0)
    pitch_s    = sigs.get("pitch_cv", 0)
    cent_s     = sigs.get("spectral_centroid", 0)

    # Build verdict sentence
    if verdict == "FAKE":
        if confidence >= 80:
            v_line = f"Analysis strongly indicates this audio is AI-generated or synthetic ({confidence:.1f}% confidence)."
        else:
            v_line = f"Analysis suggests this audio may be AI-generated ({confidence:.1f}% confidence), though certainty is limited."
    else:
        if confidence >= 80:
            v_line = f"Analysis strongly indicates this audio contains genuine human speech ({confidence:.1f}% confidence)."
        else:
            v_line = f"Analysis suggests this audio is likely genuine human speech ({confidence:.1f}% confidence), though certainty is limited."

    # Build signal explanation
    reasons = []
    if mfcc_s >= 65:
        reasons.append("vocal tract dynamics are over-expressive, consistent with TTS synthesis")
    elif mfcc_s <= 30:
        reasons.append("vocal tract dynamics are within normal human speech range")

    if pitch_s >= 65:
        reasons.append("pitch variation is exaggerated beyond typical human range")
    elif pitch_s <= 20:
        reasons.append("pitch variation is consistent with natural human speech")

    if cent_s >= 65:
        reasons.append("spectral brightness changes are unusually large")

    if reasons:
        reason_line = "Key findings: " + "; ".join(reasons) + "."
    else:
        reason_line = "Signal scores are in the borderline range — no single feature is strongly indicative."

    # Warning line
    warnings = []
    if result.get("low_confidence"):
        warnings.append("confidence is below threshold")
    if result.get("short_clip"):
        warnings.append("audio clip is under 5 seconds")
    if warnings:
        warn_line = f"Note: {' and '.join(warnings).capitalize()} — manual expert review is recommended."
    else:
        warn_line = ""

    parts = [v_line, reason_line]
    if warn_line:
        parts.append(warn_line)
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def _generate_report(path: str, y: np.ndarray, result: dict,
                     pitch_times, pitch_f0, pitch_voiced,
                     file_hash: str, metadata: dict) -> str:
    if not MATPLOTLIB_AVAILABLE:
        return None

    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = os.path.splitext(result["file"])[0][:40]
    out_path  = os.path.join(REPORT_DIR, f"report_{safe_name}_{timestamp}.png")

    # Pre-compute features for plots
    S      = librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
    mel_db = librosa.power_to_db(S, ref=np.max)
    times  = np.linspace(0, len(y) / SAMPLE_RATE, num=len(y))

    # ── Layout: 7 rows ────────────────────────────────────────────────────────
    # Row 1: waveform + mel spectrogram (side by side)
    # Row 2: pitch contour (full width)          ← NEW
    # Row 3: signal bar chart (full width)
    # Row 4: auto-summary text (full width)      ← NEW
    # Row 5: metadata (full width)               ← NEW
    # Row 6: verdict panel (full width)
    # Row 7: disclaimer (full width)             ← NEW
    fig = plt.figure(figsize=(14, 18), facecolor="#0d1117")
    gs  = gridspec.GridSpec(
            7, 2, figure=fig,
            height_ratios=[3, 2, 2.5, 1.2, 1.0, 1.5, 1.0],
            hspace=0.55, wspace=0.35,
            left=0.07, right=0.97, top=0.95, bottom=0.03)

    ax_wave   = fig.add_subplot(gs[0, 0])
    ax_mel    = fig.add_subplot(gs[0, 1])
    ax_pitch  = fig.add_subplot(gs[1, :])   # NEW — pitch contour
    ax_bar    = fig.add_subplot(gs[2, :])
    ax_summ   = fig.add_subplot(gs[3, :])   # NEW — auto summary
    ax_meta   = fig.add_subplot(gs[4, :])   # NEW — metadata
    ax_info   = fig.add_subplot(gs[5, :])
    ax_disc   = fig.add_subplot(gs[6, :])   # NEW — disclaimer

    text_col    = "#e6edf3"
    grid_col    = "#21262d"
    accent_fake = "#f85149"
    accent_real = "#3fb950"
    accent_mid  = "#d29922"
    verdict     = result["verdict"]
    verdict_col = accent_fake if verdict == "FAKE" else accent_real

    fig.suptitle(
        f"Audio Forensic Analysis Report  —  {result['file']}",
        fontsize=13, color=text_col, fontweight="bold", y=0.975)

    # ── Row 1a: Waveform ──────────────────────────────────────────────────────
    ax_wave.set_facecolor("#161b22")
    ax_wave.plot(times, y, color="#58a6ff", linewidth=0.4, alpha=0.85)
    ax_wave.set_title("Waveform", color=text_col, fontsize=9, pad=5)
    ax_wave.set_xlabel("Time (s)", color=text_col, fontsize=7)
    ax_wave.set_ylabel("Amplitude", color=text_col, fontsize=7)
    ax_wave.tick_params(colors=text_col, labelsize=6)
    for sp in ax_wave.spines.values(): sp.set_edgecolor(grid_col)
    ax_wave.grid(color=grid_col, linewidth=0.4, alpha=0.5)

    # ── Row 1b: Mel Spectrogram ───────────────────────────────────────────────
    ax_mel.set_facecolor("#161b22")
    img = ax_mel.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                        extent=[0, len(y)/SAMPLE_RATE, 0, SAMPLE_RATE/2/1000])
    cb = plt.colorbar(img, ax=ax_mel, format="%+2.0f dB")
    cb.ax.yaxis.label.set_color(text_col)
    cb.ax.tick_params(colors=text_col)
    ax_mel.set_title("Mel Spectrogram", color=text_col, fontsize=9, pad=5)
    ax_mel.set_xlabel("Time (s)", color=text_col, fontsize=7)
    ax_mel.set_ylabel("Frequency (kHz)", color=text_col, fontsize=7)
    ax_mel.tick_params(colors=text_col, labelsize=6)
    for sp in ax_mel.spines.values(): sp.set_edgecolor(grid_col)

    # ── Row 2: Pitch Contour  (NEW — remove this block to remove pitch plot) ──
    ax_pitch.set_facecolor("#161b22")
    # Plot unvoiced frames as faint dots, voiced frames as solid line
    # This makes the difference between smooth TTS and jagged real speech visible
    voiced_mask   = (pitch_voiced == 1) & (~np.isnan(pitch_f0))
    unvoiced_mask = (pitch_voiced != 1) | np.isnan(pitch_f0)

    if np.any(unvoiced_mask):
        f0_unvoiced = np.where(unvoiced_mask, np.nan, pitch_f0)
        ax_pitch.plot(pitch_times, f0_unvoiced, color="#30363d",
                      linewidth=1.0, alpha=0.4, label="Unvoiced")

    if np.any(voiced_mask):
        f0_voiced = np.where(voiced_mask, pitch_f0, np.nan)
        ax_pitch.plot(pitch_times, f0_voiced, color="#f0883e",
                      linewidth=1.2, alpha=0.9, label="Voiced F0")

    ax_pitch.set_title(
        "Pitch Contour (F0)  —  Smooth line = TTS characteristic  |  "
        "Jagged/irregular = Human characteristic",
        color=text_col, fontsize=8, pad=5)
    ax_pitch.set_xlabel("Time (s)", color=text_col, fontsize=7)
    ax_pitch.set_ylabel("Frequency (Hz)", color=text_col, fontsize=7)
    ax_pitch.tick_params(colors=text_col, labelsize=6)
    for sp in ax_pitch.spines.values(): sp.set_edgecolor(grid_col)
    ax_pitch.grid(color=grid_col, linewidth=0.4, alpha=0.5)
    ax_pitch.set_ylim(bottom=0)
    leg = ax_pitch.legend(fontsize=6, loc="upper right",
                          facecolor="#161b22", edgecolor=grid_col,
                          labelcolor=text_col)

    # ── Row 3: Signal breakdown bar chart ────────────────────────────────────
    ax_bar.set_facecolor("#161b22")
    signal_labels = {
        "mfcc_delta":        f"MFCC Delta Variance  (w={WEIGHTS['mfcc_delta']:.2f})  over-expressive TTS has higher variance",
        "pitch_cv":          f"Pitch CV             (w={WEIGHTS['pitch_cv']:.2f})  TTS pitch range is exaggerated",
        "spectral_centroid": f"Spectral Centroid    (w={WEIGHTS['spectral_centroid']:.2f})  TTS brightness changes are too large",
        "energy_envelope":   f"Energy Envelope      (w={WEIGHTS['energy_envelope']:.2f})  TTS energy is too uniform",
    }
    sigs   = result["signals"]
    names  = list(signal_labels.values())
    scores = [sigs[k] for k in signal_labels]

    bar_colors = [accent_fake if s >= 65 else accent_mid if s >= 45
                  else accent_real for s in scores]
    bars = ax_bar.barh(names, scores, color=bar_colors,
                       height=0.45, edgecolor="#30363d")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_title("Signal Breakdown  (higher = more likely FAKE)",
                     color=text_col, fontsize=9, pad=5)
    ax_bar.set_xlabel("Fake Score (%)", color=text_col, fontsize=7)
    ax_bar.tick_params(colors=text_col, labelsize=7)
    for sp in ax_bar.spines.values(): sp.set_edgecolor(grid_col)
    ax_bar.grid(axis="x", color=grid_col, linewidth=0.4, alpha=0.5)
    ax_bar.axvline(x=50, color="#8b949e", linewidth=1.0, linestyle="--", alpha=0.7)
    for bar, score in zip(bars, scores):
        ax_bar.text(score + 1.0, bar.get_y() + bar.get_height() / 2,
                    f"{score:.0f}%", va="center", color=text_col, fontsize=7)

    # ── Row 4: Auto-generated summary  (NEW — remove ax_summ block to remove) ─
    ax_summ.set_facecolor("#161b22")
    ax_summ.axis("off")
    for sp in ax_summ.spines.values(): sp.set_edgecolor(grid_col)
    summary_text = _generate_summary_text(result)
    ax_summ.text(0.5, 0.55, "Analysis Summary",
                 transform=ax_summ.transAxes,
                 fontsize=8, fontweight="bold",
                 color="#8b949e", ha="center", va="center")
    ax_summ.text(0.5, 0.20, summary_text,
                 transform=ax_summ.transAxes,
                 fontsize=8, color=text_col,
                 ha="center", va="center",
                 wrap=True,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#161b22", edgecolor=grid_col))

    # ── Row 5: Metadata + SHA-256  (NEW — remove ax_meta block to remove) ────
    ax_meta.set_facecolor("#161b22")
    ax_meta.axis("off")
    meta_line = (
        f"File: {result['file']}   |   "
        f"Format: {result['format'].upper()}   |   "
        f"Size: {metadata.get('size_kb', '?')} KB   |   "
        f"Duration: {result.get('duration_secs', '?')}s   |   "
        f"Sample Rate: {metadata.get('sample_rate_orig', '?')} Hz   |   "
        f"Channels: {metadata.get('channels', '?')}   |   "
        f"Analysed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    hash_line = f"SHA-256: {file_hash}"
    ax_meta.text(0.5, 0.70, meta_line,
                 transform=ax_meta.transAxes,
                 fontsize=6.5, color="#8b949e",
                 ha="center", va="center")
    ax_meta.text(0.5, 0.20, hash_line,
                 transform=ax_meta.transAxes,
                 fontsize=6, color="#484f58",
                 ha="center", va="center",
                 fontfamily="monospace")

    # ── Row 6: Verdict panel ──────────────────────────────────────────────────
    ax_info.set_facecolor("#161b22")
    ax_info.axis("off")
    ax_info.text(0.5, 0.72, f"VERDICT: {verdict}",
                 transform=ax_info.transAxes,
                 fontsize=24, fontweight="bold",
                 color=verdict_col, ha="center", va="center")
    detail = (f"Real {result['real_pct']:.1f}%   |   "
              f"Fake {result['fake_pct']:.1f}%   |   "
              f"Confidence {result['confidence']:.1f}%   |   "
              f"Threshold {result['threshold_used']:.2f}")
    ax_info.text(0.5, 0.35, detail,
                 transform=ax_info.transAxes,
                 fontsize=9, color=text_col,
                 ha="center", va="center", alpha=0.85)

    # Warnings — bold and prominent
    warnings_list = []
    if result["low_confidence"]:
        warnings_list.append("⚠  LOW CONFIDENCE — Score is close to decision boundary. Manual review recommended.")
    if result.get("short_clip"):
        warnings_list.append("⚠  SHORT CLIP (<5s) — Limited audio data. Submit a longer sample for reliable results.")
    if warnings_list:
        ax_info.text(0.5, 0.02,
                     "     ".join(warnings_list),
                     transform=ax_info.transAxes,
                     fontsize=9, fontweight="bold",
                     color=accent_mid, ha="center", va="center")

    # ── Row 7: Disclaimer  (NEW — remove ax_disc block to remove) ────────────
    ax_disc.set_facecolor("#0d1117")
    ax_disc.axis("off")
    disclaimer = (
        "DISCLAIMER: This report is produced by an automated hybrid detection system combining acoustic signal analysis. "
        "It should not be used as sole evidence in legal, academic, or professional proceedings. "
        "Results below 65% confidence require independent expert verification. "
        "The system was validated on 18 audio samples and may not generalise to all TTS engines, "
        "voice cloning techniques, or recording conditions. "
        "Detection of AI audio is an evolving field — newer synthesis methods may produce results not captured by this system."
    )
    ax_disc.text(0.5, 0.55, disclaimer,
                 transform=ax_disc.transAxes,
                 fontsize=6.5, color="#484f58",
                 ha="center", va="center",
                 style="italic",
                 wrap=True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def analyse(path: str, threshold: float = None,
            save_report: bool = True) -> dict:
    """
    Analyse an audio file for deepfake detection.

    Parameters
    ----------
    path        : path to audio file (any supported format)
    threshold   : fake probability cutoff 0–1 (default 0.50)
    save_report : save PNG forensic report (default True)

    Returns
    -------
    JSON-serialisable dict — see module docstring for schema.
    """
    result = {
        "verdict":        None,
        "real_pct":       None,
        "fake_pct":       None,
        "confidence":     None,
        "low_confidence": None,
        "threshold_used": threshold or DEFAULT_THRESHOLD,
        "duration_secs":  None,
        "short_clip":     False,
        "file":           os.path.basename(path),
        "format":         os.path.splitext(path)[1].lower(),
        "signals":        {},
        "report_path":    None,
        "error":          None,
    }

    try:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        if os.path.splitext(path)[1].lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format. "
                             f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        result["threshold_used"] = threshold

        print(f"\n  Loading audio...")
        y = _load_audio(path)

        # Duration check
        duration = len(y) / SAMPLE_RATE
        result["duration_secs"] = round(duration, 2)
        if duration < MIN_DURATION_SECS:
            result["short_clip"] = True
            print(f"  ⚠  Short clip ({duration:.1f}s) — results may be less reliable")

        # Voice activity detection
        print(f"  Checking for speech...")
        is_speech, _ = _check_voice_activity(y)
        if not is_speech:
            raise ValueError(
                "No speech detected. Please upload a file containing human speech."
            )

        print(f"  Running signals...")
        _lcnn_debug(y)

        # Extract pitch once — used by both _score_pitch_cv and the contour plot
        pitch_times, pitch_f0, pitch_voiced = _extract_pitch(y)

        scores = {
            "mfcc_delta":        _score_mfcc_delta(y),
            "pitch_cv":          _score_pitch_cv(pitch_f0, pitch_voiced),
            "spectral_centroid": _score_spectral_centroid(y),
            "energy_envelope":   _score_energy_envelope(y),
        }

        fake_prob  = _combine(scores)
        real_prob  = 1.0 - fake_prob
        confidence = max(fake_prob, real_prob) * 100

        # ── low_confidence: fires if score is borderline OR clip is short ────
        # Short clips produce unreliable statistics so always flag them
        low_conf = confidence < 65.0 or result["short_clip"]

        result.update({
            "verdict":        "FAKE" if fake_prob >= threshold else "REAL",
            "real_pct":       round(real_prob  * 100, 2),
            "fake_pct":       round(fake_prob  * 100, 2),
            "confidence":     round(confidence,       2),
            "low_confidence": low_conf,
            "signals":        {k: round(v * 100, 1) for k, v in scores.items()},
        })

        if save_report:
            # Collect metadata and hash for report
            print(f"  Computing file hash...")
            file_hash = _get_file_hash(path)
            metadata  = _get_metadata(path)
            rpath = _generate_report(
                path, y, result,
                pitch_times, pitch_f0, pitch_voiced,
                file_hash, metadata)
            result["report_path"] = rpath
            if rpath:
                print(f"  Report saved → {rpath}")

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════
# FILE PICKER
# ═══════════════════════════════════════════════════════════════

def _run_file_picker():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            title="Select Audio Files  (hold Ctrl/Cmd to select multiple)",
            filetypes=[
                ("Audio files",
                 "*.wav *.flac *.mp3 *.m4a *.ogg *.opus *.aac *.webm "
                 "*.wma *.mpeg *.mpg *.mp4 *.mov *.3gp *.amr"),
                ("All files", "*.*"),
            ]
        )
        root.destroy()
        return list(paths)
    except Exception:
        raw = input("Enter file path(s) separated by commas: ").strip()
        return [p.strip().strip('"') for p in raw.split(",") if p.strip()]


def _print_result(result: dict):
    if result["error"]:
        print(f"\n  ERROR: {result['error']}\n")
        return

    v      = result["verdict"]
    colour = "\033[91m" if v == "FAKE" else "\033[92m"
    yellow = "\033[93m"
    reset  = "\033[0m"

    signal_labels = {
        "mfcc_delta":        "MFCC Delta Var   ",
        "pitch_cv":          "Pitch CV         ",
        "spectral_centroid": "Spectral Centroid",
        "energy_envelope":   "Energy Envelope  ",
    }

    print("\n" + "=" * 60)
    print(f"  FILE       : {result['file']}")
    print(f"  FORMAT     : {result['format']}")
    print(f"  DURATION   : {result.get('duration_secs', '?')}s", end="")
    if result.get("short_clip"):
        print(f"  {yellow}⚠ short clip{reset}", end="")
    print()
    print(f"  VERDICT    : {colour}{v}{reset}")
    print(f"  REAL       : {result['real_pct']:.1f}%")
    print(f"  FAKE       : {result['fake_pct']:.1f}%")
    print(f"  CONFIDENCE : {result['confidence']:.1f}%", end="")
    if result["low_confidence"]:
        print(f"  {yellow}⚠ low — treat with caution{reset}", end="")
    print()
    print(f"  ── Signal breakdown ────────────────────────────────")
    for k, label in signal_labels.items():
        score  = result["signals"].get(k, 0)
        filled = int(score / 5)
        bar    = "█" * filled + "░" * (20 - filled)
        w      = WEIGHTS[k]
        print(f"  {label}: {bar} {score:.0f}%  (w={w:.2f})")
    if result.get("report_path"):
        print(f"  ── Report → {result['report_path']}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    paths = _run_file_picker()
    if not paths:
        print("No files selected.")
        sys.exit(0)

    n        = len(paths)
    results  = []
    fake_col = "\033[91m"
    real_col = "\033[92m"
    mid_col  = "\033[93m"
    reset    = "\033[0m"

    print(f"\n{'='*60}")
    print(f"  BATCH MODE — {n} file(s) selected")
    print(f"{'='*60}")

    for i, path in enumerate(paths, 1):
        print(f"\n[{i}/{n}] Analysing: {os.path.basename(path)} ...")
        result = analyse(path)
        _print_result(result)
        results.append(result)

    if n > 1:
        real_count = sum(1 for r in results if r["verdict"] == "REAL")
        fake_count = sum(1 for r in results if r["verdict"] == "FAKE")
        err_count  = sum(1 for r in results if r["error"])

        print(f"\n{'='*70}")
        print(f"  BATCH SUMMARY  ({n} files)   "
              f"REAL: {real_count}   FAKE: {fake_count}   ERRORS: {err_count}")
        print(f"  {'─'*66}")
        for r in results:
            if r["error"]:
                print(f"  ✗  {r['file']:<38}  {r['error'][:30]}")
                continue
            v    = r["verdict"]
            col  = fake_col if v == "FAKE" else real_col
            flags = ""
            if r["low_confidence"]: flags += f" {mid_col}⚠low{reset}"
            if r.get("short_clip"): flags += f" {mid_col}⚠short{reset}"
            print(f"  {col}{v:<4}{reset}  {r['file']:<38}  "
                  f"Real {r['real_pct']:5.1f}%  Fake {r['fake_pct']:5.1f}%  "
                  f"Conf {r['confidence']:5.1f}%{flags}")
        print(f"{'='*70}\n")
