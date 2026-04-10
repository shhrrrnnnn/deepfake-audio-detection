import sys
import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
TARGET_SR = 16000
DURATION = 5  # seconds


# -----------------------------
# LOAD AUDIO
# -----------------------------
def load_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    max_len = TARGET_SR * DURATION
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    return y


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(y):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y))

    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13)
    mfcc_var = np.var(mfcc)

    return {
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "mfcc_var": mfcc_var,
    }


# -----------------------------
# SCORING LOGIC
# -----------------------------
def compute_score(features):
    zcr = features["zcr"] * 100
    centroid = features["centroid"] / 1000
    bandwidth = features["bandwidth"] / 1000
    mfcc_var = features["mfcc_var"] / 50   # scaled better

    score = (
        (-1.2 * zcr) +
        (-0.8 * centroid) +
        (-0.6 * bandwidth) +
        (1.8 * mfcc_var)
    )

    return score


# -----------------------------
# CONFIDENCE (FIXED SCALING)
# -----------------------------
def compute_confidence(score):
    # clamp score range for stability
    score = max(min(score, 10), -10)

    confidence = 1 / (1 + np.exp(-score / 2))  # smoother curve
    return round(confidence * 100, 2)


# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify(confidence):
    if confidence > 65:
        return "REAL"
    elif confidence < 35:
        return "FAKE"
    else:
        return "UNCERTAIN"


# -----------------------------
# REASONING
# -----------------------------
def get_reason(features):
    if features["mfcc_var"] > 50:
        return "Rich vocal variation → likely human speech"
    elif features["zcr"] > 0.08:
        return "High signal noise → possible synthetic audio"
    else:
        return "Mixed characteristics → uncertain"


# -----------------------------
# MAIN
# -----------------------------
def analyze_audio(file_path):
    y = load_audio(file_path)
    features = extract_features(y)

    score = compute_score(features)
    confidence = compute_confidence(score)
    verdict = classify(confidence)
    reason = get_reason(features)

    print("\n=======================================================")
    print("  FORENSIC AUDIO ANALYSIS (FINAL)")
    print("-------------------------------------------------------")
    print(f"  FILE        : {os.path.basename(file_path)}")
    print(f"  VERDICT     : {verdict}")
    print(f"  CONFIDENCE  : {confidence}%")
    print(f"  SCORE       : {round(score, 4)}")
    print(f"  ZCR         : {round(features['zcr'], 5)}")
    print(f"  CENTROID    : {round(features['centroid'], 2)}")
    print(f"  BANDWIDTH   : {round(features['bandwidth'], 2)}")
    print(f"  MFCC VAR    : {round(features['mfcc_var'], 2)}")
    print(f"  INSIGHT     : {reason}")
    print("=======================================================\n")


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app_inference.py <audio_file>")
        sys.exit(1)

    analyze_audio(sys.argv[1])
