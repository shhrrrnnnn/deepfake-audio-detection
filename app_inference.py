# Inference script — Updated with Auto-Standardization for .m4a support
# Usage: python app_inference.py path/to/your_recording.m4a

import os
import sys
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from models.lcnn import LCNN
from utils.audio_handler import load_audio
from utils.features import extract_mel, save_spectrogram_plot

OUTPUT_DIR     = r"C:\Users\shara\deepfake_audio\output"
MODEL_PATH     = os.path.join(OUTPUT_DIR, "best_lcnn.pt")
# Slightly lowered to be less "paranoid" for real-world testing
CONFIDENT_THRESHOLD = 0.90 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def standardize_audio(input_path):
    """
    Forces any audio format into a 16kHz Mono WAV.
    This eliminates 'codec noise' from .m4a files that causes False Fakes.
    """
    temp_wav = os.path.join(OUTPUT_DIR, "temp_inference_standardized.wav")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Standardizing audio: {os.path.basename(input_path)} -> 16kHz Mono WAV")
    
    # Load with librosa (automatically handles various formats via audioread)
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # Save as a clean, uncompressed WAV
    sf.write(temp_wav, y, 16000)
    return temp_wav

def interpret(fake_prob):
    real_prob = 1 - fake_prob
    if fake_prob >= CONFIDENT_THRESHOLD:
        return "FAKE", fake_prob, "High confidence — AI generated audio"
    elif real_prob >= CONFIDENT_THRESHOLD:
        return "REAL", real_prob, "High confidence — authentic human speech"
    else:
        # If it's between 10% and 90%, we call it suspicious
        return "SUSPICIOUS", max(fake_prob, real_prob), \
               "Inconclusive — potential domain mismatch or low quality"

def predict(audio_path, model_path=MODEL_PATH, save_plot=True):
    # 1. PRE-PROCESS: Convert .m4a/etc to clean .wav
    clean_audio_path = standardize_audio(audio_path)

    # 2. LOAD MODEL
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model      = LCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model loaded (trained EER: {checkpoint.get('eer', 0)*100:.2f}%)")

    # 3. LOAD STANDARDIZED AUDIO
    y, sr = load_audio(clean_audio_path)

    # 4. FEATURE EXTRACTION
    mel = extract_mel(y, sr)
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)

    # 5. INFERENCE
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                logits = model(mel_tensor)
        else:
            logits = model(mel_tensor)
        
        probs     = torch.softmax(logits, dim=1)[0]
        fake_prob = probs[1].item()
        real_prob = probs[0].item()

    verdict, confidence, message = interpret(fake_prob)

    # 6. VISUALIZATION
    if save_plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base       = os.path.splitext(os.path.basename(audio_path))[0]
        plot_path  = os.path.join(OUTPUT_DIR, f"{base}_analysis.png")
        save_spectrogram_plot(
            y, sr,
            output_path=plot_path,
            title=f"Analysis: {os.path.basename(audio_path)}",
            prediction=fake_prob,
            confidence=confidence,
            verdict=verdict
        )

    # 7. PRINT RESULT
    print("\n" + "=" * 50)
    print(f"  ORIGINAL FILE : {os.path.basename(audio_path)}")
    print(f"  VERDICT       : {verdict}")
    print(f"  REAL PROB     : {real_prob*100:.1f}%")
    print(f"  FAKE PROB     : {fake_prob*100:.1f}%")
    print(f"  MESSAGE       : {message}")
    print("=" * 50)

    # Clean up temp file
    if os.path.exists(clean_audio_path):
        os.remove(clean_audio_path)

    return {
        "verdict": verdict,
        "fake_prob": round(fake_prob, 4),
        "real_prob": round(real_prob, 4),
        "message": message
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio deepfake detection")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    predict(args.audio, args.model, save_plot=not args.no_plot)