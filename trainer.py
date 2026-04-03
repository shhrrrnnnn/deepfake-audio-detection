# Updated trainer.py — Fixed ArgParsing and Validation Robustness
import os
import sys
import random
import gc
import warnings
import argparse # Added
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from models.lcnn import LCNN
from utils.audio_handler import load_audio
from utils.features import extract_mel, extract_lfcc, get_cache_path, \
                           save_to_cache, load_from_cache

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
SAMPLE_RATE  = 16000
# Defaults (Can be overridden by command line)
BATCH_SIZE   = 32
EPOCHS       = 30 
LR           = 3e-4

DATA_DIR    = r"C:\Users\shara\deepfake_audio\data\LA\LA"
OUTPUT_DIR  = r"C:\Users\shara\deepfake_audio\output"
CACHE_DIR   = r"C:\Users\shara\deepfake_audio\cache"

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# ── Loss & Augmentation (Keeping your existing logic) ─────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def augment(y, sr):
    # Added Gaussian Noise specifically to help with your laptop mic issue
    def noise(y):
        snr = np.random.uniform(20, 40)
        pwr = np.mean(y**2) + 1e-8
        n = np.random.normal(0, np.sqrt(pwr / 10**(snr/10)), len(y))
        return (y + n).astype(np.float32)

    def vol(y):
        return (y * np.random.uniform(0.8, 1.2)).astype(np.float32)

    # Use a simpler, more targeted augmentation for real-world robustness
    fns = [noise, vol]
    for fn in fns:
        y = fn(y)
    return y

# ── Dataset ───────────────────────────────────────────────────────────────────
class ASVspoofDataset(Dataset):
    def __init__(self, audio_dir, label_dict, cache_dir, augment_fn=None):
        self.audio_dir  = audio_dir
        self.cache_dir  = cache_dir
        self.augment_fn = augment_fn
        os.makedirs(cache_dir, exist_ok=True)
        all_files    = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]
        self.samples = [(f, label_dict[f.replace(".flac", "")]) for f in all_files if f.replace(".flac", "") in label_dict]
        
        real = sum(1 for _, l in self.samples if l == 0)
        fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  Dataset: {len(self.samples)} samples (Real={real}, Fake={fake})")

    def __len__(self): return len(self.samples)
    def get_labels(self): return [l for _, l in self.samples]

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        aug = self.augment_fn is not None
        cache_path = get_cache_path(self.cache_dir, fname, augmented=aug)
        
        cached = load_from_cache(cache_path)
        if cached:
            return cached["mel"], torch.tensor(label, dtype=torch.long)

        fpath = os.path.join(self.audio_dir, fname)
        y, sr = load_audio(fpath)
        if self.augment_fn:
            y = self.augment_fn(y, sr)

        mel = extract_mel(y, sr)
        save_to_cache(cache_path, mel, None)
        return torch.tensor(mel, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])

def load_protocol(path):
    d = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: d[p[1]] = 0 if p[-1] == "bonafide" else 1
    return d

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Protocols
    proto_dir = os.path.join(DATA_DIR, "ASVspoof2019_LA_cm_protocols")
    train_labels = load_protocol(os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt"))
    dev_labels   = load_protocol(os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt"))

    # Build Datasets
    train_dir = os.path.join(DATA_DIR, "ASVspoof2019_LA_train", "flac")
    dev_dir   = os.path.join(DATA_DIR, "ASVspoof2019_LA_dev", "flac")
    
    print("\nBuilding Training Dataset (with Augmentation)...")
    train_ds = ASVspoofDataset(train_dir, train_labels, os.path.join(CACHE_DIR, "train"), augment_fn=augment)
    print("Building Validation Dataset...")
    val_ds   = ASVspoofDataset(dev_dir, dev_labels, os.path.join(CACHE_DIR, "val"))

    # Balanced Sampler for Training
    labels_list = train_ds.get_labels()
    class_counts = np.bincount(labels_list)
    sample_weights = [1.0 / class_counts[l] for l in labels_list]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model & Setup
    model = LCNN(num_classes=2, dropout=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_eer = float("inf")
    print(f"\nStarting training on {device} for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss = 0.0
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(mel)
                loss = criterion(out, labels)
            scaler.scale(loss).backward() # 1. Compute gradients
            scaler.step(optimizer)        # 2. Update weights using the optimizer
            scaler.update()               # 3. Refresh the scaler for the next batch)
            t_loss += loss.item()

        # Evaluation
        model.eval()
        v_targets, v_probs = [], []
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                out = model(mel)
                v_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                v_targets.extend(labels.numpy())
        
        val_eer, thresh = compute_eer(v_targets, v_probs)
        scheduler.step()

        print(f"Epoch {epoch:02d} | Train Loss: {t_loss/len(train_loader):.4f} | Val EER: {val_eer*100:.2f}%")

        if val_eer < best_eer:
            best_eer = val_eer
            torch.save({"model_state": model.state_dict(), "eer": val_eer, "threshold": thresh}, 
                       os.path.join(args.output_dir, "best_lcnn.pt"))
            print("  New Best Model Saved!")

    print(f"\nTraining Complete. Best EER: {best_eer*100:.2f}%")