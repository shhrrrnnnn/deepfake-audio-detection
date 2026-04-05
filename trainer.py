import os
import sys
import random
import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import torchaudio.transforms as T
from sklearn.metrics import roc_curve

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
from models.lcnn import LCNN
from utils.audio_handler import load_audio
from utils.features import extract_mel, get_cache_path, save_to_cache, load_from_cache

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
BATCH_SIZE  = 32
EPOCHS      = 30 
LR          = 3e-4

DATA_DIR    = r"C:\Users\shara\deepfake_audio\data\LA\LA"
OUTPUT_DIR  = r"C:\Users\shara\deepfake_audio\output"
CACHE_DIR   = r"C:\Users\shara\deepfake_audio\cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# --- ROBUST AUGMENTATION ---
def augment_for_real_world(y, sr):
    if random.random() > 0.5:
        snr = np.random.uniform(15, 30)
        pwr = np.mean(y**2) + 1e-8
        noise = np.random.normal(0, np.sqrt(pwr / 10**(snr/10)), len(y))
        y = (y + noise).astype(np.float32)

    if random.random() > 0.5:
        cutoff = random.randint(4000, 8000)
        from scipy.signal import butter, lfilter
        b, a = butter(4, cutoff/(sr/2), btype='low')
        y = lfilter(b, a, y).astype(np.float32)

    y = (y * np.random.uniform(0.6, 1.2)).astype(np.float32)
    return y

# --- LOSS FUNCTION ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

# --- DATASET ---
class ASVspoofDataset(Dataset):
    def __init__(self, audio_dir, label_dict, cache_dir, is_train=False):
        self.audio_dir = audio_dir
        self.cache_dir = cache_dir
        self.is_train = is_train
        os.makedirs(cache_dir, exist_ok=True)
        
        all_files = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]
        self.samples = [(f, label_dict[f.replace(".flac", "")]) 
                        for f in all_files if f.replace(".flac", "") in label_dict]

    def __len__(self): return len(self.samples)
    def get_labels(self): return [l for _, l in self.samples]

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        cache_path = get_cache_path(self.cache_dir, fname, augmented=self.is_train)
        cached = load_from_cache(cache_path)
        if cached:
            return torch.tensor(cached["mel"], dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        fpath = os.path.join(self.audio_dir, fname)
        y, sr = load_audio(fpath)
        if self.is_train:
            y = augment_for_real_world(y, sr)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    proto_dir = os.path.join(DATA_DIR, "ASVspoof2019_LA_cm_protocols")
    train_labels = load_protocol(os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt"))
    dev_labels   = load_protocol(os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt"))

    train_dir = os.path.join(DATA_DIR, "ASVspoof2019_LA_train", "flac")
    dev_dir   = os.path.join(DATA_DIR, "ASVspoof2019_LA_dev", "flac")
    
    train_ds = ASVspoofDataset(train_dir, train_labels, os.path.join(CACHE_DIR, "train"), is_train=True)
    val_ds   = ASVspoofDataset(dev_dir, dev_labels, os.path.join(CACHE_DIR, "val"), is_train=False)

    labels_list = train_ds.get_labels()
    class_counts = np.bincount(labels_list)
    weights = [1.0 / class_counts[l] for l in labels_list]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LCNN(num_classes=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_eer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for mel, labels in train_loader:
            mel, labels = mel.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(mel)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        v_targets, v_scores = [], []
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(DEVICE)
                outputs = model(mel)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                v_scores.extend(probs)
                v_targets.extend(labels.numpy())
        
        val_eer, threshold = compute_eer(v_targets, v_scores)
        scheduler.step()
        print(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Val EER: {val_eer*100:.2f}%")

        if val_eer < best_eer:
            best_eer = val_eer
            torch.save({"model_state": model.state_dict(), "eer": val_eer, "threshold": threshold}, 
                       os.path.join(OUTPUT_DIR, "best_lcnn.pt"))