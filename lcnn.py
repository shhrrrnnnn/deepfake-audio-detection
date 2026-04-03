import torch
import torch.nn as nn
import torch.nn.functional as F

class MFM(nn.Module):
    def __init__(self, out_channels, mode=0):
        super().__init__()
        self.out_channels = out_channels
        self.mode = mode

    def forward(self, x):
        # Check if input is 4D (Conv) or 2D (Linear)
        if x.dim() == 4:
            out = x[:, :self.out_channels, :, :]
            rem = x[:, self.out_channels:, :, :]
        else:
            # For Linear layers (Batch, Features)
            out = x[:, :self.out_channels]
            rem = x[:, self.out_channels:]
        return torch.max(out, rem)

class LCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MFM(out_ch)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.mfm(self.bn(self.conv(x))))

class LCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            LCNNBlock(1,   32,  kernel=5, padding=2, pool=True), 
            LCNNBlock(32,  48,  pool=True),                     
            LCNNBlock(48,  64,  pool=True),                     
            LCNNBlock(64,  96),                                 
            LCNNBlock(96,  128, pool=True),                     
            LCNNBlock(128, 128),                                
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4)) 

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),         # Increased to 1024 so MFM can split it to 512
            nn.BatchNorm1d(1024),
            MFM(512),                             # Splits 1024 -> 512
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)           # Final output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)