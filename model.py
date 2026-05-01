# U-Net architecture built from scratch in PyTorch

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive (Conv → BatchNorm → ReLU) blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool. Returns both the skip feature map and the pooled output."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """Upsample → concat skip connection → DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Halves channels before concat so that after concat (in_channels) it stays manageable
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle odd spatial sizes: pad x to match skip dimensions
        if x.shape != skip.shape:
            x = nn.functional.pad(
                x,
                [0, skip.shape[3] - x.shape[3],
                 0, skip.shape[2] - x.shape[2]],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for binary brain tumor segmentation.

    Input:  (B, 3, H, W)   — RGB MRI slice
    Output: (B, 1, H, W)   — raw logits (apply sigmoid explicitly for inference)

    Encoder channel progression: 3 → 64 → 128 → 256 → 512
    Bottleneck:                        512 → 1024
    Decoder channel progression:  1024 → 512 → 256 → 128 → 64
    Head:                           64 → 1  (1×1 conv + sigmoid)
    """

    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(ch, feat))
            ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (reversed features)
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for feat in reversed(features):
            self.decoders.append(DecoderBlock(ch, feat))
            ch = feat

        # Final 1×1 classification head
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoding path
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding path (skips in reverse order)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)  # raw logits — apply sigmoid explicitly during eval


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return trainable
