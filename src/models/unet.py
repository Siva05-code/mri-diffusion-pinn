"""
2D U-Net Architecture for Accelerated MRI Reconstruction.

Implements a standard, lightweight 2D U-Net operating on single-channel magnitude
MRI images with skip connections, ConvTranspose2d upsampling, and dynamic spatial
alignment to support arbitrary 2D input dimensions.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard double convolution block:
    [Conv2d -> ReLU -> Conv2d -> ReLU]
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downscaling block: MaxPool2d(2) followed by ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Upscaling block:
    ConvTranspose2d upsampling -> dynamic padding -> skip-connection concatenation -> ConvBlock.
    Dynamic padding ensures exact spatial match with skip connections for arbitrary (H, W).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle potential spatial dimension mismatches due to odd input sizes
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight 2D U-Net for MRI Reconstruction.

    Args:
        in_channels (int): Number of input channels (default: 1 for magnitude MRI).
        out_channels (int): Number of output channels (default: 1 for magnitude MRI).
        base_channels (int): Number of channels in the initial layer (default: 32).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        c = base_channels
        # Encoder (Contracting Path)
        self.inc = ConvBlock(in_channels, c)            # e.g., 1 -> 32
        self.down1 = DownBlock(c, c * 2)               # 32 -> 64
        self.down2 = DownBlock(c * 2, c * 4)           # 64 -> 128
        self.down3 = DownBlock(c * 4, c * 8)           # 128 -> 256
        self.down4 = DownBlock(c * 8, c * 16)          # 256 -> 512 (Bottleneck)

        # Decoder (Expanding Path)
        self.up1 = UpBlock(c * 16, c * 8)              # 512 -> 256
        self.up2 = UpBlock(c * 8, c * 4)               # 256 -> 128
        self.up3 = UpBlock(c * 4, c * 2)               # 128 -> 64
        self.up4 = UpBlock(c * 2, c)                   # 64 -> 32

        # Final projection to target channels
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor [B, in_channels, H, W]
        Returns:
            torch.Tensor: Reconstructed tensor [B, out_channels, H, W]
        """
        # Save original spatial dimensions to guarantee exact spatial preservation
        orig_h, orig_w = x.shape[-2], x.shape[-1]

        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)

        # Final spatial safety check
        if out.shape[-2] != orig_h or out.shape[-1] != orig_w:
            out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        return out

