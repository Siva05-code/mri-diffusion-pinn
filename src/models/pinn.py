"""
Physics-Informed Neural Network (PINN) Refinement Module for Accelerated MRI.

Architecture:
    Lightweight residual convolutional refinement network that refines coarse
    diffusion pre-reconstructions into high-fidelity MRI images:
        x_refined = x_coarse + Residual(x_coarse, [x_zf])

Scientific Context:
    In classical PINNs for physical fields, networks minimize known PDE differential
    operators. For MRI reconstruction, physics is governed by the MRI forward acquisition
    model (undersampled Fourier encoding y = F(x) + e), Fourier data consistency,
    and image-gradient smoothness constraints. The PINN refinement module learns physical
    corrections in image space while being trained and regularized with exact k-space
    data-consistency and gradient smoothness physics losses.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with dilation and group normalization."""
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.silu(out + residual)


class PINNRefinementNet(nn.Module):
    """
    Physics-Informed Neural Network (PINN) Refinement Network.

    Takes a coarse reconstruction (and optionally zero-filled input / physics guidance)
    and predicts a residual correction:
        refined = coarse + residual_network(coarse, [zf])

    Args:
        in_channels: Input channels (1 if coarse only, 2 if coarse + zero-filled).
        out_channels: Output channels (1 for single-coil magnitude MRI).
        base_channels: Channel capacity of the intermediate layers (default: 32).
        num_blocks: Number of residual processing blocks (default: 4).
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial feature projection
        self.entry_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels),
            nn.SiLU(),
        )

        # Multi-scale receptive field blocks using alternating dilations (1, 2, 4, 1)
        dilations = [1, 2, 4, 1]
        blocks = []
        for i in range(num_blocks):
            dil = dilations[i % len(dilations)]
            blocks.append(ResidualConvBlock(base_channels, dilation=dil))
        self.residual_blocks = nn.Sequential(*blocks)

        # Exit convolutional projection to single-channel residual
        self.exit_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        coarse: torch.Tensor,
        zero_filled: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of PINN refinement.

        Args:
            coarse: Coarse pre-reconstruction from diffusion [B, 1, H, W].
            zero_filled: Optional zero-filled initial image [B, 1, H, W].

        Returns:
            torch.Tensor: Refined image [B, 1, H, W].
        """
        while coarse.ndim > 4:
            coarse = coarse.squeeze(1)
        if coarse.ndim == 2:
            coarse = coarse.unsqueeze(0).unsqueeze(0)
        elif coarse.ndim == 3:
            coarse = coarse.unsqueeze(1)

        if zero_filled is not None:
            while zero_filled.ndim > 4:
                zero_filled = zero_filled.squeeze(1)
            if zero_filled.ndim == 2:
                zero_filled = zero_filled.unsqueeze(0).unsqueeze(0)
            elif zero_filled.ndim == 3:
                zero_filled = zero_filled.unsqueeze(1)

        if self.in_channels == 2:
            if zero_filled is None:
                # If 2 channels expected but no zf provided, concatenate coarse with itself
                inp = torch.cat([coarse, coarse], dim=1)
            else:
                inp = torch.cat([coarse, zero_filled], dim=1)
        else:
            inp = coarse

        features = self.entry_conv(inp)
        features = self.residual_blocks(features)
        residual = self.exit_conv(features)

        # Residual refinement
        refined = coarse + residual

        # Clamp output to valid normalized MRI intensity range [0, 1]
        return torch.clamp(refined, 0.0, 1.0)

