"""
Stage-2 physics-guided refinement network.

This is a small, deterministic U-Net (no diffusion, no timestep) that cleans up
the output of the diffusion sampler, followed by a final data-consistency
projection:

    diffusion output -> refinement U-Net -> + correction -> data consistency

It is deliberately *not* part of stage 1. Train it only once
"diffusion + DC" already beats the zero-filled baseline, otherwise it just
learns to undo a broken sampler and the ablation becomes uninterpretable.

The output convolution is zero-initialised, so an untrained refinement network
is exactly the identity and can only improve on its input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.fft import data_consistency


def _groups(ch: int, maximum: int = 32) -> int:
    for g in (maximum, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.body(x)


class RefinementUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 32, depth: int = 3):
        super().__init__()
        self.depth = depth

        chans = [base * (2 ** i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        ch = in_channels
        for c in chans:
            self.encoders.append(ConvBlock(ch, c))
            ch = c

        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(ch, c, 2, stride=2))
            self.decoders.append(ConvBlock(c * 2, c))
            ch = c

        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        image: torch.Tensor,
        zf: torch.Tensor,
        mask: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        apply_dc: bool = True,
    ) -> torch.Tensor:
        h = torch.cat([image, zf, mask], dim=1)

        skips = []
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)

        h = self.bottleneck(h)

        for up, dec in zip(self.upconvs, self.decoders):
            h = up(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = dec(torch.cat([h, skip], dim=1))

        refined = (image + self.out_conv(h)).clamp(0.0, 1.0)

        if apply_dc and measured_kspace is not None:
            # No clamp after the projection: it would undo the measured lines.
            refined = data_consistency(refined, measured_kspace, mask)

        return refined
