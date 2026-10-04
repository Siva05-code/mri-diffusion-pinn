"""
Conditional U-Net used as the diffusion backbone.

Input channels (the 3-channel conditioning from the project design):

    x_t        noisy normalised residual
    zf         zero-filled reconstruction  (the condition)
    mask       sampling mask               (tells the net which lines are real)

Output: the diffusion target (v or eps) for the residual.

Each encoder stage pushes exactly one skip tensor and each decoder stage pops
exactly one, so the two sides stay in lock-step regardless of depth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(ch: int, maximum: int = 32) -> int:
    for g in (maximum, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class SelfAttention(nn.Module):
    """Single-head self-attention, used only at coarse resolutions."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        q = q.transpose(1, 2)                       # b, hw, c
        k = k                                       # b, c, hw
        v = v.transpose(1, 2)                       # b, hw, c
        attn = torch.softmax(q @ k / math.sqrt(c), dim=-1)   # b, hw, hw
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class ResBlock(nn.Module):
    """Time-conditioned residual block, optionally followed by attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        dropout: float = 0.0,
        attention: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = SelfAttention(out_ch) if attention else nn.Identity()

        # Residual branch starts at zero: a fresh block is the identity map.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.attn(h + self.skip(x))


class ConditionalDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base: int = 48,
        channel_mult=(1, 2, 4, 8),
        time_dim: int = 256,
        attention_levels=(2, 3),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        channel_mult = tuple(channel_mult)
        attention_levels = set(attention_levels)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        chans = [base * m for m in channel_mult]
        n_levels = len(chans)

        # ---------------- encoder ----------------
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_dims: list[int] = []

        ch = base
        for level, out_ch in enumerate(chans):
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(
                    ResBlock(ch, out_ch, time_dim, dropout, level in attention_levels)
                )
                ch = out_ch
            self.enc_blocks.append(stage)
            skip_dims.append(ch)

            is_last = level == n_levels - 1
            self.downsamples.append(
                nn.Identity() if is_last else nn.Conv2d(ch, ch, 3, stride=2, padding=1)
            )

        # ---------------- bottleneck ----------------
        self.mid1 = ResBlock(ch, ch, time_dim, dropout, attention=True)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout, attention=False)

        # ---------------- decoder ----------------
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for level in reversed(range(n_levels)):
            out_ch = chans[level]
            is_last = level == n_levels - 1
            self.upsamples.append(
                nn.Identity() if is_last else nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, ch, 3, padding=1),
                )
            )
            stage = nn.ModuleList()
            for i in range(num_res_blocks):
                in_ch = ch + skip_dims[level] if i == 0 else ch
                stage.append(
                    ResBlock(in_ch, out_ch, time_dim, dropout, level in attention_levels)
                )
                ch = out_ch
            self.dec_blocks.append(stage)

        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Zero-initialised output. Before any training the predicted residual is
        # exactly zero, so the full pipeline reproduces the zero-filled baseline
        # instead of random noise, and training can only move away from it.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)

        h = self.in_conv(torch.cat([x_t, condition, mask], dim=1))

        skips = []
        for stage, down in zip(self.enc_blocks, self.downsamples):
            for block in stage:
                h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for up, stage in zip(self.upsamples, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in stage:
                h = block(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
