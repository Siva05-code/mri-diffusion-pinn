import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2

        emb = torch.log(torch.tensor(10000.0, device=t.device))
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-emb / max(half - 1, 1))
        )

        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        time = self.time_proj(t_emb)[:, :, None, None]
        h = h + time

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h


class ConditionalDiffusionUNet(nn.Module):
    """
    Lightweight conditional diffusion network.

    Input:
        x_t       : noisy target image [B,1,H,W]
        condition : zero-filled reconstruction [B,1,H,W]
        t         : diffusion timestep [B]

    Output:
        predicted noise [B,1,H,W]
    """

    def __init__(self, base_channels=32, time_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.enc1 = ConvBlock(2, base_channels, time_dim)
        self.enc2 = ConvBlock(
            base_channels,
            base_channels * 2,
            time_dim
        )
        self.enc3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            time_dim
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 4,
            2,
            stride=2
        )

        self.dec3 = ConvBlock(
            base_channels * 8,
            base_channels * 2,
            time_dim
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels * 2,
            2,
            stride=2
        )

        self.dec2 = ConvBlock(
            base_channels * 4,
            base_channels,
            time_dim
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels,
            base_channels,
            2,
            stride=2
        )

        self.dec1 = ConvBlock(
            base_channels * 2,
            base_channels,
            time_dim
        )

        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x_t, condition, t):

        t_emb = self.time_embedding(t)

        x = torch.cat([x_t, condition], dim=1)

        e1 = self.enc1(x, t_emb)

        e2 = self.enc2(
            self.pool(e1),
            t_emb
        )

        e3 = self.enc3(
            self.pool(e2),
            t_emb
        )

        b = self.bottleneck(
            self.pool(e3),
            t_emb
        )

        d3 = self.up3(b)

        d3 = self._match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3, t_emb)

        d2 = self.up2(d3)

        d2 = self._match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)

        d1 = self.up1(d2)

        d1 = self._match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb)

        return self.out(d1)

    @staticmethod
    def _match_size(x, reference):
        if x.shape[-2:] == reference.shape[-2:]:
            return x

        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False
        )