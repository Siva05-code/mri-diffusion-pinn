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

    def __init__(self, in_channels=3, base_channels=32, time_dim=128):
        super().__init__()
        self.in_channels = in_channels

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.enc1 = ConvBlock(in_channels, base_channels, time_dim)
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

    def forward(self, x_t, condition, mask=None, t=None):
        # Support positional calling convention where 3rd argument was t: model(x_t, condition, t)
        if t is None:
            if mask is not None and mask.ndim == 1 and mask.dtype in (torch.long, torch.int64, torch.int32):
                t = mask
                mask = None
            elif mask is None:
                raise ValueError("Diffusion timestep 't' must be provided.")

        while x_t.ndim > 4:
            x_t = x_t.squeeze(1)
        while condition.ndim > 4:
            condition = condition.squeeze(1)
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0).unsqueeze(0)
        elif x_t.ndim == 3:
            x_t = x_t.unsqueeze(1)
        if condition.ndim == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)
        elif condition.ndim == 3:
            condition = condition.unsqueeze(1)

        t_emb = self.time_embedding(t)

        x = torch.cat([x_t, condition], dim=1)
        if self.in_channels == 3:
            if mask is None:
                mask_t = torch.ones_like(condition)
            else:
                mask_t = mask
                while mask_t.ndim > 4:
                    mask_t = mask_t.squeeze(1)
                if mask_t.ndim == 2:
                    mask_t = mask_t.unsqueeze(0).unsqueeze(0)
                elif mask_t.ndim == 3:
                    if mask_t.shape[0] == 1 and mask_t.shape[1] == 1:
                        mask_t = mask_t.unsqueeze(0)
                    else:
                        mask_t = mask_t.unsqueeze(1)
                mask_t = mask_t.to(device=condition.device, dtype=condition.dtype)
                if mask_t.shape[-1] > condition.shape[-1]:
                    w_from = (mask_t.shape[-1] - condition.shape[-1]) // 2
                    mask_t = mask_t[..., w_from : w_from + condition.shape[-1]]
                if mask_t.shape[-2] > condition.shape[-2]:
                    h_from = (mask_t.shape[-2] - condition.shape[-2]) // 2
                    mask_t = mask_t[..., h_from : h_from + condition.shape[-2], :]
                if mask_t.shape != condition.shape:
                    mask_t = mask_t.expand_as(condition)
            x = torch.cat([x_t, condition, mask_t], dim=1)
        elif self.in_channels == 2:
            x = torch.cat([x_t, condition], dim=1)
        else:
            raise ValueError(f"Unsupported in_channels: {self.in_channels}")

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

    def load_compatible_state_dict(self, state_dict: dict, migrate_2ch: bool = False):
        """
        Loads state dict with explicit validation of first-layer channel counts.
        Rejects incompatible checkpoints unless migrate_2ch=True is explicitly specified.

        Args:
            state_dict: State dictionary containing model weights.
            migrate_2ch: If True and checkpoint has 2 input channels while model has 3,
                         migrates the weights by zero-initializing the mask channel.
        """
        first_layer_key = "enc1.conv1.weight"
        if first_layer_key in state_dict:
            ckpt_in_ch = state_dict[first_layer_key].shape[1]
            model_in_ch = self.enc1.conv1.weight.shape[1]
            if ckpt_in_ch != model_in_ch:
                if ckpt_in_ch == 2 and model_in_ch == 3:
                    if not migrate_2ch:
                        raise RuntimeError(
                            f"Checkpoint incompatibility: Checkpoint '{first_layer_key}' has {ckpt_in_ch} input channels, "
                            f"but model requires {model_in_ch} channels ([x_t, condition, mask]). "
                            f"To explicitly migrate legacy 2-channel weights by zero-initializing the mask channel, "
                            f"call model.load_compatible_state_dict(state_dict, migrate_2ch=True)."
                        )
                    old_weight = state_dict[first_layer_key]
                    new_weight = torch.zeros(
                        (old_weight.shape[0], 3, old_weight.shape[2], old_weight.shape[3]),
                        dtype=old_weight.dtype,
                        device=old_weight.device,
                    )
                    new_weight[:, :2, :, :] = old_weight
                    state_dict = dict(state_dict)
                    state_dict[first_layer_key] = new_weight
                    print("[CHECKPOINT MIGRATION] Migrated legacy 2-channel first convolution weights to 3 channels. Mask channel initialized with zeros.")
                else:
                    raise RuntimeError(
                        f"Cannot migrate checkpoint: incompatible channel count ({ckpt_in_ch} vs {model_in_ch})."
                    )
        return self.load_state_dict(state_dict, strict=True)