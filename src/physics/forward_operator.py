"""
MRI Forward and Adjoint Physics Operators.

Defines the mathematical forward model and adjoint operations connecting the clinical
reconstruction domain (e.g. 320x320 FOV) and the raw scanner acquisition domain (e.g. 640x368 k-space).

Mathematical Model:
    Let x ∈ C^{H_recon x W_recon} be the true underlying image object in the target FOV.
    The forward acquisition operator A: C^{H_recon x W_recon} -> C^{H_raw x W_raw} is:
        A(x; φ) = M_raw ⊙ F_raw( Z( x · exp(i φ) ) )
    where:
        - φ is the spatial phase map (estimated from zero-filled central k-space or complex acquisition).
        - Z: C^{H_recon x W_recon} -> C^{H_raw x W_raw} is the zero-padding operator (formal adjoint of center-cropping).
        - F_raw is the 2D discrete centered Fourier transform on the raw acquisition grid.
        - M_raw ∈ {0, 1}^{H_raw x W_raw} is the binary sampling mask.

    The adjoint reconstruction operator A*: C^{H_raw x W_raw} -> C^{H_recon x W_recon} is:
        A*(y) = C( F_raw^{-1}(y) )
    where:
        - F_raw^{-1} is the 2D discrete centered Inverse Fourier transform.
        - C: C^{H_raw x W_raw} -> C^{H_recon x W_recon} is the center-cropping operator.

Scientific Note on Exactness:
    Because center-cropping C discards energy outside the target FOV (air, wrap-around oversampling margin),
    its formal adjoint Z is an approximation: C(Z(x)) = x, but Z(C(x_full)) ≠ x_full if x_full has non-zero
    content outside the crop window. This operator explicitly documents this adjoint relationship.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch
from src.data.preprocessing import center_crop


class MRIForwardOperator(nn.Module):
    """
    Differentiable forward, adjoint, and data-consistency operator for MRI reconstruction.
    """
    def __init__(
        self,
        raw_shape: Tuple[int, int] = (640, 368),
        recon_shape: Tuple[int, int] = (320, 320),
    ):
        super().__init__()
        self.raw_shape = raw_shape
        self.recon_shape = recon_shape

    def forward_kspace(
        self,
        x_crop: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Maps a cropped reconstruction image [..., H_recon, W_recon] to the raw frequency domain [..., H_raw, W_raw].

        Args:
            x_crop: Image tensor [..., H_recon, W_recon] (real or complex).
            phase: Optional phase tensor [..., H_recon, W_recon] to restore complex representation.
            target_shape: Destination raw k-space shape (defaults to self.raw_shape).

        Returns:
            torch.Tensor: Complex k-space tensor [..., H_raw, W_raw].
        """
        raw_h, raw_w = target_shape or self.raw_shape
        img_h, img_w = x_crop.shape[-2], x_crop.shape[-1]

        # 1. Restore complex representation using phase if provided
        if not torch.is_complex(x_crop):
            if phase is not None:
                img_complex = x_crop.to(torch.complex64) * torch.exp(1j * phase.to(x_crop.device))
            else:
                img_complex = x_crop.to(torch.complex64)
        else:
            img_complex = x_crop

        # 2. Zero-pad to raw acquisition FOV if shapes differ
        if (img_h, img_w) != (raw_h, raw_w):
            pad_top = (raw_h - img_h) // 2
            pad_bottom = raw_h - img_h - pad_top
            pad_left = (raw_w - img_w) // 2
            pad_right = raw_w - img_w - pad_left
            img_padded = F.pad(
                img_complex,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            )
        else:
            img_padded = img_complex

        # 3. Centered 2D Forward Fourier Transform
        return image_to_kspace_torch(img_padded)

    def adjoint(
        self,
        y_raw: torch.Tensor,
        crop_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Maps raw k-space measurements [..., H_raw, W_raw] to the cropped reconstruction image domain [..., H_recon, W_recon].

        Args:
            y_raw: Complex k-space tensor [..., H_raw, W_raw].
            crop_shape: Destination spatial shape (defaults to self.recon_shape).

        Returns:
            torch.Tensor: Complex image tensor [..., H_recon, W_recon].
        """
        dest_shape = crop_shape or self.recon_shape
        if not torch.is_complex(y_raw):
            if y_raw.shape[-1] == 2:
                y_raw = torch.complex(y_raw[..., 0], y_raw[..., 1])
            else:
                y_raw = y_raw.to(torch.complex64)

        img_full = kspace_to_image_torch(y_raw)
        return center_crop(img_full, dest_shape)

    def data_consistency(
        self,
        x_crop: torch.Tensor,
        y_raw: torch.Tensor,
        mask_raw: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        return_complex: bool = False,
    ) -> torch.Tensor:
        """
        Applies exact physical measurement consistency in the raw acquisition domain:
            k_pred = F_raw( Z( x_crop · exp(i φ) ) )
            k_consistent = mask_raw · y_raw + (1 - mask_raw) · k_pred
            x_consistent = C( F_raw^{-1}( k_consistent ) )

        Args:
            x_crop: Reconstructed image [..., H_recon, W_recon].
            y_raw: Measured raw k-space [..., H_raw, W_raw] (complex).
            mask_raw: Binary acquisition mask [..., H_raw, W_raw] or broadcastable.
            phase: Optional phase map [..., H_recon, W_recon].
            return_complex: Whether to return complex or real magnitude tensor.

        Returns:
            torch.Tensor: Data-consistent image matching spatial shape of x_crop.
        """
        # 1. Map predicted reconstruction to raw k-space
        raw_shape = (y_raw.shape[-2], y_raw.shape[-1])
        k_pred = self.forward_kspace(x_crop, phase=phase, target_shape=raw_shape)

        if not torch.is_complex(y_raw):
            if y_raw.shape[-1] == 2:
                y_raw = torch.complex(y_raw[..., 0], y_raw[..., 1])
            else:
                y_raw = y_raw.to(torch.complex64)

        # 2. Measurement replacement in k-space
        mask = mask_raw.to(device=k_pred.device, dtype=k_pred.real.dtype)
        k_consistent = mask * y_raw + (1.0 - mask) * k_pred

        # 3. Inverse transform and center crop
        x_full = kspace_to_image_torch(k_consistent)
        x_consistent = center_crop(x_full, (x_crop.shape[-2], x_crop.shape[-1]))

        if not return_complex:
            return torch.real(x_consistent)
        return x_consistent

