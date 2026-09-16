"""
MRI Physics Data-Consistency Operations.

Implements differentiable k-space data consistency for accelerated MRI reconstruction:
    k_pred = FFT(x_hat)
    k_corrected = M * k_measured + (1 - M) * k_pred
    x_corrected = IFFT(k_corrected)

Preserves exact measured k-space trajectories while retaining network-synthesized unmeasured k-space lines.
Fully differentiable for end-to-end PyTorch backpropagation.
"""

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn

from src.physics.fft import kspace_to_image_torch, image_to_kspace_torch
from src.data.preprocessing import center_crop


def apply_data_consistency(
    x_image: torch.Tensor,
    k_measured: torch.Tensor,
    mask: torch.Tensor,
    phase: Optional[torch.Tensor] = None,
    return_complex: bool = False,
) -> torch.Tensor:
    """
    Applies differentiable MRI k-space data consistency.

    Replaces k-space points in the predicted image's Fourier transform with the
    actually acquired k-space measurements at sampled positions dictated by the mask.

    Args:
        x_image: Predicted reconstructed image.
                 Shape [B, 1, H, W], [B, H, W], or [H, W]. Real or complex.
        k_measured: Measured acquired k-space.
                    Shape [..., H_k, W_k]. Complex tensor.
        mask: Binary undersampling mask where 1 indicates sampled, 0 indicates unacquired.
              Broadcastable to k_measured.
        phase: Optional spatial phase tensor [..., H, W] to restore complex representation.
        return_complex: If True, returns complex image; otherwise returns magnitude.

    Returns:
        torch.Tensor: Data-consistent image tensor matching the input shape of x_image.
    """
    is_4d = (x_image.ndim == 4 and x_image.shape[1] == 1)

    # Convert image to [..., H, W] complex representation
    if is_4d:
        img_2d = x_image.squeeze(1)
    else:
        img_2d = x_image

    if not torch.is_complex(img_2d):
        if phase is not None:
            if phase.ndim == 4 and phase.shape[1] == 1:
                phase = phase.squeeze(1)
            img_complex = img_2d.to(torch.complex64) * torch.exp(1j * phase)
        else:
            img_complex = img_2d.to(torch.complex64)
    else:
        img_complex = img_2d

    if not torch.is_complex(k_measured):
        if k_measured.shape[-1] == 2:
            k_measured = torch.complex(k_measured[..., 0], k_measured[..., 1])
        else:
            k_measured = k_measured.to(torch.complex64)

    # Handle shape alignment if k_measured has different FOV (e.g. 640x368 vs 320x320)
    img_h, img_w = img_complex.shape[-2], img_complex.shape[-1]
    k_h, k_w = k_measured.shape[-2], k_measured.shape[-1]

    if (img_h, img_w) == (k_h, k_w):
        # Forward FFT to k-space domain
        k_pred = image_to_kspace_torch(img_complex)

        # Ensure consistent scale if raw k_measured has different unit scale from normalized image
        # If scale discrepancy is larger than 10x, rescale k_measured to match image k-space scale
        k_pred_norm = torch.norm(k_pred)
        k_meas_norm = torch.norm(k_measured)
        if k_pred_norm > 1e-4 and k_meas_norm > 1e-4 and (k_pred_norm / k_meas_norm > 10.0 or k_meas_norm / k_pred_norm > 10.0):
            scale_ratio = k_pred_norm / (k_meas_norm + 1e-12)
            k_meas_aligned = k_measured * scale_ratio
        else:
            k_meas_aligned = k_measured

        # Differentiable data consistency in k-space
        # k_corrected = M * k_measured + (1 - M) * k_pred
        mask = mask.to(k_pred.device, dtype=k_pred.real.dtype)
        k_corrected = mask * k_meas_aligned + (1.0 - mask) * k_pred

        # Inverse FFT back to image domain
        x_corrected = kspace_to_image_torch(k_corrected)
    else:
        # If image FOV is cropped (e.g. 320x320) and k_measured is full (640x368):
        pad_top = (k_h - img_h) // 2
        pad_bottom = k_h - img_h - pad_top
        pad_left = (k_w - img_w) // 2
        pad_right = k_w - img_w - pad_left

        img_padded = torch.nn.functional.pad(
            img_complex, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
        )
        k_pred = image_to_kspace_torch(img_padded)

        k_pred_norm = torch.norm(k_pred)
        k_meas_norm = torch.norm(k_measured)
        if k_pred_norm > 1e-4 and k_meas_norm > 1e-4 and (k_pred_norm / k_meas_norm > 10.0 or k_meas_norm / k_pred_norm > 10.0):
            scale_ratio = k_pred_norm / (k_meas_norm + 1e-12)
            k_meas_aligned = k_measured * scale_ratio
        else:
            k_meas_aligned = k_measured

        mask = mask.to(k_pred.device, dtype=k_pred.real.dtype)
        k_corrected = mask * k_meas_aligned + (1.0 - mask) * k_pred
        x_full = kspace_to_image_torch(k_corrected)

        # Crop back to target image dimensions
        x_corrected = center_crop(x_full, (img_h, img_w))

    if not return_complex:
        x_corrected = torch.abs(x_corrected)

    if is_4d and x_corrected.ndim == 3:
        x_corrected = x_corrected.unsqueeze(1)
    elif is_4d and x_corrected.ndim == 2:
        x_corrected = x_corrected.unsqueeze(0).unsqueeze(0)

    return x_corrected


class DataConsistencyLayer(nn.Module):
    """
    Differentiable Neural Network Layer for MRI Data Consistency.
    Can be inserted into any PyTorch reconstruction pipeline.
    """
    def __init__(self, return_complex: bool = False):
        super().__init__()
        self.return_complex = return_complex

    def forward(
        self,
        x_image: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_data_consistency(
            x_image=x_image,
            k_measured=k_measured,
            mask=mask,
            phase=phase,
            return_complex=self.return_complex,
        )
