"""
MRI Physics Data-Consistency Operations.

Implements differentiable k-space data consistency for accelerated MRI reconstruction:
    k_pred = ForwardOperator(x_hat)
    k_corrected = M * y_measured + (1 - M) * k_pred
    x_corrected = AdjointOperator(k_corrected)

Preserves exact measured k-space trajectories while retaining network-synthesized unmeasured k-space lines.
Fully differentiable for end-to-end PyTorch backpropagation.
"""

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn

from src.physics.forward_operator import MRIForwardOperator
from src.physics.fft import kspace_to_image_torch, image_to_kspace_torch
from src.data.preprocessing import center_crop


_default_operator = MRIForwardOperator()


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

    Handles both:
    1. Raw acquisition domain (e.g. k_measured [..., 640, 368] with x_image [..., 320, 320])
    2. Same-domain Fourier consistency (when shapes match exactly).

    Args:
        x_image: Predicted reconstructed image [..., H_img, W_img].
        k_measured: Measured acquired k-space [..., H_k, W_k] (complex).
        mask: Binary undersampling mask where 1 indicates sampled, 0 indicates unacquired.
        phase: Optional spatial phase tensor to restore complex representation.
        return_complex: If True, returns complex image; otherwise returns real component.

    Returns:
        torch.Tensor: Data-consistent image tensor matching the spatial shape of x_image.
    """
    is_4d = (x_image.ndim == 4 and x_image.shape[1] == 1)
    if is_4d:
        img = x_image.squeeze(1)
        phase_in = phase.squeeze(1) if (phase is not None and phase.ndim == 4 and phase.shape[1] == 1) else phase
    else:
        img = x_image
        phase_in = phase

    img_h, img_w = img.shape[-2], img.shape[-1]
    k_h, k_w = k_measured.shape[-2], k_measured.shape[-1]

    if (img_h, img_w) != (k_h, k_w):
        # Raw acquisition domain consistency via forward operator
        res = _default_operator.data_consistency(
            x_crop=img,
            y_raw=k_measured,
            mask_raw=mask,
            phase=phase_in,
            return_complex=return_complex,
        )
    else:
        # Same-domain Fourier consistency
        if not torch.is_complex(img):
            if phase_in is not None:
                img_c = img.to(torch.complex64) * torch.exp(1j * phase_in.to(img.device))
            else:
                img_c = img.to(torch.complex64)
        else:
            img_c = img

        if not torch.is_complex(k_measured):
            if k_measured.shape[-1] == 2:
                k_measured = torch.complex(k_measured[..., 0], k_measured[..., 1])
            else:
                k_measured = k_measured.to(torch.complex64)

        k_pred = image_to_kspace_torch(img_c)
        mask = mask.to(device=k_pred.device, dtype=k_pred.real.dtype)
        k_corrected = mask * k_measured + (1.0 - mask) * k_pred
        x_rec = kspace_to_image_torch(k_corrected)
        res = x_rec if return_complex else torch.real(x_rec)

    if is_4d:
        if res.ndim == 3:
            res = res.unsqueeze(1)
        elif res.ndim == 2:
            res = res.unsqueeze(0).unsqueeze(0)

    return res


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
