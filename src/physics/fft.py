"""
Centered orthonormal Fourier operators and Fourier data consistency.

All operators here work on the 320x320 *reconstruction domain*, i.e. the
Fourier transform of the cropped, normalised magnitude image -- not on the
raw scanner k-space. This is stated explicitly so that the project does not
over-claim raw k-space consistency.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# Core transforms
# --------------------------------------------------------------------------
def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered, orthonormal 2-D FFT. Accepts real or complex input."""
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    k = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(k, dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered, orthonormal 2-D inverse FFT. Returns a complex tensor."""
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def image_to_kspace(x: torch.Tensor) -> torch.Tensor:
    return fft2c(x)


def kspace_to_image(k: torch.Tensor) -> torch.Tensor:
    return ifft2c(k)


# --------------------------------------------------------------------------
# Autocast safety
# --------------------------------------------------------------------------
# cuFFT has no float16 kernels for these sizes, and complex half arithmetic is
# not reliable. Every FFT in this project therefore runs in float32 with
# autocast explicitly disabled. Forgetting this is the single most common
# runtime crash when AMP is switched on.
def _fp32(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return x.to(torch.complex64)
    return x.float()


# --------------------------------------------------------------------------
# Data consistency
# --------------------------------------------------------------------------
def data_consistency(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Fourier data consistency.

        k_guided = mask * ( lam * k_measured + (1-lam) * k_pred )
                 + (1 - mask) * k_pred

    ``lam = 1`` gives hard replacement of the measured Fourier coefficients
    (the default). Values below 1 give a soft update, which is sometimes more
    stable at very high acceleration.

    The returned image is real: the masks produced by ``src.data.undersampling``
    are Hermitian-symmetric, so ``k_guided`` is the spectrum of a real image and
    the imaginary part is numerical noise at the 1e-7 level.
    """
    with torch.autocast(device_type=image.device.type, enabled=False):
        image = _fp32(image)
        measured_kspace = _fp32(measured_kspace)
        mask = _fp32(mask)

        pred_k = fft2c(image)
        merged = lam * measured_kspace + (1.0 - lam) * pred_k
        guided_k = mask * merged + (1.0 - mask) * pred_k
        return ifft2c(guided_k).real


def dc_error(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Relative data-consistency error:

        || mask * (FFT(recon) - k_measured) ||_2
        ---------------------------------------
                || mask * k_measured ||_2

    Computed per sample and averaged, so the value does not depend on batch size.
    """
    with torch.autocast(device_type=image.device.type, enabled=False):
        image = _fp32(image)
        measured_kspace = _fp32(measured_kspace)
        mask = _fp32(mask)

        pred_k = fft2c(image)
        diff = mask * (pred_k - measured_kspace)
        ref = mask * measured_kspace

        dims = tuple(range(1, diff.ndim))
        num = torch.linalg.vector_norm(diff, dim=dims)
        den = torch.linalg.vector_norm(ref, dim=dims).clamp_min(eps)
        return (num / den).mean()
