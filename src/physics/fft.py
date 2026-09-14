"""
PyTorch Fast Fourier Transform (FFT) & Inverse FFT Utilities for MRI.

Implements 2D Discrete Fourier Transforms (ifft2d, fft2d) with centered spatial and frequency coordinates.
"""

import torch


def fftshift(x: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    """Shift zero-frequency component to center of spectrum for PyTorch tensors."""
    for d in dim:
        x = torch.roll(x, shifts=x.shape[d] // 2, dims=d)
    return x


def ifftshift(x: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    """Inverse shift of zero-frequency component for PyTorch tensors."""
    for d in dim:
        x = torch.roll(x, shifts=(x.shape[d] + 1) // 2, dims=d)
    return x


def kspace_to_image_torch(kspace: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    """
    Transform k-space data to image space via centered 2D Inverse FFT.
    
    Args:
        kspace (torch.Tensor): Complex k-space tensor [..., H, W]
    Returns:
        torch.Tensor: Complex image domain tensor [..., H, W]
    """
    if not torch.is_complex(kspace):
        if kspace.shape[-1] == 2:
            kspace = torch.complex(kspace[..., 0], kspace[..., 1])
        else:
            kspace = kspace.to(torch.complex64)
            
    # Shift centered k-space, apply IFFT, and shift back
    kspace_shifted = ifftshift(kspace, dim=dim)
    image_domain = torch.fft.ifft2(kspace_shifted, dim=dim, norm="ortho")
    return fftshift(image_domain, dim=dim)


def image_to_kspace_torch(image: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    """
    Transform image domain data to k-space via centered 2D Forward FFT.
    
    Args:
        image (torch.Tensor): Complex image tensor [..., H, W]
    Returns:
        torch.Tensor: Complex k-space domain tensor [..., H, W]
    """
    if not torch.is_complex(image):
        image = image.to(torch.complex64)

    image_shifted = ifftshift(image, dim=dim)
    kspace_domain = torch.fft.fft2(image_shifted, dim=dim, norm="ortho")
    return fftshift(kspace_domain, dim=dim)

