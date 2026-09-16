#!/usr/bin/env python3
"""
Reconstruction Sanity Check & Diagnostic Suite.

Validates the numerical and physical correctness of the entire MRI reconstruction pipeline:
1. ZF image range in [0, 1]
2. Target image range in [0, 1]
3. Predicted diffusion x0 range in [0, 1]
4. Diffusion reverse sampling reconstruction range in [0, 1]
5. PINN output range in [0, 1]
6. NaN/Inf detection across all outputs
7. Shape consistency across all pipeline stages
8. FFT/IFFT round-trip reconstruction accuracy (< 1e-6)
9. Data-consistency projection fidelity on sampled k-space (< 1e-5)

Prints clear [PASS] / [FAIL] status for each check.
"""

import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import align_and_normalize_slice, center_crop
from src.data.undersampling import RandomMaskFunc
from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_pinn import DiffusionPINN
from src.models.diffusion_schedule import DiffusionSchedule
from src.physics.data_consistency import apply_data_consistency
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch


def run_sanity_checks():
    print("=" * 70)
    print("MRI DIFFUSION + PINN RECONSTRUCTION PIPELINE SANITY CHECKS")
    print("=" * 70)

    device = torch.device("cpu")
    all_passed = True

    # -------------------------------------------------------------
    # 1. Setup realistic test slice on standard fastMRI grid (640, 368)
    # -------------------------------------------------------------
    torch.manual_seed(42)
    H, W = 640, 368
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    knee = (x**2 / 0.6**2 + y**2 / 0.8**2) < 0.8
    bone = ((x + 0.1)**2 / 0.2**2 + (y + 0.1)**2 / 0.3**2) < 0.5
    sim_img = torch.zeros(H, W, dtype=torch.float32)
    sim_img[knee] = 0.6 + 0.3 * torch.cos(8 * x[knee])
    sim_img[bone] = 0.95
    phase = 0.5 * torch.sin(3 * x) + 0.2 * torch.cos(2 * y)
    sim_complex = sim_img.to(torch.complex64) * torch.exp(1j * phase)

    kspace_raw = image_to_kspace_torch(sim_complex)
    target_raw = center_crop(sim_img, (320, 320))

    # Undersampling (4x)
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4], seed=42)
    mask = mask_func(kspace_raw.shape, seed=42)
    undersampled = kspace_raw * mask

    zf_complex = kspace_to_image_torch(undersampled)
    zf_mag = torch.abs(zf_complex)

    zf_norm, target_norm = align_and_normalize_slice(zf_mag, target_raw, crop_size=(320, 320))
    condition = zf_norm.unsqueeze(0).unsqueeze(0).float()
    target = target_norm.unsqueeze(0).unsqueeze(0).float()

    k_tgt_320 = image_to_kspace_torch(target.squeeze(1).to(torch.complex64))
    mask_320 = center_crop(mask, (1, 320)).to(dtype=k_tgt_320.real.dtype)
    y_meas_320 = k_tgt_320 * mask_320

    # -------------------------------------------------------------
    # Check 1: Zero-Filled image range
    # -------------------------------------------------------------
    zf_min, zf_max = zf_norm.min().item(), zf_norm.max().item()
    zf_pass = (0.0 <= zf_min <= 0.05) and (0.95 <= zf_max <= 1.0)
    print(f"Check 1: Zero-Filled image range [{zf_min:.4f}, {zf_max:.4f}] -> [{'PASS' if zf_pass else 'FAIL'}]")
    if not zf_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 2: Target image range
    # -------------------------------------------------------------
    tgt_min, tgt_max = target_norm.min().item(), target_norm.max().item()
    tgt_pass = (0.0 <= tgt_min <= 0.05) and (0.95 <= tgt_max <= 1.0)
    print(f"Check 2: Target image range [{tgt_min:.4f}, {tgt_max:.4f}] -> [{'PASS' if tgt_pass else 'FAIL'}]")
    if not tgt_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 3: Predicted diffusion x0 range (Tweedie's formula)
    # -------------------------------------------------------------
    schedule = DiffusionSchedule(timesteps=10, device=device)
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=10,
        enforce_dc=True,
    )
    t = torch.tensor([5], dtype=torch.long)
    noise = torch.randn_like(target)
    noisy_target = schedule.add_noise(target, noise, t)
    pred_noise = model.diffusion(noisy_target, condition, t)
    pred_x0 = schedule.predict_x0(noisy_target, pred_noise, t, clamp=True)

    x0_min, x0_max = pred_x0.min().item(), pred_x0.max().item()
    x0_pass = (0.0 <= x0_min) and (x0_max <= 1.0)
    print(f"Check 3: Predicted diffusion x0 range [{x0_min:.4f}, {x0_max:.4f}] -> [{'PASS' if x0_pass else 'FAIL'}]")
    if not x0_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 4: Diffusion reverse sampling reconstruction range
    # -------------------------------------------------------------
    diff_recon = schedule.sample(model=model.diffusion, condition=condition, device=device)
    diff_min, diff_max = diff_recon.min().item(), diff_recon.max().item()
    diff_pass = (0.0 <= diff_min) and (diff_max <= 1.0)
    print(f"Check 4: Diffusion reconstruction range [{diff_min:.4f}, {diff_max:.4f}] -> [{'PASS' if diff_pass else 'FAIL'}]")
    if not diff_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 5: PINN output range
    # -------------------------------------------------------------
    coarse, refined = model.sample(condition=condition, y_measured=y_meas_320, mask=mask_320, device=device)
    pinn_min, pinn_max = refined.min().item(), refined.max().item()
    pinn_pass = (0.0 <= pinn_min) and (pinn_max <= 1.0)
    print(f"Check 5: PINN output range [{pinn_min:.4f}, {pinn_max:.4f}] -> [{'PASS' if pinn_pass else 'FAIL'}]")
    if not pinn_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 6: NaN/Inf detection across all outputs
    # -------------------------------------------------------------
    has_nan = any(
        torch.isnan(t).any().item() or torch.isinf(t).any().item()
        for t in [condition, target, pred_x0, diff_recon, coarse, refined]
    )
    nan_pass = not has_nan
    print(f"Check 6: NaN/Inf detection (all tensors finite) -> [{'PASS' if nan_pass else 'FAIL'}]")
    if not nan_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 7: Shape consistency across stages
    # -------------------------------------------------------------
    expected_shape = (1, 1, 320, 320)
    shape_pass = (
        condition.shape == expected_shape
        and target.shape == expected_shape
        and pred_x0.shape == expected_shape
        and diff_recon.shape == expected_shape
        and coarse.shape == expected_shape
        and refined.shape == expected_shape
    )
    print(f"Check 7: Shape consistency ({expected_shape}) -> [{'PASS' if shape_pass else 'FAIL'}]")
    if not shape_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 8: FFT/IFFT round-trip accuracy
    # -------------------------------------------------------------
    k_roundtrip = image_to_kspace_torch(sim_complex)
    img_roundtrip = kspace_to_image_torch(k_roundtrip)
    fft_err = torch.norm(img_roundtrip - sim_complex).item() / torch.norm(sim_complex).item()
    fft_pass = fft_err < 1e-6
    print(f"Check 8: FFT/IFFT round-trip error ({fft_err:.2e} < 1e-6) -> [{'PASS' if fft_pass else 'FAIL'}]")
    if not fft_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 9: Data-consistency error on sampled k-space
    # -------------------------------------------------------------
    dc_out = apply_data_consistency(
        x_image=refined,
        k_measured=y_meas_320,
        mask=mask_320,
        return_complex=True,
    )
    k_dc = image_to_kspace_torch(dc_out)
    sampled_fidelity_err = torch.norm(mask_320 * (k_dc - y_meas_320)).item() / (torch.norm(y_meas_320).item() + 1e-12)
    dc_pass = sampled_fidelity_err < 1e-5
    print(f"Check 9: Data-consistency sampled k-space fidelity ({sampled_fidelity_err:.2e} < 1e-5) -> [{'PASS' if dc_pass else 'FAIL'}]")
    if not dc_pass:
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL 9 SANITY CHECKS PASSED SUCCESSFULLY! [PASS]")
    else:
        print("SOME SANITY CHECKS FAILED! [FAIL]")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_sanity_checks()
    sys.exit(0 if success else 1)
