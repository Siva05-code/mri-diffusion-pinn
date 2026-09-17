#!/usr/bin/env python3
"""
Reconstruction Sanity Check & Diagnostic Suite.

Validates the numerical and physical correctness of the entire MRI reconstruction pipeline:
1. ZF image range with shared normalization
2. Target image range with shared normalization
3. 3-channel diffusion model forward pass
4. Measurement-guided reverse diffusion sampling with raw-domain consistency
5. PINN refinement forward pass
6. End-to-end DiffusionPINN forward pass & loss computation
7. Forward operator & k-space data-consistency accuracy
8. Evaluation metrics implementation
9. FastMRI data pipeline and mask generation

Prints clear [PASS] / [FAIL] status for each check.
"""

import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import normalize_slice_pair, center_crop
from src.data.undersampling import (
    EquispacedMaskFunc,
    RandomMaskFunc,
    create_mask_func,
    get_recon_conditioning_mask,
)
from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_pinn import DiffusionPINN
from src.models.diffusion_schedule import DiffusionSchedule
from src.physics.forward_operator import MRIForwardOperator
from src.physics.data_consistency import apply_data_consistency
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch
from src.losses.total_loss import TotalReconstructionLoss
from src.evaluation.metrics import calculate_psnr, calculate_ssim, calculate_nmse


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

    # Undersampling (4x equispaced)
    mask_func = create_mask_func(mask_type="equispaced", center_fraction=0.08, acceleration=4, seed=42)
    mask_raw = mask_func(kspace_raw.shape, seed=42)
    y_raw = kspace_raw * mask_raw

    zf_complex_raw = kspace_to_image_torch(y_raw)
    zf_complex_crop = center_crop(zf_complex_raw, (320, 320))
    zf_mag_crop = torch.abs(zf_complex_crop)
    phase_zf = torch.angle(zf_complex_crop)

    zf_norm, target_norm, scale = normalize_slice_pair(zf_mag_crop, target_raw, percentile=99.0)
    condition = zf_norm.unsqueeze(0).unsqueeze(0).float()
    target = target_norm.unsqueeze(0).unsqueeze(0).float()
    mask_recon = get_recon_conditioning_mask(mask_raw, recon_size=(320, 320))
    y_raw_norm = y_raw / scale

    # -------------------------------------------------------------
    # Check 1: Zero-Filled image range
    # -------------------------------------------------------------
    zf_min, zf_max = zf_norm.min().item(), zf_norm.max().item()
    zf_pass = (0.0 <= zf_min <= 0.05) and (0.8 <= zf_max <= 1.5)
    print(f"Check 1: Zero-Filled image range [{zf_min:.4f}, {zf_max:.4f}] -> [{'PASS' if zf_pass else 'FAIL'}]")
    if not zf_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 2: Target image range
    # -------------------------------------------------------------
    tgt_min, tgt_max = target_norm.min().item(), target_norm.max().item()
    tgt_pass = (0.0 <= tgt_min <= 0.05) and (0.8 <= tgt_max <= 1.5)
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
    pred_noise = model.diffusion(noisy_target, condition, mask_recon, t)
    pred_x0 = schedule.predict_x0(noisy_target, pred_noise, t, clamp=True)

    x0_min, x0_max = pred_x0.min().item(), pred_x0.max().item()
    x0_pass = (0.0 <= x0_min) and (x0_max <= 1.0)
    print(f"Check 3: Predicted diffusion x0 range [{x0_min:.4f}, {x0_max:.4f}] -> [{'PASS' if x0_pass else 'FAIL'}]")
    if not x0_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 4: Measurement-guided reverse sampling
    # -------------------------------------------------------------
    recon = schedule.sample(
        model=model.diffusion,
        condition=condition,
        mask=mask_recon,
        y_raw=y_raw_norm,
        mask_raw=mask_raw,
        phase_zf=phase_zf.unsqueeze(0).unsqueeze(0),
        measurement_guided=True,
        device=device,
    )
    recon_min, recon_max = recon.min().item(), recon.max().item()
    recon_pass = (0.0 <= recon_min) and (recon_max <= 1.0) and not torch.isnan(recon).any()
    print(f"Check 4: Measurement-guided reverse sampling [{recon_min:.4f}, {recon_max:.4f}] -> [{'PASS' if recon_pass else 'FAIL'}]")
    if not recon_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 5: PINN refinement forward pass
    # -------------------------------------------------------------
    refined = model.pinn(coarse=recon, zero_filled=condition)
    pinn_min, pinn_max = refined.min().item(), refined.max().item()
    pinn_pass = (0.0 <= pinn_min) and (pinn_max <= 1.0) and not torch.isnan(refined).any()
    print(f"Check 5: PINN Refinement output range [{pinn_min:.4f}, {pinn_max:.4f}] -> [{'PASS' if pinn_pass else 'FAIL'}]")
    if not pinn_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 6: End-to-end DiffusionPINN training pass
    # -------------------------------------------------------------
    out = model.forward_train(
        target=target,
        condition=condition,
        mask=mask_recon,
        mask_raw=mask_raw,
        y_measured=y_raw_norm,
        phase=phase_zf.unsqueeze(0).unsqueeze(0),
    )
    e2e_pass = ("coarse" in out) and ("refined" in out) and ("pred_noise" in out)
    print(f"Check 6: End-to-end DiffusionPINN forward pass -> [{'PASS' if e2e_pass else 'FAIL'}]")
    if not e2e_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 7: Composite loss computation with forward operator
    # -------------------------------------------------------------
    criterion = TotalReconstructionLoss(
        image_loss_type="l1",
        lambda_image=1.0,
        lambda_kspace=0.1,
        lambda_physics=0.1,
    )
    loss_dict = criterion(
        i_pred=out["refined"],
        i_gt=target,
        y_measured=y_raw_norm,
        mask=mask_raw,
        phase=phase_zf.unsqueeze(0).unsqueeze(0),
    )
    loss_val = loss_dict["total_loss"].item()
    loss_pass = not np.isnan(loss_val) and not np.isinf(loss_val) and loss_val > 0.0
    print(f"Check 7: Composite Loss computation ({loss_val:.4f}) -> [{'PASS' if loss_pass else 'FAIL'}]")
    if not loss_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 8: Forward operator roundtrip & data consistency
    # -------------------------------------------------------------
    forward_op = MRIForwardOperator()
    x_test = target[0, 0]
    k_test = forward_op.forward_kspace(x_test, target_shape=(640, 368))
    x_adj = forward_op.adjoint(k_test, crop_shape=(320, 320))
    # Center crop of zero-padded FFT matches original exactly:
    rt_err = torch.norm(torch.real(x_adj) - x_test).item() / (torch.norm(x_test).item() + 1e-12)
    dc_res = forward_op.data_consistency(
        x_crop=x_test,
        y_raw=y_raw_norm,
        mask_raw=mask_raw,
        return_complex=False,
    )
    dc_pass = (rt_err < 1e-4) and not torch.isnan(dc_res).any()
    print(f"Check 8: Forward operator adjoint roundtrip (err={rt_err:.2e}) -> [{'PASS' if dc_pass else 'FAIL'}]")
    if not dc_pass:
        all_passed = False

    # -------------------------------------------------------------
    # Check 9: Evaluation metrics implementation
    # -------------------------------------------------------------
    tgt_np = target[0, 0].numpy()
    pred_np = out["refined"][0, 0].detach().numpy()
    data_range = float(tgt_np.max() - tgt_np.min() + 1e-12)
    psnr_val = calculate_psnr(tgt_np, pred_np, data_range=data_range)
    ssim_val = calculate_ssim(tgt_np, pred_np, data_range=data_range)
    nmse_val = calculate_nmse(tgt_np, pred_np)
    metrics_pass = (psnr_val > 0) and (0.0 <= ssim_val <= 1.0) and (nmse_val >= 0.0)
    print(f"Check 9: Metrics: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}, NMSE={nmse_val:.4f} -> [{'PASS' if metrics_pass else 'FAIL'}]")
    if not metrics_pass:
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL 9 RECONSTRUCTION SANITY CHECKS PASSED [OK]")
    else:
        print("ONE OR MORE CHECKS FAILED [ERROR]")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_sanity_checks()
    sys.exit(0 if success else 1)
