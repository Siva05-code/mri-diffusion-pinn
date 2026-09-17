"""
Joint Diffusion Pre-Reconstruction + PINN Refinement Training Engine.

Trains the composite pipeline:
    Undersampled k-space -> Zero-filled -> Conditional Diffusion -> Coarse Image
    -> PINN Refinement -> (Data Consistency) -> Final Image

Supervised by the composite physics-guided objective:
    L_total = L_noise + lambda_image * L_image + lambda_kspace * L_kspace + lambda_physics * L_physics
where k-space loss compares the forward model of the reconstructed image against ACTUAL acquired
raw measurements y_raw.

Validation uses the real multi-step inference path:
    model.sample(condition=zf, mask=mask_recon, y_measured=y_raw, mask_raw=mask_raw, phase_zf=phase_zf)
and computes evaluation metrics for Zero-Filled, Coarse Diffusion, and Refined PINN outputs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FastMRIDataset
from src.data.preprocessing import normalize_slice_pair, center_crop
from src.data.undersampling import (
    EquispacedMaskFunc,
    RandomMaskFunc,
    create_mask_func,
    get_recon_conditioning_mask,
)
from src.evaluation.metrics import calculate_nmse, calculate_psnr, calculate_ssim
from src.losses.total_loss import TotalReconstructionLoss
from src.models.diffusion_pinn import DiffusionPINN
from src.physics.fft import kspace_to_image_torch


def prepare_training_batch(
    sample: dict,
    mask_func: Union[EquispacedMaskFunc, RandomMaskFunc],
    device: torch.device,
    acceleration: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares a batch for joint Diffusion + PINN training using physical raw measurements:
    1. Generates acquisition mask in raw domain: mask_raw
    2. Measures raw k-space: y_raw = mask_raw * kspace_full
    3. Computes zero-filled reconstruction via centered IFFT
    4. Center-crops and normalizes both ZF and Target to 320x320 using a shared per-slice scale
    5. Builds spatial conditioning mask for the neural network

    Returns:
        zf_out:      Normalized zero-filled condition [B, 1, 320, 320]
        tgt_out:     Normalized target reference [B, 1, 320, 320]
        mask_recon:  Conditioning mask for diffusion [B, 1, 320, 320]
        y_raw_norm:  Normalized raw measurements [B, H_raw, W_raw] (complex)
        mask_raw:    Raw acquisition mask [B, H_raw, W_raw]
        phase_zf:    Phase map from zero-filled reconstruction [B, 1, 320, 320]
    """
    kspace = sample["kspace"].to(device)
    raw_target = sample.get("target")

    if not torch.is_complex(kspace):
        if kspace.shape[-1] == 2:
            kspace = torch.complex(kspace[..., 0], kspace[..., 1])
        else:
            kspace = kspace.to(torch.complex64)

    # 1. Generate mask & measure raw k-space
    mask_raw = mask_func(kspace.shape, seed=seed).to(device)
    y_raw = kspace * mask_raw

    # 2. Zero-filled reconstruction
    zf_complex_raw = kspace_to_image_torch(y_raw)
    zf_complex_crop = center_crop(zf_complex_raw, (320, 320))
    zf_mag_crop = torch.abs(zf_complex_crop)
    phase_zf = torch.angle(zf_complex_crop)

    # 3. Target reference
    if raw_target is not None:
        tgt_tensor = raw_target.to(device)
        if torch.is_complex(tgt_tensor):
            tgt_tensor = torch.abs(tgt_tensor)
        target_crop = center_crop(tgt_tensor, (320, 320))
    else:
        full_complex = kspace_to_image_torch(kspace)
        target_crop = center_crop(torch.abs(full_complex), (320, 320))

    # 4. Shared per-slice normalization
    zf_norm, tgt_norm, scale = normalize_slice_pair(zf_mag_crop, target_crop, percentile=99.0)
    y_raw_norm = y_raw / scale

    # Guarantee 4D tensors [B, 1, 320, 320]
    if zf_norm.ndim == 2:
        zf_out = zf_norm.unsqueeze(0).unsqueeze(0).float()
        tgt_out = tgt_norm.unsqueeze(0).unsqueeze(0).float()
        phase_out = phase_zf.unsqueeze(0).unsqueeze(0).float()
    elif zf_norm.ndim == 3:
        zf_out = zf_norm.unsqueeze(1).float()
        tgt_out = tgt_norm.unsqueeze(1).float()
        phase_out = phase_zf.unsqueeze(1).float()
    else:
        zf_out = zf_norm.float()
        tgt_out = tgt_norm.float()
        phase_out = phase_zf.float()

    # 5. Build conditioning mask aligned with reconstruction domain
    mask_recon = get_recon_conditioning_mask(mask_raw, recon_size=(320, 320)).to(device)

    if mask_raw.shape != y_raw.shape:
        mask_raw = mask_raw.expand_as(y_raw)

    return zf_out, tgt_out, mask_recon, y_raw_norm, mask_raw, phase_out


def validate_epoch(
    model: DiffusionPINN,
    val_loader: DataLoader,
    mask_func: Union[EquispacedMaskFunc, RandomMaskFunc],
    criterion: TotalReconstructionLoss,
    device: torch.device,
    acceleration: int,
    seed: int = 42,
    max_val_batches: Optional[int] = 10,
) -> Dict[str, float]:
    """
    Validates model performance on validation slices using the REAL inference path
    (model.sample) with raw-domain measurement guidance and zero ground-truth leakage.
    """
    model.eval()
    val_total_loss = 0.0
    pinn_psnrs, pinn_ssims, pinn_nmses = [], [], []
    diff_psnrs, diff_ssims, diff_nmses = [], [], []
    zf_psnrs, zf_ssims, zf_nmses = [], [], []

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            if max_val_batches is not None and i >= max_val_batches:
                break

            slice_seed = seed + int(sample.get("slice_idx", [i])[0])
            zf_out, tgt_out, mask_recon, y_raw_norm, mask_raw, phase_zf = prepare_training_batch(
                sample=sample,
                mask_func=mask_func,
                device=device,
                acceleration=acceleration,
                seed=slice_seed,
            )

            # Real multi-step reverse diffusion + PINN refinement inference path
            coarse, refined = model.sample(
                condition=zf_out,
                mask=mask_recon,
                y_measured=y_raw_norm,
                mask_raw=mask_raw,
                phase_zf=phase_zf,
                device=device,
            )

            loss_dict = criterion(
                i_pred=refined,
                i_gt=tgt_out,
                y_measured=y_raw_norm,
                k_gt=None,
                mask=mask_raw,
                phase=phase_zf,
            )
            val_total_loss += loss_dict["total_loss"].item()

            target_np = tgt_out[0, 0].cpu().numpy().astype(np.float32)
            zf_np = zf_out[0, 0].cpu().numpy().astype(np.float32)
            coarse_np = coarse[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            refined_np = refined[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

            data_range = float(target_np.max() - target_np.min() + 1e-12)

            # Metrics for Zero-Filled
            zf_psnrs.append(calculate_psnr(target_np, zf_np, data_range=data_range))
            zf_ssims.append(calculate_ssim(target_np, zf_np, data_range=data_range))
            zf_nmses.append(calculate_nmse(target_np, zf_np))

            # Metrics for Coarse Diffusion
            diff_psnrs.append(calculate_psnr(target_np, coarse_np, data_range=data_range))
            diff_ssims.append(calculate_ssim(target_np, coarse_np, data_range=data_range))
            diff_nmses.append(calculate_nmse(target_np, coarse_np))

            # Metrics for Refined Diffusion+PINN
            pinn_psnrs.append(calculate_psnr(target_np, refined_np, data_range=data_range))
            pinn_ssims.append(calculate_ssim(target_np, refined_np, data_range=data_range))
            pinn_nmses.append(calculate_nmse(target_np, refined_np))

    n = max(len(pinn_psnrs), 1)
    return {
        "val_total_loss": val_total_loss / n,
        "zf_psnr": float(np.mean(zf_psnrs)),
        "zf_ssim": float(np.mean(zf_ssims)),
        "zf_nmse": float(np.mean(zf_nmses)),
        "diff_psnr": float(np.mean(diff_psnrs)),
        "diff_ssim": float(np.mean(diff_ssims)),
        "diff_nmse": float(np.mean(diff_nmses)),
        "pinn_psnr": float(np.mean(pinn_psnrs)),
        "pinn_psnr_std": float(np.std(pinn_psnrs)),
        "pinn_ssim": float(np.mean(pinn_ssims)),
        "pinn_ssim_std": float(np.std(pinn_ssims)),
        "pinn_nmse": float(np.mean(pinn_nmses)),
    }


def train_diffusion_pinn(
    train_dir: str,
    val_dir: str,
    acceleration: int = 4,
    center_fraction: Optional[float] = None,
    mask_type: str = "equispaced",
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    timesteps: int = 100,
    subset_files: Optional[int] = 20,
    ablation: str = "full",
    lambda_image: float = 1.0,
    lambda_kspace: float = 0.1,
    lambda_physics: float = 0.1,
    lambda_gradient: float = 0.1,
    enforce_dc: bool = True,
    seed: int = 42,
    max_val_batches: Optional[int] = 10,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Main training routine for joint Diffusion + PINN MRI reconstruction framework.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_device = torch.device(
        device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    if center_fraction is None:
        center_fraction = 0.08 if acceleration == 4 else 0.04

    # Configure ablation parameters
    ablation = ablation.lower()
    enable_pinn = True
    if ablation == "diffusion_only":
        enable_pinn = False
        lambda_physics = 0.0
    elif ablation == "diffusion_kspace":
        enable_pinn = False
        lambda_physics = 0.0
    elif ablation == "diffusion_physics":
        enable_pinn = False
        lambda_kspace = 0.0
    elif ablation == "diffusion_pinn_no_physics":
        enable_pinn = True
        lambda_kspace = 0.0
        lambda_physics = 0.0

    print("=" * 65)
    print("PHYSICS-INFORMED DIFFUSION-PINN MRI RECONSTRUCTION")
    print("=" * 65)
    print(f"Acceleration    : {acceleration}x")
    print(f"Center Fraction : {center_fraction}")
    print(f"Mask Protocol   : {mask_type.capitalize()} Cartesian")
    print(f"Epochs          : {epochs}")
    print(f"Batch Size      : {batch_size}")
    print(f"Learning Rate   : {learning_rate}")
    print(f"Timesteps       : {timesteps}")
    print(f"Device          : {torch_device}")
    print(f"Ablation Mode   : {ablation} (PINN enabled: {enable_pinn})")
    print(f"Loss Weights    : Image={lambda_image}, K-space={lambda_kspace}, Physics={lambda_physics}")
    print(f"Seed            : {seed}")
    print("=" * 65)

    # 1. Datasets & Loaders
    train_dataset = FastMRIDataset(train_dir, max_files=subset_files)
    val_dataset = FastMRIDataset(val_dir, max_files=10)

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # 2. Mask generators
    mask_func = create_mask_func(
        mask_type=mask_type,
        center_fraction=center_fraction,
        acceleration=acceleration,
        seed=seed,
    )

    # 3. Model
    model = DiffusionPINN(
        diffusion_base_channels=32,
        diffusion_time_dim=128,
        pinn_base_channels=32,
        pinn_num_blocks=4,
        timesteps=timesteps,
        enforce_dc=enforce_dc,
    ).to(torch_device)

    # 4. Losses & Optimizer
    criterion = TotalReconstructionLoss(
        image_loss_type="l1",
        lambda_image=lambda_image,
        lambda_kspace=lambda_kspace,
        lambda_physics=lambda_physics,
        lambda_gradient=lambda_gradient,
        relative_kspace=True,
    ).to(torch_device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)

    history = []
    best_psnr = -float("inf")

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()

        train_total_loss_accum = 0.0
        train_noise_loss_accum = 0.0
        train_img_loss_accum = 0.0
        train_ksp_loss_accum = 0.0
        train_phys_loss_accum = 0.0

        for step, sample in enumerate(train_loader):
            batch_seed = seed + epoch * 10000 + step
            zf, target, mask_recon, y_raw_norm, mask_raw, phase_zf = prepare_training_batch(
                sample=sample,
                mask_func=mask_func,
                device=torch_device,
                acceleration=acceleration,
                seed=batch_seed,
            )

            # End-to-end forward pass through Diffusion + PINN (training denoising path)
            out = model.forward_train(
                target=target,
                condition=zf,
                mask=mask_recon,
                mask_raw=mask_raw,
                y_measured=y_raw_norm,
                phase=phase_zf,
            )

            # If PINN disabled in ablation, evaluate loss on coarse image; else on refined image
            output_image = out["refined"] if enable_pinn else out["coarse"]

            # Compute composite physics-guided reconstruction loss
            loss_dict = criterion(
                i_pred=output_image,
                i_gt=target,
                y_measured=y_raw_norm,
                k_gt=None,
                mask=mask_raw,
                phase=phase_zf,
            )

            # Diffusion noise MSE loss
            noise_loss = torch.nn.functional.mse_loss(out["pred_noise"], out["noise"])
            total_step_loss = loss_dict["total_loss"] + noise_loss

            optimizer.zero_grad()
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_total_loss_accum += total_step_loss.item()
            train_noise_loss_accum += noise_loss.item()
            train_img_loss_accum += loss_dict["image_loss"].item()
            train_ksp_loss_accum += loss_dict["kspace_loss"].item()
            train_phys_loss_accum += loss_dict["physics_loss"].item()

        n_batches = max(len(train_loader), 1)
        avg_train_loss = train_total_loss_accum / n_batches
        avg_noise_loss = train_noise_loss_accum / n_batches
        avg_img_loss = train_img_loss_accum / n_batches
        avg_ksp_loss = train_ksp_loss_accum / n_batches
        avg_phys_loss = train_phys_loss_accum / n_batches

        # Real validation on multi-step inference path
        val_results = validate_epoch(
            model=model,
            val_loader=val_loader,
            mask_func=mask_func,
            criterion=criterion,
            device=torch_device,
            acceleration=acceleration,
            seed=seed,
            max_val_batches=max_val_batches,
        )

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Loss: {avg_train_loss:.4f} (Noise: {avg_noise_loss:.4f}, Img: {avg_img_loss:.4f}, Ksp: {avg_ksp_loss:.4f}) | "
            f"Val PSNR: ZF={val_results['zf_psnr']:.2f}, Diff={val_results['diff_psnr']:.2f}, PINN={val_results['pinn_psnr']:.2f} dB | "
            f"{epoch_time:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "noise_loss": avg_noise_loss,
            "image_loss": avg_img_loss,
            "kspace_loss": avg_ksp_loss,
            "physics_loss": avg_phys_loss,
            "val_loss": val_results["val_total_loss"],
            "zf_psnr": val_results["zf_psnr"],
            "diff_psnr": val_results["diff_psnr"],
            "pinn_psnr": val_results["pinn_psnr"],
            "pinn_ssim": val_results["pinn_ssim"],
            "pinn_nmse": val_results["pinn_nmse"],
            "epoch_time": epoch_time,
        }
        history.append(record)

        # Save latest checkpoint
        latest_path = checkpoints_dir / f"diffusion_pinn_{ablation}_accel{acceleration}_latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acceleration": acceleration,
                "ablation": ablation,
                "val_psnr": val_results["pinn_psnr"],
            },
            latest_path,
        )

        # Save best checkpoint
        current_psnr = val_results["pinn_psnr"]
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_path = checkpoints_dir / f"diffusion_pinn_{ablation}_accel{acceleration}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "acceleration": acceleration,
                    "ablation": ablation,
                    "val_psnr": current_psnr,
                    "val_ssim": val_results["pinn_ssim"],
                },
                best_path,
            )

    log_path = logs_dir / f"diffusion_pinn_{ablation}_accel{acceleration}_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete: Best PINN PSNR = {best_psnr:.2f} dB")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Log history:       {log_path}")

    return {
        "history": history,
        "best_psnr": best_psnr,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path) if best_psnr > -float("inf") else str(latest_path),
    }
