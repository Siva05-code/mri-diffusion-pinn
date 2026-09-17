"""
Conditional Diffusion MRI Reconstruction Training Engine.

Trains a conditional diffusion U-Net (DDPM) conditioned on zero-filled MRI inputs and sampling masks.
Optimizes a joint objective:
    L_total = L_noise + lambda_image * L_image + lambda_kspace * L_kspace
where:
    L_noise  = MSE(predicted_noise, noise)
    L_image  = L1(x0_pred, target)
    L_kspace = relative Fourier error against ACTUAL acquired raw measurements y_raw:
               || M_raw ⊙ (A(x0_pred; φ_zf) - y_raw) ||_1 / (|| M_raw ⊙ y_raw ||_1 + eps)
    x0_pred  = (x_t - sqrt(1 - alpha_bar) * predicted_noise) / sqrt(alpha_bar)  (Tweedie's formula)

Validation uses the real multi-step reverse diffusion inference path (schedule.sample)
with raw-domain measurement guidance and zero ground-truth leakage.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import FastMRIDataset
from src.data.preprocessing import normalize_slice_pair, center_crop
from src.data.undersampling import (
    EquispacedMaskFunc,
    RandomMaskFunc,
    create_mask_func,
    get_recon_conditioning_mask,
)
from src.evaluation.metrics import calculate_nmse, calculate_psnr, calculate_ssim
from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_schedule import DiffusionSchedule
from src.physics.forward_operator import MRIForwardOperator
from src.physics.fft import kspace_to_image_torch


def prepare_batch(
    sample: dict,
    mask_func: Union[EquispacedMaskFunc, RandomMaskFunc],
    device: torch.device,
    acceleration: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare one MRI sample for diffusion training or validation.

    Physical Process:
        1. Acquisition: y_raw = mask_raw * kspace_full  (in raw [640, 368] domain)
        2. Reconstruction: zf_complex = IFFT(y_raw)
        3. FOV Cropping: zf_crop = center_crop(zf_complex, (320, 320))
        4. Target Extraction: target_crop = center_crop(target, (320, 320))
        5. Shared Normalization: scale = percentile(zf_mag, 99). Divide all by scale.

    Returns:
        zf_out:      Normalized 320x320 zero-filled magnitude image [B, 1, 320, 320]
        tgt_out:     Normalized 320x320 target magnitude image [B, 1, 320, 320]
        mask_recon:  320x320 conditioning mask for model channel 3 [B, 1, 320, 320]
        y_raw_norm:  Normalized raw complex measurements [B, H_raw, W_raw]
        mask_raw:    Raw acquisition mask [B, H_raw, W_raw]
        phase_zf:    Phase map from zero-filled reconstruction [B, 1, 320, 320]
    """
    kspace = sample["kspace"].to(device)
    raw_target = sample.get("target")

    # Ensure complex k-space
    if not torch.is_complex(kspace):
        if kspace.shape[-1] == 2:
            kspace = torch.complex(kspace[..., 0], kspace[..., 1])
        else:
            kspace = kspace.to(torch.complex64)

    # 1. Raw acquisition mask and measurement
    mask_raw = mask_func(kspace.shape, seed=seed).to(device)
    y_raw = kspace * mask_raw

    # 2. Zero-filled complex reconstruction via centered 2D IFFT
    zf_complex_raw = kspace_to_image_torch(y_raw)
    zf_complex_crop = center_crop(zf_complex_raw, (320, 320))
    zf_mag_crop = torch.abs(zf_complex_crop)
    phase_zf = torch.angle(zf_complex_crop)

    # 3. Target reference image
    if raw_target is not None:
        tgt_tensor = raw_target.to(device)
        if torch.is_complex(tgt_tensor):
            tgt_tensor = torch.abs(tgt_tensor)
        target_crop = center_crop(tgt_tensor, (320, 320))
    else:
        full_complex = kspace_to_image_torch(kspace)
        target_crop = center_crop(torch.abs(full_complex), (320, 320))

    # 4. Shared per-slice normalization derived strictly from zero-filled condition
    zf_norm, tgt_norm, scale = normalize_slice_pair(zf_mag_crop, target_crop, percentile=99.0)
    y_raw_norm = y_raw / scale

    # Guarantee 4D shape [B, 1, 320, 320]
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

    # Broadcast mask_raw to match y_raw shape if necessary
    if mask_raw.shape != y_raw.shape:
        mask_raw = mask_raw.expand_as(y_raw)

    return zf_out, tgt_out, mask_recon, y_raw_norm, mask_raw, phase_out


def validate_diffusion_epoch(
    model: ConditionalDiffusionUNet,
    val_loader: DataLoader,
    schedule: DiffusionSchedule,
    mask_func: Union[EquispacedMaskFunc, RandomMaskFunc],
    device: torch.device,
    acceleration: int,
    max_val_batches: Optional[int] = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Validates diffusion reconstruction on validation slices using the REAL inference path
    (schedule.sample) with raw-domain measurement guidance and without ground truth leakage.
    """
    model.eval()
    psnrs, ssims, nmses = [], [], []
    zf_psnrs, zf_ssims, zf_nmses = [], [], []

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            if max_val_batches is not None and max_val_batches > 0 and i >= max_val_batches:
                break

            slice_seed = seed + int(sample.get("slice_idx", [i])[0])
            zf_out, tgt_out, mask_recon, y_raw_norm, mask_raw, phase_zf = prepare_batch(
                sample=sample,
                mask_func=mask_func,
                device=device,
                acceleration=acceleration,
                seed=slice_seed,
            )

            # Measurement-guided reverse diffusion inference from random Gaussian noise
            recon = schedule.sample(
                model=model,
                condition=zf_out,
                mask=mask_recon,
                y_raw=y_raw_norm,
                mask_raw=mask_raw,
                phase_zf=phase_zf,
                measurement_guided=True,
                device=device,
            )

            pred_np = recon[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            tgt_np = tgt_out[0, 0].cpu().numpy().astype(np.float32)
            zf_np = zf_out[0, 0].cpu().numpy().astype(np.float32)

            data_range = float(tgt_np.max() - tgt_np.min() + 1e-12)
            psnrs.append(calculate_psnr(tgt_np, pred_np, data_range=data_range))
            ssims.append(calculate_ssim(tgt_np, pred_np, data_range=data_range))
            nmses.append(calculate_nmse(tgt_np, pred_np))

            zf_psnrs.append(calculate_psnr(tgt_np, zf_np, data_range=data_range))
            zf_ssims.append(calculate_ssim(tgt_np, zf_np, data_range=data_range))
            zf_nmses.append(calculate_nmse(tgt_np, zf_np))

    return {
        "val_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "val_psnr_std": float(np.std(psnrs)) if psnrs else 0.0,
        "val_ssim": float(np.mean(ssims)) if ssims else 0.0,
        "val_ssim_std": float(np.std(ssims)) if ssims else 0.0,
        "val_nmse": float(np.mean(nmses)) if nmses else 0.0,
        "val_nmse_std": float(np.std(nmses)) if nmses else 0.0,
        "val_zf_psnr": float(np.mean(zf_psnrs)) if zf_psnrs else 0.0,
        "val_zf_ssim": float(np.mean(zf_ssims)) if zf_ssims else 0.0,
        "val_zf_nmse": float(np.mean(zf_nmses)) if zf_nmses else 0.0,
    }


def train_diffusion(
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
    lambda_image: float = 1.0,
    lambda_kspace: float = 0.1,
    seed: int = 42,
    max_val_batches: Optional[int] = 10,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Main training function for Conditional Diffusion MRI Reconstruction.
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

    print("=" * 65)
    print("CONDITIONAL DIFFUSION MRI RECONSTRUCTION TRAINING")
    print("=" * 65)
    print(f"Acceleration    : {acceleration}x")
    print(f"Center Fraction : {center_fraction}")
    print(f"Mask Protocol   : {mask_type.capitalize()} Cartesian")
    print(f"Epochs          : {epochs}")
    print(f"Batch Size      : {batch_size}")
    print(f"Learning Rate   : {learning_rate}")
    print(f"Timesteps       : {timesteps}")
    print(f"Loss Weights    : Noise=1.0, Image={lambda_image}, K-space={lambda_kspace}")
    print(f"Device          : {torch_device}")
    print(f"Seed            : {seed}")
    print("=" * 65)

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

    model = ConditionalDiffusionUNet(in_channels=3, base_channels=32, time_dim=128).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = DiffusionSchedule(timesteps=timesteps, device=torch_device)
    forward_op = MRIForwardOperator()

    mask_func = create_mask_func(
        mask_type=mask_type,
        center_fraction=center_fraction,
        acceleration=acceleration,
        seed=seed,
    )

    history = []
    best_val_psnr = -float("inf")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / f"diffusion_accel{acceleration}_latest.pt"
    best_path = checkpoint_dir / f"diffusion_accel{acceleration}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_noise_loss = 0.0
        total_image_loss = 0.0
        total_kspace_loss = 0.0
        start_time = time.time()

        for step, sample in enumerate(train_loader):
            batch_seed = seed + epoch * 10000 + step
            condition, target, mask_recon, y_raw_norm, mask_raw, phase_zf = prepare_batch(
                sample=sample,
                mask_func=mask_func,
                device=torch_device,
                acceleration=acceleration,
                seed=batch_seed,
            )

            # Sample random diffusion timesteps
            t = torch.randint(0, timesteps, (target.shape[0],), device=torch_device)
            noise = torch.randn_like(target)

            # Add noise to ground-truth target
            noisy_target = schedule.add_noise(target, noise, t)

            # Diffusion model predicts injected noise: [noisy_target, condition, mask_recon]
            predicted_noise = model(noisy_target, condition, mask_recon, t)

            # 1. Noise prediction loss (DDPM objective)
            loss_noise = F.mse_loss(predicted_noise, noise)

            # 2. Tweedie differentiable predicted clean image
            x0_pred = schedule.predict_x0(noisy_target, predicted_noise, t, clamp=True)

            # 3. Image reconstruction loss (L1)
            loss_image = F.l1_loss(x0_pred, target)

            # 4. K-space fidelity loss using actual raw measurements
            if lambda_kspace > 0.0:
                raw_shape = (y_raw_norm.shape[-2], y_raw_norm.shape[-1])
                k_pred_raw = forward_op.forward_kspace(
                    x0_pred.squeeze(1),
                    phase=phase_zf.squeeze(1),
                    target_shape=raw_shape,
                )
                mask_k = mask_raw.to(device=k_pred_raw.device, dtype=k_pred_raw.real.dtype)
                diff = mask_k * (k_pred_raw - y_raw_norm)
                diff_mag = torch.sqrt(diff.real**2 + diff.imag**2 + 1e-12)
                ref_mag = torch.sqrt((mask_k * y_raw_norm).real**2 + (mask_k * y_raw_norm).imag**2 + 1e-12)
                loss_kspace = torch.sum(diff_mag) / (torch.sum(ref_mag) + 1e-12)
            else:
                loss_kspace = torch.tensor(0.0, device=torch_device)

            # Composite weighted loss
            loss = loss_noise + lambda_image * loss_image + lambda_kspace * loss_kspace

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_epoch_loss += loss.item()
            total_noise_loss += loss_noise.item()
            total_image_loss += loss_image.item()
            total_kspace_loss += loss_kspace.item()

        n_batches = max(len(train_loader), 1)
        avg_loss = total_epoch_loss / n_batches
        avg_noise = total_noise_loss / n_batches
        avg_img = total_image_loss / n_batches
        avg_kspace = total_kspace_loss / n_batches

        # Real validation on inference path
        val_metrics = validate_diffusion_epoch(
            model=model,
            val_loader=val_loader,
            schedule=schedule,
            mask_func=mask_func,
            device=torch_device,
            acceleration=acceleration,
            max_val_batches=max_val_batches,
            seed=seed,
        )

        elapsed = time.time() - start_time
        val_psnr = val_metrics["val_psnr"]
        val_ssim = val_metrics["val_ssim"]
        val_nmse = val_metrics["val_nmse"]

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Loss: {avg_loss:.4f} (Noise: {avg_noise:.4f}, Img: {avg_img:.4f}, Ksp: {avg_kspace:.4f}) | "
            f"Val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}, NMSE: {val_nmse:.4f} | "
            f"{elapsed:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "noise_loss": avg_noise,
            "image_loss": avg_img,
            "kspace_loss": avg_kspace,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_nmse": val_nmse,
            "epoch_time": elapsed,
        }
        history.append(record)

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acceleration": acceleration,
                "timesteps": timesteps,
                "val_psnr": val_psnr,
            },
            latest_path,
        )

        # Save best checkpoint by validation PSNR
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "acceleration": acceleration,
                    "timesteps": timesteps,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                },
                best_path,
            )

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"diffusion_train_accel{acceleration}_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Latest checkpoint : {latest_path}")
    print(f"Best checkpoint   : {best_path} (PSNR: {best_val_psnr:.2f} dB)")
    print(f"Training log      : {log_path}")

    return {
        "history": history,
        "best_val_psnr": best_val_psnr,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
    }


if __name__ == "__main__":
    train_diffusion(
        train_dir="/kaggle/working/data/fastmri/knee_singlecoil_train",
        val_dir="/kaggle/working/data/fastmri/knee_singlecoil_val",
        acceleration=4,
        epochs=5,
        batch_size=1,
        learning_rate=1e-4,
        timesteps=100,
        subset_files=20,
        lambda_image=1.0,
        lambda_kspace=0.1,
        seed=42,
        device="cuda",
    )