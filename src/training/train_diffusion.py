"""
Conditional Diffusion MRI Reconstruction Training Engine.

Trains a conditional diffusion U-Net (DDPM) conditioned on zero-filled MRI inputs.
Optimizes a joint objective:
    L_total = L_noise + lambda_image * L_image + lambda_kspace * L_kspace
where:
    L_noise  = MSE(predicted_noise, noise)
    L_image  = L1(x0_pred, target)
    L_kspace = relative Fourier error of x0_pred at sampled locations in 320x320 domain
    x0_pred  = (x_t - sqrt(1 - alpha_bar) * predicted_noise) / sqrt(alpha_bar)  (Tweedie's formula)

Validation uses the real multi-step reverse diffusion inference path (schedule.sample)
without access to ground truth.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import FastMRIDataset
from src.data.preprocessing import align_and_normalize_slice, center_crop
from src.data.undersampling import RandomMaskFunc
from src.evaluation.metrics import calculate_nmse, calculate_psnr, calculate_ssim
from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_schedule import DiffusionSchedule
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch


def prepare_batch(
    sample: dict,
    mask_func: RandomMaskFunc,
    device: torch.device,
    acceleration: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares a batch for diffusion training:
    - Undersamples k-space
    - Creates zero-filled reconstruction
    - Aligns and normalizes both to [0, 1] at 320x320
    - Generates 320x320 Fourier domain reference and mask for data fidelity
    """
    kspace = sample["kspace"].to(device)
    raw_target = sample.get("target")

    # Ensure complex k-space
    if not torch.is_complex(kspace):
        if kspace.shape[-1] == 2:
            kspace = torch.complex(kspace[..., 0], kspace[..., 1])
        else:
            kspace = kspace.to(torch.complex64)

    # 1. Mask and undersample
    mask = mask_func(kspace.shape, seed=seed).to(device)
    undersampled = kspace * mask

    # 2. Zero-filled image
    zf_complex = kspace_to_image_torch(undersampled)
    zf_mag = torch.abs(zf_complex)

    # 3. Target reference
    if raw_target is not None:
        tgt_mag = raw_target.to(device)
        if torch.is_complex(tgt_mag):
            tgt_mag = torch.abs(tgt_mag)
    else:
        full_complex = kspace_to_image_torch(kspace)
        tgt_mag = torch.abs(full_complex)

    # 4. Spatial alignment & [0, 1] normalization
    if zf_mag.ndim == 2:
        zf_norm, tgt_norm = align_and_normalize_slice(zf_mag, tgt_mag)
        zf_out = zf_norm.unsqueeze(0).unsqueeze(0).float()
        tgt_out = tgt_norm.unsqueeze(0).unsqueeze(0).float()
    else:
        zf_list, tgt_list = [], []
        for b in range(zf_mag.shape[0]):
            z_n, t_n = align_and_normalize_slice(zf_mag[b], tgt_mag[b])
            zf_list.append(z_n)
            tgt_list.append(t_n)
        zf_out = torch.stack(zf_list, dim=0).unsqueeze(1).float()
        tgt_out = torch.stack(tgt_list, dim=0).unsqueeze(1).float()

    while zf_out.ndim > 4:
        zf_out = zf_out.squeeze(1)
    while tgt_out.ndim > 4:
        tgt_out = tgt_out.squeeze(1)

    # 5. Mathematically valid common 320x320 Fourier domain reference
    k_tgt_320 = image_to_kspace_torch(tgt_out.squeeze(1).to(torch.complex64))
    mask_h = mask.shape[-2]
    mask_320 = center_crop(mask, (mask_h, tgt_out.shape[-1]))
    mask_320 = mask_320.to(k_tgt_320.device, dtype=k_tgt_320.real.dtype)

    return zf_out, tgt_out, mask_320, k_tgt_320


def validate_diffusion_epoch(
    model: ConditionalDiffusionUNet,
    val_loader: DataLoader,
    schedule: DiffusionSchedule,
    mask_func: RandomMaskFunc,
    device: torch.device,
    acceleration: int,
    max_val_batches: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Validates diffusion reconstruction on validation slices using the REAL inference path
    (schedule.sample) without using ground truth target during reconstruction.
    """
    model.eval()
    psnrs, ssims, nmses = [], [], []
    zf_psnrs = []

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            if i >= max_val_batches:
                break

            slice_seed = seed + int(sample.get("slice_idx", [i])[0])
            condition, target, _, _ = prepare_batch(
                sample=sample,
                mask_func=mask_func,
                device=device,
                acceleration=acceleration,
                seed=slice_seed,
            )

            # Real reverse diffusion inference from random noise
            recon = schedule.sample(
                model=model,
                condition=condition,
                device=device,
            )

            pred_np = recon[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            tgt_np = target[0, 0].cpu().numpy().astype(np.float32)
            zf_np = condition[0, 0].cpu().numpy().astype(np.float32)

            psnrs.append(calculate_psnr(tgt_np, pred_np))
            ssims.append(calculate_ssim(tgt_np, pred_np))
            nmses.append(calculate_nmse(tgt_np, pred_np))
            zf_psnrs.append(calculate_psnr(tgt_np, zf_np))

    return {
        "val_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "val_ssim": float(np.mean(ssims)) if ssims else 0.0,
        "val_nmse": float(np.mean(nmses)) if nmses else 0.0,
        "val_zf_psnr": float(np.mean(zf_psnrs)) if zf_psnrs else 0.0,
    }


def train_diffusion(
    train_dir: str,
    val_dir: str,
    acceleration: int = 4,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    timesteps: int = 100,
    subset_files: Optional[int] = 20,
    lambda_image: float = 1.0,
    lambda_kspace: float = 0.1,
    seed: int = 42,
    max_val_batches: int = 10,
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

    print("=" * 65)
    print("CONDITIONAL DIFFUSION MRI RECONSTRUCTION TRAINING")
    print("=" * 65)
    print(f"Acceleration    : {acceleration}x")
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

    model = ConditionalDiffusionUNet(base_channels=32, time_dim=128).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = DiffusionSchedule(timesteps=timesteps, device=torch_device)
    mask_func = RandomMaskFunc(
        center_fractions=[0.08 if acceleration == 4 else 0.04],
        accelerations=[acceleration],
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
            condition, target, mask_320, k_tgt_320 = prepare_batch(
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

            # Diffusion model predicts injected noise
            predicted_noise = model(noisy_target, condition, t)

            # 1. Noise prediction loss (DDPM objective)
            loss_noise = F.mse_loss(predicted_noise, noise)

            # 2. Tweedie differentiable predicted clean image
            x0_pred = schedule.predict_x0(noisy_target, predicted_noise, t, clamp=True)

            # 3. Image reconstruction loss (L1)
            loss_image = F.l1_loss(x0_pred, target)

            # 4. K-space fidelity loss in 320x320 Fourier domain
            if lambda_kspace > 0.0:
                k_pred_320 = image_to_kspace_torch(x0_pred.squeeze(1).to(torch.complex64))
                diff = mask_320 * (k_pred_320 - k_tgt_320)
                diff_mag = torch.sqrt(diff.real**2 + diff.imag**2 + 1e-12)
                ref_mag = torch.sqrt((mask_320 * k_tgt_320).real**2 + (mask_320 * k_tgt_320).imag**2 + 1e-12)
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