"""
Joint Diffusion Pre-Reconstruction + PINN Refinement Training Engine.

Trains the composite pipeline:
    Undersampled k-space -> Zero-filled -> Conditional Diffusion -> Coarse Image
    -> PINN Refinement -> (Data Consistency) -> Final Image

Supervised by the composite physics-guided objective:
    L_total = L_noise + lambda_image * L_image + lambda_kspace * L_kspace + lambda_physics * L_physics

Validation uses the real multi-step inference path:
    model.sample(condition=zf, y_measured=y_meas, mask=mask)
and computes evaluation metrics for Zero-Filled, Coarse Diffusion, and Refined PINN outputs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FastMRIDataset
from src.data.preprocessing import align_and_normalize_slice, center_crop
from src.data.undersampling import RandomMaskFunc
from src.evaluation.metrics import calculate_nmse, calculate_psnr, calculate_ssim
from src.losses.total_loss import TotalReconstructionLoss
from src.models.diffusion_pinn import DiffusionPINN
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch


def prepare_training_batch(
    sample: dict,
    mask_func: RandomMaskFunc,
    device: torch.device,
    acceleration: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares a batch for joint Diffusion + PINN training:
    1. Undersamples k-space using mask_func
    2. Computes zero-filled reconstruction via IFFT
    3. Center-crops and normalizes both ZF and Target to [0, 1] at 320x320
    4. Computes matched 320x320 Fourier domain reference and sampled measurements
    """
    kspace = sample["kspace"].to(device)
    raw_target = sample.get("target")

    if not torch.is_complex(kspace):
        if kspace.shape[-1] == 2:
            kspace = torch.complex(kspace[..., 0], kspace[..., 1])
        else:
            kspace = kspace.to(torch.complex64)

    # 1. Generate mask & undersample
    mask = mask_func(kspace.shape, seed=seed).to(device)
    undersampled = kspace * mask

    # 2. Zero-filled reconstruction
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

    # 4. Spatial alignment and [0, 1] normalization
    if zf_mag.ndim == 2:
        zf_norm, tgt_norm = align_and_normalize_slice(zf_mag, tgt_mag)
        zf_out = zf_norm.unsqueeze(0).unsqueeze(0).float()
        tgt_out = tgt_norm.unsqueeze(0).unsqueeze(0).float()
    else:
        zf_list = []
        tgt_list = []
        for b in range(zf_mag.shape[0]):
            z_n, t_n = align_and_normalize_slice(zf_mag[b], tgt_mag[b])
            zf_list.append(z_n)
            tgt_list.append(t_n)
        zf_out = torch.stack(zf_list, dim=0).unsqueeze(1).float()
        tgt_out = torch.stack(tgt_list, dim=0).unsqueeze(1).float()

    # Guarantee strictly 4D [B, 1, H, W]
    while zf_out.ndim > 4:
        zf_out = zf_out.squeeze(1)
    while tgt_out.ndim > 4:
        tgt_out = tgt_out.squeeze(1)

    # 5. Common 320x320 Fourier domain representation
    k_tgt_320 = image_to_kspace_torch(tgt_out.squeeze(1).to(torch.complex64))
    mask_h = mask.shape[-2]
    mask_320 = center_crop(mask, (mask_h, tgt_out.shape[-1]))
    mask_320 = mask_320.to(k_tgt_320.device, dtype=k_tgt_320.real.dtype)
    y_meas_320 = k_tgt_320 * mask_320

    return zf_out, tgt_out, y_meas_320, k_tgt_320, mask_320


def validate_epoch(
    model: DiffusionPINN,
    val_loader: DataLoader,
    mask_func: RandomMaskFunc,
    criterion: TotalReconstructionLoss,
    device: torch.device,
    acceleration: int,
    seed: int = 42,
    max_val_batches: int = 10,
) -> Dict[str, float]:
    """
    Validates model performance on validation slices using the REAL inference path
    (model.sample) without using ground truth target during reconstruction.
    """
    model.eval()
    val_total_loss = 0.0
    pinn_psnrs, pinn_ssims, pinn_nmses = [], [], []
    diff_psnrs, diff_ssims, diff_nmses = [], [], []
    zf_psnrs, zf_ssims, zf_nmses = [], [], []

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            if i >= max_val_batches:
                break

            slice_seed = seed + int(sample.get("slice_idx", [i])[0])
            zf, target, y_meas, k_full, mask = prepare_training_batch(
                sample=sample,
                mask_func=mask_func,
                device=device,
                acceleration=acceleration,
                seed=slice_seed,
            )

            # Real multi-step reverse diffusion + PINN refinement inference path
            coarse, refined = model.sample(
                condition=zf,
                y_measured=y_meas,
                mask=mask,
                device=device,
            )

            loss_dict = criterion(
                i_pred=refined,
                i_gt=target,
                y_measured=y_meas,
                k_gt=k_full,
                mask=mask,
            )
            val_total_loss += loss_dict["total_loss"].item()

            target_np = target[0, 0].cpu().numpy().astype(np.float32)
            zf_np = zf[0, 0].cpu().numpy().astype(np.float32)
            coarse_np = coarse[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            refined_np = refined[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

            # Metrics for Zero-Filled
            zf_psnrs.append(calculate_psnr(target_np, zf_np))
            zf_ssims.append(calculate_ssim(target_np, zf_np))
            zf_nmses.append(calculate_nmse(target_np, zf_np))

            # Metrics for Coarse Diffusion
            diff_psnrs.append(calculate_psnr(target_np, coarse_np))
            diff_ssims.append(calculate_ssim(target_np, coarse_np))
            diff_nmses.append(calculate_nmse(target_np, coarse_np))

            # Metrics for Refined Diffusion+PINN
            pinn_psnrs.append(calculate_psnr(target_np, refined_np))
            pinn_ssims.append(calculate_ssim(target_np, refined_np))
            pinn_nmses.append(calculate_nmse(target_np, refined_np))

    num_samples = max(len(pinn_psnrs), 1)
    return {
        "val_total_loss": val_total_loss / num_samples,
        "val_psnr": float(np.mean(pinn_psnrs)) if pinn_psnrs else 0.0,
        "val_ssim": float(np.mean(pinn_ssims)) if pinn_ssims else 0.0,
        "val_nmse": float(np.mean(pinn_nmses)) if pinn_nmses else 0.0,
        "val_coarse_psnr": float(np.mean(diff_psnrs)) if diff_psnrs else 0.0,
        "val_coarse_ssim": float(np.mean(diff_ssims)) if diff_ssims else 0.0,
        "val_coarse_nmse": float(np.mean(diff_nmses)) if diff_nmses else 0.0,
        "val_zf_psnr": float(np.mean(zf_psnrs)) if zf_psnrs else 0.0,
        "val_zf_ssim": float(np.mean(zf_ssims)) if zf_ssims else 0.0,
        "val_zf_nmse": float(np.mean(zf_nmses)) if zf_nmses else 0.0,
    }


def train_diffusion_pinn(
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
    lambda_physics: float = 0.1,
    lambda_gradient: float = 0.1,
    enforce_dc: bool = True,
    ablation: str = "full",
    seed: int = 42,
    max_val_batches: int = 10,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Joint training function for Diffusion + PINN.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_device = torch.device(
        device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # Configure ablation weights
    if ablation == "diffusion_only":
        enable_pinn = False
        lambda_kspace = 0.0
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
    else:  # 'full'
        enable_pinn = True

    print("=" * 65)
    print("PHYSICS-INFORMED DIFFUSION-PINN MRI RECONSTRUCTION")
    print("=" * 65)
    print(f"Acceleration    : {acceleration}x")
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
    center_frac = 0.08 if acceleration == 4 else 0.04
    train_mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[acceleration], seed=seed)
    val_mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[acceleration], seed=seed + 9999)

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
            zf, target, y_meas, k_full, mask = prepare_training_batch(
                sample=sample,
                mask_func=train_mask_func,
                device=torch_device,
                acceleration=acceleration,
                seed=batch_seed,
            )

            # End-to-end forward pass through Diffusion + PINN (training denoising path)
            out = model.forward_train(
                target=target,
                condition=zf,
                y_measured=y_meas,
                mask=mask,
            )

            # If PINN disabled in ablation, evaluate loss on coarse image; else on refined image
            output_image = out["refined"] if enable_pinn else out["coarse"]

            # Compute composite physics-guided reconstruction loss
            loss_dict = criterion(
                i_pred=output_image,
                i_gt=target,
                y_measured=y_meas,
                k_gt=k_full,
                mask=mask,
            )

            # Add diffusion epsilon matching loss
            noise_loss = torch.mean((out["pred_noise"] - out["noise"]) ** 2)
            total_loss = loss_dict["total_loss"] + noise_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_total_loss_accum += total_loss.item()
            train_noise_loss_accum += noise_loss.item()
            train_img_loss_accum += loss_dict["image_loss"].item()
            train_ksp_loss_accum += loss_dict["kspace_loss"].item()
            train_phys_loss_accum += loss_dict["physics_loss"].item()

        num_train_batches = max(len(train_loader), 1)
        avg_train_total = train_total_loss_accum / num_train_batches
        avg_train_noise = train_noise_loss_accum / num_train_batches
        avg_train_img = train_img_loss_accum / num_train_batches
        avg_train_ksp = train_ksp_loss_accum / num_train_batches
        avg_train_phys = train_phys_loss_accum / num_train_batches

        # Validation using REAL inference path
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            mask_func=val_mask_func,
            criterion=criterion,
            device=torch_device,
            acceleration=acceleration,
            seed=seed,
            max_val_batches=max_val_batches,
        )

        epoch_duration = time.time() - epoch_start
        val_psnr = val_metrics["val_psnr"]
        val_ssim = val_metrics["val_ssim"]
        val_nmse = val_metrics["val_nmse"]

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Loss: {avg_train_total:.4f} (Noise: {avg_train_noise:.4f}, Img: {avg_train_img:.4f}, "
            f"Ksp: {avg_train_ksp:.4f}, Phys: {avg_train_phys:.4f}) | "
            f"Val Loss: {val_metrics['val_total_loss']:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB (Coarse: {val_metrics['val_coarse_psnr']:.2f}, ZF: {val_metrics['val_zf_psnr']:.2f}) | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"{epoch_duration:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_total_loss": avg_train_total,
            "train_noise_loss": avg_train_noise,
            "train_image_loss": avg_train_img,
            "train_kspace_loss": avg_train_ksp,
            "train_physics_loss": avg_train_phys,
            "validation_total_loss": val_metrics["val_total_loss"],
            "validation_PSNR": val_psnr,
            "validation_SSIM": val_ssim,
            "validation_NMSE": val_nmse,
            "validation_coarse_PSNR": val_metrics["val_coarse_psnr"],
            "validation_ZF_PSNR": val_metrics["val_zf_psnr"],
            "epoch_time": epoch_duration,
        }
        history.append(record)

        # Save latest checkpoint
        latest_ckpt = checkpoints_dir / f"diffusion_pinn_accel{acceleration}_latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acceleration": acceleration,
                "ablation": ablation,
                "val_psnr": val_psnr,
                "history": history,
            },
            latest_ckpt,
        )

        # Save best checkpoint selected by validation PSNR from the real inference path
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ckpt = checkpoints_dir / f"diffusion_pinn_accel{acceleration}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_psnr": best_psnr,
                    "val_ssim": val_ssim,
                    "val_nmse": val_nmse,
                    "acceleration": acceleration,
                    "ablation": ablation,
                },
                best_ckpt,
            )

    # Save log history
    log_file = logs_dir / f"diffusion_pinn_accel{acceleration}_history.json"
    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed.")
    print(f"Latest checkpoint : {latest_ckpt}")
    print(f"Best checkpoint   : {checkpoints_dir / f'diffusion_pinn_accel{acceleration}_best.pt'} (PSNR: {best_psnr:.2f} dB)")
    print(f"History saved to  : {log_file}")
    return {"history": history, "best_psnr": best_psnr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion + PINN MRI Model")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--accel", type=int, default=4, help="Acceleration factor (4 or 8)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=100, help="Diffusion timesteps")
    parser.add_argument("--subset_files", type=int, default=20, help="Subset of training HDF5 files")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "diffusion_only", "diffusion_kspace", "diffusion_physics", "diffusion_pinn_no_physics"],
        help="Ablation mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    args = parser.parse_args()

    # Load base config if present
    cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    train_dir = cfg.get("data", {}).get("train_dir", "/kaggle/working/data/fastmri/knee_singlecoil_train")
    val_dir = cfg.get("data", {}).get("val_dir", "/kaggle/working/data/fastmri/knee_singlecoil_val")
    phys_cfg = cfg.get("physics", {})

    train_diffusion_pinn(
        train_dir=train_dir,
        val_dir=val_dir,
        acceleration=args.accel,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        timesteps=args.timesteps,
        subset_files=args.subset_files,
        lambda_image=phys_cfg.get("lambda_image", 1.0),
        lambda_kspace=phys_cfg.get("lambda_kspace", 0.1),
        lambda_physics=phys_cfg.get("lambda_physics", 0.1),
        lambda_gradient=phys_cfg.get("lambda_gradient", 0.1),
        ablation=args.ablation,
        seed=args.seed,
        device=args.device,
    )
