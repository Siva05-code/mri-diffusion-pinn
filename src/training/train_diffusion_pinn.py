"""
Joint End-to-End Training for Diffusion + PINN Accelerated MRI Reconstruction.

Pipeline:
1. Load raw fastMRI k-space.
2. Generate undersampling mask (deterministic for validation, random for train).
3. Undersample k-space: y = M * kspace.
4. Generate zero-filled image via centered IFFT.
5. Forward pass through Conditional Diffusion -> coarse image.
6. Forward pass through PINN refinement -> refined final image.
7. Compute composite multi-domain loss:
       L_total = lambda_image * L_image + alpha * L_kspace + beta * L_physics
8. Backpropagate end-to-end through BOTH Diffusion and PINN networks simultaneously.
9. Validate periodically reporting validation loss, PSNR, SSIM, and NMSE.
10. Save checkpoints and structured training log JSON.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FastMRIDataset
from src.data.preprocessing import align_and_normalize_slice
from src.data.undersampling import RandomMaskFunc
from src.physics.fft import kspace_to_image_torch
from src.models.diffusion_pinn import DiffusionPINN
from src.losses.total_loss import TotalReconstructionLoss
from src.evaluation.metrics import calculate_psnr, calculate_ssim, calculate_nmse


def prepare_training_batch(
    sample: dict,
    mask_func: RandomMaskFunc,
    device: torch.device,
    acceleration: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts, undersamples, and aligns a batch sample for joint training.

    Returns:
        zero_filled: [B, 1, 320, 320]
        target: [B, 1, 320, 320]
        undersampled_kspace: [B, H, W] (complex)
        kspace: [B, H, W] (complex)
        mask: [1, 1, W] or broadcastable
    """
    kspace = sample["kspace"].to(device)
    raw_target = sample.get("target")

    if not torch.is_complex(kspace):
        if kspace.shape[-1] == 2:
            kspace = torch.complex(kspace[..., 0], kspace[..., 1])
        else:
            kspace = kspace.to(torch.complex64)

    # 1. Generate mask
    mask = mask_func(kspace.shape, seed=seed).to(device)

    # 2. Undersampled k-space
    undersampled = kspace * mask

    # 3. Zero-filled reconstruction
    zf_complex = kspace_to_image_torch(undersampled)
    zf_mag = torch.abs(zf_complex)

    # 4. Target reference
    if raw_target is not None:
        tgt_mag = raw_target.to(device)
        if torch.is_complex(tgt_mag):
            tgt_mag = torch.abs(tgt_mag)
    else:
        full_complex = kspace_to_image_torch(kspace)
        tgt_mag = torch.abs(full_complex)

    # 5. Spatial alignment and [0, 1] normalization
    # Handle single slice or batched
    if zf_mag.ndim == 2:
        zf_norm, tgt_norm = align_and_normalize_slice(zf_mag, tgt_mag)
        zf_out = zf_norm.unsqueeze(0).unsqueeze(0).float()
        tgt_out = tgt_norm.unsqueeze(0).unsqueeze(0).float()
    else:
        # Batched slices
        zf_list = []
        tgt_list = []
        for b in range(zf_mag.shape[0]):
            z_n, t_n = align_and_normalize_slice(zf_mag[b], tgt_mag[b])
            zf_list.append(z_n.unsqueeze(0))
            tgt_list.append(t_n.unsqueeze(0))
        zf_out = torch.stack(zf_list, dim=0).unsqueeze(1).float()
        tgt_out = torch.stack(tgt_list, dim=0).unsqueeze(1).float()

    return zf_out, tgt_out, undersampled, kspace, mask


def validate_epoch(
    model: DiffusionPINN,
    val_loader: DataLoader,
    mask_func: RandomMaskFunc,
    criterion: TotalReconstructionLoss,
    device: torch.device,
    acceleration: int,
    seed: int = 42,
    max_val_batches: int = 20,
) -> Dict[str, float]:
    """Validates model performance on validation slices."""
    model.eval()
    val_total_loss = 0.0
    psnrs = []
    ssims = []
    nmses = []

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

            # Sampling or train-eval mode
            out_dict = model.forward_train(
                target=target,
                condition=zf,
                y_measured=y_meas,
                mask=mask,
            )
            refined = out_dict["refined"]

            loss_dict = criterion(
                i_pred=refined,
                i_gt=target,
                y_measured=y_meas,
                k_gt=k_full,
                mask=mask,
            )
            val_total_loss += loss_dict["total_loss"].item()

            pred_np = refined[0, 0].clamp(0.0, 1.0).cpu().numpy()
            target_np = target[0, 0].cpu().numpy()

            psnrs.append(calculate_psnr(target_np, pred_np))
            ssims.append(calculate_ssim(target_np, pred_np))
            nmses.append(calculate_nmse(target_np, pred_np))

    num_samples = max(len(psnrs), 1)
    return {
        "val_total_loss": val_total_loss / num_samples,
        "val_psnr": float(np.mean(psnrs)),
        "val_ssim": float(np.mean(ssims)),
        "val_nmse": float(np.mean(nmses)),
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
    lambda_kspace: float = 1.0,
    lambda_physics: float = 1.0,
    lambda_gradient: float = 0.1,
    enforce_dc: bool = True,
    ablation: str = "full",
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Joint training function for Diffusion + PINN.
    """
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

    # 2. Model
    model = DiffusionPINN(
        diffusion_base_channels=32,
        diffusion_time_dim=128,
        pinn_base_channels=32,
        pinn_num_blocks=4,
        timesteps=timesteps,
        enforce_dc=enforce_dc,
    ).to(torch_device)

    # If ablation disables PINN, freeze PINN parameters
    if not enable_pinn:
        for p in model.pinn.parameters():
            p.requires_grad = False

    # 3. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # 4. Composite Loss Criterion
    criterion = TotalReconstructionLoss(
        image_loss_type="l1",
        lambda_image=lambda_image,
        lambda_kspace=lambda_kspace,
        lambda_physics=lambda_physics,
        lambda_gradient=lambda_gradient,
    ).to(torch_device)

    # 5. Mask Generator
    center_fraction = 0.08 if acceleration == 4 else 0.04
    train_mask_func = RandomMaskFunc(
        center_fractions=[center_fraction],
        accelerations=[acceleration],
    )
    val_mask_func = RandomMaskFunc(
        center_fractions=[center_fraction],
        accelerations=[acceleration],
        seed=42,
    )

    history = []
    best_psnr = -float("inf")

    # Output paths
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()

        train_total_loss_accum = 0.0
        train_img_loss_accum = 0.0
        train_ksp_loss_accum = 0.0
        train_phys_loss_accum = 0.0

        for sample in train_loader:
            zf, target, y_meas, k_full, mask = prepare_training_batch(
                sample=sample,
                mask_func=train_mask_func,
                device=torch_device,
                acceleration=acceleration,
            )

            # End-to-end forward pass through Diffusion + PINN
            out = model.forward_train(
                target=target,
                condition=zf,
                y_measured=y_meas,
                mask=mask,
            )

            # If PINN disabled in ablation, evaluate loss on coarse image; else on refined image
            output_image = out["refined"] if enable_pinn else out["coarse"]

            # Compute composite loss
            loss_dict = criterion(
                i_pred=output_image,
                i_gt=target,
                y_measured=y_meas,
                k_gt=k_full,
                mask=mask,
            )

            # Add diffusion epsilon matching loss to guide pre-reconstruction
            noise_loss = torch.mean((out["pred_noise"] - out["noise"]) ** 2)
            total_loss = loss_dict["total_loss"] + noise_loss

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_total_loss_accum += loss_dict["total_loss"].item()
            train_img_loss_accum += loss_dict["image_loss"].item()
            train_ksp_loss_accum += loss_dict["kspace_loss"].item()
            train_phys_loss_accum += loss_dict["physics_loss"].item()

        num_train_batches = max(len(train_loader), 1)
        avg_train_total = train_total_loss_accum / num_train_batches
        avg_train_img = train_img_loss_accum / num_train_batches
        avg_train_ksp = train_ksp_loss_accum / num_train_batches
        avg_train_phys = train_phys_loss_accum / num_train_batches

        # Validation
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            mask_func=val_mask_func,
            criterion=criterion,
            device=torch_device,
            acceleration=acceleration,
        )

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Train Loss: {avg_train_total:.4f} "
            f"(Img: {avg_train_img:.4f}, Ksp: {avg_train_ksp:.4f}, Phys: {avg_train_phys:.4f}) | "
            f"Val Loss: {val_metrics['val_total_loss']:.4f} | "
            f"Val PSNR: {val_metrics['val_psnr']:.2f} dB | "
            f"Val SSIM: {val_metrics['val_ssim']:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_total_loss": avg_train_total,
            "train_image_loss": avg_train_img,
            "train_kspace_loss": avg_train_ksp,
            "train_physics_loss": avg_train_phys,
            "validation_total_loss": val_metrics["val_total_loss"],
            "validation_PSNR": val_metrics["val_psnr"],
            "validation_SSIM": val_metrics["val_ssim"],
            "validation_NMSE": val_metrics["val_nmse"],
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
                "history": history,
            },
            latest_ckpt,
        )

        # Save best checkpoint
        if val_metrics["val_psnr"] > best_psnr:
            best_psnr = val_metrics["val_psnr"]
            best_ckpt = checkpoints_dir / f"diffusion_pinn_accel{acceleration}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_psnr": best_psnr,
                    "acceleration": acceleration,
                    "ablation": ablation,
                },
                best_ckpt,
            )

    # Save log history
    log_file = logs_dir / f"diffusion_pinn_accel{acceleration}_history.json"
    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed. History saved to: {log_file}")
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
    parser.add_argument("--ablation", type=str, default="full", choices=[
        "full", "diffusion_only", "diffusion_kspace", "diffusion_physics", "diffusion_pinn_no_physics"
    ], help="Ablation mode")
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
        lambda_kspace=phys_cfg.get("lambda_kspace", 1.0),
        lambda_physics=phys_cfg.get("lambda_physics", 1.0),
        lambda_gradient=phys_cfg.get("lambda_gradient", 0.1),
        ablation=args.ablation,
        device=args.device,
    )

