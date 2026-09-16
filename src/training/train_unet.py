"""
Training Engine for U-Net Baseline on Accelerated MRI.

Supports reproducible seeding, Adam optimizer, modern mixed-precision training (torch.amp),
sanity checks against accidental validation-training contamination, checkpoint persistence,
and consistent spatial preprocessing.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.losses.total_loss import TotalReconstructionLoss
from src.data.dataset import FastMRIDataset
from src.data.undersampling import RandomMaskFunc
from src.data.preprocessing import complex_abs, align_and_normalize_slice
from src.physics.fft import kspace_to_image_torch
from src.evaluation.metrics import calculate_psnr, calculate_ssim, calculate_nmse


def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_grad_scaler(device: torch.device):
    """Instantiate GradScaler using modern torch.amp API."""
    if device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()
    return None


def get_autocast(device: torch.device):
    """Context manager for autocast using modern torch.amp API."""
    if device.type == "cuda":
        try:
            return torch.amp.autocast("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast()
    return nullcontext()


class SliceTransform:
    """
    Transforms raw k-space sample into (zero-filled input, reference target) pair.
    Handles dimension verification and consistent [0, 1] normalization.
    """
    def __init__(
        self,
        mask_func: RandomMaskFunc,
        seed: int = 42,
        crop_size: Optional[Tuple[int, int]] = (320, 320),
    ):
        self.mask_func = mask_func
        self.seed = seed
        self.crop_size = crop_size
        self._dim_verified = False

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        kspace = sample["kspace"]
        slice_seed = self.seed + int(sample.get("slice_idx", 0))
        mask = self.mask_func(kspace.shape, seed=slice_seed)

        # 1. Undersample k-space
        y_undersampled = mask * kspace

        # 2. Reconstruct zero-filled image via centered IFFT
        zf_complex = kspace_to_image_torch(y_undersampled)
        zf_mag = complex_abs(zf_complex)

        # 3. Obtain ground truth target
        if "target" in sample and sample["target"] is not None:
            target_mag = sample["target"]
        else:
            gt_complex = kspace_to_image_torch(kspace)
            target_mag = complex_abs(gt_complex)

        # 4. Consistent spatial preprocessing & normalization to standard (320, 320) FOV
        zf_norm, target_norm = align_and_normalize_slice(
            zf_mag,
            target_mag,
            crop_size=self.crop_size,
        )

        if not self._dim_verified:
            print(
                f"[Data Pipeline Verification] Raw k-space: {tuple(kspace.shape)} | "
                f"Zero-Filled input: {tuple(zf_norm.shape)} | "
                f"Target: {tuple(target_norm.shape)} | "
                f"(Dimensions match: {zf_norm.shape == target_norm.shape})"
            )
            self._dim_verified = True

        # Reshape to [1, H, W] for 2D Convolution
        zf_tensor = zf_norm.unsqueeze(0).float()
        target_tensor = target_norm.unsqueeze(0).float()

        return {
            "input": zf_tensor,
            "target": target_tensor,
            "mask": mask,
            "filename": sample.get("filename", ""),
            "slice_idx": sample.get("slice_idx", 0),
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any] = None,
) -> float:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    count = 0

    use_amp = scaler is not None and device.type == "cuda"

    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with get_autocast(device):
                pred = model(x)
                loss_dict = criterion(pred, y)
                loss = loss_dict.get("loss", loss_dict.get("total_loss"))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss_dict = criterion(pred, y)
            loss = loss_dict.get("loss", loss_dict.get("total_loss"))
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        count += x.size(0)

    return running_loss / max(1, count)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    psnr_vals = []
    ssim_vals = []
    nmse_vals = []
    count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)

            pred = model(x)
            loss_dict = criterion(pred, y)
            val_loss = loss_dict.get("loss", loss_dict.get("total_loss"))
            running_loss += val_loss.item() * x.size(0)
            count += x.size(0)

            y_np = y.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()

            psnr_vals.append(calculate_psnr(y_np, pred_np))
            ssim_vals.append(calculate_ssim(y_np, pred_np))
            nmse_vals.append(calculate_nmse(y_np, pred_np))

    return {
        "val_loss": running_loss / max(1, count),
        "val_psnr": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        "val_ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        "val_nmse": float(np.mean(nmse_vals)) if nmse_vals else 0.0,
    }


def train_unet(
    data_dir: str,
    val_data_dir: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    acceleration: int = 4,
    center_fraction: float = 0.08,
    subset_files: Optional[int] = None,
    device_str: Optional[str] = None,
    checkpoints_dir: str = "checkpoints",
    logs_dir: str = "results/logs",
    seed: int = 42,
    is_smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Main training execution function for U-Net baseline.
    """
    # Sanity check: Disallow training on validation data for real experiments
    if not is_smoke_test and val_data_dir:
        train_path_resolved = Path(data_dir).resolve()
        val_path_resolved = Path(val_data_dir).resolve()
        if train_path_resolved == val_path_resolved:
            raise ValueError(
                f"FATAL ERROR: Training directory '{data_dir}' is identical to validation directory '{val_data_dir}'.\n"
                f"Real scientific training must require a separate official fastMRI training directory.\n"
                f"If you intended to run a pipeline verification test, explicitly pass --smoke_test (is_smoke_test=True)."
            )

    set_seed(seed)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n========================================================")
    print(f"  Training U-Net Baseline ({acceleration}x Acceleration)")
    print(f"  Execution Device  : {device}")
    print(f"  Mode              : {'SMOKE TEST (Pipeline Verification Only)' if is_smoke_test else 'Standard Experiment'}")
    print(f"  Epochs            : {epochs}")
    print(f"  Learning Rate     : {learning_rate}")
    print(f"  Subset Files      : {subset_files if subset_files else 'Full'}")
    print(f"========================================================\n")

    # Undersampling mask generator
    mask_func = RandomMaskFunc(
        center_fractions=[center_fraction],
        accelerations=[acceleration],
        seed=seed,
    )
    transform = SliceTransform(mask_func=mask_func, seed=seed, crop_size=(320, 320))

    # Training Dataset
    train_dataset = FastMRIDataset(
        root_dir=data_dir,
        transform=transform,
        max_files=subset_files,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"No usable MRI slices found in {data_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"Training dataset loaded: {len(train_dataset)} slices across files.")

    # Validation Dataset
    val_loader = None
    if val_data_dir and Path(val_data_dir).exists():
        val_dataset = FastMRIDataset(
            root_dir=val_data_dir,
            transform=transform,
            max_files=max(1, (subset_files or 5) // 2),
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            print(f"Validation dataset loaded: {len(val_dataset)} slices.")

    # Model, Optimizer, Loss
    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)

    # U-Net Baseline uses pure image L1 loss (lambda_kspace=0.0 strictly maintained)
    criterion = TotalReconstructionLoss(
        image_loss_type="l1",
        lambda_image=1.0,
        lambda_kspace=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = create_grad_scaler(device)

    # Checkpoints & Logs setup
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    best_val_psnr = -float("inf")
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        metrics = {"epoch": epoch, "train_loss": train_loss, "epoch_time": round(time.time() - epoch_start, 2)}

        if val_loader is not None:
            val_metrics = validate_epoch(model, val_loader, criterion, device)
            metrics.update(val_metrics)
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_metrics['val_loss']:.5f} | "
                f"Val PSNR: {val_metrics['val_psnr']:.2f} dB | "
                f"Val SSIM: {val_metrics['val_ssim']:.3f} | "
                f"Time: {metrics['epoch_time']}s"
            )

            if val_metrics["val_psnr"] > best_val_psnr:
                best_val_psnr = val_metrics["val_psnr"]
                best_ckpt_path = Path(checkpoints_dir) / f"unet_accel{acceleration}_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_psnr": best_val_psnr,
                        "acceleration": acceleration,
                        "is_smoke_test": is_smoke_test,
                    },
                    best_ckpt_path,
                )
        else:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.5f} | Time: {metrics['epoch_time']}s")

        history.append(metrics)

    total_time = round(time.time() - start_time, 2)
    latest_ckpt_path = Path(checkpoints_dir) / f"unet_accel{acceleration}_latest.pt"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "acceleration": acceleration,
            "is_smoke_test": is_smoke_test,
        },
        latest_ckpt_path,
    )

    # Save training history
    history_file = Path(logs_dir) / f"unet_train_accel{acceleration}_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "acceleration": acceleration,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "subset_files": subset_files,
                "is_smoke_test": is_smoke_test,
                "total_time_seconds": total_time,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"\nTraining completed in {total_time}s.")
    print(f"Latest checkpoint saved to: {latest_ckpt_path}")
    print(f"Training log saved to     : {history_file}")

    return {
        "latest_checkpoint": str(latest_ckpt_path),
        "history_file": str(history_file),
        "history": history,
    }
