import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import FastMRIDataset
from src.data.preprocessing import align_and_normalize_slice
from src.data.undersampling import RandomMaskFunc
from src.physics.fft import (
    kspace_to_image_torch,
)
from src.models.diffusion import (
    ConditionalDiffusionUNet,
)
from src.models.diffusion_schedule import (
    DiffusionSchedule,
)


def prepare_batch(sample, mask_func, device, acceleration):

    kspace = sample["kspace"].to(device)

    target = sample["target"].to(device)

    # Complex k-space
    if not torch.is_complex(kspace):
        kspace = torch.view_as_complex(kspace)

    mask = mask_func(
        kspace.shape,
    ).to(device)

    undersampled = kspace * mask

    zero_filled = kspace_to_image_torch(
        undersampled
    )

    zero_filled = torch.abs(zero_filled)

    target = torch.abs(target)

    zero_filled = align_and_normalize_slice(
        zero_filled
    )

    target = align_and_normalize_slice(
        target
    )

    return (
        zero_filled.unsqueeze(0),
        target.unsqueeze(0),
    )


def train_diffusion(
    train_dir,
    val_dir,
    acceleration=4,
    epochs=5,
    batch_size=1,
    learning_rate=1e-4,
    timesteps=100,
    subset_files=20,
    device="cuda",
):

    device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Conditional Diffusion MRI Reconstruction")
    print(f"Acceleration : {acceleration}x")
    print(f"Epochs       : {epochs}")
    print(f"Device       : {device}")
    print(f"Timesteps    : {timesteps}")
    print("=" * 60)

    train_dataset = FastMRIDataset(
        train_dir,
        max_files=subset_files,
    )

    val_dataset = FastMRIDataset(
        val_dir,
        max_files=10,
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

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

    model = ConditionalDiffusionUNet(
        base_channels=32,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    schedule = DiffusionSchedule(
        timesteps=timesteps,
        device=device,
    )

    mask_func = RandomMaskFunc(
        center_fractions=[0.08],
        accelerations=[acceleration],
    )

    history = []

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0
        start = time.time()

        for sample in train_loader:

            condition, target = prepare_batch(
                sample,
                mask_func,
                device,
                acceleration,
            )

            condition = condition.float()
            target = target.float()

            t = torch.randint(
                0,
                timesteps,
                (target.shape[0],),
                device=device,
            )

            noise = torch.randn_like(target)

            noisy_target = schedule.add_noise(
                target,
                noise,
                t,
            )

            predicted_noise = model(
                noisy_target,
                condition,
                t,
            )

            loss = torch.mean(
                (predicted_noise - noise) ** 2
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        train_loss = (
            total_loss / max(len(train_loader), 1)
        )

        elapsed = time.time() - start

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Time: {elapsed:.2f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "epoch_time": elapsed,
            }
        )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"diffusion_accel{acceleration}_latest.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "acceleration": acceleration,
            "timesteps": timesteps,
        },
        checkpoint_path,
    )

    log_dir = Path("results/logs")
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / f"diffusion_train_accel{acceleration}_history.json"
    )

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(
        f"Checkpoint saved to: {checkpoint_path}"
    )

    print(
        f"Training log saved to: {log_path}"
    )


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
        device="cuda",
    )