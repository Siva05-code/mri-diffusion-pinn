import os
import json
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_schedule import DiffusionSchedule
from src.data.undersampling import RandomMaskFunc
from src.data.preprocessing import align_and_normalize_slice
from src.physics.fft import kspace_to_image_torch
from src.evaluation.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_nmse,
)


# ============================================================
# Configuration
# ============================================================

ACCELERATION = 4
CENTER_FRACTION = 0.08
TIMESTEPS = 100

FILE_PATH = (
    "/kaggle/working/data/fastmri/"
    "knee_singlecoil_val/file1000196.h5"
)

SLICE_INDEX = 18

CHECKPOINT_PATH = (
    "checkpoints/diffusion_accel4_latest.pt"
)

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DIFFUSION MRI RECONSTRUCTION EVALUATION")
    print("=" * 60)
    print(f"Acceleration    : {ACCELERATION}x")
    print(f"Device           : {device}")
    print(f"Timesteps        : {TIMESTEPS}")
    print(f"Slice            : {SLICE_INDEX}")
    print("=" * 60)


    # ========================================================
    # 1. Load FastMRI sample
    # ========================================================

    with h5py.File(FILE_PATH, "r") as hf:

        kspace = hf["kspace"][SLICE_INDEX]

        target = hf["reconstruction_rss"][SLICE_INDEX]

    print("\nRaw data:")
    print("K-space:", kspace.shape, kspace.dtype)
    print("Target :", target.shape, target.dtype)


    # ========================================================
    # 2. Convert k-space to PyTorch tensor
    # ========================================================

    kspace_t = torch.from_numpy(kspace)


    # ========================================================
    # 3. Generate deterministic undersampling mask
    # ========================================================

    mask_func = RandomMaskFunc(
        center_fractions=[CENTER_FRACTION],
        accelerations=[ACCELERATION]
    )

    mask = mask_func(
        kspace_t.shape,
        seed=42
    )

    while mask.ndim > 2:
        mask = mask.squeeze(0)

    print("\nMask:")
    print("Shape:", mask.shape)


    # ========================================================
    # 4. Apply undersampling
    # ========================================================

    undersampled_kspace = (
        kspace_t * mask
    )

    print(
        "Undersampled k-space:",
        undersampled_kspace.shape,
        undersampled_kspace.dtype
    )


    # ========================================================
    # 5. Zero-filled reconstruction
    # ========================================================

    zf = kspace_to_image_torch(
        undersampled_kspace
    )

    print(
        "\nZF before magnitude:",
        zf.shape,
        zf.dtype
    )

    # Convert complex image to magnitude
    if torch.is_complex(zf):
        zf = torch.abs(zf)

    print(
        "ZF after magnitude:",
        zf.shape,
        zf.dtype
    )


    # ========================================================
    # 6. Align and normalize
    # ========================================================

    zf, target = align_and_normalize_slice(
        zf,
        target
    )

    print("\nAfter preprocessing:")
    print("ZF     :", zf.shape, zf.dtype)
    print("Target :", target.shape, target.dtype)


    # ========================================================
    # 7. Convert to [B,C,H,W]
    # ========================================================

    condition = (
        zf
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    target_tensor = (
        torch.from_numpy(target)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    print("\nModel input:")
    print("Condition:", condition.shape)
    print("Target   :", target_tensor.shape)


    # ========================================================
    # 8. Load diffusion model
    # ========================================================

    model = ConditionalDiffusionUNet(
        base_channels=32,
        time_dim=128
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nCheckpoint loaded:")
    print(CHECKPOINT_PATH)


    # ========================================================
    # 9. Diffusion schedule
    # ========================================================

    schedule = DiffusionSchedule(
        timesteps=TIMESTEPS
    )

    print(
        "Timesteps:",
        schedule.timesteps
    )


    # ========================================================
    # 10. Reverse diffusion
    # ========================================================

    print("\nStarting reverse diffusion...")

    with torch.no_grad():

        diffusion_recon = schedule.sample(
            model,
            condition,
            device
        )

    diffusion_recon = diffusion_recon.clamp(
        0.0,
        1.0
    )

    print(
        "Diffusion reconstruction:",
        diffusion_recon.shape
    )

    print(
        "Range:",
        diffusion_recon.min().item(),
        "to",
        diffusion_recon.max().item()
    )


    # ========================================================
    # 11. Convert to NumPy
    # ========================================================

    zf_np = (
        condition[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    diffusion_np = (
        diffusion_recon[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    target_np = (
        target_tensor[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    # ========================================================
    # 12. Calculate metrics
    # ========================================================

    psnr_zf = calculate_psnr(
        target_np,
        zf_np
    )

    ssim_zf = calculate_ssim(
        target_np,
        zf_np
    )

    nmse_zf = calculate_nmse(
        target_np,
        zf_np
    )


    psnr_diffusion = calculate_psnr(
        target_np,
        diffusion_np
    )

    ssim_diffusion = calculate_ssim(
        target_np,
        diffusion_np
    )

    nmse_diffusion = calculate_nmse(
        target_np,
        diffusion_np
    )


    # ========================================================
    # 13. Print results
    # ========================================================

    print("\n")
    print("=" * 60)
    print("4x DIFFUSION EVALUATION")
    print("=" * 60)

    print("\nZero-Filled:")
    print(f"PSNR : {psnr_zf:.4f} dB")
    print(f"SSIM : {ssim_zf:.4f}")
    print(f"NMSE : {nmse_zf:.4f}")

    print("\nDiffusion:")
    print(f"PSNR : {psnr_diffusion:.4f} dB")
    print(f"SSIM : {ssim_diffusion:.4f}")
    print(f"NMSE : {nmse_diffusion:.4f}")

    print("\nImprovement:")
    print(
        f"PSNR : "
        f"{psnr_diffusion - psnr_zf:+.4f} dB"
    )

    print(
        f"SSIM : "
        f"{ssim_diffusion - ssim_zf:+.4f}"
    )

    print(
        f"NMSE : "
        f"{nmse_diffusion - nmse_zf:+.4f}"
    )

    print("=" * 60)


    # ========================================================
    # 14. Error maps
    # ========================================================

    zf_error = np.abs(
        zf_np - target_np
    )

    diffusion_error = np.abs(
        diffusion_np - target_np
    )


    # ========================================================
    # 15. Visualization
    # ========================================================

    figure_path = os.path.join(
        FIGURES_DIR,
        "diffusion_accel4_sample.png"
    )

    plt.figure(
        figsize=(15, 8)
    )

    plt.subplot(2, 3, 1)
    plt.imshow(
        zf_np,
        cmap="gray"
    )
    plt.title(
        f"Zero-Filled\n"
        f"PSNR {psnr_zf:.2f} dB"
    )
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(
        diffusion_np,
        cmap="gray"
    )
    plt.title(
        f"Diffusion\n"
        f"PSNR {psnr_diffusion:.2f} dB"
    )
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(
        target_np,
        cmap="gray"
    )
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(
        zf_error,
        cmap="hot"
    )
    plt.title("ZF Error Map")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(
        diffusion_error,
        cmap="hot"
    )
    plt.title("Diffusion Error Map")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nFigure saved to:",
        figure_path
    )


    # ========================================================
    # 16. Save metrics
    # ========================================================

    metrics = {
        "acceleration": ACCELERATION,
        "slice_index": SLICE_INDEX,
        "zero_filled": {
            "psnr": float(psnr_zf),
            "ssim": float(ssim_zf),
            "nmse": float(nmse_zf)
        },
        "diffusion": {
            "psnr": float(psnr_diffusion),
            "ssim": float(ssim_diffusion),
            "nmse": float(nmse_diffusion)
        }
    }

    metrics_path = os.path.join(
        TABLES_DIR,
        "diffusion_accel4_sample_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        "Metrics saved to:",
        metrics_path
    )


if __name__ == "__main__":
    main()