"""
Evaluate a trained checkpoint and save qualitative figures.

Reports PSNR / SSIM / NMSE / DC error for the model *and* for the zero-filled
baseline on identical masks, plus the standard deviation across slices, so a
claimed improvement can be judged against its own spread.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FastMRIDataset
from src.evaluation.metrics import MetricTracker, summarize
from src.physics.fft import dc_error
from src.training.train_diffusion import build_model, load_config


def save_figures(samples, out_dir, limit=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figures.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (zf, pred, target) in enumerate(samples[:limit]):
        err = np.abs(pred - target)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
        for a, img, title, kw in [
            (ax[0], target, "Ground truth", dict(cmap="gray", vmin=0, vmax=1)),
            (ax[1], zf, "Zero-filled", dict(cmap="gray", vmin=0, vmax=1)),
            (ax[2], pred, "Reconstruction", dict(cmap="gray", vmin=0, vmax=1)),
            (ax[3], err, "|error| x4", dict(cmap="inferno", vmin=0, vmax=0.25)),
        ]:
            a.imshow(img, **kw)
            a.set_title(title)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:02d}.png", dpi=110)
        plt.close(fig)
    print(f"Figures -> {out_dir}")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--inference_steps", type=int, default=0)
    p.add_argument("--figures", default="results/figures")
    p.add_argument("--out", default="results/evaluation.json")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    cfg = load_config(args.config)
    steps = args.inference_steps or cfg.get("inference_steps", 50)

    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    if "residual_scale" in ckpt:
        model.set_residual_scale(ckpt["residual_scale"])
    model.eval()

    ds = FastMRIDataset(
        args.val_dir,
        image_size=cfg["image_size"],
        accelerations=cfg.get("val_accelerations", cfg["accelerations"]),
        center_fraction=cfg["center_fraction"],
        mask_mode=cfg.get("mask_mode", "random"),
        residual_scale=float(model.residual_scale),
        max_slices=args.samples,
        slice_range=tuple(cfg.get("slice_range", [0.15, 0.95])),
        training=False,
        seed=cfg["seed"],
        cache=cfg.get("val_cache"),
    )
    loader = DataLoader(ds, batch_size=cfg.get("val_batch_size", 4), shuffle=False)

    model_t, zf_t = MetricTracker(), MetricTracker()
    figures = []

    for batch in tqdm(loader, desc="evaluate"):
        target = batch["target"].to(device)
        zf = batch["zf"].to(device)
        mask = batch["mask"].to(device)
        measured = batch["measured_kspace"].to(device)

        pred = model.reconstruct(
            zf, mask, measured,
            inference_steps=steps,
            eta=cfg.get("eta", 0.0),
        )

        model_t.update(summarize(pred, target, dc_error(pred, measured, mask)))
        zf_t.update(summarize(zf, target, dc_error(zf, measured, mask)))

        if len(figures) < 6:
            figures.append((
                zf[0, 0].cpu().numpy(),
                pred[0, 0].cpu().numpy(),
                target[0, 0].cpu().numpy(),
            ))

    model_mean, model_std = model_t.mean(), model_t.std()
    zf_mean = zf_t.mean()

    print(f"\nSlices evaluated: {len(ds)}   sampling steps: {steps}")
    print(f"{'Metric':<12}{'Zero-filled':>14}{'Model':>14}{'Delta':>12}")
    print("-" * 52)
    for k in ("psnr", "ssim", "nmse", "dc_error"):
        print(f"{k:<12}{zf_mean[k]:>14.5f}{model_mean[k]:>14.5f}"
              f"{model_mean[k] - zf_mean[k]:>+12.5f}")
    print(f"\nPSNR spread across slices: +/- {model_std['psnr']:.3f} dB")

    save_figures(figures, args.figures)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": len(ds),
                "inference_steps": steps,
                "model": model_mean,
                "model_std": model_std,
                "zero_filled": zf_mean,
            },
            f,
            indent=2,
        )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
