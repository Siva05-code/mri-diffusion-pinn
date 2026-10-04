"""
Sweep the inference-time settings on a trained checkpoint.

None of these require retraining, and together they are usually worth more than
another few epochs. Run this once after training, pick the row you like, then
put those values in your config for the final evaluation.

    python scripts/tune_inference.py --config configs/accel4.yaml \
        --checkpoint checkpoints/best.pt --val_dir data/fastmri/val --samples 40

Reading the output:
  t_start  where the reverse chain begins. Lower keeps more of the zero-filled
           image and usually raises PSNR; higher is more generative and may look
           sharper while scoring worse.
  steps    DDIM steps. Past a point this stops helping and only costs time.
  avg      samples averaged. PSNR rewards the posterior mean and one draw is not
           the mean, so this reliably helps -- at linear cost.
  tta      flip ensembling, 4x cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FastMRIDataset
from src.evaluation.metrics import MetricTracker, summarize
from src.physics.fft import dc_error
from src.training.train_diffusion import build_model, load_config


@torch.no_grad()
def run(model, loader, device, **kw):
    tracker = MetricTracker()
    for batch in loader:
        target = batch["target"].to(device)
        zf = batch["zf"].to(device)
        mask = batch["mask"].to(device)
        measured = batch["measured_kspace"].to(device)
        pred = model.reconstruct(zf, mask, measured, **kw)
        tracker.update(summarize(pred, target, dc_error(pred, measured, mask)))
    return tracker.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--quick", action="store_true", help="skip the expensive rows")
    p.add_argument("--out", default="results/inference_sweep.json")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    cfg = load_config(args.config)

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

    steps = cfg.get("inference_steps", 30)
    configs = [
        ("pure noise start (old default)", dict(t_start_frac=1.0, inference_steps=steps)),
        ("t_start 0.8", dict(t_start_frac=0.8, inference_steps=steps)),
        ("t_start 0.6", dict(t_start_frac=0.6, inference_steps=steps)),
        ("t_start 0.4", dict(t_start_frac=0.4, inference_steps=steps)),
        ("t_start 0.2", dict(t_start_frac=0.2, inference_steps=steps)),
    ]
    if not args.quick:
        configs += [
            ("best t_start + avg 4", dict(t_start_frac=0.4, inference_steps=steps, num_samples=4)),
            ("best t_start + avg 8", dict(t_start_frac=0.4, inference_steps=steps, num_samples=8)),
            ("best t_start + avg 4 + tta", dict(t_start_frac=0.4, inference_steps=steps, num_samples=4, tta=True)),
            ("more steps (2x)", dict(t_start_frac=0.4, inference_steps=steps * 2, num_samples=4)),
        ]

    header = f"{'setting':<34}{'PSNR':>8}{'SSIM':>9}{'NMSE':>10}{'DC':>10}{'sec':>7}"
    print("\n" + header)
    print("-" * len(header))

    rows = {}
    for label, kw in configs:
        t0 = time.time()
        m = run(model, loader, device, **kw)
        rows[label] = {**m, "settings": kw, "seconds": time.time() - t0}
        print(f"{label:<34}{m['psnr']:>8.3f}{m['ssim']:>9.4f}"
              f"{m['nmse']:>10.5f}{m['dc_error']:>10.1e}{time.time() - t0:>7.1f}")

    best = max(rows.items(), key=lambda kv: kv[1]["psnr"])
    print(f"\nBest: {best[0]}  ->  PSNR {best[1]['psnr']:.3f}")
    print(f"Put these in your config: {best[1]['settings']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
