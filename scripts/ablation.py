"""
Ablation study: isolate the contribution of each component.

Rows produced (matching the project plan):

    A  Zero-filled                                  baseline
    B  Diffusion, no mask channel, no DC
    C  Diffusion + mask conditioning, no DC
    D  Diffusion + mask + Fourier data consistency
    E  Full model + refinement                      (if --refinement given)

Rows B and C are only meaningful if the corresponding checkpoint was actually
trained that way. Passing --no_mask_checkpoint scores a model trained without
mask conditioning; without it, row B is reported as "mask channel disabled at
inference only" and is labelled as such, because switching a conditioning
channel off at test time is not the same experiment as training without it.

Usage:
    python scripts/ablation.py --config configs/accel4.yaml \
        --checkpoint checkpoints/best.pt --val_dir data/fastmri/val --samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FastMRIDataset
from src.evaluation.metrics import MetricTracker, summarize
from src.models.refinement import RefinementUNet
from src.physics.fft import dc_error
from src.training.train_diffusion import build_model, load_config


def load_reconstructor(cfg, checkpoint, device):
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state, strict=True)
    if "residual_scale" in ckpt:
        model.set_residual_scale(ckpt["residual_scale"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def score(model, loader, device, cfg, use_mask, use_dc, refine=None, label=""):
    tracker = MetricTracker()
    original = model.use_mask_conditioning
    model.use_mask_conditioning = use_mask

    try:
        for batch in tqdm(loader, desc=label, leave=False):
            target = batch["target"].to(device)
            zf = batch["zf"].to(device)
            mask = batch["mask"].to(device)
            measured = batch["measured_kspace"].to(device)

            pred = model.reconstruct(
                zf, mask, measured,
                inference_steps=cfg.get("inference_steps", 50),
                eta=cfg.get("eta", 0.0),
                use_data_consistency=use_dc,
            )
            if refine is not None:
                pred = refine(pred, zf, mask, measured, apply_dc=True)

            tracker.update(summarize(pred, target, dc_error(pred, measured, mask)))
    finally:
        model.use_mask_conditioning = original

    return tracker.mean()


@torch.no_grad()
def score_zero_filled(loader, device):
    tracker = MetricTracker()
    for batch in tqdm(loader, desc="zero-filled", leave=False):
        target = batch["target"].to(device)
        zf = batch["zf"].to(device)
        mask = batch["mask"].to(device)
        measured = batch["measured_kspace"].to(device)
        tracker.update(summarize(zf, target, dc_error(zf, measured, mask)))
    return tracker.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--no_mask_checkpoint", default=None,
                   help="checkpoint of a model trained WITHOUT mask conditioning")
    p.add_argument("--refinement", default=None, help="refinement checkpoint")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/ablation.json")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    cfg = load_config(args.config)

    model, ckpt = load_reconstructor(cfg, args.checkpoint, device)

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

    rows = {}
    rows["A. Zero-filled"] = score_zero_filled(loader, device)

    if args.no_mask_checkpoint:
        nm_cfg = dict(cfg)
        nm_cfg["use_mask_conditioning"] = False
        nm_model, _ = load_reconstructor(nm_cfg, args.no_mask_checkpoint, device)
        rows["B. Diffusion (trained without mask)"] = score(
            nm_model, loader, device, cfg, use_mask=False, use_dc=False, label="B")
    else:
        rows["B. Diffusion (mask channel off at inference)"] = score(
            model, loader, device, cfg, use_mask=False, use_dc=False, label="B")

    rows["C. Diffusion + mask"] = score(
        model, loader, device, cfg, use_mask=True, use_dc=False, label="C")
    rows["D. Diffusion + mask + DC"] = score(
        model, loader, device, cfg, use_mask=True, use_dc=True, label="D")

    if args.refinement:
        refine = RefinementUNet(
            base=cfg.get("refine_base_channels", 32),
            depth=cfg.get("refine_depth", 3),
        ).to(device)
        r_ckpt = torch.load(args.refinement, map_location=device, weights_only=False)
        refine.load_state_dict(r_ckpt.get("ema") or r_ckpt["model"])
        refine.eval()
        rows["E. Full model + refinement"] = score(
            model, loader, device, cfg, use_mask=True, use_dc=True,
            refine=refine, label="E")

    # ---- report ----
    header = f"{'Method':<46}{'PSNR':>8}{'SSIM':>9}{'NMSE':>11}{'DC Error':>12}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in rows.items():
        print(
            f"{name:<46}{m['psnr']:>8.3f}{m['ssim']:>9.4f}"
            f"{m['nmse']:>11.5f}{m['dc_error']:>12.2e}"
        )

    base = rows["A. Zero-filled"]["psnr"]
    print(f"\nGain over zero-filled baseline (PSNR, dB):")
    for name, m in rows.items():
        if not name.startswith("A."):
            print(f"  {name:<46}{m['psnr'] - base:+.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"samples": len(ds), "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
