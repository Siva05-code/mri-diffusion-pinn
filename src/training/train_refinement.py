"""
Stage-2 training: the physics-guided refinement U-Net.

The diffusion model is frozen. For each batch the sampler produces a
reconstruction (this is the expensive part, so use few inference steps here),
and the refinement network learns the correction that is still missing, with a
final data-consistency projection applied afterwards.

Only run this once stage 1 already beats the zero-filled baseline. Training a
refinement network on top of a sampler that is not yet working means the
refinement learns to compensate for the sampler, and the ablation table can no
longer tell you which part actually helped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.evaluation.metrics import MetricTracker, summarize
from src.losses.image_loss import gradient_loss, l1_loss
from src.losses.kspace_loss import kspace_consistency_loss
from src.models.diffusion import EMA
from src.models.refinement import RefinementUNet
from src.physics.fft import dc_error
from src.training.train_diffusion import (
    build_datasets,
    build_loaders,
    build_model,
    load_config,
    seed_everything,
)


@torch.no_grad()
def _sample(model, zf, mask, measured, steps, eta):
    return model.reconstruct(
        zf, mask, measured, inference_steps=steps, eta=eta, use_data_consistency=True
    )


@torch.no_grad()
def evaluate_refinement(model, refine, loader, device, cfg, max_batches=0):
    refine.eval()
    base_t, ref_t = MetricTracker(), MetricTracker()

    for i, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches > 0 and i >= max_batches:
            break
        target = batch["target"].to(device)
        zf = batch["zf"].to(device)
        mask = batch["mask"].to(device)
        measured = batch["measured_kspace"].to(device)

        coarse = _sample(model, zf, mask, measured,
                         cfg.get("refine_inference_steps", 20), cfg.get("eta", 0.0))
        refined = refine(coarse, zf, mask, measured, apply_dc=True)

        base_t.update(summarize(coarse, target, dc_error(coarse, measured, mask)))
        ref_t.update(summarize(refined, target, dc_error(refined, measured, mask)))

    return base_t.mean(), ref_t.mean()


def run_refinement_training(
    config_path: str,
    train_dir: str,
    val_dir: str,
    diffusion_checkpoint: str,
    device: torch.device,
    output_dir: str = "checkpoints",
    smoke_test: bool = False,
):
    cfg = load_config(config_path)
    if smoke_test:
        cfg.update(
            refine_epochs=1,
            max_train_slices=8,
            max_val_slices=4,
            num_workers=0,
            refine_inference_steps=3,
            val_max_batches=1,
        )

    seed_everything(cfg["seed"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- frozen stage-1 model ----
    model = build_model(cfg, device)
    ckpt = torch.load(diffusion_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=True)
    if "residual_scale" in ckpt:
        model.set_residual_scale(ckpt["residual_scale"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    scale = float(model.residual_scale)
    cfg["residual_scale"] = scale

    train_ds, val_ds = build_datasets(cfg, train_dir, val_dir)
    train_ds.residual_scale = scale
    val_ds.residual_scale = scale
    train_loader, val_loader = build_loaders(cfg, train_ds, val_ds)

    refine = RefinementUNet(
        base=cfg.get("refine_base_channels", 32),
        depth=cfg.get("refine_depth", 3),
    ).to(device)
    ema = EMA(refine, cfg.get("ema_decay", 0.999))

    optimizer = AdamW(
        refine.parameters(),
        lr=cfg.get("refine_learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    epochs = cfg.get("refine_epochs", 10)
    steps = cfg.get("refine_inference_steps", 20)
    history = []
    best = -float("inf")

    for epoch in range(epochs):
        refine.train()
        t0, running = time.time(), 0.0

        for batch in tqdm(train_loader, desc=f"refine {epoch + 1}", leave=False):
            target = batch["target"].to(device)
            zf = batch["zf"].to(device)
            mask = batch["mask"].to(device)
            measured = batch["measured_kspace"].to(device)

            coarse = _sample(model, zf, mask, measured, steps, cfg.get("eta", 0.0))
            refined = refine(coarse, zf, mask, measured, apply_dc=True)

            loss = (
                l1_loss(refined, target)
                + cfg.get("lambda_gradient", 0.01) * gradient_loss(refined, target)
                + cfg.get("lambda_kspace", 0.1)
                * kspace_consistency_loss(refined, measured, mask)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refine.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()
            ema.update(refine)
            running += float(loss.detach())

        coarse_m, refined_m = evaluate_refinement(
            model, ema.shadow, val_loader, device, cfg,
            max_batches=cfg.get("val_max_batches", 16),
        )
        gain = refined_m["psnr"] - coarse_m["psnr"]
        print(
            f"epoch {epoch + 1}/{epochs}  loss={running / max(len(train_loader), 1):.4f}  "
            f"diffusion PSNR={coarse_m['psnr']:.3f} -> refined={refined_m['psnr']:.3f} "
            f"({gain:+.3f})  SSIM={refined_m['ssim']:.4f}  [{time.time() - t0:.0f}s]"
        )
        history.append({"epoch": epoch + 1, "coarse": coarse_m, "refined": refined_m})

        if refined_m["psnr"] > best:
            best = refined_m["psnr"]
            torch.save(
                {"model": refine.state_dict(), "ema": ema.state_dict(),
                 "epoch": epoch, "metrics": refined_m, "config": cfg},
                out_dir / "refinement_best.pt",
            )

    Path("results").mkdir(exist_ok=True)
    with open("results/refinement_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history
