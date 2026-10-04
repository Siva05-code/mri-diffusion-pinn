"""Stage-1 training loop: conditional diffusion + Fourier data consistency."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import FastMRIDataset, estimate_residual_scale
from src.evaluation.metrics import MetricTracker, summarize
from src.losses.composite_loss import LossWeights
from src.models.diffusion import DiffusionReconstructor, EMA
from src.physics.fft import dc_error


# ----------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> DiffusionReconstructor:
    return DiffusionReconstructor(
        image_size=cfg["image_size"],
        diffusion_steps=cfg["diffusion_steps"],
        base_channels=cfg["base_channels"],
        channel_mult=tuple(cfg["channel_mult"]),
        time_dim=cfg["time_dim"],
        attention_levels=tuple(cfg.get("attention_levels", [2, 3])),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        dropout=cfg.get("dropout", 0.0),
        schedule=cfg.get("beta_schedule", "cosine"),
        prediction_type=cfg.get("prediction_type", "v"),
        residual_scale=cfg.get("residual_scale", 0.1) or 0.1,
        use_mask_conditioning=cfg.get("use_mask_conditioning", True),
        min_snr_gamma=cfg.get("min_snr_gamma", 5.0),
    ).to(device)


def build_datasets(cfg: dict, train_dir: str, val_dir: str):
    common = dict(
        image_size=cfg["image_size"],
        center_fraction=cfg["center_fraction"],
        mask_mode=cfg.get("mask_mode", "random"),
        slice_range=tuple(cfg.get("slice_range", [0.15, 0.95])),
        seed=cfg["seed"],
    )
    train_ds = FastMRIDataset(
        train_dir,
        accelerations=cfg["accelerations"],
        max_files=cfg.get("max_train_files", 0),
        max_slices=cfg.get("max_train_slices", 0),
        training=True,
        cache=cfg.get("train_cache"),
        **common,
    )
    val_ds = FastMRIDataset(
        val_dir,
        accelerations=cfg.get("val_accelerations", cfg["accelerations"]),
        max_files=cfg.get("max_val_files", 0),
        max_slices=cfg.get("max_val_slices", 0),
        training=False,
        cache=cfg.get("val_cache"),
        **common,
    )
    return train_ds, val_ds


def build_loaders(cfg: dict, train_ds, val_ds):
    nw = cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader


# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: DiffusionReconstructor,
    loader: DataLoader,
    device: torch.device,
    inference_steps: int = 50,
    eta: float = 0.0,
    max_batches: int = 0,
    use_data_consistency: bool = True,
    include_baseline: bool = False,
    progress: bool = True,
    t_start_frac: float = 0.6,
    num_samples: int = 1,
    tta: bool = False,
) -> dict[str, float]:
    model.eval()
    tracker = MetricTracker()
    baseline = MetricTracker()

    iterator = enumerate(loader)
    if progress:
        total = max_batches if max_batches > 0 else len(loader)
        iterator = enumerate(tqdm(loader, total=total, desc="eval", leave=False))

    for i, batch in iterator:
        if max_batches > 0 and i >= max_batches:
            break

        target = batch["target"].to(device)
        zf = batch["zf"].to(device)
        mask = batch["mask"].to(device)
        measured = batch["measured_kspace"].to(device)

        pred = model.reconstruct(
            zf,
            mask,
            measured,
            inference_steps=inference_steps,
            eta=eta,
            use_data_consistency=use_data_consistency,
            t_start_frac=t_start_frac,
            num_samples=num_samples,
            tta=tta,
        )

        tracker.update(summarize(pred, target, dc_error(pred, measured, mask)))
        if include_baseline:
            baseline.update(summarize(zf, target, dc_error(zf, measured, mask)))

    out = tracker.mean()
    if include_baseline:
        out.update({f"zf_{k}": v for k, v in baseline.mean().items()})
    return out


# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, ema, cfg, device, weights, epoch):
    model.train()
    accum = max(1, cfg.get("gradient_accumulation", 1))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    running = {}
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"epoch {epoch + 1}", leave=False)
    for step, batch in enumerate(pbar):
        target = batch["target"].to(device, non_blocking=True)
        zf = batch["zf"].to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        measured = batch["measured_kspace"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            losses = model.training_loss(
                target, zf, residual, mask, measured, weights
            )
            loss = losses["total"] / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.detach())
        pbar.set_postfix(loss=f"{float(losses['total'].detach()):.4f}")

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


# ----------------------------------------------------------------------
def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, metrics, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg,
            "residual_scale": float(model.residual_scale),
        },
        path,
    )


def run_training(
    config_path: str,
    train_dir: str,
    val_dir: str,
    device: torch.device,
    smoke_test: bool = False,
    output_dir: str = "checkpoints",
    resume: str | None = None,
):
    cfg = load_config(config_path)

    if smoke_test:
        cfg.update(
            epochs=1,
            max_train_slices=min(cfg.get("max_train_slices") or 16, 16),
            max_val_slices=min(cfg.get("max_val_slices") or 4, 4),
            max_train_files=min(cfg.get("max_train_files") or 2, 2),
            max_val_files=min(cfg.get("max_val_files") or 1, 1),
            inference_steps=4,
            val_max_batches=2,
            num_workers=0,
        )

    seed_everything(cfg["seed"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(cfg, train_dir, val_dir)

    # --- residual scale calibration -----------------------------------
    scale = cfg.get("residual_scale", 0)
    if not scale:
        print("Estimating residual scale from training data...")
        scale = estimate_residual_scale(
            train_ds, num_samples=min(128, len(train_ds))
        )
        print(f"  residual_scale = {scale:.5f}")
    train_ds.residual_scale = scale
    val_ds.residual_scale = scale
    cfg["residual_scale"] = float(scale)

    train_loader, val_loader = build_loaders(cfg, train_ds, val_ds)

    model = build_model(cfg, device)
    model.set_residual_scale(scale)
    ema = EMA(model, cfg.get("ema_decay", 0.999))

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"train slices={len(train_ds)}  val slices={len(val_ds)}  "
        f"params={n_params / 1e6:.2f}M  device={device}"
    )

    weights = LossWeights(
        image=cfg.get("lambda_image", 1.0),
        kspace=cfg.get("lambda_kspace", 0.1),
        gradient=cfg.get("lambda_gradient", 0.01),
        ssim=cfg.get("lambda_ssim", 0.0),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-5),
        betas=(0.9, 0.999),
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, cfg.get("gradient_accumulation", 1)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        total_steps=max(cfg["epochs"] * steps_per_epoch, 2),
        pct_start=cfg.get("warmup_pct", 0.05),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=20.0,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg.get("amp", True)) and device.type == "cuda"
    )

    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {resume} at epoch {start_epoch}")

    # --- baseline before training -------------------------------------
    print("Zero-filled baseline on the validation subset:")
    base = evaluate_model(
        ema.shadow,
        val_loader,
        device,
        inference_steps=2,
        max_batches=cfg.get("val_max_batches", 20),
        include_baseline=True,
        progress=False,
    )
    print(
        f"  ZF   PSNR={base['zf_psnr']:.3f}  SSIM={base['zf_ssim']:.4f}  "
        f"NMSE={base['zf_nmse']:.5f}"
    )
    baseline_psnr = base["zf_psnr"]

    # --- stop early if the data pipeline is obviously broken -----------------
    # A healthy zero-filled image is far above an empty one. If it is not, no
    # amount of training will rescue the run, and it is better to stop in
    # seconds than to burn a Colab session finding out. Override with
    # `allow_low_baseline: true` only if you know why the baseline is low.
    min_ok = float(cfg.get("min_baseline_psnr", 15.0))
    if baseline_psnr < min_ok and not cfg.get("allow_low_baseline", False):
        raise SystemExit(
            f"\nSTOPPED: zero-filled baseline PSNR is {baseline_psnr:.2f} dB "
            f"(threshold {min_ok:.0f}).\n"
            "That is close to what an almost-empty image would score, so the data "
            "going into the model is wrong, not the model.\n"
            "Run:  python scripts/diagnose_data.py --dir <your val folder> "
            "--config <your config>\n"
            "and fix what it reports. To train anyway, set "
            "`allow_low_baseline: true` in the config."
        )

    history = []
    best_psnr = -float("inf")

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        losses = train_one_epoch(
            model, train_loader, optimizer, scaler, ema, cfg, device, weights, epoch
        )
        for _ in range(steps_per_epoch):
            if scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()

        metrics = evaluate_model(
            ema.shadow,
            val_loader,
            device,
            inference_steps=cfg.get("inference_steps", 50),
            eta=cfg.get("eta", 0.0),
            max_batches=cfg.get("val_max_batches", 20),
            t_start_frac=cfg.get("t_start_frac", 0.6),
            num_samples=cfg.get("val_num_samples", 1),
        )

        gain = metrics["psnr"] - baseline_psnr
        print(
            f"epoch {epoch + 1}/{cfg['epochs']}  "
            f"loss={losses['total']:.4f} (diff={losses['diffusion']:.4f} "
            f"img={losses['image']:.4f})  "
            f"PSNR={metrics['psnr']:.3f} ({gain:+.2f} vs ZF)  "
            f"SSIM={metrics['ssim']:.4f}  NMSE={metrics['nmse']:.5f}  "
            f"DC={metrics['dc_error']:.2e}  [{time.time() - t0:.0f}s]"
        )

        history.append({"epoch": epoch + 1, **losses, **metrics})
        save_checkpoint(out_dir / "latest.pt", model, ema, optimizer, scheduler, epoch, metrics, cfg)

        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            save_checkpoint(out_dir / "best.pt", model, ema, optimizer, scheduler, epoch, metrics, cfg)

    Path("results").mkdir(exist_ok=True)
    with open("results/train_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {"baseline_psnr": baseline_psnr, "history": history, "config": cfg},
            f,
            indent=2,
        )

    print(f"\nBest validation PSNR: {best_psnr:.3f} (zero-filled baseline {baseline_psnr:.3f})")
    return history
