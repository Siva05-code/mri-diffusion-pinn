"""
Diagnose the data pipeline. Run this BEFORE training, and whenever the
zero-filled baseline looks wrong.

    python scripts/diagnose_data.py --dir data/fastmri/val --config configs/accel4.yaml

It answers the questions that decide whether training can possibly work:

  * What is actually inside your .h5 files (keys, shapes, single vs multi-coil)?
  * Does the target image look sane (range, how much is clipped, how bright)?
  * How good is the zero-filled image at 4x and 8x, compared with the score you
    would get by predicting an all-zero image?

That last comparison is the key calibration. If zero-filled PSNR is close to the
"predict nothing" PSNR, the zero-filled image carries almost no information and
something upstream of the model is broken. Training cannot fix that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FastMRIDataset
from src.data.undersampling import effective_acceleration
from src.evaluation.metrics import nmse, psnr, ssim
from src.training.train_diffusion import load_config


def inspect_files(folder: Path, limit: int = 3) -> None:
    files = sorted(list(folder.rglob("*.h5")) + list(folder.rglob("*.hdf5")))
    print(f"\n== Files ==\nfound {len(files)} .h5 file(s) under {folder}")
    for path in files[:limit]:
        with h5py.File(path, "r") as f:
            print(f"\n  {path.name}")
            for key in f.keys():
                d = f[key]
                print(f"    {key:<22} shape={tuple(d.shape)}  dtype={d.dtype}")
            acq = f.attrs.get("acquisition", "?")
            print(f"    acquisition attr: {acq}")
            if "kspace" in f and f["kspace"].ndim == 4:
                print("    !! kspace has 4 dims: this is MULTI-COIL data. "
                      "This project needs the SINGLE-COIL knee set.")
            if "reconstruction_esc" not in f:
                print("    note: no 'reconstruction_esc'; target will be computed "
                      "from raw k-space.")


def study(args, cfg, accel: int):
    ds = FastMRIDataset(
        args.dir,
        image_size=cfg["image_size"],
        accelerations=[accel],
        center_fraction=cfg["center_fraction"] if accel == cfg["accelerations"][0]
        else (0.08 if accel == 4 else 0.04),
        mask_mode=cfg.get("mask_mode", "random"),
        max_slices=args.samples,
        slice_range=tuple(cfg.get("slice_range", [0.15, 0.95])),
        training=False,
        seed=cfg["seed"],
        cache=args.cache,
    )

    stats = {k: [] for k in
             ("mean", "mean_sq", "clipped", "dark", "p_zf", "s_zf", "n_zf",
              "p_zero", "eff", "zf_min", "zf_max")}
    first = None

    for i in range(len(ds)):
        item = ds[i]
        t, z, m = item["target"][None], item["zf"][None], item["mask"]
        zeros = torch.zeros_like(t)

        stats["mean"].append(float(t.mean()))
        stats["mean_sq"].append(float((t ** 2).mean()))
        stats["clipped"].append(float((t >= 0.999).float().mean()))
        stats["dark"].append(float((t < 0.01).float().mean()))
        stats["p_zf"].append(psnr(z, t))
        stats["s_zf"].append(ssim(z, t))
        stats["n_zf"].append(nmse(z, t))
        stats["p_zero"].append(psnr(zeros, t))
        stats["eff"].append(effective_acceleration(m[0].numpy()))
        stats["zf_min"].append(float(z.min()))
        stats["zf_max"].append(float(z.max()))
        if first is None:
            first = (t[0, 0].numpy(), z[0, 0].numpy(), m[0].numpy())

    return {k: float(np.mean(v)) for k, v in stats.items()}, first, len(ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="folder of .h5 files (ignored if --cache)")
    p.add_argument("--cache", default=None, help="optional .npy cache")
    p.add_argument("--config", default="configs/accel4.yaml")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--out", default="results/diagnose")
    args = p.parse_args()

    cfg = load_config(args.config)
    if not args.cache:
        inspect_files(Path(args.dir))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n== Zero-filled quality ==")
    header = (f"{'accel':>6}{'eff':>7}{'ZF PSNR':>10}{'ZF SSIM':>9}{'ZF NMSE':>9}"
              f"{'all-zero PSNR':>15}{'target mean':>13}{'clipped%':>10}")
    print(header)
    print("-" * len(header))

    verdicts = []
    first_images = None
    for accel in (4, 8):
        s, first, n = study(args, cfg, accel)
        first_images = first_images or first
        print(f"{accel:>5}x{s['eff']:>7.2f}{s['p_zf']:>10.2f}{s['s_zf']:>9.3f}"
              f"{s['n_zf']:>9.4f}{s['p_zero']:>15.2f}{s['mean']:>13.3f}"
              f"{100 * s['clipped']:>10.2f}")
        verdicts.append((accel, s))
    print(f"\n({n} slices per row)")

    print("\n== Verdict ==")
    ok = True
    for accel, s in verdicts:
        gap = s["p_zf"] - s["p_zero"]
        if s["n_zf"] > 0.5:
            ok = False
            print(f"[BAD ] {accel}x: ZF NMSE {s['n_zf']:.3f} -- the zero-filled image "
                  "carries almost none of the target. Upstream bug or wrong data.")
        elif gap < 6:
            ok = False
            print(f"[BAD ] {accel}x: ZF PSNR is only {gap:.1f} dB above an all-zero "
                  "image. Expected a much larger gap.")
        else:
            print(f"[ ok ] {accel}x: ZF is {gap:.1f} dB above an all-zero image.")

        if abs(s["eff"] - accel) > 0.25 * accel:
            ok = False
            print(f"[BAD ] {accel}x: effective acceleration is {s['eff']:.2f}, "
                  "not close to the requested value.")
        if s["clipped"] > 0.05:
            print(f"[warn] {accel}x: {100 * s['clipped']:.1f}% of target pixels are clipped "
                  "at 1.0 -- the intensity normalisation may be too aggressive.")
        if s["mean"] < 0.02:
            print(f"[warn] {accel}x: target mean is {s['mean']:.3f}; the images are "
                  "almost entirely dark, which compresses PSNR and makes it unreliable.")

    if ok:
        print("\nData pipeline looks healthy. A low model PSNR is then a training-"
              "length / data-volume issue, not a bug.")
    else:
        print("\nFix the data pipeline before training. Send this whole output back "
              "if you want help reading it.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t, z, m = first_images
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
        for a, img, title in [(ax[0], t, "target"), (ax[1], z, "zero-filled"),
                              (ax[2], np.abs(t - z), "|target - zf|"), (ax[3], m, "mask")]:
            a.imshow(img, cmap="gray")
            a.set_title(title)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / "look_at_this.png", dpi=100)
        print(f"\nImage saved -> {out / 'look_at_this.png'}  (open it and look)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
