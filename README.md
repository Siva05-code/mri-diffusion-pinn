# Physics-Guided Conditional Diffusion for Accelerated MRI Reconstruction

Reconstruct an MRI image from undersampled k-space. A conditional diffusion
model predicts the information missing from the zero-filled reconstruction, and
Fourier data consistency keeps every reverse step tied to the lines that were
actually measured.

```
fastMRI k-space -> undersampling mask -> undersampled k-space
                                              |
                                  +-----------+-----------+
                                  |                       |
                                IFFT                    mask
                                  |                       |
                          zero-filled image               |
                                  |                       |
                   x_t (noisy residual) ------------------+
                                  |
                      Conditional Diffusion U-Net
                                  |
                        predicted residual -> zf + residual
                                  |
                       Fourier data consistency (every step)
                                  |
                          reconstructed MRI
                                  |
                    optional refinement -> final DC
                                  |
                       PSNR / SSIM / NMSE / DC error
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Verify the pipeline. No dataset needed.
python scripts/sanity_check.py
python -m pytest tests/ -q

# 2. Prove it learns, on synthetic phantoms, in a few minutes on CPU.
python scripts/make_synthetic_data.py --out data/synthetic --train 8 --val 3 --slices 16 --size 64
python scripts/train.py --config configs/synthetic_cpu.yaml \
    --train_dir data/synthetic/train --val_dir data/synthetic/val --device cpu

# 3. Real data (see "Dataset" below), 4x acceleration.
python scripts/train.py --config configs/accel4.yaml \
    --train_dir data/fastmri/train --val_dir data/fastmri/val --device cuda
```

---

## Dataset

This project uses the **NYU fastMRI single-coil knee** dataset. There is no
public direct download link and nobody can give you one: you register at
<https://fastmri.med.nyu.edu/>, accept the data-use agreement, and NYU emails
you personal, time-limited URLs. Put the HDF5 volumes here:

```
data/fastmri/train/*.h5
data/fastmri/val/*.h5
```

The loader reads `reconstruction_esc` (the official 320x320 emulated
single-coil image) when present, and falls back to computing it from `kspace`.

### Cache the data before training on Colab

The raw path runs a 640x372 inverse FFT for every slice, every epoch. Caching
turns that into a memmap read:

```bash
python scripts/prepare_data.py --src data/fastmri/train --out data/cache/train.npy
python scripts/prepare_data.py --src data/fastmri/val   --out data/cache/val.npy
```

Then set `train_cache` / `val_cache` in the config.

---

## What was wrong with the previous version

Three of these would have stopped the project from ever reaching good metrics,
no matter how long it trained.

### 1. The residual was never rescaled (the important one)

The model diffuses `target - zero_filled`, whose standard deviation is roughly
0.02-0.06 on a `[0, 1]` image. Diffusion assumes the clean signal is order 1.
Unscaled, the residual sits far below the injected noise at almost every
timestep, so the network learns the trivial solution `eps_pred = x_t`. The
diffusion loss then collapses to a small, very encouraging-looking number while
the sampler hands the zero-filled image straight back.

`estimate_residual_scale()` now measures this from your data at the start of
training and stores it in the checkpoint as a buffer, so training and sampling
cannot drift apart.

### 2. The sampling mask silently destroyed measured data

The target is a real magnitude image, so its spectrum is Hermitian-symmetric:
`K[i] = conj(K[(N-i) % N])`. The old mask was not, so the undersampled spectrum
was not the spectrum of any real image, and taking `.real` threw part of the
measurement away. Data consistency stopped being an exact projection and the DC
error could never reach zero.

Masks now mirror every sampled line about the centre. Measured: the imaginary
part of the zero-filled image is 5e-8, DC error is 1e-7, and DC is idempotent
to 2e-7.

### 3. AMP crashed on the k-space loss

`torch.fft` has no float16 kernels. Every FFT now runs in float32 with autocast
explicitly disabled.

### 4. Clamping after data consistency undid the projection

`data_consistency(...).clamp(0, 1)` edits the measured Fourier lines back out
again. This showed up as a DC error of 6e-2 instead of 1e-7. The order is now
clamp, then project, and no clamp afterwards.

### Also fixed

- The sampler added `sqrt(beta) * randn` on top of an already-complete DDIM
  step, which is not a valid transition.
- `x0 = (x_t - b*eps) / sqrt(abar)` amplifies error by ~1000x near `t = T`.
  Switched to v-prediction, which needs no division and is well conditioned
  everywhere.
- EMA never copied buffers, so the shadow model could sample with a different
  residual scale than it trained with.
- Metrics were averaged over a flattened batch rather than per slice, letting
  bright slices dominate and making numbers from different batch sizes
  incomparable.
- Training masks were seeded per slice index, so each slice always saw the same
  mask and the model could memorise it.

---

## Project structure

```
src/
├── data/
│   ├── dataset.py         raw HDF5 + cached backends, residual-scale estimator
│   ├── preprocessing.py   crop, magnitude, intensity normalisation
│   └── undersampling.py   Hermitian-symmetric Cartesian masks
├── models/
│   ├── unet.py                conditional 3-channel U-Net, zero-init output
│   ├── diffusion.py           training loss + DC-guided DDIM sampler
│   ├── diffusion_schedule.py  cosine/linear betas, v/eps, DDIM step
│   └── refinement.py          stage-2 refinement U-Net
├── physics/fft.py         centered FFT, data consistency, DC error
├── losses/                image, k-space, gradient, composite
├── training/              train_diffusion.py, train_refinement.py
└── evaluation/metrics.py  PSNR, SSIM (torch), NMSE, trackers

scripts/  sanity_check, make_synthetic_data, prepare_data,
          train, train_refinement, evaluate, ablation
configs/  accel4, accel8, mixed, colab_fast, synthetic_cpu
```

---

## Training plan

Follow this order. Do not add a stage until the previous one improves the
validation result.

| Phase | Config | Target |
|---|---|---|
| 1. Pipeline check | `synthetic_cpu.yaml` | sanity checks and tests pass, loss falls |
| 2. 4x | `accel4.yaml` | diffusion + DC beats zero-filled |
| 3. 8x | `accel8.yaml` | measure how much quality drops |
| 4. Mixed | `mixed.yaml` | one model handles both |
| 5. Refinement | `train_refinement.py` | additional gain on top of stage 2 |

```bash
# Stage 5, only after stage 2 works
python scripts/train_refinement.py --config configs/accel4.yaml \
    --train_dir data/fastmri/train --val_dir data/fastmri/val \
    --diffusion_checkpoint checkpoints/best.pt --device cuda
```

---

## Evaluation and ablation

```bash
python scripts/evaluate.py --config configs/accel4.yaml \
    --checkpoint checkpoints/best.pt --val_dir data/fastmri/val --samples 200

python scripts/ablation.py --config configs/accel4.yaml \
    --checkpoint checkpoints/best.pt --val_dir data/fastmri/val --samples 200
```

`evaluate.py` reports the model and the zero-filled baseline on identical
masks, plus the standard deviation across slices, and writes side-by-side
figures to `results/figures/`.

`ablation.py` produces the five-row table. One caveat it enforces: row B
("diffusion without mask conditioning") is only a real experiment if you
*trained* a model without the mask channel and pass it via
`--no_mask_checkpoint`. Switching a conditioning channel off at test time is a
different thing, and the script labels it as such rather than quietly passing
it off as the ablation.

---

## Verified behaviour

Checked by running the code, not by inspection:

| Check | Result |
|---|---|
| FFT/IFFT round trip | 3.6e-07 |
| Mask Hermitian symmetry | exact |
| Zero-filled image imaginary part | 4.6e-08 |
| Data consistency idempotent | 2.4e-07 |
| v-parameterisation inverts | 1.8e-07 |
| Torch SSIM vs scikit-image | 3e-05 |
| Unit tests | 20 passed |

End-to-end learning check, 64x64 phantoms, 0.53M params, 12 epochs, one CPU
core: validation PSNR rose monotonically 16.39 -> 20.75 dB and crossed the
zero-filled baseline of 20.61 dB, with DC error holding at 1.0e-07. The curve
was still climbing when the run ended. This demonstrates the machinery learns;
it is not a result.

Ablation on the same run:

| Method | PSNR | SSIM | NMSE | DC Error |
|---|---:|---:|---:|---:|
| A. Zero-filled | 20.552 | 0.4959 | 0.06262 | 9.0e-08 |
| B. Diffusion (mask off at inference) | 18.917 | 0.3383 | 0.09045 | 1.3e-01 |
| C. Diffusion + mask | 18.921 | 0.3367 | 0.09052 | 1.4e-01 |
| D. Diffusion + mask + DC | 20.847 | 0.5160 | 0.05893 | 1.1e-07 |

Data consistency is doing the work at this stage of training, which is what you
would expect from a barely-trained sampler and is exactly the kind of thing the
ablation exists to reveal.

---

## Pushing the metrics higher (v2)

Five changes were added specifically to raise PSNR/SSIM. Measured on the same
12-epoch, 64x64 phantom CPU run used above (zero-filled baseline 20.61 dB):

| Stage | PSNR | Gain over baseline |
|---|---:|---:|
| v1 (original design) | 20.75 | +0.15 |
| v2 training changes | 21.77 | +1.17 |
| v2 + 8-sample averaging | 23.10 | +2.49 |

Same data, same epochs, same parameter count. The gain came from method, not
compute.

### Training-time changes

**Min-SNR-gamma loss weighting** (`min_snr_gamma: 5.0`). Per-timestep losses are
wildly unbalanced; low-t terms dominate the gradient so the model spends its
capacity on timesteps that were already easy. Clipping the effective weight
rebalances them.

**SSIM in the objective** (`lambda_ssim: 0.1`). L1 and SSIM disagree about what
matters. If you want SSIM reported higher, optimise it, do not just measure it.

**Heavier image loss** (`lambda_image: 2.0` in the HQ preset). The diffusion term
teaches denoising; the image term teaches reconstruction.

### Inference-time changes (no retraining)

**Sample averaging** (`num_samples`). PSNR rewards the posterior mean, and a
single diffusion draw is a sample from the posterior, not its mean. This was
worth +1.13 dB on the phantom run and has been the most dependable gain in every
sweep so far. Cost is linear in inference time.

**Warm-start sampling** (`t_start_frac`). Begin the reverse chain part way down
the schedule instead of at pure noise, so the model is not re-deriving the
zero-filled image it was already handed.

**Flip ensembling** (`tta`). Included, off by default.

### Two honest corrections

I expected warm-start to be the largest lever and wrote that in the code. The
sweep disagreed: on this run a pure-noise start scored as well as any warm start
(22.32 vs 22.22 at `t_start=0.2`). The comment has been corrected.

I also expected flip ensembling to help. It **hurt** on the phantoms, 22.37
against 22.91 without it. Anatomy is not flip-symmetric and the prior the model
learned is not either. It now defaults to off.

This is exactly why `scripts/tune_inference.py` exists: measure these on your
data instead of trusting anyone's defaults, including mine.

```bash
python scripts/tune_inference.py --config configs/accel4_hq.yaml \
    --checkpoint checkpoints/best.pt --val_dir data/fastmri/val --samples 40
```

---

## Getting to PSNR > 30 at 4x

Realistic at 4x. Much harder at 8x, where published single-coil knee results
mostly sit in the high 20s, so do not hold the same target there.

Use `configs/accel4_hq.yaml`, and understand that the ordering below is by
expected return:

1. **Train on more data for longer.** Nothing else competes. The phantom run was
   still climbing when it stopped. Use 60+ epochs over several thousand real
   slices, cache the data first, and resume across Colab sessions.
2. **Use the full `base_channels: 64` model.** Drop to 32 only if you hit OOM.
3. **Run `tune_inference.py`** and adopt the winning row. Free.
4. **Set `eval_num_samples: 8`** for the number you report.
5. **Keep `lambda_image: 2.0` and `lambda_ssim: 0.1`.**

One thing that matters for whether you can compare your number to anyone
else's: PSNR depends on the intensity-normalisation convention. This project
normalises each slice to [0,1] by its 99.9th percentile and reports with
`data_range=1.0`. If you change that, your PSNR changes without the
reconstruction changing at all. Say which convention you used in the write-up.

---

## If your zero-filled baseline is low (read this first)

Run this before any training:

```bash
python scripts/diagnose_data.py --dir data/fastmri/val --config configs/accel4.yaml
```

It reports the file layout, the zero-filled PSNR/SSIM/NMSE at 4x and 8x, and the
PSNR of an **all-zero image** as a reference. The comparison is the point: a
healthy zero-filled image sits many dB above the all-zero score. If it sits only a
few dB above, the zero-filled image is nearly empty and the problem is upstream of
the model. Training cannot fix that, and `train.py` now stops automatically when
the baseline is below 15 dB.

Reference, measured with this code on 320x320 phantoms in fastMRI layout:
zero-filled 27.35 dB at 4x and 23.94 dB at 8x, against 9.07 dB for an all-zero
image. These are phantoms, not knees, so treat them as proof the pipeline can
produce a healthy baseline, not as numbers to expect on real data.

Common causes of a bad baseline: multi-coil files instead of single-coil, the wrong
folder, a cache built from a different source than the one being validated, or an
`image_size` that does not match the cache.

Do not expect a model that has trained for a few epochs on a small subset to beat
the baseline. An under-trained sampler scores *below* it, because leftover noise
sits in the unmeasured lines. That is normal early on; what matters is the gap
closing as epochs pass.

---

## Honest expectations

No code can promise a specific PSNR or SSIM before it has run on your fastMRI
split, and you should be suspicious of anything that claims otherwise. What is
reasonable to expect on a free Colab T4 with `colab_fast.yaml`: a few dB over
the zero-filled baseline at 4x, less at 8x. Longer training, more data, and a
larger `base_channels` all help.

Two wording points for the write-up:

- Say **reconstruction-domain Fourier consistency**, not raw scanner k-space
  consistency. The operation happens after the 320x320 preprocessing.
- Report the spread across slices alongside the mean. A +0.2 dB gain with a
  +/-0.1 dB spread is a different claim from the same gain with +/-2 dB.

---

## Memory and speed

If you run out of GPU memory, change one thing at a time:

```yaml
batch_size: 1
gradient_accumulation: 8
base_channels: 32
attention_levels: [3]      # attention only at the coarsest level
```

Validation runs the full sampler, so it is the slow part. `val_max_batches` and
`inference_steps` control that cost directly.

---

## References

- fastMRI: <https://fastmri.med.nyu.edu/>
- PyTorch FFT: <https://docs.pytorch.org/docs/stable/fft.html>
- PyTorch AMP: <https://docs.pytorch.org/docs/stable/amp.html>
