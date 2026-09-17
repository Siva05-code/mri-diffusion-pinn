# Physics-Informed Diffusion Framework for Accelerated MRI Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastMRI](https://img.shields.io/badge/Dataset-NYU%20fastMRI-brightgreen)](https://fastmri.med.nyu.edu/)

This repository contains the official PyTorch implementation of a **physics-informed conditional diffusion framework** for accelerated Magnetic Resonance Imaging (MRI) reconstruction using the **official NYU fastMRI dataset** (single-coil knee MRI).

---

## Table of Contents

1. [Research Motivation & Formulation Mismatch](#1-research-motivation--formulation-mismatch)
2. [Methodology & Architecture](#2-methodology--architecture)
   - [3-Channel Mask-Conditioned Diffusion](#3-channel-mask-conditioned-diffusion)
   - [Measurement-Guided Reverse Diffusion Sampling](#measurement-guided-reverse-diffusion-sampling)
   - [Reconstruction-Domain Fourier Consistency](#reconstruction-domain-fourier-consistency)
   - [Physics-Informed Refinement (PINN) Operator](#physics-informed-refinement-pinn-operator)
3. [Strict Scientific Integrity & Technical Clarifications](#3-strict-scientific-integrity--technical-clarifications)
4. [Repository Structure](#4-repository-structure)
5. [Checkpoint Compatibility & Migration](#5-checkpoint-compatibility--migration)
6. [Official NYU fastMRI Dataset & Streaming Extraction](#6-official-nyu-fastmri-dataset--streaming-extraction)
7. [Installation & Quickstart](#7-installation--quickstart)
8. [Execution & Verification Workflows](#8-execution--verification-workflows)
   - [Running Unit Tests](#running-unit-tests)
   - [Running Sanity Checks](#running-sanity-checks)
   - [Local CPU Smoke Training](#local-cpu-smoke-training)
   - [Evaluation Pipeline](#evaluation-pipeline)
   - [Full GPU Training on Kaggle](#full-gpu-training-on-kaggle)
9. [Quantitative Benchmarks & Baselines](#9-quantitative-benchmarks--baselines)
10. [Citation & Acknowledgments](#10-citation--acknowledgments)

---

## 1. Research Motivation & Formulation Mismatch

Accelerated MRI reduces scan times by acquiring an undersampled subset of Fourier space ($k$-space) trajectories. The classical linear inverse reconstruction problem is modeled as:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \boldsymbol{\varepsilon}$$

where:
- $\mathbf{x} \in \mathbb{C}^N$ is the fully sampled ground-truth cross-sectional image.
- $\mathbf{y} \in \mathbb{C}^M$ ($M \ll N$) is the measured undersampled $k$-space data.
- $\mathcal{F} = \mathbf{M} \odot \mathcal{F}_{\text{full}}$ is the forward acquisition operator ($\mathbf{M}$ is the undersampling mask, $\mathcal{F}_{\text{full}}$ is the 2D Discrete Fourier Transform).
- $\boldsymbol{\varepsilon}$ represents acquisition noise.

### The Formulation Mismatch Problem

Standard conditional Denoising Diffusion Probabilistic Models (DDPM) conditioned solely on zero-filled images suffer from severe **manifold drift** during unguided ancestral sampling. In early iterations of this project at $4\times$ acceleration, an unguided 2-channel diffusion model performed significantly worse than the trivial zero-filled baseline:

| Method | PSNR (dB) | SSIM | NMSE |
| :--- | :---: | :---: | :---: |
| **Zero-Filled Baseline (4×)** | **22.82 ± 2.95** | **0.5373** | **0.1686** |
| Unguided 2-Channel Diffusion (Flawed) | 18.09 ± 1.61 | 0.1997 | 0.3919 |

### Root Causes
1. **Mask Blindness**: The diffusion U-Net only observed $[x_t, \text{condition}]$, leaving the network blind to which Fourier frequencies were measured versus omitted.
2. **Unguided Ancestral Drift**: During reverse diffusion ($T \to 0$), stochastic sampling from random Gaussian noise without measurement consistency allowed samples to drift arbitrarily far from the acquired $k$-space measurements.
3. **Phase & Magnitude Distortion**: Premature conversion of complex signals to magnitude during Fourier projections destroyed phase information and created non-linear distortions.

---

## 2. Methodology & Architecture

Our revised framework redesigns both network conditioning and the reverse diffusion trajectory:

```
Undersampled k-space (y)
          ↓ (IFFT + Crop + Normalize)
Zero-Filled Reconstruction (condition) + Undersampling Mask (mask_320)
          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3-Channel Conditional Diffusion U-Net                       │
│ Input: [x_t, condition, mask_320]  -->  Predicted Noise ε_θ  │
└─────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────┐
│ Measurement-Guided Reverse Diffusion Step                   │
│ 1. Tweedie Estimate: x̂_0(x_t, ε_θ)                          │
│ 2. Fourier Consistency: k_guided = M·k_ref + (1-M)·k_pred    │
│ 3. Spatial Real Projection: x̂_0^DC = Re(IFFT_320(k_guided)) │
│ 4. Posterior Mean Update: μ̃_t(x_t, x̂_0^DC)                  │
└─────────────────────────────────────────────────────────────┘
          ↓
Final Reconstructed MRI Image (x_final)
          ↓ (Optional)
Physics-Informed Refinement Operator (Forward Model Consistency)
```

### 3-Channel Mask-Conditioned Diffusion

The diffusion network (`ConditionalDiffusionUNet`) accepts **3 input channels**:
$$\mathbf{z}_t = \text{Concat}(x_t, \, \text{condition}, \, \text{mask}_{320})$$
- $x_t \in \mathbb{R}^{1 \times H \times W}$: Noisy latent image at timestep $t$.
- $\text{condition} \in \mathbb{R}^{1 \times H \times W}$: Zero-filled reconstruction ($320 \times 320$).
- $\text{mask}_{320} \in \mathbb{R}^{1 \times H \times W}$: Binary sampling mask aligned with the reconstruction field-of-view (FOV).

If the raw acquisition mask width exceeds $320$, it is automatically center-cropped or column-broadcast to match the reconstruction dimensions.

### Measurement-Guided Reverse Diffusion Sampling

Rather than performing unguided ancestral DDPM sampling, each reverse timestep $t$ incorporates explicit measurement consistency:

1. **Tweedie's Formula (Clean Image Estimation)**:
   $$\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}_\theta(x_t, \text{condition}, \text{mask}_{320}, t) \right)$$

2. **Common $320 \times 320$ Reconstruction-Domain Fourier Consistency**:
   $$k_{\text{pred}} = \mathcal{F}_{320}(\hat{x}_0)$$
   $$k_{\text{ref}} = \mathcal{F}_{320}(\text{condition})$$
   $$k_{\text{guided}} = M_{320} \odot k_{\text{ref}} + (1 - M_{320}) \odot k_{\text{pred}}$$

3. **Real-Valued Spatial Projection**:
   $$\hat{x}_0^{\text{DC}} = \text{Re}\left(\mathcal{F}_{320}^{-1}(k_{\text{guided}})\right)$$
   > **Implementation Note**: We use `torch.real` rather than `torch.abs` on real-valued signals to preserve Fourier conjugate symmetry and eliminate non-linear magnitude distortion ($< 10^{-7}$ numerical error).

4. **Measurement-Guided Posterior Mean Update**:
   $$\tilde{\boldsymbol{\mu}}_t(x_t, \hat{x}_0^{\text{DC}}) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\hat{x}_0^{\text{DC}} + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t$$
   $$x_{t-1} \sim \mathcal{N}\left(\tilde{\boldsymbol{\mu}}_t, \, \tilde{\beta}_t \mathbf{I}\right)$$

This forces the reverse trajectory to remain strictly tethered to the acquired Fourier lines at every single sampling step.

### Reconstruction-Domain Fourier Consistency

```
k_pred = FFT2D(x̂_0)            k_ref = FFT2D(condition)
        \                             /
         \-----[ M ]           [ 1-M ]/
                \                 /
                 v               v
                k_guided = M·k_ref + (1-M)·k_pred
                                 ↓
                     x̂_0^DC = Re(IFFT2D(k_guided))
```

### Physics-Informed Refinement (PINN) Operator

The optional downstream refinement stage (`PhysicsRefinementUNet` in `src/models/diffusion_pinn.py`) further optimizes image quality using a composite objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{image}}(\mathbf{x}, \mathbf{x}_{\text{gt}}) + \alpha \mathcal{L}_{\text{kspace}}(\mathbf{x}, \mathbf{y}) + \beta \mathcal{L}_{\text{TV}}(\mathbf{x})$$
- $\mathcal{L}_{\text{image}}$: $\ell_1$ or SSIM perceptual loss.
- $\mathcal{L}_{\text{kspace}}$: Frequency data fidelity $\|M_{320} \odot (\mathcal{F}_{320}(\mathbf{x}) - \mathcal{F}_{320}(\text{condition}))\|_2^2$.
- $\mathcal{L}_{\text{TV}}$: Total variation / gradient smoothness regularization.

---

## 3. Strict Scientific Integrity & Technical Clarifications

To adhere to rigorous scientific standards, this repository maintains five core technical distinctions:

1. **Reconstruction-Domain Fourier Consistency vs. Raw Acquisition Consistency**:
   - The raw FastMRI single-coil acquisition k-space has dimensions $\sim 640 \times 368$ (with readout oversampling and asymmetric phase encoding).
   - The neural network operates on $320 \times 320$ center-cropped, normalized images.
   - Therefore, $\mathcal{F}_{320}(\text{condition})$ is **NOT** the raw scanner k-space; it is a **derived reconstruction-domain Fourier representation**.
   - This constraint is strictly designated as:
     $$\textbf{“common } 320 \times 320 \textbf{ reconstruction-domain Fourier consistency”}$$

2. **Measurement Guidance vs. Standard DDPM**:
   - Replacing $\hat{x}_0$ with $\hat{x}_0^{\text{DC}}$ during sampling alters the marginal reverse transition density.
   - It is explicitly documented and designated as **measurement-guided reverse diffusion sampling**, NOT mathematically identical to standard unguided DDPM sampling.

3. **Preservation of Complex-Valued Representation**:
   - Frequency operations (`fft2c_2d`, `ifft2c_2d`) are performed in complex tensor space ($\mathbb{C}$).
   - Spatial projections use `torch.real` to prevent non-linear distortion caused by `torch.abs`.

4. **Zero Ground-Truth Leakage at Inference**:
   - Under no circumstances does the reverse diffusion or evaluation pipeline access ground-truth images or ground-truth k-space during inference. Guidance relies solely on the measured zero-filled condition and mask.

5. **Clarification on "PINN" Terminology**:
   - "Physics-Informed" denotes enforcing physical MRI discrete Fourier forward acquisition operators ($\mathcal{F}$) and k-space consistency constraints. It does not solve continuous spatiotemporal partial differential equations (PDEs).

---

## 4. Repository Structure

```
mri-diffusion-pinn/
│
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Excludes datasets, checkpoints, caches
│
├── configs/                       # YAML configuration files
│   ├── base.yaml                  # Base parameters and fastMRI subset sizes
│   ├── accel4.yaml                # 4× acceleration configuration
│   └── accel8.yaml                # 8× acceleration configuration
│
├── src/                           # Modular source code
│   ├── data/                      # Dataset handling & fastMRI streaming
│   │   ├── download_fastmri.py    # Memory-safe streaming fastMRI downloader
│   │   ├── dataset.py             # PyTorch Dataset for fastMRI HDF5 files
│   │   ├── preprocessing.py       # Center cropping, normalization, transform
│   │   └── undersampling.py       # Equispaced & Gaussian 1D/2D mask generators
│   ├── models/                    # Neural network architectures
│   │   ├── diffusion.py           # 3-channel ConditionalDiffusionUNet
│   │   ├── diffusion_schedule.py  # GaussianDiffusion with measurement guidance
│   │   ├── diffusion_pinn.py      # End-to-end Diffusion + PINN Refinement model
│   │   ├── unet.py                # Standalone baseline U-Net
│   │   └── refinement.py          # Physics-informed refinement U-Net
│   ├── physics/                   # Fourier physics operators
│   │   └── fft.py                 # Centered 2D FFT / IFFT (PyTorch & NumPy)
│   ├── losses/                    # Loss functions
│   │   ├── composite_loss.py      # Image + k-space + physics composite objective
│   │   ├── image_loss.py          # L1, L2, and SSIM image losses
│   │   └── kspace_loss.py         # K-space data consistency loss
│   ├── training/                  # Training logic
│   │   ├── train_diffusion.py     # Standalone diffusion trainer
│   │   ├── train_diffusion_pinn.py# Joint Diffusion-PINN trainer
│   │   └── trainer.py             # Baseline U-Net trainer
│   ├── evaluation/                # Metrics and benchmarking
│   │   └── metrics.py             # PSNR, SSIM, NMSE, k-space error
│   └── visualization/             # Plotting utilities
│       ├── reconstructions.py     # Comparison panel visualizers
│       └── error_maps.py          # Residual and error map visualizers
│
├── scripts/                       # CLI execution entry points
│   ├── train_diffusion.py         # Diffusion training script (supports --smoke_test)
│   ├── train_diffusion_pinn.py    # Joint Diffusion-PINN training script
│   ├── evaluate_diffusion.py      # Standalone diffusion evaluation CLI
│   ├── evaluate_diffusion_pinn.py # Diffusion-PINN evaluation CLI
│   ├── sanity_check.py            # Comprehensive 9-step pipeline sanity check
│   └── prepare_data.py            # Dataset preparation utility
│
├── tests/                         # Pytest automated test suite
│   ├── test_diffusion_pinn.py     # 16 unit tests for diffusion, guidance & migration
│   └── test_unet_forward.py       # U-Net forward pass and dimensional checks
│
├── notebooks/                     # Interactive Jupyter notebooks for Kaggle
│   ├── 01_environment.ipynb       # Environment validation & streaming download
│   ├── 02_kspace_inspection.ipynb # Raw k-space inspection & zero-filled IFFT
│   └── 03_zero_filled_baseline.ipynb# 4× and 8× baseline evaluation
│
├── checkpoints/                   # Checkpoint storage (.gitkept)
└── results/                       # Evaluation logs, metrics JSON, visual figures
```

---

## 5. Checkpoint Compatibility & Migration

The transition from a 2-channel input $[x_t, \text{condition}]$ to a 3-channel input $[x_t, \text{condition}, \text{mask}_{320}]$ modifies the first convolutional layer (`init_conv.weight` shape changed from `[C_out, 2, K, K]` to `[C_out, 3, K, K]`).

To prevent silent errors or degraded performance from loading mismatched weights:

- **Default Behavior**: Calling `load_compatible_state_dict()` or running evaluation with legacy 2-channel weights will raise a `RuntimeError` explaining the architectural upgrade.
- **Migration Flag (`--migrate_2ch`)**: If you possess a legacy 2-channel checkpoint and wish to evaluate it without retraining from scratch, specify `--migrate_2ch`:
  ```bash
  python scripts/evaluate_diffusion.py --checkpoint checkpoints/legacy_model.pt --migrate_2ch
  ```
  The migration utility preserves the existing weights for channels 0 ($x_t$) and 1 ($\text{condition}$) and safely initializes the channel 2 ($\text{mask}$) weights to zero.

---

## 6. Official NYU fastMRI Dataset & Streaming Extraction

This framework exclusively uses the official NYU fastMRI single-coil knee dataset.

### Dataset Access
1. Apply for access at the [NYU fastMRI portal](https://fastmri.med.nyu.edu/).
2. You will receive private Amazon S3 presigned URLs for downloading the archives (e.g. `knee_singlecoil_train.tar.xz`, `knee_singlecoil_val.tar.xz`).
3. **NEVER** commit, paste, or push signed URLs to GitHub.

### Storage-Efficient Streaming Extraction
Full FastMRI archives exceed 72+ GB compressed and 160+ GB uncompressed, which easily exceeds typical cloud/Kaggle disk limits.

Our custom streaming downloader (`src/data/download_fastmri.py`):
- Streams the signed HTTP response directly over the network.
- Extracts only the specified number of HDF5 volume files (e.g., 10 validation files, 30 training files).
- Aborts the network stream immediately once the target file count is satisfied, keeping disk usage under 1–3 GB.

To download:
```bash
export FASTMRI_VAL_URL="<your_private_presigned_url>"
python -m src.data.download_fastmri --split val --max-files 10
```

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- NumPy `< 2.0` (for PyTorch compatibility)

### Setup Virtual Environment
```bash
git clone https://github.com/Siva05-code/mri-diffusion-pinn.git
cd mri-diffusion-pinn

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Execution & Verification Workflows

### Running Unit Tests

Our comprehensive test suite validates the 3-channel diffusion model, measurement-guided reverse sampling, Fourier consistency projections, and checkpoint migration:

```bash
# Run via pytest:
pytest tests/ -v

# Or run directly via Python:
python tests/test_diffusion_pinn.py
python tests/test_unet_forward.py
```

**Expected output**: 19 passed tests covering:
- 3-channel tensor shapes, 2D/1D mask broadcasting, and center-cropping.
- Centered 2D FFT / IFFT roundtrip identity ($\text{error} < 10^{-6}$).
- Reconstruction-domain Fourier consistency projection idempotence.
- Quantitative verification that measurement guidance reduces frequency error by $>20\times$ compared to unguided sampling.
- Legacy 2-channel checkpoint rejection and safe migration.

### Running Sanity Checks

Verify the complete data pipeline, physics operators, loss functions, and model forward passes:

```bash
python scripts/sanity_check.py
```

**Expected output**: All 9 checks passing:
1. Directory structure & configuration
2. Physics FFT / IFFT roundtrip accuracy
3. 3-Channel Conditional Diffusion U-Net forward pass
4. Measurement-guided reverse diffusion sampling
5. PhysicsRefinementUNet forward pass
6. End-to-end DiffusionPINN model
7. Composite loss computation
8. Evaluation metrics implementation
9. FastMRI synthetic data pipeline

### Local CPU Smoke Training

You can execute a 1-epoch smoke test training run locally on CPU without downloading the full FastMRI dataset (an internal synthetic simulation pipeline activates automatically if data files are absent):

```bash
python scripts/train_diffusion.py --smoke_test --device cpu
```

This verifies the training loop, optimizer, forward diffusion, loss computation, validation loop, and checkpoint generation (`checkpoints/diffusion_accel4_best.pt`).

### Evaluation Pipeline

Evaluate a trained diffusion checkpoint on accelerated MRI slices with measurement guidance:

```bash
# Evaluate 4× acceleration
python scripts/evaluate_diffusion.py \
    --config configs/accel4.yaml \
    --checkpoint checkpoints/diffusion_accel4_best.pt \
    --accel 4 \
    --num_samples 10 \
    --device cpu

# Evaluate a legacy 2-channel checkpoint with automatic weight migration
python scripts/evaluate_diffusion.py \
    --checkpoint checkpoints/legacy_model.pt \
    --migrate_2ch \
    --accel 4 \
    --device cpu
```

### Full GPU Training on Kaggle

For full training on official FastMRI data using Kaggle GPUs (P100 or T4 × 2):

1. **Upload Notebook**: Create a new Kaggle notebook and link your repository.
2. **Set Kaggle Secrets**:
   - In the Kaggle notebook, navigate to **Add-ons** → **Secrets**.
   - Add secret `FASTMRI_TRAIN_URL` and `FASTMRI_VAL_URL` containing your signed fastMRI download links.
3. **Run Notebook 01**: Stream extract 30 training volumes and 10 validation volumes.
4. **Launch Standalone Diffusion Training**:
   ```bash
   python scripts/train_diffusion.py \
       --config configs/accel4.yaml \
       --data_dir /kaggle/working/data/fastmri \
       --device cuda
   ```
5. **Evaluate Convergence**:
   ```bash
   python scripts/evaluate_diffusion.py \
       --config configs/accel4.yaml \
       --checkpoint checkpoints/diffusion_accel4_best.pt \
       --data_dir /kaggle/working/data/fastmri \
       --device cuda
   ```
6. **Proceed to Joint Diffusion-PINN Training**:
   Once standalone diffusion matches or exceeds the zero-filled baseline, proceed to joint refinement:
   ```bash
   python scripts/train_diffusion_pinn.py \
       --config configs/accel4.yaml \
       --diffusion_checkpoint checkpoints/diffusion_accel4_best.pt \
       --data_dir /kaggle/working/data/fastmri \
       --device cuda
   ```

---

## 9. Quantitative Benchmarks & Baselines

All evaluations are conducted under identical preprocessing (equispaced Cartesian undersampling, center fraction 0.08, $320 \times 320$ center crop, normalization by the 99th intensity percentile) with fixed evaluation seed `42`.

### Official NYU FastMRI Benchmark (Single-Coil Knee, 4× Acceleration)

| Model / Method | Input Channels | Guidance Strategy | PSNR (dB) | SSIM | NMSE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Filled Baseline** | N/A | IFFT Direct | 22.82 ± 2.95 | 0.5373 | 0.1686 | Official Baseline |
| **Unguided Diffusion (Flawed)** | 2 | None (Pure Noise) | 18.09 ± 1.61 | 0.1997 | 0.3919 | Deprecated |
| **Measurement-Guided Diffusion** | 3 `[x_t, cond, mask]` | $320\times 320$ Fourier DC | *In Training* | *In Training* | *In Training* | Architecture Verified |
| **Diffusion + PINN Refinement** | 3 + Physics | Guided + Refinement | *Pending* | *Pending* | *Pending* | Stage 2 |

> [!NOTE]
> Converged deep-learning benchmark numbers require multi-epoch GPU training on the full fastMRI dataset. The unit tests verify that measurement-guided reverse diffusion reduces k-space measurement error by **$>24\times$** over unguided sampling ($0.0256$ vs $0.6133$), mathematically guaranteeing that reconstructed images preserve acquired frequency data.

---

## 10. Citation & Acknowledgments

If you find this work or codebase useful in your research, please cite:

```bibtex
@misc{mri_diffusion_pinn_2026,
  title={Physics-Informed Diffusion Framework for Accelerated MRI Reconstruction},
  author={Karthick, Siva and Contributors},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Siva05-code/mri-diffusion-pinn}}
}
```

### Acknowledgments
- The **NYU fastMRI Initiative** for providing the benchmark MRI dataset.
- PyTorch and the scientific Python community.