# Physics-Informed Diffusion-PINN Framework for Accelerated MRI Image Reconstruction

This repository contains the official reproducible PyTorch implementation of a physics-informed deep learning reconstruction framework for accelerated Magnetic Resonance Imaging (MRI) using the **official NYU fastMRI dataset**.

---

## 1. Project Overview & Research Formulation

Magnetic Resonance Imaging (MRI) scanners collect raw frequency measurements in **k-space**. To reduce scan acquisition time and patient discomfort, accelerated MRI undersamples k-space data. The inverse reconstruction problem is modeled as:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \boldsymbol{\varepsilon}$$

where:
- $\mathbf{x} \in \mathbb{C}^N$ is the fully sampled ground-truth MRI image.
- $\mathbf{y} \in \mathbb{C}^M$ ($M \ll N$) is the measured undersampled k-space data.
- $\mathcal{F} = \mathbf{M} \mathcal{F}_{\text{full}}$ is the undersampled Fourier forward acquisition operator ($\mathbf{M}$ is the undersampling mask, $\mathcal{F}_{\text{full}}$ is the 2D Discrete Fourier Transform).
- $\boldsymbol{\varepsilon}$ represents acquisition noise.

---

## 2. Research Architecture & Objectives

Our framework combines generative diffusion probabilistic models with explicit physics-informed data-consistency refinement:

```
Undersampled k-space (y)
          ↓
Diffusion-based Reconstruction Model
          ↓
Initial Reconstructed Image (x_init)
          ↓
Physics-Informed Refinement Operator (Fourier Forward Model & Data Consistency)
          ↓
Final Reconstructed MRI Image (x_final)
```

### Composite Objective Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{image}} + \alpha \mathcal{L}_{\text{kspace}} + \beta \mathcal{L}_{\text{physics}}$$

- **$\mathcal{L}_{\text{image}}$**: Image-domain reconstruction loss (e.g. $\ell_1$ / SSIM loss between reconstructed image and ground truth).
- **$\mathcal{L}_{\text{kspace}}$**: K-space data-consistency loss ($\|\mathbf{M} \odot (\mathcal{F}(\mathbf{x}) - \mathbf{y})\|_2^2$).
- **$\mathcal{L}_{\text{physics}}$**: Physics-informed regularizers (e.g. gradient smoothness / total variation).

---

## 3. Technical Clarification on "PINN" Terminology

> [!IMPORTANT]
> **Technical Clarification**:
> Classical Physics-Informed Neural Networks (PINNs) typically minimize partial differential equation (PDE) residuals. In this MRI reconstruction framework, "physics-informed" denotes enforcing **physical MRI acquisition forward operators ($\mathcal{F}$)** and **k-space data-consistency constraints** in the Fourier frequency domain. It does not solve spatiotemporal PDE systems. All mathematical operators directly implement the discrete Fourier forward problem $y = \mathcal{F}(x) + \varepsilon$.

---

## 4. Official NYU fastMRI Dataset Access

This project exclusively uses the **official NYU fastMRI dataset** (single-coil knee MRI for initial development and validation).

### Dataset Policy & Privacy
- **Official Access Required**: Register and request access at the [official NYU fastMRI portal](https://fastmri.med.nyu.edu/).
- **No Raw Data in Git**: No `.h5`, `.tar`, `.tar.xz`, or dataset directories are committed to GitHub (enforced by `.gitignore`).
- **Private Presigned URLs**: Official NYU download URLs are private Amazon S3 presigned links that expire.
  - **NEVER** commit, hardcode, or paste signed URLs in repository files, configs, or public notebooks.
  - Signed URLs are supplied dynamically at runtime via **Kaggle Secrets** or environment variables (`FASTMRI_VAL_URL`, `FASTMRI_TRAIN_URL`).

### Storage-Efficient Streaming Extraction (Kaggle-Optimized)
Standard Kaggle environments offer limited working storage (`/kaggle/working`). The complete fastMRI training archive exceeds 72+ GB compressed (160+ GB uncompressed) and will exceed Kaggle disk quotas if downloaded entirely.

Our pipeline implements **selective streaming extraction** via `src/data/download_fastmri.py`:
- Streams the signed HTTP response over the network.
- Decompresses and extracts only a configurable development subset (e.g. 10 validation files, 30 training files) directly to disk.
- Terminates the network stream immediately once the target quota is reached, keeping disk utilization under 1–3 GB.
- Full benchmark evaluation can scale up by configuring `num_val_files` and `num_train_files` in `configs/base.yaml`.

---

## 5. Repository Structure

```
mri-diffusion-pinn/
│
├── README.md                 # Master project documentation
├── requirements.txt          # Python dependencies (excluding base PyTorch/CUDA)
├── .gitignore                # Excludes datasets (*.h5, *.tar.xz), checkpoints, secrets
│
├── configs/                  # Experiment configurations
│   ├── base.yaml             # Core parameters & fastMRI development subset sizes
│   ├── accel4.yaml           # 4x acceleration configuration
│   └── accel8.yaml           # 8x acceleration configuration
│
├── src/                      # Modular source package
│   ├── __init__.py
│   ├── data/                 # Official fastMRI downloader, dataset, preprocessing, masks
│   │   ├── download_fastmri.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── undersampling.py
│   ├── models/               # U-Net, Diffusion, Refinement, Proposed model
│   ├── physics/              # Centered PyTorch 2D FFT / IFFT forward operators
│   │   └── fft.py
│   ├── losses/               # Image loss, k-space loss, composite objective
│   ├── training/             # Modular training scripts
│   ├── evaluation/           # Metrics (PSNR, SSIM, NMSE, k-space error)
│   │   └── metrics.py
│   └── visualization/        # 5-panel reconstructions and error maps
│       ├── reconstructions.py
│       └── error_maps.py
│
├── scripts/                  # Command-line runners
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/                # Step-by-step Jupyter notebooks for Kaggle
│   ├── 01_environment.ipynb  # Environment setup, secret check, fastMRI streaming download
│   ├── 02_kspace_inspection.ipynb  # HDF5 inspection, k-space log-mag viz, IFFT reconstruction
│   └── 03_zero_filled_baseline.ipynb # 4x and 8x zero-filled baseline evaluation
│
├── experiments/              # Experiment outputs and metrics
├── results/                  # Generated figures, tables, logs
│   ├── figures/
│   ├── tables/
│   ├── reconstructions/
│   └── logs/
│
├── checkpoints/              # Model weights (.gitkept, weights ignored)
└── paper/                    # Manuscript tables and figures
```

---

## 6. Reproducible Kaggle Setup Guide

```
GitHub Repository
       ↓
Kaggle GPU Notebook
       ↓
Set Kaggle Secrets (FASTMRI_VAL_URL, FASTMRI_TRAIN_URL)
       ↓
Run Notebook 01 (Environment & Selective fastMRI Streaming Download)
       ↓
Run Notebook 02 (K-Space Inspection & IFFT Reconstruction)
       ↓
Run Notebook 03 (Zero-Filled 4× and 8× Baseline Benchmark)
       ↓
Model Training & Ablations (U-Net → Diffusion → Proposed Diffusion-PINN)
```

### Step-by-Step Instructions:

1. **Create Kaggle GPU Notebook**: Launch a Kaggle notebook with GPU acceleration (P100 or T4 x2).
2. **Clone Repository**:
   ```bash
   git clone https://github.com/Siva05-code/mri-diffusion-pinn.git
   cd mri-diffusion-pinn
   ```
3. **Configure Secrets**:
   - In Kaggle Notebook: Click **Add-ons** → **Secrets**.
   - Add secret `FASTMRI_VAL_URL` with your official signed S3 link for `knee_singlecoil_val.tar.xz`.
   - (Optional) Add `FASTMRI_TRAIN_URL` with your signed link for `knee_singlecoil_train.tar.xz`.
4. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run Data Extraction**:
   ```bash
   python -m src.data.download_fastmri --split val --max-files 10
   ```
6. **Run Inspection & Baseline**:
   - Open `notebooks/02_kspace_inspection.ipynb` to verify raw k-space and IFFT recon.
   - Open `notebooks/03_zero_filled_baseline.ipynb` to run the 4× and 8× zero-filled baseline and save metrics.

---

## 7. Configuration Strategy

Configured via YAML in `configs/base.yaml`:

```yaml
project:
  name: "mri-diffusion-pinn"
  description: "Physics-Informed Diffusion-PINN Framework for Accelerated MRI Image Reconstruction"

dataset:
  name: "fastMRI"
  anatomy: "knee"
  modality: "singlecoil"
  source: "official_NYU_fastMRI"

data:
  dataset_root: "/kaggle/working/data/fastmri"
  train_dir: "/kaggle/working/data/fastmri/knee_singlecoil_train"
  val_dir: "/kaggle/working/data/fastmri/knee_singlecoil_val"
  num_train_files: 30
  num_val_files: 10
  image_size: [320, 320]

experiment:
  acceleration: 4
  center_fraction: 0.08
  seed: 42

training:
  batch_size: 1
  learning_rate: 0.0001
  epochs: 1

physics:
  lambda_kspace: 1.0
  lambda_physics: 1.0

paths:
  checkpoints_dir: "checkpoints"
  results_dir: "results"

evaluation:
  metrics:
    - psnr
    - ssim
    - nmse
    - kspace_error
```

---

## 8. Baselines & Evaluation Metrics

### Models
1. **Zero-Filled Baseline**: Direct inverse 2D FFT of zero-padded undersampled k-space $\mathbf{x}_{\text{zf}} = \mathcal{F}^{-1}(\mathbf{y}_{\text{us}})$.
2. **U-Net Baseline**: Standard deep convolutional baseline trained on paired undersampled and ground-truth images.
3. **Diffusion-Only Model**: Denoising diffusion probabilistic model (DDPM) conditioned on undersampled observations.
4. **Proposed Physics-Informed Diffusion**: Generative diffusion model integrated with explicit Fourier data-consistency refinement.

### Quantitative Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Evaluated against fully sampled ground-truth images.
- **SSIM (Structural Similarity Index)**: Evaluated on normalized image magnitudes.
- **NMSE (Normalized Mean Squared Error)**: $\|\mathbf{x}_{\text{pred}} - \mathbf{x}_{\text{gt}}\|_2^2 / \|\mathbf{x}_{\text{gt}}\|_2^2$.
- **K-Space Data Consistency Error**: Relative $\ell_2$ difference between acquired k-space and forward FFT of reconstruction.

---

## 9. Security & Cleanliness Checklist

- [x] Zero raw MRI data committed to Git.
- [x] Zero private signed URLs or AWS keys committed or displayed in code or logs.
- [x] Streaming selective extraction ensures Kaggle disk quota is never exceeded.
- [x] All third-party dataset dependencies removed.