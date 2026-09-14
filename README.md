# Physics-Informed Diffusion-PINN Framework for MRI Image Reconstruction
# Physics-Informed Diffusion-PINN Framework for Accelerated MRI Image Reconstruction

This repository contains the implementation of a physics-informed deep learning framework for accelerated MRI image reconstruction from undersampled k-space data.
This repository contains the official reproducible PyTorch implementation of a physics-informed deep learning reconstruction framework for accelerated Magnetic Resonance Imaging (MRI).

## Project Overview
---

MRI reconstruction can be formulated as an inverse problem:
## 1. Project Overview

y = F(x) + ε
Magnetic Resonance Imaging (MRI) scanners collect raw measurement samples in **k-space** (Fourier domain). To reduce scan acquisition time and improve patient comfort, accelerated MRI undersamples k-space data. The inverse reconstruction problem is formulated as:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \boldsymbol{\varepsilon}$$

where:
- $\mathbf{x} \in \mathbb{C}^N$ is the fully sampled ground-truth MRI image.
- $\mathbf{y} \in \mathbb{C}^M$ ($M \ll N$) is the measured undersampled k-space data.
- $\mathcal{F} = \mathbf{M} \mathcal{F}_{\text{full}}$ is the undersampled Fourier forward acquisition operator ($\mathbf{M}$ is the undersampling mask, $\mathcal{F}_{\text{full}}$ is the 2D Discrete Fourier Transform).
- $\boldsymbol{\varepsilon}$ represents measurement noise.

- x is the original MRI image
- F is the undersampled Fourier acquisition operator
- y is the measured k-space data
- ε represents measurement noise
---

The proposed framework combines:
## 2. Research Architecture

1. Diffusion-based image reconstruction
2. Physics-informed refinement
3. Fourier/k-space consistency
4. Data fidelity
5. Image-domain reconstruction losses
Our framework integrates generative diffusion probabilistic modeling with explicit physics-informed data consistency refinement:

## Dataset
```
Undersampled k-space (y)
          ↓
Diffusion-based Reconstruction Model
          ↓
Initial Reconstructed Image (x_init)
          ↓
Physics-Informed Refinement Operator (Data Consistency / Fourier Forward Model)
          ↓
Final Reconstructed MRI Image (x_final)
```

The project uses the fastMRI dataset.
---

Raw MRI data is not stored in this repository.
## 3. Terminology & Research Methodology Note

## Experimental Pipeline
> [!IMPORTANT]
> **Clarification on "PINN" Terminology**:
> Classical Physics-Informed Neural Networks (PINNs) typically minimize partial differential equation (PDE) residuals. In this MRI reconstruction framework, "physics-informed" refers to enforcing **MRI acquisition domain constraints** via the **Fourier forward operator $\mathcal{F}$** and **k-space data consistency**. It does not use PDE spatial/temporal differential operators. All mathematical formulations strictly reflect physical MRI k-space sampling and Fourier transform properties.

Undersampled k-space
        ↓
Diffusion reconstruction
        ↓
Initial MRI image
        ↓
Physics-informed refinement
        ↓
Final reconstructed MRI
---

## Baselines
## 4. Repository Structure

- Zero-filled reconstruction
- U-Net reconstruction
- Diffusion-only reconstruction
- Proposed Diffusion + Physics-informed refinement
```
mri-diffusion-pinn/
│
├── README.md                 # Master project documentation
├── requirements.txt          # Python dependencies (excluding base PyTorch/CUDA)
├── .gitignore                # Git exclusions (datasets, checkpoints, credentials)
│
├── configs/                  # Experiment configuration files
│   ├── base.yaml             # Base settings & dataset slug configuration
│   ├── accel4.yaml           # 4x acceleration hyperparameter settings
│   └── accel8.yaml           # 8x acceleration hyperparameter settings
│
├── src/                      # Source package
│   ├── __init__.py
│   ├── data/                 # Downloaders, dataset wrappers, preprocessing, masks
│   ├── models/               # U-Net, Diffusion, PINN Refiner, Proposed architecture
│   ├── physics/              # PyTorch FFT operators, forward model, data consistency
│   ├── losses/               # Image-domain, k-space, and composite loss functions
│   ├── training/             # Modular training scripts for models
│   ├── evaluation/           # Metrics (PSNR, SSIM, NMSE, k-space error)
│   └── visualization/        # Error maps, k-space plots, image reconstructions
│
├── scripts/                  # CLI scripts
│   ├── prepare_data.py       # Data preparation utility
│   ├── train.py              # Unified training launcher
│   └── evaluate.py           # Unified evaluation launcher
│
├── notebooks/                # Step-by-step Jupyter notebooks for Kaggle / Local
│   ├── 01_environment.ipynb  # Environment setup, GitHub clone, GPU & Kaggle check
│   ├── 02_kspace_inspection.ipynb  # HDF5 inspection, k-space viz, inverse FFT
│   └── 03_zero_filled_baseline.ipynb # 4x and 8x zero-filled baseline evaluation
│
├── experiments/              # Experiment logs, configs, and metrics
│   ├── zero_filled/
│   ├── unet/
│   ├── diffusion/
│   ├── diffusion_pinn/
│   └── ablations/
│
├── results/                  # Generated plots, quantitative tables, figures
│   ├── figures/
│   ├── tables/
│   ├── reconstructions/
│   └── logs/
│
├── checkpoints/              # Model weights (.pt / .pth, ignored by Git)
│
└── paper/                    # Research manuscript assets
    ├── figures/
    ├── tables/
    └── manuscript/
```

## Evaluation Metrics
---

- PSNR
- SSIM
- NMSE
- k-space reconstruction error
## 5. New Developer Quick Start

## Experiments
Follow these step-by-step instructions to initialize the project in a fresh Kaggle GPU environment:

Experiments will evaluate reconstruction quality under different acceleration factors and sampling masks, including 4× and 8× acceleration.
1. **Fork/Clone Repository**: Clone this repository to your local machine or directly inside your Kaggle notebook session.
2. **Open Kaggle**: Create a new GPU-accelerated Kaggle Notebook.
3. **Configure Kaggle Secret**: Add your Kaggle API key (`KAGGLE_USERNAME` and `KAGGLE_KEY`) to Kaggle Secrets or set up `~/.kaggle/kaggle.json`.
4. **Run Notebook 01**: Execute `notebooks/01_environment.ipynb` to verify Python, PyTorch, GPU acceleration, and Kaggle CLI access.
5. **Configure Dataset Slug**: Update `data.kaggle_dataset` in `configs/base.yaml` with the verified Kaggle dataset slug (`OWNER/DATASET-NAME`).
6. **Download Dataset**: Execute `python -m src.data.download_kaggle` to download and extract the dataset to `/kaggle/working/data/fastmri`.
7. **Run Data Inspection**: Open and run `notebooks/02_kspace_inspection.ipynb` to inspect HDF5 slices and verify inverse FFT reconstructions.
8. **Run Zero-Filled Baseline**: Execute `notebooks/03_zero_filled_baseline.ipynb` to compute initial 4× and 8× zero-filled baseline metrics (PSNR, SSIM, NMSE).
9. **Train & Evaluate Models**: Follow subsequent phase guidelines to train U-Net, Diffusion, and Physics-Refining architectures.

## Environment
---

Training and GPU-based experiments are performed using Kaggle GPU environments.
## 6. Kaggle Authentication & Secrets Management

GitHub is used for source-code version control, while experiment outputs and trained checkpoints are downloaded from Kaggle for local analysis.
To download datasets programmatically without hardcoding tokens:

> [!CAUTION]
> **NEVER hardcode API credentials, tokens, or Kaggle API keys in source files, notebooks, or git commits.**

- **Kaggle API Key**: In your Kaggle account settings, generate a `kaggle.json` API token.
- **Kaggle Secrets (Notebooks)**: Navigate to `Add-ons` -> `Secrets` in your Kaggle notebook editor. Add `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- **Private Repository Cloning**: If working in a private repo, store a GitHub Personal Access Token (PAT) as a Kaggle Secret `GH_TOKEN` and clone via environment variables:
  ```bash
  git clone https://${GH_TOKEN}@github.com/USER/mri-diffusion-pinn.git
  ```

---

## 7. Configuration Strategy

Project configuration is stored in YAML format under `configs/`:

```yaml
project:
  name: "mri-diffusion-pinn"

data:
  dataset_source: "kaggle"
  kaggle_dataset: ""  # Developer sets this to "OWNER/DATASET-NAME"
  data_dir: "/kaggle/working/data/fastmri"

experiment:
  acceleration: 4
  seed: 42

training:
  batch_size: 1
  learning_rate: 0.0001
  epochs: 1

physics:
  lambda_kspace: 1.0
  lambda_physics: 1.0

evaluation:
  metrics:
    - psnr
    - ssim
    - nmse
    - kspace_error
```

---

## 8. Evaluation Metrics & Baselines

### Quantitative Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index Measure
- **NMSE**: Normalized Mean Squared Error
- **k-space Consistency Error**: Relative $\ell_2$ norm difference in measured k-space coordinates

### Baselines & Ablations
1. **Zero-Filled Reconstruction**
2. **U-Net Baseline**
3. **Diffusion-Only Model**
4. **Proposed Physics-Informed Diffusion (Diffusion + Data Consistency)**

---

## 9. Security & Reproducibility Guidelines

- **No Dataset Files in Git**: All `.h5`, `.hdf5`, and dataset directories are ignored by `.gitignore`.
- **No Checkpoints in Git**: All `.pt`, `.pth`, and `.ckpt` files are excluded.
- **Fixed Random Seeds**: All random number generators (PyTorch, NumPy, Python) are seeded with `seed: 42` for exact reproducibility.
- **Environment Parity**: PyTorch and CUDA versions supplied by Kaggle runtime environment are verified directly in `notebooks/01_environment.ipynb`.