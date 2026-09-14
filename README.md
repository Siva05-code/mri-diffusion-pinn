# Physics-Informed Diffusion-PINN Framework for MRI Image Reconstruction

This repository contains the implementation of a physics-informed deep learning framework for accelerated MRI image reconstruction from undersampled k-space data.

## Project Overview

MRI reconstruction can be formulated as an inverse problem:

y = F(x) + ε

where:

- x is the original MRI image
- F is the undersampled Fourier acquisition operator
- y is the measured k-space data
- ε represents measurement noise

The proposed framework combines:

1. Diffusion-based image reconstruction
2. Physics-informed refinement
3. Fourier/k-space consistency
4. Data fidelity
5. Image-domain reconstruction losses

## Dataset

The project uses the fastMRI dataset.

Raw MRI data is not stored in this repository.

## Experimental Pipeline

Undersampled k-space
        ↓
Diffusion reconstruction
        ↓
Initial MRI image
        ↓
Physics-informed refinement
        ↓
Final reconstructed MRI

## Baselines

- Zero-filled reconstruction
- U-Net reconstruction
- Diffusion-only reconstruction
- Proposed Diffusion + Physics-informed refinement

## Evaluation Metrics

- PSNR
- SSIM
- NMSE
- k-space reconstruction error

## Experiments

Experiments will evaluate reconstruction quality under different acceleration factors and sampling masks, including 4× and 8× acceleration.

## Environment

Training and GPU-based experiments are performed using Kaggle GPU environments.

GitHub is used for source-code version control, while experiment outputs and trained checkpoints are downloaded from Kaggle for local analysis.