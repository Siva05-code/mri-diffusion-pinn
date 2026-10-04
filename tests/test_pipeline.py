"""Unit tests for the reconstruction pipeline. Run with: pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.undersampling import make_cartesian_mask, effective_acceleration
from src.evaluation.metrics import nmse, psnr, ssim
from src.losses.composite_loss import LossWeights
from src.models.diffusion import DiffusionReconstructor, EMA
from src.models.refinement import RefinementUNet
from src.physics.fft import data_consistency, dc_error, fft2c, ifft2c

SIZE = 32


def tiny_model(**kw):
    return DiffusionReconstructor(
        image_size=SIZE, diffusion_steps=100, base_channels=8,
        channel_mult=(1, 2), attention_levels=(1,), time_dim=32,
        residual_scale=0.1, **kw,
    )


def sample_batch(accel=4, batch=2):
    torch.manual_seed(0)
    x = torch.rand(batch, 1, SIZE, SIZE)
    mask_np = make_cartesian_mask(SIZE, accel, 0.1, np.random.default_rng(0))
    mask = torch.from_numpy(mask_np)[None, None].expand(batch, 1, SIZE, SIZE).contiguous()
    measured = mask.to(torch.complex64) * fft2c(x)
    zf = ifft2c(measured).real
    return x, zf, mask, measured


# --------------------------------------------------------------- physics
def test_fft_roundtrip():
    x = torch.rand(2, 1, SIZE, SIZE)
    assert (ifft2c(fft2c(x)).real - x).abs().max() < 1e-4


def test_fft_is_orthonormal():
    """Parseval: an orthonormal transform preserves energy."""
    x = torch.rand(1, 1, SIZE, SIZE)
    assert abs(float(fft2c(x).abs().pow(2).sum() - x.pow(2).sum())) < 1e-2


@pytest.mark.parametrize("accel", [4, 8])
def test_mask_is_hermitian(accel):
    m = make_cartesian_mask(SIZE, accel, 0.1, np.random.default_rng(1))
    idx = np.arange(SIZE)
    assert np.allclose(m[0], m[0][(SIZE - idx) % SIZE])


@pytest.mark.parametrize("accel", [4, 8])
def test_effective_acceleration(accel):
    m = make_cartesian_mask(SIZE, accel, 0.08, np.random.default_rng(2))
    assert 0.6 * accel <= effective_acceleration(m) <= 1.6 * accel


def test_zero_filled_is_real():
    """The whole point of the Hermitian mask: no information lost to .real"""
    _, _, mask, measured = sample_batch()
    assert ifft2c(measured).imag.abs().max() < 1e-5


def test_data_consistency_is_a_projection():
    _, _, mask, measured = sample_batch()
    noise = torch.rand(2, 1, SIZE, SIZE)
    once = data_consistency(noise, measured, mask)
    twice = data_consistency(once, measured, mask)
    assert dc_error(once, measured, mask) < 1e-4
    assert (twice - once).abs().max() < 1e-4


def test_zero_filled_is_data_consistent():
    _, zf, mask, measured = sample_batch()
    assert dc_error(zf, measured, mask) < 1e-4


# --------------------------------------------------------------- diffusion
@pytest.mark.parametrize("pred_type", ["v", "eps"])
def test_parameterisation_inverts(pred_type):
    model = tiny_model(prediction_type=pred_type)
    x0 = torch.rand(2, 1, SIZE, SIZE) * 2 - 1
    t = torch.tensor([5, 80])
    noise = torch.randn_like(x0)
    xt = model.diffusion.q_sample(x0, t, noise)
    target = model.diffusion.target_for(x0, noise, t)
    assert (model.diffusion.to_x0(xt, t, target) - x0).abs().max() < 1e-2


def test_q_sample_endpoints():
    """At t=0 x_t is nearly clean; at t=T-1 it is nearly pure noise."""
    model = tiny_model()
    x0 = torch.rand(1, 1, SIZE, SIZE)
    noise = torch.randn_like(x0)
    near_clean = model.diffusion.q_sample(x0, torch.tensor([0]), noise)
    near_noise = model.diffusion.q_sample(x0, torch.tensor([99]), noise)
    assert (near_clean - x0).abs().mean() < (near_noise - x0).abs().mean()


def test_training_loss_finite_and_differentiable():
    model = tiny_model()
    x, zf, mask, measured = sample_batch()
    residual = ((x - zf) / 0.1).clamp(-1, 1)
    losses = model.training_loss(x, zf, residual, mask, measured, LossWeights())
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_sampler_output_is_data_consistent():
    model = tiny_model().eval()
    _, zf, mask, measured = sample_batch()
    out = model.reconstruct(zf, mask, measured, inference_steps=5)
    assert out.shape == zf.shape
    assert torch.isfinite(out).all()
    assert dc_error(out, measured, mask) < 1e-4


def test_disabling_dc_changes_result():
    """The ablation switch must actually do something."""
    model = tiny_model().eval()
    _, zf, mask, measured = sample_batch()
    torch.manual_seed(0)
    with_dc = model.reconstruct(zf, mask, measured, inference_steps=5, use_data_consistency=True)
    torch.manual_seed(0)
    without = model.reconstruct(zf, mask, measured, inference_steps=5, use_data_consistency=False)
    assert dc_error(with_dc, measured, mask) < dc_error(without, measured, mask)


def test_residual_scale_survives_state_dict():
    model = tiny_model()
    model.set_residual_scale(0.1234)
    reloaded = tiny_model()
    reloaded.load_state_dict(model.state_dict())
    assert abs(float(reloaded.residual_scale) - 0.1234) < 1e-6


def test_ema_tracks_and_copies_buffers():
    model = tiny_model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    model.set_residual_scale(0.42)
    ema.update(model)
    assert abs(float(ema.shadow.residual_scale) - 0.42) < 1e-6


# --------------------------------------------------------------- refinement
def test_refinement_starts_as_identity():
    refine = RefinementUNet(base=8, depth=2).eval()
    x, zf, mask, measured = sample_batch()
    # Feed an image already inside [0, 1]: the zero-filled image is not, and the
    # output clamp would legitimately change it, which is not what we are testing.
    inp = zf.clamp(0, 1)
    out = refine(inp, zf, mask, measured, apply_dc=False)
    assert (out - inp).abs().max() < 1e-5


# --------------------------------------------------------------- metrics
def test_metrics_on_identity():
    x = torch.rand(2, 1, SIZE, SIZE)
    assert psnr(x, x) > 100
    assert abs(ssim(x, x) - 1.0) < 1e-3
    assert nmse(x, x) < 1e-9


def test_psnr_decreases_with_noise():
    x = torch.rand(2, 1, SIZE, SIZE)
    small = (x + 0.01 * torch.randn_like(x)).clamp(0, 1)
    large = (x + 0.10 * torch.randn_like(x)).clamp(0, 1)
    assert psnr(small, x) > psnr(large, x)
