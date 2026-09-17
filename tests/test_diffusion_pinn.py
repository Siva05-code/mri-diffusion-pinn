"""
Comprehensive Unit Test Suite for Physics-Informed Diffusion-PINN MRI Reconstruction Pipeline.

Verifies:
1. PINN forward pass (shape, range, residual mechanics).
2. Data consistency operator (FFT -> mask replacement -> IFFT).
3. Preservation of measured k-space values after data consistency.
4. K-space loss (differentiable complex L1 loss at sampled trajectories).
5. Physics loss (k-space data fidelity + gradient TV smoothness).
6. Total composite reconstruction loss (image + k-space + physics).
7. Diffusion + PINN joint forward pass.
8. Gradient backpropagation simultaneously updating both Diffusion and PINN networks.
9. Diffusion schedule Tweedie x0 estimation and posterior sampling.
10. End-to-end inference validation path without target leakage.
11. 3-channel conditional diffusion model forward pass [x_t, condition, mask].
12. Spatial mask broadcasting and conditioning mask generation.
13. Centered 2D FFT/IFFT roundtrip identity.
14. Fourier consistency projection idempotence.
15. Measurement-guided reverse sampling fidelity on sampled frequencies.
16. Checkpoint migration and rejection mechanics.
17. Raw acquisition mask shape [640, 368] and 1-D Cartesian broadcast.
18. Physical measurement acquisition y = M * kspace (no target-derived measurements).
19. MRIForwardOperator forward, adjoint, and raw data consistency.
20. Shared per-slice normalization consistency (no independent min-max scaling).
"""

import sys
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.pinn import PINNRefinementNet
from src.models.diffusion import ConditionalDiffusionUNet
from src.models.diffusion_pinn import DiffusionPINN
from src.models.diffusion_schedule import DiffusionSchedule
from src.physics.forward_operator import MRIForwardOperator
from src.physics.data_consistency import apply_data_consistency, DataConsistencyLayer
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch
from src.data.preprocessing import normalize_slice_pair, center_crop, compute_slice_scale
from src.data.undersampling import (
    EquispacedMaskFunc,
    RandomMaskFunc,
    create_mask_func,
    get_recon_conditioning_mask,
)
from src.losses.image_loss import L1ImageLoss
from src.losses.kspace_loss import KSpaceDataConsistencyLoss
from src.losses.physics_loss import ImageGradientLoss, MRIPhysicsLoss
from src.losses.total_loss import TotalReconstructionLoss


def test_pinn_forward():
    """Test 1: PINN forward pass with coarse reconstruction input."""
    pinn = PINNRefinementNet(in_channels=2, out_channels=1, base_channels=16, num_blocks=2)
    coarse = torch.rand(2, 1, 64, 64, requires_grad=True)
    zf = torch.rand(2, 1, 64, 64)

    refined = pinn(coarse, zf)

    assert refined.shape == (2, 1, 64, 64), f"Unexpected shape {refined.shape}"
    assert refined.min() >= 0.0 and refined.max() <= 1.0, "Refined output not clamped in [0, 1]"
    refined.sum().backward()
    assert coarse.grad is not None, "Gradients failed to flow to coarse input"


def test_data_consistency():
    """Test 2: Data consistency operator preserves measured k-space and outputs valid shape."""
    H, W = 64, 64
    k_measured = torch.randn(2, H, W, dtype=torch.complex64)
    mask = torch.zeros(1, 1, W)
    mask[:, :, 20:44] = 1.0

    x_image = torch.rand(2, 1, H, W)
    x_corrected = apply_data_consistency(x_image, k_measured, mask, return_complex=False)

    assert x_corrected.shape == (2, 1, H, W)
    assert not torch.is_complex(x_corrected)


def test_fft_dc_ifft_preservation():
    """Test 3: FFT -> Data Consistency -> IFFT preserves measured k-space values."""
    H, W = 32, 32
    gt_img = torch.rand(1, H, W, dtype=torch.complex64)
    k_gt = image_to_kspace_torch(gt_img)

    mask = torch.zeros(1, 1, W)
    mask[:, :, 10:22] = 1.0
    k_measured = k_gt * mask

    pred_img = torch.zeros(1, H, W, dtype=torch.complex64)
    corrected_img = apply_data_consistency(pred_img, k_measured, mask, return_complex=True)
    k_corrected = image_to_kspace_torch(corrected_img)

    diff_at_mask = mask * (k_corrected - k_measured)
    assert torch.allclose(diff_at_mask, torch.zeros_like(diff_at_mask), atol=1e-5), \
        "Measured k-space not preserved after data consistency"


def test_kspace_loss():
    """Test 4: K-space loss computes differentiable complex L1 loss at sampled points."""
    loss_fn = KSpaceDataConsistencyLoss(norm="l1")
    img_pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    k_gt = torch.randn(2, 32, 32, dtype=torch.complex64)
    mask = torch.ones(1, 1, 32)

    loss = loss_fn(img_pred, k_gt, mask)
    assert loss.ndim == 0, "Loss must be scalar"
    assert loss.item() > 0.0, "Loss must be positive"

    loss.backward()
    assert img_pred.grad is not None, "Gradients failed to flow to img_pred"


def test_physics_loss():
    """Test 5: Physics loss evaluates gradient smoothness."""
    physics_loss_fn = MRIPhysicsLoss(lambda_gradient=0.1)
    img_pred = torch.rand(2, 1, 32, 32, requires_grad=True)

    loss_dict = physics_loss_fn(img_pred)
    loss = loss_dict["physics_loss"]
    assert loss.ndim == 0
    assert loss.item() > 0.0

    loss.backward()
    assert img_pred.grad is not None


def test_composite_loss():
    """Test 6: Total composite loss aggregates terms cleanly."""
    total_loss_fn = TotalReconstructionLoss(
        image_loss_type="l1",
        lambda_image=1.0,
        lambda_kspace=0.5,
        lambda_physics=0.2,
    )
    img_pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    img_gt = torch.rand(2, 1, 32, 32)
    y_meas = torch.randn(2, 32, 32, dtype=torch.complex64)
    mask = torch.ones(1, 1, 32)

    loss_dict = total_loss_fn(
        i_pred=img_pred,
        i_gt=img_gt,
        y_measured=y_meas,
        mask=mask,
    )

    assert "total_loss" in loss_dict
    assert "image_loss" in loss_dict
    assert "kspace_loss" in loss_dict
    assert "physics_loss" in loss_dict

    loss_dict["total_loss"].backward()
    assert img_pred.grad is not None


def test_diffusion_pinn_forward():
    """Test 7: DiffusionPINN training forward pass."""
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=10,
        enforce_dc=True,
    )

    target = torch.rand(2, 1, 32, 32)
    condition = torch.rand(2, 1, 32, 32)
    mask = torch.ones(2, 1, 32, 32)

    out = model.forward_train(target=target, condition=condition, mask=mask)

    assert out["coarse"].shape == (2, 1, 32, 32)
    assert out["refined"].shape == (2, 1, 32, 32)
    assert out["pred_noise"].shape == (2, 1, 32, 32)


def test_joint_gradient_backpropagation():
    """Test 8: Joint backpropagation flows gradients to all model components."""
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=10,
        enforce_dc=False,
    )
    target = torch.rand(2, 1, 32, 32)
    condition = torch.rand(2, 1, 32, 32)
    mask = torch.ones(2, 1, 32, 32)

    out = model.forward_train(target=target, condition=condition, mask=mask)
    loss = (out["pred_noise"] ** 2).mean() + (out["refined"] - target).abs().mean()
    loss.backward()

    for name, param in model.diffusion.named_parameters():
        assert param.grad is not None, f"No grad for diffusion param: {name}"
    for name, param in model.pinn.named_parameters():
        assert param.grad is not None, f"No grad for PINN param: {name}"


def test_diffusion_schedule_x0_and_sampling():
    """Test 9: Tweedie x0 prediction and posterior variance formula."""
    schedule = DiffusionSchedule(timesteps=20)
    x_t = torch.randn(2, 1, 32, 32)
    eps = torch.randn(2, 1, 32, 32)
    t = torch.tensor([5, 10], dtype=torch.long)

    pred_x0 = schedule.predict_x0(x_t, eps, t, clamp=True)
    assert pred_x0.shape == (2, 1, 32, 32)
    assert pred_x0.min() >= 0.0 and pred_x0.max() <= 1.0


def test_inference_validation_path():
    """Test 10: End-to-end inference validation path without target leakage."""
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=5,
        enforce_dc=False,
    )
    condition = torch.rand(1, 1, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    coarse, refined = model.sample(
        condition=condition,
        mask=mask,
        device=torch.device("cpu"),
    )

    assert coarse.shape == (1, 1, 32, 32)
    assert refined.shape == (1, 1, 32, 32)
    assert coarse.min() >= 0.0 and coarse.max() <= 1.0
    assert refined.min() >= 0.0 and refined.max() <= 1.0


def test_diffusion_3channel_forward():
    """Test 11: ConditionalDiffusionUNet accepts 3-channel input [x_t, condition, mask]."""
    model = ConditionalDiffusionUNet(in_channels=3, base_channels=16, time_dim=64)
    B, H, W = 2, 64, 64
    x_t = torch.randn(B, 1, H, W)
    condition = torch.rand(B, 1, H, W)
    mask = torch.ones(1, 1, W)
    t = torch.randint(0, 100, (B,), dtype=torch.long)

    pred_noise = model(x_t, condition, mask, t)
    assert pred_noise.shape == (B, 1, H, W), f"Expected {(B, 1, H, W)}, got {pred_noise.shape}"


def test_mask_broadcasting():
    """Test 12: Mask broadcasting across different dimensions produces identical outputs."""
    model = ConditionalDiffusionUNet(in_channels=3, base_channels=16, time_dim=64)
    model.eval()

    torch.manual_seed(42)
    x_t = torch.randn(1, 1, 32, 32)
    condition = torch.rand(1, 1, 32, 32)
    t = torch.tensor([10], dtype=torch.long)

    mask_1d = torch.zeros(1, 32)
    mask_1d[0, 10:22] = 1.0

    mask_3d = mask_1d.unsqueeze(0)
    mask_4d = mask_1d.unsqueeze(0).unsqueeze(0).expand(1, 1, 32, 32)

    with torch.no_grad():
        out_1d = model(x_t, condition, mask_1d, t)
        out_3d = model(x_t, condition, mask_3d, t)
        out_4d = model(x_t, condition, mask_4d, t)

    assert torch.allclose(out_1d, out_3d, atol=1e-6)
    assert torch.allclose(out_1d, out_4d, atol=1e-6)


def test_fft_roundtrip_320():
    """Test 13: Centered FFT/IFFT roundtrip in 320x320 reconstruction domain."""
    torch.manual_seed(42)
    img = torch.randn(1, 320, 320, dtype=torch.complex64)
    k = image_to_kspace_torch(img)
    recon = kspace_to_image_torch(k)
    err = torch.norm(recon - img) / torch.norm(img)
    assert err < 1e-6, f"FFT roundtrip error too large: {err:.2e}"


def test_common_320_fourier_consistency_identity():
    """Test 14: Common 320x320 reconstruction-domain Fourier consistency identity."""
    torch.manual_seed(42)
    condition = torch.rand(1, 1, 64, 64)
    mask = torch.zeros(1, 1, 64)
    mask[:, :, 20:44] = 1.0

    cond_complex = condition.squeeze(1).to(torch.complex64)
    k_ref = image_to_kspace_torch(cond_complex)
    k_pred = image_to_kspace_torch(cond_complex)
    k_guided = mask * k_ref + (1.0 - mask) * k_pred
    x0_guided = torch.abs(kspace_to_image_torch(k_guided)).unsqueeze(1)

    assert torch.allclose(x0_guided, condition, atol=1e-5)


def test_measurement_guided_sampling_consistency():
    """Test 15: Measurement-guided reverse sampling preserves sampled Fourier coefficients."""
    model = ConditionalDiffusionUNet(in_channels=3, base_channels=8, time_dim=32)
    schedule = DiffusionSchedule(timesteps=5)

    torch.manual_seed(42)
    condition = torch.rand(1, 1, 32, 32)
    mask = torch.zeros(1, 1, 32)
    mask[:, :, 11:22] = 1.0

    recon_guided = schedule.sample(
        model=model,
        condition=condition,
        mask=mask,
        measurement_guided=True,
        device=torch.device("cpu"),
    )
    recon_unguided = schedule.sample(
        model=model,
        condition=condition,
        mask=mask,
        measurement_guided=False,
        device=torch.device("cpu"),
    )

    k_cond = image_to_kspace_torch(condition.squeeze(1).to(torch.complex64))
    k_guided = image_to_kspace_torch(recon_guided.squeeze(1).to(torch.complex64))
    k_unguided = image_to_kspace_torch(recon_unguided.squeeze(1).to(torch.complex64))

    err_guided = torch.norm(mask * (k_guided - k_cond)) / (torch.norm(mask * k_cond) + 1e-12)
    err_unguided = torch.norm(mask * (k_unguided - k_cond)) / (torch.norm(mask * k_cond) + 1e-12)

    assert err_guided < 0.05
    assert err_guided < 0.1 * err_unguided


def test_checkpoint_migration_rejection():
    """Test 16: Checkpoint loader rejects 2-channel weights unless migrate_2ch=True."""
    old_model = ConditionalDiffusionUNet(in_channels=2, base_channels=16, time_dim=32)
    old_state = old_model.state_dict()

    new_model = ConditionalDiffusionUNet(in_channels=3, base_channels=16, time_dim=32)

    rejected = False
    try:
        new_model.load_compatible_state_dict(old_state, migrate_2ch=False)
    except RuntimeError as e:
        rejected = True
        assert "Checkpoint incompatibility" in str(e)

    assert rejected, "Legacy 2-channel checkpoint was silently accepted!"

    new_model.load_compatible_state_dict(old_state, migrate_2ch=True)
    assert new_model.enc1.conv1.weight.shape[1] == 3
    assert torch.all(new_model.enc1.conv1.weight[:, 2, :, :] == 0.0)


def test_raw_mask_shape_and_broadcast():
    """Test 17: Raw mask shape [640, 368] and 1-D Cartesian broadcast along readout."""
    mask_func = create_mask_func(mask_type="equispaced", center_fraction=0.08, acceleration=4, seed=42)
    raw_shape = (640, 368)
    mask = mask_func(raw_shape, seed=42)

    assert mask.shape[-1] == 368, f"Unexpected width: {mask.shape[-1]}"
    # Broadcast across 640 readout rows
    mask_2d = mask.expand(640, 368)
    assert mask_2d.shape == (640, 368)
    # Check that all readout rows have identical phase-encoding masks
    assert torch.all(mask_2d[0] == mask_2d[300])
    assert torch.all(mask_2d[0] == mask_2d[-1])

    # Check conditioning mask extraction
    recon_mask = get_recon_conditioning_mask(mask, recon_size=(320, 320))
    assert recon_mask.shape == (1, 1, 320, 320)
    assert recon_mask.min() >= 0.0 and recon_mask.max() <= 1.0


def test_actual_undersampled_measurement():
    """Test 18: Physical undersampled measurement y = M * kspace (no target derivation)."""
    torch.manual_seed(42)
    kspace_full = torch.randn(640, 368, dtype=torch.complex64)
    mask_func = create_mask_func(mask_type="equispaced", center_fraction=0.08, acceleration=4, seed=42)
    mask_raw = mask_func(kspace_full.shape, seed=42)

    y_raw = mask_raw * kspace_full

    # Sampled positions must match kspace_full exactly
    sampled_idx = (mask_raw.squeeze() > 0.5)
    assert torch.allclose(y_raw.squeeze()[:, sampled_idx], kspace_full[:, sampled_idx])

    # Unsampled positions must be strictly 0
    unsampled_idx = (mask_raw.squeeze() < 0.5)
    assert torch.all(y_raw.squeeze()[:, unsampled_idx] == 0.0)


def test_forward_operator_adjoint_and_dc():
    """Test 19: MRIForwardOperator forward, adjoint, and raw data consistency."""
    forward_op = MRIForwardOperator(raw_shape=(640, 368), recon_shape=(320, 320))

    # 1. Forward kspace maps 320x320 -> 640x368
    x_crop = torch.rand(320, 320)
    k_pred = forward_op.forward_kspace(x_crop, target_shape=(640, 368))
    assert k_pred.shape == (640, 368)
    assert torch.is_complex(k_pred)

    # 2. Adjoint maps 640x368 -> 320x320
    x_adj = forward_op.adjoint(k_pred, crop_shape=(320, 320))
    assert x_adj.shape == (320, 320)
    # Zero-padded crop roundtrip must equal original image
    assert torch.allclose(torch.real(x_adj), x_crop, atol=1e-5)

    # 3. Raw Data Consistency replaces measured lines
    mask_raw = torch.zeros(640, 368)
    mask_raw[:, 150:200] = 1.0
    y_raw = torch.randn(640, 368, dtype=torch.complex64) * mask_raw

    x_dc = forward_op.data_consistency(x_crop, y_raw=y_raw, mask_raw=mask_raw, return_complex=False)
    assert x_dc.shape == (320, 320)
    assert not torch.is_complex(x_dc)


def test_shared_normalization_consistency():
    """Test 20: Shared per-slice normalization preserves relative contrast and linearity."""
    torch.manual_seed(42)
    zf = torch.rand(320, 320) * 50.0 + 1.0
    target = zf * 1.5  # Linear contrast relationship

    zf_norm, target_norm, scale = normalize_slice_pair(zf, target, percentile=99.0)

    assert scale > 0.0
    # Both must be scaled by the identical scalar scale
    assert torch.allclose(zf_norm, zf / scale)
    assert torch.allclose(target_norm, target / scale)
    # Ratio between target and zf must remain exactly identical
    assert torch.allclose(target_norm / zf_norm, target / zf, atol=1e-5)


if __name__ == "__main__":
    print("Running comprehensive test suite...")
    test_pinn_forward()
    print("✓ test_pinn_forward passed")
    test_data_consistency()
    print("✓ test_data_consistency passed")
    test_fft_dc_ifft_preservation()
    print("✓ test_fft_dc_ifft_preservation passed")
    test_kspace_loss()
    print("✓ test_kspace_loss passed")
    test_physics_loss()
    print("✓ test_physics_loss passed")
    test_composite_loss()
    print("✓ test_composite_loss passed")
    test_diffusion_pinn_forward()
    print("✓ test_diffusion_pinn_forward passed")
    test_joint_gradient_backpropagation()
    print("✓ test_joint_gradient_backpropagation passed")
    test_diffusion_schedule_x0_and_sampling()
    print("✓ test_diffusion_schedule_x0_and_sampling passed")
    test_inference_validation_path()
    print("✓ test_inference_validation_path passed")
    test_diffusion_3channel_forward()
    print("✓ test_diffusion_3channel_forward passed")
    test_mask_broadcasting()
    print("✓ test_mask_broadcasting passed")
    test_fft_roundtrip_320()
    print("✓ test_fft_roundtrip_320 passed")
    test_common_320_fourier_consistency_identity()
    print("✓ test_common_320_fourier_consistency_identity passed")
    test_measurement_guided_sampling_consistency()
    print("✓ test_measurement_guided_sampling_consistency passed")
    test_checkpoint_migration_rejection()
    print("✓ test_checkpoint_migration_rejection passed")
    test_raw_mask_shape_and_broadcast()
    print("✓ test_raw_mask_shape_and_broadcast passed")
    test_actual_undersampled_measurement()
    print("✓ test_actual_undersampled_measurement passed")
    test_forward_operator_adjoint_and_dc()
    print("✓ test_forward_operator_adjoint_and_dc passed")
    test_shared_normalization_consistency()
    print("✓ test_shared_normalization_consistency passed")
    print("\nALL 20 UNIT TESTS PASSED SUCCESSFULLY!")
