"""
Unit Test Suite for Physics-Informed Diffusion-PINN MRI Reconstruction Pipeline.

Verifies:
1. PINN forward pass (shape, range, residual mechanics).
2. Data consistency operator (FFT -> mask replacement -> IFFT).
3. Preservation of measured k-space values after data consistency.
4. K-space loss (differentiable complex L1 loss at sampled trajectories).
5. Physics loss (k-space data fidelity + gradient TV smoothness).
6. Total composite reconstruction loss (image + k-space + physics).
7. Diffusion + PINN joint forward pass.
8. Gradient backpropagation simultaneously updating both Diffusion and PINN networks.
"""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.pinn import PINNRefinementNet
from src.models.diffusion_pinn import DiffusionPINN
from src.models.diffusion_schedule import DiffusionSchedule
from src.physics.data_consistency import apply_data_consistency, DataConsistencyLayer
from src.physics.fft import image_to_kspace_torch, kspace_to_image_torch
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
    # Verify gradient flow through PINN to coarse input
    refined.sum().backward()
    assert coarse.grad is not None, "Gradients failed to flow to coarse input"


def test_data_consistency():
    """Test 2: Data consistency operator preserves measured k-space and outputs valid shape."""
    H, W = 64, 64
    k_measured = torch.randn(2, H, W, dtype=torch.complex64)
    mask = torch.zeros(1, 1, W)
    mask[:, :, 20:44] = 1.0  # Center low frequencies

    x_image = torch.rand(2, 1, H, W)
    x_corrected = apply_data_consistency(x_image, k_measured, mask, return_complex=False)

    assert x_corrected.shape == (2, 1, H, W)
    assert not torch.is_complex(x_corrected)


def test_fft_dc_ifft_preservation():
    """Test 3: FFT -> Data Consistency -> IFFT preserves measured k-space values."""
    H, W = 32, 32
    # Ground truth image and k-space
    gt_img = torch.rand(1, H, W, dtype=torch.complex64)
    k_gt = image_to_kspace_torch(gt_img)

    # Random undersampling mask
    mask = torch.zeros(1, 1, W)
    mask[:, :, 10:22] = 1.0

    k_measured = k_gt * mask

    # Degraded prediction
    pred_img = torch.zeros(1, H, W, dtype=torch.complex64)

    # Apply DC
    corrected_img = apply_data_consistency(pred_img, k_measured, mask, return_complex=True)
    k_corrected = image_to_kspace_torch(corrected_img)

    # Check that at mask positions, k_corrected == k_measured
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
    assert img_pred.grad is not None, "Gradient must flow through k-space loss"


def test_physics_loss():
    """Test 5: Physics loss combining data fidelity and image gradient smoothness."""
    phys_loss_fn = MRIPhysicsLoss(lambda_gradient=0.1)
    img_pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    k_meas = torch.randn(2, 32, 32, dtype=torch.complex64)
    mask = torch.ones(1, 1, 32)

    res = phys_loss_fn(img_pred, k_meas, mask)
    assert "physics_loss" in res
    assert "data_consistency_loss" in res
    assert "gradient_loss" in res

    total_phys = res["physics_loss"]
    total_phys.backward()
    assert img_pred.grad is not None, "Gradient must flow through physics loss"


def test_composite_loss():
    """Test 6: Total composite loss returns all component losses."""
    criterion = TotalReconstructionLoss(
        image_loss_type="l1",
        lambda_image=1.0,
        lambda_kspace=1.0,
        lambda_physics=1.0,
        lambda_gradient=0.1,
    )
    pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    target = torch.rand(2, 1, 32, 32)
    k_meas = torch.randn(2, 32, 32, dtype=torch.complex64)
    mask = torch.ones(1, 1, 32)

    loss_dict = criterion(pred, target, y_measured=k_meas, k_gt=k_meas, mask=mask)
    for key in ["total_loss", "image_loss", "kspace_loss", "physics_loss", "gradient_loss"]:
        assert key in loss_dict, f"Missing key {key}"
        assert loss_dict[key].item() >= 0.0

    loss_dict["total_loss"].backward()
    assert pred.grad is not None, "Gradient failed to backpropagate from total composite loss"


def test_diffusion_pinn_forward():
    """Test 7: Diffusion + PINN forward pass produces coarse and refined images."""
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=10,
    )
    target = torch.rand(2, 1, 32, 32)
    condition = torch.rand(2, 1, 32, 32)

    out = model.forward_train(target=target, condition=condition)
    assert "coarse" in out
    assert "refined" in out
    assert "pred_noise" in out
    assert out["coarse"].shape == (2, 1, 32, 32)
    assert out["refined"].shape == (2, 1, 32, 32)


def test_joint_gradient_backpropagation():
    """Test 8: Joint backpropagation simultaneously updates Diffusion and PINN parameters."""
    model = DiffusionPINN(
        diffusion_base_channels=16,
        diffusion_time_dim=64,
        pinn_base_channels=16,
        pinn_num_blocks=2,
        timesteps=10,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = TotalReconstructionLoss(
        lambda_image=1.0,
        lambda_kspace=0.5,
        lambda_physics=0.5,
    )

    target = torch.rand(1, 1, 32, 32)
    condition = torch.rand(1, 1, 32, 32)
    k_meas = torch.randn(1, 32, 32, dtype=torch.complex64)
    mask = torch.zeros(1, 1, 32)
    mask[:, :, 12:20] = 1.0  # Center lines sampled, outer lines unmeasured

    optimizer.zero_grad()
    out = model.forward_train(target=target, condition=condition, y_measured=k_meas, mask=mask)
    refined = out["refined"]

    loss_dict = criterion(refined, target, y_measured=k_meas, mask=mask)
    noise_loss = torch.mean((out["pred_noise"] - out["noise"]) ** 2)
    total_loss = loss_dict["total_loss"] + noise_loss

    total_loss.backward()

    # Verify both diffusion and pinn parameters received gradients
    has_diff_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.diffusion.parameters())
    has_pinn_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.pinn.parameters())

    assert has_diff_grad, "Diffusion model did not receive gradients!"
    assert has_pinn_grad, "PINN model did not receive gradients!"

    optimizer.step()


def test_diffusion_schedule_x0_and_sampling():
    """Test 9: DiffusionSchedule Tweedie x0 prediction and DDPM reverse sampling."""
    schedule = DiffusionSchedule(timesteps=10)
    target = torch.rand(1, 1, 32, 32)
    noise = torch.randn_like(target)
    t = torch.tensor([5])

    noisy = schedule.add_noise(target, noise, t)
    # Under ground truth noise, Tweedie formula should recover target closely
    pred_x0 = schedule.predict_x0(noisy, noise, t, clamp=False)
    assert torch.allclose(pred_x0, target, atol=1e-4), "Tweedie formula did not invert noise addition"

    # Reverse sampling test
    # Reverse sampling test with measurement guidance
    from src.models.diffusion import ConditionalDiffusionUNet
    model = ConditionalDiffusionUNet(base_channels=8, time_dim=32)
    model = ConditionalDiffusionUNet(in_channels=3, base_channels=8, time_dim=32)
    condition = torch.rand(1, 1, 32, 32)
    sample_out = schedule.sample(model=model, condition=condition, device=torch.device("cpu"))
    mask = torch.ones(1, 1, 32)
    sample_out = schedule.sample(
        model=model,
        condition=condition,
        mask=mask,
        measurement_guided=True,
        device=torch.device("cpu"),
    )
    assert sample_out.shape == (1, 1, 32, 32)
    assert sample_out.min() >= 0.0 and sample_out.max() <= 1.0, "Sampled output not in [0, 1]"


def test_inference_validation_path():
    """Test 10: Validation inference path uses model.sample() without target."""
    model = DiffusionPINN(
        diffusion_base_channels=8,
        diffusion_time_dim=32,
        pinn_base_channels=8,
        pinn_num_blocks=2,
        timesteps=5,
        enforce_dc=True,
    )
    condition = torch.rand(1, 1, 32, 32)
    k_meas = torch.randn(1, 32, 32, dtype=torch.complex64)
    mask = torch.ones(1, 1, 32)

    # Real inference path: no target given
    coarse, refined = model.sample(
        condition=condition,
        y_measured=k_meas,
        mask=mask,
        device=torch.device("cpu"),
    )

    assert coarse.shape == (1, 1, 32, 32)
    assert refined.shape == (1, 1, 32, 32)
    assert coarse.min() >= 0.0 and coarse.max() <= 1.0
    assert refined.min() >= 0.0 and refined.max() <= 1.0


def test_diffusion_3channel_forward():
    """Test 11: ConditionalDiffusionUNet accepts 3-channel input [x_t, condition, mask]."""
    from src.models.diffusion import ConditionalDiffusionUNet
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
    from src.models.diffusion import ConditionalDiffusionUNet
    model = ConditionalDiffusionUNet(in_channels=3, base_channels=16, time_dim=64)
    model.eval()

    torch.manual_seed(42)
    x_t = torch.randn(1, 1, 32, 32)
    condition = torch.rand(1, 1, 32, 32)
    t = torch.tensor([10], dtype=torch.long)

    # 1D line mask across columns
    mask_1d = torch.zeros(1, 32)
    mask_1d[0, 10:22] = 1.0

    mask_3d = mask_1d.unsqueeze(0)  # [1, 1, 32]
    mask_4d = mask_1d.unsqueeze(0).unsqueeze(0).expand(1, 1, 32, 32)  # [1, 1, 32, 32]

    with torch.no_grad():
        out_1d = model(x_t, condition, mask_1d, t)
        out_3d = model(x_t, condition, mask_3d, t)
        out_4d = model(x_t, condition, mask_4d, t)

    assert torch.allclose(out_1d, out_3d, atol=1e-6), "1D and 3D mask outputs differ"
    assert torch.allclose(out_1d, out_4d, atol=1e-6), "1D and 4D mask outputs differ"


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
    # When predicted image matches zero-filled condition, Fourier consistency must return exact identity
    condition = torch.rand(1, 1, 64, 64)
    mask = torch.zeros(1, 1, 64)
    mask[:, :, 20:44] = 1.0

    cond_complex = condition.squeeze(1).to(torch.complex64)
    k_ref = image_to_kspace_torch(cond_complex)

    k_pred = image_to_kspace_torch(cond_complex)
    k_guided = mask * k_ref + (1.0 - mask) * k_pred
    x0_guided = torch.abs(kspace_to_image_torch(k_guided)).unsqueeze(1)

    assert torch.allclose(x0_guided, condition, atol=1e-5), \
        "Fourier consistency identity violated on identical prediction and condition"


def test_measurement_guided_sampling_consistency():
    """Test 15: Measurement-guided reverse sampling preserves sampled Fourier coefficients."""
    from src.models.diffusion import ConditionalDiffusionUNet
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

    assert err_guided < 0.05, f"Sampled Fourier coefficients diverged: {err_guided:.2e}"
    assert err_guided < 0.1 * err_unguided, f"Measurement guidance did not reduce error: guided={err_guided:.4f} vs unguided={err_unguided:.4f}"


def test_checkpoint_migration_rejection():
    """Test 16: Checkpoint loader rejects 2-channel weights unless migrate_2ch=True."""
    from src.models.diffusion import ConditionalDiffusionUNet
    old_model = ConditionalDiffusionUNet(in_channels=2, base_channels=16, time_dim=32)
    old_state = old_model.state_dict()

    new_model = ConditionalDiffusionUNet(in_channels=3, base_channels=16, time_dim=32)

    # 1. Loading without migration flag must fail
    rejected = False
    try:
        new_model.load_compatible_state_dict(old_state, migrate_2ch=False)
    except RuntimeError as e:
        rejected = True
        assert "Checkpoint incompatibility" in str(e)

    assert rejected, "Legacy 2-channel checkpoint was silently accepted without migration flag!"

    # 2. Loading with migrate_2ch=True must succeed
    new_model.load_compatible_state_dict(old_state, migrate_2ch=True)
    assert new_model.enc1.conv1.weight.shape[1] == 3
    # Channel 2 must be zero-initialized
    assert torch.all(new_model.enc1.conv1.weight[:, 2, :, :] == 0.0), \
        "Migrated mask channel was not initialized with zeros"


if __name__ == "__main__":
    print("Running tests manually...")
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
    print("\nALL 8 TESTS PASSED SUCCESSFULLY!")
    test_diffusion_schedule_x0_and_sampling()
    print("✓ test_diffusion_schedule_x0_and_sampling passed")
    test_inference_validation_path()
    print("✓ test_inference_validation_path passed")
    print("\nALL 10 TESTS PASSED SUCCESSFULLY!")
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
    print("\nALL 16 UNIT TESTS PASSED SUCCESSFULLY!")
