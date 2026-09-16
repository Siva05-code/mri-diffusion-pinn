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
