"""
Automated Verification Suite for U-Net Baseline and Preprocessing Pipeline.

Tests:
1. U-Net instantiation and parameter counts.
2. Forward pass with standard fastMRI FOV: [1, 1, 320, 320] -> [1, 1, 320, 320].
3. Forward pass with full k-space IFFT dimensions: [1, 1, 640, 368] -> [1, 1, 640, 368].
4. Forward pass with non-standard / odd dimensions: [1, 1, 317, 319] -> [1, 1, 317, 319].
5. Spatial alignment and center-cropping verification:
   (640, 368) zero-filled + (320, 320) target -> both (320, 320).
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_tests():
    try:
        import torch
    except ImportError:
        print("[SKIP] PyTorch is not installed in the local environment.")
        print("[INFO] Run this test in the Kaggle GPU environment: python3 tests/test_unet_forward.py")
        return

    from src.models.unet import UNet
    from src.data.preprocessing import align_and_normalize_slice, center_crop

    print("========================================================")
    print("      RUNNING U-NET ARCHITECTURE & PIPELINE TESTS       ")
    print("========================================================")

    # 1. Model Instantiation
    model = UNet(in_channels=1, out_channels=1, base_channels=32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ U-Net initialized successfully ({param_count:,} trainable parameters).")

    # 2. Forward pass: standard fastMRI cropped FOV [1, 1, 320, 320]
    x_320 = torch.randn(1, 1, 320, 320)
    with torch.no_grad():
        out_320 = model(x_320)
    assert out_320.shape == (1, 1, 320, 320), f"Mismatch: expected (1, 1, 320, 320), got {out_320.shape}"
    print(f"✓ Forward pass [1, 1, 320, 320] -> {tuple(out_320.shape)} passed.")

    # 3. Forward pass: full k-space image [1, 1, 640, 368]
    x_640 = torch.randn(1, 1, 640, 368)
    with torch.no_grad():
        out_640 = model(x_640)
    assert out_640.shape == (1, 1, 640, 368), f"Mismatch: expected (1, 1, 640, 368), got {out_640.shape}"
    print(f"✓ Forward pass [1, 1, 640, 368] -> {tuple(out_640.shape)} passed.")

    # 4. Forward pass: odd / non-divisible dimensions [1, 1, 317, 319]
    x_odd = torch.randn(1, 1, 317, 319)
    with torch.no_grad():
        out_odd = model(x_odd)
    assert out_odd.shape == (1, 1, 317, 319), f"Mismatch: expected (1, 1, 317, 319), got {out_odd.shape}"
    print(f"✓ Dynamic padding forward pass [1, 1, 317, 319] -> {tuple(out_odd.shape)} passed.")

    # 5. Preprocessing alignment: (640, 368) zf vs (320, 320) target
    zf_raw = torch.rand(640, 368)
    target_320 = torch.rand(320, 320)
    zf_norm, target_norm = align_and_normalize_slice(zf_raw, target_320)
    assert zf_norm.shape == (320, 320), f"Expected zf (320, 320), got {zf_norm.shape}"
    assert target_norm.shape == (320, 320), f"Expected target (320, 320), got {target_norm.shape}"
    assert 0.0 <= zf_norm.min() and zf_norm.max() <= 1.0, "zf normalization outside [0, 1]"
    assert 0.0 <= target_norm.min() and target_norm.max() <= 1.0, "target normalization outside [0, 1]"
    print(f"✓ Preprocessing alignment: zf (640, 368) + target (320, 320) -> both {tuple(zf_norm.shape)} passed.")

    # 6. Preprocessing alignment: both (640, 368) full IFFT -> center crop to (320, 320)
    target_raw = torch.rand(640, 368)
    zf_norm2, target_norm2 = align_and_normalize_slice(zf_raw, target_raw, crop_size=(320, 320))
    assert zf_norm2.shape == (320, 320) and target_norm2.shape == (320, 320)
    print(f"✓ Preprocessing alignment: both (640, 368) -> both {tuple(zf_norm2.shape)} passed.")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_tests()

