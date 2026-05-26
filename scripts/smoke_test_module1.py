"""
Smoke test for Module 1 — runs a single forward-model call and prints shapes.
Run from the repo root: python scripts/smoke_test_module1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from aura.data import (
    AURASyntheticDataset,
    AURAValidationDataset,
    DatasetConfig,
    TelescopeConfig,
    AtmosphericConfig,
    TelescopeType,
    get_worker_init_fn,
)


def main() -> None:
    print("── Building config ─────────────────────────────────────")
    cfg = DatasetConfig(
        telescope=TelescopeConfig(
            telescope_type=TelescopeType.NEWTONIAN,
            image_size_px=256,
            kernel_size_px=15,
            pupil_grid_size=128,   # smaller for fast smoke test
        ),
        atmosphere=AtmosphericConfig(n_frames=8),   # fewer frames for speed
        dataset_length=4,
    )
    print(f"  kernel_size_px  : {cfg.telescope.kernel_size_px}")
    print(f"  image_size_px   : {cfg.telescope.image_size_px}")
    print(f"  n_zernike_terms : {cfg.atmosphere.n_zernike_terms}")

    print("\n── Instantiating dataset ────────────────────────────────")
    ds = AURASyntheticDataset(cfg, psf_grid_step=64)  # 4×4 PSF grid for speed
    print(f"  image_shape  : {ds.image_shape}")
    print(f"  psf_map_shape: {ds.psf_map_shape}")

    print("\n── Generating one sample ────────────────────────────────")
    image, psf_map = ds[0]
    print(f"  image dtype={image.dtype}  shape={tuple(image.shape)}  "
          f"min={image.min():.4f}  max={image.max():.4f}")
    print(f"  psf_map dtype={psf_map.dtype}  shape={tuple(psf_map.shape)}")

    # Each spatial kernel should sum to ~1.0
    kernel_sums = psf_map.sum(dim=0)  # sum over K² channels → (H, W)
    print(f"  kernel sum  min={kernel_sums.min():.4f}  "
          f"max={kernel_sums.max():.4f}  mean={kernel_sums.mean():.4f}")
    assert 0.98 < kernel_sums.mean().item() < 1.02, "Kernel normalisation failed!"
    print("  [PASS] kernel normalisation")

    print("\n── DataLoader (1 worker, no pin_memory for smoke test) ──")
    loader = DataLoader(
        ds,
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        worker_init_fn=get_worker_init_fn(),
    )
    batch_img, batch_psf = next(iter(loader))
    print(f"  batch image  : {tuple(batch_img.shape)}")
    print(f"  batch psf_map: {tuple(batch_psf.shape)}")

    print("\n── Validation dataset (deterministic, noiseless) ────────")
    val_ds = AURAValidationDataset(cfg, n_samples=2, psf_grid_step=64)
    img1, _ = val_ds[0]
    img1b, _ = val_ds[0]
    assert torch.allclose(img1, img1b), "Validation dataset is not deterministic!"
    print("  [PASS] deterministic validation samples")

    print("\n✓  All smoke tests passed.")


if __name__ == "__main__":
    main()
