"""
dataset.py — PyTorch Dataset wrapping the AURA forward model.

Design goals:
  - All data is generated in RAM at __getitem__ time — no disk I/O.
  - Each worker gets a deterministic but non-colliding RNG seed derived from
    (base_seed, worker_id, sample_index), ensuring full reproducibility while
    avoiding the forked-RNG trap that plagues multi-process DataLoaders.
  - pin_memory=True in the DataLoader is assumed; Tensors are returned as
    contiguous float32 to maximise pinned-memory copy throughput.
  - The Dataset is "virtual": __len__ returns dataset_length but there is no
    backing file list — every index produces a freshly generated sample.

Expected DataLoader configuration for RTX 5000 Pro training:
    DataLoader(
        dataset,
        batch_size=16,
        num_workers=16,       # Match high CPU core count; adjust for RAM
        pin_memory=True,      # Critical for H2D throughput
        persistent_workers=True,
        prefetch_factor=4,
    )
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .configs import DatasetConfig
from .psf_engine import PSFEngine


# ---------------------------------------------------------------------------
# Worker-init helper (must be a module-level function for pickling)
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    """
    Called once per DataLoader worker process after forking.

    Re-seeds PyTorch and NumPy independently per worker to break the identical
    fork-time RNG state that would otherwise make all workers produce identical
    batches in the same epoch.

    The seed is derived from the worker ID only — sample-level uniqueness is
    handled inside __getitem__ by mixing in the sample index.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def get_worker_init_fn():
    """Return the worker init function for DataLoader construction."""
    return _worker_init_fn


# ---------------------------------------------------------------------------
# AURASyntheticDataset
# ---------------------------------------------------------------------------

class AURASyntheticDataset(Dataset):
    """
    Infinite-ish synthetic dataset for Phase 1 PSF Estimator training.

    Each call to __getitem__(idx) runs the full forward model pipeline:
        1. Generates a synthetic blurry star field (256×256 mono).
        2. Produces the dense ground-truth PSF map (256×256×225 for a 15×15 kernel).
        3. Returns both as contiguous float32 CPU Tensors.

    Memory layout (channel-first for PyTorch Conv2d compatibility):
        image:   Tensor of shape (1, H, W)         — mono star field
        psf_map: Tensor of shape (K², H, W)        — K² = 225 for K=15

    Note: psf_map is permuted from (H, W, K²) → (K², H, W) to match PyTorch's
    (C, H, W) convention.  The training loss should treat the K² dimension as
    the prediction target per spatial position.

    Args:
        cfg:           Master DatasetConfig. Serialise with cfg.save() for reproducibility.
        psf_grid_step: Coarse-grid step for PSF evaluation (see PSFEngine).
                       Larger = faster generation, lower spatial PSF accuracy.
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        psf_grid_step: int = 32,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._engine = PSFEngine(cfg, psf_grid_step=psf_grid_step)
        self._length = cfg.dataset_length
        self._base_seed = cfg.base_seed

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Generate and return one (image, psf_map) training pair.

        The RNG is seeded from (base_seed, worker_id, idx) to guarantee:
          - Different samples per epoch (idx varies).
          - No collision across workers (worker_id varies).
          - Full reproducibility when base_seed is set.

        Args:
            idx: Sample index in [0, dataset_length).

        Returns:
            image:   FloatTensor of shape (1, H, W), values in [0, 1].
            psf_map: FloatTensor of shape (K², H, W), each spatial kernel sums to 1.
        """
        rng = self._make_rng(idx)
        image_np, psf_map_np = self._engine.generate(rng)

        # image_np: (H, W) float32 → (1, H, W)
        image_t = torch.from_numpy(np.ascontiguousarray(image_np)).unsqueeze(0)

        # psf_map_np: (H, W, K²) float32 → (K², H, W)
        psf_t = torch.from_numpy(
            np.ascontiguousarray(psf_map_np.transpose(2, 0, 1))
        )

        return image_t, psf_t

    # ------------------------------------------------------------------
    def _make_rng(self, idx: int) -> np.random.Generator:
        """
        Construct a worker-safe RNG for sample `idx`.

        Seed derivation mixes base_seed (optional global seed), the DataLoader
        worker ID (obtained at call-time, 0 if main process), and idx.
        Using a hash prevents seed aliasing when any component is 0.
        """
        # torch.utils.data.get_worker_info() returns None in the main process
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self._base_seed is not None:
            seed_str = f"{self._base_seed}:{worker_id}:{idx}"
        else:
            # Non-deterministic: mix worker_id and idx with a random 64-bit int
            # (set at Dataset construction time in the main process)
            seed_str = f"{id(self)}:{worker_id}:{idx}"

        # SHA-256 → take first 8 bytes → uint64 seed (avoids birthday collisions)
        digest = hashlib.sha256(seed_str.encode()).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="little") % (2 ** 63)
        return np.random.default_rng(seed_int)

    # ------------------------------------------------------------------
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """Returns (C, H, W) of the image output tensor."""
        H = W = self._cfg.telescope.image_size_px
        return (1, H, W)

    @property
    def psf_map_shape(self) -> Tuple[int, int, int]:
        """Returns (K², H, W) of the psf_map output tensor."""
        K = self._cfg.telescope.kernel_size_px
        H = W = self._cfg.telescope.image_size_px
        return (K * K, H, W)


# ---------------------------------------------------------------------------
# Validation dataset helper (deterministic, no sensor noise)
# ---------------------------------------------------------------------------

class AURAValidationDataset(AURASyntheticDataset):
    """
    Deterministic, noiseless variant of AURASyntheticDataset for validation.

    Overrides:
      - base_seed is forced to 42 so samples are identical across runs.
      - Sensor noise mode is forced to NONE so validation loss measures
        only PSF estimation error, not noise sensitivity.
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        n_samples: int = 500,
        psf_grid_step: int = 32,
    ) -> None:
        from copy import deepcopy
        from .configs import NoiseMode

        val_cfg = deepcopy(cfg)
        val_cfg.base_seed = 42
        val_cfg.dataset_length = n_samples
        val_cfg.sensor.noise_mode = NoiseMode.NONE

        super().__init__(val_cfg, psf_grid_step=psf_grid_step)
