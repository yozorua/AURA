"""
configs.py — Typed configuration dataclasses for the AURA forward model.

Every physical parameter that drives the synthetic data generator lives here.
Keeping configs separate from logic allows reproducible experiments via JSON/YAML
serialisation without touching generator code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TelescopeType(Enum):
    """Optical design driving the pupil geometry and spider vane simulation."""
    REFRACTOR = auto()          # Clear circular aperture, no obstruction
    NEWTONIAN = auto()          # Central obstruction + 4-vane spider
    SCT = auto()                # Schmidt-Cassegrain: central obstruction, no spiders


class NoiseMode(Enum):
    """Controls which noise sources are active during sensor simulation."""
    NONE = auto()               # Noiseless — useful for ground-truth kernel extraction
    SHOT_ONLY = auto()          # Poisson shot noise only
    FULL = auto()               # Poisson + Gaussian read noise


# ---------------------------------------------------------------------------
# Telescope / Optics
# ---------------------------------------------------------------------------

@dataclass
class TelescopeConfig:
    """
    Physical description of the telescope optical train.

    All angular quantities are in arcseconds; linear quantities in metres.
    The plate scale (arcsec/pixel) bridges optics to the sensor plane.
    """

    telescope_type: TelescopeType = TelescopeType.NEWTONIAN

    # ---- Aperture geometry ------------------------------------------------
    aperture_diameter_m: float = 0.20          # Primary mirror / lens diameter [m]
    focal_length_m: float = 1.00               # Effective focal length [m]
    obstruction_ratio: float = 0.35            # Secondary / primary diameter ratio (0 = no obstruction)

    # ---- Spider vanes (Newtonians only) ------------------------------------
    n_spider_vanes: int = 4                    # Typically 3 or 4
    spider_vane_width_px: int = 2              # Rendered width in the pupil grid [px]

    # ---- Plate scale / sensor geometry ------------------------------------
    # cdelt1 is expected to arrive in arcsec/pixel directly — no unit conversion applied.
    pixel_scale_arcsec: float = 0.62           # arcsec per pixel on the sensor
    image_size_px: int = 256                   # Square image side length [px]

    # ---- PSF kernel grid --------------------------------------------------
    kernel_size_px: int = 15                   # Must be odd; dense output kernel side
    pupil_grid_size: int = 256                 # Internal FFT pupil resolution


# ---------------------------------------------------------------------------
# Atmospheric Seeing
# ---------------------------------------------------------------------------

@dataclass
class AtmosphericConfig:
    """
    Parameters governing the Kolmogorov turbulence simulation.

    The Fried parameter r0 [m] sets the coherence length of the atmosphere.
    Smaller r0 → worse seeing. Typical amateur site: 0.05 – 0.15 m.
    """

    # ---- Fried parameter range (randomised per sample) --------------------
    r0_min_m: float = 0.05                     # Worst seeing limit [m]
    r0_max_m: float = 0.15                     # Best seeing limit [m]

    # ---- Zernike decomposition --------------------------------------------
    n_zernike_terms: int = 36                  # Noll index up to which to fit (j=1..N)
    # Tip/tilt (j=2,3) are partially compensated by mount; still included but damped.
    tiptilt_damping: float = 0.3               # Fraction of tip/tilt variance retained

    # ---- Temporal averaging -----------------------------------------------
    # Long amateur exposures (60-300 s) average over thousands of speckle patterns.
    # We approximate this by averaging n_frames independent phase screen realisations.
    n_frames: int = 48                         # Phase screen realisations to average
    wavelength_m: float = 550e-9               # Effective wavelength for PSF [m] (green)


# ---------------------------------------------------------------------------
# Mount Mechanics
# ---------------------------------------------------------------------------

@dataclass
class MountConfig:
    """
    Simulates mechanical imperfections that smear the long-exposure PSF.

    Periodic Error models worm-gear imperfection; wind buffeting models
    short-frequency random motion during the exposure.
    """

    # ---- Periodic Error ---------------------------------------------------
    pe_enabled: bool = True
    pe_amplitude_arcsec_range: Tuple[float, float] = (0.5, 8.0)  # Peak-to-peak range
    pe_period_s_range: Tuple[float, float] = (4.0, 12.0)          # Worm period range [s]
    exposure_time_s: float = 120.0             # Integration time; sets PE smear length

    # ---- Wind buffeting (2-D Brownian motion) -----------------------------
    wind_enabled: bool = True
    wind_sigma_arcsec_range: Tuple[float, float] = (0.1, 1.5)     # RMS wind shake
    wind_n_steps: int = 200                    # Random-walk steps during exposure


# ---------------------------------------------------------------------------
# Sensor Physics
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """
    CCD / CMOS sensor model.

    Converts the continuous photon-flux image to a quantised, noisy integer array,
    then normalises back to float for network consumption.
    """

    noise_mode: NoiseMode = NoiseMode.FULL

    # ---- Sky background ---------------------------------------------------
    sky_background_e: float = 50.0             # Mean sky background electrons / pixel
    sky_background_std: float = 10.0           # Sample-to-sample variation in background

    # ---- Sensor parameters ------------------------------------------------
    read_noise_e: float = 5.0                  # Gaussian read noise [e-]
    gain_e_per_adu: float = 1.0                # electrons per ADU (quantisation)
    full_well_e: float = 65535.0               # Saturation ceiling [e-]
    bit_depth: int = 16                        # ADU bit depth

    # ---- Star flux range --------------------------------------------------
    # Stars are drawn with random peak fluxes to cover faint-to-bright range.
    star_peak_e_min: float = 500.0
    star_peak_e_max: float = 40000.0


# ---------------------------------------------------------------------------
# Spatial Variance
# ---------------------------------------------------------------------------

@dataclass
class SpatialVarianceConfig:
    """
    Controls how PSF properties vary across the focal plane.

    Real telescope fields exhibit coma, astigmatism, and field curvature that
    worsen toward the edges. The generator must reproduce this so the network
    learns spatial dependence rather than a single global PSF.
    """

    enabled: bool = True

    # Coma amplitude scales with (field_radius / coma_reference_radius)^2.
    # Set in units of the image half-width (0.5 = midway to corner).
    coma_strength_range: Tuple[float, float] = (0.0, 2.0)   # Zernike Z7/Z8 amplitude [rad rms]
    coma_reference_radius: float = 0.5                        # Fractional image radius

    # Field curvature blurs edges by increasing effective r0.
    # defocus_strength is added Zernike Z4 [rad rms] at the image corner.
    defocus_strength_range: Tuple[float, float] = (0.0, 1.5)

    # Astigmatism (Z5/Z6) at field edge [rad rms]
    astig_strength_range: Tuple[float, float] = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Master dataset config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Top-level config aggregating all sub-configs for the Dataset class."""

    telescope: TelescopeConfig = field(default_factory=TelescopeConfig)
    atmosphere: AtmosphericConfig = field(default_factory=AtmosphericConfig)
    mount: MountConfig = field(default_factory=MountConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    spatial_variance: SpatialVarianceConfig = field(default_factory=SpatialVarianceConfig)

    # ---- Dataset sizing ---------------------------------------------------
    dataset_length: int = 10_000               # Virtual epoch length (in-RAM generation)
    n_stars_per_image_range: Tuple[int, int] = (15, 60)  # Stars scattered per sample

    # ---- Reproducibility --------------------------------------------------
    base_seed: Optional[int] = None            # None → non-deterministic per worker

    def save(self, path: Path) -> None:
        """Serialise to JSON for experiment logging."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _serialise(obj):
            # Store enums as their bare name so load() can re-hydrate with Enum[name]
            if isinstance(obj, Enum):
                return obj.name
            raise TypeError(f"Cannot serialise {type(obj)}")

        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=_serialise)

    @classmethod
    def load(cls, path: Path) -> "DatasetConfig":
        """Deserialise from JSON. Enum values are re-hydrated by name."""
        with open(path) as fh:
            raw = json.load(fh)
        # Re-hydrate nested enum fields
        raw["telescope"]["telescope_type"] = TelescopeType[raw["telescope"]["telescope_type"]]
        raw["sensor"]["noise_mode"] = NoiseMode[raw["sensor"]["noise_mode"]]
        return cls(
            telescope=TelescopeConfig(**raw["telescope"]),
            atmosphere=AtmosphericConfig(**raw["atmosphere"]),
            mount=MountConfig(**{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in raw["mount"].items()
            }),
            sensor=SensorConfig(**raw["sensor"]),
            spatial_variance=SpatialVarianceConfig(**{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in raw["spatial_variance"].items()
            }),
            dataset_length=raw["dataset_length"],
            n_stars_per_image_range=tuple(raw["n_stars_per_image_range"]),
            base_seed=raw.get("base_seed"),
        )
