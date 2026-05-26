"""
psf_engine.py — Physics-based PSF forward model.

Produces a spatially-variant PSF map: for every pixel (x, y) in a 256×256
field, a 15×15 kernel representing the local Point Spread Function.

Pipeline per pixel location:
  1. ApertureGenerator   → complex pupil function (with obstruction / spider vanes)
  2. ZernikePhaseScreen  → Kolmogorov atmospheric wavefront error
  3. SpatialAberrations  → field-position-dependent coma / defocus / astigmatism
  4. AtmosphericSeeing   → time-averaged long-exposure PSF via FFT
  5. MountMechanics      → PE streak + wind Brownian-motion smear
  6. PSFEngine (top)     → assembles the dense (H, W, K²) ground-truth tensor

All heavy math runs on NumPy (CPU); the Dataset wrapper converts to Tensors.
Worker-level RNG is seeded per-sample so multi-process DataLoaders are safe.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve as nd_convolve
from scipy.special import j1  # Bessel J1 for Airy pattern sanity checks

from .configs import (
    AtmosphericConfig,
    MountConfig,
    SensorConfig,
    SpatialVarianceConfig,
    TelescopeConfig,
    TelescopeType,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n (for FFT efficiency)."""
    return 1 << (n - 1).bit_length()


def _normalise_kernel(kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalise a PSF kernel so it sums to 1.0 (energy-conserving convolution).
    Guards against degenerate all-zero kernels by returning a delta fallback.
    """
    total = kernel.sum()
    if total < 1e-12:
        # Degenerate kernel: return a centred delta function
        delta = np.zeros_like(kernel)
        cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
        delta[cy, cx] = 1.0
        return delta
    return kernel / total


def _crop_centre(arr: NDArray, out_size: int) -> NDArray:
    """Crop a 2-D array to (out_size × out_size) around the centre."""
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    half = out_size // 2
    return arr[cy - half : cy + half + 1, cx - half : cx + half + 1]


# ---------------------------------------------------------------------------
# Module 1a: Aperture / Pupil Generator
# ---------------------------------------------------------------------------

class ApertureGenerator:
    """
    Generates the complex telescope pupil function on a square grid.

    The pupil encodes:
      - Circular aperture support
      - Central obstruction (SCT / Newtonian secondary mirror shadow)
      - Spider vanes (Newtonian, rendered as thin rectangular masks)

    The OTF / PSF is later obtained by FFT of (pupil × phase_screen).
    """

    def __init__(self, cfg: TelescopeConfig) -> None:
        self._cfg = cfg
        self._grid_size = cfg.pupil_grid_size

        # Pre-compute the static binary pupil mask (does not change between samples)
        self._base_mask: NDArray[np.float64] = self._build_base_mask()

    # ------------------------------------------------------------------
    def _build_base_mask(self) -> NDArray[np.float64]:
        """
        Build the binary pupil mask encoding aperture geometry.

        Coordinates run from -1 to +1 across the primary mirror diameter.
        """
        N = self._grid_size
        # Normalised coordinate grid: radius = 1.0 at primary mirror edge
        y, x = np.mgrid[-1 : 1 : N * 1j, -1 : 1 : N * 1j]
        r = np.hypot(x, y)

        # Primary aperture circle
        mask = (r <= 1.0).astype(np.float64)

        t = self._cfg.telescope_type
        obs = self._cfg.obstruction_ratio

        # Central obstruction (SCT and Newtonian secondary shadow)
        if t in (TelescopeType.NEWTONIAN, TelescopeType.SCT) and obs > 0:
            mask[r <= obs] = 0.0

        # Spider vanes — only for Newtonians
        if t == TelescopeType.NEWTONIAN:
            mask = self._apply_spider_vanes(mask, x, y)

        return mask

    def _apply_spider_vanes(
        self,
        mask: NDArray[np.float64],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Carve spider vane slots out of the pupil mask.

        A 4-vane spider has vanes at 0°, 90°, 180°, 270°.
        Width is converted from pixels to the normalised [-1, 1] coordinate.
        """
        N = self._grid_size
        n_vanes = self._cfg.n_spider_vanes
        # Width in normalised coordinates
        width_norm = self._cfg.spider_vane_width_px / N * 2.0

        for k in range(n_vanes):
            angle_rad = math.pi * k / n_vanes
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            # Signed perpendicular distance from the vane centre-line
            perp_dist = np.abs(-sin_a * x + cos_a * y)
            # Along-vane coordinate (must be within aperture, handled by base mask)
            mask[perp_dist < width_norm / 2.0] = 0.0

        return mask

    # ------------------------------------------------------------------
    def get_pupil(self, phase_screen: NDArray[np.float64]) -> NDArray[np.complex128]:
        """
        Apply a wavefront phase screen to the pupil and return the complex pupil.

        Args:
            phase_screen: Real-valued 2-D phase map in radians, shape (N, N).

        Returns:
            Complex pupil P(u,v) = mask * exp(i * phase), shape (N, N).
        """
        if phase_screen.shape != (self._grid_size, self._grid_size):
            raise ValueError(
                f"Phase screen shape {phase_screen.shape} does not match "
                f"pupil grid {self._grid_size}×{self._grid_size}."
            )
        return self._base_mask * np.exp(1j * phase_screen)

    def get_psf_from_phase(self, phase_screen: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the instantaneous PSF intensity from a phase screen via FFT.

        PSF = |FFT(pupil)|² normalised to unit energy.
        """
        pupil = self.get_pupil(phase_screen)
        # Zero-pad to next power-of-two for FFT efficiency
        pad = _next_power_of_two(self._grid_size * 2)
        amplitude = np.fft.fftshift(np.fft.fft2(pupil, s=(pad, pad)))
        psf = np.abs(amplitude) ** 2
        return _normalise_kernel(psf)


# ---------------------------------------------------------------------------
# Module 1b: Zernike Phase Screen Generator
# ---------------------------------------------------------------------------

class ZernikePhaseScreen:
    """
    Generates Kolmogorov-statistics wavefront phase screens using Zernike polynomials.

    Uses the Noll (1976) covariance matrix for the Zernike coefficient variances
    under Kolmogorov turbulence.  Each call to `sample()` returns an independent
    realisation drawn from this distribution.

    Reference:
        Noll, R.J. (1976). Zernike polynomials and atmospheric turbulence.
        J. Opt. Soc. Am., 66(3), 207-211.
    """

    # Noll index → (radial degree n, azimuthal frequency m) mapping (1-indexed)
    # Pre-computed for j = 1 … 66 (covers up to n=10)
    _NOLL_TO_NM: Dict[int, Tuple[int, int]] = {}

    def __init__(self, cfg: AtmosphericConfig, grid_size: int) -> None:
        self._cfg = cfg
        self._grid_size = grid_size
        self._n_terms = cfg.n_zernike_terms

        # Build coordinate grid in normalised pupil coordinates (r ≤ 1)
        N = grid_size
        y, x = np.mgrid[-1 : 1 : N * 1j, -1 : 1 : N * 1j]
        self._r = np.hypot(x, y)
        self._theta = np.arctan2(y, x)
        self._pupil_mask = self._r <= 1.0

        # Pre-evaluate all Zernike basis functions on the grid: shape (n_terms, N, N)
        self._basis: NDArray[np.float64] = self._precompute_basis()

        # Pre-compute Noll covariance diagonal (variance of each Zernike coefficient)
        # Shape: (n_terms,)  — scaled by (D/r0)^(5/3) at sample time
        self._noll_variances: NDArray[np.float64] = self._compute_noll_variances()

    # ------------------------------------------------------------------
    @staticmethod
    def _noll_to_nm(j: int) -> Tuple[int, int]:
        """
        Convert Noll index j (1-based) to (radial order n, azimuthal order m).

        m > 0 → cosine mode, m < 0 → sine mode, m = 0 → rotationally symmetric.

        Verified against Noll (1976) Table 1:
          j=1→(0,0), j=2→(1,1), j=3→(1,-1), j=4→(2,0), j=5→(2,-2),
          j=6→(2,2), j=7→(3,-1), j=8→(3,1), j=9→(3,-3), j=10→(3,3), j=11→(4,0)
        """
        # Radial order n: smallest n such that (n+1)(n+2)/2 >= j
        n = int(math.ceil((-3.0 + math.sqrt(1.0 + 8.0 * j)) / 2.0))

        # 0-based position within row n; row n starts at Noll index n(n+1)/2 + 1
        j_in_row = j - n * (n + 1) // 2 - 1

        # Azimuthal absolute value |m|.
        # Even-n rows:  [m=0], [|m|=2, |m|=2], [|m|=4, |m|=4], ...
        # Odd-n rows:   [|m|=1, |m|=1], [|m|=3, |m|=3], ...
        if n % 2 == 0:
            m_abs = 0 if j_in_row == 0 else 2 * ((j_in_row + 1) // 2)
        else:
            m_abs = 2 * (j_in_row // 2) + 1

        # Sign: even j → cosine mode (+m), odd j → sine mode (−m); m=0 is unsigned
        if m_abs == 0:
            m = 0
        elif j % 2 == 0:
            m = m_abs
        else:
            m = -m_abs
        return n, m

    @staticmethod
    def _radial_poly(n: int, m: int, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the Zernike radial polynomial R_n^|m|(r) using the standard
        sum formula.  Points outside the unit disk return 0.
        """
        m_abs = abs(m)
        R = np.zeros_like(r)
        for s in range((n - m_abs) // 2 + 1):
            coeff = (
                ((-1) ** s)
                * math.factorial(n - s)
                / (
                    math.factorial(s)
                    * math.factorial((n + m_abs) // 2 - s)
                    * math.factorial((n - m_abs) // 2 - s)
                )
            )
            R = R + coeff * r ** (n - 2 * s)
        return R

    def _zernike(self, j: int) -> NDArray[np.float64]:
        """
        Evaluate the j-th Noll Zernike polynomial on the pupil grid.

        Returns a (grid_size, grid_size) array, zero outside the unit disk.
        """
        n, m = self._noll_to_nm(j)
        R = self._radial_poly(n, m, self._r)
        norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)

        if m == 0:
            Z = norm * R
        elif m > 0:
            Z = norm * R * np.cos(m * self._theta)
        else:
            Z = norm * R * np.sin(-m * self._theta)

        Z[~self._pupil_mask] = 0.0
        return Z

    def _precompute_basis(self) -> NDArray[np.float64]:
        """Evaluate and stack all n_terms Zernike bases (1-indexed, j=1…N)."""
        basis = np.stack([self._zernike(j) for j in range(1, self._n_terms + 1)])
        return basis.astype(np.float64)

    def _compute_noll_variances(self) -> NDArray[np.float64]:
        """
        Compute Noll covariance diagonal entries c_j = <a_j²> / (D/r0)^(5/3).

        These are the coefficients such that Var(a_j) = c_j * (D/r0)^(5/3).
        Approximation from Noll (1976) Table 1: c_j ∝ j^(-sqrt(3)/2) for large j.
        We use the exact analytic expressions for low orders and the power-law
        approximation for j > 11.
        """
        # Exact Noll residual variances (in units of (D/r0)^5/3 radians²)
        # from Noll 1976, Table 1, expressed as per-term variances.
        # These are approximate but sufficient for simulation purposes.
        exact = {
            1: 1.0299,   # Piston
            2: 0.5820,   # Tip
            3: 0.5820,   # Tilt
            4: 0.1336,   # Defocus
            5: 0.0875, 6: 0.0875,   # Astigmatism
            7: 0.0595, 8: 0.0595,   # Coma
            9: 0.0215, 10: 0.0215,  # Trefoil
            11: 0.0175,              # Spherical
        }
        variances = np.zeros(self._n_terms, dtype=np.float64)
        for j in range(1, self._n_terms + 1):
            if j in exact:
                variances[j - 1] = exact[j]
            else:
                # Power-law approximation: c_j ≈ 0.2944 * j^(-sqrt(3)/2)
                variances[j - 1] = 0.2944 * (j ** (-math.sqrt(3) / 2.0))
        return variances

    # ------------------------------------------------------------------
    def sample(
        self,
        r0_m: float,
        D_m: float,
        rng: np.random.Generator,
        extra_coeffs: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Draw one random Kolmogorov phase screen realisation.

        Args:
            r0_m:        Fried parameter in metres.
            D_m:         Telescope aperture diameter in metres.
            rng:         Seeded NumPy random Generator (worker-safe).
            extra_coeffs: Optional additive Zernike coefficients for deterministic
                          field-position aberrations (coma, defocus, astigmatism).
                          Shape (n_terms,).

        Returns:
            phase: Real-valued phase map in radians, shape (grid_size, grid_size).
        """
        D_over_r0 = D_m / r0_m
        # Variance of each coefficient = noll_variance * (D/r0)^(5/3)
        std_devs = np.sqrt(self._noll_variances * (D_over_r0 ** (5.0 / 3.0)))

        # Damp tip/tilt (j=2, j=3, indices 1 & 2) to simulate partial AO / guiding
        std_devs[1] *= self._cfg.tiptilt_damping
        std_devs[2] *= self._cfg.tiptilt_damping

        # Draw random Zernike coefficients
        coeffs = rng.normal(0.0, std_devs)

        # Inject deterministic field-position aberrations
        if extra_coeffs is not None:
            coeffs = coeffs + extra_coeffs

        # Synthesise phase screen as linear combination of basis functions
        # Einsum: (n_terms,) dot (n_terms, N, N) → (N, N)
        phase = np.einsum("j,jnm->nm", coeffs, self._basis)
        return phase


# ---------------------------------------------------------------------------
# Module 1c: Spatial Aberration Map
# ---------------------------------------------------------------------------

class SpatialAberrationMap:
    """
    Computes per-pixel deterministic Zernike coefficient offsets that simulate
    field-position-dependent aberrations (coma, field curvature, astigmatism).

    For a pixel at normalised field position (fx, fy) ∈ [-1, 1]², the returned
    coefficient vector is ADDED to the stochastic atmospheric coefficients,
    producing a spatially-varying total wavefront error.
    """

    def __init__(self, cfg: SpatialVarianceConfig, n_zernike_terms: int, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._n_terms = n_zernike_terms

        if not cfg.enabled:
            self._coma_strength = 0.0
            self._defocus_strength = 0.0
            self._astig_strength = 0.0
            return

        # Sample aberration magnitudes once per image (not per pixel)
        lo, hi = cfg.coma_strength_range
        self._coma_strength: float = rng.uniform(lo, hi)

        lo, hi = cfg.defocus_strength_range
        self._defocus_strength: float = rng.uniform(lo, hi)

        lo, hi = cfg.astig_strength_range
        self._astig_strength: float = rng.uniform(lo, hi)

        # Random orientation angle for coma / astig direction (breaks symmetry)
        self._coma_angle: float = rng.uniform(0, 2 * math.pi)
        self._astig_angle: float = rng.uniform(0, 2 * math.pi)

    def get_extra_coeffs(self, fx: float, fy: float) -> NDArray[np.float64]:
        """
        Return additive Zernike coefficient vector for field position (fx, fy).

        Args:
            fx, fy: Normalised field coordinates in [-1, 1].
                    (0, 0) is image centre; (±1, ±1) are image corners.

        Returns:
            extra_coeffs: shape (n_terms,), in radians of wavefront error.
        """
        if not self._cfg.enabled:
            return np.zeros(self._n_terms, dtype=np.float64)

        extra = np.zeros(self._n_terms, dtype=np.float64)
        field_r = math.hypot(fx, fy)  # 0 at centre, √2 at corner

        if self._n_terms < 8:
            return extra

        # ---- Coma (Noll j=7,8 → indices 6,7): linear in field radius ----------
        # Coma grows ∝ (field_r / reference_radius), directed along field angle
        field_angle = math.atan2(fy, fx)
        coma_scale = (field_r / self._cfg.coma_reference_radius) * self._coma_strength
        extra[6] = coma_scale * math.cos(field_angle + self._coma_angle)
        extra[7] = coma_scale * math.sin(field_angle + self._coma_angle)

        # ---- Defocus / field curvature (Noll j=4 → index 3): quadratic ---------
        # Defocus grows ∝ field_r²
        extra[3] = (field_r ** 2) * self._defocus_strength

        # ---- Astigmatism (Noll j=5,6 → indices 4,5): quadratic in field --------
        if self._n_terms >= 6:
            astig_scale = (field_r ** 2) * self._astig_strength
            extra[4] = astig_scale * math.cos(2 * (field_angle + self._astig_angle))
            extra[5] = astig_scale * math.sin(2 * (field_angle + self._astig_angle))

        return extra


# ---------------------------------------------------------------------------
# Module 1d: Time-Averaged Long-Exposure PSF
# ---------------------------------------------------------------------------

class AtmosphericSeeing:
    """
    Produces a time-averaged long-exposure PSF by averaging n_frames independent
    Kolmogorov phase screen realisations.

    Amateur astrophotography exposures (60-300 s) see thousands of atmospheric
    speckles; the average converges to a smooth Moffat-like profile whose width
    is set by the Fried parameter r0.
    """

    def __init__(
        self,
        atm_cfg: AtmosphericConfig,
        tel_cfg: TelescopeConfig,
        aperture_gen: ApertureGenerator,
        zernike_gen: ZernikePhaseScreen,
    ) -> None:
        self._atm_cfg = atm_cfg
        self._tel_cfg = tel_cfg
        self._aperture = aperture_gen
        self._zernike = zernike_gen

    def generate(
        self,
        r0_m: float,
        extra_coeffs: Optional[NDArray[np.float64]],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """
        Generate the long-exposure PSF at one field location.

        Args:
            r0_m:        Fried parameter for this sample.
            extra_coeffs: Field-position Zernike offsets from SpatialAberrationMap.
            rng:         Worker-safe random Generator.

        Returns:
            psf: Normalised 2-D PSF, shape (pupil_grid_size, pupil_grid_size).
        """
        n_frames = self._atm_cfg.n_frames
        pad = _next_power_of_two(self._tel_cfg.pupil_grid_size * 2)

        accum = np.zeros((pad, pad), dtype=np.float64)

        for _ in range(n_frames):
            phase = self._zernike.sample(
                r0_m=r0_m,
                D_m=self._tel_cfg.aperture_diameter_m,
                rng=rng,
                extra_coeffs=extra_coeffs,
            )
            pupil = self._aperture.get_pupil(phase)
            amplitude = np.fft.fftshift(np.fft.fft2(pupil, s=(pad, pad)))
            accum += np.abs(amplitude) ** 2

        psf = accum / n_frames
        return _normalise_kernel(psf)


# ---------------------------------------------------------------------------
# Module 1e: Mount Mechanics
# ---------------------------------------------------------------------------

class MountMechanics:
    """
    Applies mount-related motion blur to a PSF:
      1. Periodic Error (PE) — worm-gear tracking drift → 1-D linear streak
      2. Wind buffeting    — random Brownian walk → isotropic blur

    Both effects are applied by constructing motion-trail kernels and
    convolving them with the input PSF.
    """

    def __init__(self, cfg: MountConfig, pixel_scale_arcsec: float) -> None:
        self._cfg = cfg
        self._px_scale = pixel_scale_arcsec  # arcsec / pixel

    # ------------------------------------------------------------------
    def _pe_kernel(self, amplitude_arcsec: float, rng: np.random.Generator) -> NDArray[np.float64]:
        """
        Build a 1-D Periodic Error motion kernel.

        The worm gear drifts the star along RA by a sinusoidal profile; the
        time-averaged trail produces a top-hat-like smear of width = peak-to-peak.
        We model it as a uniform line kernel along a random angle (RA axis on sky).
        """
        half_px = max(1, int(round(amplitude_arcsec / self._px_scale / 2.0)))
        length = 2 * half_px + 1

        # Kernel size must accommodate the line; pad to make it square and centred
        kernel_size = length + 2  # small safety border
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)

        # RA tracking axis: random rotation to simulate arbitrary camera orientation
        angle = rng.uniform(0, math.pi)  # Line angle in the image plane
        cy, cx = kernel_size // 2, kernel_size // 2

        # Rasterise a line of `length` pixels through the kernel centre
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for t in range(-half_px, half_px + 1):
            iy = int(round(cy + t * sin_a))
            ix = int(round(cx + t * cos_a))
            if 0 <= iy < kernel_size and 0 <= ix < kernel_size:
                kernel[iy, ix] = 1.0

        return _normalise_kernel(kernel)

    def _wind_kernel(self, sigma_arcsec: float, rng: np.random.Generator) -> NDArray[np.float64]:
        """
        Build a wind-buffeting kernel via 2-D Brownian (random-walk) motion.

        Simulates the stellar centroid path during the exposure under wind shake.
        The resulting density map is the convolution kernel.
        """
        sigma_px = sigma_arcsec / self._px_scale
        n_steps = self._cfg.wind_n_steps

        # Step size: each step is drawn from N(0, sigma_px / sqrt(n_steps))
        step_sigma = sigma_px / math.sqrt(n_steps)
        dy = rng.normal(0.0, step_sigma, n_steps)
        dx = rng.normal(0.0, step_sigma, n_steps)

        # Cumulative walk gives the centroid trajectory
        y_path = np.cumsum(dy)
        x_path = np.cumsum(dx)

        # Render the density of the trajectory on a pixel grid
        pad = int(np.ceil(3 * sigma_px)) + 2
        grid_size = 2 * pad + 1
        kernel = np.zeros((grid_size, grid_size), dtype=np.float64)

        cy, cx = pad, pad
        for y, x in zip(y_path, x_path):
            iy = int(round(cy + y))
            ix = int(round(cx + x))
            if 0 <= iy < grid_size and 0 <= ix < grid_size:
                kernel[iy, ix] += 1.0

        return _normalise_kernel(kernel)

    # ------------------------------------------------------------------
    def apply(
        self,
        psf: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """
        Convolve the input PSF with PE and wind kernels.

        Args:
            psf: Input PSF array (any square size).
            rng: Worker-safe random Generator.

        Returns:
            Smeared PSF, same shape as input, normalised.
        """
        result = psf.copy()

        if self._cfg.pe_enabled:
            lo, hi = self._cfg.pe_amplitude_arcsec_range
            amplitude = rng.uniform(lo, hi)
            pe_k = self._pe_kernel(amplitude, rng)
            result = nd_convolve(result, pe_k, mode="constant", cval=0.0)

        if self._cfg.wind_enabled:
            lo, hi = self._cfg.wind_sigma_arcsec_range
            sigma = rng.uniform(lo, hi)
            wind_k = self._wind_kernel(sigma, rng)
            result = nd_convolve(result, wind_k, mode="constant", cval=0.0)

        return _normalise_kernel(result)


# ---------------------------------------------------------------------------
# Module 1f: Sensor Noise Model
# ---------------------------------------------------------------------------

class SensorModel:
    """
    Converts a continuous normalised flux map to a noisy, quantised image.

    Pipeline:
        1. Scale flux to electron counts (set peak star flux from config).
        2. Add sky background electrons.
        3. Apply Poisson shot noise.
        4. Apply Gaussian read noise.
        5. Quantise to ADU integers and clip to full-well / bit-depth.
        6. Return normalised float32 in [0, 1] for network consumption.
    """

    def __init__(self, cfg: SensorConfig) -> None:
        self._cfg = cfg

    def apply(
        self,
        flux_image: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """
        Args:
            flux_image: Normalised continuous flux image, values in [0, ∞).
                        Stars should have arbitrary positive flux; background ≈ 0.
            rng:        Worker-safe random Generator.

        Returns:
            Noisy image as float32 normalised to [0, 1].
        """
        cfg = self._cfg

        # ---- Scale to electron counts -------------------------------------
        # Randomly sample a peak flux within the configured range
        peak_e = rng.uniform(cfg.star_peak_e_min, cfg.star_peak_e_max)
        flux_max = flux_image.max()
        if flux_max > 0:
            electrons = (flux_image / flux_max) * peak_e
        else:
            electrons = flux_image.copy()

        # ---- Add sky background -------------------------------------------
        sky_e = max(0.0, rng.normal(cfg.sky_background_e, cfg.sky_background_std))
        electrons = electrons + sky_e

        # ---- Poisson shot noise ------------------------------------------
        if cfg.noise_mode.value >= 2:  # SHOT_ONLY or FULL
            electrons = rng.poisson(np.clip(electrons, 0, None)).astype(np.float64)

        # ---- Gaussian read noise -----------------------------------------
        if cfg.noise_mode.value == 3:  # FULL
            electrons = electrons + rng.normal(0.0, cfg.read_noise_e, electrons.shape)

        # ---- Quantise to ADU and clip ------------------------------------
        adu = np.clip(electrons / cfg.gain_e_per_adu, 0, cfg.full_well_e).astype(np.float32)

        # ---- Normalise to [0, 1] for network input -----------------------
        max_adu = float(2 ** cfg.bit_depth - 1)
        return np.clip(adu / max_adu, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Module 1g: Top-level PSF Engine
# ---------------------------------------------------------------------------

class PSFEngine:
    """
    Orchestrates the full forward model to produce:
      - A noisy synthetic star-field image: shape (H, W)
      - A dense PSF ground-truth tensor:    shape (H, W, K*K)
        where K = kernel_size_px.

    The PSF map is computed on a coarser spatial grid (psf_grid_step pixels)
    and bilinearly interpolated to full image resolution for efficiency.
    Generating a full per-pixel PSF via n_frames FFTs is prohibitively expensive;
    the real optical PSF varies slowly across the field, so coarse sampling is valid.

    Usage:
        engine = PSFEngine(cfg)
        image, psf_map = engine.generate(rng)
    """

    def __init__(self, cfg: "DatasetConfig", psf_grid_step: int = 32) -> None:  # noqa: F821
        """
        Args:
            cfg:           Master DatasetConfig.
            psf_grid_step: Spacing in pixels between PSF evaluation points.
                           32 gives an 8×8 evaluation grid for a 256×256 image.
        """
        self._cfg = cfg
        self._tel = cfg.telescope
        self._atm = cfg.atmosphere
        self._mnt = cfg.mount
        self._sen = cfg.sensor
        self._spv = cfg.spatial_variance
        self._grid_step = psf_grid_step

        # Instantiate sub-modules
        self._aperture = ApertureGenerator(self._tel)
        self._zernike = ZernikePhaseScreen(self._atm, self._tel.pupil_grid_size)
        self._seeing = AtmosphericSeeing(self._atm, self._tel, self._aperture, self._zernike)
        self._mount = MountMechanics(self._mnt, self._tel.pixel_scale_arcsec)
        self._sensor = SensorModel(self._sen)

    # ------------------------------------------------------------------
    def generate(
        self, rng: np.random.Generator
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Generate one (image, psf_map) training pair.

        Returns:
            image:   Float32 array, shape (H, W) or (H, W, 3).
            psf_map: Float32 array, shape (H, W, K²).
        """
        H = W = self._tel.image_size_px
        K = self._tel.kernel_size_px
        assert K % 2 == 1, "kernel_size_px must be odd"

        # ---- Sample atmospheric conditions for this image ----------------
        r0 = rng.uniform(self._atm.r0_min_m, self._atm.r0_max_m)

        # ---- Sample spatial aberration magnitudes (once per image) -------
        spv_map = SpatialAberrationMap(self._spv, self._atm.n_zernike_terms, rng)

        # ---- Compute PSF at each grid node --------------------------------
        # Grid of evaluation points in pixel coordinates
        xs = np.arange(self._grid_step // 2, W, self._grid_step)
        ys = np.arange(self._grid_step // 2, H, self._grid_step)
        n_gx, n_gy = len(xs), len(ys)

        # Store the cropped K×K kernels for each grid node
        # Shape: (n_gy, n_gx, K, K)
        grid_kernels = np.zeros((n_gy, n_gx, K, K), dtype=np.float32)

        for gi, gy_px in enumerate(ys):
            for gj, gx_px in enumerate(xs):
                # Normalised field coordinates in [-1, 1]
                fx = (gx_px / W) * 2.0 - 1.0
                fy = (gy_px / H) * 2.0 - 1.0

                extra_coeffs = spv_map.get_extra_coeffs(fx, fy)

                # Long-exposure PSF from atmospheric seeing
                psf_full = self._seeing.generate(r0, extra_coeffs, rng)

                # Apply mount mechanics
                psf_full = self._mount.apply(psf_full, rng)

                # Crop to K×K centred kernel
                kernel = _crop_centre(psf_full, K)
                kernel = _normalise_kernel(kernel)
                grid_kernels[gi, gj] = kernel.astype(np.float32)

        # ---- Interpolate PSF map to full image resolution ----------------
        # We interpolate each of the K² kernel pixels independently via bilinear
        # interp over the (n_gy × n_gx) coarse grid → (H × W × K²)
        psf_map = self._interpolate_psf_map(grid_kernels, H, W, xs, ys)

        # ---- Generate synthetic star field --------------------------------
        n_lo, n_hi = self._cfg.n_stars_per_image_range
        n_stars = rng.integers(n_lo, n_hi + 1)
        flux_image = self._render_stars(n_stars, psf_map, H, W, K, rng)

        # ---- Apply sensor noise ------------------------------------------
        noisy_image = self._sensor.apply(flux_image, rng)

        return noisy_image, psf_map

    # ------------------------------------------------------------------
    def _interpolate_psf_map(
        self,
        grid_kernels: NDArray[np.float32],
        H: int,
        W: int,
        xs: NDArray[np.intp],
        ys: NDArray[np.intp],
    ) -> NDArray[np.float32]:
        """
        Bilinearly interpolate the coarse PSF grid to full image resolution.

        For each of the K² kernel components, perform 2-D bilinear interpolation
        from the (n_gy × n_gx) evaluation grid to the full (H × W) image grid.
        """
        K = self._tel.kernel_size_px
        n_gy, n_gx = grid_kernels.shape[:2]

        # Flatten kernels: (n_gy, n_gx, K²)
        flat_kernels = grid_kernels.reshape(n_gy, n_gx, K * K)

        # Full-resolution coordinate grids
        iy_full = np.arange(H, dtype=np.float32)
        ix_full = np.arange(W, dtype=np.float32)

        # Scale full coords to coarse grid indices (fractional)
        # xs, ys are the pixel coordinates of coarse evaluation nodes
        x_coarse = np.interp(ix_full, xs.astype(np.float32), np.arange(n_gx, dtype=np.float32))
        y_coarse = np.interp(iy_full, ys.astype(np.float32), np.arange(n_gy, dtype=np.float32))

        psf_map = np.zeros((H, W, K * K), dtype=np.float32)

        for k_idx in range(K * K):
            # Bilinear interp of one kernel component over the 2-D grid
            # Using scipy/numpy meshgrid approach
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (np.arange(n_gy, dtype=np.float32), np.arange(n_gx, dtype=np.float32)),
                flat_kernels[:, :, k_idx],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            grid_yy, grid_xx = np.meshgrid(y_coarse, x_coarse, indexing="ij")
            points = np.stack([grid_yy.ravel(), grid_xx.ravel()], axis=-1)
            psf_map[:, :, k_idx] = interp(points).reshape(H, W)

        # Re-normalise each spatial kernel to sum to 1 (interp can violate this)
        kernel_sums = psf_map.sum(axis=-1, keepdims=True)
        kernel_sums = np.where(kernel_sums < 1e-12, 1.0, kernel_sums)
        psf_map /= kernel_sums

        return psf_map

    def _render_stars(
        self,
        n_stars: int,
        psf_map: NDArray[np.float32],
        H: int,
        W: int,
        K: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """
        Place `n_stars` point sources on the image, each convolved with its
        local PSF kernel drawn from psf_map.

        Stars are placed at random sub-pixel positions; their PSFs are taken
        from the nearest pixel in psf_map (already full-resolution).
        """
        flux_image = np.zeros((H, W), dtype=np.float64)
        half = K // 2

        star_y = rng.uniform(0, H, n_stars)
        star_x = rng.uniform(0, W, n_stars)
        # Relative flux weights (magnitude-like distribution via exponential)
        star_flux = rng.exponential(1.0, n_stars)
        star_flux /= star_flux.max()  # Normalise; absolute scaling done by SensorModel

        for sy, sx, sf in zip(star_y, star_x, star_flux):
            iy = int(round(sy))
            ix = int(round(sx))
            # Clamp to valid range
            iy = np.clip(iy, half, H - half - 1)
            ix = np.clip(ix, half, W - half - 1)

            # Local PSF kernel at this pixel
            local_psf = psf_map[iy, ix].reshape(K, K).astype(np.float64)

            # Add star flux convolved with its local PSF to the image
            flux_image[iy - half : iy + half + 1, ix - half : ix + half + 1] += sf * local_psf

        return flux_image
